"""
AURA — Adaptive Update through Representation Adaptation (FC + normalization-layer variant).

The adapted normalization layers are BatchNorm for CNN backbones and LayerNorm
for transformer backbones (timm ViT); the grouping/stage machinery is shared and
lives in models/arch_utils.py.
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from dataclasses import dataclass
from copy import deepcopy

from methods.base import TTAMethod
from models.arch_utils import (
    STAGE_ORDER,
    build_stage_matchers,
    disable_fused_attention,
    get_classifier_name,
    is_affine_norm,
    is_vit,
    unwrap_backbone,
)
from utils.registry import ADAPTATION_REGISTRY

logger = logging.getLogger(__name__)


class NonlinearCorrection(nn.Module):
    """
    Residual correction network for the nonlinear lifting map.

    Architecture:  z -> Linear(d, h) -> activation -> Linear(h, p) * output_scale
    Final output layer is zero-initialized by default (ZERO_INIT_OUTPUT=True),
    so delta(z) = 0 at initialization: the model starts exactly at the linear
    solution phi + z @ A, preserving the nonzero Jacobian A.
    """
    _ACTIVATIONS = {
        "gelu":       F.gelu,
        "relu":       F.relu,
        "leaky_relu": F.leaky_relu,
        "tanh":       torch.tanh,
    }

    def __init__(
        self,
        latent_dim: int,
        param_dim: int,
        h: int = 0,
        activation: str = "gelu",
        output_scale: float = 1.0,
        zero_init_output: bool = True,
    ):
        super().__init__()
        if h <= 0:
            h = min(latent_dim, 512)
        if activation not in self._ACTIVATIONS:
            raise ValueError(f"NonlinearCorrection: unknown activation '{activation}'. Choose from {list(self._ACTIVATIONS)}.")
        self.act_fn = self._ACTIVATIONS[activation]
        self.output_scale = output_scale
        self.fc1 = nn.Linear(latent_dim, h)
        self.fc2 = nn.Linear(h, param_dim)
        if zero_init_output:
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act_fn(self.fc1(z))) * self.output_scale


@ADAPTATION_REGISTRY.register()
class AURA(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        # Hyperparameters specific to AURA
        self.default_latent_dim = cfg.AURA.LATENT_DIM
        self.obs_cov_scale = cfg.AURA.OBS_COV_SCALE
        self.gamma = cfg.AURA.GAMMA
        self.init_cov_scale = cfg.AURA.INIT_COV_SCALE
        
        self.bn_grouping_mode = getattr(cfg.AURA, "BN_GROUPING_MODE", "shared_all_bn")
        self.stage_selection = getattr(cfg.AURA, "STAGE_SELECTION", [])
        
        self.model.eval()
        self.model.requires_grad_(False)

        # ------------------------------------------------------------------
        # Nonlinear lifting: residual correction on top of the linear map.
        # theta_i = phi_i + z_i @ A_i + delta_i(z_i)
        # AURA.LIFTING.TYPE = "linear"    -> delta is None (default, backward-compat)
        # AURA.LIFTING.TYPE = "nonlinear" -> one NonlinearCorrection per block
        # ------------------------------------------------------------------
        _lifting = getattr(cfg.AURA, "LIFTING", None)
        if _lifting is not None:
            self._lifting_type        = _lifting.TYPE.lower()
            _hidden_dims              = list(_lifting.HIDDEN_DIMS)
            self._lifting_h           = int(_hidden_dims[0]) if _hidden_dims else 0
            self._lifting_activation  = _lifting.ACTIVATION.lower()
            self._lifting_output_scale= float(_lifting.OUTPUT_SCALE)
            self._lifting_zero_init   = bool(_lifting.ZERO_INIT_OUTPUT)
            self._lifting_residual    = bool(_lifting.RESIDUAL_LINEAR)
        else:
            # Legacy flat-key fallback (v1 configs without LIFTING sub-node)
            self._lifting_type        = getattr(cfg.AURA, "LIFTING_TYPE", "linear").lower()
            self._lifting_h           = int(getattr(cfg.AURA, "LIFTING_H", 0))
            self._lifting_activation  = "gelu"
            self._lifting_output_scale= 1.0
            self._lifting_zero_init   = True
            self._lifting_residual    = True
        self._use_nonlinear = (self._lifting_type == "nonlinear")

        # Discover blocks
        # We store everything fundamentally needed sequentially inside lists.
        self.block_names = []
        self.A_list = nn.ParameterList()
        self.phi_list = nn.ParameterList()
        # Nonlinear corrections (empty ModuleList when linear — keeps state_dict clean)
        self.correction_list = nn.ModuleList()
        # Whether corrections are frozen (True after checkpoint load in online mode)
        self._corrections_frozen = False
        
        self._meta_param_names_tuple = ()
        self._meta_shapes_tuple = ()
        self._meta_latent_dims_tuple = ()
        self._meta_numels_tuple = ()
        
        # Initial stats
        self.z_list = []
        self.cov_list = []
        self.z_init_list = []
        self.cov_init_list = []
        
        self._setup_blocks(cfg)
        self._A_tuple = tuple(self.A_list)
        self._phi_tuple = tuple(self.phi_list)
        # Build correction tuple: None for linear (fast path), tuple for nonlinear
        self._correction_tuple = tuple(self.correction_list) if self._use_nonlinear else None
        self._cache_init_state()
        self._setup_slicing()
        
        self.R = torch.eye(num_classes, device=self.device) * self.obs_cov_scale

        # ------------------------------------------------------------------
        # Jacobian computation mode.
        # jacrev scales with num_classes (output dim): OOM risk at 1000 classes.
        # jacfwd scales with the sum of latent_dims (input dim): cheaper when
        # num_classes >= latent_dim. Chosen once at init and held for the run.
        # ------------------------------------------------------------------
        _mode_cfg = getattr(cfg.AURA, "JACOBIAN_MODE", "auto").lower()
        _total_latent_dim = sum(self._meta_latent_dims_tuple)
        # Optional chunking for jacrev: bounds peak memory when vmapping
        # over num_classes (1000 on ImageNet). 0 = unchunked.
        _chunk = int(getattr(cfg.AURA, "JAC_CHUNK_SIZE", 0)) or None
        def _mk_rev():
            return (lambda f, argnums=0: torch.func.jacrev(f, argnums=argnums, chunk_size=_chunk)) if _chunk else torch.func.jacrev
        if _mode_cfg == "rev":
            self._jac_fn = _mk_rev()
            _chosen = f"rev (forced, chunk={_chunk})"
        elif _mode_cfg == "fwd":
            self._jac_fn = torch.func.jacfwd
            _chosen = "fwd (forced)"
        elif _mode_cfg == "auto":
            # When any non-fc block is selected, jacfwd vmaps full-network
            # forwards over `total_latent_dim` tangent vectors, blowing peak
            # activation memory through the backbone. Force jacrev (with
            # chunking over num_classes) for any backbone-touching config.
            _has_backbone_block = any(b != "fc" for b in self.block_names)
            if _has_backbone_block:
                # Default chunk_size=1 if user didn't set one — strictly bounds peak.
                if _chunk is None:
                    _chunk = 1
                    _mk_rev_local = lambda: (lambda f, argnums=0: torch.func.jacrev(f, argnums=argnums, chunk_size=1))
                else:
                    _mk_rev_local = _mk_rev
                self._jac_fn = _mk_rev_local()
                _chosen = f"rev (auto; backbone block present, chunk={_chunk})"
            elif num_classes >= max(_total_latent_dim, 100):
                self._jac_fn = torch.func.jacfwd
                _chosen = f"fwd (auto; fc-only, num_classes={num_classes} >= latent_dim={_total_latent_dim})"
            else:
                self._jac_fn = _mk_rev()
                _chosen = f"rev (auto; num_classes={num_classes} < latent_dim={_total_latent_dim}, chunk={_chunk})"
        else:
            raise ValueError(
                f"AURA: unknown JACOBIAN_MODE '{_mode_cfg}'. "
                "Expected one of: 'auto', 'rev', 'fwd'."
            )
        logger.info(
            f"AURA: Jacobian mode = {_chosen} "
            f"(num_classes={num_classes}, total_latent_dim={_total_latent_dim})"
        )

        # Timing state
        self._timing_enabled = cfg.TIMING.ENABLED
        self._timing_warmup = cfg.TIMING.WARMUP_SAMPLES
        self._timing_target = cfg.TIMING.NUM_MEASUREMENTS
        self._timing_results = []
        self._timing_sample_counter = 0

        if hasattr(cfg.AURA, "OFFLINE_CKPT") and cfg.AURA.OFFLINE_CKPT != "":
            self._load_and_validate_checkpoint(cfg.AURA.OFFLINE_CKPT)

    def _setup_blocks(self, cfg):
        # Backbone introspection: strips the normalize / class-mask wrappers and
        # tells us whether the stage names map to CNN layers or ViT block groups.
        core, _, _ = unwrap_backbone(self.model)
        self.is_vit_backbone = is_vit(core)
        stage_matchers = build_stage_matchers(core)
        logger.info(
            f"AURA: backbone family = {'vit' if self.is_vit_backbone else 'cnn'} "
            f"({type(core).__name__}), available stages = {sorted(stage_matchers.keys()) + ['fc']}"
        )

        # 1. Setup FC block (Only if requested)
        if "fc" in self.stage_selection:
            last_fc_name = get_classifier_name(self.model, self.num_classes)
            logger.info(f"AURA: Identified classifier FC layer as '{last_fc_name}'")
            self._register_block("fc", [f"{last_fc_name}.weight", f"{last_fc_name}.bias"])

        # 2. Setup normalization-layer blocks (BatchNorm on CNNs, LayerNorm on ViTs).
        # "stage_N" resolves per backbone family:
        #   CNN — torchvision ResNets (layerN.), RobustBench WRN (blockN.),
        #         Hendrycks2020AugMix_ResNeXt (stage_N.). WRN-28-10 / ResNeXt-29
        #         have only stage_1..stage_3; ResNet-50 adds stage_4.
        #   ViT — the transformer block list split into 4 contiguous groups
        #         (ViT-B/16: blocks 0-2, 3-5, 6-8, 9-11), with the patch embed
        #         norm in stage_1 and the final norm / fc_norm in stage_4.
        if "stage_4" in self.stage_selection and "stage_4" not in stage_matchers:
            raise ValueError(
                "AURA: STAGE_SELECTION contains 'stage_4' but this backbone has no fourth "
                "stage (no layer4/block4 module). stage_4 is valid for ResNet-50-style and "
                "ViT backbones — WRN-28-10 has only stage_1..stage_3."
            )

        if self.bn_grouping_mode == "shared_all_bn":
            # Collect every affine normalization layer into a single block
            shared_bn_names = []
            for name, module in self.model.named_modules():
                if is_affine_norm(module):
                    shared_bn_names.extend([f"{name}.weight", f"{name}.bias"])
            if len(shared_bn_names) > 0:
                self._register_block("shared_all_bn", shared_bn_names)

        elif self.bn_grouping_mode == "per_stage_bn":
            for stage_target in self.stage_selection:
                if stage_target == "fc":
                    continue
                if stage_target not in stage_matchers:
                    # Fallback: treat as a literal norm-layer leaf name (e.g., "bn1",
                    # "norm1") or module-name suffix (e.g., "layer4.2.bn3",
                    # "blocks.11.norm2"). Register every affine norm layer whose name
                    # == stage_target or ends with ".<stage_target>".
                    bn_names = []
                    for name, module in self.model.named_modules():
                        if not is_affine_norm(module):
                            continue
                        if name == stage_target or name.endswith("." + stage_target):
                            bn_names.extend([f"{name}.weight", f"{name}.bias"])
                    if len(bn_names) > 0:
                        logger.info(f"AURA: name-filter '{stage_target}' matched {len(bn_names)//2} norm modules")
                        self._register_block(stage_target.replace(".", "_"), bn_names)
                    else:
                        logger.warning(f"AURA: name-filter '{stage_target}' yielded no norm parameters")
                    continue

                matches = stage_matchers[stage_target]
                stage_bn_names = []
                for name, module in self.model.named_modules():
                    if is_affine_norm(module) and matches(name):
                        stage_bn_names.extend([f"{name}.weight", f"{name}.bias"])

                if len(stage_bn_names) > 0:
                    logger.info(f"AURA: Stage '{stage_target}' matched {len(stage_bn_names)//2} norm modules")
                    self._register_block(stage_target, stage_bn_names)
                else:
                    logger.warning(f"AURA: Stage selection '{stage_target}' yielded no norm parameters!")
        else:
            raise ValueError(f"Unknown BN_GROUPING_MODE: {self.bn_grouping_mode}")

    def _register_block(self, block_name, param_names):
        state_dict = dict(self.model.named_parameters())
        
        shapes = []
        tensors = []
        for p_name in param_names:
            if p_name not in state_dict:
                raise ValueError(f"Parameter {p_name} not found in model.")
            p_tensor = state_dict[p_name]
            shapes.append(p_tensor.shape)
            tensors.append(p_tensor.detach().clone().flatten())
            
        phi_vector = torch.cat(tensors).to(self.device).unsqueeze(0)
        num_params = phi_vector.shape[1]
        
        # Use a properly scaled random normal bounding to uniformly wake up all parameter streams across depth bounds natively!
        A_init = torch.randn(self.default_latent_dim, num_params, device=self.device) / (num_params ** 0.5)
        A = nn.Parameter(A_init)
        phi = nn.Parameter(phi_vector, requires_grad=False)
        
        self.block_names.append(block_name)
        self.A_list.append(A)
        self.phi_list.append(phi)

        # Optionally add a nonlinear residual correction for this block
        if self._use_nonlinear:
            corr = NonlinearCorrection(
                latent_dim=self.default_latent_dim,
                param_dim=num_params,
                h=self._lifting_h,
                activation=self._lifting_activation,
                output_scale=self._lifting_output_scale,
                zero_init_output=self._lifting_zero_init,
            ).to(self.device)
            self.correction_list.append(corr)
            logger.info(
                f"AURA: NonlinearCorrection for '{block_name}': "
                f"h={corr.fc1.out_features}, act={self._lifting_activation}, "
                f"params={sum(p.numel() for p in corr.parameters())}"
            )
        
        # Append metadata functionally
        self._meta_param_names_tuple += (tuple(param_names),)
        self._meta_shapes_tuple += (tuple(shapes),)
        self._meta_latent_dims_tuple += (self.default_latent_dim,)
        import math
        self._meta_numels_tuple += (tuple(math.prod(s) for s in shapes),)
        
        # Init Kalman state for this block
        z = torch.zeros(self.default_latent_dim, device=self.device)
        cov = torch.eye(self.default_latent_dim, device=self.device) * self.init_cov_scale
        
        self.z_list.append(z)
        self.cov_list.append(cov)
        self.z_init_list.append(z.clone())
        self.cov_init_list.append(cov.clone())
        
        logger.info(f"AURA: Registered block '{block_name}' with {num_params} params, latent_dim={self.default_latent_dim}")

    def _setup_slicing(self):
        self.can_slice = False
        # stage_order gives the depth index used by the head/tail builders below.
        # fc is the final "stage" (deepest), so the earliest adapted stage determines
        # how much of the backbone can be cached as a frozen head.
        # Indices: stage_1=0, stage_2=1, stage_3=2, stage_4=3, fc=4.
        earliest_stage_idx = 999
        unknown_blocks = []
        for b in self.block_names:
            if b in STAGE_ORDER:
                earliest_stage_idx = min(earliest_stage_idx, STAGE_ORDER.index(b))
            elif b == "shared_all_bn":
                earliest_stage_idx = 0
            else:
                # A literal name-filter block (e.g. "layer4_2_bn3"): its depth is
                # unknown to the stage ordering, so caching a head could silently
                # freeze the very parameters being adapted.
                unknown_blocks.append(b)

        inner, norm_layer, post_layer = unwrap_backbone(self.model)
        has_norm_wrapper = norm_layer is not None

        if is_vit(inner):
            # ViT: derive the cut from the adapted parameter names themselves, which
            # also covers literal name filters such as "blocks.11.norm2".
            self._setup_slicing_vit(inner, norm_layer, post_layer)
            return

        if unknown_blocks:
            logger.warning(
                f"AURA: blocks {unknown_blocks} are not part of the stage ordering — "
                f"disabling graph splicing to avoid caching adapted parameters."
            )
            return

        if earliest_stage_idx == 0 or earliest_stage_idx == 999:
            return

        is_tv = hasattr(inner, 'layer2') and hasattr(inner, 'layer3')
        is_rb = hasattr(inner, 'block2') and hasattr(inner, 'block3')

        if not (is_tv or is_rb): return

        # Does this TV backbone have a layer4 (ResNet-50) or not (WRN-28-10)?
        tv_has_layer4 = is_tv and hasattr(inner, 'layer4')
        # Analogous RB check (none of the current backbones have block4, but keep symmetric).
        rb_has_block4 = is_rb and hasattr(inner, 'block4')

        self.can_slice = True
        self.inner_model = inner

        if is_tv:
            # Torchvision ResNet-style head/tail.
            # Head runs all stages strictly BEFORE the earliest adapted stage.
            # Tail replays from the earliest adapted stage through fc.
            # idx 0 -> stage_1, 1 -> stage_2, 2 -> stage_3, 3 -> stage_4, 4 -> fc.
            def head_impl(self_outer, x):
                if has_norm_wrapper: x = norm_layer(x)
                # Typical torchvision ResNet stem: conv1 -> bn1 -> relu -> maxpool.
                x = inner.conv1(x)
                if hasattr(inner, 'bn1'): x = inner.bn1(x)
                if hasattr(inner, 'relu'): x = inner.relu(x)
                if hasattr(inner, 'maxpool'): x = inner.maxpool(x)
                if earliest_stage_idx > 0: x = inner.layer1(x)
                if earliest_stage_idx > 1: x = inner.layer2(x)
                if earliest_stage_idx > 2: x = inner.layer3(x)
                if earliest_stage_idx > 3 and tv_has_layer4: x = inner.layer4(x)
                return x

            def tail_impl(self_outer, z_frozen):
                x = z_frozen
                if earliest_stage_idx <= 1: x = inner.layer2(x)
                if earliest_stage_idx <= 2: x = inner.layer3(x)
                if earliest_stage_idx <= 3 and tv_has_layer4: x = inner.layer4(x)
                if earliest_stage_idx <= 4:
                    # Final pooling + classifier. WRN uses bn+relu+avgpool+view; TV ResNet uses avgpool+flatten.
                    if tv_has_layer4:
                        # Torchvision ResNet-50 path: avgpool is AdaptiveAvgPool2d, then flatten, then fc.
                        x = inner.avgpool(x)
                        x = torch.flatten(x, 1)
                    else:
                        # RobustBench WRN-28-10 TV-style path (no layer4): final bn + relu before avgpool.
                        if hasattr(inner, 'bn'): x = inner.bn(x)
                        if hasattr(inner, 'relu'): x = inner.relu(x)
                        x = inner.avgpool(x)
                        x = x.view(x.size(0), -1)
                out = inner.fc(x)
                # ImageNet-R/A/V2/D109 mask the 1000-way logits down to the subset
                return post_layer(out) if post_layer is not None else out
        else:
            def head_impl(self_outer, x):
                if has_norm_wrapper: x = norm_layer(x)
                out = inner.conv1(x)
                if earliest_stage_idx > 0: out = inner.block1(out)
                if earliest_stage_idx > 1: out = inner.block2(out)
                if earliest_stage_idx > 2: out = inner.block3(out)
                if earliest_stage_idx > 3 and rb_has_block4: out = inner.block4(out)
                return out

            def tail_impl(self_outer, z_frozen):
                out = z_frozen
                if earliest_stage_idx <= 1: out = inner.block2(out)
                if earliest_stage_idx <= 2: out = inner.block3(out)
                if earliest_stage_idx <= 3 and rb_has_block4: out = inner.block4(out)
                if earliest_stage_idx <= 4:
                    out = inner.relu(inner.bn1(out))
                    out = F.adaptive_avg_pool2d(out, (1, 1))
                    out = torch.flatten(out, 1)
                out = inner.fc(out)
                return post_layer(out) if post_layer is not None else out

        self.head_fn = head_impl
        self.tail_fn = tail_impl
        self.original_forward = self.model.forward
        logger.info(f"AURA: Graph splicing initialized successfully. Truncating up to stage block {earliest_stage_idx}.")

    def _vit_slice_cut(self, num_blocks):
        """
        Shallowest transformer block index touched by any adapted parameter.

        Derived from the registered parameter names rather than the stage labels,
        so stage groups, `shared_all_bn` and literal name filters (e.g.
        "blocks.11.norm2") all yield a cut that keeps every adapted parameter in
        the replayed tail. Returns 0 (no caching possible) if anything before the
        block list is adapted.
        """
        cut = num_blocks
        pre_block_segments = {"patch_embed", "norm_pre", "pos_embed", "cls_token"}
        for param_names in self._meta_param_names_tuple:
            for name in param_names:
                segments = name.split(".")
                if "blocks" in segments:
                    pos = segments.index("blocks")
                    if pos + 1 < len(segments) and segments[pos + 1].isdigit():
                        cut = min(cut, int(segments[pos + 1]))
                    else:
                        return 0
                elif any(seg in pre_block_segments for seg in segments):
                    return 0
                # post-block params (norm, fc_norm, head) never lower the cut
        return cut

    def _setup_slicing_vit(self, core, norm_layer, post_layer):
        """
        Graph splicing for timm VisionTransformer backbones.

        Head (frozen, cached): input norm -> patch embed -> pos embed -> pre-norm
                               -> transformer blocks [0, k)
        Tail (re-executed under the Jacobian): blocks [k, L) -> final norm -> pool
                                               -> fc_norm -> head (-> class mask)

        k is the shallowest block holding an adapted parameter, so every adapted
        parameter lives in the tail. With STAGE_SELECTION=["fc"] the whole
        backbone is cached and only the classifier is replayed.
        """
        num_blocks = len(core.blocks)
        k = self._vit_slice_cut(num_blocks)

        if k == 0:
            return  # earliest adapted stage starts at block 0 — nothing to cache

        def head_impl(self_outer, x):
            if norm_layer is not None: x = norm_layer(x)
            x = core.patch_embed(x)
            x = core._pos_embed(x)
            if hasattr(core, 'patch_drop'): x = core.patch_drop(x)
            if hasattr(core, 'norm_pre'): x = core.norm_pre(x)
            for blk in core.blocks[:k]:
                x = blk(x)
            return x

        def tail_impl(self_outer, feats):
            x = feats
            for blk in core.blocks[k:]:
                x = blk(x)
            x = core.norm(x)
            # Pooling: timm >= 1.0 exposes `pool()`; older versions only `global_pool`
            if callable(getattr(core, 'pool', None)):
                x = core.pool(x)
            else:
                num_prefix = getattr(core, 'num_prefix_tokens', 1)
                x = x[:, num_prefix:].mean(dim=1) if getattr(core, 'global_pool', 'token') == 'avg' else x[:, 0]
            if hasattr(core, 'fc_norm'): x = core.fc_norm(x)
            if hasattr(core, 'head_drop'): x = core.head_drop(x)
            out = core.head(x)
            return post_layer(out) if post_layer is not None else out

        self.can_slice = True
        self.inner_model = core
        self.head_fn = head_impl
        self.tail_fn = tail_impl
        self.original_forward = self.model.forward
        logger.info(
            f"AURA: ViT graph splicing initialized — caching patch embed + blocks [0,{k}) "
            f"of {num_blocks}; Jacobian replays blocks [{k},{num_blocks}) + head."
        )

    def _cache_init_state(self):
        self._aura_fc_z_init_list = [z.detach().clone() for z in self.z_list]
        self._aura_fc_cov_init_list = [c.detach().clone() for c in self.cov_list]

    def reset(self):
        for model, model_state in zip(self.models, self.model_states):
            model.load_state_dict(model_state, strict=True)
        self.z_list = [z.detach().clone() for z in self._aura_fc_z_init_list]
        self.cov_list = [c.detach().clone() for c in self._aura_fc_cov_init_list]

    def _load_and_validate_checkpoint(self, ckpt_path):
        ckpt_file = ckpt_path + "/last_params.pth" if not ckpt_path.endswith(".pth") else ckpt_path
        state = torch.load(ckpt_file, map_location=self.device)
        
        # --- Topological validation ---
        ckpt_blocks = state.get("blocks", {})
        if set(ckpt_blocks.keys()) != set(self.block_names):
            raise ValueError(
                f"AURA: Checkpoint blocks {sorted(ckpt_blocks.keys())} "
                f"do not match configured blocks {sorted(self.block_names)}!"
            )

        for i, b_name in enumerate(self.block_names):
            b_state = ckpt_blocks[b_name]
            if b_state["A"].shape != self.A_list[i].shape:
                raise ValueError(
                    f"AURA: Checkpoint A shape {b_state['A'].shape} "
                    f"mismatches configured {self.A_list[i].shape} for block '{b_name}'!"
                )
            self.A_list[i].data = b_state["A"].to(self.device)
            self.phi_list[i].data = b_state["phi"].to(self.device)

            # Load nonlinear correction if present
            if self._use_nonlinear:
                if "correction" in b_state:
                    corr_sd = {k: v.to(self.device) for k, v in b_state["correction"].items()}
                    self.correction_list[i].load_state_dict(corr_sd)
                    logger.info(f"  [{b_name}] Loaded NonlinearCorrection state_dict")
                else:
                    logger.warning(
                        f"  [{b_name}] Nonlinear mode but checkpoint has no 'correction' key "
                        f"— correction stays zero-initialized (linear start)."
                    )

        # --- Re-cache tuples so all downstream code uses the loaded values ---
        self._A_tuple = tuple(self.A_list)
        self._phi_tuple = tuple(self.phi_list)

        # --- Freeze and eval corrections for online TTA (only z evolves) ---
        if self._use_nonlinear and len(self.correction_list) > 0:
            for corr in self.correction_list:
                corr.requires_grad_(False)
                corr.eval()
            self._corrections_frozen = True
            self._correction_tuple = tuple(self.correction_list)
            logger.info("AURA: NonlinearCorrection frozen and set to eval for online TTA.")

        # --- Verification: confirm the loaded values are actually used ---
        for i, b_name in enumerate(self.block_names):
            a_match = torch.equal(self._A_tuple[i].data, self.A_list[i].data)
            phi_match = torch.equal(self._phi_tuple[i].data, self.phi_list[i].data)
            logger.info(
                f"  [{b_name}] A shape={tuple(self.A_list[i].shape)}, "
                f"A_norm={self.A_list[i].data.norm():.4f}, "
                f"phi_norm={self.phi_list[i].data.norm():.4f}, "
                f"tuple_synced={a_match and phi_match}"
            )
            if not (a_match and phi_match):
                raise RuntimeError(
                    f"AURA: CRITICAL — cached tuple for block '{b_name}' "
                    f"does not match loaded A_list/phi_list after checkpoint load!"
                )

        # --- Metadata consistency checks (warn, don't crash) ---
        ckpt_latent = state.get("latent_dim")
        if ckpt_latent is not None and ckpt_latent != self.default_latent_dim:
            logger.warning(
                f"AURA: Checkpoint latent_dim={ckpt_latent} differs from "
                f"config latent_dim={self.default_latent_dim}"
            )
        ckpt_nc = state.get("num_classes")
        if ckpt_nc is not None and ckpt_nc != self.num_classes:
            logger.warning(
                f"AURA: Checkpoint num_classes={ckpt_nc} differs from "
                f"current num_classes={self.num_classes}"
            )
        ckpt_gamma = state.get("gamma")
        if ckpt_gamma is not None and ckpt_gamma != self.gamma:
            logger.warning(
                f"AURA: Checkpoint gamma={ckpt_gamma} differs from "
                f"config gamma={self.gamma}"
            )
        ckpt_lifting = state.get("lifting_type", "linear")
        if ckpt_lifting != self._lifting_type:
            logger.warning(
                f"AURA: Checkpoint lifting_type='{ckpt_lifting}' differs from "
                f"config LIFTING_TYPE='{self._lifting_type}'."
            )

        logger.info(f"AURA: Successfully loaded and verified offline checkpoint from {ckpt_file}")

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)
        for m in self.model.modules():
            if hasattr(m, 'inplace'):
                m.inplace = False
        # Transformer backbones: the fused SDPA kernels support neither forward-mode
        # AD nor vmap, which both jacfwd and chunked jacrev rely on.
        disable_fused_attention(self.model)

    def train(self, mode=True):
        super().train(mode)
        # Protect base model from jacrev inplace training traces recursively applying state buffers
        if hasattr(self, 'model'):
            self.model.eval()
        # Keep correction in eval when it has been frozen (online mode)
        if mode and getattr(self, '_corrections_frozen', False):
            for corr in self.correction_list:
                corr.eval()
        return self

    def collect_params(self):
        return [], []

    def setup_optimizer(self):
        return None

    def copy_model_and_optimizer(self):
        model_states = [deepcopy(model.state_dict()) for model in self.models]
        return model_states, None

    @staticmethod
    def _reconstruct_params(z_global, A_tuple, phi_tuple, param_names_tuple, shapes_tuple, latent_dims_tuple, numels_tuple, correction_tuple=None):
        zs = torch.split(z_global, latent_dims_tuple, dim=-1)
        overrides = {}
        for i, (z, A, phi, pnames, shapes, numels) in enumerate(
            zip(zs, A_tuple, phi_tuple, param_names_tuple, shapes_tuple, numels_tuple)
        ):
            # Linear base: theta = phi + z @ A
            theta = z @ A + phi.squeeze(0)
            # Optional nonlinear residual correction: theta += delta(z)
            if correction_tuple is not None:
                theta = theta + correction_tuple[i](z)
            theta_split = torch.split(theta, numels, dim=-1)
            for name, shape, t in zip(pnames, shapes, theta_split):
                overrides[name] = t.view(t.shape[:-1] + shape)
        return overrides

    @staticmethod
    def _obs_fn(z_global, x, base_model, A_tuple, phi_tuple, param_names_tuple, shapes_tuple, latent_dims_tuple, numels_tuple, correction_tuple=None):
        overrides_dict = AURA._reconstruct_params(
            z_global, A_tuple, phi_tuple, param_names_tuple, shapes_tuple, latent_dims_tuple, numels_tuple,
            correction_tuple=correction_tuple,
        )
        y_pred = torch.func.functional_call(base_model, overrides_dict, (x,), strict=False)
        return y_pred

    def _aura_fc_update_step(self, x, y, z_list, cov_list, y_pred_probs=None):
        z_global = torch.cat(z_list)
        
        # We fetch the exact static A/phi state from mathematically cached tuples
        A_tuple = self._A_tuple
        phi_tuple = self._phi_tuple
        # Nonlinear correction tuple (None for linear mode)
        correction_tuple = self._correction_tuple
        
        if y_pred_probs is None:
            y_pred_batch = self._obs_fn(
                z_global, x, self.model, A_tuple, phi_tuple,
                self._meta_param_names_tuple, self._meta_shapes_tuple,
                self._meta_latent_dims_tuple, self._meta_numels_tuple,
                correction_tuple,
            )
            y_pred = F.softmax(y_pred_batch[0], dim=0)
        else:
            y_pred = y_pred_probs

        # Jacobian of obs fn w.r.t. the latent state (jacrev by default, jacfwd when
        # num_classes is large — chosen once in __init__; see self._jac_fn).
        # NOTE: for nonlinear mode the Jacobian traces through both the correction MLP
        # and the backbone functional_call — measure actual runtime rather than assuming.
        H_global_batch = self._jac_fn(AURA._obs_fn, argnums=0)(
            z_global, x, self.model, A_tuple, phi_tuple,
            self._meta_param_names_tuple, self._meta_shapes_tuple,
            self._meta_latent_dims_tuple, self._meta_numels_tuple,
            correction_tuple,
        )
        H_global = H_global_batch[0]
        
        H_blocks = torch.split(H_global, self._meta_latent_dims_tuple, dim=1)
        
        z_new_list = []
        cov_new_list = []
        
        for i in range(len(self.block_names)):
            z_i, cov_i, H_i = z_list[i], cov_list[i], H_blocks[i]
            z_init, cov_init = self.z_init_list[i], self.cov_init_list[i]
            
            predict_mean = self.gamma * z_i + (1 - self.gamma) * z_init
            predict_cov = (self.gamma ** 2) * cov_i + (1 - self.gamma ** 2) * cov_init

            S = self.R + H_i @ predict_cov @ H_i.T
            C = predict_cov @ H_i.T
            K = torch.linalg.solve(S, C.T).T

            z_new = predict_mean + K @ (y - y_pred)
            cov_new = (torch.eye(self._meta_latent_dims_tuple[i], device=self.device) - K @ H_i) @ predict_cov
            
            z_new_list.append(z_new)
            cov_new_list.append(cov_new)

        return z_new_list, cov_new_list

    def forward_offline_step(self, ctx_imgs, ctx_labels, qry_imgs, qry_labels, z_list, cov_list):
        # Mirror the online forward_and_adapt slicing pattern: when can_slice
        # is set, precompute frozen-head features under no_grad and run jacrev
        # only over the tail. Eliminates per-class vmap activations through
        # the entire backbone.
        if getattr(self, "can_slice", False):
            import types
            with torch.no_grad():
                z_frozen_ctx = self.head_fn(self.model, ctx_imgs)
                z_frozen_qry = self.head_fn(self.model, qry_imgs)
            self.model.forward = types.MethodType(self.tail_fn, self.model)
            try:
                return self._forward_offline_step_inner(
                    z_frozen_ctx, ctx_labels, z_frozen_qry, qry_labels, z_list, cov_list
                )
            finally:
                self.model.forward = self.original_forward
        else:
            return self._forward_offline_step_inner(
                ctx_imgs, ctx_labels, qry_imgs, qry_labels, z_list, cov_list
            )

    def _forward_offline_step_inner(self, ctx_inp, ctx_labels, qry_inp, qry_labels, z_list, cov_list):
        x_c = ctx_inp[0:1]
        y_c = ctx_labels[0]

        y = torch.zeros(self.num_classes, device=self.device, dtype=x_c.dtype)
        y[y_c.long()] = 1.0

        z_new_list, cov_new_list = self._aura_fc_update_step(x_c, y, z_list, cov_list)

        z_global_new = torch.cat(z_new_list)
        A_tuple = self._A_tuple
        phi_tuple = self._phi_tuple

        logits_q = self._obs_fn(
            z_global_new, qry_inp, self.model, A_tuple, phi_tuple,
            self._meta_param_names_tuple, self._meta_shapes_tuple,
            self._meta_latent_dims_tuple, self._meta_numels_tuple,
            self._correction_tuple,
        )
        loss_q = F.cross_entropy(logits_q, qry_labels)

        return logits_q, loss_q, z_new_list, cov_new_list

    @torch.no_grad()
    def forward_and_adapt(self, x, labels=None):
        """
        Two-phase online adaptation:
          Phase 1 (Adapt):  Perform NUM_ADAPT_SAMPLES supervised Kalman updates
                            using ground-truth labels on a subset of the batch.
          Phase 2 (Evaluate): Run a single batched forward pass on the FULL batch
                              using the updated latent state, and return those outputs.

        Args:
            x      : list with x[0] = imgs [B, C, H, W]
            labels : [B] integer ground-truth class labels (required)
        """
        imgs_test = x[0]
        batch_size = imgs_test.shape[0]

        if labels is None:
            logger.warning(
                "AURA.forward_and_adapt called WITHOUT labels — "
                "falling back to pseudo-labels. Pass true labels for supervised mode."
            )

        # Determine how many samples to use for adaptation (from config)
        num_adapt_samples = getattr(self.cfg.AURA, "NUM_ADAPT_SAMPLES", -1)
        if num_adapt_samples > 0:
            num_updates = min(num_adapt_samples, batch_size)
        else:
            num_updates = batch_size

        z_list = [z.clone() for z in self.z_list]
        cov_list = [c.clone() for c in self.cov_list]

        # ================================================================
        # Phase 1: Adaptation — update z/cov on the configured subset
        # ================================================================
        if getattr(self, "can_slice", False):
            import types
            # Cache the frozen head features for the adaptation subset only
            adapt_imgs = imgs_test[:num_updates]
            with torch.no_grad():
                z_frozen_adapt = self.head_fn(self.model, adapt_imgs)
            self.model.forward = types.MethodType(self.tail_fn, self.model)
            adapt_input = z_frozen_adapt
        else:
            adapt_input = imgs_test[:num_updates]

        try:
            self._timing_units_per_frame = num_updates   # updates performed per frame
            for i in range(num_updates):
                _should_time = (
                    self._timing_enabled
                    and self._timing_sample_counter >= self._timing_warmup
                    and len(self._timing_results) < self._timing_target
                )
                if _should_time:
                    torch.cuda.synchronize()
                    _t0 = time.perf_counter()

                x_i = adapt_input[i:i+1]
                z_global = torch.cat(z_list)

                # Forward pass to get prediction for this sample
                y_pred_batch = self._obs_fn(
                    z_global, x_i, self.model,
                    self._A_tuple, self._phi_tuple,
                    self._meta_param_names_tuple, self._meta_shapes_tuple,
                    self._meta_latent_dims_tuple, self._meta_numels_tuple,
                    self._correction_tuple,
                )
                y_pred_probs = F.softmax(y_pred_batch[0], dim=0)

                # Build one-hot observation from ground truth (or pseudo-label fallback)
                y = torch.zeros(self.num_classes, device=self.device)
                if labels is not None:
                    y[labels[i].long()] = 1.0
                else:
                    y[y_pred_probs.argmax()] = 1.0

                z_list, cov_list = self._aura_fc_update_step(
                    x_i, y, z_list, cov_list, y_pred_probs=y_pred_probs
                )

                if _should_time:
                    torch.cuda.synchronize()
                    self._timing_results.append(time.perf_counter() - _t0)
                self._timing_sample_counter += 1
        finally:
            if getattr(self, "can_slice", False):
                self.model.forward = self.original_forward

        # Persist the updated latent state
        self.z_list = [z.detach() for z in z_list]
        self.cov_list = [c.detach() for c in cov_list]

        # ================================================================
        # Phase 2: Evaluation — single batched forward on the FULL batch
        # ================================================================
        z_global_final = torch.cat(self.z_list)
        outputs = self._obs_fn(
            z_global_final, imgs_test, self.model,
            self._A_tuple, self._phi_tuple,
            self._meta_param_names_tuple, self._meta_shapes_tuple,
            self._meta_latent_dims_tuple, self._meta_numels_tuple,
            self._correction_tuple,
        )
        return outputs

    @torch.no_grad()
    def forward_eval(self, imgs):
        """
        Evaluate classification logits on a batch of test images using the current
        adapted latent state z without modifying latent state z or covariance P.
        """
        z_global_final = torch.cat(self.z_list)
        return self._obs_fn(
            z_global_final, imgs, self.model,
            self._A_tuple, self._phi_tuple,
            self._meta_param_names_tuple, self._meta_shapes_tuple,
            self._meta_latent_dims_tuple, self._meta_numels_tuple,
            self._correction_tuple,
        )
