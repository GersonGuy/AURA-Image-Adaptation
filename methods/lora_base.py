"""
LoRA_Base — Online LoRA adaptation baseline.

A parameter-efficient comparison to AURA.  All backbone parameters are frozen;
only lightweight LoRA adapter weights are trained online via supervised
cross-entropy on the same labeled budget used by AURA and SGD_Base.

Design:
  LoRALinear  : wraps nn.Linear.  A is (in_f, r), B is (r, out_f, zero-init).
  LoRAConv2d  : wraps nn.Conv2d (groups=1 only).
                A-conv preserves the original kernel_size/stride/padding/
                dilation, mapping in_channels → r.
                B-conv is a 1×1 conv mapping r → out_channels (zero-init).
                Output shape is identical to the original convolution.
  inject_lora : replaces target modules in-place; skips groups>1 with a
                logged warning.
  LoRA_Base   : TTAMethod that freezes the model, injects LoRA, and runs
                per-sample supervised gradient updates.  Parameters are
                never reset across the complete online stream.
"""

import logging
import time
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-rank wrapper modules
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """
    LoRA wrapper for nn.Linear.

    Forward:  F.linear(x, W0 + (alpha/r) * (A @ B).T, bias)
    where A is (in_features, r)  — Kaiming-uniform init
          B is (r, out_features) — zero-init so delta = 0 at t=0.
    """

    def __init__(self, linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        assert rank > 0
        self.rank = rank
        self.scale = alpha / rank

        # Freeze original parameters (kept as direct tensor refs)
        self.weight = linear.weight   # (out_f, in_f)
        self.bias   = linear.bias     # (out_f,) or None
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

        in_f  = linear.in_features
        out_f = linear.out_features
        device = linear.weight.device
        dtype  = linear.weight.dtype

        self.lora_A = nn.Parameter(torch.empty(in_f, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

        self.lora_B = nn.Parameter(torch.zeros(rank, out_f, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta_w = self.lora_A @ self.lora_B          # (in_f, out_f)
        w = self.weight + self.scale * delta_w.T      # (out_f, in_f)
        return F.linear(x, w, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in={self.lora_A.shape[0]}, out={self.lora_B.shape[1]}, "
            f"rank={self.rank}, scale={self.scale:.4g}"
        )


class LoRAConv2d(nn.Module):
    """
    LoRA wrapper for nn.Conv2d (groups=1 only).

    A-conv: same kernel_size / stride / padding / dilation as the original,
            maps in_channels → r  (Kaiming-uniform init, no bias).
    B-conv: 1×1 kernel, maps r → out_channels  (zero-init, no bias).

    The spatial dimensions of conv_a(x) exactly match those of original_conv(x),
    so conv_b (1×1, no stride) preserves that shape.

    Forward:  original_conv(x)  +  (alpha/r) * conv_b(conv_a(x))

    Raises ValueError at construction if groups > 1.
    """

    def __init__(self, conv: nn.Conv2d, rank: int, alpha: float):
        super().__init__()
        if conv.groups > 1:
            raise ValueError(
                f"LoRAConv2d requires groups=1 (got groups={conv.groups})."
            )
        assert rank > 0
        self.rank  = rank
        self.scale = alpha / rank

        # Freeze original weight and bias
        self.weight = conv.weight   # (out_ch, in_ch, kH, kW)
        self.bias   = conv.bias     # (out_ch,) or None
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

        # Store geometry for the frozen-path F.conv2d call
        self.stride   = conv.stride
        self.padding  = conv.padding
        self.dilation = conv.dilation

        device = conv.weight.device
        dtype  = conv.weight.dtype

        # A-conv: same geometry, in_ch → r
        self.conv_a = nn.Conv2d(
            conv.in_channels, rank,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=1,
            bias=False,
        ).to(device=device, dtype=dtype)
        nn.init.kaiming_uniform_(self.conv_a.weight, a=5 ** 0.5)

        # B-conv: 1×1, r → out_ch, zero-init
        self.conv_b = nn.Conv2d(rank, conv.out_channels, kernel_size=1, bias=False
                                ).to(device=device, dtype=dtype)
        nn.init.zeros_(self.conv_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.conv2d(x, self.weight, self.bias,
                        self.stride, self.padding, self.dilation, 1)
        lora = self.conv_b(self.conv_a(x))
        return base + self.scale * lora

    def extra_repr(self) -> str:
        return (
            f"in={self.conv_a.in_channels}, out={self.conv_b.out_channels}, "
            f"rank={self.rank}, scale={self.scale:.4g}, "
            f"kernel={self.conv_a.kernel_size}"
        )


# ---------------------------------------------------------------------------
# Injection utility
# ---------------------------------------------------------------------------

def inject_lora(model: nn.Module, target_module_names: list,
                rank: int, alpha: float):
    """
    Replace target modules in *model* with LoRA wrappers in-place.

    Returns:
        injected (list[str])  : names of successfully wrapped modules.
        skipped  (list[str])  : names skipped (e.g. grouped convolutions).
    """
    target_set = set(target_module_names)
    injected, skipped = [], []

    for name, module in list(model.named_modules()):
        if name not in target_set:
            continue

        # Navigate to the parent so we can setattr the replacement
        parts  = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        attr = parts[-1]

        if isinstance(module, nn.Linear):
            wrapper = LoRALinear(module, rank, alpha)
            setattr(parent, attr, wrapper)
            injected.append(name)
            logger.info(
                f"LoRA: Injected LoRALinear at '{name}' "
                f"({module.in_features}→{module.out_features}, r={rank})"
            )

        elif isinstance(module, nn.Conv2d):
            if module.groups > 1:
                logger.warning(
                    f"LoRA: Skipping '{name}' — "
                    f"groups={module.groups} not supported "
                    f"(ResNeXt grouped convolution)."
                )
                skipped.append(name)
                continue
            wrapper = LoRAConv2d(module, rank, alpha)
            setattr(parent, attr, wrapper)
            injected.append(name)
            logger.info(
                f"LoRA: Injected LoRAConv2d at '{name}' "
                f"({module.in_channels}→{module.out_channels}, "
                f"k={module.kernel_size}, r={rank})"
            )

        else:
            logger.warning(
                f"LoRA: '{name}' is {type(module).__name__}, "
                f"not nn.Linear or nn.Conv2d — skipping."
            )
            skipped.append(name)

    return injected, skipped


# ---------------------------------------------------------------------------
# Scope → module name resolution
# ---------------------------------------------------------------------------

def _resolve_scope(model: nn.Module, scope: str, explicit: list) -> list:
    """
    Return the list of module names to inject LoRA into.

    If *explicit* is non-empty it is used directly (LORA.TARGET_MODULES).
    Otherwise *scope* must be one of: "fc" | "fc_stage1".

    Supported backbones
    -------------------
    WRN-28-10 (Standard, CIFAR-10-C)
        "fc"          → ["fc"]
        "fc_stage1"   → ["fc"] + all Conv2d inside block1

    Hendrycks2020AugMix_ResNeXt (CIFAR-100-C)
        "fc"          → ["classifier"]
        "fc_stage1"   → ["classifier"] + all Conv2d inside stage_1
                         (conv_conv layers have groups=4 and will be
                          auto-skipped by inject_lora with a warning)
    """
    if explicit:
        return list(explicit)

    # Unwrap possible Sequential(normalizer, backbone) wrapper
    inner = model
    if isinstance(inner, nn.Sequential) and len(inner) >= 2:
        inner = inner[1]

    is_wrn     = hasattr(inner, "block1") and hasattr(inner, "fc")
    is_resnext = hasattr(inner, "stage_1") and hasattr(inner, "classifier")

    if not is_wrn and not is_resnext:
        raise RuntimeError(
            "LoRA: Backbone family not recognised "
            "(expected WRN block1/fc or ResNeXt stage_1/classifier). "
            "Set LORA.TARGET_MODULES explicitly."
        )

    if is_wrn:
        fc_names = ["fc"]
        if scope == "fc":
            return fc_names
        stage1_convs = [
            name for name, mod in inner.named_modules()
            if name.startswith("block1.") and isinstance(mod, nn.Conv2d)
        ]
        if scope == "fc_stage1":
            return fc_names + stage1_convs
        raise ValueError(f"LoRA: Unknown scope '{scope}'. Choose 'fc' or 'fc_stage1'.")

    # is_resnext
    fc_names = ["classifier"]
    if scope == "fc":
        return fc_names
    stage1_convs = [
        name for name, mod in inner.named_modules()
        if name.startswith("stage_1.") and isinstance(mod, nn.Conv2d)
    ]
    if scope == "fc_stage1":
        return fc_names + stage1_convs
    raise ValueError(f"LoRA: Unknown scope '{scope}'. Choose 'fc' or 'fc_stage1'.")


# ---------------------------------------------------------------------------
# TTAMethod
# ---------------------------------------------------------------------------

@ADAPTATION_REGISTRY.register()
class LoRA_Base(TTAMethod):
    """
    Online LoRA adaptation baseline.

    Freezes all backbone parameters and injects LoRA adapters into the
    target modules (classifier only, or classifier + stage-1 conv layers).
    Only LoRA parameters are trained via per-sample supervised cross-entropy.
    BatchNorm layers remain frozen in eval mode throughout.
    Parameters are never reset across the full stream.
    """

    def __init__(self, cfg, model, num_classes):
        self._lora_cfg = cfg.LORA
        super().__init__(cfg, model, num_classes)

        self.num_updates_per_sample = self._lora_cfg.NUM_UPDATES_PER_SAMPLE
        self.num_adapt_samples      = self._lora_cfg.NUM_ADAPT_SAMPLES

        # Timing — same mechanism as SGD_Base
        self._timing_enabled  = cfg.TIMING.ENABLED
        self._timing_warmup   = cfg.TIMING.WARMUP_SAMPLES
        self._timing_target   = cfg.TIMING.NUM_MEASUREMENTS
        self._timing_results  = []
        self._timing_sample_counter = 0

        logger.info(
            f"LoRA_Base: scope={self._lora_cfg.SCOPE}, "
            f"rank={self._lora_cfg.RANK}, alpha={self._lora_cfg.ALPHA}, "
            f"lr={self._lora_cfg.LR}, optim={self._lora_cfg.OPTIM_METHOD}, "
            f"updates_per_sample={self.num_updates_per_sample}, "
            f"adapt_samples={self.num_adapt_samples}"
        )

    # ------------------------------------------------------------------
    # Model configuration
    # ------------------------------------------------------------------
    def configure_model(self):
        """Freeze backbone, inject LoRA adapters, keep BN in eval."""
        self.model.eval()
        self.model.requires_grad_(False)

        explicit = list(self._lora_cfg.TARGET_MODULES)
        scope    = self._lora_cfg.SCOPE
        targets  = _resolve_scope(self.model, scope, explicit)

        logger.info(
            f"LoRA_Base: Resolved {len(targets)} target module(s) "
            f"for scope '{scope}': {targets}"
        )

        injected, skipped = inject_lora(
            self.model, targets,
            rank=self._lora_cfg.RANK,
            alpha=self._lora_cfg.ALPHA,
        )

        if skipped:
            logger.info(
                f"LoRA_Base: {len(skipped)} module(s) skipped "
                f"(grouped convolutions unsupported): {skipped}"
            )
        logger.info(
            f"LoRA_Base: {len(injected)} module(s) successfully wrapped: "
            f"{injected}"
        )

        # Keep all BN layers frozen in eval mode
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.eval()
                m.requires_grad_(False)

    # ------------------------------------------------------------------
    # Collect only LoRA parameters
    # ------------------------------------------------------------------
    def collect_params(self):
        params, names = [], []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params.append(param)
                names.append(name)
        logger.info(
            f"LoRA_Base: {len(params)} trainable LoRA parameter tensors collected."
        )
        return params, names

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    def setup_optimizer(self):
        if len(self.params) == 0:
            return None
        method = self._lora_cfg.OPTIM_METHOD
        lr     = self._lora_cfg.LR
        wd     = self._lora_cfg.WEIGHT_DECAY
        if method == "Adam":
            opt = torch.optim.Adam(self.params, lr=lr, weight_decay=wd)
        elif method == "AdamW":
            opt = torch.optim.AdamW(self.params, lr=lr, weight_decay=wd)
        elif method == "SGD":
            opt = torch.optim.SGD(
                self.params, lr=lr,
                momentum=self._lora_cfg.MOMENTUM,
                weight_decay=wd,
                nesterov=self._lora_cfg.NESTEROV,
            )
        else:
            raise ValueError(f"LoRA_Base: Unsupported optimizer '{method}'.")
        logger.info(f"LoRA_Base: {method} optimizer, lr={lr}, wd={wd}")
        return opt

    # ------------------------------------------------------------------
    # forward_and_adapt
    # ------------------------------------------------------------------
    @torch.enable_grad()
    def forward_and_adapt(self, x, labels=None):
        """
        Phase 1 — Adapt: per-sample supervised CE on NUM_ADAPT_SAMPLES samples.
        Phase 2 — Evaluate: no-grad batched forward on the full batch.
        """
        imgs       = x[0]
        batch_size = imgs.shape[0]

        if labels is None:
            logger.warning(
                "LoRA_Base.forward_and_adapt called without labels — "
                "skipping adaptation."
            )
            with torch.no_grad():
                return self.model(imgs)

        num_adapt = (
            min(self.num_adapt_samples, batch_size)
            if self.num_adapt_samples > 0
            else batch_size
        )

        # Phase 1 — Adapt
        for i in range(num_adapt):
            _time_this = (
                self._timing_enabled
                and self._timing_sample_counter >= self._timing_warmup
                and len(self._timing_results) < self._timing_target
            )
            if _time_this:
                torch.cuda.synchronize()
                _t0 = time.perf_counter()

            x_i = imgs[i:i + 1]
            for _ in range(self.num_updates_per_sample):
                logits = self.model(x_i)
                loss   = F.cross_entropy(logits, labels[i:i + 1])
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            if _time_this:
                torch.cuda.synchronize()
                self._timing_results.append(time.perf_counter() - _t0)
            self._timing_sample_counter += 1

        # Phase 2 — Evaluate
        with torch.no_grad():
            return self.model(imgs)

    # ------------------------------------------------------------------
    # State management — continual (no reset across the stream)
    # ------------------------------------------------------------------
    def copy_model_and_optimizer(self):
        model_states    = [deepcopy(m.state_dict()) for m in self.models]
        optimizer_state = (
            deepcopy(self.optimizer.state_dict()) if self.optimizer else None
        )
        return model_states, optimizer_state

    def reset(self):
        """Restore initial snapshot (called only in episodic mode)."""
        if self.model_states is None or self.optimizer_state is None:
            raise Exception("LoRA_Base: Cannot reset without saved state.")
        self.load_model_and_optimizer()
