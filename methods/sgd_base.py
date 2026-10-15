"""
SGD_Base — Online SGD / Adam adaptation baseline.

A clean, standalone online adaptation method that performs per-sample sequential
gradient-based updates.  Independent of AURA_FC / AURA internals — no latent
state, no Kalman filter, no offline training stage.

Two-phase `forward_and_adapt`:
  Phase 1 (Adapt)   – Perform supervised gradient updates on a configurable
                       subset of the batch, one sample at a time.
  Phase 2 (Evaluate) – Run a single batched forward pass on the *full* batch
                        using the updated model, and return those outputs.
"""

import logging
import time
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.base import TTAMethod
from models.arch_utils import is_vit, unwrap_backbone, vit_stage_modules
from utils.registry import ADAPTATION_REGISTRY

logger = logging.getLogger(__name__)


@ADAPTATION_REGISTRY.register()
class SGD_Base(TTAMethod):
    """Online per-sample SGD / Adam adaptation baseline."""

    def __init__(self, cfg, model, num_classes):
        # Store SGD_BASE config before super().__init__ because
        # configure_model / collect_params / setup_optimizer are called there.
        self._sgd_cfg = cfg.SGD_BASE

        super().__init__(cfg, model, num_classes)

        self.num_updates_per_sample = self._sgd_cfg.NUM_UPDATES_PER_SAMPLE
        self.num_adapt_samples = self._sgd_cfg.NUM_ADAPT_SAMPLES
        self.loss_type = getattr(self._sgd_cfg, "LOSS", "cross_entropy")

        logger.info(
            f"SGD_Base: loss={self.loss_type}, "
            f"updates_per_sample={self.num_updates_per_sample}, "
            f"adapt_samples={self.num_adapt_samples}, "
            f"stage_selection={list(self._sgd_cfg.STAGE_SELECTION)}"
        )

        # Timing state
        self._timing_enabled = cfg.TIMING.ENABLED
        self._timing_warmup = cfg.TIMING.WARMUP_SAMPLES
        self._timing_target = cfg.TIMING.NUM_MEASUREMENTS
        self._timing_results = []
        self._timing_sample_counter = 0

    # ------------------------------------------------------------------
    # Model configuration — freeze everything, then selectively unfreeze
    # ------------------------------------------------------------------
    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)

        stage_selection = list(self._sgd_cfg.STAGE_SELECTION)

        if "all" in stage_selection:
            self.model.requires_grad_(True)
            # Keep BN in eval mode (use frozen running stats) — option (A)
            logger.info("SGD_Base: STAGE_SELECTION='all' — all parameters trainable.")
            return

        # Resolve the inner backbone (skip normalization / class-mask wrappers)
        inner, _, _ = unwrap_backbone(self.model)

        # Map stage names to WRN module references
        # WRN (robustbench "Standard" for CIFAR-10) uses:
        #   conv1, block1, block2, block3, bn1, relu, fc
        stage_to_modules = {}
        if is_vit(inner):
            # timm VisionTransformer: transformer blocks split into 4 contiguous
            # groups (ViT-B/16 -> 0-2, 3-5, 6-8, 9-11); stage_1 also owns the
            # patch embedding, stage_4 the final norm(s).
            stage_to_modules = vit_stage_modules(inner)
        elif hasattr(inner, "block1"):
            # RobustBench WRN layout
            stage_to_modules = {
                "fc":      [inner.fc],
                "stage_1": [inner.block1],
                "stage_2": [inner.block2],
                "stage_3": [inner.block3, inner.bn1],
            }
        elif hasattr(inner, "layer1"):
            # torchvision-style ResNet layout
            stage_to_modules = {
                "fc":      [inner.fc],
                "stage_1": [inner.layer1],
                "stage_2": [inner.layer2],
                "stage_3": [inner.layer3],
            }
        elif hasattr(inner, "stage_1") and hasattr(inner, "classifier"):
            # Hendrycks2020AugMix_ResNeXt layout (top-level: conv_1_3x3, bn_1,
            # stage_1..3, avgpool, classifier)
            stage_to_modules = {
                "fc":      [inner.classifier],
                "stage_1": [inner.stage_1],
                "stage_2": [inner.stage_2],
                "stage_3": [inner.stage_3],
            }

        for stage in stage_selection:
            if stage not in stage_to_modules:
                raise ValueError(
                    f"SGD_Base: Unknown stage '{stage}'. "
                    f"Choose from {list(stage_to_modules.keys())} or 'all'."
                )
            for module in stage_to_modules[stage]:
                module.requires_grad_(True)
                logger.info(f"SGD_Base: Enabled gradients for stage '{stage}' module: {type(module).__name__}")

    # ------------------------------------------------------------------
    # Collect only the parameters that were unfrozen by configure_model
    # ------------------------------------------------------------------
    def collect_params(self):
        params = []
        names = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params.append(param)
                names.append(name)
        logger.info(f"SGD_Base: Collected {len(params)} trainable parameter groups.")
        return params, names

    # ------------------------------------------------------------------
    # Create optimizer from SGD_BASE config (not global OPTIM)
    # ------------------------------------------------------------------
    def setup_optimizer(self):
        if len(self.params) == 0:
            return None

        optim_method = getattr(self._sgd_cfg, "OPTIM_METHOD", "Adam")
        lr = self._sgd_cfg.LR
        wd = self._sgd_cfg.WEIGHT_DECAY

        if optim_method == "Adam":
            optimizer = torch.optim.Adam(
                self.params,
                lr=lr,
                weight_decay=wd,
            )
        elif optim_method == "AdamW":
            optimizer = torch.optim.AdamW(
                self.params,
                lr=lr,
                weight_decay=wd,
            )
        elif optim_method == "SGD":
            optimizer = torch.optim.SGD(
                self.params,
                lr=lr,
                momentum=self._sgd_cfg.MOMENTUM,
                weight_decay=wd,
                nesterov=self._sgd_cfg.NESTEROV,
            )
        else:
            raise ValueError(f"SGD_Base: Unsupported optimizer '{optim_method}'.")

        logger.info(f"SGD_Base: Using {optim_method} optimizer with lr={lr}, wd={wd}")
        return optimizer

    # ------------------------------------------------------------------
    # Two-phase forward_and_adapt
    # ------------------------------------------------------------------
    @torch.enable_grad()
    def forward_and_adapt(self, x, labels=None):
        """
        Two-phase online adaptation:
          Phase 1 (Adapt):    Per-sample supervised gradient updates on a
                              configured subset of the batch.
          Phase 2 (Evaluate): Single batched forward pass on the FULL batch
                              using the updated parameters.

        Args:
            x      : list with x[0] = imgs [B, C, H, W]
            labels : [B] integer ground-truth class labels (required for
                     cross_entropy loss; optional for entropy loss)
        """
        imgs_test = x[0]
        batch_size = imgs_test.shape[0]

        if labels is None and self.loss_type == "cross_entropy":
            logger.warning(
                "SGD_Base.forward_and_adapt called WITHOUT labels while using "
                "cross_entropy loss — skipping adaptation for this batch."
            )
            with torch.no_grad():
                return self.model(imgs_test)

        # Determine how many samples to use for adaptation
        if self.num_adapt_samples > 0:
            num_updates = min(self.num_adapt_samples, batch_size)
        else:
            num_updates = batch_size

        # ================================================================
        # Phase 1: Adaptation — per-sample gradient updates
        # ================================================================
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

            x_i = imgs_test[i:i+1]

            for _ in range(self.num_updates_per_sample):
                output_i = self.model(x_i)

                if self.loss_type == "cross_entropy":
                    loss = F.cross_entropy(output_i, labels[i:i+1])
                elif self.loss_type == "entropy":
                    # Unsupervised entropy minimization (like Tent)
                    probs = F.softmax(output_i, dim=1)
                    loss = -(probs * probs.log()).sum(dim=1).mean()
                else:
                    raise ValueError(f"SGD_Base: Unknown loss type '{self.loss_type}'")

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            if _should_time:
                torch.cuda.synchronize()
                self._timing_results.append(time.perf_counter() - _t0)
            self._timing_sample_counter += 1

        # ================================================================
        # Phase 2: Evaluation — single batched forward on the FULL batch
        # ================================================================
        with torch.no_grad():
            outputs = self.model(imgs_test)

        return outputs

    # ------------------------------------------------------------------
    # copy / reset — use base class defaults for continual adaptation
    # ------------------------------------------------------------------
    def copy_model_and_optimizer(self):
        model_states = [deepcopy(model.state_dict()) for model in self.models]
        optimizer_state = deepcopy(self.optimizer.state_dict()) if self.optimizer else None
        return model_states, optimizer_state

    def reset(self):
        if self.model_states is None or self.optimizer_state is None:
            raise Exception("SGD_Base: Cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer()
