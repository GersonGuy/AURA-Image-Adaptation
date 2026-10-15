import time
import torch
import torch.nn as nn
from methods.source import Source
from methods.base import forward_decorator
from methods.bn import AlphaBatchNorm, EMABatchNorm
from utils.registry import ADAPTATION_REGISTRY


def _get_num_adapt_samples(cfg):
    """Read the subset size from MIXED_GRADUAL.NUM_UPDATES_PER_EPISODE.
    Returns -1 (= use full batch) when the key is absent or non-positive."""
    try:
        k = cfg.MIXED_GRADUAL.NUM_UPDATES_PER_EPISODE
        return k if k > 0 else -1
    except AttributeError:
        return -1


@ADAPTATION_REGISTRY.register()
class BNTest(Source):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)
        self._timing_enabled = cfg.TIMING.ENABLED
        self._timing_warmup = cfg.TIMING.WARMUP_SAMPLES
        self._timing_target = cfg.TIMING.NUM_MEASUREMENTS
        self._timing_results = []
        self._timing_sample_counter = 0

    @forward_decorator
    def forward_and_adapt(self, x):
        imgs_test = x[0]
        batch_size = imgs_test.shape[0]
        k = _get_num_adapt_samples(self.cfg)

        _should_time = (
            getattr(self, "_timing_enabled", False)
            and self._timing_sample_counter >= self._timing_warmup
            and len(self._timing_results) < self._timing_target
        )

        if _should_time:
            torch.cuda.synchronize()
            _t0 = time.perf_counter()

        if k <= 0 or k >= batch_size:
            # Original behaviour: single forward with full batch (BN in train mode)
            outputs = self.model(imgs_test)
        else:
            k = min(k, batch_size)
            subset_idx = torch.randperm(batch_size, device=imgs_test.device)[:k]

            # Phase 1: forward k samples — BN layers in train mode update running stats
            with torch.no_grad():
                _ = self.model(imgs_test[subset_idx])

            # Phase 2: switch BN to eval (use just-updated running stats), forward full batch
            bn_layers = []
            for m in self.model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    if m.training:
                        bn_layers.append(m)
                        m.eval()

            with torch.no_grad():
                outputs = self.model(imgs_test)

            # Restore BN train mode for next batch
            for m in bn_layers:
                m.train()

        if _should_time:
            torch.cuda.synchronize()
            batch_time = time.perf_counter() - _t0
            time_per_sample = batch_time / batch_size
            self._timing_results.extend([time_per_sample] * batch_size)
            
        self._timing_sample_counter += batch_size

        return outputs

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)

        for m in self.model.modules():
            # Re-activate batchnorm layer
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                m.train()
            elif isinstance(m, nn.BatchNorm1d):
                # Only activate BN1d when batch_size > 1 (single sample stats are degenerate)
                if self.cfg.TEST.BATCH_SIZE > 1:
                    m.train()


@ADAPTATION_REGISTRY.register()
class BNAlpha(Source):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)
        self._timing_enabled = cfg.TIMING.ENABLED
        self._timing_warmup = cfg.TIMING.WARMUP_SAMPLES
        self._timing_target = cfg.TIMING.NUM_MEASUREMENTS
        self._timing_results = []
        self._timing_sample_counter = 0

    @forward_decorator
    def forward_and_adapt(self, x):
        imgs_test = x[0]
        batch_size = imgs_test.shape[0]
        k = _get_num_adapt_samples(self.cfg)

        _should_time = (
            getattr(self, "_timing_enabled", False)
            and self._timing_sample_counter >= self._timing_warmup
            and len(self._timing_results) < self._timing_target
        )

        if _should_time:
            torch.cuda.synchronize()
            _t0 = time.perf_counter()

        if k <= 0 or k >= batch_size:
            # Original behaviour: single forward, AlphaBatchNorm computes stats from full batch
            outputs = self.model(imgs_test)
        else:
            k = min(k, batch_size)
            subset_idx = torch.randperm(batch_size, device=imgs_test.device)[:k]

            # Phase 1: forward k samples — AlphaBatchNorm computes and caches blended stats
            AlphaBatchNorm.set_use_cached(self.model, False)
            with torch.no_grad():
                _ = self.model(imgs_test[subset_idx])

            # Phase 2: forward full batch using the cached stats from Phase 1
            AlphaBatchNorm.set_use_cached(self.model, True)
            with torch.no_grad():
                outputs = self.model(imgs_test)

            # Clear cache for next batch
            AlphaBatchNorm.set_use_cached(self.model, False)

        if _should_time:
            torch.cuda.synchronize()
            batch_time = time.perf_counter() - _t0
            time_per_sample = batch_time / batch_size
            self._timing_results.extend([time_per_sample] * batch_size)

        self._timing_sample_counter += batch_size

        return outputs

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)

        # (1-alpha) * src_stats + alpha * test_stats
        self.model = AlphaBatchNorm.adapt_model(self.model, alpha=self.cfg.BN.ALPHA).to(self.device)


@ADAPTATION_REGISTRY.register()
class BNEMA(Source):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)
        self._timing_enabled = cfg.TIMING.ENABLED
        self._timing_warmup = cfg.TIMING.WARMUP_SAMPLES
        self._timing_target = cfg.TIMING.NUM_MEASUREMENTS
        self._timing_results = []
        self._timing_sample_counter = 0

    @forward_decorator
    def forward_and_adapt(self, x):
        imgs_test = x[0]
        batch_size = imgs_test.shape[0]
        k = _get_num_adapt_samples(self.cfg)

        _should_time = (
            getattr(self, "_timing_enabled", False)
            and self._timing_sample_counter >= self._timing_warmup
            and len(self._timing_results) < self._timing_target
        )

        if _should_time:
            torch.cuda.synchronize()
            _t0 = time.perf_counter()

        if k <= 0 or k >= batch_size:
            # Original behaviour: EMABatchNorm does train-forward + eval-forward on full batch
            outputs = self.model(imgs_test)
        else:
            k = min(k, batch_size)
            subset_idx = torch.randperm(batch_size, device=imgs_test.device)[:k]

            # Access the inner model wrapped by EMABatchNorm
            inner_model = self.model.model

            # Phase 1: update running stats with k samples (train mode forward)
            inner_model.train()
            with torch.no_grad():
                _ = inner_model(imgs_test[subset_idx])

            # Phase 2: evaluate full batch with updated running stats (eval mode)
            inner_model.eval()
            with torch.no_grad():
                outputs = inner_model(imgs_test)

        if _should_time:
            torch.cuda.synchronize()
            batch_time = time.perf_counter() - _t0
            time_per_sample = batch_time / batch_size
            self._timing_results.extend([time_per_sample] * batch_size)

        self._timing_sample_counter += batch_size

        return outputs

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)
        self.model = EMABatchNorm.adapt_model(self.model).to(self.device)
