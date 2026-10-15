"""
Shared adaptation-latency measurement.

One convention, enforced in one place: `_timing_results` holds **seconds per
adapted sample**, because `test_time.py` (and the `timing.json` that
`scripts/plot_pareto.py` consumes) multiplies the stored values by 1000. A method
that stored milliseconds instead reported a latency 1000x too large — which is
exactly what happened to ROID before this helper existed.

"Per adapted sample" means the wall-clock of one adaptation step divided by the
number of samples that step consumed:

  * streaming methods (AURA, EKF-FC, Online GD) adapt one sample at a time, so a
    step is one update and the divisor is 1;
  * batch-wise methods (Tent, SAR, ROID) adapt once per frame, so the divisor is
    the number of samples the update was computed from — the whole batch for
    Tent/SAR, `ROID.ADAPT_SAMPLES` for ROID.
"""
import time

import torch


class TimedAdaptation:
    """
    Context manager timing a single adaptation step.

        with TimedAdaptation(self, num_samples=imgs.shape[0]):
            ...  # the adaptation step only, not the evaluation forward

    Honours cfg.TIMING.{ENABLED,WARMUP_SAMPLES,NUM_MEASUREMENTS}, synchronizes CUDA
    around the region, and skips recording if the step raised.
    """

    def __init__(self, method, num_samples: int):
        self.method = method
        self.num_samples = max(int(num_samples), 1)
        self.measuring = False
        self._t0 = None

        timing_cfg = getattr(getattr(method, "cfg", None), "TIMING", None)
        self.enabled = bool(timing_cfg is not None and timing_cfg.ENABLED)
        if self.enabled:
            self.warmup = timing_cfg.WARMUP_SAMPLES
            self.target = timing_cfg.NUM_MEASUREMENTS
            if not hasattr(method, "_timing_results"):
                method._timing_results = []
            if not hasattr(method, "_timing_samples_seen"):
                method._timing_samples_seen = 0
            # How many recorded units make up one frame, i.e. multiply a stored
            # measurement by this to get the per-frame adaptation cost. For a
            # batch-wise method one step covers the whole frame, so it is the
            # batch size; a streaming method records one update and performs
            # `num_updates` of them per frame.
            method._timing_units_per_frame = self.num_samples

    def __enter__(self):
        if self.enabled:
            method = self.method
            self.measuring = (method._timing_samples_seen >= self.warmup
                              and len(method._timing_results) < self.target)
            if self.measuring:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.enabled:
            method = self.method
            if self.measuring and exc_type is None:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - self._t0
                method._timing_results.append(elapsed / self.num_samples)   # seconds
            method._timing_samples_seen += self.num_samples
        return False
