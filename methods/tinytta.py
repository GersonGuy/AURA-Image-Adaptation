"""
TinyTTA: Efficient Test-time Adaptation via Early-exit Ensembles on Edge Devices
Jia, Kwon, Orsino, Dang, Talia, Mascolo -- NeurIPS 2024.
https://proceedings.neurips.cc/paper_files/paper/2024/hash/4c454d34f3a4c8d6b4ca85a918e5d7ba-Abstract-Conference.html

RECONSTRUCTION FROM PROSE -- NOT A PORT
=======================================
The official release cited by the paper (github.com/h-jia/TTE) contains NO TinyTTA algorithm:
no backward pass, no optimizer, no entropy, no early-exit logic, no weight standardization, no
submodule partitioning, and none of the four paper backbones. It is a vendored mid-2023
TensorFlow-Lite-Micro snapshot plus a ~470-line Mbed demo that runs forward inference on upstream
TFLM's `hello_world` sine-wave regressor with Gaussian noise as input. `grep -ril entropy` over the
whole repository returns zero hits. See docs/tinytta/code_forensics.md for the proof.

Consequence: **0 of the 19 ambiguities catalogued in docs/tinytta/paper_spec.md section 8 are
resolvable from code**, and there is no third-party reimplementation to cross-check against. Every
mechanism below is labelled PAPER / INFERRED / GUESS, and every behavioural guess is reachable from
a `cfg.TINYTTA` key. Full mapping and provenance: docs/tinytta/implementation_notes.md.


Method
------
The frozen backbone is split into K contiguous submodules theta_1..theta_K by ACTIVATION memory
(paper Sec. 3.1: "memory usage is primarily driven by activations ... therefore we focus on
activation memory for self-ensembling"). Each submodule carries an exit head phi_k. A test sample
is pushed through the submodules one at a time and stops at the first exit whose predictive entropy
falls below a threshold:

    z^k = phi_k(theta_k(...theta_1(x)))                         (exit logits)
    p^k = softmax(z^k)                                          (paper Eq. 1)
    H   = -sum_j p^k_j log p^k_j                                (paper Eq. 2, natural log, nats)
    exit at the first k with H < gamma_k                        (paper Sec. 3.2, strict '<')
    exit K is unconditional                                     (A11, guess)

The backbone is frozen throughout; only the exit heads are updated (Sec. 3.1: "only update the
heads at early exits, freezing the submodules ... this process does not require backpropagation to
the submodules, thus eliminating the need for activation memory").


The five load-bearing guesses
-----------------------------
A1  TEST-TIME LOSS -- **INFERRED, NOT STATED**. The paper never writes down the loss it minimizes
    at test time; Sec. 3.3/3.4 say only "adapts phi_k" / "only early exit branches are updated
    on-device". We minimize the Shannon entropy of the EXITED head's softmax -- i.e. TENT's
    objective applied to the exit head. The inference rests on Sec. 1's framing ("TTA aims to align
    the model with the current data distribution through entropy minimization"), on App. A Eq. 7
    which supplies H(p_hat) = -sum_c p_hat_c log(p_hat_c) under the heading "Entropy Minimization",
    and on self-consistency with the exit rule, which uses the same entropy. It is NOT a finding.
    Key: TINYTTA.TEST_LOSS.

A2  WHICH HEAD IS UPDATED -- the paper contradicts itself three ways: Sec. 3.1 "only update the
    heads at early exits"; Sec. 3.2 "the adaptation of the submodules preceding this kth
    submodule"; Sec. 3.3 "adapts only phi_k". We update the head the sample ACTUALLY EXITED AT --
    Sec. 3.3 is the most specific and the only reading consistent with per-sample exits.
    Key: TINYTTA.UPDATE_HEADS.

A3  STEPS PER BATCH -- never stated. 1. (TENT's default, and the only value compatible with
    Table 1's 0.22 s/sample being BELOW EATA's 0.25 s.) Enforced: OPTIM.STEPS must be 1.

A4  OPTIMIZER / LR -- no image-domain optimizer, learning rate, momentum or weight decay exists
    anywhere in the paper (verified by exhaustive grep, paper_spec.md Sec. 6.1). SGD / lr 1e-5 /
    momentum 0.9 is TRANSPLANTED from App. H, which is the AUDIO experiment (MicroNets on Speech
    Commands), and coincidentally appears again in App. J describing the TTT BASELINE.
    lr 1e-5 was tuned at BATCH SIZE 1. See "Batch 100" below.

A5  WEIGHT STANDARDIZATION AXES -- Eq. 4 as printed says "over all channels and dimensions" with a
    single (mu_w, sigma_w), i.e. whole-tensor; the cited reference [12] (Qiao et al., 2019)
    standardizes PER OUTPUT CHANNEL. The paper is in direct conflict with the reference it claims
    to be using, and the code cannot arbitrate. Default here is PER-OUTPUT-CHANNEL (standard
    practice, and what [12] means); whole-tensor is `TINYTTA.WS_MODE="whole_tensor"`. Sec. 5.4 /
    Fig. 7 attributes 12-15% accuracy to WS alone, so BOTH must be measured before any WS claim.
    Eq. 3 also divides by (sigma + eps) rather than sqrt(var + eps) -- a second, independent fork,
    `TINYTTA.WS_EPS_IN_SQRT`.


Exit-head architecture -- recovered from App. D Table 3, not guessed
-------------------------------------------------------------------
The paper never states the head's kernel size, channel count or pooling (A8). Table 3 does pin the
FORM. It reports per-exit parameter counts for three backbones at C=10 and C=100; C is the only
difference between the two columns, so

    params(C) = P + c*C + C   ==>   params(100) - params(10) = 90*(c + 1)

determines the final linear layer's in_features `c` exactly. Solving all nine exits gives integer
c, and the remainder P factors as C_in*c + a*c:

    backbone      exit    c (linear in_features)   C_in            a
    MobileNet     1/2/3   96 / 192 / 288           16 / 32 / 48    2
    EfficientNet  1/2/3   144 / 480 / 672          24 / 80 / 192   2
    RegNet        1/2/3   152 / 368 / 368          152 / 368 / 368 3

All 12 MobileNet/EfficientNet cells reproduce Table 3 EXACTLY; the 6 RegNet cells reproduce exactly
with one extra parameter per output channel. The recovered C_in values are clean architectural
widths (RegNetX-200MF's stage widths are 24/56/152/368) and for MobileNet/EfficientNet `c` is the
inverted-residual EXPANSION width at the tap (6*16=96, 6*32=192, 6*48=288, 6*24=144, 6*80=480).
So the head is

    tap (C_in ch) -> Conv2d(C_in -> c, k=1, no bias, WEIGHT-STANDARDIZED)
                  -> per-channel affine (2c params: gain + shift)
                  -> GlobalAvgPool -> flatten -> Linear(c, C)

The affine explains itself: per-output-channel WS is scale-invariant in the weight, so without a
learnable gain the head could not set its own logit scale. (The paper's "WS introduces no
additional parameters" refers to WS itself and is consistent -- the affine is separate.)
Neither of OUR backbones is in Table 3, so `c` is unconstrained for us; the only non-arbitrary
choice is c == C_in (the RegNet reading), i.e. TINYTTA.HEAD_WIDTH_MULT = 1.0.

A12: App. D says "the last exit layer retains the same memory and parameters as the original
model" and Table 3 lists only exits 1-3 for K=4, so exit K IS the pretrained classifier and only
K-1 new heads are created (TINYTTA.EXIT_K_MODE="reuse_classifier").


Partitioning -- measured on this repo's backbones
-------------------------------------------------
No published split exists for WRN-28-10 or ResNeXt-29. The paper's criterion is activation memory,
and its published ResNet50 cuts (layers 1-15/16-28/29-44/45-52) fall about one bottleneck PAST each
architectural stage boundary (which sit at ~11/24/43) -- i.e. inside stages. That rules out
"homogeneous per-layer size within a group" and confirms BALANCED CUMULATIVE ACTIVATION as the
criterion. Measured with forward hooks at the protocol's input (batch 100, 32x32, fp32 output
bytes of every Conv/BN/Linear), the optimal contiguous 4-way splits at block boundaries are:

    WRN-28-10   [2,2,3,5]  537.40 / 524.29 / 458.75 / 442.37 MB   max/min 1.215, std/mean 8.3%
    ResNeXt-29  [1,2,2,4]  681.57 / 838.86 / 681.57 / 655.40 MB   max/min 1.280, std/mean 10.2%

Both are the global optimum over all contiguous 4-way block splits (runners-up 1.750 and 1.833).
Exit taps: WRN 160x32^2, 160x32^2, 320x16^2 + the original fc; ResNeXt 256x32^2, 256x32^2,
512x16^2 + the original classifier. Full tables in docs/tinytta/implementation_notes.md Sec. 2.

NOTE: in a PyTorch port the partition has no effect on memory (the backbone is frozen, so no
activations are retained for it). It determines only where the exits tap and how much compute an
early exit skips. The activation-balanced split is a FIDELITY choice, not an efficiency one.


Batch 100 vs the paper's batch 1
--------------------------------
The paper runs batch size 1 in every reported experiment (Sec. 5.1; Limitations: "TinyTTA has only
been tested with a batch size of one"). This repo's protocol is batch 100.

`TINYTTA.EXIT_MODE="grouped"` (default): one forward in which the active-sample tensor SHRINKS as
samples exit, per-sample exit decisions, samples grouped by their chosen exit, and ONE optimizer
step per batch on the summed loss. This is exact, not an approximation of the forward, because the
frozen backbone runs in eval() with source BN running statistics (A19) and is therefore pointwise
in the batch -- dropping rows cannot change the surviving rows' activations. That equivalence is
asserted by tests/test_tinytta.py and is why FROZEN_BN_MODE must be "eval" in grouped mode.

`TINYTTA.EXIT_MODE="per_sample"` is the faithful batch-1 loop (B forwards, B optimizer steps).
~20-50x slower; for validation runs only.

**The lr consequence, stated as arithmetic rather than opinion.** Over one 1e5-sample run:
    batch 1, 1 step/sample  ->  100,000 optimizer steps at lr 1e-5
    batch 100, 1 step/batch ->    1,000 optimizer steps at lr 1e-5, on gradients of similar
                                  magnitude (an average over ~100 samples, not 100x larger)
so total parameter displacement is ~100x smaller in grouped mode. Combined with lr 1e-5 on a head
whose linear layer has O(10^3) parameters, TinyTTA at batch 100 may be very nearly INERT -- i.e. it
would measure static self-ensemble early-exit inference, not adaptation. The default lr is left at
the transplanted 1e-5 and nothing is rescaled here. Instead `TINYTTA.LOG_PARAM_DRIFT` (on by
default) measures it: per-exit cumulative ||delta theta||, displacement from the offline init, and
the fraction of steps that produced a non-zero update. If TinyTTA is inert, the run says so with a
number.


TWO DEGENERACIES, ONE OF WHICH LOOKS LIKE SUCCESS
-------------------------------------------------
(1) NOTHING EXITS EARLY. If every gamma_k sits below the observed entropy distribution, no early
exit ever fires. With `EXIT_K_MODE="reuse_classifier"` and `EXIT_K_TRAINABLE=True`, what is left is
a frozen backbone plus entropy-SGD on the original classifier -- which is byte-for-byte
`methods/sgd_base.py` with `STAGE_SELECTION=["fc"]`, `LOSS="entropy"`. There is no early exit and
no self-ensemble, only an existing baseline under a new name. Detector: the exit histogram.

(2) EVERYTHING EXITS EARLY, CONFIDENTLY AND WRONGLY. MEASURED (tests/test_tinytta.py section 4):
randomly initialised heads of THIS design are NOT near-uniform. Per-exit median entropies on random
32x32 input came out at 0.137 / 0.011 / 0.273 nats versus ln 10 = 2.303 -- i.e. an untrained head is
strongly OVER-confident. The cause is structural: per-output-channel WS forces the conv weight to
unit variance, so the conv amplifies its input by ~sqrt(fan_in) (~12.6 at C_in=160), and the default
Linear init on top of that yields a wide logit spread. So the naive expectation "untrained head ->
H ~ ln C -> never fires" is FALSE for this head; an untrained head fires eagerly and returns
garbage. Detector: `per_exit_clean_acc`, stored in the offline checkpoint and re-checked on load.

Both must be watched, and neither is detectable from the error rate alone.

This is dangerous because the paper's own effect size is small: Fig. 5's TinyTTA-vs-source gaps are
4.1-7.5 pp (mean 4.3 pp), so a degenerate implementation lands in exactly the reported band.
**Accuracy alone cannot validate this implementation.** The diagnostics below are therefore
load-bearing, not optional:
  * the per-exit histogram (TINYTTA.LOG_EXIT_DIST) plus a loud warning when fewer than
    TINYTTA.MIN_EARLY_EXIT_FRAC of samples exit before exit K;
  * the fraction of backbone blocks skipped, for the Table-2 cross-check (TinyTTA is 50.7 ms vs
    55.8 ms inference-only = 9.1% faster INCLUDING an on-device backward, which implies early exits
    fire RARELY in the paper's own configuration);
  * per-exit clean source accuracy stored in the offline checkpoint and re-checked on load;
  * the parameter-drift numbers above.
Any TinyTTA number must be reported next to the exit histogram and the drift numbers.


Entropy thresholds gamma_k
--------------------------
A10. App. G publishes exactly four numbers per dataset -- ONE PER BACKBONE, not per exit
(CIFAR10C: 1.12/1.34/1.53/1.26 for MCUNet/EfficientNet/MobileNet/RegNet; CIFAR100C:
3.26/2.58/3.10/2.31) -- and none of those backbones is ours. So gamma_1..gamma_K are unpublished
for every model in the paper and unpublishable for ours. The stated method is "grid search", with
no grid, no objective and no split mentioned, so it was presumably tuned on the test corruptions.

We do NOT inherit that leak. `GAMMA_MODE="calibrated"` (default) reads thresholds produced by
offline_train_tinytta.py on a held-out slice of the CLEAN SOURCE TRAIN split, chosen so a target
fraction of samples exits at each head (see `calibrate_gammas`). Calibration data is never the test
split and never corrupted. `"scalar"` reproduces the paper's single-value form, `"explicit"`
requires the numbers to be typed into a yaml by hand -- so any leak has to be deliberate and
visible in the config. This is unavoidable tuning and is labelled as such.


Not ported, deliberately
------------------------
* The int8-backbone / fp32-head mixed-precision boundary of Fig. 4. Nothing in the release
  implements it and the paper never characterizes the quantization error. Everything here is fp32,
  which makes our accuracies an UPPER BOUND on MCU-deployed accuracy.
* The static tensor arena and in-place fixed-point arithmetic -- the mechanism behind the paper's
  memory claims, and meaningless in PyTorch. Do not report Table 1 / Table 2 numbers: the harness
  that produced them timed a sine-wave regressor on random noise with a 5 KB arena against
  Table 2's claimed 123 KB.
"""

import json
import logging
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.base import TTAMethod
# BackboneShim and friends are reused UNCHANGED from the EcoTTA port. TinyTTA needs exactly the
# surface it already exposes: blocks() for the ordered residual blocks, block_geometry() for
# (c_in, c_out, stride) of a contiguous run -- c_out is the exit tap's channel count and the stride
# product gives the tap's spatial reduction -- and build_stem()/build_head() for the two ends.
# build_stem() is what reproduces the ResNeXt's (x - mu)/sigma normalization, which lives INSIDE
# Hendrycks2020AugMixResNeXtNet.forward and would otherwise be silently dropped by a graph rebuild.
from methods.ecotta import get_backbone_shim
from utils.losses import Entropy
from utils.registry import ADAPTATION_REGISTRY

logger = logging.getLogger(__name__)


# Measured activation-balanced partitions (see the module docstring and
# docs/tinytta/implementation_notes.md Sec. 2). Blocks per submodule, shallow -> deep.
TINYTTA_PARTITIONS = {
    "wrn28":     {2: [4, 8],    3: [2, 3, 7],    4: [2, 2, 3, 5]},
    "resnext29": {2: [2, 7],    3: [1, 3, 5],    4: [1, 2, 2, 4]},
}

# Regression guard on the head construction, for the DEFAULT head configuration only
# (kernel 1, width_mult 1.0, affine, gap, exit_k_mode="reuse_classifier"):
#     params = C_in^2 + 2*C_in + C_in*C + C
# WRN-28-10 K=4, C=10:  160 -> 25600+320+1600+10 = 27530 ; 320 -> 102400+640+3200+10 = 106250
# ResNeXt-29 K=4, C=100: 256 -> 65536+512+25600+100 = 91748 ; 512 -> 262144+1024+51200+100 = 314468
EXPECTED_HEAD_PARAMS = {
    ("wrn28", 4, 10):      [27530, 27530, 106250],
    ("resnext29", 4, 100): [91748, 91748, 314468],
}

WS_MODES = ("per_out_channel", "whole_tensor")
WS_TARGETS = ("conv", "linear", "both", "none")
GAMMA_MODES = ("calibrated", "scalar", "explicit")
TEST_LOSSES = ("entropy_exited", "entropy_mean_upto_k", "none")
UPDATE_HEAD_MODES = ("exited", "upto_k", "all")
EXIT_MODES = ("grouped", "per_sample")
EXIT_K_MODES = ("reuse_classifier", "new_head")
POOL_MODES = ("gap", "avgpool2")


# ---------------------------------------------------------------------------------------------
# Weight Standardization (paper Eq. 3 / Eq. 4 vs. ref. [12])
# ---------------------------------------------------------------------------------------------

def _standardize(w: torch.Tensor, dims, eps: float, eps_in_sqrt: bool) -> torch.Tensor:
    """Eq. 3 / Eq. 4.

    `dims` selects the fork:
      (1, 2, 3) / (1,)  -> per output channel, N = fan_in       (ref. [12], the default)
      all dims          -> whole tensor,      N = numel         (Eq. 4 read literally)

    `unbiased=False` because Eq. 4 divides by N, not N-1.
    `eps_in_sqrt=False` reproduces Eq. 3 as printed, (W - mu) / (sigma + eps); True uses the
    standard sqrt(var + eps). No gain term -- neither the paper nor ref. [12] mentions one; the
    learnable per-channel affine that Table 3's +2c implies is a separate module (see ExitHead).
    """
    mu = w.mean(dim=dims, keepdim=True)
    var = w.var(dim=dims, unbiased=False, keepdim=True)
    if eps_in_sqrt:
        return (w - mu) / torch.sqrt(var + eps)
    return (w - mu) / (torch.sqrt(var) + eps)


class WSConv2d(nn.Conv2d):
    """Conv whose weight is standardized on every forward.

    Sec. 3.3: "During inference, WS is computed solely from the weights, eliminating the need to
    accumulate batch statistics on the fly, thus achieving batch-agnostic adaptation." So the
    standardization is recomputed from the LIVE weight every call -- there is no buffer and nothing
    to reset. That also resolves A7 ("adapts only phi_k, i.e. the WS statistics and one linear
    layer"): the adapted state is the conv weight, the affine and the linear; mu_w/sigma_w have no
    independent existence and move only as a consequence.
    """

    def __init__(self, *args, ws_mode="per_out_channel", ws_eps=1e-5, ws_eps_in_sqrt=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if ws_mode not in WS_MODES:
            raise ValueError(f"ws_mode must be one of {WS_MODES}, got {ws_mode!r}")
        self.ws_mode = ws_mode
        self.ws_eps = ws_eps
        self.ws_eps_in_sqrt = ws_eps_in_sqrt

    def standardized_weight(self) -> torch.Tensor:
        dims = (1, 2, 3) if self.ws_mode == "per_out_channel" else (0, 1, 2, 3)
        return _standardize(self.weight, dims, self.ws_eps, self.ws_eps_in_sqrt)

    def forward(self, x):
        return self._conv_forward(x, self.standardized_weight(), self.bias)


class WSLinear(nn.Linear):
    """Linear whose weight is standardized on every forward (TINYTTA.WS_TARGET in {linear, both}).

    Sec. 3.3 alternately describes WS as "a WS layer preceding the linear layer" and as "applied
    directly to the early exit CNN layers phi_k"; "WS introduces no additional parameters within
    the CNN layers" settles the default in favour of the conv (A6). This class exists so the other
    reading is measurable.
    """

    def __init__(self, *args, ws_mode="per_out_channel", ws_eps=1e-5, ws_eps_in_sqrt=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if ws_mode not in WS_MODES:
            raise ValueError(f"ws_mode must be one of {WS_MODES}, got {ws_mode!r}")
        self.ws_mode = ws_mode
        self.ws_eps = ws_eps
        self.ws_eps_in_sqrt = ws_eps_in_sqrt

    def standardized_weight(self) -> torch.Tensor:
        dims = (1,) if self.ws_mode == "per_out_channel" else (0, 1)
        return _standardize(self.weight, dims, self.ws_eps, self.ws_eps_in_sqrt)

    def forward(self, x):
        return F.linear(x, self.standardized_weight(), self.bias)


class ChannelAffine(nn.Module):
    """Per-channel gain + shift: the 2c parameters that App. D Table 3's counts require.

    Initialised to the identity so a freshly built head is exactly (WS conv -> pool -> linear).
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


def _gn_groups(c: int, maximum: int = 32) -> int:
    for g in range(min(maximum, c), 0, -1):
        if c % g == 0:
            return g
    return 1


# ---------------------------------------------------------------------------------------------
# Exit head
# ---------------------------------------------------------------------------------------------

class ExitHead(nn.Module):
    """phi_k: WS conv -> per-channel affine -> pool -> Linear(c, C).

    Recovered from App. D Table 3; see the module docstring for the fit. `width_mult` scales c
    relative to the tap's channel count: 1.0 is the RegNet reading (c == C_in), 6.0 the
    MobileNet/EfficientNet reading (the inverted-residual expansion width). Neither of our
    backbones is in Table 3, so 1.0 is the only non-arbitrary choice.

    `ws_target="none"` is Fig. 7's "without WS" ablation (claimed -12 to -15% accuracy);
    `ws_ablation_replacement="groupnorm"` makes it App. E / Fig. 11's TinyTTA+GN comparator
    (WS 83.2/78.3/73.6/68.7/64.3 vs GN 73.4/67.2/63.1/57.9/53.2 at severities 1-5).
    """

    def __init__(self, c_in: int, num_classes: int, kernel_size: int = 1, width_mult: float = 1.0,
                 affine: bool = True, pool: str = "gap", ws_target: str = "conv",
                 ws_mode: str = "per_out_channel", ws_eps: float = 1e-5,
                 ws_eps_in_sqrt: bool = False, ws_ablation_replacement: str = "identity"):
        super().__init__()
        if pool not in POOL_MODES:
            raise ValueError(f"HEAD_POOL must be one of {POOL_MODES}, got {pool!r}")
        if ws_target not in WS_TARGETS:
            raise ValueError(f"WS_TARGET must be one of {WS_TARGETS}, got {ws_target!r}")

        c = max(1, int(round(width_mult * c_in)))
        self.c_in, self.c, self.num_classes = c_in, c, num_classes
        self.pool_mode = pool
        self.ws_target = ws_target

        ws_kwargs = dict(ws_mode=ws_mode, ws_eps=ws_eps, ws_eps_in_sqrt=ws_eps_in_sqrt)
        pad = kernel_size // 2
        if ws_target in ("conv", "both"):
            self.conv = WSConv2d(c_in, c, kernel_size, padding=pad, bias=False, **ws_kwargs)
        else:
            self.conv = nn.Conv2d(c_in, c, kernel_size, padding=pad, bias=False)

        # The normalization slot. With WS active this is the 2c affine of Table 3; with
        # ws_target="none" it becomes either nothing (Fig. 7) or GroupNorm (App. E / Fig. 11).
        if ws_target == "none" and ws_ablation_replacement == "groupnorm":
            self.norm = nn.GroupNorm(_gn_groups(c), c)
        elif affine:
            self.norm = ChannelAffine(c)
        else:
            self.norm = nn.Identity()

        self.pool = nn.AdaptiveAvgPool2d(1) if pool == "gap" else nn.AdaptiveAvgPool2d((2, 2))
        in_features = c * (1 if pool == "gap" else 4)
        if ws_target in ("linear", "both"):
            self.fc = WSLinear(in_features, num_classes, **ws_kwargs)
        else:
            self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        out = self.norm(self.conv(x))
        out = self.pool(out)
        return self.fc(out.flatten(1))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------------------------

class TinyTTANet(nn.Module):
    """Frozen backbone split into K submodules with K-1 (or K) exit heads.

    Both supported backbones have hand-written `forward` methods, so the split has to REBUILD the
    graph rather than swap children. Frozen parameter tensors are reused by reference (no copy, no
    re-download); the shim's stem reproduces any input normalization.

    Exposes both a stage-wise API (`stage`, `exit_logits`, `run_exits`) used by
    TinyTTA.forward_and_adapt, and a plain `forward` for every no_grad caller -- so the host's
    single-`self.model` assumption (methods/base.py:64) holds and `state_dict()` round-trips
    through reset().
    """

    def __init__(self, model: nn.Module, K: int, num_classes: int, partition=None,
                 head_kernel_size: int = 1, head_width_mult: float = 1.0,
                 head_affine: bool = True, head_pool: str = "gap",
                 exit_k_mode: str = "reuse_classifier", ws_target: str = "conv",
                 ws_mode: str = "per_out_channel", ws_eps: float = 1e-5,
                 ws_eps_in_sqrt: bool = False, ws_ablation_replacement: str = "identity"):
        super().__init__()
        if exit_k_mode not in EXIT_K_MODES:
            raise ValueError(f"EXIT_K_MODE must be one of {EXIT_K_MODES}, got {exit_k_mode!r}")

        shim = get_backbone_shim(model)
        self.family = shim.family
        self.num_classes = num_classes
        self.exit_k_mode = exit_k_mode
        self.head_cfg = dict(kernel_size=head_kernel_size, width_mult=head_width_mult,
                             affine=head_affine, pool=head_pool, ws_target=ws_target,
                             ws_mode=ws_mode, ws_eps=ws_eps, ws_eps_in_sqrt=ws_eps_in_sqrt,
                             ws_ablation_replacement=ws_ablation_replacement)

        blocks = shim.blocks()
        # True when the caller supplied TINYTTA.PARTITION rather than using the measured default.
        # collect_params() uses this to decide whether the published-derived head-param table applies.
        self.partition_is_override = bool(partition)
        if partition:
            partition = [int(v) for v in partition]
        else:
            if K not in TINYTTA_PARTITIONS.get(self.family, {}):
                raise ValueError(
                    f"No measured TinyTTA partition for family {self.family!r} at K={K}. "
                    f"Available: {sorted(TINYTTA_PARTITIONS.get(self.family, {}))}. Supply "
                    f"TINYTTA.PARTITION explicitly (blocks per submodule, shallow -> deep)."
                )
            partition = list(TINYTTA_PARTITIONS[self.family][K])
        if len(partition) != K:
            raise ValueError(f"TINYTTA.PARTITION has {len(partition)} entries but K={K}.")
        if sum(partition) != len(blocks):
            raise ValueError(
                f"Partition {partition} covers {sum(partition)} blocks but backbone family "
                f"{self.family!r} exposes {len(blocks)}."
            )
        self.partition = partition

        self.stem = shim.build_stem()

        stages, taps, cum_stride, start = [], [], 1, 0
        for n_blocks in partition:
            group = blocks[start:start + n_blocks]
            _, c_out, stride = shim.block_geometry(group)
            cum_stride *= stride
            stages.append(nn.Sequential(*group))
            taps.append((c_out, cum_stride))
            start += n_blocks
        self.stages = nn.ModuleList(stages)
        # (channels, cumulative stride) at each submodule's output. Cumulative stride rather than a
        # spatial size, so the signature is independent of the input resolution.
        self.taps = taps

        self.tail = shim.build_head()

        num_new_heads = K - 1 if exit_k_mode == "reuse_classifier" else K
        self.exits = nn.ModuleList([
            ExitHead(taps[k][0], num_classes, **self.head_cfg) for k in range(num_new_heads)
        ])

        # gamma_1..gamma_{K-1}; exit K is unconditional (A11). Registered as a buffer so it travels
        # with .to(device) and is restored by TTAMethod.reset().
        self.register_buffer("gammas", torch.zeros(K - 1))

        # Exit bookkeeping. Only adaptation passes count (see `count=` in run_exits): a no_grad
        # prediction forward must not inflate the histogram.
        self.register_buffer("exit_counts", torch.zeros(K, dtype=torch.long), persistent=False)
        self.register_buffer("blocks_executed", torch.zeros(1, dtype=torch.double),
                             persistent=False)
        self.register_buffer("blocks_possible", torch.zeros(1, dtype=torch.double),
                             persistent=False)

    # -- properties ----------------------------------------------------------
    @property
    def K(self) -> int:
        return len(self.stages)

    def head_params(self):
        return [h.num_params() for h in self.exits]

    def tail_classifier(self):
        """The pretrained classifier inside the tail (WRN: `fc`; ResNeXt: `classifier`)."""
        for m in self.tail.modules():
            if isinstance(m, nn.Linear):
                return m
        return None

    # -- stage-wise API ------------------------------------------------------
    def stage(self, h, k):
        return self.stages[k](h)

    def exit_logits(self, h, k):
        """Exit k's logits. Exit K is the original classifier when EXIT_K_MODE='reuse_classifier'."""
        if k < len(self.exits):
            return self.exits[k](h)
        return self.tail(h)

    @staticmethod
    def entropy(z: torch.Tensor) -> torch.Tensor:
        """Paper Eq. 1 + Eq. 2: softmax then Shannon entropy in NATS, per sample, unreduced.

        Identical to utils/losses.py::Entropy, written out here so the exit rule and the loss
        provably use the same quantity (asserted in tests/test_tinytta.py).
        """
        return -(z.softmax(1) * z.log_softmax(1)).sum(1)

    def run_exits(self, x, count: bool = False, force_full: bool = False):
        """The algorithm.

        Returns
        -------
        logits    : [B, C], each row from the exit that row took, in ORIGINAL batch order
        exit_idx  : [B] long, 0-based index of the exit each row took
        per_exit  : {k: (row_index[n_k], logits[n_k, C])} -- the exiting rows only
        evaluated : {k: (row_index[m_k], logits[m_k, C])} -- every row that reached exit k

        `force_full=True` disables early exiting so every exit is evaluated on every sample
        (UPDATE_HEADS="all"). It defeats the method's compute saving and exists only for that mode.
        """
        B = x.shape[0]
        h = self.stem(x)
        alive = torch.arange(B, device=x.device)
        logits = None
        exit_idx = torch.full((B,), -1, dtype=torch.long, device=x.device)
        per_exit, evaluated = {}, {}

        for k in range(self.K):
            h = self.stage(h, k)
            z = self.exit_logits(h, k)
            if logits is None:
                logits = z.new_zeros((B, z.shape[1]))
            evaluated[k] = (alive, z)

            if k == self.K - 1:
                # A11: exit K is unconditional.
                fire = torch.ones_like(alive, dtype=torch.bool)
            else:
                fire = self.entropy(z) < self.gammas[k]        # strict '<', paper Sec. 3.2

            if force_full:
                # Record the exit decision but keep every row propagating, so that every exit is
                # evaluated on every sample. Only the FIRST firing exit owns a row (first-match).
                fire = fire & (exit_idx[alive] < 0)
                if bool(fire.any()):
                    rows = alive[fire]
                    logits.index_copy_(0, rows, z[fire])
                    exit_idx.index_fill_(0, rows, k)
                    per_exit[k] = (rows, z[fire])
                continue

            if bool(fire.any()):
                rows = alive[fire]
                logits.index_copy_(0, rows, z[fire])
                exit_idx.index_fill_(0, rows, k)
                per_exit[k] = (rows, z[fire])

            keep = ~fire
            alive = alive[keep]
            if alive.numel() == 0:
                break
            h = h[keep]                                        # <- the actual early exit

        if count:
            self.exit_counts += torch.bincount(exit_idx, minlength=self.K)
            cum = torch.tensor(self.partition, device=x.device).cumsum(0).double()
            self.blocks_executed += cum[exit_idx].sum()
            self.blocks_possible += float(sum(self.partition)) * B

        return logits, exit_idx, per_exit, evaluated

    def forward(self, x):
        """Plain forward for no_grad callers. Does not touch the exit histogram."""
        return self.run_exits(x, count=False)[0]

    # -- diagnostics ---------------------------------------------------------
    def reset_counts(self):
        self.exit_counts.zero_()
        self.blocks_executed.zero_()
        self.blocks_possible.zero_()

    def exit_distribution(self):
        total = int(self.exit_counts.sum().item())
        if total == 0:
            return None
        return [c / total for c in self.exit_counts.tolist()]

    def skipped_block_fraction(self):
        possible = float(self.blocks_possible.item())
        if possible == 0.0:
            return None
        return 1.0 - float(self.blocks_executed.item()) / possible

    def topology_signature(self) -> str:
        """Changes whenever the learned function's shape changes.

        Stored in the offline checkpoint and enforced on load. Tensor shapes can coincide across
        genuinely different architectures (both WRN exits 1 and 2 tap 160 channels), so
        shape-compatibility is NOT evidence of correctness -- this string is.

        Deliberately EXCLUDED: TEST_LOSS, UPDATE_HEADS, LR, EXIT_MODE, GAMMA_*. Those affect only
        the online update, so one offline artifact is valid across all of them. WS settings and
        the head geometry ARE included, because they change the function that was fitted.
        Taps carry the CUMULATIVE STRIDE rather than a spatial size, so the signature does not
        depend on the input resolution.
        """
        hc = self.head_cfg
        n = sum(self.head_params())
        return (
            f"{self.family}|K{self.K}|part={self.partition}|taps={self.taps}"
            f"|ws={hc['ws_target']}/{hc['ws_mode']}/eps{hc['ws_eps']}"
            f"/sqrt{int(hc['ws_eps_in_sqrt'])}/abl{hc['ws_ablation_replacement']}"
            f"|head=k{hc['kernel_size']}_w{hc['width_mult']}_affine{int(hc['affine'])}"
            f"_pool{hc['pool']}"
            f"|exitK={self.exit_k_mode}|C={self.num_classes}|head_params={n}"
        )


# ---------------------------------------------------------------------------------------------
# Construction / checkpoint helpers
# ---------------------------------------------------------------------------------------------

def default_offline_ckpt_path(cfg) -> str:
    """Auto-derived offline-checkpoint path.

    Follows the ckpt/ convention of methods/rmt.py:113-120 and
    methods/ecotta.py::default_warmup_path, with a distinct `ckpt_tinytta_` prefix so it can never
    collide. Only COMPATIBILITY keys go in the filename -- dataset, arch, K, WS mode, seed.
    Training hyperparameters live in the payload and are logged on load, so an online run does not
    fail to find a perfectly usable artifact merely because OFFLINE_EPOCHS differs (the lesson from
    docs/ecotta/implementation_notes.md Sec. 5.3).
    """
    fname = (
        f"ckpt_tinytta_{cfg.CORRUPTION.DATASET}_{cfg.MODEL.ARCH}"
        f"_K{cfg.TINYTTA.K}"
        f"_ws{cfg.TINYTTA.WS_MODE}"
        f"_seed{cfg.RNG_SEED}.pth"
    )
    return os.path.join(cfg.CKPT_DIR, "tinytta", fname)


def build_tinytta_model(cfg, model, device, num_classes: int) -> TinyTTANet:
    t = cfg.TINYTTA
    for key, value, allowed in (("WS_MODE", t.WS_MODE, WS_MODES),
                                ("WS_TARGET", t.WS_TARGET, WS_TARGETS),
                                ("GAMMA_MODE", t.GAMMA_MODE, GAMMA_MODES),
                                ("TEST_LOSS", t.TEST_LOSS, TEST_LOSSES),
                                ("UPDATE_HEADS", t.UPDATE_HEADS, UPDATE_HEAD_MODES),
                                ("EXIT_MODE", t.EXIT_MODE, EXIT_MODES),
                                ("EXIT_K_MODE", t.EXIT_K_MODE, EXIT_K_MODES),
                                ("HEAD_POOL", t.HEAD_POOL, POOL_MODES)):
        if value not in allowed:
            raise ValueError(f"TINYTTA.{key} must be one of {list(allowed)}, got {value!r}")

    return TinyTTANet(
        model,
        K=t.K,
        num_classes=num_classes,
        partition=list(t.PARTITION),
        head_kernel_size=t.HEAD_KERNEL_SIZE,
        head_width_mult=t.HEAD_WIDTH_MULT,
        head_affine=t.HEAD_AFFINE,
        head_pool=t.HEAD_POOL,
        exit_k_mode=t.EXIT_K_MODE,
        ws_target=t.WS_TARGET,
        ws_mode=t.WS_MODE,
        ws_eps=t.WS_EPS,
        ws_eps_in_sqrt=t.WS_EPS_IN_SQRT,
        ws_ablation_replacement=t.WS_ABLATION_REPLACEMENT,
    ).to(device)


def apply_frozen_bn_mode(net: TinyTTANet, mode: str):
    """Train/eval modes for the frozen backbone. Same three-way contract as
    methods/ecotta.py::apply_bn_modes, minus the meta networks (TinyTTA's heads have no BN).

    A19. The paper never says whether the frozen backbone's BN runs on source statistics at test
    time, but "eval" is the only option compatible with batch size 1 and with the claim that the
    method is batch-agnostic (Sec. 3.3).

    "train" and "batch_stats" SILENTLY GRANT THE FROZEN BACKBONE FULL BN ADAPTATION -- i.e. they
    add a BN-Stats baseline (19.66% error on this repo's CIFAR-10-C protocol) on top of TinyTTA and
    would report it as TinyTTA's doing. They also break the grouped/per-sample forward equivalence,
    because BN over a shrinking batch is not pointwise. Kept only as labelled ablations.
    """
    if mode == "eval":
        net.eval()
    elif mode == "train":
        net.train()
    elif mode == "batch_stats":
        net.eval()
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
    else:
        raise ValueError(
            f"TINYTTA.FROZEN_BN_MODE must be one of ['eval', 'train', 'batch_stats'], got {mode!r}"
        )


@torch.no_grad()
def calibrate_gammas(net: TinyTTANet, loader, device, target_fractions=None,
                     max_samples: int = -1):
    """Choose gamma_1..gamma_{K-1} on CLEAN SOURCE data so a target fraction exits at each head.

    A10 in full: App. G publishes ONE threshold per (model, dataset) -- not per exit -- and none of
    the four published backbones is ours, so gamma_1..gamma_K have never been published for any
    model. The paper's stated method is "grid search" with no grid, no objective and no split
    mentioned, i.e. presumably on the test corruptions. This procedure does not inherit that leak:
    the loader MUST be a held-out slice of the clean source TRAIN split (never the test split,
    never corrupted data).

    Procedure, honouring the first-match nature of the exit rule:
        R = all calibration samples, N = |R|
        for k in 1..K-1:
            n_k     = round(f_k * N)                 # share of the FULL set that should exit at k
            gamma_k = quantile({H_k(x) : x in R}, n_k / |R|)
            R      -= {x in R : H_k(x) < gamma_k}

    Returns (gammas, achieved_fractions, entropy_percentiles).

    HONEST CAVEAT: calibration is on clean data; the test stream is corrupted at severity 5, where
    entropies are systematically HIGHER. The achieved test-time early-exit fractions will therefore
    be BELOW the targets, possibly far below. That is the correct leak-free behaviour and is
    exactly why the runtime exit histogram is mandatory rather than optional.
    """
    K = net.K
    if target_fractions is None or len(target_fractions) == 0:
        target_fractions = [1.0 / K] * (K - 1)
    if len(target_fractions) != K - 1:
        raise ValueError(f"GAMMA_TARGET_FRACTIONS needs {K - 1} values, got {len(target_fractions)}")

    net.eval()
    ent_rows = []
    seen = 0
    for batch in loader:
        imgs = batch[0]
        imgs = imgs[0] if isinstance(imgs, list) else imgs
        imgs = imgs.to(device)
        _, _, _, evaluated = net.run_exits(imgs, count=False, force_full=True)
        cols = []
        for k in range(K):
            rows, z = evaluated[k]
            assert rows.numel() == imgs.shape[0], "force_full must evaluate every exit on every row"
            cols.append(net.entropy(z))
        ent_rows.append(torch.stack(cols, dim=1).cpu())
        seen += imgs.shape[0]
        if 0 < max_samples <= seen:
            break

    H = torch.cat(ent_rows, dim=0)                       # [N, K], nats
    N = H.shape[0]
    remaining = torch.ones(N, dtype=torch.bool)
    gammas = []
    for k in range(K - 1):
        n_k = int(round(target_fractions[k] * N))
        pool = H[remaining, k]
        if pool.numel() == 0 or n_k <= 0:
            gammas.append(float("-inf"))
            continue
        q = min(1.0, n_k / float(pool.numel()))
        gamma_k = float(torch.quantile(pool.double(), q).item())
        gammas.append(gamma_k)
        remaining = remaining & ~(H[:, k] < gamma_k)

    # Achieved fractions, measured by actually running the first-match rule.
    achieved = [0] * K
    assigned = torch.zeros(N, dtype=torch.bool)
    for k in range(K - 1):
        fire = (~assigned) & (H[:, k] < gammas[k])
        achieved[k] = int(fire.sum().item())
        assigned |= fire
    achieved[K - 1] = int((~assigned).sum().item())
    achieved = [a / N for a in achieved]

    pcts = {}
    qs = torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99], dtype=torch.double)
    for k in range(K):
        pcts[k] = [float(v) for v in torch.quantile(H[:, k].double(), qs).tolist()]

    return gammas, achieved, pcts


# ---------------------------------------------------------------------------------------------
# The TTA method
# ---------------------------------------------------------------------------------------------

@ADAPTATION_REGISTRY.register()
class TinyTTA(TTAMethod):
    """TinyTTA. Unsupervised at test time (entropy only); labels are used ONLY in the offline stage.

    Deliberately NOT added to utils/eval_utils._is_supervised_tta -- it must go through
    TTAMethod.forward so that self.steps, the episodic reset and the reset_after_num_updates
    counter all apply.
    """

    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        t = cfg.TINYTTA
        self.softmax_entropy = Entropy()
        self.test_loss = t.TEST_LOSS
        self.update_heads = t.UPDATE_HEADS
        self.exit_mode = t.EXIT_MODE
        self.loss_norm = t.LOSS_NORM

        # ---- host-contract guards (see docs/tinytta/implementation_notes.md Sec. 4.2) ----
        if self.steps != 1:
            raise ValueError(
                f"TinyTTA expects OPTIM.STEPS=1 (A3: one step per batch) but OPTIM.STEPS="
                f"{self.steps}. TTAMethod.forward loops `for _ in range(self.steps)` "
                f"(base.py:117), which would multiply the update count."
            )
        if cfg.TEST.WINDOW_LENGTH > 1:
            raise ValueError(
                "TinyTTA rejects TEST.WINDOW_LENGTH > 1. base.py:83-114 would re-adapt every "
                "sample window_length times and would take the reported logit from a no_grad "
                "forward of the whole buffer, making the exit histogram meaningless. For the "
                "paper's batch-1 protocol use TEST.BATCH_SIZE 1 with WINDOW_LENGTH 1, or "
                "TINYTTA.EXIT_MODE 'per_sample'."
            )
        if cfg.MIXED_PRECISION:
            raise ValueError(
                "TinyTTA rejects MIXED_PRECISION: Weight Standardization computes a variance over "
                "the weight tensor, which is needlessly fragile in fp16."
            )
        if self.exit_mode == "grouped" and t.FROZEN_BN_MODE != "eval":
            raise ValueError(
                f"TINYTTA.EXIT_MODE='grouped' requires FROZEN_BN_MODE='eval' (got "
                f"{t.FROZEN_BN_MODE!r}). The grouped forward drops exited rows from the tensor, "
                f"which is only equivalent to a per-sample forward when the frozen backbone is "
                f"pointwise in the batch. Use EXIT_MODE='per_sample' to combine them."
            )

        # ---- parameter-drift diagnostics: the inertness detector ----
        # lr 1e-5 was tuned at batch 1 (App. H, audio). At batch 100 in grouped mode the run takes
        # ~1/100 as many optimizer steps, so total displacement is ~100x smaller and TinyTTA may be
        # very nearly inert. These counters let us state that with a number instead of inferring it
        # from an unchanged error rate.
        self.log_param_drift = t.LOG_PARAM_DRIFT
        self._param_groups = self._head_param_groups()
        self._drift_path = [0.0 for _ in self._param_groups]      # cumulative sum ||delta theta||
        self._drift_steps = [0 for _ in self._param_groups]       # steps with a non-zero update
        self._num_steps = 0
        self._init_flat = [self._flat(g).detach().clone() for g in self._param_groups]

        # ---- exit diagnostics ----
        self.log_exit_dist = t.LOG_EXIT_DIST
        self.log_every = max(1, int(t.LOG_EVERY))
        self._num_batches = 0
        self._diag_dir = None
        self._warned_no_early_exit = False

        logger.info(
            f"TinyTTA ready: family={self.model.family}, K={self.model.K}, "
            f"partition={self.model.partition}, taps(c,cum_stride)={self.model.taps}, "
            f"head_params={self.model.head_params()} (total {sum(self.model.head_params()):,}), "
            f"exit_k_mode={t.EXIT_K_MODE} (trainable={t.EXIT_K_TRAINABLE}), "
            f"ws={t.WS_TARGET}/{t.WS_MODE} eps={t.WS_EPS} eps_in_sqrt={t.WS_EPS_IN_SQRT}, "
            f"gammas={[round(g, 4) for g in self.model.gammas.tolist()]} "
            f"(mode={t.GAMMA_MODE}, ln C = {math.log(self.num_classes):.4f}), "
            f"test_loss={self.test_loss} (INFERRED, A1), update_heads={self.update_heads}, "
            f"exit_mode={self.exit_mode}, lr={t.LR} (TRANSPLANTED from App. H audio, A4)"
        )
        if self.exit_mode == "grouped" and cfg.TEST.BATCH_SIZE > 1:
            logger.warning(
                f"TinyTTA: the paper runs BATCH SIZE 1 in every experiment (Sec. 5.1) and its lr "
                f"1e-5 was tuned there. This run is batch {cfg.TEST.BATCH_SIZE} with one optimizer "
                f"step per batch, i.e. ~{cfg.TEST.BATCH_SIZE}x fewer steps and therefore ~"
                f"{cfg.TEST.BATCH_SIZE}x less total parameter displacement. Check the "
                f"PARAM-DRIFT lines before interpreting the error rate: TinyTTA may be inert here."
            )

    # -- setup ---------------------------------------------------------------
    def configure_model(self):
        """Build the split network, freeze the backbone, load the offline heads.

        Replacing `self.model` here is the sanctioned host pattern (methods/norm.py:157, :219,
        methods/ecotta.py:897): methods/base.py takes `self.models = [self.model]` and the reset
        snapshot at base.py:64-65, i.e. AFTER configure_model runs. **The offline checkpoint must
        therefore be loaded HERE and not after super().__init__()** -- otherwise self.model_states
        holds the RANDOM head init and the unconditional model.reset() at test_time.py:76 silently
        throws the offline stage away. That is the exact defect the EcoTTA port hit.
        """
        cfg = self.cfg
        self.model = build_tinytta_model(cfg, self.model, self.device, self.num_classes)

        self.model.requires_grad_(False)
        for head in self.model.exits:
            head.requires_grad_(True)
        if cfg.TINYTTA.EXIT_K_MODE == "reuse_classifier" and cfg.TINYTTA.EXIT_K_TRAINABLE:
            clf = self.model.tail_classifier()
            if clf is None:
                raise RuntimeError(
                    "EXIT_K_TRAINABLE=True but no nn.Linear was found in the backbone tail."
                )
            clf.requires_grad_(True)

        apply_frozen_bn_mode(self.model, cfg.TINYTTA.FROZEN_BN_MODE)

        self._load_offline_ckpt()
        self._resolve_gammas()

    def _load_offline_ckpt(self):
        cfg = self.cfg
        self.offline_ckpt_path = None
        self.offline_ckpt = None

        if not cfg.TINYTTA.LOAD_OFFLINE:
            logger.info("TINYTTA.LOAD_OFFLINE=False: exit heads left at their random "
                        "initialisation, no offline checkpoint loaded.")
            return

        path = cfg.TINYTTA.OFFLINE_CKPT or default_offline_ckpt_path(cfg)
        if not os.path.isfile(path):
            msg = (
                f"TinyTTA offline checkpoint not found at:\n    {path}\n"
                f"The offline self-ensemble stage is MANDATORY -- the paper's Limitations section "
                f"says so verbatim: 'TinyTTA requires retraining the model via self-ensemble.' "
                f"Randomly initialised heads of this design are strongly OVER-confident, not "
                f"near-uniform (MEASURED: median entropy 0.01-0.27 nats vs ln C = "
                f"{math.log(self.num_classes):.3f}, because per-channel WS amplifies by "
                f"~sqrt(fan_in)), so they fire eagerly and return garbage. Produce a checkpoint "
                f"with:\n"
                f"    CUDA_VISIBLE_DEVICES=0 python offline_train_tinytta.py --cfg <this cfg>\n"
                f"or set TINYTTA.REQUIRE_OFFLINE False to deliberately run cold-started (the exit "
                f"histogram will then show 100% at the last exit, which is information, not a bug)."
            )
            if cfg.TINYTTA.REQUIRE_OFFLINE:
                raise RuntimeError(msg)
            logger.warning(msg)
            return

        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        for key, expected in (("arch", cfg.MODEL.ARCH),
                              ("dataset", cfg.CORRUPTION.DATASET),
                              ("K", cfg.TINYTTA.K),
                              ("num_classes", self.num_classes)):
            if key in ckpt and ckpt[key] != expected:
                raise RuntimeError(
                    f"Offline checkpoint {path} was built with {key}={ckpt[key]!r} but this run "
                    f"uses {key}={expected!r}. Re-run offline_train_tinytta.py or point "
                    f"TINYTTA.OFFLINE_CKPT at the matching file."
                )
        if ckpt.get("source_kind") not in (None, cfg.TINYTTA.OFFLINE_SOURCE):
            raise RuntimeError(
                f"Offline checkpoint {path} was trained on source_kind="
                f"{ckpt['source_kind']!r} but this run declares TINYTTA.OFFLINE_SOURCE="
                f"{cfg.TINYTTA.OFFLINE_SOURCE!r}. Heads trained on CORRUPTED data have seen the "
                f"test corruption pool -- that is a leak and must be declared, not inherited "
                f"silently."
            )

        expected_sig = self.model.topology_signature()
        ckpt_sig = ckpt.get("topology_signature")
        if ckpt_sig is None:
            raise RuntimeError(
                f"Offline checkpoint {path} carries no topology signature and cannot be validated "
                f"against the current architecture. Regenerate it.\n  expected: {expected_sig}"
            )
        if ckpt_sig != expected_sig:
            raise RuntimeError(
                f"Offline checkpoint {path} was trained for a DIFFERENT topology.\n"
                f"  checkpoint: {ckpt_sig}\n"
                f"  this run:   {expected_sig}\n"
                f"Tensor shapes can coincide across genuinely different architectures, so this "
                f"check -- not load_state_dict -- is what makes the artifact safe. Re-run "
                f"offline_train_tinytta.py with the current TINYTTA settings."
            )

        # Only exit tensors are stored (< 1 MB), hence strict=False -- with an explicit check that
        # every exit tensor was in fact restored. That check is what keeps strict=False honest.
        state = dict(ckpt["exit_state"])
        if ckpt.get("tail_state"):
            state.update(ckpt["tail_state"])
        incompatible = self.model.load_state_dict(state, strict=False)
        missing = [k for k in incompatible.missing_keys if k.startswith("exits.")]
        if missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Offline checkpoint {path} does not match the built model.\n"
                f"  missing exit keys: {missing[:8]}\n"
                f"  unexpected keys:   {list(incompatible.unexpected_keys)[:8]}"
            )

        # Risk 1 guard: a head that was never trained (or a mis-wired sum over exits that trained
        # only the last one) is caught here rather than by a plausible-looking accuracy number.
        accs = ckpt.get("per_exit_clean_acc")
        chance = 1.0 / self.num_classes
        if accs is not None:
            logger.info(f"Offline per-exit CLEAN source accuracy: "
                        f"{[round(a, 4) for a in accs]} (chance = {chance:.4f})")
            bad = [k for k, a in enumerate(accs[:len(self.model.exits)]) if a < 2.0 * chance]
            if bad:
                raise RuntimeError(
                    f"Offline checkpoint {path} has exit head(s) {bad} at or near chance accuracy "
                    f"({[round(accs[k], 4) for k in bad]} vs chance {chance:.4f}). Such a head "
                    f"is still CONFIDENT -- WS amplification makes an untrained head "
                    f"over-confident, not uniform -- so it will fire and return garbage rather "
                    f"than be harmlessly ignored. Re-run the offline stage."
                )

        self.offline_ckpt_path = path
        self.offline_ckpt = ckpt
        logger.info(
            f"Loaded TinyTTA offline checkpoint {path} (epochs={ckpt.get('offline_epochs')}, "
            f"lr={ckpt.get('offline_lr')}, lambda={ckpt.get('offline_lambda')}, "
            f"bs={ckpt.get('offline_batch_size')}, source_kind={ckpt.get('source_kind')}, "
            f"gammas={ckpt.get('gammas')}, achieved_exit_frac={ckpt.get('gamma_achieved_frac')})"
        )
        if (ckpt.get("offline_epochs") is not None
                and ckpt["offline_epochs"] != self.cfg.TINYTTA.OFFLINE_EPOCHS):
            logger.warning(
                f"Offline checkpoint was trained for {ckpt['offline_epochs']} epoch(s) but "
                f"TINYTTA.OFFLINE_EPOCHS is {self.cfg.TINYTTA.OFFLINE_EPOCHS}. Used as is."
            )

    def _resolve_gammas(self):
        """Fill self.model.gammas according to GAMMA_MODE. Called at the end of configure_model."""
        cfg, t = self.cfg, self.cfg.TINYTTA
        K = self.model.K
        if t.GAMMA_MODE == "explicit":
            if len(t.GAMMA) != K - 1:
                raise ValueError(f"GAMMA_MODE='explicit' needs {K - 1} values in TINYTTA.GAMMA, "
                                 f"got {len(t.GAMMA)}.")
            gammas = [float(g) for g in t.GAMMA]
        elif t.GAMMA_MODE == "scalar":
            if not t.GAMMA_SCALAR > 0:
                raise ValueError("GAMMA_MODE='scalar' requires TINYTTA.GAMMA_SCALAR > 0. App. G's "
                                 "values are per (model, dataset) and none of its backbones is "
                                 "ours -- see the module docstring.")
            gammas = [float(t.GAMMA_SCALAR)] * (K - 1)
        else:                                                     # "calibrated"
            if self.offline_ckpt is not None and self.offline_ckpt.get("gammas") is not None:
                gammas = [float(g) for g in self.offline_ckpt["gammas"]]
                if len(gammas) != K - 1:
                    raise RuntimeError(f"Checkpoint holds {len(gammas)} gammas, expected {K - 1}.")
            else:
                placeholder = 0.5 * math.log(self.num_classes)
                gammas = [placeholder] * (K - 1)
                logger.warning(
                    f"TINYTTA.GAMMA_MODE='calibrated' but no calibrated gammas are available "
                    f"(no offline checkpoint). Falling back to the PLACEHOLDER "
                    f"{placeholder:.4f} = 0.5*ln({self.num_classes}) for every exit. This is NOT "
                    f"a calibrated threshold and NOT a paper value -- it is only in the ballpark "
                    f"of App. G's per-(model,dataset) numbers (CIFAR10C 1.12-1.53, CIFAR100C "
                    f"2.31-3.26). Run offline_train_tinytta.py before reporting anything."
                )
        self.model.gammas.copy_(torch.tensor(gammas, dtype=self.model.gammas.dtype))

    def _head_param_groups(self):
        """Trainable parameters grouped per exit (index 0..K-1), for the drift diagnostic."""
        groups = [list(h.parameters()) for h in self.model.exits]
        if len(groups) < self.model.K:
            clf = self.model.tail_classifier()
            tail = [p for p in (clf.parameters() if clf is not None else []) if p.requires_grad]
            groups.append(tail)
        return groups

    @staticmethod
    def _flat(params):
        if len(params) == 0:
            return torch.zeros(0)
        return torch.cat([p.detach().reshape(-1) for p in params])

    def collect_params(self):
        """Exit-head parameters only (plus the reused classifier when EXIT_K_TRAINABLE)."""
        params, names = [], []
        for nm, p in self.model.named_parameters():
            if p.requires_grad:
                params.append(p)
                names.append(nm)

        head_counts = self.model.head_params()
        hc = self.model.head_cfg
        default_head = (hc["kernel_size"] == 1 and hc["width_mult"] == 1.0 and hc["affine"]
                        and hc["pool"] == "gap" and hc["ws_target"] in ("conv", "linear", "both")
                        and self.model.exit_k_mode == "reuse_classifier")
        if default_head:
            # Check App. D Table 3's fitted FORM per head against that head's own tap width, rather
            # than against a table keyed by (family, K, C). The form is what Table 3 pins down and it
            # holds for ANY partition; a hard-coded expectation only holds for the default partition
            # and would reject a legitimate TINYTTA.PARTITION override.
            C = self.num_classes
            expected = [c * c + 2 * c + c * C + C
                        for c, _ in self.model.taps[:len(head_counts)]]
            if head_counts != expected:
                raise RuntimeError(
                    f"TinyTTA built exit heads with {head_counts} parameters but the default head "
                    f"form (App. D Table 3: C_in^2 + 2*C_in + C_in*C + C) predicts {expected} for "
                    f"taps {self.model.taps[:len(head_counts)]} and C={C}. The head was not built "
                    f"as intended."
                )
            # Additionally cross-check the DEFAULT partition against the published-derived table, so
            # a regression in the partition itself is still caught.
            key = (self.model.family, self.model.K, C)
            ref = EXPECTED_HEAD_PARAMS.get(key)
            if ref is not None and not self.model.partition_is_override and head_counts != ref:
                raise RuntimeError(
                    f"TinyTTA's DEFAULT partition for {key} produced head params {head_counts} but "
                    f"{ref} was expected (EXPECTED_HEAD_PARAMS). The default partition changed."
                )
        if not params:
            # base.py:52 would set self.optimizer=None and base.py:203 would then die with an
            # opaque AttributeError. Fail readably instead.
            raise RuntimeError(
                "TinyTTA found no trainable parameters. Check TINYTTA.EXIT_K_MODE / "
                "EXIT_K_TRAINABLE and that at least one exit head exists."
            )
        return params, names

    def setup_optimizer(self):
        t = self.cfg.TINYTTA
        if not t.USE_OWN_OPTIM:
            return super().setup_optimizer()
        if t.OPTIM_METHOD == "SGD":
            return torch.optim.SGD(self.params, lr=t.LR, momentum=t.MOMENTUM,
                                   weight_decay=t.WD)
        if t.OPTIM_METHOD == "Adam":
            return torch.optim.Adam(self.params, lr=t.LR, weight_decay=t.WD)
        if t.OPTIM_METHOD == "AdamW":
            return torch.optim.AdamW(self.params, lr=t.LR, weight_decay=t.WD)
        raise NotImplementedError(f"TINYTTA.OPTIM_METHOD={t.OPTIM_METHOD!r}")

    # -- loss ----------------------------------------------------------------
    def build_loss(self, batch_size, exit_idx, per_exit, evaluated):
        """Entropy over the selected (exit, rows) pairs. A1 + A2, both inferred/adjudicated.

        UPDATE_HEADS selects the pairs:
          "exited" -> each sample contributes H only at the exit it actually took (default; the
                      only reading consistent with per-sample exits and Sec. 3.3's "adapts only
                      phi_k").
          "upto_k" -> every exit a sample was evaluated at contributes (Sec. 3.2's literal
                      "adaptation of the submodules preceding this kth submodule").
          "all"    -> every exit for every sample (requires force_full; defeats the early exit).

        TEST_LOSS="entropy_mean_upto_k" divides each sample's contributions by their count, so the
        sample contributes the MEAN of H(p^1..p^k) rather than their sum. With UPDATE_HEADS="exited"
        the two are identical (one exit per sample).

        LOSS_NORM: "batch_mean" divides the total by B -- the exact B->1 limit of TENT's .mean(0),
        and occupancy-weighted across exits. "per_exit_mean" averages within each exit and sums,
        which gives a 3-sample exit the same gradient magnitude as a 90-sample one.

        Returns None when there is nothing to optimize (TEST_LOSS="none", or no rows selected).
        """
        if self.test_loss == "none":
            return None

        pairs = per_exit if self.update_heads == "exited" else evaluated
        if not pairs:
            return None

        # How many exits each sample contributes to (for the "mean_upto_k" weighting).
        counts = torch.zeros(batch_size, device=exit_idx.device)
        for rows, _ in pairs.values():
            counts.index_add_(0, rows, torch.ones_like(rows, dtype=counts.dtype))
        counts = counts.clamp_min(1.0)

        terms = []
        for _, (rows, z) in sorted(pairs.items()):
            ent = self.softmax_entropy(z)
            if self.test_loss == "entropy_mean_upto_k":
                ent = ent / counts[rows]
            if self.loss_norm == "per_exit_mean":
                terms.append(ent.mean())
            else:
                terms.append(ent.sum() / batch_size)
        if not terms:
            return None
        return torch.stack(terms).sum()

    # -- online step ---------------------------------------------------------
    @torch.enable_grad()
    def forward_and_adapt(self, x):
        timing_on = bool(getattr(self.cfg, "TIMING", None)) and self.cfg.TIMING.ENABLED
        measuring = False
        batch_size = x[0].shape[0]
        if timing_on:
            if not hasattr(self, "_timing_results"):
                self._timing_results = []
                self._timing_samples_seen = 0
            measuring = (self._timing_samples_seen >= self.cfg.TIMING.WARMUP_SAMPLES
                         and len(self._timing_results) < self.cfg.TIMING.NUM_MEASUREMENTS)
            if measuring:
                if self.device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

        if self.exit_mode == "grouped":
            outputs = self._adapt_grouped(x[0])
        else:
            outputs = self._adapt_per_sample(x[0])

        if timing_on:
            if measuring:
                if self.device == "cuda":
                    torch.cuda.synchronize()
                # SECONDS PER SAMPLE: test_time.py:146 multiplies by 1000 (cf. norm.py:76).
                self._timing_results.append((time.perf_counter() - t0) / batch_size)
                # See the note in methods/ecotta.py: per-sample storage -> one frame is
                # batch_size units. Without this test_time.py cannot amortize and prints a
                # different line that cluster/collect_results.py does not parse.
                self._timing_units_per_frame = batch_size
            self._timing_samples_seen += batch_size

        self._num_batches += 1
        self._maybe_log()
        return outputs

    def _adapt_grouped(self, imgs):
        force_full = (self.update_heads == "all")
        logits, exit_idx, per_exit, evaluated = self.model.run_exits(
            imgs, count=True, force_full=force_full
        )
        loss = self.build_loss(imgs.shape[0], exit_idx, per_exit, evaluated)
        if loss is not None:
            pre = [self._flat(g).clone() for g in self._param_groups] if self.log_param_drift \
                else None
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._num_steps += 1
            if pre is not None:
                for i, g in enumerate(self._param_groups):
                    d = float(torch.linalg.vector_norm(self._flat(g) - pre[i]).item()) \
                        if pre[i].numel() else 0.0
                    self._drift_path[i] += d
                    if d > 0.0:
                        self._drift_steps[i] += 1
        # Pre-update logits, per the host convention (methods/tent.py:43-47).
        return logits.detach()

    def _adapt_per_sample(self, imgs):
        """A3/Sec. 5.1-faithful loop: one sample, one exit decision, one optimizer step."""
        outs = []
        for i in range(imgs.shape[0]):
            outs.append(self._adapt_grouped(imgs[i:i + 1]))
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def forward_sliding_window(self, x):
        # base.py:139-147. Prediction only: must not touch the exit histogram or adapted state.
        return self.model(x[0])

    # -- diagnostics ---------------------------------------------------------
    def param_drift(self):
        """Per-exit displacement from the offline init, plus the cumulative path length.

        `rel_displacement` is ||theta - theta_init|| / ||theta_init||; `path` is the sum of the
        per-step ||delta theta||, which is larger whenever the update direction changes.
        `nonzero_step_frac` is the fraction of optimizer steps that moved the group at all.
        """
        out = []
        for i, g in enumerate(self._param_groups):
            init = self._init_flat[i]
            now = self._flat(g)
            if init.numel() == 0:
                out.append(dict(exit=i, params=0, rel_displacement=0.0, displacement=0.0,
                                path=0.0, nonzero_step_frac=0.0))
                continue
            disp = float(torch.linalg.vector_norm(now - init.to(now.device)).item())
            base = float(torch.linalg.vector_norm(init).item())
            out.append(dict(
                exit=i,
                params=int(init.numel()),
                displacement=disp,
                rel_displacement=disp / base if base > 0 else float("nan"),
                path=self._drift_path[i],
                nonzero_step_frac=(self._drift_steps[i] / self._num_steps)
                if self._num_steps else 0.0,
            ))
        return out

    def exit_stats(self):
        return dict(
            counts=self.model.exit_counts.tolist(),
            distribution=self.model.exit_distribution(),
            skipped_block_fraction=self.model.skipped_block_fraction(),
            gammas=[round(g, 6) for g in self.model.gammas.tolist()],
            num_batches=self._num_batches,
            num_optimizer_steps=self._num_steps,
        )

    def _maybe_log(self):
        if self._num_batches % self.log_every != 0:
            return
        if self.log_exit_dist:
            st = self.exit_stats()
            dist = st["distribution"]
            skipped = st["skipped_block_fraction"]
            skipped_str = "n/a" if skipped is None else f"{skipped:.4f}"
            logger.info(
                f"TinyTTA EXIT-DIST after {self._num_batches} batches: counts={st['counts']} "
                f"frac={[round(d, 4) for d in dist] if dist else None} "
                f"skipped_backbone_blocks={skipped_str}"
            )
            self._write_diag(st)
            last = dist[-1] if dist else 1.0
            if (not self._warned_no_early_exit
                    and self._num_batches >= self.cfg.TINYTTA.WARN_AFTER_BATCHES
                    and (1.0 - last) < self.cfg.TINYTTA.MIN_EARLY_EXIT_FRAC):
                self._warned_no_early_exit = True
                logger.warning(
                    f"TinyTTA: only {(1.0 - last):.4%} of samples exited before the LAST exit "
                    f"after {self._num_batches} batches. A run in which everything exits at the "
                    f"last head is NOT TinyTTA -- there is no early exit and no self-ensemble, "
                    f"only entropy-SGD on the reused classifier (i.e. sgd_base with "
                    f"STAGE_SELECTION=['fc'], LOSS='entropy'). Likely causes, in order: "
                    f"(1) gammas too tight for corrupted data (gammas="
                    f"{[round(g, 4) for g in self.model.gammas.tolist()]}, ln C = "
                    f"{math.log(self.num_classes):.4f}); (2) exit heads mis-scaled (an "
                    f"under-trained head is over-confident, not uniform, so this usually indicts "
                    f"gamma rather than the heads); "
                    f"(3) the clean-calibration / corrupted-test entropy shift. Do NOT report an "
                    f"accuracy number from this run as TinyTTA without saying this."
                )
        if self.log_param_drift:
            for d in self.param_drift():
                logger.info(
                    f"TinyTTA PARAM-DRIFT exit {d['exit']}: params={d['params']:,} "
                    f"rel_disp={d['rel_displacement']:.3e} disp={d['displacement']:.3e} "
                    f"path={d['path']:.3e} nonzero_steps={d['nonzero_step_frac']:.3f} "
                    f"(steps={self._num_steps})"
                )

    def _write_diag(self, stats):
        try:
            if self._diag_dir is None:
                self._diag_dir = os.path.join(self.cfg.CKPT_DIR, "tinytta_diag")
                os.makedirs(self._diag_dir, exist_ok=True)
            payload = dict(stats)
            payload["param_drift"] = self.param_drift()
            payload["cfg"] = dict(
                dataset=self.cfg.CORRUPTION.DATASET, arch=self.cfg.MODEL.ARCH,
                K=self.cfg.TINYTTA.K, lr=self.cfg.TINYTTA.LR,
                exit_mode=self.exit_mode, ws_mode=self.cfg.TINYTTA.WS_MODE,
                seed=self.cfg.RNG_SEED, batch_size=self.cfg.TEST.BATCH_SIZE,
            )
            fname = (f"exit_hist_{self.cfg.CORRUPTION.DATASET}_{self.cfg.MODEL.ARCH}"
                     f"_K{self.cfg.TINYTTA.K}_seed{self.cfg.RNG_SEED}.json")
            with open(os.path.join(self._diag_dir, fname), "w") as f:
                json.dump(payload, f, indent=2)
        except OSError as exc:                                    # never kill a run over a log
            logger.warning(f"TinyTTA could not write its exit-histogram diagnostic: {exc}")

    def reset(self):
        super().reset()
        self.model.reset_counts()
