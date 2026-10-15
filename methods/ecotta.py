"""
EcoTTA: Memory-Efficient Continual Test-time Adaptation via Self-distilled Regularization
Song et al., CVPR 2023 -- https://arxiv.org/abs/2303.01904

Mechanics cross-checked against the unofficial reference implementation
https://github.com/Lily-Le/EcoTTA (files ecotta_models/wideresnet28.py, main.py, utils.py).
Where the two disagree, this file follows the PAPER and exposes the reference behaviour
behind a cfg flag. The disagreements are documented inline.


Architecture
------------
The frozen pre-trained encoder is split into K contiguous groups of residual blocks.
Each group gets one lightweight, trainable "meta network": a single BN layer placed BEFORE the
frozen group, plus a residual Conv-BN-ReLU bypass around it (`ECOTTA.BN_PLACEMENT="before"`,
the default):

    h_k         = meta_bn_k(x_{k-1})                          # BN on the group INPUT
    x_k         = frozen_group_k(h_k)                          # the frozen part's output
    x_tilde_k   = x_k + ReLU(BN(Conv3x3_s(h_k)))               # meta network output
    R_k         = || x_tilde_k - sg(x_k) ||_1                   # self-distilled regularization

(`sg` = stop-gradient. `ECOTTA.REG_TEACHER` selects the teacher; see below.)

so one meta network = 1 conv + 2 BN layers. Only the meta networks are ever trained; the
pre-trained encoder, the stem conv and the classifier stay frozen for the whole run.

Both the frozen path AND the conv bypass consume the NORMALIZED input h_k. That placement is
the load-bearing part of the design: it is what re-normalizes the features entering the frozen
convolutions under domain shift. Every BN inside the frozen residual blocks keeps its source
statistics (25 of them on WRN-28-10, 31 on ResNeXt-29), so if the meta BN sat downstream of the
frozen group instead, nothing in the network would adapt the statistics of the features those
frozen convs actually see.

Paper Sec. 4.2 "Architecture design" evaluates exactly that downstream arrangement as ablation
"design (i)" and REJECTS it:

    "In design (i), we enforce both BN parameters and Conv layers in the meta networks to take
     the output of the original networks as inputs. Such a design brings performance drop. We
     speculate that it is because the original network, which is not adapted to the target
     domain, lacks the ability to extract sufficiently meaningful features from the target
     image."

Table 4a, error rate (%), lower is better:

    proposed (BN before)   16.8 (WRN-28)  12.1 (WRN-40)  36.3 (WRN-40, CIFAR100-C)
    design (i)  BN after   18.1           12.6           37.2
    design (ii) w/o BN     18.7           13.7           38.2
    design (iii) w/o Conv  20.7           14.9           40.1
    design (iv) Conv only  60.6           73.3           77.2

`ECOTTA.BN_PLACEMENT="after"` reproduces design (i) for the BN branch and is kept as a
legitimate ablation. It is NOT the paper's method and must not be labelled EcoTTA.


Why the conv bypass reads the group INPUT (and not the group output)
-------------------------------------------------------------------
A sequential output-side conv (C_out -> C_out, stride 1) would cost, at K=5:
    9 * (160^2 + 160^2 + 320^2 + 320^2 + 640^2) = 5,990,400 = 16.42% of 36,479,194
against the parallel input-side form's 3,484,192 = 9.55%. The paper reports single-digit
parameter overhead for WRN-28, so the output-side conv is excluded. Both forms are
shape-consistent, so shape is not the discriminator here; the magnitude of the overhead is.

CAUTION -- the parameter count does NOT discriminate the BN PLACEMENT, and the widely quoted
"9.5%" cannot be used as evidence for it:
  * BN before (this default): 3,479,040 conv + 2*sum(C_in) + 2*sum(C_out) = 3,484,192 = 9.551%
  * BN after  (design (i)):   3,479,040 conv + 4*sum(C_out)               = 3,485,440 = 9.554%
    The two differ by 1,248 parameters -- 0.003 percentage points. Indistinguishable.
  * Table 14's partition factor is stated as K=4 (verified against the paper), yet its
    WRN-28/kernel-3 entry of 9.5% only reproduces at K=5 under either BN placement; at K=4 this
    topology gives 7.02%. Its kernel-1 column is also internally inconsistent for any topology
    (solving A+B = 0.8%, 9A+B = 9.5% yields a negative BN contribution).
So the BN placement is settled by the Sec. 4.2 / Table 4a ablation quoted above, NOT by
arithmetic. Do not re-derive it from the parameter table.


Which tensor is the regularization teacher -- ECOTTA.REG_TEACHER
---------------------------------------------------------------
Eq. (4)'s `x_k` is "the output of the frozen part". Under BN-before that admits two readings:

  "inner_frozen": x_k = frozen_group(h_k), the frozen part's output as it sits in the
      adapted network. R_k is then a DRIFT penalty: how far this group's output has been moved
      off the frozen path by the residual bypass. Measured R_k ~ 0.3, which is commensurate with
      the entropy term and makes the paper's lambda = 0.5 dimensionally sensible.

  "source_path" (DEFAULT): x_k = frozen_group(x_{k-1}), the frozen part applied to the UN-normalized
      input. This is the faithful analogue of the reference, whose target
      `out1 = original_part(x)` (wideresnet28.py:122, 133) is a function of the group INPUT only
      and is therefore PARAMETER-INDEPENDENT -- a genuine anchor to source behaviour.

Note the distinction between detaching and parameter-independence. Both settings detach the
target, so neither leaks gradient. But under "inner_frozen" the target frozen_group(meta_bn(x))
still MOVES as meta_bn is updated from step to step, so R_k only penalizes the residual bypass
and pins nothing to source behaviour; numerically R_k = ||bypass(h)||_1. Only "source_path"
holds a fixed reference. Measured on real gaussian_noise-5 data, lambda*sum_k R_k = 1.61
(source_path) vs 0.86 (inner_frozen), against a loss_ent of 0.18.


KNOWN DEFECT -- the online update collapses on the paper's continual protocol
----------------------------------------------------------------------------
Measured on CIFAR-10-C severity 5, continual, 15 corruptions, batch 64, K=5, lambda=0.5,
lr 5e-3, after a 10-epoch warm-up reaching 93.45% clean test accuracy:

    ECOTTA.LR = 0 (no online update, static warmed net):  22.24% mean error  -- stable, sane
    ECOTTA.LR = 5e-3, REG_TEACHER=inner_frozen:           82.43% mean error  -- collapses
    ECOTTA.LR = 5e-3, REG_TEACHER=source_path:            81.33% mean error  -- collapses
    paper, EcoTTA K=5:                                    16.8%  mean error

So the ARCHITECTURE is sound (22.24% static, and per-corruption errors are sensible: brightness
8.64%, defocus_blur 11.46%) and the defect is localized to the online update, which is actively
destructive rather than merely unhelpful.

Mechanism, measured on the first batch of gaussian_noise-5:

    loss_ent  = 0.183   ||grad|| = 8.41e-01
    l*sum R_k = 0.855   ||grad|| = 9.59e-02      -> reg/ent gradient-norm ratio = 0.11

The two loss VALUES are comparable, but the entropy gradient dominates the regularizer's by ~9x.
Entropy minimization is self-reinforcing (as the model grows confident, more samples pass the
H_0 filter and reinforce the collapse -- the empty-filter counter never fires during a collapsed
run), so a regularizer contributing 11% of the gradient cannot anchor it.

DIAGNOSED: the destabilizer is SGD MOMENTUM, not the reduction and not the teacher.

Two hypotheses were tested and REJECTED:
  * Reduction. The reference uses mean-over-all-elements (wideresnet28.py:132-133,
    `L1Loss(reduction='none')(...).mean()`), which is exactly what "mean_all" does. The literal
    ||.||_1 reading ("per_sample_l1") is catastrophically mis-scaled: measured
    lambda*sum_k R_k = 87,425 with a reg/ent gradient-norm ratio of 16,181 (vs 0.11 for
    mean_all). So the paper's notation cannot be taken literally and the reference's reading is
    the only viable one.
  * Teacher. Both teachers collapse at momentum 0.9 (82.43% / 81.33% mean error).

Momentum is the actual cause. The paper never reports it; this file chose 0.9 because the
reference uses 0.9 (main.py:282) -- but the reference pairs 0.9 with the BN-AFTER placement,
where the trainable BN sits downstream of the frozen convs and is a far weaker lever. Under
BN-BEFORE the meta BN feeds an entire frozen stack, so the same step size diverges. Measured on
the first three severity-5 corruptions (gaussian / shot / impulse):

    momentum 0.9, source_path:   27.98 / 39.01 / 78.34   -> collapses
    momentum 0.0, source_path:   29.57 / 25.59 / 38.80   -> stable
    ECOTTA.LR = 0 (static):      30.66 / 27.10 / 42.58
    local BN-Stats (bntest):     29.11 / 26.32 / 36.56
    paper EcoTTA K=5:            23.8  / 18.7  / 25.7

Momentum 0.0 removes the collapse and beats the static warmed network, but only reaches roughly
BN-Stats parity and remains 6-13 points off the paper.

The divergence is NOT WideResNet-specific. Reproduced on CIFAR-100-C / ResNeXt-29 (continual,
severity 5, first three corruptions, batch 64, after a 10-epoch warm-up reaching 76.39% clean
test accuracy vs the source model's 78.89%):

    BN-Stats (bntest)              43.06 / 40.67 / 43.31   mean 42.35
    ECOTTA.LR = 0 (static)         51.39 / 48.62 / 51.35   mean 50.45
    momentum 0.9                   59.49 / 91.17 / 98.90   mean 83.19  -> collapses to chance (99%)
    momentum 0.0                   51.27 / 49.23 / 53.38   mean 51.29  -> stable

So BN-before + momentum diverges on both backbones. On this backbone momentum 0.0 does not even
beat the static line (51.29 vs 50.45), and both trail BN-Stats by ~8 points -- consistent with
FROZEN_BN_MODE="eval" leaving all 31 frozen ResNeXt BNs on source statistics while BN-Stats
adapts every one of them, with only K=5 meta BNs available to compensate. `ECOTTA.MOMENTUM` is therefore left at the
reference's 0.9 as the documented default, and momentum 0.0 is recorded here as a TUNED value
supported by neither the paper nor the reference. The residual gap to 23.8 / 18.7 / 25.7 is
unexplained and this implementation should not be presented as a reproduction of EcoTTA.

Online objective (paper Eq. (3) + App. D Eq. (7))
-------------------------------------------------
    L = (1/N) * sum_i 1{H(y_hat_i) < H_0} * H(y_hat_i)  +  lambda * sum_k R_k
    H_0 = ECOTTA.E_MARGIN * ln(C)

The entropy term is normalised by the FULL batch size N, not by the number of surviving
samples (App. D Eq. (7)). If no sample survives the filter the entropy term is exactly 0.


How Eq. (3) is actually optimized -- ECOTTA.UPDATE_RULE
------------------------------------------------------
Eq. (3) reads as one combined objective, but the paper's Algorithm 1 (App., p. 13) evaluates
it in two passes, because the stop-gradient needed by the regularizer must NOT apply to the
entropy term:

    1. entropy pass:  no detach. L_ent.backward().  Gradients reach ALL K meta networks,
                      since the entropy is a function of the final logits.
    2. Detach_parts(model), then regularization pass: each partition's input is detached, so
       dR_k only touches theta_k and never back-propagates into an upstream meta network.
       (lambda * sum_k R_k).backward() ACCUMULATES onto the pass-1 gradients.
    3. a SINGLE optimizer step on the accumulated gradients.

This is `UPDATE_RULE="algorithm1"` (the default) and it is why two forwards per batch are
inherent to the method rather than a reference-implementation quirk: detaching cannot be
selective within a single shared forward without recomputing the frozen groups anyway.

  - "algorithm1"     : paper Algorithm 1 as above -- 2 forwards, 2 backwards, 1 optimizer step.
  - "two_step"       : the unofficial reference (main.py:59-87) -- same two passes but TWO
                       separate optimizer steps. Kept for fidelity checks. NOTE: the reference
                       additionally reports the post-update forward's logits (main.py:88), which
                       inflates accuracy; this implementation always reports pre-update logits.
  - "single_forward" : 1 forward, combined loss, 1 step. DEVIATION: with only one graph the
                       detach cannot be selective, so dR_k also flows into upstream meta
                       networks (Algorithm 1's isolation is lost). Roughly halves the per-batch
                       cost, which matters for the error-vs-latency Pareto comparison, so it is
                       offered explicitly -- but it is not the paper's method.

The same subtlety governs warm-up: cross-entropy must reach every meta network, so warm-up runs
with detaching OFF (see warmup_train_ecotta.py). Leaving the detach on during warm-up would
train only the deepest meta network and silently produce a much weaker checkpoint.
"""

import os
import time
import math
import logging
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import Entropy

logger = logging.getLogger(__name__)


BN_PLACEMENTS = ("before", "after")


# ===========================================================================
# Backbone shims
# ===========================================================================
# EcoTTA has to (a) enumerate the encoder's residual blocks in order, (b) partition them,
# (c) know each group's input/output channels and stride, and (d) rebuild the stem and head
# because both backbones use hand-written `forward` methods. All of that is backbone-specific,
# so it lives in one shim per family behind a single interface rather than in branches inside
# the group builder.
#
# Partition principle (paper Table 13): the ENCODER only is partitioned -- the input conv and
# the classifier stay outside every group -- non-uniformly, denser at the shallow end.
#
# Conv kernel size (paper Table 14 + Sec. 4.2, verbatim): "we obtain better performances by
# setting the kernel size to three with WideResNet (with 10% additional number of model
# parameters). On the other hand, utilizing the kernel size of three with ResNet leads to
# significant increases in parameters and memory sizes. Thus, we use one and three as the kernel
# size with ResNet and WideResNet, respectively." Hence `default_kernel_size` is per family.


class BackboneShim:
    """Interface every backbone shim implements."""

    family = None                 # str, part of the warm-up topology signature
    partitions = {}               # {K: [blocks per group, shallow -> deep]}
    default_kernel_size = 3

    @staticmethod
    def matches(model) -> bool:
        raise NotImplementedError

    def __init__(self, model):
        self.model = model

    def blocks(self):
        """Ordered list of the encoder's residual blocks."""
        raise NotImplementedError

    def block_geometry(self, group_blocks):
        """(c_in, c_out, stride) for a contiguous run of blocks."""
        raise NotImplementedError

    def build_stem(self) -> nn.Module:
        """Everything before the first group. MUST include any input normalization."""
        raise NotImplementedError

    def build_head(self) -> nn.Module:
        """Everything after the last group, including the classifier."""
        raise NotImplementedError


class _WRNStem(nn.Module):
    def __init__(self, wrn):
        super().__init__()
        self.conv1 = wrn.conv1

    def forward(self, x):
        return self.conv1(x)


class _WRNHead(nn.Module):
    """WRN tail: relu(bn1(x)) -> avg_pool2d(8) -> flatten -> fc."""

    def __init__(self, wrn):
        super().__init__()
        self.bn1, self.relu, self.fc = wrn.bn1, wrn.relu, wrn.fc
        self.n_channels = wrn.nChannels

    def forward(self, x):
        out = self.relu(self.bn1(x))
        out = F.avg_pool2d(out, 8)
        return self.fc(out.view(-1, self.n_channels))


class WideResNetShim(BackboneShim):
    """RobustBench `Standard` = WideResNet-28-10 (CIFAR-10-C).

    12 pre-activation blocks: block{1,2,3}.layer.{0..3}. No input normalizer (this entry is a
    bare WideResNet, unlike most other RobustBench CIFAR models).
    """

    family = "wrn28"
    partitions = {4: [2, 2, 4, 4], 5: [2, 2, 2, 2, 4]}   # paper Table 13
    default_kernel_size = 3                              # paper Sec. 4.2 for WideResNet

    @staticmethod
    def matches(model):
        return type(model).__name__ == "WideResNet" and hasattr(model, "nChannels")

    def blocks(self):
        out = []
        for stage in (self.model.block1, self.model.block2, self.model.block3):
            out.extend(list(stage.layer.children()))
        return out

    def block_geometry(self, group_blocks):
        c_in = group_blocks[0].bn1.num_features      # pre-activation: bn1 sees the block input
        c_out = group_blocks[-1].conv2.out_channels
        stride = 1
        for blk in group_blocks:
            stride *= blk.conv1.stride[0]
        return c_in, c_out, stride

    def build_stem(self):
        return _WRNStem(self.model)

    def build_head(self):
        return _WRNHead(self.model)


class _ResNeXtStem(nn.Module):
    """ResNeXt stem, INCLUDING the input normalization.

    `Hendrycks2020AugMixResNeXtNet.forward` does `x = (x - self.mu) / self.sigma` before calling
    the base `CifarResNeXt.forward` (robustbench/model_zoo/cifar100.py). The normalization is a
    subclass forward override, not a Sequential wrapper, so rebuilding the graph would silently
    drop it and feed un-normalized [0,1] images to the frozen convs. mu/sigma are registered
    buffers, so re-registering them here keeps `.to(device)` working.
    """

    def __init__(self, net):
        super().__init__()
        self.normalize = hasattr(net, "mu") and hasattr(net, "sigma")
        if self.normalize:
            self.register_buffer("mu", net.mu.clone())
            self.register_buffer("sigma", net.sigma.clone())
        self.conv_1_3x3 = net.conv_1_3x3
        self.bn_1 = net.bn_1

    def forward(self, x):
        if self.normalize:
            x = (x - self.mu) / self.sigma
        return F.relu(self.bn_1(self.conv_1_3x3(x)))


class _ResNeXtHead(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.avgpool = net.avgpool
        self.classifier = net.classifier

    def forward(self, x):
        out = self.avgpool(x)
        return self.classifier(out.view(out.size(0), -1))


class ResNeXtShim(BackboneShim):
    """RobustBench `Hendrycks2020AugMix_ResNeXt` = CifarResNeXt-29 (CIFAR-100-C).

    9 ResNeXtBottleneck blocks in 3 stages of 3: stage_{1,2,3}.{0,1,2}.
    Per-block geometry (verified by instantiation):

        idx  module      C_in  C_out  stride  H_in
        0    stage_1.0     64    256     1     32
        1    stage_1.1    256    256     1     32
        2    stage_1.2    256    256     1     32
        3    stage_2.0    256    512     2     32
        4    stage_2.1    512    512     1     16
        5    stage_2.2    512    512     1     16
        6    stage_3.0    512   1024     2     16
        7    stage_3.1   1024   1024     1      8
        8    stage_3.2   1024   1024     1      8

    Partition: with only 3 blocks per stage, the WRN K=5 pattern (split the two shallow stages
    in half, keep the deepest whole) maps to [1,2, 1,2, 3]. Every group stays INSIDE one stage,
    and each stride-2 transition is isolated as its own group's first block. Denser at the
    shallow end, per Table 13. K=4 uses [1,2,3,3] -- stage_1 split, stages 2 and 3 whole.

    The frozen blocks' 3x3 convs use `groups=4` (ResNeXt cardinality). That is irrelevant to the
    meta networks: `block_geometry` reads the bottleneck's OUTER channel counts
    (`conv_reduce.in_channels` / `conv_expand.out_channels`), which are the group's actual input
    and output streams, and the meta conv is a fresh dense `groups=1` conv on those streams.
    """

    family = "resnext29"
    partitions = {4: [1, 2, 3, 3], 5: [1, 2, 1, 2, 3]}
    default_kernel_size = 1        # paper Sec. 4.2: kernel 1 for the ResNet family

    @staticmethod
    def matches(model):
        return hasattr(model, "stage_1") and hasattr(model, "classifier") \
            and hasattr(model, "conv_1_3x3")

    def blocks(self):
        out = []
        for stage in (self.model.stage_1, self.model.stage_2, self.model.stage_3):
            out.extend(list(stage.children()))
        return out

    def block_geometry(self, group_blocks):
        # Outer bottleneck streams, NOT the grouped inner conv's channels.
        c_in = group_blocks[0].conv_reduce.in_channels
        c_out = group_blocks[-1].conv_expand.out_channels
        stride = 1
        for blk in group_blocks:
            stride *= blk.conv_conv.stride[0]
        return c_in, c_out, stride

    def build_stem(self):
        return _ResNeXtStem(self.model)

    def build_head(self):
        return _ResNeXtHead(self.model)


BACKBONE_SHIMS = (WideResNetShim, ResNeXtShim)


def get_backbone_shim(model) -> BackboneShim:
    for shim_cls in BACKBONE_SHIMS:
        if shim_cls.matches(model):
            return shim_cls(model)
    raise ValueError(
        f"EcoTTA has no backbone shim for {type(model).__name__}. Supported families: "
        f"{[s.family for s in BACKBONE_SHIMS]}. Add a BackboneShim subclass exposing blocks(), "
        f"block_geometry(), build_stem() and build_head() -- note build_stem() must reproduce any "
        f"input normalization the original forward applied."
    )


# Regression guards: expected trainable meta-network parameter count, keyed by
# family -> BN placement -> K, at that family's default kernel size.
# The two BN placements differ by only ~1.2k parameters, which is exactly why the parameter
# count cannot settle the BN placement (see the module docstring).
EXPECTED_META_PARAMS = {
    "wrn28": {                              # kernel 3
        "before": {4: 2_561_312, 5: 3_484_192},
        "after":  {4: 2_562_560, 5: 3_485_440},
    },
    "resnext29": {                          # kernel 1
        "before": {4: 743_552, 5: 1_007_744},    # 10.776% / 14.605% of 6,900,132
        "after":  {4: 745_472, 5: 1_009_664},
    },
}


class MetaBlock(nn.Module):
    """One meta network: a single BN layer plus a Conv-BN-ReLU bypass.

    `bn_placement="before"` (paper Sec. 4.2, default): `meta_bn` normalises the group's INPUT,
    so it is sized to `c_in` and both the frozen path and the bypass consume its output.
    `bn_placement="after"` (rejected design (i)): `meta_bn` normalises the frozen group's
    OUTPUT, so it is sized to `c_out`.

    `conv -> conv_bn -> ReLU` is the bypass branch; it carries the group's spatial stride and
    maps c_in -> c_out so the two summands line up.
    """

    def __init__(self, c_in: int, c_out: int, stride: int, zero_init_conv: bool = False,
                 bn_placement: str = "before", kernel_size: int = 3):
        super().__init__()
        if bn_placement not in BN_PLACEMENTS:
            raise ValueError(f"bn_placement must be one of {BN_PLACEMENTS}, got {bn_placement!r}")
        if kernel_size not in (1, 3):
            raise ValueError(f"meta conv kernel_size must be 1 or 3 (paper Tab. 14), got {kernel_size}")
        self.c_in, self.c_out, self.stride = c_in, c_out, stride
        self.bn_placement = bn_placement
        self.kernel_size = kernel_size
        self.meta_bn = nn.BatchNorm2d(c_in if bn_placement == "before" else c_out)
        # Fresh dense conv (groups=1) regardless of the frozen backbone's cardinality: c_in/c_out
        # are the group's OUTER streams, so no grouping assumption is inherited from the backbone.
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride,
                              padding=1 if kernel_size == 3 else 0, bias=False)
        self.conv_bn = nn.BatchNorm2d(c_out)

        if zero_init_conv:
            # Make the bypass an exact no-op at init so that, before warm-up, the network
            # reproduces the source function up to `meta_bn`. Only for the cold-start ablation;
            # the paper (and the reference) rely on warm-up instead.
            nn.init.zeros_(self.conv.weight)
            nn.init.zeros_(self.conv_bn.weight)
            nn.init.zeros_(self.conv_bn.bias)

    def bypass(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv_bn(self.conv(x)))


class MetaGroup(nn.Module):
    """A frozen block group wrapped by its meta network.

    Port of `one_part_of_networks` (reference ecotta_models/wideresnet28.py:111-134).

    `reg_loss` holds R_k for the most recent forward when `compute_reg` is True.

    `detach_input` is a per-pass switch, NOT a fixed property: Algorithm 1 applies
    `Detach_parts` only for the regularization pass. It must be False for the entropy pass and
    for warm-up, otherwise gradients cannot reach anything but the deepest meta network.
    """

    def __init__(self, frozen_group: nn.Sequential, meta: MetaBlock, detach_input: bool = False,
                 reg_teacher: str = "inner_frozen", reg_reduction: str = "mean_all",
                 reg_detach_target: bool = True):
        super().__init__()
        self.frozen_group = frozen_group
        self.meta = meta
        self.detach_input = detach_input
        self.reg_teacher = reg_teacher
        self.reg_reduction = reg_reduction
        self.reg_detach_target = reg_detach_target
        self.compute_reg = False
        self.reg_loss = None
        # Calibration-only switch: when True the meta BN is skipped entirely, so with a
        # zero-initialised bypass the network computes exactly the source function. Used by
        # calibrate_identity_init to capture each group's true source input statistics.
        self.bypass_meta_bn = False

    @contextlib.contextmanager
    def _frozen_bns_in_eval(self):
        """Temporarily force the frozen group's BNs into eval mode.

        Only needed for the teacher forward under `FROZEN_BN_MODE="train"`: without this, the
        extra frozen pass would fold the same batch into the frozen running statistics a second
        time. A no-op under the default `FROZEN_BN_MODE="eval"`.
        """
        flipped = [m for m in self.frozen_group.modules()
                   if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.training]
        for m in flipped:
            m.eval()
        try:
            yield
        finally:
            for m in flipped:
                m.train()

    def _reg(self, out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """R_k = ||out - target||_1 under the configured reduction.

        Reference ground truth (ecotta_models/wideresnet28.py:132-133):
            loss = nn.L1Loss(reduction='none'); btsloss = loss(out, out1.detach()).mean()
        i.e. mean over ALL elements -- which is `reduction="mean_all"`, the default here.
        The paper writes ||.||_1, which literally denotes a sum; "per_sample_l1" and "sum"
        expose those readings because the paper's notation and the reference disagree.
        """
        if self.reg_detach_target:
            target = target.detach()
        diff = (out - target).abs()
        if self.reg_reduction == "mean_all":
            return diff.mean()
        if self.reg_reduction == "per_sample_l1":
            # ||.||_1 per sample (sum over features), averaged over the batch.
            return diff.flatten(1).sum(1).mean()
        if self.reg_reduction == "sum":
            return diff.sum()
        raise ValueError(f"unknown reg_reduction {self.reg_reduction!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Algorithm 1 `Detach_parts`: in the regularization pass, detach between partitions so
        # each R_k only produces gradients for its own theta_k.
        if self.detach_input:
            x = x.detach()

        if self.meta.bn_placement == "before":
            # Paper Sec. 4.2: the meta BN normalizes the features ENTERING the frozen group, so
            # both the frozen path and the conv bypass consume the normalized input.
            h = x if self.bypass_meta_bn else self.meta.meta_bn(x)
            frozen_out = self.frozen_group(h)
            out = frozen_out + self.meta.bypass(h)
            if self.compute_reg:
                if self.reg_teacher == "inner_frozen":  # noqa: SIM108
                    # Eq. (4) teacher: the output of the frozen part AS IT SITS IN THE ADAPTED
                    # NETWORK. R_k then measures how far this group's output has been moved away
                    # from the frozen path -- a drift/proximity penalty, which is what
                    # "self-distilled regularization ... preserving the source knowledge" around
                    # a residual connection means, and what makes the paper's lambda = 0.5
                    # dimensionally sensible.
                    self.reg_loss = self._reg(out, frozen_out)
                else:
                    # "source_path": compare against the frozen group on the UN-normalized input.
                    # Do NOT use this as a default. It is dominated by a large constant offset
                    # (frozen(h) vs frozen(x) differ by the whole normalization), so R_k is O(5)
                    # instead of O(0.1), swamps the entropy term at lambda = 0.5, and its gradient
                    # pushes meta_bn toward undoing its own normalization. Measured: it drives
                    # entropy collapse to chance level by the 3rd corruption of a continual run.
                    with torch.no_grad(), self._frozen_bns_in_eval():
                        teacher = self.frozen_group(x)
                    self.reg_loss = self._reg(out, teacher)
            else:
                self.reg_loss = None
        else:
            # Rejected design (i) (paper Sec. 4.2 / Table 4a): the meta BN consumes the frozen
            # group's OUTPUT. Kept as an ablation only -- see the module docstring.
            frozen_out = self.frozen_group(x)
            out = self.meta.meta_bn(frozen_out) + self.meta.bypass(x)
            if self.compute_reg:
                self.reg_loss = self._reg(out, frozen_out)
            else:
                self.reg_loss = None
        return out


class EcoTTANet(nn.Module):
    """A frozen backbone rebuilt with K meta networks inserted, generic over BackboneShim.

    Both supported backbones use hand-written `forward` methods (WideResNet:
    robustbench/.../wide_resnet.py:86-94; ResNeXt: robustbench/model_zoo/cifar100.py, which also
    applies input normalization inside `forward`), so meta networks cannot be inserted by
    swapping children in place -- the graph has to be rebuilt. The frozen parameter tensors are
    reused by reference (no copy, no re-download); the shim's stem reproduces any normalization.
    """

    def __init__(self, model: nn.Module, K: int, zero_init_conv=False, detach_input=False,
                 bn_placement="before", reg_teacher="inner_frozen", reg_reduction="mean_all",
                 reg_detach_target=True, conv_kernel_size=0):
        super().__init__()
        self.bn_placement = bn_placement
        self.reg_teacher = reg_teacher
        self.reg_reduction = reg_reduction
        self.reg_detach_target = reg_detach_target

        shim = get_backbone_shim(model)
        self.family = shim.family
        if K not in shim.partitions:
            raise ValueError(
                f"ECOTTA.K must be one of {sorted(shim.partitions)} for backbone family "
                f"'{shim.family}', got {K}."
            )
        num_blocks_per_group = shim.partitions[K]
        # 0 -> the family's paper-justified default (3 for WideResNet, 1 for the ResNet family).
        self.conv_kernel_size = conv_kernel_size or shim.default_kernel_size

        blocks = shim.blocks()
        if len(blocks) != sum(num_blocks_per_group):
            raise ValueError(
                f"Group split {num_blocks_per_group} covers {sum(num_blocks_per_group)} blocks "
                f"but backbone family '{shim.family}' exposes {len(blocks)}."
            )

        # Stem and head stay outside every group (paper Table 13 / Fig. 3 "input_conv").
        self.stem = shim.build_stem()

        groups, start = [], 0
        for n_blocks in num_blocks_per_group:
            group_blocks = blocks[start:start + n_blocks]
            c_in, c_out, stride = shim.block_geometry(group_blocks)
            groups.append(
                MetaGroup(
                    nn.Sequential(*group_blocks),
                    MetaBlock(c_in, c_out, stride, zero_init_conv=zero_init_conv,
                              bn_placement=bn_placement,
                              kernel_size=self.conv_kernel_size),
                    detach_input=detach_input,
                    reg_teacher=reg_teacher,
                    reg_reduction=reg_reduction,
                    reg_detach_target=reg_detach_target,
                )
            )
            start += n_blocks
        self.groups = nn.ModuleList(groups)

        self.head = shim.build_head()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        for group in self.groups:
            out = group(out)
        return self.head(out)

    # -- meta-network plumbing ------------------------------------------------
    def set_compute_reg(self, flag: bool):
        for group in self.groups:
            group.compute_reg = flag

    def set_detach(self, flag: bool):
        """Algorithm 1 `Detach_parts`. True only for the regularization pass."""
        for group in self.groups:
            group.detach_input = flag

    def set_bypass_meta_bn(self, flag: bool):
        for group in self.groups:
            group.bypass_meta_bn = flag

    def reg_losses(self):
        return [g.reg_loss for g in self.groups]

    def meta_modules(self):
        return [g.meta for g in self.groups]

    def geometry(self):
        """(c_in, c_out, stride) per group -- used by the tests."""
        return [(g.meta.c_in, g.meta.c_out, g.meta.stride) for g in self.groups]

    def topology_signature(self) -> str:
        """A string that changes whenever the meta-network topology changes.

        Stored in the warm-up checkpoint and enforced on load. A warm-up artifact is only valid
        for the exact topology it was trained against: the backbone family and the BN placement
        both change the module tree AND the function being learned, so silently loading across
        them would produce a plausible-looking but meaningless run.
        """
        n_meta = sum(p.numel() for m in self.meta_modules() for p in m.parameters())
        # NOTE: ECOTTA.REG_TEACHER is deliberately NOT part of the signature. It only affects the
        # online regularization target; warm-up is a plain cross-entropy fit with the regularizer
        # switched off, so one warm-up artifact is valid for every teacher setting. The family,
        # kernel size, K, geometry and BN_PLACEMENT *are* included.
        return (f"{self.family}|parallel_conv_k{self.conv_kernel_size}|bn_{self.bn_placement}"
                f"|K{len(self.groups)}|geom={self.geometry()}|meta_params={n_meta}")


# Backwards-compatible alias: the class was WRN-only before the backbone-shim refactor.
EcoTTAWideResNet = EcoTTANet


def default_warmup_path(cfg) -> str:
    """Auto-derived warm-up checkpoint path.

    Follows the `ckpt/warmup/` convention of methods/rmt.py:113-120 but with a distinct
    `ckpt_ecotta_` prefix so it can never collide with RMT's `ckpt_warmup_*` files.
    Shared by warmup_train_ecotta.py (writer) and EcoTTA.configure_model (reader).

    The filename carries only the keys that determine CHECKPOINT COMPATIBILITY -- dataset,
    architecture, K, BN placement and seed. Training hyperparameters (epochs, lr, batch size) are
    recorded in the checkpoint payload and logged on load instead of being baked into the path, so
    that an online run does not silently fail to find a perfectly usable warm-up just because e.g.
    WARMUP_EPOCHS was left at its default. A mismatch is reported as a warning at load time.

    BN placement is in the filename so that the paper default and the design-(i) ablation can
    coexist; the payload additionally carries a full topology signature that is enforced on load.
    """
    fname = (
        f"ckpt_ecotta_{cfg.CORRUPTION.DATASET}_{cfg.MODEL.ARCH}"
        f"_K{cfg.ECOTTA.K}"
        f"_bn{cfg.ECOTTA.BN_PLACEMENT}"
        f"_seed{cfg.RNG_SEED}.pth"
    )
    # ARCH already distinguishes the backbone family in the filename; the payload's topology
    # signature (which embeds the family explicitly) is what actually enforces compatibility.
    return os.path.join(cfg.CKPT_DIR, "warmup", fname)


def build_ecotta_model(cfg, model, device, zero_init_conv=None):
    """Wrap a supported backbone into an EcoTTANet. Backbone family is auto-detected."""
    if cfg.ECOTTA.BN_PLACEMENT not in BN_PLACEMENTS:
        raise ValueError(
            f"ECOTTA.BN_PLACEMENT must be one of {list(BN_PLACEMENTS)}, "
            f"got {cfg.ECOTTA.BN_PLACEMENT!r}."
        )

    zero_init = cfg.ECOTTA.ZERO_INIT_CONV if zero_init_conv is None else zero_init_conv
    # detach_input starts False; the online update rule flips it on per regularization pass.
    return EcoTTANet(
        model,
        cfg.ECOTTA.K,
        zero_init_conv=zero_init,
        detach_input=False,
        bn_placement=cfg.ECOTTA.BN_PLACEMENT,
        reg_teacher=cfg.ECOTTA.REG_TEACHER,
        reg_reduction=cfg.ECOTTA.REG_REDUCTION,
        reg_detach_target=cfg.ECOTTA.REG_DETACH_TARGET,
        conv_kernel_size=cfg.ECOTTA.CONV_KERNEL_SIZE,
    ).to(device)


def apply_bn_modes(ecotta_model, frozen_bn_mode: str):
    """Set train/eval modes: frozen backbone per `frozen_bn_mode`, meta BNs always in train mode.

    Paper default `frozen_bn_mode="eval"`: the original network is frozen and keeps its source
    running statistics, so ALL test-time normalization comes from the meta networks' own BNs
    (which run in train mode and therefore use test-batch statistics).

    The reference implementation instead puts the whole net in train() (main.py:332), which
    silently grants the frozen encoder full BN-Stats adaptation and destroys the source
    running statistics -- that confounds EcoTTA with a BN-adaptation baseline. Reachable via
    `frozen_bn_mode="train"` for a fidelity check, but it is not the paper's method.
    """
    if frozen_bn_mode == "train":
        ecotta_model.train()
    elif frozen_bn_mode == "eval":
        ecotta_model.eval()
    elif frozen_bn_mode == "batch_stats":
        # Host idiom (cf. methods/tent.py:75-81): eval mode but BN uses batch statistics and
        # the source running statistics are discarded.
        ecotta_model.eval()
        frozen_bns = []
        meta_bn_ids = {id(m) for meta in ecotta_model.meta_modules()
                       for m in (meta.meta_bn, meta.conv_bn)}
        for module in ecotta_model.modules():
            if isinstance(module, nn.BatchNorm2d) and id(module) not in meta_bn_ids:
                frozen_bns.append(module)
        for bn in frozen_bns:
            bn.track_running_stats = False
            bn.running_mean = None
            bn.running_var = None
    else:
        raise ValueError(
            f"ECOTTA.FROZEN_BN_MODE must be one of ['eval', 'train', 'batch_stats'], "
            f"got '{frozen_bn_mode}'."
        )

    # The meta networks' BNs always use test-batch statistics -- this is EcoTTA's
    # normalization adaptation mechanism.
    for meta in ecotta_model.meta_modules():
        meta.meta_bn.train()
        meta.conv_bn.train()


@ADAPTATION_REGISTRY.register()
class EcoTTA(TTAMethod):
    """EcoTTA. Fully unsupervised: entropy minimization + self-distilled L1 regularization.

    Deliberately NOT added to utils/eval_utils._is_supervised_tta -- it never sees labels.
    """

    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        self.softmax_entropy = Entropy()
        # H_0 = E_MARGIN * ln(C). Scaled from num_classes, not hardcoded to ln(10), so a
        # CIFAR-100 port stays correct (the reference hardcodes math.log(10), main.py:44).
        self.e_margin = cfg.ECOTTA.E_MARGIN * math.log(self.num_classes)
        self.lambda_reg = cfg.ECOTTA.LAMBDA_REG
        self.normalize_reg_by_k = cfg.ECOTTA.NORMALIZE_REG_BY_K
        self.detach_group_input = cfg.ECOTTA.DETACH_GROUP_INPUT
        self.update_rule = cfg.ECOTTA.UPDATE_RULE
        if self.update_rule not in ("algorithm1", "two_step", "single_forward"):
            raise ValueError(
                f"ECOTTA.UPDATE_RULE must be one of ['algorithm1', 'two_step', 'single_forward'], "
                f"got '{self.update_rule}'."
            )

        # Entropy-filter diagnostics (see entropy_loss / filter_stats). Without these, a run in
        # which the entropy term was inert for most batches is indistinguishable from a healthy one.
        self.num_empty_filter_batches = 0
        self.num_filter_batches = 0
        self.num_samples_seen = 0
        self.num_samples_kept = 0

        if self.steps > 1:
            # TTAMethod.forward loops `for _ in range(self.steps)` (base.py:117), so any
            # OPTIM.STEPS > 1 multiplies the optimizer steps per batch and contradicts
            # Algorithm 1's single step. Guard every update rule, not just two_step.
            raise ValueError(
                f"EcoTTA expects OPTIM.STEPS=1 (paper Algorithm 1 performs exactly one optimizer "
                f"step per batch) but OPTIM.STEPS={self.steps}. TTAMethod.forward would repeat "
                f"the whole update {self.steps} times."
            )
        if self.update_rule == "single_forward" and self.detach_group_input:
            logger.warning(
                "ECOTTA.UPDATE_RULE='single_forward' cannot honour ECOTTA.DETACH_GROUP_INPUT: "
                "with a single shared graph, dR_k also flows into upstream meta networks, so "
                "Algorithm 1's per-partition isolation is lost. Proceeding without detaching."
            )
        if (cfg.ECOTTA.BN_PLACEMENT == "before" and cfg.ECOTTA.USE_OWN_OPTIM
                and cfg.ECOTTA.MOMENTUM > 0 and cfg.ECOTTA.LR > 0):
            logger.warning(
                f"EcoTTA: BN_PLACEMENT='before' with ECOTTA.MOMENTUM={cfg.ECOTTA.MOMENTUM} is "
                f"MEASURED TO DIVERGE on the paper's continual protocol -- error collapses to "
                f"chance (~90%) by the 3rd corruption; 15-corruption mean 81-82% vs the paper's "
                f"16.8%. ECOTTA.MOMENTUM 0.0 is stable (29.57/25.59/38.80 on gaussian/shot/"
                f"impulse) but is a TUNED value supported by neither the paper nor the reference "
                f"(which uses 0.9 with the BN-AFTER placement). See the KNOWN DEFECT section of "
                f"methods/ecotta.py before reporting any number from this configuration."
            )
        logger.info(
            f"EcoTTA ready: K={cfg.ECOTTA.K}, lambda_reg={self.lambda_reg}, "
            f"H_0={self.e_margin:.4f} ({cfg.ECOTTA.E_MARGIN} * ln {self.num_classes}), "
            f"frozen_bn_mode={cfg.ECOTTA.FROZEN_BN_MODE}, update_rule={self.update_rule}, "
            f"detach_parts={self.detach_group_input}"
        )

    # -- setup ---------------------------------------------------------------
    def configure_model(self):
        """Insert the meta networks, freeze everything else, and load the warm-up checkpoint.

        Mutating/replacing `self.model` here is the sanctioned host pattern: methods/base.py
        takes `self.models = [self.model]` and the reset snapshot at base.py:64-65, i.e. AFTER
        configure_model() runs (cf. methods/norm.py:157, :219). Loading the warm-up checkpoint
        here -- rather than after super().__init__() -- is what makes reset() restore the
        WARMED-UP state; loading it later would let the first model.reset() in test_time.py:76
        silently discard the warm-up.
        """
        self.model = build_ecotta_model(self.cfg, self.model, self.device)

        # Freeze everything, then re-enable only the meta networks.
        self.model.requires_grad_(False)
        for meta in self.model.meta_modules():
            meta.requires_grad_(True)

        apply_bn_modes(self.model, self.cfg.ECOTTA.FROZEN_BN_MODE)

        self._load_warmup_ckpt()

    def _load_warmup_ckpt(self):
        cfg = self.cfg
        self.warmup_ckpt_path = None

        if not cfg.ECOTTA.LOAD_WARMUP:
            # Used by warmup_train_ecotta.py: a warm-up run must start from the random meta-net
            # init, never silently resume (and restart the lr schedule on top of) an existing
            # checkpoint that happens to sit at the auto-derived path.
            logger.info("ECOTTA.LOAD_WARMUP=False: meta networks left at their random "
                        "initialisation, no warm-up checkpoint loaded.")
            return

        path = cfg.ECOTTA.WARMUP_CKPT or default_warmup_path(cfg)

        if not os.path.isfile(path):
            msg = (
                f"EcoTTA warm-up checkpoint not found at:\n    {path}\n"
                f"EcoTTA's meta networks are randomly initialised and are SUMMED into the frozen "
                f"feature path, so without warm-up the network is near-random -- it will not "
                f"merely be 'worse', it will be meaningless. Produce one with:\n"
                f"    CUDA_VISIBLE_DEVICES=0 python warmup_train_ecotta.py --cfg cfgs/cifar10_c/ecotta.yaml\n"
                f"or set ECOTTA.REQUIRE_WARMUP False to deliberately cold-start "
                f"(consider ECOTTA.ZERO_INIT_CONV True for a meaningful cold-start ablation)."
            )
            if cfg.ECOTTA.REQUIRE_WARMUP:
                raise RuntimeError(msg)
            logger.warning(msg)
            return

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        for key, expected in (("arch", cfg.MODEL.ARCH),
                              ("dataset", cfg.CORRUPTION.DATASET),
                              ("K", cfg.ECOTTA.K),
                              ("num_classes", self.num_classes),
                              ("bn_placement", cfg.ECOTTA.BN_PLACEMENT)):
            if key in ckpt and ckpt[key] != expected:
                raise RuntimeError(
                    f"Warm-up checkpoint {path} was built with {key}={ckpt[key]!r} but this run "
                    f"uses {key}={expected!r}. Re-run the warm-up or point ECOTTA.WARMUP_CKPT "
                    f"at the matching file."
                )

        # A warm-up artifact is only meaningful for the exact topology it was trained against.
        # Refuse rather than load a stale checkpoint into a changed meta-network arrangement:
        # the tensor shapes can coincide (e.g. group 2 of WRN-28 has c_in == c_out == 160) while
        # the learned function is for a different architecture entirely.
        expected_sig = self.model.topology_signature()
        ckpt_sig = ckpt.get("topology_signature")
        if ckpt_sig is None:
            raise RuntimeError(
                f"Warm-up checkpoint {path} predates topology signatures and cannot be validated "
                f"against the current architecture. Re-run warmup_train_ecotta.py to regenerate "
                f"it.\n  expected: {expected_sig}"
            )
        if ckpt_sig != expected_sig:
            raise RuntimeError(
                f"Warm-up checkpoint {path} was trained for a DIFFERENT meta-network topology.\n"
                f"  checkpoint: {ckpt_sig}\n"
                f"  this run:   {expected_sig}\n"
                f"Re-run warmup_train_ecotta.py with the current ECOTTA settings."
            )

        # Only meta-network tensors are stored (~14 MB for K=5 vs. 583 MB for a full state dict),
        # hence strict=False -- with an explicit check that every meta tensor was in fact restored.
        incompatible = self.model.load_state_dict(ckpt["meta_state"], strict=False)
        missing_meta = [k for k in incompatible.missing_keys if ".meta." in k]
        if missing_meta or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Warm-up checkpoint {path} does not match the built model.\n"
                f"  missing meta keys: {missing_meta[:8]}\n"
                f"  unexpected keys:   {list(incompatible.unexpected_keys)[:8]}"
            )

        # If the warm-up ran with FROZEN BNs in train mode, the frozen running statistics drifted
        # away from the RobustBench values. They must be restored, otherwise the meta networks
        # were trained against normalization statistics that no longer exist at test time.
        if ckpt.get("frozen_bn_buffers"):
            self.model.load_state_dict(ckpt["frozen_bn_buffers"], strict=False)
            logger.info("Restored frozen BN buffers from the warm-up checkpoint "
                        "(warm-up ran with WARMUP_BN_MODE='train').")

        self.warmup_ckpt_path = path
        logger.info(
            f"Loaded EcoTTA warm-up checkpoint from {path} "
            f"(src_acc={ckpt.get('src_acc', 'n/a')}, epochs={ckpt.get('warmup_epochs', 'n/a')}, "
            f"lr={ckpt.get('warmup_lr', 'n/a')}, bs={ckpt.get('warmup_batch_size', 'n/a')}, "
            f"bn_mode={ckpt.get('warmup_bn_mode', 'n/a')})"
        )
        ckpt_epochs = ckpt.get("warmup_epochs")
        if ckpt_epochs is not None and ckpt_epochs != cfg.ECOTTA.WARMUP_EPOCHS:
            logger.warning(
                f"Warm-up checkpoint was trained for {ckpt_epochs} epoch(s) but "
                f"ECOTTA.WARMUP_EPOCHS is {cfg.ECOTTA.WARMUP_EPOCHS}. The checkpoint is used as "
                f"is -- re-run warmup_train_ecotta.py if you want the configured schedule "
                f"(the paper uses 10 epochs for CIFAR)."
            )

    def collect_params(self):
        """Collect the meta-network parameters only."""
        params, names = [], []
        for nm, module in self.model.named_modules():
            for np_, p in module.named_parameters(recurse=False):
                if ".meta." in f"{nm}.{np_}" and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np_}")

        num = sum(p.numel() for p in params)
        placement = self.cfg.ECOTTA.BN_PLACEMENT
        family = self.model.family
        expected = (EXPECTED_META_PARAMS.get(family, {})
                    .get(placement, {}).get(self.cfg.ECOTTA.K))
        # Only a guard at the family's default kernel size; a CONV_KERNEL_SIZE override changes
        # the count legitimately.
        if (expected is not None and num != expected
                and self.cfg.ECOTTA.CONV_KERNEL_SIZE == 0):
            raise RuntimeError(
                f"EcoTTA collected {num:,} trainable meta parameters for family={family!r}, "
                f"K={self.cfg.ECOTTA.K}, BN_PLACEMENT={placement!r} but expected {expected:,} "
                f"(see EXPECTED_META_PARAMS in methods/ecotta.py). The module tree was probably "
                f"not built as intended."
            )
        if not params:
            # base.py:52 would set self.optimizer=None and copy_model_and_optimizer would then
            # die with an opaque AttributeError. Fail with something readable instead.
            raise RuntimeError("EcoTTA found no trainable meta parameters.")
        return params, names

    def setup_optimizer(self):
        if not self.cfg.ECOTTA.USE_OWN_OPTIM:
            return super().setup_optimizer()
        # Paper: SGD, lr 5e-3, momentum 0.9.
        return torch.optim.SGD(
            self.params,
            lr=self.cfg.ECOTTA.LR,
            momentum=self.cfg.ECOTTA.MOMENTUM,
            weight_decay=self.cfg.ECOTTA.WD,
        )

    # -- losses --------------------------------------------------------------
    def entropy_loss(self, outputs):
        """Filtered entropy, normalised by the FULL batch size N (paper App. D Eq. (7)).

        Returns exactly 0 (differentiably, on the right device/dtype) when no sample passes the
        threshold. The reference computes `entropys[mask].mean(0)` (utils.py:25-26), which is NaN
        for an empty selection and would silently poison every meta parameter for the rest of
        the run while the accuracy log keeps printing chance level.
        """
        entropys = self.softmax_entropy(outputs)          # per-sample, unreduced
        keep = entropys < self.e_margin
        num_kept = int(keep.sum())

        self.num_filter_batches += 1
        self.num_samples_seen += outputs.shape[0]
        self.num_samples_kept += num_kept

        if num_kept == 0:
            self.num_empty_filter_batches += 1
            if self.num_empty_filter_batches in (1, 10, 100, 1000):
                logger.warning(
                    f"EcoTTA entropy filter rejected an ENTIRE batch "
                    f"(H_0={self.e_margin:.4f}); such batches adapt on the regularizer alone. "
                    f"Count so far: {self.num_empty_filter_batches}/{self.num_filter_batches} "
                    f"batches."
                )
            return outputs.sum() * 0.0, 0
        return entropys[keep].sum() / outputs.shape[0], num_kept

    def filter_stats(self):
        """Entropy-filter health. Reported so an inert-entropy run is distinguishable."""
        return {
            "batches": self.num_filter_batches,
            "empty_filter_batches": self.num_empty_filter_batches,
            "empty_filter_frac": (self.num_empty_filter_batches / self.num_filter_batches
                                  if self.num_filter_batches else 0.0),
            "samples_kept_frac": (self.num_samples_kept / self.num_samples_seen
                                  if self.num_samples_seen else 0.0),
        }

    def regularization_loss(self):
        reg_losses = [r for r in self.model.reg_losses() if r is not None]
        if not reg_losses:
            raise RuntimeError("EcoTTA regularization requested but no group recorded an R_k.")
        total = torch.stack(reg_losses).sum()
        if self.normalize_reg_by_k:
            total = total / len(reg_losses)
        return total

    def loss_calculation(self, x):
        """Single-graph combined objective (UPDATE_RULE='single_forward').

        Cannot honour Algorithm 1's selective detach -- see the module docstring.
        """
        imgs_test = x[0]
        self.model.set_compute_reg(True)
        self.model.set_detach(False)
        outputs = self.model(imgs_test)
        loss_ent, _ = self.entropy_loss(outputs)
        loss = loss_ent + self.lambda_reg * self.regularization_loss()
        return outputs, loss

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

        if self.update_rule == "algorithm1":
            outputs = self._adapt_algorithm1(x)
        elif self.update_rule == "two_step":
            outputs = self._adapt_two_step(x)
        else:
            outputs = self._adapt_single_forward(x)

        if timing_on:
            if measuring:
                if self.device == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                # Stored in SECONDS PER SAMPLE: test_time.py:146 multiplies by 1000 for the
                # reported milliseconds (cf. methods/norm.py:76).
                self._timing_results.append(elapsed / batch_size)
                # Declare the divisor so test_time.py can emit the canonical
                # "Adaptation latency: X ms/sample" line. We store PER-SAMPLE seconds, so one
                # frame is batch_size recorded units (cf. utils/timing.py, which sets the same).
                self._timing_units_per_frame = batch_size
            self._timing_samples_seen += batch_size

        return outputs

    def _entropy_pass(self, x):
        """Pass 1: no detach, so the entropy gradient reaches all K meta networks."""
        self.model.set_compute_reg(False)
        self.model.set_detach(False)
        outputs = self.model(x[0])
        loss_ent, _ = self.entropy_loss(outputs)
        return outputs, loss_ent

    def _reg_pass(self, x):
        """Pass 2: Detach_parts, so dR_k only touches theta_k.

        Also returns the logits. Detaching changes gradient flow, not forward values, so these
        are numerically the same predictions as the entropy pass produces on the same weights.
        """
        self.model.set_compute_reg(True)
        self.model.set_detach(self.detach_group_input)
        outputs = self.model(x[0])
        return outputs, self.lambda_reg * self.regularization_loss()

    def _adapt_algorithm1(self, x):
        """Paper Algorithm 1: two backward passes accumulating into ONE optimizer step.

        Note: with two forwards per batch the meta BNs' running statistics get their EMA applied
        twice on the same batch. That is inconsequential here (they are in train mode, so
        normalization always uses batch statistics and the running buffers are never read
        online), and the reference behaves identically -- but it matters if anyone ever
        evaluates this model with the meta BNs switched to eval mode.
        """
        if self.mixed_precision and self.device == "cuda":
            with torch.cuda.amp.autocast():
                outputs, loss_ent = self._entropy_pass(x)
            self.scaler.scale(loss_ent).backward()
            with torch.cuda.amp.autocast():
                _, loss_reg = self._reg_pass(x)
            self.scaler.scale(loss_reg).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        else:
            outputs, loss_ent = self._entropy_pass(x)
            loss_ent.backward()
            _, loss_reg = self._reg_pass(x)
            loss_reg.backward()
            # Single step on the accumulated gradients (Algorithm 1).
            self.optimizer.step()
            self.optimizer.zero_grad()
        # Pre-update logits, per the host convention (methods/tent.py:43-47).
        return outputs

    def _adapt_single_forward(self, x):
        """Cheap approximation: one forward, one combined loss, one optimizer step.

        Deviation: Algorithm 1's per-partition gradient isolation is lost.
        """
        if self.mixed_precision and self.device == "cuda":
            with torch.cuda.amp.autocast():
                outputs, loss = self.loss_calculation(x)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        else:
            outputs, loss = self.loss_calculation(x)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        # Pre-update logits, per the host convention (methods/tent.py:43-47).
        return outputs

    def _adapt_two_step(self, x):
        """Reference schedule: step on lambda * sum_k R_k, THEN a second step on the entropy loss.

        Reproduces the reference's optimizer dynamics (main.py:59-87). Two differences from the
        reference are deliberate: the reported logits come from the pre-update regularization
        forward rather than the post-update entropy forward (main.py:88 inflates accuracy that
        way), and the entropy pass runs without detaching so its gradient reaches every meta
        network -- which is what the reference does too, since it only detaches when
        `cal_mseloss` is set.
        """
        if self.mixed_precision and self.device == "cuda":
            with torch.cuda.amp.autocast():
                outputs, loss_reg = self._reg_pass(x)
            self.scaler.scale(loss_reg).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                _, loss_ent = self._entropy_pass(x)
            self.scaler.scale(loss_ent).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        else:
            outputs, loss_reg = self._reg_pass(x)
            loss_reg.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            _, loss_ent = self._entropy_pass(x)
            loss_ent.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return outputs

    @torch.no_grad()
    def forward_sliding_window(self, x):
        # Single-sample path (base.py:139-147): never leave reg computation on for a
        # prediction-only forward.
        self.model.set_compute_reg(False)
        return self.model(x[0])


@torch.no_grad()
def calibrate_identity_init(net, loader, device, num_batches: int = 50):
    """Initialise the meta networks so the whole network starts AT the source function.

    Rationale (measured, see the warm-up section of this docstring block below):
    warm-up from random init converges to 97.3% train / 76.4% test clean accuracy on CIFAR-100
    while the source ResNeXt scores 78.89% -- i.e. it OVERFITS away from a source function that
    the architecture can represent exactly (proved in tests/test_ecotta_cifar100.py: a neutralised
    EcoTTANet reproduces the source model at max|diff| = 0). Starting warm-up at the identity
    removes that failure mode: cross-entropy then refines a network that already computes the
    source function instead of searching for it from scratch.

    Construction, for each group k:
      * bypass zeroed  -> conv.weight = 0, conv_bn.{weight,bias} = 0, so ReLU(BN(conv(h))) == 0.
      * meta_bn made an approximate identity -> with per-channel source statistics (mu, var) of
        that group's INPUT, set running_mean = mu, running_var = var, weight = sqrt(var + eps),
        bias = mu. In train mode BN computes weight*(x - mu_B)/sqrt(var_B + eps) + bias, which for
        mu_B ~ mu and var_B ~ var returns x; in eval mode it is exact.

    Statistics are captured with the meta BNs bypassed, so they are the true source activations.
    NOTE this is an initialisation choice the paper does not specify (it says only that warm-up
    trains the meta networks on source data). It is not tuned against any test metric.
    """
    net.eval()
    net.set_compute_reg(False)

    # --- zero the bypass so the network is exactly the source function ---
    for meta in net.meta_modules():
        meta.conv.weight.zero_()
        meta.conv_bn.weight.zero_()
        meta.conv_bn.bias.zero_()
        meta.conv_bn.running_mean.zero_()
        meta.conv_bn.running_var.fill_(1.0)

    # --- capture each group's source input statistics ---
    sums, sqsums, counts = {}, {}, {}
    handles = []

    def make_hook(i):
        def hook(mod, inputs):
            x = inputs[0].detach().float()
            c = x.shape[1]
            xt = x.transpose(0, 1).reshape(c, -1)
            sums[i] = sums.get(i, 0) + xt.sum(1)
            sqsums[i] = sqsums.get(i, 0) + (xt * xt).sum(1)
            counts[i] = counts.get(i, 0) + xt.shape[1]
        return hook

    net.set_bypass_meta_bn(True)
    for i, g in enumerate(net.groups):
        handles.append(g.register_forward_pre_hook(make_hook(i)))
    seen = 0
    for batch in loader:
        net(batch[0].to(device))
        seen += 1
        if seen >= num_batches:
            break
    for h in handles:
        h.remove()
    net.set_bypass_meta_bn(False)

    # --- write the identity-preserving affine + running stats ---
    for i, g in enumerate(net.groups):
        n = counts[i]
        mu = sums[i] / n
        var = (sqsums[i] / n - mu * mu).clamp_min(0.0)
        bn = g.meta.meta_bn
        bn.running_mean.copy_(mu.to(bn.running_mean.dtype))
        bn.running_var.copy_(var.to(bn.running_var.dtype))
        bn.weight.copy_(torch.sqrt(var + bn.eps).to(bn.weight.dtype))
        bn.bias.copy_(mu.to(bn.bias.dtype))
    logger.info(f"calibrate_identity_init: calibrated {len(net.groups)} meta BNs from "
                f"{seen} source batches; bypass zeroed.")
    return net
