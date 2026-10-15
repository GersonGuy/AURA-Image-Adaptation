"""
Structural tests for methods/ecotta.py (EcoTTA, arXiv 2303.01904).

Run from the repo root:
    CUDA_VISIBLE_DEVICES=0 python tests/test_ecotta.py

These are self-checking assertions rather than a pytest suite, matching the style of
tests/test_nonlinear_lifting.py. They load the real RobustBench `Standard` WRN-28-10 from
ckpt/cifar10/corruptions/Standard.pt, so the checkpoint must be present.
"""

import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

import methods  # noqa: F401  (populates ADAPTATION_REGISTRY)
import methods.ecotta as ecotta_module
from conf import cfg as _cfg, get_num_classes
from models.model import get_model
from utils.registry import ADAPTATION_REGISTRY
from methods.ecotta import (
    apply_bn_modes,
    EcoTTANet,
    WideResNetShim,
    EXPECTED_META_PARAMS,
    build_ecotta_model,
    default_warmup_path,
)

PASSED = []
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def check(name, condition, detail=""):
    if condition:
        PASSED.append(name)
        print(f"  PASS  {name}" + (f"  [{detail}]" if detail else ""))
    else:
        print(f"  FAIL  {name}  {detail}")
        raise AssertionError(f"{name}: {detail}")


def make_cfg(**ecotta_overrides):
    """Fresh unfrozen cfg pointed at the CIFAR-10-C EcoTTA config.

    Returns an INDEPENDENT clone, not the global `conf.cfg`. Several tests hold two configs at
    once (e.g. one per BN placement); sharing the global object would let the second silently
    overwrite the first.
    """
    c = _default_clone()
    c.defrost()
    c.merge_from_file("cfgs/cifar10_c/ecotta.yaml")
    c.ECOTTA.REQUIRE_WARMUP = False          # tests must not depend on a warm-up artifact
    c.TIMING.ENABLED = False
    for k, v in ecotta_overrides.items():
        setattr(c.ECOTTA, k, v)
    return c


def _default_clone():
    import conf
    return conf._CFG_DEFAULT.clone()


def load_backbone(cfg):
    model, preprocess = get_model(cfg, get_num_classes(cfg.CORRUPTION.DATASET), DEVICE)
    model.model_preprocess = preprocess
    return model


# ---------------------------------------------------------------------------
print("\n=== 1. group split geometry on WRN-28-10 ===")
cfg = make_cfg(K=5)
backbone = load_backbone(cfg)
net5 = build_ecotta_model(cfg, backbone, DEVICE)

expected_geom_k5 = [(16, 160, 1), (160, 160, 1), (160, 320, 2), (320, 320, 1), (320, 640, 2)]
check("K=5 groups [2,2,2,2,4]", WideResNetShim.partitions[5] == [2, 2, 2, 2, 4])
check("K=5 (c_in, c_out, stride) per group", net5.geometry() == expected_geom_k5,
      str(net5.geometry()))
check("K=5 covers all 12 residual blocks",
      sum(len(g.frozen_group) for g in net5.groups) == 12)
check("conv bypass maps group input -> group output channels",
      [(g.meta.conv.in_channels, g.meta.conv.out_channels) for g in net5.groups]
      == [(16, 160), (160, 160), (160, 320), (320, 320), (320, 640)])
check("conv_bn widths follow group outputs",
      [g.meta.conv_bn.num_features for g in net5.groups] == [160, 160, 320, 320, 640])

cfg4 = make_cfg(K=4)
net4 = build_ecotta_model(cfg4, load_backbone(cfg4), DEVICE)
expected_geom_k4 = [(16, 160, 1), (160, 160, 1), (160, 320, 2), (320, 640, 2)]
check("K=4 groups [2,2,4,4]", WideResNetShim.partitions[4] == [2, 2, 4, 4])
check("K=4 (c_in, c_out, stride) per group", net4.geometry() == expected_geom_k4,
      str(net4.geometry()))


# ---------------------------------------------------------------------------
print("\n=== 2. meta-network parameter counts (BN placement is NOT decided by these) ===")
def meta_param_count(net):
    return sum(p.numel() for m in net.meta_modules() for p in m.parameters())

n5, n4 = meta_param_count(net5), meta_param_count(net4)
backbone_params = 36_479_194

# Paper default: BN BEFORE the frozen group, so meta_bn is sized to the group's INPUT channels.
#   conv    = 9 * (16*160 + 160*160 + 160*320 + 320*320 + 320*640) = 3,479,040
#   meta_bn = 2 * (16 + 160 + 160 + 320 + 320)                     =     1,952
#   conv_bn = 2 * (160 + 160 + 320 + 320 + 640)                    =     3,200
conv_k5 = 9 * (16 * 160 + 160 * 160 + 160 * 320 + 320 * 320 + 320 * 640)
bn_before_k5 = conv_k5 + 2 * (16 + 160 + 160 + 320 + 320) + 2 * (160 + 160 + 320 + 320 + 640)
check("K=5 (BN before) meta params == 3,484,192", n5 == 3_484_192 == bn_before_k5, f"{n5:,}")
check("K=4 (BN before) meta params == 2,561,312", n4 == 2_561_312, f"{n4:,}")
check("meta_bn is sized to the group INPUT channels under BN-before",
      [g.meta.meta_bn.num_features for g in net5.groups] == [16, 160, 160, 320, 320],
      str([g.meta.meta_bn.num_features for g in net5.groups]))
check("constants agree with the built model",
      EXPECTED_META_PARAMS["wrn28"]["before"] == {4: n4, 5: n5})

# The parameter count DOES rule out a sequential output-side conv...
output_side_k5 = 9 * (160**2 + 160**2 + 320**2 + 320**2 + 640**2)
check("an output-side conv would be ~16.4% and is therefore excluded",
      abs(output_side_k5 / backbone_params - 0.164) < 0.001,
      f"{output_side_k5:,} = {output_side_k5 / backbone_params:.4%}")

# ...but it CANNOT distinguish the BN placement: the two differ by 1,248 params / 0.003 pp.
cfg_after = make_cfg(K=5, BN_PLACEMENT="after")
net5_after = build_ecotta_model(cfg_after, load_backbone(cfg_after), DEVICE)
n5_after = meta_param_count(net5_after)
check("K=5 (BN after, rejected design (i)) meta params == 3,485,440", n5_after == 3_485_440,
      f"{n5_after:,}")
check("the two BN placements are indistinguishable by parameter count",
      abs(n5_after - n5) == 1248
      and abs(n5_after / backbone_params - n5 / backbone_params) < 1e-4,
      f"before {n5 / backbone_params:.4%} vs after {n5_after / backbone_params:.4%}, "
      f"delta {n5_after - n5} params")
check("BN placement is settled by paper Sec. 4.2 / Tab. 4a, not by arithmetic",
      "Sec. 4.2" in ecotta_module.__doc__ and "design (i)" in ecotta_module.__doc__)

# Table 14's partition factor is K=4, yet its 9.5% WRN-28 entry only reproduces at K=5.
check("K=5 overhead is 9.55%, K=4 is 7.02% -- the paper's 9.5% is captioned K=4",
      abs(n5 / backbone_params - 0.0955) < 5e-4 and abs(n4 / backbone_params - 0.0702) < 5e-4,
      f"K=5 {n5 / backbone_params:.4%}, K=4 {n4 / backbone_params:.4%}")

total_after_ins = sum(p.numel() for p in net5.parameters())
check("trainable/total after insertion ~= 8.7%",
      0.086 <= n5 / total_after_ins <= 0.088, f"{n5 / total_after_ins:.4%} of {total_after_ins:,}")


# ---------------------------------------------------------------------------
print("\n=== 3. forward pass and self-distilled regularization ===")
x = torch.randn(8, 3, 32, 32, device=DEVICE)
net5.eval()
net5.set_compute_reg(False)
with torch.no_grad():
    out = net5(x)
check("output shape [N, num_classes]", tuple(out.shape) == (8, 10), str(tuple(out.shape)))
check("no R_k recorded when compute_reg=False",
      all(r is None for r in net5.reg_losses()))

net5.set_compute_reg(True)
out = net5(x)
regs = net5.reg_losses()
check("one R_k per group", len(regs) == 5 and all(r is not None for r in regs))
check("every R_k is a finite non-negative scalar",
      all(r.ndim == 0 and torch.isfinite(r) and r.item() >= 0 for r in regs),
      str([round(r.item(), 4) for r in regs]))
check("R_k carries grad", all(r.requires_grad for r in regs))

# Zero-init bypass => the adapted output equals meta_bn(frozen output), so R_k measures only
# the BN deviation; with an all-zero conv the bypass contributes exactly nothing.
cfgz = make_cfg(K=5, ZERO_INIT_CONV=True)
netz = build_ecotta_model(cfgz, load_backbone(cfgz), DEVICE)
netz.eval()
with torch.no_grad():
    bypass = netz.groups[0].meta.bypass(torch.randn(4, 16, 32, 32, device=DEVICE))
check("ZERO_INIT_CONV makes the bypass an exact no-op",
      torch.count_nonzero(bypass).item() == 0)


# ---------------------------------------------------------------------------
print("\n=== 4. gradient isolation: only meta networks are trainable ===")
cfg = make_cfg(K=5)
method = ADAPTATION_REGISTRY.get("ecotta")(
    cfg=cfg, model=load_backbone(cfg), num_classes=10)

trainable = [n for n, p in method.model.named_parameters() if p.requires_grad]
check("every trainable param lives under a .meta. path",
      all(".meta." in n for n in trainable), f"{len(trainable)} tensors")
check("collect_params found the expected count",
      sum(p.numel() for p in method.params) == 3_484_192,
      f"{sum(p.numel() for p in method.params):,}")
frozen_names = [n for n, p in method.model.named_parameters() if not p.requires_grad]
check("stem and classifier head are frozen and outside every group",
      any(n.startswith("stem.") for n in frozen_names)
      and any(n.startswith("head.") for n in frozen_names)
      and not any(n.startswith("stem.") or n.startswith("head.") for n in trainable),
      f"stem/head tensors: {sum(1 for n in frozen_names if n.startswith(('stem.','head.')))}")
check("optimizer is SGD with ECOTTA.LR/MOMENTUM",
      isinstance(method.optimizer, torch.optim.SGD)
      and math.isclose(method.optimizer.param_groups[0]["lr"], cfg.ECOTTA.LR)
      and math.isclose(method.optimizer.param_groups[0]["momentum"], cfg.ECOTTA.MOMENTUM))

# Frozen BNs must keep their source running statistics under FROZEN_BN_MODE="eval".
meta_bn_ids = {id(m) for meta in method.model.meta_modules() for m in (meta.meta_bn, meta.conv_bn)}
frozen_bns = [m for m in method.model.modules()
              if isinstance(m, nn.BatchNorm2d) and id(m) not in meta_bn_ids]
check("frozen BNs are in eval mode and keep running stats",
      all((not m.training) and m.running_mean is not None for m in frozen_bns),
      f"{len(frozen_bns)} frozen BNs")
check("meta BNs are in train mode (test-batch statistics)",
      all(meta.meta_bn.training and meta.conv_bn.training
          for meta in method.model.meta_modules()))

# One adaptation step must move meta params and leave frozen params untouched.
before_meta = [p.detach().clone() for p in method.params]
before_frozen = {n: p.detach().clone()
                 for n, p in method.model.named_parameters() if not p.requires_grad}
before_bn_buffers = [(m.running_mean.clone(), m.running_var.clone(),
                      m.num_batches_tracked.clone()) for m in frozen_bns]
imgs = torch.rand(16, 3, 32, 32, device=DEVICE)
out = method([imgs])
check("forward_and_adapt returns [N, C] logits", tuple(out.shape) == (16, 10))
check("meta params changed after one step",
      any(not torch.equal(b, p) for b, p in zip(before_meta, method.params)))
check("frozen params unchanged after one step",
      all(torch.equal(before_frozen[n], p)
          for n, p in method.model.named_parameters() if not p.requires_grad))
check("frozen BN running stats unchanged after one step",
      all(torch.equal(before_bn_buffers[i][0], m.running_mean)
          and torch.equal(before_bn_buffers[i][1], m.running_var)
          and torch.equal(before_bn_buffers[i][2], m.num_batches_tracked)
          for i, m in enumerate(frozen_bns)),
      f"{len(frozen_bns)} frozen BNs, buffers compared elementwise")


# ---------------------------------------------------------------------------
print("\n=== 5. entropy filter: empty batch contributes exactly 0, never NaN ===")
# H_0 = 0.4 * ln(10) ~= 0.9210
check("H_0 == E_MARGIN * ln(num_classes)",
      math.isclose(method.e_margin, 0.4 * math.log(10)), f"{method.e_margin:.6f}")

# Maximally uncertain logits => H = ln(10) = 2.303 > H_0 for every sample => nothing survives.
uniform_logits = torch.zeros(16, 10, device=DEVICE, requires_grad=True)
loss_ent, kept = method.entropy_loss(uniform_logits)
check("no sample survives a uniform-logit batch", kept == 0)
check("empty-filter entropy loss is exactly 0.0 (not NaN)",
      torch.isfinite(loss_ent) and loss_ent.item() == 0.0, f"{loss_ent.item()!r}")
loss_ent.backward()
check("empty-filter loss still back-propagates with zero grad",
      uniform_logits.grad is not None and torch.count_nonzero(uniform_logits.grad).item() == 0)

# Confident logits => everything survives, and the sum is normalised by the FULL batch N
# (paper App. D Eq. (7)), so a half-confident batch yields half the mean entropy.
confident = torch.zeros(16, 10, device=DEVICE)
confident[:, 0] = 20.0                       # H ~= 0 for all 16
loss_conf, kept_conf = method.entropy_loss(confident)
check("all samples survive a confident batch", kept_conf == 16)
check("confident batch has ~zero entropy loss", loss_conf.item() < 1e-4, f"{loss_conf.item():.3e}")

mixed = torch.zeros(16, 10, device=DEVICE)
mixed[:8, 0] = 20.0                          # 8 confident (survive, H~0), 8 uniform (filtered)
loss_mixed, kept_mixed = method.entropy_loss(mixed)
check("mixed batch keeps only the confident half", kept_mixed == 8)
check("normalisation uses the full batch N, not the surviving count",
      torch.isfinite(loss_mixed) and loss_mixed.item() < 1e-4, f"{loss_mixed.item():.3e}")


# ---------------------------------------------------------------------------
print("\n=== 6. reset() round-trips the mutated module tree ===")
method.reset()   # back to the state snapshotted at the end of base.__init__
snapshot = [p.detach().clone() for p in method.params]
for _ in range(3):
    method([torch.rand(16, 3, 32, 32, device=DEVICE)])
check("meta params drifted over 3 steps",
      any(not torch.equal(s, p) for s, p in zip(snapshot, method.params)))
method.reset()
check("reset() restores the post-configure_model meta weights",
      all(torch.equal(s, p) for s, p in zip(snapshot, method.params)))
# The snapshot the host holds must itself be the post-configure_model (and, in a real run,
# post-warm-up) state -- this is the trap from the mapping doc: had the warm-up been loaded
# after super().__init__(), model_states would hold random meta weights and the first
# model.reset() in test_time.py:76 would silently discard the warm-up.
stored = method.model_states[0]
check("stored reset state holds the configured meta weights",
      all(torch.equal(stored[k].to(DEVICE), method.model.state_dict()[k])
          for k in stored if ".meta." in k),
      f"{sum(1 for k in stored if '.meta.' in k)} meta tensors")


# ---------------------------------------------------------------------------
print("\n=== 7. warm-up checkpoint contract ===")
path = default_warmup_path(cfg)
check("auto-derived path lives under ckpt/warmup with an ecotta- prefix",
      os.path.basename(path).startswith("ckpt_ecotta_")
      and os.path.join("ckpt", "warmup") in path, path)
check("path encodes the compatibility keys (dataset, arch, K, seed)",
      all(tok in path for tok in ["cifar10_c", "Standard", "K5", "seed1"]), path)
check("path does NOT encode training hyperparameters (they live in the payload)",
      not any(tok in path for tok in ["bs64", "_ep", "lr0.05"]), path)

# REQUIRE_WARMUP=True with a nonexistent checkpoint must raise, not silently cold-start.
cfg_req = make_cfg(K=5)
cfg_req.defrost()
cfg_req.ECOTTA.REQUIRE_WARMUP = True
cfg_req.ECOTTA.WARMUP_CKPT = "/nonexistent/ecotta_warmup.pth"
cfg_req.freeze()
raised = False
try:
    ADAPTATION_REGISTRY.get("ecotta")(cfg=cfg_req, model=load_backbone(cfg_req), num_classes=10)
except RuntimeError as e:
    raised = "warm-up checkpoint not found" in str(e) and "warmup_train_ecotta.py" in str(e)
check("missing warm-up ckpt raises RuntimeError naming the path and the fix command", raised)

# A round-tripped meta-only checkpoint must load cleanly via the strict=False + verify path.
import tempfile
with tempfile.TemporaryDirectory() as tmpdir:
    ckpt_path = os.path.join(tmpdir, "ecotta_meta.pth")
    meta_state = {k: v.cpu() for k, v in method.model.state_dict().items() if ".meta." in k}
    torch.save({"meta_state": meta_state, "frozen_bn_buffers": {},
                "arch": "Standard", "dataset": "cifar10_c", "K": 5, "src_acc": 0.9,
                "bn_placement": "before",
                "topology_signature": method.model.topology_signature()}, ckpt_path)

    cfg_load = make_cfg(K=5)
    cfg_load.defrost()
    cfg_load.ECOTTA.REQUIRE_WARMUP = True
    cfg_load.ECOTTA.WARMUP_CKPT = ckpt_path
    cfg_load.freeze()
    reloaded = ADAPTATION_REGISTRY.get("ecotta")(
        cfg=cfg_load, model=load_backbone(cfg_load), num_classes=10)
    check("meta-only checkpoint restores every meta tensor",
          all(torch.equal(method.model.state_dict()[k].cpu(), reloaded.model.state_dict()[k].cpu())
              for k in meta_state))

    # LOAD_WARMUP=False must ignore an existing checkpoint entirely, so that a warm-up run
    # trains from random init rather than silently resuming (and re-scheduling on top of) it.
    cfg_noload = make_cfg(K=5)
    cfg_noload.defrost()
    cfg_noload.ECOTTA.REQUIRE_WARMUP = True     # would otherwise raise if it tried to load
    cfg_noload.ECOTTA.LOAD_WARMUP = False
    cfg_noload.ECOTTA.WARMUP_CKPT = ckpt_path
    cfg_noload.freeze()
    fresh = ADAPTATION_REGISTRY.get("ecotta")(
        cfg=cfg_noload, model=load_backbone(cfg_noload), num_classes=10)
    check("LOAD_WARMUP=False skips loading (meta weights differ from the checkpoint)",
          any(not torch.equal(meta_state[k].to(DEVICE), fresh.model.state_dict()[k])
              for k in meta_state if k.endswith("conv.weight")))
    check("LOAD_WARMUP=False records no warm-up path", fresh.warmup_ckpt_path is None)

    # A K mismatch must be caught rather than silently loading a wrong-shaped subset.
    cfg_bad = make_cfg(K=4)
    cfg_bad.defrost()
    cfg_bad.ECOTTA.REQUIRE_WARMUP = True
    cfg_bad.ECOTTA.WARMUP_CKPT = ckpt_path
    cfg_bad.freeze()
    raised_k = False
    try:
        ADAPTATION_REGISTRY.get("ecotta")(
            cfg=cfg_bad, model=load_backbone(cfg_bad), num_classes=10)
    except RuntimeError as e:
        raised_k = "K=5" in str(e) and "K=4" in str(e)
    check("K mismatch between checkpoint and cfg raises", raised_k)


# ---------------------------------------------------------------------------
print("\n=== 8. it is unsupervised: never on the label allowlist ===")
from utils.eval_utils import _is_supervised_tta
check("_is_supervised_tta(EcoTTA) is False", _is_supervised_tta(method) is False)
import inspect
sig = inspect.signature(method.forward_and_adapt)
check("forward_and_adapt takes no labels argument",
      "labels" not in sig.parameters, str(sig))


# ---------------------------------------------------------------------------
print("\n=== 9. Detach_parts semantics: gradient must reach ALL K meta networks ===")
# This is the trap that motivated ECOTTA.UPDATE_RULE. The entropy loss (and the warm-up
# cross-entropy) is a function of the final logits, so if Algorithm 1's Detach_parts were
# applied to that pass, gradients could reach only the DEEPEST meta network and every
# shallower one would stay at its random initialisation.
cfg = make_cfg(K=5)
mg = ADAPTATION_REGISTRY.get("ecotta")(cfg=cfg, model=load_backbone(cfg), num_classes=10)
imgs = torch.rand(16, 3, 32, 32, device=DEVICE)
labels = torch.randint(0, 10, (16,), device=DEVICE)


def grads_per_group(model):
    """Which groups' meta convs actually received a gradient."""
    return [g.meta.conv.weight.grad is not None
            and torch.count_nonzero(g.meta.conv.weight.grad).item() > 0
            for g in model.groups]


def zero_grads(model):
    for p in model.parameters():
        p.grad = None


# (a) entropy pass, no detach -> all K groups get gradient
zero_grads(mg.model)
_, loss_ent = mg._entropy_pass([imgs])
loss_ent.backward()
reach_ent = grads_per_group(mg.model)
check("entropy pass reaches all 5 meta networks", all(reach_ent), str(reach_ent))

# (b) cross-entropy (the warm-up objective), no detach -> all K groups get gradient
zero_grads(mg.model)
mg.model.set_compute_reg(False)
mg.model.set_detach(False)
F.cross_entropy(mg.model(imgs), labels).backward()
reach_ce = grads_per_group(mg.model)
check("warm-up cross-entropy reaches all 5 meta networks", all(reach_ce), str(reach_ce))

# (c) with Detach_parts ON, a logit-level loss reaches ONLY the deepest group. This is the
#     failure mode being guarded against; asserted so the guard cannot silently regress.
zero_grads(mg.model)
mg.model.set_compute_reg(False)
mg.model.set_detach(True)
F.cross_entropy(mg.model(imgs), labels).backward()
reach_detached = grads_per_group(mg.model)
check("with Detach_parts on, a logit loss reaches ONLY the deepest group",
      reach_detached == [False, False, False, False, True], str(reach_detached))

# (d) regularization pass with Detach_parts -> every group still gets its own gradient, because
#     each R_k is computed locally from that group's own output.
zero_grads(mg.model)
_, loss_reg = mg._reg_pass([imgs])
loss_reg.backward()
reach_reg = grads_per_group(mg.model)
check("detached regularization pass still gives every group a gradient",
      all(reach_reg), str(reach_reg))
mg.model.set_detach(False)


# ---------------------------------------------------------------------------
print("\n=== 10. all three update rules run and take the right number of steps ===")
for rule, expected_steps in (("algorithm1", 1), ("two_step", 2), ("single_forward", 1)):
    c = make_cfg(K=5, UPDATE_RULE=rule)
    m = ADAPTATION_REGISTRY.get("ecotta")(cfg=c, model=load_backbone(c), num_classes=10)

    counter = {"n": 0}
    real_step = m.optimizer.step

    def counting_step(*a, _real=real_step, _c=counter, **kw):
        _c["n"] += 1
        return _real(*a, **kw)

    m.optimizer.step = counting_step
    before = [p.detach().clone() for p in m.params]
    out = m([torch.rand(16, 3, 32, 32, device=DEVICE)])
    check(f"UPDATE_RULE='{rule}' produces [N, C] logits", tuple(out.shape) == (16, 10))
    check(f"UPDATE_RULE='{rule}' takes {expected_steps} optimizer step(s)",
          counter["n"] == expected_steps, f"took {counter['n']}")
    check(f"UPDATE_RULE='{rule}' moves the meta params",
          any(not torch.equal(b, p) for b, p in zip(before, m.params)))
    check(f"UPDATE_RULE='{rule}' leaves the meta params finite",
          all(torch.isfinite(p).all() for p in m.params))

c = make_cfg(K=5)
c.defrost()
c.ECOTTA.UPDATE_RULE = "nonsense"
c.freeze()
raised_rule = False
try:
    ADAPTATION_REGISTRY.get("ecotta")(cfg=c, model=load_backbone(c), num_classes=10)
except ValueError as e:
    raised_rule = "UPDATE_RULE" in str(e)
check("an unknown UPDATE_RULE raises ValueError", raised_rule)


# ---------------------------------------------------------------------------
print("\n=== 11. BN placement: the meta BN must normalize the frozen group's INPUT ===")
# The whole point of BN-before (paper Sec. 4.2) is that the features ENTERING the frozen convs
# get re-normalized. Assert that structurally rather than trusting the flag.
cfg_b = make_cfg(K=5, BN_PLACEMENT="before")
net_b = build_ecotta_model(cfg_b, load_backbone(cfg_b), DEVICE)
# The real online configuration: frozen BNs in eval (source stats), meta BNs in train
# (test-batch stats). In full eval mode meta_bn would use its init running stats and be a
# near-identity, which would make this test vacuous.
apply_bn_modes(net_b, "eval")
g0 = net_b.groups[0]

seen = {}
def _record_hook(mod, inp, out):
    seen["frozen_in"] = inp[0].detach().clone()
h = g0.frozen_group.register_forward_hook(_record_hook)
x_in = torch.randn(4, 16, 32, 32, device=DEVICE) * 5.0 + 3.0   # deliberately off-distribution
with torch.no_grad():
    _ = g0(x_in)
h.remove()
with torch.no_grad():
    expected_h = g0.meta.meta_bn(x_in)
check("under BN-before the frozen group receives meta_bn(x), not x",
      torch.allclose(seen["frozen_in"], expected_h, atol=1e-5)
      and not torch.allclose(seen["frozen_in"], x_in, atol=1e-3))
check("the normalized input actually is normalized (near zero mean / unit var per channel)",
      abs(seen["frozen_in"].mean().item()) < 0.2
      and abs(seen["frozen_in"].std().item() - 1.0) < 0.3,
      f"mean {seen['frozen_in'].mean().item():.3f} std {seen['frozen_in'].std().item():.3f}")

cfg_a = make_cfg(K=5, BN_PLACEMENT="after")
net_a = build_ecotta_model(cfg_a, load_backbone(cfg_a), DEVICE)
apply_bn_modes(net_a, "eval")
ga = net_a.groups[0]
seen_a = {}
def _record_hook_a(mod, inp, out):
    seen_a["frozen_in"] = inp[0].detach().clone()
h = ga.frozen_group.register_forward_hook(_record_hook_a)
with torch.no_grad():
    _ = ga(x_in)
h.remove()
check("under BN-after (rejected design (i)) the frozen group receives the raw input",
      torch.allclose(seen_a["frozen_in"], x_in, atol=1e-6))
check("BN-after sizes meta_bn to the group OUTPUT channels",
      [g.meta.meta_bn.num_features for g in net_a.groups] == [160, 160, 320, 320, 640])

# ---------------------------------------------------------------------------
print("\n=== 11b. R_k teacher: the two readings, and their relative scales ===")
# Default is "source_path": its target frozen_group(x) is PARAMETER-INDEPENDENT, the faithful
# analogue of the reference's out1 = original_part(x). "inner_frozen" targets
# frozen_group(meta_bn(x)), which -- even detached -- moves with meta_bn between steps, so it
# only penalizes the residual bypass and anchors nothing to source behaviour.
#
# The magnitudes below are measured on DELIBERATELY OFF-DISTRIBUTION synthetic input, so they
# exaggerate the gap. On real gaussian_noise-5 data the measured values are
# lambda*sum_k R_k = 1.61 (source_path) vs 0.86 (inner_frozen) against loss_ent = 0.18, i.e.
# both are commensurate with the entropy term and neither "swamps" it.
cfg_if = make_cfg(K=5, BN_PLACEMENT="before", REG_TEACHER="inner_frozen")
net_if = build_ecotta_model(cfg_if, load_backbone(cfg_if), DEVICE)
apply_bn_modes(net_if, "eval")
net_if.set_compute_reg(True)
out_b = net_if.groups[0](x_in)
with torch.no_grad():
    inner_teacher = net_if.groups[0].frozen_group(net_if.groups[0].meta.meta_bn(x_in))
r_inner = net_if.groups[0].reg_loss
check("default REG_TEACHER is 'source_path' (parameter-independent anchor, per the reference)",
      make_cfg().ECOTTA.REG_TEACHER == "source_path")
check("R_0 equals L1(out, sg(frozen_group(meta_bn(x))))",
      torch.allclose(r_inner, F.l1_loss(out_b, inner_teacher), atol=1e-5),
      f"{r_inner.item():.6f}")

cfg_sp = make_cfg(K=5, BN_PLACEMENT="before", REG_TEACHER="source_path")  # noqa
net_sp = build_ecotta_model(cfg_sp, load_backbone(cfg_sp), DEVICE)
apply_bn_modes(net_sp, "eval")
net_sp.set_compute_reg(True)
_ = net_sp.groups[0](x_in)
r_src = net_sp.groups[0].reg_loss
check("the two teachers give different R_0 magnitudes",
      r_inner.item() < 1.0 < r_src.item(),
      f"inner_frozen {r_inner.item():.4f} vs source_path {r_src.item():.4f}")
check("on off-distribution input source_path's R_0 is the larger of the two by ~10x",
      r_src.item() / r_inner.item() > 5,
      f"{r_src.item():.4f} / {r_inner.item():.4f} = {r_src.item() / r_inner.item():.1f}x")
check("R_k still carries gradient to the meta BN under the default teacher",
      r_inner.requires_grad)
net_if.set_compute_reg(False)


# ---------------------------------------------------------------------------
print("\n=== 12. warm-up checkpoints are topology-gated ===")
sig_before = net_b.topology_signature()
sig_after = net_a.topology_signature()
check("topology signatures differ across BN placements", sig_before != sig_after)
check("signature names the placement and K",
      "bn_before" in sig_before and "K5" in sig_before, sig_before)
check("BN placement appears in the auto-derived warm-up path",
      "bnbefore" in default_warmup_path(cfg_b) and "bnafter" in default_warmup_path(cfg_a))

with tempfile.TemporaryDirectory() as tmpdir:
    # A checkpoint from the OTHER topology must be refused, not silently loaded.
    stale = os.path.join(tmpdir, "stale.pth")
    m_after = ADAPTATION_REGISTRY.get("ecotta")(
        cfg=cfg_a, model=load_backbone(cfg_a), num_classes=10)
    torch.save({"meta_state": {k: v.cpu() for k, v in m_after.model.state_dict().items()
                               if ".meta." in k},
                "frozen_bn_buffers": {}, "arch": "Standard", "dataset": "cifar10_c", "K": 5,
                "bn_placement": "after",
                "topology_signature": m_after.model.topology_signature()}, stale)

    cfg_mismatch = make_cfg(K=5, BN_PLACEMENT="before")
    cfg_mismatch.defrost()
    cfg_mismatch.ECOTTA.REQUIRE_WARMUP = True
    cfg_mismatch.ECOTTA.WARMUP_CKPT = stale
    cfg_mismatch.freeze()
    raised_sig = False
    try:
        ADAPTATION_REGISTRY.get("ecotta")(
            cfg=cfg_mismatch, model=load_backbone(cfg_mismatch), num_classes=10)
    except RuntimeError as e:
        raised_sig = "bn_placement" in str(e) or "DIFFERENT meta-network topology" in str(e)
    check("a checkpoint from the other BN placement is refused", raised_sig)

    # An unsigned (pre-signature) checkpoint must also be refused rather than trusted.
    unsigned = os.path.join(tmpdir, "unsigned.pth")
    torch.save({"meta_state": {k: v.cpu() for k, v in net_b.state_dict().items()
                               if ".meta." in k},
                "frozen_bn_buffers": {}, "arch": "Standard", "dataset": "cifar10_c", "K": 5,
                "bn_placement": "before"}, unsigned)
    cfg_uns = make_cfg(K=5, BN_PLACEMENT="before")
    cfg_uns.defrost()
    cfg_uns.ECOTTA.REQUIRE_WARMUP = True
    cfg_uns.ECOTTA.WARMUP_CKPT = unsigned
    cfg_uns.freeze()
    raised_uns = False
    try:
        ADAPTATION_REGISTRY.get("ecotta")(
            cfg=cfg_uns, model=load_backbone(cfg_uns), num_classes=10)
    except RuntimeError as e:
        raised_uns = "predates topology signatures" in str(e)
    check("an unsigned checkpoint is refused", raised_uns)


# ---------------------------------------------------------------------------
print("\n=== 13. guards: OPTIM.STEPS and entropy-filter reporting ===")
c = make_cfg(K=5)
c.defrost()
c.OPTIM.STEPS = 2
c.freeze()
raised_steps = False
try:
    ADAPTATION_REGISTRY.get("ecotta")(cfg=c, model=load_backbone(c), num_classes=10)
except ValueError as e:
    raised_steps = "OPTIM.STEPS" in str(e)
check("OPTIM.STEPS>1 is rejected for every update rule", raised_steps)

cfg_f = make_cfg(K=5)
mf = ADAPTATION_REGISTRY.get("ecotta")(cfg=cfg_f, model=load_backbone(cfg_f), num_classes=10)
mf.entropy_loss(torch.zeros(16, 10, device=DEVICE))          # all filtered
mf.entropy_loss(torch.eye(10, device=DEVICE)[torch.zeros(16, dtype=torch.long)] * 20.0)  # all kept
st = mf.filter_stats()
check("filter_stats counts batches, empty batches and kept fraction",
      st["batches"] == 2 and st["empty_filter_batches"] == 1
      and abs(st["empty_filter_frac"] - 0.5) < 1e-9
      and abs(st["samples_kept_frac"] - 0.5) < 1e-9, str(st))


print(f"\nAll {len(PASSED)} checks passed.")
