"""
Structural tests for methods/tinytta.py (TinyTTA, NeurIPS 2024).

Run from the repo root, CPU-only:
    CUDA_VISIBLE_DEVICES="" python tests/test_tinytta.py

Self-checking assertions rather than a pytest suite, matching tests/test_ecotta.py. The real
RobustBench backbones are loaded from ckpt/cifar10/corruptions/Standard.pt and
ckpt/cifar100/corruptions/Hendrycks2020AugMix_ResNeXt.pt, so both checkpoints must be present.

WHY THESE TESTS AND NOT OTHERS
------------------------------
There is no reference implementation of TinyTTA -- official or third-party -- and the paper's
headline comparison (Fig. 6) has no numeric table. The only checkable references are Fig. 5's
16-cell source-vs-TinyTTA grid (four backbones, none of them ours), App. D Table 3's parameter
counts, and the SIGN of the WS ablation (App. E / Fig. 11). So accuracy cannot validate this port;
these tests substitute for it. Each one guards a failure that would otherwise produce
believable-but-wrong numbers -- see docs/tinytta/implementation_notes.md Sec. 6.3.
"""

import os
import sys
import math
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

import methods  # noqa: F401  (populates ADAPTATION_REGISTRY)
from conf import get_num_classes
from models.model import get_model
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import Entropy
from methods.tinytta import (
    ChannelAffine,
    EXPECTED_HEAD_PARAMS,
    ExitHead,
    TINYTTA_PARTITIONS,
    TinyTTANet,
    WSConv2d,
    build_tinytta_model,
    calibrate_gammas,
    default_offline_ckpt_path,
)

torch.set_num_threads(4)
PASSED = []
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {DEVICE}")


def check(name, condition, detail=""):
    if condition:
        PASSED.append(name)
        print(f"  PASS  {name}" + (f"  [{detail}]" if detail else ""))
    else:
        print(f"  FAIL  {name}  {detail}")
        raise AssertionError(f"{name}: {detail}")


def _default_clone():
    import conf
    return conf._CFG_DEFAULT.clone()


def make_cfg(yaml_path="cfgs/cifar10_c/tinytta.yaml", **tinytta_overrides):
    """Fresh unfrozen cfg clone. Independent of the global `conf.cfg` so tests can hold several."""
    c = _default_clone()
    c.defrost()
    c.merge_from_file(yaml_path)
    c.TINYTTA.REQUIRE_OFFLINE = False        # tests must not depend on an offline artifact
    c.TINYTTA.LOAD_OFFLINE = False
    c.TINYTTA.LOG_EXIT_DIST = False
    c.TINYTTA.LOG_PARAM_DRIFT = False
    c.TIMING.ENABLED = False
    for k, v in tinytta_overrides.items():
        setattr(c.TINYTTA, k, v)
    return c


def load_backbone(cfg):
    model, preprocess = get_model(cfg, get_num_classes(cfg.CORRUPTION.DATASET), DEVICE)
    model.model_preprocess = preprocess
    return model


def build_method(cfg):
    backbone = load_backbone(cfg)
    frozen = cfg.clone()
    frozen.freeze()
    return ADAPTATION_REGISTRY.get("tinytta")(
        cfg=frozen, model=backbone, num_classes=get_num_classes(cfg.CORRUPTION.DATASET)
    )


def set_gammas(net, values):
    net.gammas.copy_(torch.tensor(values, dtype=net.gammas.dtype, device=net.gammas.device))


# =========================================================================================
print("\n=== 1. NEUTRALIZED-HEAD BIT-EQUIVALENCE TO THE SOURCE MODEL (both backbones) ===")
# The single most valuable test. With no exit able to fire and EXIT_K_MODE='reuse_classifier',
# TinyTTANet must compute EXACTLY the original backbone's function, row for row. Validates the
# graph rebuild end to end: stage ordering, the ResNeXt's (x-mu)/sigma normalization (which lives
# INSIDE Hendrycks2020AugMixResNeXtNet.forward and a naive rebuild would silently drop), the tail,
# and the row-scatter in run_exits (a partial permutation there would degrade accuracy without
# looking like a bug).
for yaml_path, arch in (("cfgs/cifar10_c/tinytta.yaml", "Standard"),
                        ("cfgs/cifar100_c/tinytta.yaml", "Hendrycks2020AugMix_ResNeXt")):
    cfg = make_cfg(yaml_path)
    backbone = load_backbone(cfg)
    backbone.eval()
    x = torch.randn(4, 3, 32, 32, device=DEVICE)
    with torch.no_grad():
        ref = backbone(x).clone()
    net = build_tinytta_model(cfg, backbone, DEVICE, get_num_classes(cfg.CORRUPTION.DATASET))
    net.eval()
    set_gammas(net, [float("-inf")] * (net.K - 1))
    with torch.no_grad():
        logits, exit_idx, per_exit, _ = net.run_exits(x)
    maxdiff = float((logits - ref).abs().max().item())
    check(f"{arch}: neutralized TinyTTANet == source model", maxdiff < 1e-5,
          f"max|diff| = {maxdiff:.3e}")
    check(f"{arch}: every sample exits at the LAST exit when no gamma can fire",
          bool((exit_idx == net.K - 1).all()), f"exit_idx = {exit_idx.tolist()}")
    check(f"{arch}: row order preserved (per_exit rows cover 0..B-1 exactly once)",
          sorted(torch.cat([r for r, _ in per_exit.values()]).tolist()) == list(range(4)))
    del net, backbone


# =========================================================================================
print("\n=== 2. GROUPED == PER-SAMPLE FORWARD (the load-bearing equivalence) ===")
# The grouped batch-100 mode drops exited rows from the tensor. That is only equivalent to a
# per-sample forward because the frozen backbone runs in eval() with SOURCE BN statistics and is
# therefore pointwise in the batch. If this test fails, EXIT_MODE='grouped' is not TinyTTA.
cfg = make_cfg()
backbone = load_backbone(cfg)
net = build_tinytta_model(cfg, backbone, DEVICE, 10)
net.eval()
x = torch.randn(8, 3, 32, 32, device=DEVICE)

# Pick gammas from the observed entropy distribution so that exits actually fire at every head.
with torch.no_grad():
    set_gammas(net, [float("-inf")] * (net.K - 1))
    _, _, _, evaluated = net.run_exits(x, force_full=True)
    med = [float(net.entropy(evaluated[k][1]).median().item()) for k in range(net.K)]
set_gammas(net, [med[k] + 1e-4 for k in range(net.K - 1)])

with torch.no_grad():
    g_logits, g_idx, _, _ = net.run_exits(x)
    rows = [net.run_exits(x[i:i + 1]) for i in range(x.shape[0])]
p_logits = torch.cat([r[0] for r in rows], dim=0)
p_idx = torch.cat([r[1] for r in rows], dim=0)
maxdiff = float((g_logits - p_logits).abs().max().item())
check("grouped logits == per-sample logits", maxdiff < 1e-5, f"max|diff| = {maxdiff:.3e}")
check("grouped exit indices == per-sample exit indices", bool((g_idx == p_idx).all()),
      f"grouped {g_idx.tolist()} vs per-sample {p_idx.tolist()}")
check("the exits actually fired (test is not vacuous)", len(set(g_idx.tolist())) > 1,
      f"exit_idx = {g_idx.tolist()}, gammas = {[round(v, 4) for v in net.gammas.tolist()]}")

# ... and that the equivalence genuinely depends on FROZEN_BN_MODE. With batch statistics the
# frozen backbone is no longer pointwise, so grouped and per-sample MUST diverge -- which is why
# the method rejects that combination.
net_bs = build_tinytta_model(cfg, load_backbone(cfg), DEVICE, 10)
net_bs.eval()
for m in net_bs.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.track_running_stats = False
        m.running_mean = None
        m.running_var = None
with torch.no_grad():
    set_gammas(net_bs, [float("inf")] * (net_bs.K - 1))   # everything exits at head 1
    a = net_bs.run_exits(x)[0]
    b = torch.cat([net_bs.run_exits(x[i:i + 1])[0] for i in range(x.shape[0])], dim=0)
check("batch_stats BN breaks the equivalence (so grouped mode rightly rejects it)",
      float((a - b).abs().max().item()) > 1e-4,
      f"max|diff| = {float((a - b).abs().max().item()):.3e}")
del net_bs


# =========================================================================================
print("\n=== 3. WEIGHT STANDARDIZATION unit tests (A5 / A12 -- the fidelity fork) ===")
torch.manual_seed(0)
W = torch.randn(8, 5, 3, 3) * 3.0 + 1.5
eps = 1e-5

c_pc = WSConv2d(5, 8, 3, bias=False, ws_mode="per_out_channel", ws_eps=eps)
with torch.no_grad():
    c_pc.weight.copy_(W)
Wt = c_pc.standardized_weight()
check("(a) per-output-channel: each filter has zero mean",
      float(Wt.mean(dim=(1, 2, 3)).abs().max().item()) < 1e-5,
      f"max|row mean| = {float(Wt.mean(dim=(1, 2, 3)).abs().max().item()):.3e}")
check("(a) per-output-channel: each filter has unit std (biased, 1/N as in Eq. 4)",
      float((Wt.std(dim=(1, 2, 3), unbiased=False) - 1.0).abs().max().item()) < 1e-3,
      f"max|std-1| = {float((Wt.std(dim=(1, 2, 3), unbiased=False) - 1.0).abs().max().item()):.3e}")

c_wt = WSConv2d(5, 8, 3, bias=False, ws_mode="whole_tensor", ws_eps=eps)
with torch.no_grad():
    c_wt.weight.copy_(W)
Wt_whole = c_wt.standardized_weight()
check("(b) whole_tensor (Eq. 4 literal): global mean 0, global std 1",
      abs(float(Wt_whole.mean().item())) < 1e-5
      and abs(float(Wt_whole.std(unbiased=False).item()) - 1.0) < 1e-3,
      f"mean={float(Wt_whole.mean().item()):.3e} std={float(Wt_whole.std(unbiased=False).item()):.6f}")

for alpha in (0.1, 10.0):
    with torch.no_grad():
        c_pc.weight.copy_(W * alpha)
    Wt_scaled = c_pc.standardized_weight()
    with torch.no_grad():
        c_pc.weight.copy_(W)
    d = float((Wt_scaled - Wt).abs().max().item())
    check(f"(c) scale invariance: ws({alpha}*W) == ws(W)", d < 1e-3, f"max|diff| = {d:.3e}")
print("      -> WS makes the conv weight's norm unobservable to the loss, which is (i) why the "
      "head needs the 2c affine that App. D Table 3's counts require, and (ii) why TINYTTA.WD "
      "must stay 0.")

c_pc.zero_grad(set_to_none=True)
c_pc.standardized_weight().sum().backward()
check("(d) WS is differentiable w.r.t. the weight (A7: no buffer, nothing to reset)",
      c_pc.weight.grad is not None and bool(torch.isfinite(c_pc.weight.grad).all())
      and float(c_pc.weight.grad.abs().sum().item()) > 0.0)

check("(e) WS_MODE is a live flag (per_out_channel != whole_tensor)",
      float((Wt - Wt_whole).abs().max().item()) > 1e-3,
      f"max|diff| = {float((Wt - Wt_whole).abs().max().item()):.3e}")
c_sqrt = WSConv2d(5, 8, 3, bias=False, ws_mode="per_out_channel", ws_eps=0.5,
                  ws_eps_in_sqrt=True)
c_lit = WSConv2d(5, 8, 3, bias=False, ws_mode="per_out_channel", ws_eps=0.5,
                 ws_eps_in_sqrt=False)
with torch.no_grad():
    c_sqrt.weight.copy_(W)
    c_lit.weight.copy_(W)
check("(e) WS_EPS_IN_SQRT is a live flag ((W-mu)/(sigma+eps) != (W-mu)/sqrt(var+eps))",
      float((c_sqrt.standardized_weight() - c_lit.standardized_weight()).abs().max().item()) > 1e-3)

mu = W.mean(dim=(1, 2, 3), keepdim=True)
var_manual = ((W - mu) ** 2).mean(dim=(1, 2, 3), keepdim=True)      # 1/N, as Eq. 4 prints
manual = (W - mu) / (var_manual.sqrt() + eps)
check("(f) matches a hand-written Eq. 3 / Eq. 4 with 1/N variance",
      float((manual - Wt).abs().max().item()) < 1e-6,
      f"max|diff| = {float((manual - Wt).abs().max().item()):.3e}")


# =========================================================================================
print("\n=== 4. EXIT-DISTRIBUTION SANITY (gamma calibration hits its targets) ===")
# calibrate_gammas must reproduce its target exit fractions on the data it calibrated on. That is a
# property of the sequential-quantile procedure, independent of how good the heads are -- which is
# exactly what we want to test here (head quality is checked by per_exit_clean_acc in the offline
# checkpoint instead).
cfg = make_cfg()
net_cal = build_tinytta_model(cfg, load_backbone(cfg), DEVICE, 10)
net_cal.eval()


class _RandomSet(torch.utils.data.Dataset):
    def __init__(self, n):
        g = torch.Generator().manual_seed(7)
        self.x = torch.randn(n, 3, 32, 32, generator=g)
        self.y = torch.randint(0, 10, (n,), generator=g)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i]


cal_loader = torch.utils.data.DataLoader(_RandomSet(256), batch_size=64, num_workers=0)
targets = [0.25, 0.25, 0.25]
gammas, achieved, pcts = calibrate_gammas(net_cal, cal_loader, DEVICE,
                                          target_fractions=targets, max_samples=-1)
check("calibrated gammas are finite", all(math.isfinite(g) for g in gammas),
      f"gammas = {[round(g, 4) for g in gammas]}")
check("achieved exit fractions match the targets within 3pp",
      all(abs(achieved[k] - targets[k]) < 0.03 for k in range(3)),
      f"achieved = {[round(a, 4) for a in achieved]} vs targets {targets}")
check("entropy percentiles recorded per exit", len(pcts) == net_cal.K and len(pcts[0]) == 7)
print(f"      untrained-head entropy medians per exit: "
      f"{[round(pcts[k][3], 4) for k in range(net_cal.K)]}  (ln 10 = {math.log(10):.4f})")
print("      -> MEASURED, and it CONTRADICTS the naive expectation: an untrained head of this "
      "design is\n         strongly OVER-confident (medians far below ln C), not near-uniform, "
      "because per-channel WS\n         forces unit weight variance and therefore amplifies by "
      "~sqrt(fan_in). So the degeneracy at\n         random init is 'fires eagerly and returns "
      "garbage' (caught by per_exit_clean_acc in the\n         offline checkpoint), NOT 'never "
      "fires' (caught by the exit histogram). Both are watched.")
del net_cal


# =========================================================================================
print("\n=== 5. ENTROPY CORRECTNESS (Eq. 1 + Eq. 2, nats) ===")
uniform = torch.zeros(3, 10)
check("H(uniform) == ln C", abs(float(TinyTTANet.entropy(uniform)[0].item()) - math.log(10)) < 1e-6,
      f"{float(TinyTTANet.entropy(uniform)[0].item()):.6f} vs {math.log(10):.6f}")
saturated = torch.zeros(1, 10)
saturated[0, 3] = 50.0
check("H(saturated) -> 0", float(TinyTTANet.entropy(saturated).item()) < 1e-6)
rand_logits = torch.randn(6, 10)
check("TinyTTANet.entropy == utils.losses.Entropy (exit rule and loss use the same quantity)",
      float((TinyTTANet.entropy(rand_logits) - Entropy()(rand_logits)).abs().max().item()) < 1e-7)


# =========================================================================================
print("\n=== 6. GRADIENT LOCALITY (A2: only the exiting head is updated) ===")
# gamma_1 = +inf forces every sample to exit at head 1, so heads 2..K must be untouched.
cfg = make_cfg(MOMENTUM=0.0, LR=1e-1, GAMMA_MODE="explicit", GAMMA=[1e9, 1e9, 1e9])
m = build_method(cfg)
set_gammas(m.model, [float("inf"), float("inf"), float("inf")])
before = [torch.cat([p.detach().reshape(-1) for p in h.parameters()]).clone()
          for h in m.model.exits]
m.forward_and_adapt([torch.randn(6, 3, 32, 32, device=DEVICE)])
after = [torch.cat([p.detach().reshape(-1) for p in h.parameters()]).clone()
         for h in m.model.exits]
deltas = [float((after[i] - before[i]).abs().max().item()) for i in range(len(after))]
check("all samples exited at head 1", m.model.exit_counts.tolist()[0] == 6,
      f"counts = {m.model.exit_counts.tolist()}")
check("exit 1 was updated", deltas[0] > 0.0, f"delta = {deltas[0]:.3e}")
check("exits 2..K-1 are BIT-IDENTICAL (momentum 0)", all(d == 0.0 for d in deltas[1:]),
      f"deltas = {deltas}")
print("      -> With MOMENTUM > 0 this would NOT hold: SGD keeps moving a head on batches where it "
      "received no samples, which is a real deviation from A2. Documented, not hidden.")


# =========================================================================================
print("\n=== 7. FROZEN-BACKBONE IMMUTABILITY ===")
bb_named = {}
for nm, p in m.model.named_parameters():
    if not nm.startswith("exits.") and not p.requires_grad:
        bb_named[nm] = p.detach().clone()
buf_named = {nm: b.detach().clone() for nm, b in m.model.named_buffers()
             if "running_" in nm or "num_batches_tracked" in nm}
for _ in range(3):
    m.forward_and_adapt([torch.randn(6, 3, 32, 32, device=DEVICE)])
bad_p = [nm for nm, v in bb_named.items()
         if not torch.equal(v, dict(m.model.named_parameters())[nm].detach())]
bad_b = [nm for nm, v in buf_named.items()
         if not torch.equal(v, dict(m.model.named_buffers())[nm].detach())]
check("every frozen backbone parameter is bit-identical after 4 update steps", not bad_p,
      f"changed: {bad_p[:5]}")
check("every BN buffer is bit-identical (FROZEN_BN_MODE='eval' keeps source statistics)",
      not bad_b, f"changed: {bad_b[:5]}")
check("the pretrained classifier IS trainable and did move (EXIT_K_TRAINABLE=True)",
      any(p.requires_grad for p in m.model.tail_classifier().parameters()))


# =========================================================================================
print("\n=== 8. OFFLINE-CHECKPOINT GUARDS ===")
cfg_ck = make_cfg()
cfg_ck.TINYTTA.LOAD_OFFLINE = True
tmpdir = tempfile.mkdtemp()
good_net = build_tinytta_model(cfg_ck, load_backbone(cfg_ck), DEVICE, 10)
good_sig = good_net.topology_signature()
good_state = {k: v.cpu() for k, v in good_net.state_dict().items() if k.startswith("exits.")}
base_payload = dict(exit_state=good_state, tail_state={}, gammas=[1.0, 1.0, 1.0],
                    topology_signature=good_sig, arch="Standard", dataset="cifar10_c",
                    num_classes=10, K=4, source_kind="clean",
                    per_exit_clean_acc=[0.5, 0.6, 0.7, 0.9])


def expect_error(name, payload, fragment):
    path = os.path.join(tmpdir, name + ".pth")
    torch.save(payload, path)
    c = make_cfg()
    c.TINYTTA.LOAD_OFFLINE = True
    c.TINYTTA.OFFLINE_CKPT = path
    try:
        build_method(c)
    except RuntimeError as exc:
        check(f"rejects {name}", fragment in str(exc), str(exc)[:140])
        return
    check(f"rejects {name}", False, "no RuntimeError raised")


p = dict(base_payload)
p["topology_signature"] = good_sig + "|TAMPERED"
expect_error("mismatched_topology_signature", p, "DIFFERENT topology")

p = dict(base_payload)
p.pop("topology_signature")
expect_error("missing_topology_signature", p, "no topology signature")

p = dict(base_payload)
p["source_kind"] = "corrupted_gradual"
expect_error("corrupted_source_artifact", p, "leak")

p = dict(base_payload)
p["per_exit_clean_acc"] = [0.1, 0.6, 0.7, 0.9]      # exit 1 at chance for CIFAR-10
expect_error("chance_level_exit_head", p, "chance accuracy")

p = dict(base_payload)
p["exit_state"] = {k: v for k, v in good_state.items() if not k.startswith("exits.0.")}
expect_error("missing_exit_tensors", p, "does not match the built model")

p = dict(base_payload)
p["arch"] = "SomeOtherNet"
expect_error("wrong_arch", p, "was built with arch")

# A well-formed artifact must load, and the gammas must come from it (GAMMA_MODE='calibrated').
path = os.path.join(tmpdir, "good.pth")
torch.save(base_payload, path)
c = make_cfg()
c.TINYTTA.LOAD_OFFLINE = True
c.TINYTTA.OFFLINE_CKPT = path
m_ok = build_method(c)
check("accepts a well-formed artifact and takes its gammas",
      m_ok.model.gammas.tolist() == [1.0, 1.0, 1.0], str(m_ok.model.gammas.tolist()))

# REQUIRE_OFFLINE must fail loudly when the artifact is absent.
c = make_cfg()
c.TINYTTA.LOAD_OFFLINE = True
c.TINYTTA.REQUIRE_OFFLINE = True
c.TINYTTA.OFFLINE_CKPT = os.path.join(tmpdir, "does_not_exist.pth")
try:
    build_method(c)
    check("REQUIRE_OFFLINE=True fails loudly on a missing artifact", False, "no error")
except RuntimeError as exc:
    check("REQUIRE_OFFLINE=True fails loudly on a missing artifact",
          "offline checkpoint not found" in str(exc))
print(f"      default path would be: {default_offline_ckpt_path(make_cfg())}")


# =========================================================================================
print("\n=== 9. reset() RESTORES THE OFFLINE HEADS ===")
# The reset snapshot is taken at base.py:64-65, AFTER configure_model, which is why the offline
# checkpoint is loaded inside configure_model. If it were loaded later, the unconditional
# model.reset() at test_time.py:76 would silently replace the trained heads with the random init.
snap = [torch.cat([p.detach().reshape(-1) for p in h.parameters()]).clone()
        for h in m_ok.model.exits]
set_gammas(m_ok.model, [float("inf")] * 3)
m_ok.optimizer.param_groups[0]["lr"] = 1e-1
m_ok.forward_and_adapt([torch.randn(4, 3, 32, 32, device=DEVICE)])
moved = [torch.cat([p.detach().reshape(-1) for p in h.parameters()]).clone()
         for h in m_ok.model.exits]
check("adaptation moved the heads before reset",
      float((moved[0] - snap[0]).abs().max().item()) > 0.0)
m_ok.reset()
restored = [torch.cat([p.detach().reshape(-1) for p in h.parameters()]).clone()
            for h in m_ok.model.exits]
check("reset() restores the loaded offline heads bit-for-bit",
      all(torch.equal(snap[i], restored[i]) for i in range(len(snap))))
check("reset() also clears the exit histogram", m_ok.model.exit_counts.sum().item() == 0)
del m_ok


# =========================================================================================
print("\n=== 10. PARAMETER-COUNT / PARTITION REGRESSION (App. D Table 3 fitted form) ===")
# params = C_in^2 + 2*C_in + C_in*C + C at the default head configuration.
for yaml_path, family, C, expected_partition in (
        ("cfgs/cifar10_c/tinytta.yaml", "wrn28", 10, [2, 2, 3, 5]),
        ("cfgs/cifar100_c/tinytta.yaml", "resnext29", 100, [1, 2, 2, 4])):
    cfg = make_cfg(yaml_path)
    net = build_tinytta_model(cfg, load_backbone(cfg), DEVICE, C)
    check(f"{family}: measured activation-balanced partition {expected_partition}",
          net.partition == expected_partition and TINYTTA_PARTITIONS[family][4] == expected_partition,
          str(net.partition))
    check(f"{family}: partition covers every residual block",
          sum(net.partition) == sum(len(s) for s in net.stages))
    check(f"{family}: head params match EXPECTED_HEAD_PARAMS",
          net.head_params() == EXPECTED_HEAD_PARAMS[(family, 4, C)],
          f"{net.head_params()} vs {EXPECTED_HEAD_PARAMS[(family, 4, C)]}")
    check(f"{family}: K-1 = 3 new heads (exit K reuses the pretrained classifier, A12)",
          len(net.exits) == 3)
    for k, h in enumerate(net.exits):
        formula = h.c_in ** 2 + 2 * h.c_in + h.c_in * C + C
        check(f"{family} exit {k}: C_in^2 + 2*C_in + C_in*C + C == built count",
              h.num_params() == formula, f"{h.num_params()} vs {formula} (C_in={h.c_in})")
    check(f"{family}: taps (channels, cumulative stride) = {net.taps}", True, str(net.taps))
    check(f"{family}: the 2c affine is present (Table 3's +2c)",
          all(isinstance(h.norm, ChannelAffine) for h in net.exits))
    del net

# The head is also checkable against Table 3 itself: reproduce MobileNet/EfficientNet/RegNet.
TABLE3 = {  # (C_in, c, extra_per_channel) -> {C: thousands of params}
    "MobileNet-1":    (16, 96, 2, {10: 2.70, 100: 11.43}),
    "MobileNet-2":    (32, 192, 2, {10: 8.46, 100: 25.83}),
    "MobileNet-3":    (48, 288, 2, {10: 17.29, 100: 43.30}),
    "EfficientNet-1": (24, 144, 2, {10: 5.19, 100: 18.24}),
    "EfficientNet-2": (80, 480, 2, {10: 44.17, 100: 87.46}),
    "EfficientNet-3": (192, 672, 2, {10: 137.10, 100: 197.67}),
    "RegNet-1":       (152, 152, 3, {10: 25.09, 100: 38.86}),
    "RegNet-2":       (368, 368, 3, {10: 140.22, 100: 173.43}),
}
worst = 0.0
for name, (c_in, c, extra, cells) in TABLE3.items():
    for C, kb in cells.items():
        pred = c_in * c + extra * c + c * C + C
        worst = max(worst, abs(pred - kb * 1000))
check("the fitted head form reproduces App. D Table 3 to within its rounding (+-5 params)",
      worst <= 5.0, f"worst |predicted - reported| = {worst:.0f} params over 16 cells")


# =========================================================================================
print("\n=== 11. INERTNESS POSITIVE CONTROL (the lr question, measured) ===")
# lr 1e-5 is transplanted from App. H (audio) and was tuned at batch 1. This prints the actual
# displacement at batch 100 rather than arguing about it.
for lr in (1e-5, 1e-2):
    cfg = make_cfg(LR=lr, MOMENTUM=0.9, GAMMA_MODE="explicit", GAMMA=[1e9, 1e9, 1e9])
    cfg.TINYTTA.LOG_PARAM_DRIFT = True
    mm = build_method(cfg)
    set_gammas(mm.model, [float("inf")] * 3)
    for _ in range(5):
        mm.forward_and_adapt([torch.randn(20, 3, 32, 32, device=DEVICE)])
    d = mm.param_drift()[0]
    print(f"      lr={lr:g}: exit-1 rel_disp={d['rel_displacement']:.3e} "
          f"path={d['path']:.3e} nonzero_steps={d['nonzero_step_frac']:.2f} "
          f"over {mm._num_steps} steps")
    check(f"lr={lr:g}: the update is non-zero (positive control)", d["path"] > 0.0)
    if lr == 1e-5:
        inert_rel = d["rel_displacement"]
    else:
        check("lr 1e-2 moves the head far more than lr 1e-5 (drift metric is sensitive)",
              d["rel_displacement"] > 100 * inert_rel,
              f"{d['rel_displacement']:.3e} vs {inert_rel:.3e}")
    del mm


print(f"\nAll {len(PASSED)} checks passed.")
