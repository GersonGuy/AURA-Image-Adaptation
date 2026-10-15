"""
Structural tests for methods/elatta.py (ELaTTA, arXiv:2510.11068).

Run from the repo root, CPU-only:
    CUDA_VISIBLE_DEVICES="" python tests/test_elatta.py

Self-checking assertions rather than a pytest suite, matching tests/test_ecotta.py and
tests/test_tinytta.py. The RobustBench `Standard` WRN-28-10 is loaded from
ckpt/cifar10/corruptions/Standard.pt, so that checkpoint must be present. The offline basis is built
on the fly if it is not cached (~20 clean CIFAR-10 train images).

WHY THESE TESTS AND NOT OTHERS
------------------------------
ELaTTA has no released code, no third-party reimplementation, and -- crucially -- **no published
CIFAR number at all** (the paper never touches CIFAR). So there is nothing to compare an error rate
against, and accuracy cannot validate this port. These tests substitute for it by pinning the three
claims the port would otherwise be trusted on:

  1. THE IDENTITY. With the search neutralized (N_ITERS=0), ELaTTA must reproduce the frozen source
     model's logits BITWISE. Any accidental feature perturbation, wrong split, or leaked mutation of
     the backbone shows up here and nowhere else.
  2. THE BATCH-1 EQUIVALENCE (the port's central engineering claim, docs/elatta/implementation_notes.md
     Sec. "D2"). The paper is strictly batch-1; this repo runs batch 100. Batching is only exact if
     (a) the encoder and head are pointwise in the batch and (b) the B searches never couple. Both
     are measured here; (b) is also measured directly on the ES in tests/test_cma_es.py test 3.
  3. THE COST MODEL. One encoder pass and lambda*n head evaluations per sample -- the paper's entire
     efficiency claim (16.95 vs MEMO's 1096.14 GFLOPs). Counted, not asserted in prose.

Plus the basis invariants (orthonormality; the whitened/orthonormal reparameterization really is a
reparameterization) and the host-contract checks that a parameterless method breaks by default
(optimizer is None -> base.py's reset() would raise).
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import methods  # noqa: F401  (populates ADAPTATION_REGISTRY)
from conf import get_num_classes
from methods.cma_es import BatchedCMAES
from methods.elatta import default_basis_path
from models.model import get_model
from utils.registry import ADAPTATION_REGISTRY

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(f"{name}: {detail}")


def _default_clone():
    import conf
    return conf._CFG_DEFAULT.clone()


def make_cfg(yaml_path="cfgs/cifar10_c/elatta.yaml", **overrides):
    """Fresh unfrozen cfg clone, independent of the global `conf.cfg`."""
    c = _default_clone()
    c.defrost()
    c.merge_from_file(yaml_path)
    c.TIMING.ENABLED = False
    c.ELATTA.LOG_EVERY = 10 ** 9        # silence the periodic diagnostics inside tests
    for k, v in overrides.items():
        setattr(c.ELATTA, k, v)
    return c


def build_method(cfg):
    model, preprocess = get_model(cfg, get_num_classes(cfg.CORRUPTION.DATASET), DEVICE)
    model.model_preprocess = preprocess
    frozen = cfg.clone()
    frozen.freeze()
    return ADAPTATION_REGISTRY.get("elatta")(
        cfg=frozen, model=model, num_classes=get_num_classes(cfg.CORRUPTION.DATASET))


def fake_batch(b=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(b, 3, 32, 32, generator=g).to(DEVICE)


# =================================================================================================
print("\n=== 1. IDENTITY: N_ITERS=0 reproduces the frozen source model BITWISE ===")
# The hard identity test. With no generations there is nothing in the sampled set S, so the argmin
# is over {p = 0} and ELaTTA must be the source model exactly -- not approximately.
m0 = build_method(make_cfg(N_ITERS=0))
x = fake_batch(16)
with torch.no_grad():
    src = m0.model(x)
    out = m0.forward_and_adapt([x])
check("logits are bitwise identical to model(x)", bool(torch.equal(out, src)),
      f"max|delta| = {float((out - src).abs().max()):.3e}")
check("predictions identical", int((out.argmax(1) != src.argmax(1)).sum()) == 0)
check("head evals == B (one source forward, no candidates)", m0._head_evals == 16,
      str(m0._head_evals))
check("encoder forwards == 1", m0._encoder_forwards == 1, str(m0._encoder_forwards))

print("\n  ... and the split itself is exact: Cls(Enc(x)) == model(x)")
with torch.no_grad():
    spliced = m0.classifier(m0.encoder(x).flatten(1))
check("Cls(Enc(x)) == model(x) bitwise", bool(torch.equal(spliced, src)),
      f"max|delta| = {float((spliced - src).abs().max()):.3e}")


# =================================================================================================
print("\n=== 2. BATCH-1 EQUIVALENCE (the port's central claim) ===")
m = build_method(make_cfg())
x = fake_batch(16, seed=1)

print("  2a. the encoder is pointwise in the batch (frozen eval() BN, no cross-sample coupling)")
# The claim under test is the ABSENCE OF CROSS-SAMPLE COUPLING, which would show up at the scale of
# the feature itself. What actually limits the agreement is cuDNN picking a different convolution
# algorithm at batch 1 than at batch 16 -- and on Ampere+ that choice includes TF32 kernels
# (torch.backends.cudnn.allow_tf32 defaults to True), whose ~1e-3 relative precision puts the gap at
# ~1e-4 and blows any 1e-5 threshold for reasons that have nothing to do with ELaTTA. So measure
# BOTH: assert on the true-fp32 path, and report the as-configured number.
_tf32_was = torch.backends.cudnn.allow_tf32
_measured = {}
for _tf32 in (_tf32_was, False):
    torch.backends.cudnn.allow_tf32 = _tf32
    with torch.no_grad():
        z_batch = m.encoder(x).flatten(1)
        z_single = torch.cat([m.encoder(x[i:i + 1]).flatten(1) for i in range(x.shape[0])], dim=0)
    _measured[_tf32] = (float((z_batch - z_single).abs().max()), float(z_batch.abs().max()))
torch.backends.cudnn.allow_tf32 = _tf32_was
delta, scale = _measured[False]
print(f"      as configured (cudnn.allow_tf32={_tf32_was}): relative gap "
      f"{_measured[_tf32_was][0] / _measured[_tf32_was][1]:.2e} "
      f"-- reduced-precision conv kernels, not coupling")
check("features match a batch-1 loop to fp32 tolerance (TF32 off)", delta / scale < 1e-5,
      f"max|delta| = {delta:.3e} vs feature scale {scale:.3f} (relative {delta / scale:.2e})")

print("  2b. the classifier head is pointwise in the batch")
with torch.no_grad():
    l_batch = m.classifier(z_batch)
    l_single = torch.cat([m.classifier(z_batch[i:i + 1]) for i in range(z_batch.shape[0])], dim=0)
check("head logits match a per-row loop", float((l_batch - l_single).abs().max()) < 1e-5,
      f"max|delta| = {float((l_batch - l_single).abs().max()):.3e}")

print("  2c. BN is frozen: track_running_stats is untouched and running stats never move")
bns = [mod for mod in m.model.modules()
       if isinstance(mod, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))]
check("model has BatchNorm layers (otherwise the claim is vacuous)", len(bns) > 0, str(len(bns)))
check("all BN track_running_stats True (source stats retained)",
      all(b.track_running_stats for b in bns))
check("all BN in eval() mode", all(not b.training for b in bns))
before = [b.running_mean.clone() for b in bns]
with torch.no_grad():
    m.forward_and_adapt([x])
check("running_mean unchanged after adaptation",
      all(torch.equal(b.running_mean, t) for b, t in zip(bns, before)))

print("  2d. the B searches do not couple: per-row ES trajectories are independent")
# Drive a batched ES and a 1-element ES with the SAME populations and the SAME per-row fitness; the
# batched ES's row must equal the scalar ES exactly. (The complementary direction -- that mangling
# one row's fitness cannot move another row -- is tests/test_cma_es.py test 3.)
B, row = 8, 3
g = torch.Generator(device=DEVICE).manual_seed(0)
es_b = BatchedCMAES(B, m.k, m.sigma0, population=m.lam, device=DEVICE, generator=g,
                    constants=m.cma_constants)
es_1 = BatchedCMAES(1, m.k, m.sigma0, population=m.lam, device=DEVICE,
                    constants=m.cma_constants)
worst = 0.0
with torch.no_grad():
    z = m.encoder(x[:B]).flatten(1)
    for _ in range(m.n_iters):
        P = es_b.ask()
        Z = z.unsqueeze(1) + torch.matmul(P.to(z.dtype), m.V.T)
        lg = m.classifier(Z.reshape(-1, m.D)).reshape(B, m.lam, -1)
        fit = m._entropy(lg)
        es_b.tell(P, fit)
        es_1.tell(P[row:row + 1], fit[row:row + 1])
        worst = max(worst, float((es_b.mean[row] - es_1.mean[0]).abs().max()),
                    abs(float(es_b.sigma[row]) - float(es_1.sigma[0])),
                    float((es_b.C[row] - es_1.C[0]).abs().max()))
# Tolerance rather than exact equality: the batched einsums reduce in a different order than the
# 1-element ones, so float64 rounding shows up at ~1e-16. Coupling would show up at the scale of the
# state itself (O(1)), not at the last bit.
check("row-3 ES state matches a standalone 1-element ES to float64 rounding", worst < 1e-12,
      f"{worst:.3e}")

print("  2e. per-sample candidate fitness is unaffected by the other rows")
# The fitness of sample i's candidate j depends only on z_i, so evaluating a sub-batch must give
# identical fitness values for the rows it contains.
with torch.no_grad():
    P = torch.randn(B, m.lam, m.k, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(2)).to(DEVICE)
    Z_all = z.unsqueeze(1) + torch.matmul(P.to(z.dtype), m.V.T)
    fit_all = m._entropy(m.classifier(Z_all.reshape(-1, m.D)).reshape(B, m.lam, -1))
    Z_one = z[row:row + 1].unsqueeze(1) + torch.matmul(P[row:row + 1].to(z.dtype), m.V.T)
    fit_one = m._entropy(m.classifier(Z_one.reshape(-1, m.D)).reshape(1, m.lam, -1))
# Tolerance 1e-5, not 1e-6: `classifier` is an fp32 matmul whose reduction order depends on the
# batch shape it is handed ([B*lam, D] vs [lam, D]), and _entropy's softmax/log amplifies that.
# Measured residual is 2e-7..1.5e-6 depending on cuBLAS kernel choice and on whether V_k came from
# cache or was freshly rebuilt -- against entropy values of order 1e-1..2, i.e. 5+ orders of
# magnitude below anything meaningful. This asserts pointwise-ness, not bitwise equality; the
# bitwise claims live in 1a/2b/7a. Same class of too-tight threshold as 2a.
_subbatch = float((fit_all[row] - fit_one[0]).abs().max())
check("sub-batch fitness identical (pointwise, fp32 tol)", _subbatch < 1e-5, f"{_subbatch:.3e}")


# =================================================================================================
print("\n=== 3. COST MODEL: 1 encoder pass + lambda*n head evals per sample ===")
m3 = build_method(make_cfg())
with torch.no_grad():
    for i in range(3):
        m3.forward_and_adapt([fake_batch(10, seed=10 + i)])
check("encoder forwards == number of batches", m3._encoder_forwards == 3, str(m3._encoder_forwards))
check("encoder samples == 30", m3._encoder_samples == 30, str(m3._encoder_samples))
expected = 30 * (1 + m3.lam * m3.n_iters)
check(f"head evals == B*(1 + lambda*n) = {expected}", m3._head_evals == expected,
      str(m3._head_evals))
# Pin k/n explicitly: the shipped yaml carries the TUNED CIFAR-10 config (k=2, n=2 -> 12 evals),
# so reading them from it would make this check assert whatever the yaml happens to say. The claim
# under test is the PAPER's cost model, E = lambda*n ~ 96 at its k=16, n=8.
m3_paper = build_method(make_cfg(K=16, N_ITERS=8))
check("lambda*n == 96 at the paper's k=16, n=8", m3_paper.lam * m3_paper.n_iters == 96,
      f"{m3_paper.lam} * {m3_paper.n_iters}")
check("lambda*n == 12 at the shipped tuned k=2, n=2 (8x cheaper than the paper)",
      m3.lam * m3.n_iters == 12, f"{m3.lam} * {m3.n_iters}")


# =================================================================================================
print("\n=== 4. BASIS INVARIANTS ===")
payload = torch.load(default_basis_path(make_cfg()), map_location="cpu", weights_only=False)
V = payload["V_k"].double()
gram = V.T @ V
check("V_k has orthonormal columns (V^T V = I)",
      float((gram - torch.eye(V.shape[1], dtype=torch.float64)).abs().max()) < 1e-5,
      f"max|V^T V - I| = {float((gram - torch.eye(V.shape[1], dtype=torch.float64)).abs().max()):.2e}")
check("D == classifier in_features", payload["D"] == m.classifier.in_features,
      f"{payload['D']} vs {m.classifier.in_features}")
check("k == ELATTA.K", payload["k"] == m.k, f"{payload['k']} vs {m.k}")
check("source split is 'train' (leak-free: the test stream is the TEST split)",
      payload.get("source_split") == "train", str(payload.get("source_split")))
check("singular values are descending",
      bool((payload["singular_values"][:-1] >= payload["singular_values"][1:]).all()))

print("  4b. whitened vs orthonormal really is a REPARAMETERIZATION (same subspace)")
# Build BOTH arms explicitly with SIGMA0=-1.0 (auto). Do NOT reuse `m` here: `m` inherits whatever
# cfgs/cifar10_c/elatta.yaml ships, and that yaml carries the TUNED CIFAR-10 config (orthonormal
# basis, explicit SIGMA0) rather than the auto-derivation these four checks are about. Reusing it
# made these checks silently depend on the shipped defaults.
m_whit = build_method(make_cfg(BASIS_SCALING="whitened", SIGMA0=-1.0))
m_orth = build_method(make_cfg(BASIS_SCALING="orthonormal", SIGMA0=-1.0))
Vo, Vw = m_orth.V.double(), m_whit.V.double()
# Both must span the same k-dimensional subspace: the orthogonal projector onto span(V) is identical.
Po = Vo @ torch.linalg.pinv(Vo)
Pw = Vw @ torch.linalg.pinv(Vw)
check("projectors onto span(V) coincide", float((Po - Pw).abs().max()) < 1e-4,
      f"max|P_orth - P_whit| = {float((Po - Pw).abs().max()):.2e}")
ratio = (Vw.norm(dim=0) / Vo.norm(dim=0)).cpu()
check("whitened column norms == coeff_std",
      float((ratio - payload["coeff_std"].double()).abs().max()) < 1e-5,
      f"max|delta| = {float((ratio - payload['coeff_std'].double()).abs().max()):.2e}")
check("auto sigma0: whitened == 1.0", abs(m_whit.sigma0 - 1.0) < 1e-12, f"{m_whit.sigma0}")
check("auto sigma0: orthonormal == mean(S_k)/sqrt(N-1)",
      abs(m_orth.sigma0 - float(payload["coeff_std"].mean())) < 1e-5,
      f"{m_orth.sigma0} vs {float(payload['coeff_std'].mean())}")


# =================================================================================================
print("\n=== 5. HOST CONTRACT for a method with ZERO trainable parameters ===")
check("collect_params returns nothing", m.params == [], str(m.params))
check("no optimizer was built (base.py:52)", m.optimizer is None, str(m.optimizer))
check("no parameter requires grad",
      not any(p.requires_grad for p in m.model.parameters()))
try:
    m.reset()                      # base.py's reset() raises when optimizer_state is None
    ok, why = True, ""
except Exception as exc:           # noqa: BLE001
    ok, why = False, repr(exc)
check("reset() works despite optimizer=None (override present)", ok, why)
check("reset() clears p_prev", float(m.p_prev.abs().max()) == 0.0)

print("  5b. TEST.WINDOW_LENGTH > 1 and OPTIM.STEPS != 1 are rejected loudly")
for field, value in (("WINDOW_LENGTH", 4),):
    c = make_cfg()
    setattr(c.TEST, field, value)
    try:
        build_method(c)
        ok, why = False, "no exception"
    except ValueError as exc:
        ok, why = True, str(exc)[:60]
    check(f"TEST.{field}={value} rejected", ok, why)
c = make_cfg()
c.OPTIM.STEPS = 2
try:
    build_method(c)
    ok, why = False, "no exception"
except ValueError as exc:
    ok, why = True, str(exc)[:60]
check("OPTIM.STEPS=2 rejected", ok, why)


# =================================================================================================
print("\n=== 6. SEARCH BEHAVIOUR ===")
m6 = build_method(make_cfg())
x6 = fake_batch(16, seed=42)
with torch.no_grad():
    src6 = m6.classifier(m6.encoder(x6).flatten(1))
    out6 = m6.forward_and_adapt([x6])
h_src = m6._entropy(src6)
h_out = m6._entropy(out6)
check("adaptation lowers the mean entropy (it minimizes entropy, by construction)",
      float(h_out.mean()) <= float(h_src.mean()),
      f"{float(h_src.mean()):.4f} -> {float(h_out.mean()):.4f}")

print("  6b. INCLUDE_UNADAPTED=True makes the returned fitness <= the source's, per sample")
m6b = build_method(make_cfg(INCLUDE_UNADAPTED=True))
with torch.no_grad():
    src6b = m6b.classifier(m6b.encoder(x6).flatten(1))
    out6b = m6b.forward_and_adapt([x6])
viol = int((m6b._entropy(out6b) > m6b._entropy(src6b) + 1e-6).sum())
check("no sample ends with higher entropy than source", viol == 0, f"{viol} violations")

print("  6c. reproducibility: same seed -> identical predictions")
a = build_method(make_cfg())
b = build_method(make_cfg())
with torch.no_grad():
    oa = a.forward_and_adapt([x6])
    ob = b.forward_and_adapt([x6])
check("two methods with the same RNG_SEED agree bitwise", bool(torch.equal(oa, ob)),
      f"max|delta| = {float((oa - ob).abs().max()):.3e}")
c_seed = make_cfg()
c_seed.RNG_SEED = 7
d = build_method(c_seed)
with torch.no_grad():
    od = d.forward_and_adapt([x6])
check("a different RNG_SEED changes the search", not torch.equal(oa, od))

print("  6d. THE DEGENERACY WARNING (measured, see docs/elatta/implementation_notes.md A11): "
      "entropy in span(V_k) is trivially saturable")
with torch.no_grad():
    z6 = m6.encoder(x6).flatten(1)
    es = BatchedCMAES(z6.shape[0], m6.k, m6.sigma0, population=m6.lam, device=DEVICE,
                      generator=torch.Generator(device=DEVICE).manual_seed(0),
                      constants=m6.cma_constants)
    for gen in range(m6.n_iters):
        P = es.ask()
        Z = z6.unsqueeze(1) + torch.matmul(P.to(z6.dtype), m6.V.T)
        fit = m6._entropy(m6.classifier(Z.reshape(-1, m6.D)).reshape(z6.shape[0], m6.lam, -1))
        es.tell(P, fit)
    frac_zero = float((fit < 1e-6).float().mean())
    ties = float((fit <= fit.min(1, keepdim=True).values + 1e-9).float().sum(1).mean())
print(f"      final generation: {frac_zero * 100:.0f}% of candidates have entropy < 1e-6, "
      f"{ties:.1f} of {m6.lam} tie at the numerical minimum, ||mean|| = "
      f"{float(es.mean.norm(dim=1).mean()):.1f} (sigma0 = {m6.sigma0})")
check("this is REPORTED, not asserted -- see the docs (informational)", True)


# =================================================================================================
print("\n=== 7. ELATTA.NORM_MODE (the NON-PAPER BN-recalibration composition knob) ===")
# NORM_MODE="bn_test" is a deliberate, labelled departure from the paper ("the framework ... does not
# require updating normalization statistics"). Two things must be pinned:
#   (a) the DEFAULT path is bit-for-bit the old behaviour -- no existing number may move;
#   (b) the departure is EXACTLY "BN-Test backbone + ELaTTA search on top" and nothing else: with the
#       search neutralized (N_ITERS=0) it must reproduce this repo's `bntest` baseline, and it must
#       still be gradient-free, weight-immutable and stateless across batches.
print("  7a. the DEFAULT is 'frozen_source' and it changes nothing")
check("cfg default NORM_MODE == 'frozen_source'", _default_clone().ELATTA.NORM_MODE == "frozen_source",
      _default_clone().ELATTA.NORM_MODE)
check("cfgs/cifar10_c/elatta.yaml ships 'frozen_source'",
      make_cfg().ELATTA.NORM_MODE == "frozen_source", make_cfg().ELATTA.NORM_MODE)
m7 = build_method(make_cfg(NORM_MODE="frozen_source"))
bns7 = [b for b in m7.model.modules() if isinstance(b, torch.nn.modules.batchnorm._BatchNorm)]
check("frozen_source: every BN stays in eval() with track_running_stats=True",
      all((not b.training) and b.track_running_stats for b in bns7), f"{len(bns7)} BN layers")
x7 = fake_batch(16, seed=77)
with torch.no_grad():
    src7 = m7.model(x7)
    out7 = m7.forward_and_adapt([x7])
# The stronger claim -- identical to the code as it stood BEFORE the knob existed -- was verified
# once, out of band, by capturing this method's logits on fixed batches on both revisions
# (bitwise equal on three batch shapes; see docs/elatta/implementation_notes.md A16). What is
# asserted here is the hardware-portable part: the identity path is still bitwise the source model.
m7z = build_method(make_cfg(NORM_MODE="frozen_source", N_ITERS=0))
with torch.no_grad():
    check("frozen_source + N_ITERS=0 is still bitwise the source model",
          bool(torch.equal(m7z.forward_and_adapt([x7]), m7z.model(x7))))

print("  7b. bn_test switches to CURRENT-BATCH statistics and mutates NOTHING")
m7b = build_method(make_cfg(NORM_MODE="bn_test"))
bns7b = [b for b in m7b.model.modules() if isinstance(b, torch.nn.modules.batchnorm._BatchNorm)]
check("bn_test: every BN is in train() with track_running_stats=False (buffers read-only)",
      all(b.training and (not b.track_running_stats) for b in bns7b), f"{len(bns7b)} BN layers")
check("bn_test: no non-BN module was put in train() mode",
      not any(mod.training and not isinstance(mod, torch.nn.modules.batchnorm._BatchNorm)
              for mod in m7b.model.modules()))
_before = {k: v.clone() for k, v in m7b.model.state_dict().items()}
torch.set_grad_enabled(True)          # nothing may hide behind an outer no_grad
out7b = m7b.forward(x7)               # the public entry point, grad ENABLED
out7b2 = m7b.forward(fake_batch(16, seed=78))
_moved = [k for k, v in m7b.model.state_dict().items() if not torch.equal(v, _before[k])]
check("bn_test: not one state_dict tensor changed (weights AND all BN buffers)", _moved == [],
      f"{len(_moved)} changed: {_moved[:4]}")
check("bn_test: output carries no autograd graph even with grad enabled",
      (not out7b.requires_grad) and out7b.grad_fn is None)
check("bn_test: no parameter requires grad and no .grad was populated",
      not any(p.requires_grad or p.grad is not None for p in m7b.model.parameters()))
check("bn_test: the encoder still runs exactly once per batch",
      m7b._encoder_forwards == m7b._num_batches == 2, f"{m7b._encoder_forwards} vs {m7b._num_batches}")
_exp = 32 * (1 + m7b.lam * m7b.n_iters)
check(f"bn_test: head still receives exactly B*(1 + lambda*n) = {_exp} rows",
      m7b._head_evals == _exp, str(m7b._head_evals))
check("bn_test really changes the prediction vs frozen_source (the knob is not a no-op)",
      not torch.equal(out7b, out7),
      f"max|bn_test - frozen_source| = {float((out7b - out7).abs().max()):.3f}")

print("  7c. bn_test + N_ITERS=0 reproduces this repo's `bntest` baseline BITWISE")
# The key correctness check: it proves the composition is exactly "BN-Test backbone + the search",
# with nothing else changed. CAVEAT: methods/norm.py::BNTest has a repo-local subset path keyed on
# MIXED_GRADUAL.NUM_UPDATES_PER_EPISODE (default 15!) which, when 0 < k < batch, instead blends
# k-sample statistics into the RUNNING buffers with momentum 0.1 and forwards in eval() -- a
# different, stateful estimator. Plain BN-Test (a single train()-mode forward on the full batch) is
# the k <= 0 path, so the comparison sets it to 0 on both sides. Batch sizes must also match.
c_bn = _default_clone()
c_bn.defrost()
c_bn.merge_from_file("cfgs/cifar10_c/norm_test.yaml")
c_bn.TEST.BATCH_SIZE = 100
c_bn.MIXED_GRADUAL.NUM_UPDATES_PER_EPISODE = 0
c_bn.TIMING.ENABLED = False
_mdl, _pre = get_model(c_bn, get_num_classes(c_bn.CORRUPTION.DATASET), DEVICE)
_mdl.model_preprocess = _pre
_frozen = c_bn.clone(); _frozen.freeze()
m_bnt = ADAPTATION_REGISTRY.get("bntest")(cfg=_frozen, model=_mdl, num_classes=10)
c_el = make_cfg(NORM_MODE="bn_test", N_ITERS=0)
c_el.TEST.BATCH_SIZE = 100
c_el.MIXED_GRADUAL.NUM_UPDATES_PER_EPISODE = 0
m_el = build_method(c_el)
torch.set_grad_enabled(False)
worst7 = 0.0
for _b, _s in ((100, 0), (100, 1), (37, 2)):
    _x = fake_batch(_b, seed=100 + _s)
    with torch.no_grad():
        o_el, o_bn = m_el.forward_and_adapt([_x]), m_bnt.forward_and_adapt([_x])
    worst7 = max(worst7, float((o_el - o_bn).abs().max()))
check("elatta(NORM_MODE=bn_test, N_ITERS=0) == bntest, bitwise, on 3 batches", worst7 == 0.0,
      f"worst max|delta| = {worst7:.3e}")

print("  7d. an unknown NORM_MODE is rejected")
try:
    build_method(make_cfg(NORM_MODE="bn_alpha"))
    ok7, why7 = False, "no exception"
except ValueError as exc:
    ok7, why7 = True, str(exc)[:60]
check("NORM_MODE='bn_alpha' rejected", ok7, why7)


# =================================================================================================
print("\n" + "=" * 95)
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print("  - " + f)
    sys.exit(1)
print("All ELaTTA checks passed.")
