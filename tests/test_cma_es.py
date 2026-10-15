"""
Validation of methods/cma_es.py (the batched CMA-ES used by methods/elatta.py).

Run from the repo root, CPU-only:
    CUDA_VISIBLE_DEVICES="" python tests/test_cma_es.py

Self-checking assertions rather than a pytest suite, matching tests/test_ecotta.py and
tests/test_tinytta.py.

WHY THESE TESTS
---------------
ELaTTA has no released code, so the CMA-ES it uses is specified only as "CMA-ES" with the default
population formula. Our implementation is therefore held to Hansen's tutorial recipe, and the check
that it IS that recipe is a numerical cross-check against `pycma` (`cma.CMAEvolutionStrategy`), the
reference implementation by the same author.

`pycma` is NOT a runtime dependency of this repo -- test 2 skips itself when it is absent. Install
it only to validate:  pip install cma

Test 2 drives both implementations with the SAME populations and the SAME fitness values and
compares the resulting (mean, sigma, C) after every generation. That isolates the update equations
(Eqs. 41-47), which is the part that can be silently wrong; matching the RNG draw-for-draw is
neither possible nor interesting. `pycma`'s active CMA (negative weights, on by default) is disabled
because it is not part of the tutorial's baseline recipe and not part of ours -- see the
"DELIBERATE OMISSIONS" note in methods/cma_es.py.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from methods.cma_es import BatchedCMAES, default_population

TOL = 1e-9


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    assert ok, f"{name}: {detail}"


# -------------------------------------------------------------------------------------------------
def test_constants():
    """Default strategy parameters against Hansen's Table 1, computed by hand at n = 16."""
    print("\n1. Default strategy parameters (Hansen, arXiv:1604.00772 Table 1), n = 16")
    n = 16
    check("lambda = 4 + floor(3 ln n) = 12", default_population(n) == 12, str(default_population(n)))
    es = BatchedCMAES(3, n, 1.0)
    s = es.state_summary()
    check("mu = floor(lambda/2) = 6", es.mu == 6, str(es.mu))

    # Hand-recomputed from the formulas, independently of the implementation.
    w = torch.tensor([math.log(12 / 2 + 0.5) - math.log(i) for i in range(1, 7)],
                     dtype=torch.float64)
    w = w / w.sum()
    mueff = float(1.0 / (w ** 2).sum())
    cs = (mueff + 2) / (n + mueff + 5)
    ds = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    # Hansen 2016 Table 1 Eq. (58) includes the 1/4 ("rankmu offset"); it is NOT a pycma extension.
    # This line previously omitted it, which is how the implementation's 11 %-low c_mu went unnoticed.
    # With the 1/4 the formula reproduces cma 4.4.4's `sp.cmu` exactly at n=16, lambda=12.
    cmu = min(1 - c1, 2 * (0.25 + mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    chiN = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
    for key, want in (("mueff", mueff), ("cs", cs), ("ds", ds), ("cc", cc), ("c1", c1),
                      ("cmu", cmu), ("chiN", chiN)):
        got = getattr(es, key)
        check(f"{key} = {want:.9f}", abs(got - want) < 1e-12, f"got {got}")
    check("weights sum to 1", abs(float(es.weights.sum()) - 1.0) < 1e-12)
    check("weights are decreasing", bool((es.weights[:-1] >= es.weights[1:]).all()))


# -------------------------------------------------------------------------------------------------
def test_against_pycma():
    """Same populations + fitness -> same (mean, sigma, C) as pycma, generation by generation."""
    print("\n2. Cross-check against pycma (reference implementation, same author)")
    try:
        import cma
    except ImportError:
        print("  [SKIP] pycma is not installed (it is NOT a runtime dependency). "
              "pip install cma to run this check.")
        return

    n, sigma0, gens = 6, 0.3, 12
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    # constants="pycma" switches the three constants where pycma's defaults differ from the
    # tutorial's Table 1 (c_sigma's +3 vs +5 and c_mu's rankmu_offset). With those matched the two
    # implementations must agree to machine precision; the DEFAULT for ELaTTA is "tutorial".
    ours = BatchedCMAES(1, n, sigma0, population=default_population(n), generator=g,
                        constants="pycma")

    ref = cma.CMAEvolutionStrategy(
        [0.0] * n, sigma0,
        {"popsize": ours.lam, "CMA_active": False, "verbose": -9, "seed": 1,
         "AdaptSigma": cma.sigma_adaptation.CMAAdaptSigmaCSA, "maxiter": gens + 5})

    # pycma does NOT start from C = I: it seeds the initial eigenvalue spectrum with a deliberate
    # ~1e-5 spread, D_i = 1 + i * 5e-5/n (measured: [1, 1.00000833, ...] at n = 6), presumably to
    # avoid a degenerate spectrum. Ours starts from exactly I, per the tutorial. Copy pycma's
    # initial C over so the comparison isolates the UPDATE equations; without this the two agree
    # only to ~1e-4 after 12 generations, entirely from that initialisation.
    ours.C = torch.diag(torch.tensor(ref.sm.D, dtype=torch.float64) ** 2).unsqueeze(0)
    ours._eigen_decompose()
    # Second known difference: E||N(0,I)||. We use the tutorial's series
    # sqrt(n)(1 - 1/(4n) + 1/(21n^2)) = 2.3506677 at n = 6; current pycma uses a more accurate
    # expression, cma.utilities.math.Mh.chiN(6) = 2.3499640 (relative gap 3e-4), which alone
    # produces a ~1e-4 divergence in sigma over 12 generations. Adopt pycma's value here so the
    # comparison isolates the update equations rather than re-measuring a known approximation.
    from cma.utilities.math import Mh
    ours.chiN = float(Mh.chiN(n))

    def sphere(x):  # any deterministic objective works; only the ranking is used
        return float(((x - 0.5) ** 2).sum() + 0.1 * float(torch.sin(5 * x).sum()))

    worst = {"mean": 0.0, "sigma": 0.0, "C": 0.0}
    for _ in range(gens):
        X = ours.ask()[0]                                     # [lam, n], float64
        fit = torch.tensor([sphere(x) for x in X], dtype=torch.float64)
        ref.ask()                                             # advance pycma's bookkeeping
        ref.tell([list(map(float, x)) for x in X], [float(f) for f in fit])
        ours.tell(X.unsqueeze(0), fit.unsqueeze(0))

        m_ref = torch.tensor(ref.mean, dtype=torch.float64)
        C_ref = torch.tensor(ref.C, dtype=torch.float64) if not callable(getattr(ref, "C", None)) \
            else torch.tensor(ref.sm.covariance_matrix, dtype=torch.float64)
        worst["mean"] = max(worst["mean"], float((ours.mean[0] - m_ref).abs().max()))
        worst["sigma"] = max(worst["sigma"], abs(float(ours.sigma[0]) - float(ref.sigma)))
        worst["C"] = max(worst["C"], float((ours.C[0] - C_ref).abs().max()))

    print(f"  worst |delta| over {gens} generations: mean {worst['mean']:.3e}, "
          f"sigma {worst['sigma']:.3e}, C {worst['C']:.3e}")
    check("mean matches pycma", worst["mean"] < 1e-8, f"{worst['mean']:.3e}")
    check("sigma matches pycma", worst["sigma"] < 1e-8, f"{worst['sigma']:.3e}")
    check("C matches pycma", worst["C"] < 1e-8, f"{worst['C']:.3e}")


# -------------------------------------------------------------------------------------------------
def test_batch_independence():
    """B searches must not couple. Perturbing one row's fitness may not move another row's state."""
    print("\n3. Batch independence (the premise of ELaTTA's batch-100 grouping)")
    B, n = 5, 4
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    a = BatchedCMAES(B, n, 0.5, generator=g1)
    b = BatchedCMAES(B, n, 0.5, generator=g2)
    for _ in range(5):
        X = a.ask()
        fit = (X ** 2).sum(-1)
        fit_perturbed = fit.clone()
        fit_perturbed[0] = fit_perturbed[0].flip(0) * 3.0 + 1.0   # mangle row 0 only
        a.tell(X, fit)
        b.tell(X, fit_perturbed)
    check("rows 1.. unchanged in mean", float((a.mean[1:] - b.mean[1:]).abs().max()) == 0.0)
    check("rows 1.. unchanged in sigma", float((a.sigma[1:] - b.sigma[1:]).abs().max()) == 0.0)
    check("rows 1.. unchanged in C", float((a.C[1:] - b.C[1:]).abs().max()) == 0.0)
    check("row 0 DID change", float((a.mean[0] - b.mean[0]).abs().max()) > 0.0)


# -------------------------------------------------------------------------------------------------
def test_optimizes():
    """Sanity: it actually minimizes, including on an ill-conditioned rotated problem.

    The rotated ellipsoid (condition number 1e6) is the standard check that the covariance
    adaptation -- not just the step-size control -- is working: a plain isotropic search or a
    diagonal-only update cannot solve it in this budget.
    """
    print("\n4. It optimizes (rotated ellipsoid, cond 1e6, n = 8; then Rosenbrock)")
    B, n = 4, 8
    g = torch.Generator().manual_seed(3)
    scales = torch.tensor([1e6 ** (i / (n - 1)) for i in range(n)], dtype=torch.float64)
    Q = torch.linalg.qr(torch.randn(n, n, generator=torch.Generator().manual_seed(5),
                                    dtype=torch.float64))[0]

    def ellipsoid(X):
        return ((X @ Q) ** 2 * scales).sum(-1)

    es = BatchedCMAES(B, n, 1.0, mean0=torch.ones(B, n, dtype=torch.float64), generator=g)
    best = torch.full((B,), float("inf"), dtype=torch.float64)
    for _ in range(600):
        X = es.ask()
        f = ellipsoid(X)
        es.tell(X, f)
        best = torch.minimum(best, f.min(dim=1).values)
    print(f"  ellipsoid best per run: {['%.2e' % float(v) for v in best]}")
    check("all runs reach f < 1e-10 (from f0 ~ 1e6)", bool((best < 1e-10).all()), str(best.tolist()))

    g = torch.Generator().manual_seed(3)
    es = BatchedCMAES(B, n, 0.5, mean0=torch.full((B, n), 0.1, dtype=torch.float64), generator=g)

    def rosen(X):
        return (100.0 * (X[..., 1:] - X[..., :-1] ** 2) ** 2 + (1.0 - X[..., :-1]) ** 2).sum(-1)

    best = torch.full((B,), float("inf"), dtype=torch.float64)
    for _ in range(600):
        X = es.ask()
        f = rosen(X)
        es.tell(X, f)
        best = torch.minimum(best, f.min(dim=1).values)
    print(f"  Rosenbrock best per run: {['%.2e' % float(v) for v in best]} (f0 = "
          f"{float(rosen(torch.full((1, n), 0.1, dtype=torch.float64))):.2f}). Without restarts, "
          f"non-active CMA-ES stalls in Rosenbrock's local basin on a minority of runs -- expected, "
          f"and irrelevant to ELaTTA, which runs 8 generations.")
    check("majority of Rosenbrock runs reach f < 1e-6",
          int((best < 1e-6).sum()) >= B // 2 + 1, str(best.tolist()))


# -------------------------------------------------------------------------------------------------
def test_reproducible():
    """Same seed -> identical trajectory. ELaTTA's reproducibility rests on this."""
    print("\n5. Reproducibility from the seed")
    def run(seed):
        g = torch.Generator().manual_seed(seed)
        es = BatchedCMAES(3, 5, 0.7, generator=g)
        for _ in range(6):
            X = es.ask()
            es.tell(X, (X ** 2).sum(-1))
        return es.mean.clone(), es.sigma.clone()
    m1, s1 = run(11)
    m2, s2 = run(11)
    m3, _ = run(12)
    check("identical for equal seeds", float((m1 - m2).abs().max()) == 0.0
          and float((s1 - s2).abs().max()) == 0.0)
    check("different for different seeds", float((m1 - m3).abs().max()) > 0.0)


if __name__ == "__main__":
    print("=" * 95)
    print("methods/cma_es.py validation")
    print("=" * 95)
    test_constants()
    test_against_pycma()
    test_batch_independence()
    test_optimizes()
    test_reproducible()
    print("\nAll CMA-ES checks passed.")
