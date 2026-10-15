"""
Batched (mu/mu_w, lambda)-CMA-ES in pure PyTorch.

Written for methods/elatta.py. ELaTTA (Luo et al., arXiv:2510.11068) is a strictly batch-1 method
whose per-sample search is CMA-ES; this repo's evaluation protocol runs batches of 100. Because
ELaTTA's encoder is frozen and in eval() (frozen normalization statistics), the per-sample features
of a batch-100 forward are identical to 100 separate batch-1 forwards, so the faithful thing to do
is run B *fully independent* searches. This module does exactly that: every state tensor carries a
leading batch dimension and all B searches step together, with no coupling of any kind between
batch elements.

    mean   [B, n]        distribution mean m
    sigma  [B]           global step size
    C      [B, n, n]     covariance matrix
    pc     [B, n]        evolution path for the rank-one update
    ps     [B, n]        conjugate evolution path for step-size control

For ELaTTA, n = k = 16 and B = 100, so C is 100*16*16 = 25.6k floats and the per-generation
eigendecomposition is negligible. We therefore refresh the eigendecomposition of C every generation
(no lazy-update heuristic) and keep the ES state in float64.

REFERENCE
---------
All constants and update equations follow

    N. Hansen, "The CMA Evolution Strategy: A Tutorial", arXiv:1604.00772,
    Table 1 ("default strategy parameters") and Algorithm/Eqs. (41)-(47).

i.e. the same recipe `pycma`'s `cma.CMAEvolutionStrategy` implements. Specifically:

    lambda  = 4 + floor(3 ln n)                                      (default population)
    mu      = floor(lambda / 2)
    w_i'    = ln(lambda/2 + 1/2) - ln(i),  i = 1..mu,  w = w'/sum(w') (Eq. 49, log-weighted
                                                                       recombination)
    mu_eff  = 1 / sum(w_i^2)
    c_sigma = (mu_eff + 2) / (n + mu_eff + 5)                         (Eq. 43 / Table 1)
    d_sigma = 1 + 2 max(0, sqrt((mu_eff - 1)/(n + 1)) - 1) + c_sigma
    c_c     = (4 + mu_eff/n) / (n + 4 + 2 mu_eff/n)                   (Eq. 45 / Table 1)
    c_1     = 2 / ((n + 1.3)^2 + mu_eff)                              (Eq. 47 / Table 1)
    c_mu    = min(1 - c_1, 2 (1/4 + mu_eff - 2 + 1/mu_eff) / ((n + 2)^2 + mu_eff))  (Eq. 58)
    E||N(0,I)|| ~ sqrt(n) (1 - 1/(4n) + 1/(21 n^2))                   (chiN)

Note on the weights: Hansen's tutorial Eq. (49) and `pycma` both use ln(lambda/2 + 1/2); the older
`purecmaes.m` uses ln(mu + 1/2). The two are IDENTICAL whenever lambda is even -- and ELaTTA's
lambda = 12 (k = 16) is even -- so the choice is immaterial for the configuration that matters, and
we follow the tutorial.

`constants="tutorial"` (the default) is the recipe above. `constants="pycma"` switches the three
constants where `cma.CMAEvolutionStrategy`'s defaults differ from Table 1, so that the two
implementations agree to machine precision (tests/test_cma_es.py test 2 exercises this):

    quantity   tutorial (Table 1)                 pycma default
    c_sigma    (mu_eff + 2) / (n + mu_eff + 5)     (mu_eff + 2) / (n + mu_eff + 3)
    c_mu       identical (Table 1 Eq. 58, incl. its 0.25 term)

c_sigma is the ONLY divergence: 0.23169 (tutorial) vs 0.25207 (pycma) at n = 16, i.e. pycma adapts
the step size ~9 % faster. c_mu is 0.0137162 in BOTH -- the 0.25 belongs to Hansen's Eq. (58) and is
not a pycma extension, contrary to what this docstring claimed before. Verified against
`cma.CMAEvolutionStrategy(...).adapt_sigma.cs` / `.sp.cmu` at n=16, popsize=12.
ELaTTA never states which library it used, so `ELATTA.CMA_CONSTANTS` exposes both
(docs/elatta/implementation_notes.md A8).

DELIBERATE OMISSIONS relative to `pycma`'s defaults, each of which would be a *different* algorithm
than the tutorial's baseline and none of which the ELaTTA paper mentions:

  * **Active CMA (negative weights) is NOT implemented.** Note this IS part of Hansen 2016 --
    Table 1 Eq. (49) defines w'_i for i = 1..lambda and Eqs. (50)-(53) scale the negative weights;
    only the tutorial's appended MATLAB listing omits them ("This code does not implement negative
    weights"). `pycma` enables it by default (`CMA_active=True`). So omitting it is a deviation from
    BOTH references, not a faithful reading of the tutorial as this docstring previously claimed.
    Measured cost at ELaTTA's budget (n=16, lambda=12, 8 generations, same seed): sigma 0.5048
    active vs 0.5634 non-active (10.4 %), ||dm||/||m|| 112 %, ||dC||_F 0.55 -- NOT negligible.
    Consequence: the most likely thing the authors actually ran -- pycma with its defaults, i.e.
    active CMA plus the +3 c_sigma -- is not reachable from any current CMA_CONSTANTS setting.
    The validation test in tests/test_cma_es.py disables it on the `pycma` side
    (`CMA_active: False`) so the two are comparable.
  * No restarts, no termination criteria, no boundary handling, no diagonal-only warm-up phase, no
    step-size limits, no `sigma_vec` coordinate-wise scaling, no TPA/median-based sigma adaptation.
    ELaTTA runs a hard `n = 8` generations and stops (paper hazard #5: running longer *hurts*), so
    stopping criteria are dead code here.
  * Fitness values are used only through their *ranking*, exactly as in CMA-ES; no scaling of the
    objective can change the trajectory.

`pycma` is NOT a runtime dependency of this repo. tests/test_cma_es.py cross-checks this
implementation against it when it happens to be installed and skips otherwise.
"""

import math

import torch


def default_population(n: int) -> int:
    """lambda = 4 + floor(3 ln n) -- Hansen's default, and the formula the ELaTTA paper prints (it prints the FORMULA only --
    the integers 12 and 96 appear nowhere in v1/v2/v3, so they must not be attributed to it; the
    natural-log-and-floor reading is justified instead by Hansen 2016 Table 1 Eq. (48), which is
    literally lambda = 4 + floor(3 ln n) and which the paper cites in the same sentence, and is
    independently confirmed by FOA's released code: 4 + 3*ln(2304) = 27.2 -> popsize 27).

    n = 16 -> 4 + floor(3 * 2.7726) = 4 + 8 = 12, hence lambda * n = 12 * 8 = 96 fitness
    evaluations per test sample at the paper's IC setting. Both integers are DERIVED here, not
    quoted: the paper writes only E = lambda * n.
    """
    return 4 + int(math.floor(3.0 * math.log(n)))


class BatchedCMAES:
    """B independent (mu/mu_w, lambda)-CMA-ES instances stepping in lockstep.

    Parameters
    ----------
    batch_size : int      number of independent searches, B
    dim : int             search-space dimension, n
    sigma0 : float or Tensor[B]   initial step size
    mean0 : Tensor[B, n] or None  initial mean; None -> zeros (ELaTTA's "strict" init)
    population : int or None      lambda; None -> default_population(dim)
    generator : torch.Generator   RNG for `ask` (seed it off cfg.RNG_SEED for reproducibility)

    Notes
    -----
    * The ES state is float64 regardless of the model dtype: n is tiny, and the covariance
      eigendecomposition is the one place where fp32 would actually bite.
    * All B searches draw from a single `generator`, so the sampled numbers -- and hence the
      results -- depend on B. That is unavoidable without B separate generators and is documented
      in docs/elatta/implementation_notes.md (A9).
    """

    CONSTANT_SETS = ("tutorial", "pycma")

    def __init__(self, batch_size, dim, sigma0, mean0=None, population=None, mu=None,
                 device="cpu", dtype=torch.float64, generator=None, constants="tutorial"):
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if constants not in self.CONSTANT_SETS:
            raise ValueError(f"constants must be one of {list(self.CONSTANT_SETS)}, got {constants!r}")
        self.constants = constants
        B, n = int(batch_size), int(dim)
        self.B, self.n = B, n
        self.device, self.dtype = device, dtype
        self.generator = generator

        self.lam = int(population) if population else default_population(n)
        if self.lam < 2:
            raise ValueError(f"population must be >= 2, got {self.lam}")
        self.mu = int(mu) if mu else self.lam // 2
        self.mu = max(1, min(self.mu, self.lam))

        # ---- recombination weights (Eq. 49) ---------------------------------------------------
        # ln(lambda/2 + 1/2), per the tutorial and pycma; == ln(mu + 1/2) for even lambda.
        i = torch.arange(1, self.mu + 1, device=device, dtype=dtype)
        w = math.log(self.lam / 2.0 + 0.5) - torch.log(i)
        w = torch.clamp(w, min=0.0)
        w = w / w.sum()
        self.weights = w                                    # [mu], sums to 1
        self.mueff = float(1.0 / (w ** 2).sum().item())

        # ---- learning rates / damping (Table 1, or pycma's defaults) --------------------------
        mueff, nf = self.mueff, float(n)
        # The ONLY difference between the two sets is c_sigma's offset. The 0.25 in c_mu is in
        # Hansen 2016 Table 1 Eq. (58) ITSELF -- it is not a pycma extension. An earlier version of
        # this file omitted it from the "tutorial" branch, which produced a third variant matching
        # neither reference (c_mu 0.0121905 vs the correct 0.0137162 at n=16, 11 % low). Verified:
        # with the 0.25 term the formula reproduces cma 4.4.4's `sp.cmu` to 1e-16 at n=16, lam=12.
        cs_offset = 5.0 if constants == "tutorial" else 3.0
        rankmu_offset = 0.25
        self.cs = (mueff + 2.0) / (nf + mueff + cs_offset)
        self.ds = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (nf + 1.0)) - 1.0) + self.cs
        self.cc = (4.0 + mueff / nf) / (nf + 4.0 + 2.0 * mueff / nf)
        self.c1 = 2.0 / ((nf + 1.3) ** 2 + mueff)
        self.cmu = min(1.0 - self.c1,
                       2.0 * (rankmu_offset + mueff - 2.0 + 1.0 / mueff)
                       / ((nf + 2.0) ** 2 + mueff))
        self.cm = 1.0                                       # mean learning rate (Hansen: 1)
        self.chiN = math.sqrt(nf) * (1.0 - 1.0 / (4.0 * nf) + 1.0 / (21.0 * nf * nf))

        # ---- distribution state ---------------------------------------------------------------
        if mean0 is None:
            self.mean = torch.zeros(B, n, device=device, dtype=dtype)
        else:
            self.mean = mean0.to(device=device, dtype=dtype).expand(B, n).clone()
        if torch.is_tensor(sigma0):
            self.sigma = sigma0.to(device=device, dtype=dtype).expand(B).clone()
        else:
            self.sigma = torch.full((B,), float(sigma0), device=device, dtype=dtype)
        self.C = torch.eye(n, device=device, dtype=dtype).expand(B, n, n).clone()
        self.pc = torch.zeros(B, n, device=device, dtype=dtype)
        self.ps = torch.zeros(B, n, device=device, dtype=dtype)
        self.countiter = 0

        self._eigen_decompose()

    # ------------------------------------------------------------------------------------------
    def _eigen_decompose(self):
        """Refresh B, D from C. Done every generation -- n is 16, this is free."""
        C = 0.5 * (self.C + self.C.transpose(-1, -2))       # enforce symmetry
        evals, evecs = torch.linalg.eigh(C)                 # ascending eigenvalues
        evals = torch.clamp(evals, min=1e-30)
        self.C = C
        self.eigvecs = evecs                                # [B, n, n], columns are eigenvectors
        self.eigvals = evals                                # [B, n]
        self.D = torch.sqrt(evals)                          # [B, n]

    def ask(self):
        """Sample lambda candidates per batch element.

        Returns
        -------
        X : Tensor[B, lambda, n] -- x_i = m + sigma * B * D * z_i,  z_i ~ N(0, I)   (Eq. 38-40)
        """
        z = torch.randn(self.B, self.lam, self.n, device=self.device, dtype=self.dtype,
                        generator=self.generator)
        y = torch.einsum("bij,bkj->bki", self.eigvecs, z * self.D.unsqueeze(1))
        return self.mean.unsqueeze(1) + self.sigma.view(-1, 1, 1) * y

    def tell(self, X, fitness):
        """Update m, sigma, C from the population X and its fitness (lower is better).

        X       : Tensor[B, lambda, n] -- the population returned by `ask`
        fitness : Tensor[B, lambda]    -- objective values; only the ranking is used
        """
        X = X.to(device=self.device, dtype=self.dtype)
        fitness = fitness.to(device=self.device, dtype=self.dtype)
        B, n, mu = self.B, self.n, self.mu

        order = torch.argsort(fitness, dim=1)                       # ascending
        idx = order[:, :mu]                                         # [B, mu]
        Xsel = torch.gather(X, 1, idx.unsqueeze(-1).expand(B, mu, n))

        m_old = self.mean
        # y_i = (x_i - m_old) / sigma ; y_w = sum_i w_i y_i                       (Eq. 41)
        Y = (Xsel - m_old.unsqueeze(1)) / self.sigma.view(-1, 1, 1)
        yw = torch.einsum("m,bmn->bn", self.weights, Y)

        self.mean = m_old + self.cm * self.sigma.unsqueeze(-1) * yw               # (Eq. 42)

        # p_sigma <- (1-c_s) p_sigma + sqrt(c_s (2-c_s) mu_eff) C^{-1/2} y_w      (Eq. 43)
        invsqrtC_yw = torch.einsum(
            "bij,bj->bi", self.eigvecs,
            torch.einsum("bji,bj->bi", self.eigvecs, yw) / self.D)
        self.ps = (1.0 - self.cs) * self.ps + \
            math.sqrt(self.cs * (2.0 - self.cs) * self.mueff) * invsqrtC_yw

        # Heaviside step h_sigma (Eq. 46's stalling guard)
        ps_norm = self.ps.norm(dim=1)
        denom = math.sqrt(max(1e-30, 1.0 - (1.0 - self.cs) ** (2 * (self.countiter + 1))))
        hsig = (ps_norm / denom / self.chiN < (1.4 + 2.0 / (n + 1.0))).to(self.dtype)  # [B]

        # p_c <- (1-c_c) p_c + h_sigma sqrt(c_c (2-c_c) mu_eff) y_w               (Eq. 45)
        self.pc = (1.0 - self.cc) * self.pc + \
            (hsig * math.sqrt(self.cc * (2.0 - self.cc) * self.mueff)).unsqueeze(-1) * yw

        # C <- (1 - c_1 - c_mu sum w) C + c_1 (p_c p_c^T + dh C) + c_mu sum_i w_i y_i y_i^T
        #                                                                        (Eq. 47)
        dh = (1.0 - hsig ** 2) * self.cc * (2.0 - self.cc)           # [B]
        sum_w = float(self.weights.sum().item())                     # == 1 for positive weights
        rank_mu = torch.einsum("m,bmi,bmj->bij", self.weights, Y, Y)
        rank_one = self.pc.unsqueeze(-1) * self.pc.unsqueeze(-2)
        decay = (1.0 - self.c1 - self.cmu * sum_w) + self.c1 * dh    # [B]
        self.C = decay.view(-1, 1, 1) * self.C + self.c1 * rank_one + self.cmu * rank_mu

        # sigma <- sigma exp( (c_s/d_sigma) (||p_sigma||/chiN - 1) )              (Eq. 44)
        self.sigma = self.sigma * torch.exp(
            (self.cs / self.ds) * (self.ps.norm(dim=1) / self.chiN - 1.0))

        self.countiter += 1
        self._eigen_decompose()

    # ------------------------------------------------------------------------------------------
    def state_summary(self):
        """Small dict of scalars for logging."""
        return {
            "constants": self.constants,
            "lambda": self.lam,
            "mu": self.mu,
            "mueff": round(self.mueff, 4),
            "cs": round(self.cs, 6),
            "ds": round(self.ds, 6),
            "cc": round(self.cc, 6),
            "c1": round(self.c1, 6),
            "cmu": round(self.cmu, 6),
            "chiN": round(self.chiN, 6),
        }
