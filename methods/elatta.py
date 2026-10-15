"""
ELaTTA: Efficient Test-Time Adaptation through Latent Subspace Coefficients Search
Luo, Liu, Chen, Yang, Ding, Basu, Li -- arXiv:2510.11068 (v2), under review.

RECONSTRUCTION FROM PROSE -- NOT A PORT
=======================================
No code has been released for this paper, in any language (docs/elatta/paper_spec.md, "Code
released: none"). There is no third-party reimplementation either. Every unstated detail below is a
judgement call, labelled PAPER / INFERRED / DEVIATION, and every behavioural choice is reachable
from a `cfg.ELATTA` key. The full ambiguity list with provenance is docs/elatta/implementation_notes.md
(A1..A15); this docstring covers only what a reader of the code needs.

The paper never evaluates CIFAR-10-C, and never evaluates a gradual/mixed corruption schedule, so
**no published number this file can be checked against exists**. Its reported effect size is small
(ImageNet-C ViT-B: 55.03 -> 57.82 % accuracy, +2.79 points), so calibrate expectations accordingly.


Method
------
Every network weight is frozen forever. There is no optimizer, no gradient, no BN recalibration, no
augmentation, no memory bank, no teacher.

OFFLINE (once, cheap -- ~20 forward passes):
    Z_s    = [Enc(x_1); ...; Enc(x_N)]                   N clean SOURCE TRAIN images, [N, D]
    Z_c    = Z_s - mean(Z_s)                             centered
    U,S,Vh = svd(Z_c);  V_k = Vh[:k].T                   [D, k], orthonormal columns
    cache V_k (and S, for the sigma0 auto-derivation)    ~0.015 % of model size

ONLINE (per test sample -- ONE encoder pass, lambda*n ~ 96 classifier-head evaluations):
    z_t = Enc(x)                                                        [D]
    search p in R^k with CMA-ES to minimise   L(p) = H(softmax(Cls(z_t + V_k p)))
    return the logits of the p with the smallest fitness EVER SAMPLED   (paper Step 3)

`H` is Shannon entropy in nats. `Enc` = backbone minus the final linear layer, `Cls` = that linear
layer -- i.e. exactly this repo's `split_up_model` / `aura_fc` cut (paper: "the final
post-global-average-pool feature; INFERRED, not quoted -- see below). On CIFAR-10 with RobustBench
`Standard` (WRN-28-10) that is D = 640, C = 10.

The paper's argument is that CMA-ES does not merely dodge backprop: it descends the Gaussian-
smoothed objective J(m, Sigma) = E_{p ~ N(m, Sigma)}[L(p)] (its Eq. 3), so the returned solution is
low-entropy over a whole neighbourhood of span(V_k) rather than at a point. Prop. 3.1 is the honest
counterpart: writing z_t = z* + xi and splitting xi w.r.t. span(V_k),
||z_adapted - z*||^2 = ||xi_par + V_k p||^2 + ||xi_perp||^2, so only the IN-SUBSPACE part of a shift
is correctable. A corruption orthogonal to the top-k source directions makes the method a no-op.


Batch 100 vs the paper's batch 1  (the load-bearing engineering point)
---------------------------------------------------------------------
The paper's MAIN results are batch size 1, but it is not "batch 1 everywhere": it says "Unless
otherwise specified, all main results use the strict protocol with batch size 1" and then specifies
otherwise -- v3 Sec. IV-C / Table V give a continual BATCH protocol at batch size 64 with a SHARED
latent coefficient p per batch, warm-started from the previous batch's solution. This repo's protocol
runs
batch 100. That is NOT a variant here, because of what ELaTTA does not do:

  * the model is in eval() with FROZEN BN running statistics, so the encoder is pointwise in the
    batch -- feature i of a batch-100 forward equals the feature of a batch-1 forward of sample i,
    up to floating-point associativity in cuDNN's kernel selection;
  * `Cls` is a single matmul, also pointwise;
  * the search state is per sample and never touches another sample's state (methods/cma_es.py
    carries a leading batch dimension through every tensor and never reduces over it).

So we run the encoder ONCE on the whole batch and then B fully independent CMA-ES searches. The
*mechanics* are exact, and that is what is asserted, not assumed: tests/test_elatta.py section 2
measures the pointwiseness of Enc and Cls, the frozenness of BN, and that a row of the batched ES
equals a standalone 1-element ES driven with the same populations/fitness. `ELATTA.LOG_EVERY` logs
encoder forwards vs head evaluations so the paper's efficiency claim (1 encoder pass, ~96 head
passes per sample) is verifiable from the run log.

CAVEAT, and it is not small: the *outcome* is nevertheless batch-dependent, because all B searches
draw from one shared torch.Generator, so the random stream a given sample sees depends on B
(docs/elatta/implementation_notes.md A9). Measured on 100 fog/sev-5 frames: a batch-100 run and a
literal TEST.BATCH_SIZE=1 run of the same seed agree on only ~79 % of predictions and differ by
~3 accuracy points. The two are statistically equivalent, NOT per-sample identical -- do not claim
otherwise. (The A11 tie-saturation below is what amplifies a different RNG draw into a different
argmax.)

We deliberately do NOT enter methods/base.py's sliding-window path (which fires at batch size 1):
ELaTTA has no window, no buffer and no BN statistics to accumulate. `TEST.WINDOW_LENGTH > 1` is
rejected.


The sigma0 hazard (paper spec hazard #4 -- the single biggest reproduction risk)
-------------------------------------------------------------------------------
sigma0, the CMA-ES initial step size, is NOT REPORTED anywhere in the paper, and the scale it must
have depends entirely on how V_k is scaled. With paper-literal orthonormal columns, a coefficient p_j
is in raw feature-projection units, whose source standard deviation is S_j / sqrt(N-1) -- for
WRN-28-10 features that is O(1)-O(10), so sigma0 = 0.5 (a typical CMA-ES default) would search a
region hundreds of times too small and the method would be a silent no-op.

Two knobs, near-equivalent by construction (they are the same search space in different units --
this is a reparameterization, NOT a change of method):

  ELATTA.BASIS_SCALING = "orthonormal"   paper-literal V_k; sigma0 auto = mean(S)/sqrt(N-1)
                       = "whitened"      columns scaled by S_j/sqrt(N-1); sigma0 auto = 1.0
                                         ("one source standard deviation along this direction")

  ELATTA.SIGMA0 = -1.0                   auto-derive per the above; any positive float overrides.

"whitened" is the DEFAULT because it makes sigma0 scale-free and dimensionless, so the same value
transfers across backbones, datasets and k without retuning -- the safer reproduction. It is
recorded as a deviation-of-parameterization in docs/elatta/implementation_notes.md (A3). The
resolved sigma0 is logged at INFO on every run. Setting the two paths against each other is a
useful internal consistency check: they should land within noise of one another, and if they do not,
something is wrong with the scaling.

The cached checkpoint stores the ORTHONORMAL V_k plus the singular values, and the scaling is
applied at load time, so switching BASIS_SCALING never requires a rebuild.


What is not the paper's
-----------------------
* `ELATTA.INIT_MODE="continual"` warm-starts m^(0) from the previous test SAMPLE in the paper. Our
  batch grouping makes intra-batch order arbitrary, so we keep ONE global p_prev, use it as m^(0)
  for every sample of the next batch, and set it to the mean of the winning p* over the batch at the
  end of each batch. Batch-grouping-forced deviation (A6). Only the mean is inherited -- never sigma
  or C, which is the literal reading of the paper's algorithm box.
* `ELATTA.INCLUDE_UNADAPTED=True` admits p = 0 (the untouched source prediction) as a candidate in
  the argmin. The paper does NOT do this; default False. It is the natural safety net (it makes the
  method monotone in its own objective w.r.t. source) and is kept as an available ablation.
* `ELATTA.SELECTION` in {"best_ever" (paper), "last_gen_best", "final_mean"}.
* `ELATTA.N_ITERS = 0` is a non-paper value that makes the method return the source prediction
  exactly. It exists as a hard identity test (tests/test_elatta.py::test_identity).
* `ELATTA.NORM_MODE = "bn_test"` composes ELaTTA with BatchNorm recalibration. NOT the paper's
  method -- see the section below.


ELATTA.NORM_MODE: composing ELaTTA with BN recalibration (an explicit, labelled departure)
-----------------------------------------------------------------------------------------
The paper states flatly that the framework "does not require updating normalization statistics", and
`NORM_MODE="frozen_source"` (the DEFAULT) is that: eval() everywhere, BN on the stored source running
statistics, `track_running_stats` untouched.

`NORM_MODE="bn_test"` is a DELIBERATE NON-PAPER MODE. It makes the encoder's BatchNorm layers
normalize with the CURRENT TEST BATCH's statistics -- the same recalibration this repo's `bntest` /
norm_test baseline performs (methods/norm.py::BNTest.configure_model) -- while leaving every other
part of ELaTTA alone: gradient-free, no weight update, frozen affine parameters, one encoder pass per
batch, lambda*n head evaluations, the same V_k, the same CMA-ES.

Why it exists: on CIFAR-10-C with WRN-28-10 the dominant corruption failure is the BN-statistic
mismatch, not the classifier's decision geometry (BN recalibration alone: 43.52 % -> ~20 % error;
faithful ELaTTA cannot beat Source at all). With frozen source statistics the search minimizes the
entropy of a confidently WRONG model, which carries no information about correctness. This mode
measures what the latent-coefficient search adds on top of a backbone whose features are not broken.

Mechanism, and what it does NOT mutate: BN modules are put in train() mode with
`track_running_stats = False`. nn.BatchNorm then passes `None` for the running buffers, so the forward
uses batch statistics and NOTHING is written back -- `running_mean`, `running_var` and
`num_batches_tracked` stay bitwise unchanged and the method stays stateless across batches. (BNTest
itself keeps `track_running_stats=True` and therefore does mutate the buffers; because it is always in
train mode those buffers never affect its own output, so the predictions are identical either way.)

What it costs, stated plainly:
  * the method becomes BATCH-DEPENDENT and is therefore NO LONGER SINGLE-INSTANCE TTA. The paper's
    batch-1 protocol is not reachable in this mode (batch-1 BN2d statistics are degenerate); a
    warning is logged if you try.
  * the encoder stops being pointwise in the batch, so the "batching is exact" argument above applies
    only to `frozen_source`.
Any number produced here is "ELaTTA over a BN-recalibrated backbone", never "ELaTTA".
`NORM_MODE="bn_test"` with `N_ITERS=0` reproduces the `bntest` baseline exactly at equal
TEST.BATCH_SIZE -- the check that the composition is nothing more than BN-Test plus the search.
"""

import logging
import math
import os
import time
from copy import deepcopy

import torch
import torch.nn as nn

from methods.base import TTAMethod
from methods.cma_es import BatchedCMAES, default_population
from models.model import split_up_model
from utils.registry import ADAPTATION_REGISTRY

logger = logging.getLogger(__name__)

BASIS_SCALINGS = ("orthonormal", "whitened")
INIT_MODES = ("strict", "continual")
SELECTIONS = ("best_ever", "last_gen_best", "final_mean")
SAMPLE_ORDERS = ("random", "first")
NORM_MODES = ("frozen_source", "bn_test")


# -------------------------------------------------------------------------------------------------
# Offline basis
# -------------------------------------------------------------------------------------------------

def default_basis_path(cfg) -> str:
    """`ckpt/elatta/<dataset>__<arch>__k<k>__N<N>__seed<seed>.pth`.

    Only COMPATIBILITY keys go in the filename. BASIS_SCALING is deliberately absent: the payload
    stores the orthonormal V_k plus the singular values and the scaling is applied on load, so one
    artifact serves both settings (the lesson from docs/ecotta/implementation_notes.md Sec. 5.3 --
    do not fail to find a perfectly usable artifact over a key that does not change it).
    """
    fname = (f"{cfg.CORRUPTION.DATASET}__{cfg.MODEL.ARCH}"
             f"__k{cfg.ELATTA.K}__N{cfg.ELATTA.N_SOURCE_SAMPLES}__seed{cfg.RNG_SEED}.pth")
    return os.path.join(cfg.CKPT_DIR, "elatta", fname)


@torch.no_grad()
def build_basis(cfg, encoder, device, preprocess=None):
    """Build V_k from N clean source images and return the cache payload (does not write it).

    PAPER (offline phase, steps 1-4): forward N clean source samples through the frozen encoder,
    center, rank-k truncated SVD, keep V_k. mu_s is used only for the centering and is NOT used
    online, so it is stored for diagnostics only.

    A1  WHICH SPLIT. The paper says "clean source samples" without naming a split. We use the
        source TRAIN split (`train_split=True`), which is the only leak-free choice: the test stream
        is built from the CIFAR-10 *test* images, so drawing the basis from the test split would
        show the method the exact images it is evaluated on. Logged explicitly.
    A2  WHICH N IMAGES. Unspecified. Default `BASIS_SAMPLE_ORDER="random"`: a subset drawn with a
        torch.Generator seeded off RNG_SEED, so it is reproducible and not order- or class-biased.
        "first" takes the first N of the split.

    With the paper's N=20 and k=16, rank(Z_c) <= N-1 = 19, which the paper is explicit about being
    content with. N < k+1 is rejected -- the trailing directions would be pure numerical noise.
    """
    from datasets.data_loading import get_source_loader     # local import: heavy module

    e = cfg.ELATTA
    N, k = int(e.N_SOURCE_SAMPLES), int(e.K)
    if N < k + 1:
        raise ValueError(
            f"ELATTA.N_SOURCE_SAMPLES={N} with ELATTA.K={k}: the centered source feature matrix has "
            f"rank at most N-1={N - 1} < k, so the trailing basis directions would be numerical "
            f"noise. Raise N_SOURCE_SAMPLES to at least K+1 (the paper uses N=20, k=16)."
        )
    if e.BASIS_SAMPLE_ORDER not in SAMPLE_ORDERS:
        raise ValueError(f"ELATTA.BASIS_SAMPLE_ORDER must be one of {list(SAMPLE_ORDERS)}, "
                         f"got {e.BASIS_SAMPLE_ORDER!r}")

    source_dataset, _ = get_source_loader(
        dataset_name=cfg.CORRUPTION.DATASET,
        adaptation=cfg.MODEL.ADAPTATION,
        preprocess=preprocess,
        data_root_dir=cfg.DATA_DIR,
        batch_size=min(N, 64),
        use_clip=getattr(cfg.MODEL, "USE_CLIP", False),
        train_split=e.BASIS_SOURCE_SPLIT == "train",
        ckpt_path=cfg.MODEL.CKPT_PATH,
        workers=0,
    )
    n_avail = len(source_dataset)
    if e.BASIS_SAMPLE_ORDER == "first":
        indices = list(range(min(N, n_avail)))
    else:
        g = torch.Generator().manual_seed(int(cfg.RNG_SEED))
        indices = torch.randperm(n_avail, generator=g)[:N].tolist()

    subset = torch.utils.data.Subset(source_dataset, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=min(N, 64), shuffle=False, num_workers=0)

    encoder.eval()
    feats = []
    for batch in loader:
        imgs = batch[0]
        imgs = imgs[0] if isinstance(imgs, list) else imgs
        feats.append(encoder(imgs.to(device)).flatten(1).double().cpu())
    Z = torch.cat(feats, dim=0)                                  # [N, D]
    if Z.shape[0] < N:
        logger.warning(f"ELaTTA basis: asked for N={N} source samples but the split only yielded "
                       f"{Z.shape[0]}.")
    N_eff, D = Z.shape
    mu_s = Z.mean(0)
    Zc = Z - mu_s
    # full_matrices=False: Vh is [min(N,D), D], and we only ever want its first k rows.
    _, S, Vh = torch.linalg.svd(Zc, full_matrices=False)
    V_k = Vh[:k].T.contiguous()                                  # [D, k], orthonormal columns
    S_k = S[:k].contiguous()                                     # [k]

    coeff_std = S_k / math.sqrt(max(1, N_eff - 1))               # source std of each coefficient
    payload = {
        "V_k": V_k.float(),                  # ORTHONORMAL -- scaling is applied at load time
        "singular_values": S_k.float(),
        "coeff_std": coeff_std.float(),
        "sigma0_auto_orthonormal": float(coeff_std.mean().item()),
        "mu_s": mu_s.float(),                # diagnostics only; never used online (paper step 4)
        "D": int(D),
        "k": int(k),
        "N": int(N_eff),
        "arch": cfg.MODEL.ARCH,
        "dataset": cfg.CORRUPTION.DATASET,
        "seed": int(cfg.RNG_SEED),
        "source_split": e.BASIS_SOURCE_SPLIT,
        "sample_order": e.BASIS_SAMPLE_ORDER,
        "sample_indices": list(indices),
        "explained_variance_ratio": float(((S_k ** 2).sum() / (S ** 2).sum()).item()),
    }
    logger.info(
        f"ELaTTA basis built: D={D}, k={k}, N={N_eff} clean images from the "
        f"{e.BASIS_SOURCE_SPLIT.upper()} split of "
        f"{cfg.CORRUPTION.DATASET.split('_')[0]} (order={e.BASIS_SAMPLE_ORDER}, seed={cfg.RNG_SEED}); "
        f"top-{k} explained variance {payload['explained_variance_ratio'] * 100:.2f}%; "
        f"coeff std min/mean/max = {coeff_std.min():.4f}/{coeff_std.mean():.4f}/{coeff_std.max():.4f}"
    )
    return payload


# -------------------------------------------------------------------------------------------------
# The method
# -------------------------------------------------------------------------------------------------

@ADAPTATION_REGISTRY.register()
class ELaTTA(TTAMethod):
    """ELaTTA. Fully gradient-free and unsupervised; labels are never used, offline or online.

    Deliberately NOT added to utils/eval_utils._is_supervised_tta -- it goes through
    TTAMethod.forward like every other unsupervised method.
    """

    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)
        e = cfg.ELATTA

        # ---- validation ----------------------------------------------------------------------
        for key, value, allowed in (("BASIS_SCALING", e.BASIS_SCALING, BASIS_SCALINGS),
                                    ("INIT_MODE", e.INIT_MODE, INIT_MODES),
                                    ("SELECTION", e.SELECTION, SELECTIONS),
                                    ("NORM_MODE", e.NORM_MODE, NORM_MODES),
                                    ("CMA_CONSTANTS", e.CMA_CONSTANTS,
                                     BatchedCMAES.CONSTANT_SETS)):
            if value not in allowed:
                raise ValueError(f"ELATTA.{key} must be one of {list(allowed)}, got {value!r}")
        if cfg.TEST.WINDOW_LENGTH > 1:
            raise ValueError(
                "ELaTTA rejects TEST.WINDOW_LENGTH > 1. base.py:77-114's sliding-window path exists "
                "to accumulate BN statistics over a buffer; ELaTTA has no buffer, no BN update and "
                "no persistent per-sample state, and the window would re-search every sample "
                "WINDOW_LENGTH times. Run TEST.BATCH_SIZE 1 WINDOW_LENGTH 1 for the paper's literal "
                "protocol, or any batch size with WINDOW_LENGTH 1 (see the module docstring on why "
                "batching is exact here)."
            )
        if self.steps != 1:
            raise ValueError(
                f"ELaTTA expects OPTIM.STEPS=1 but got {self.steps}. There is no optimizer and no "
                f"persistent parameter state, so base.py's `for _ in range(self.steps)` loop would "
                f"merely repeat the identical search {self.steps} times (in strict mode) at "
                f"{self.steps}x the cost. The number of CMA-ES generations is ELATTA.N_ITERS."
            )
        if cfg.MIXED_PRECISION:
            logger.warning("ELaTTA with MIXED_PRECISION: entropy differences between candidates can "
                           "be O(1e-4) nats late in the search; fp16 may flatten the ranking. "
                           "fp32 is recommended.")

        self.k = int(e.K)
        self.n_iters = int(e.N_ITERS)
        self.lam = int(e.POPULATION) if int(e.POPULATION) > 0 else default_population(self.k)
        self.selection = e.SELECTION
        self.init_mode = e.INIT_MODE
        self.include_unadapted = bool(e.INCLUDE_UNADAPTED)
        self.basis_scaling = e.BASIS_SCALING
        self.norm_mode = e.NORM_MODE
        self.cma_constants = e.CMA_CONSTANTS
        self.log_every = max(1, int(e.LOG_EVERY))

        # ---- encoder / classifier cut (D1) -----------------------------------------------------
        self.encoder, self.classifier = split_up_model(
            self.model, cfg.MODEL.ARCH, cfg.CORRUPTION.DATASET)
        self.encoder.eval()
        self.encoder.requires_grad_(False)
        self.classifier.eval()
        self.classifier.requires_grad_(False)
        self.classifier.to(self.device)
        # The three .eval() calls above walk the SAME modules configure_model() just configured (the
        # split is a view over the backbone's children, not a copy), so under NORM_MODE="bn_test"
        # they have just put the BN layers back into eval(). Re-apply. No-op for "frozen_source".
        self._apply_norm_mode()

        # CLAUDE.md known-limitation #4: for timm backbones split_up_model calls
        # reset_classifier(0) on self.model IN PLACE, which invalidates the state snapshot
        # TTAMethod.__init__ took two lines before this. Re-take it, exactly as aura_fc does, or
        # reset() dies on the missing head keys.
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()

        # ---- offline basis ---------------------------------------------------------------------
        self._load_or_build_basis(cfg)
        # _build_basis() calls encoder.eval() before its forward pass -- correct for the basis
        # itself (it must be computed on CLEAN source data with SOURCE statistics, which is what
        # the paper specifies), but under NORM_MODE="bn_test" it silently reverts the BN layers to
        # eval() for the whole run. Only bites on a cache MISS, so it produced a run that looked
        # like plain frozen-source ELaTTA on exactly those seeds whose basis was not yet cached.
        # Re-apply. No-op for "frozen_source", and no-op when the basis came from cache.
        self._apply_norm_mode(verbose=False)

        # ---- sigma0 (D4 / A3) ------------------------------------------------------------------
        self.sigma0 = self._resolve_sigma0(cfg)

        # ---- search state ----------------------------------------------------------------------
        # One global p_prev in continual mode (A6). Registered as a plain tensor, not a buffer: it
        # must NOT ride along in state_dict() / the reset snapshot, because reset() has to clear it.
        self.p_prev = torch.zeros(self.k, device=self.device, dtype=torch.float64)
        self.generator = torch.Generator(device=self.device).manual_seed(int(cfg.RNG_SEED))

        # ---- counters / diagnostics (D8, D9) ---------------------------------------------------
        self._encoder_forwards = 0       # number of encoder CALLS
        self._encoder_samples = 0        # number of images pushed through the encoder
        self._head_evals = 0             # number of classifier-head candidate evaluations
        self._num_batches = 0
        self._timing_enabled = cfg.TIMING.ENABLED
        self._timing_warmup = cfg.TIMING.WARMUP_SAMPLES
        self._timing_target = cfg.TIMING.NUM_MEASUREMENTS
        self._timing_results = []
        self._timing_sample_counter = 0

        es_probe = BatchedCMAES(1, self.k, 1.0, population=self.lam, device="cpu",
                                constants=self.cma_constants)
        logger.info(
            f"ELaTTA ready: k={self.k} (PAPER), n_iters={self.n_iters} (PAPER 8), "
            f"lambda={self.lam} (PAPER formula 4+floor(3 ln k)={default_population(self.k)}), "
            f"fitness evals/sample={self.lam * self.n_iters} (paper ~96), "
            f"D={self.D}, C={self.num_classes}, N_source={self.basis_meta['N']}, "
            f"basis_scaling={self.basis_scaling}, sigma0={self.sigma0:.6g} "
            f"({'auto' if float(e.SIGMA0) <= 0 else 'explicit'}), "
            f"init_mode={self.init_mode}, selection={self.selection}, "
            f"norm_mode={self.norm_mode}"
            f"{' (PAPER)' if self.norm_mode == 'frozen_source' else ' (NON-PAPER DEPARTURE)'}, "
            f"include_unadapted={self.include_unadapted}, "
            f"cma constants={es_probe.state_summary()}"
        )
        if self.n_iters == 0:
            logger.warning("ELATTA.N_ITERS=0: the search is disabled and ELaTTA returns the SOURCE "
                           "prediction exactly. This is the identity test, not a valid method "
                           "configuration.")

    # ---------------------------------------------------------------------------------------------
    # setup
    # ---------------------------------------------------------------------------------------------

    def configure_model(self):
        """Frozen everything, eval() everywhere, source normalization statistics untouched.

        NOTE ON EVIDENCE: earlier revisions of this file quoted the paper as saying "The framework
        is model-agnostic and does not require updating normalization statistics". THAT SENTENCE DOES
        NOT EXIST in v1, v2 or v3 -- it was fabricated. The paper makes no statement at all about its
        own normalization handling. The correct evidence that it runs with FROZEN SOURCE BN is
        indirect but strong: v3 Table VI's ResNet-50 ImageNet-C "No Adapt" row is 4.47 % on Gaussian
        noise, a number only reachable with frozen source statistics (test-batch BN would put it near
        30 %), and its batch-1 main protocol makes batch statistics unavailable by construction.
        activation is ever retained. So: eval(), requires_grad_(False), and `track_running_stats`
        is deliberately NOT touched -- BN keeps its source running statistics, which is what makes
        batch size 1 viable in the first place.

        ELATTA.NORM_MODE="bn_test" is a NON-PAPER departure layered on top of exactly this (see the
        module docstring); it is applied by _apply_norm_mode() and is a no-op under the default.
        """
        self.model.eval()
        self.model.requires_grad_(False)
        self._apply_norm_mode(verbose=False)

    def _apply_norm_mode(self, verbose=True):
        """Realize ELATTA.NORM_MODE. Idempotent, so it can be re-applied after any .eval() call.

        "frozen_source" (PAPER, default): nothing to do. The model is already in eval() with
        `track_running_stats` untouched, so BN uses the stored SOURCE running statistics.

        "bn_test" (NON-PAPER): the BN layers normalize with the current test batch's statistics, as
        methods/norm.py::BNTest does. Mechanism, reused rather than reinvented: train() mode makes
        nn.BatchNorm compute batch statistics, and `track_running_stats = False` makes it pass None
        for the running buffers -- so `running_mean` / `running_var` / `num_batches_tracked` are never
        written and the method stays stateless across batches (BNTest keeps the buffers live and
        mutates them, but since it never leaves train mode those buffers cannot affect its output;
        the predictions are identical). Affine weight/bias stay frozen and requires_grad=False.
        """
        mode = self.cfg.ELATTA.NORM_MODE
        if mode not in NORM_MODES:
            raise ValueError(f"ELATTA.NORM_MODE must be one of {list(NORM_MODES)}, got {mode!r}")
        if mode == "frozen_source":
            return

        n_bn, n_skipped = 0, 0
        for m in self.model.modules():
            if not isinstance(m, nn.modules.batchnorm._BatchNorm):
                continue
            if isinstance(m, nn.BatchNorm1d) and self.cfg.TEST.BATCH_SIZE <= 1:
                # Same guard as BNTest.configure_model: a single sample has no BN1d statistics.
                n_skipped += 1
                continue
            m.train()
            m.track_running_stats = False       # -> running buffers are passed as None: read-only
            n_bn += 1

        if n_bn == 0:
            raise ValueError(
                "ELATTA.NORM_MODE='bn_test' found no adaptable BatchNorm layer in this backbone "
                f"(arch={self.cfg.MODEL.ARCH}). On a LayerNorm backbone (ViT) there are no running "
                "statistics to recalibrate, so the mode would be a silent no-op -- exactly the trap "
                "CLAUDE.md notes for Table 1's BN-Test/BN-Alpha/BN-EMA rows on ViT. Use "
                "NORM_MODE='frozen_source' (the paper's method) there."
            )
        if not verbose:
            return
        if self.cfg.TEST.BATCH_SIZE <= 1:
            logger.warning(
                "ELATTA.NORM_MODE='bn_test' with TEST.BATCH_SIZE<=1: BatchNorm2d statistics from a "
                "single image are degenerate (per-channel spatial statistics only). This mode is "
                "batch-dependent by construction and cannot reproduce the paper's single-instance "
                "protocol -- use NORM_MODE='frozen_source' for that."
            )
        logger.warning(
            f"ELATTA.NORM_MODE='bn_test': {n_bn} BatchNorm layers switched to CURRENT-TEST-BATCH "
            f"statistics ({n_skipped} BatchNorm1d skipped at TEST.BATCH_SIZE<=1). THIS IS NOT THE "
            f"PAPER'S METHOD -- the paper states the framework 'does not require updating "
            f"normalization statistics'. Report anything measured here as 'ELaTTA over a "
            f"BN-recalibrated backbone', and note that the method is now batch-dependent, i.e. no "
            f"longer single-instance TTA. Running statistics are NOT mutated (track_running_stats "
            f"False -> the buffers are read-only); weights, affine parameters and the "
            f"gradient-free/one-encoder-pass cost model are unchanged."
        )

    def collect_params(self):
        """No trainable parameters -- the adapted quantity is a per-sample activation offset.

        Returning ([], []) makes base.py:52 set self.optimizer = None. copy_model_and_optimizer /
        reset below are overridden to cope with that.
        """
        return [], []

    def setup_optimizer(self):
        return None

    def copy_model_and_optimizer(self):
        model_states = [deepcopy(model.state_dict()) for model in self.models]
        return model_states, None

    def reset(self):
        """Restore the frozen weights (a no-op in practice) and clear the only piece of state.

        base.py's reset() raises when optimizer_state is None, and load_model_and_optimizer() would
        dereference self.optimizer, hence this override. p_prev is cleared because an episodic /
        reset_each_shift protocol must not carry a coefficient across the boundary.
        """
        for model, model_state in zip(self.models, self.model_states):
            model.load_state_dict(model_state, strict=True)
        self.p_prev = torch.zeros(self.k, device=self.device, dtype=torch.float64)

    # ---------------------------------------------------------------------------------------------
    # basis / sigma0
    # ---------------------------------------------------------------------------------------------

    def _load_or_build_basis(self, cfg):
        e = cfg.ELATTA
        path = e.BASIS_CKPT or default_basis_path(cfg)
        payload = None
        if os.path.isfile(path):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            for key, expected in (("arch", cfg.MODEL.ARCH),
                                  ("dataset", cfg.CORRUPTION.DATASET),
                                  ("k", self.k)):
                if key in payload and payload[key] != expected:
                    raise RuntimeError(
                        f"ELaTTA basis {path} was built with {key}={payload[key]!r} but this run "
                        f"uses {key}={expected!r}. Delete it or point ELATTA.BASIS_CKPT elsewhere."
                    )
            logger.info(f"ELaTTA: loaded cached basis from {path} "
                        f"(N={payload['N']}, split={payload.get('source_split')}, "
                        f"order={payload.get('sample_order')})")
        elif e.BASIS_CKPT:
            raise RuntimeError(
                f"ELATTA.BASIS_CKPT points at a file that does not exist:\n    {e.BASIS_CKPT}\n"
                f"Build one with:  python offline_build_elatta_basis.py --cfg <this cfg>"
            )
        elif not e.AUTO_BUILD_BASIS:
            raise RuntimeError(
                f"No ELaTTA basis at {path} and ELATTA.AUTO_BUILD_BASIS=False. The offline phase is "
                f"MANDATORY -- the paper has no cold-start variant (spec hazard #9). Build it with:\n"
                f"    python offline_build_elatta_basis.py --cfg <this cfg>"
            )
        else:
            # ~20 encoder forwards; auto-building on a cache miss costs seconds and keeps the
            # method runnable straight from a yaml.
            logger.info(f"ELaTTA: no cached basis at {path} -- building it now "
                        f"(N={e.N_SOURCE_SAMPLES} clean source images, one forward pass).")
            # get_source_loader wants the restored model's preprocessing, which the host attaches
            # to the base model in test_time.py (it is None for the RobustBench CIFAR backbones,
            # whose source transform is a plain ToTensor()).
            payload = build_basis(cfg, self.encoder, self.device,
                                  preprocess=getattr(self.model, "model_preprocess", None))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(payload, path)
            logger.info(f"ELaTTA: cached basis to {path}")

        self.basis_meta = {kk: vv for kk, vv in payload.items() if not torch.is_tensor(vv)}
        self.basis_path = path
        V = payload["V_k"].to(self.device).float()                      # [D, k], orthonormal
        self.D = V.shape[0]
        if V.shape[1] != self.k:
            raise RuntimeError(f"Cached basis has k={V.shape[1]} but ELATTA.K={self.k}.")
        if self.D != self.classifier.in_features:
            raise RuntimeError(
                f"Cached basis has D={self.D} but the classifier expects "
                f"{self.classifier.in_features} input features.")
        self.coeff_std = payload["coeff_std"].to(self.device).float()   # [k]
        if self.basis_scaling == "whitened":
            # Columns rescaled so that p_j = 1 means "one source standard deviation along
            # direction j". Same subspace, same reachable set of z_adapted; different units.
            self.V = (V * self.coeff_std.unsqueeze(0)).contiguous()
        else:
            self.V = V.contiguous()

    def _resolve_sigma0(self, cfg) -> float:
        e = cfg.ELATTA
        if float(e.SIGMA0) > 0:
            return float(e.SIGMA0)
        if self.basis_scaling == "whitened":
            sigma0 = 1.0
            why = "whitened basis -> one source std per unit coefficient"
        else:
            sigma0 = float(self.basis_meta.get("sigma0_auto_orthonormal",
                                               self.coeff_std.mean().item()))
            why = "orthonormal basis -> mean(S_k)/sqrt(N-1), the mean source coefficient std"
        logger.info(f"ELaTTA sigma0 auto-derived to {sigma0:.6g} ({why}). The paper reports NO "
                    f"value for sigma0 anywhere (spec hazard #4); this is our choice, not theirs.")
        return sigma0

    # ---------------------------------------------------------------------------------------------
    # online
    # ---------------------------------------------------------------------------------------------

    @staticmethod
    def _entropy(logits: torch.Tensor) -> torch.Tensor:
        """Shannon entropy of softmax(logits), in nats, over the last dim, unreduced."""
        return -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1)

    @torch.no_grad()
    def forward_and_adapt(self, x):
        imgs = x[0] if isinstance(x, (list, tuple)) else x
        B = imgs.shape[0]
        self._num_batches += 1

        _should_time = (self._timing_enabled
                        and self._timing_sample_counter >= self._timing_warmup
                        and len(self._timing_results) < self._timing_target)
        if _should_time:
            torch.cuda.synchronize()
            _t0 = time.perf_counter()

        # ---- Step 1: ONE encoder pass for the whole batch (paper hazard #3) -------------------
        z = self.encoder(imgs).flatten(1)                              # [B, D]
        self._encoder_forwards += 1
        self._encoder_samples += B

        src_logits = self.classifier(z)                                # [B, C], the p = 0 answer
        self._head_evals += B

        if self.n_iters <= 0:
            # Hard identity path: no generations, nothing sampled, so the argmin is over {p = 0}.
            logits = src_logits
            p_star = torch.zeros(B, self.k, device=self.device, dtype=torch.float64)
        else:
            logits, p_star = self._search(z, src_logits)

        if _should_time:
            torch.cuda.synchronize()
            self._timing_results.append((time.perf_counter() - _t0) / max(1, B))
            # Per-sample storage -> one frame is B recorded units. Required for test_time.py to
            # emit the canonical "Adaptation latency" line (see methods/ecotta.py).
            self._timing_units_per_frame = B
        self._timing_sample_counter += B

        # ---- continual init state (A6) --------------------------------------------------------
        if self.init_mode == "continual":
            self.p_prev = p_star.mean(0).detach()

        self._log_diagnostics(src_logits, logits, p_star)
        return logits

    def _search(self, z, src_logits):
        """B independent CMA-ES searches over the coefficient vector p (paper Step 2).

        Returns (logits [B, C], p_star [B, k]).
        """
        B = z.shape[0]
        mean0 = None
        if self.init_mode == "continual":
            mean0 = self.p_prev.unsqueeze(0).expand(B, self.k)
        es = BatchedCMAES(B, self.k, self.sigma0, mean0=mean0, population=self.lam,
                          device=self.device, generator=self.generator,
                          constants=self.cma_constants)

        if self.include_unadapted:
            # NON-PAPER (D7): admit p = 0 into S. Makes the returned fitness <= the source's.
            best_fit = self._entropy(src_logits).double()
            best_logits = src_logits.clone()
            best_p = torch.zeros(B, self.k, device=self.device, dtype=torch.float64)
        else:
            best_fit = torch.full((B,), float("inf"), device=self.device, dtype=torch.float64)
            best_logits = src_logits.clone()      # only a placeholder; overwritten in generation 0
            best_p = torch.zeros(B, self.k, device=self.device, dtype=torch.float64)

        rows = torch.arange(B, device=self.device)
        for _ in range(self.n_iters):
            P = es.ask()                                               # [B, lam, k] float64
            # Eq. (1): z_adapted = z_t + V_k p, for every candidate of every sample at once.
            Z = z.unsqueeze(1) + torch.matmul(P.to(z.dtype), self.V.T)  # [B, lam, D]
            logits = self.classifier(Z.reshape(B * self.lam, self.D)).reshape(B, self.lam, -1)
            self._head_evals += B * self.lam
            fit = self._entropy(logits)                                # [B, lam]
            es.tell(P, fit)

            j = fit.argmin(dim=1)                                      # [B]
            gen_fit = fit[rows, j].double()
            if self.selection == "last_gen_best":
                # Overwritten every generation, so the LAST generation's winner survives.
                best_fit, best_logits, best_p = gen_fit, logits[rows, j], P[rows, j]
            else:
                # "best_ever": arg min over all coefficients sampled in all generations (paper's
                # stated rule, spec hazard #6). "final_mean" ignores this and is handled below, but
                # we still track it so the two can be compared in one run.
                better = gen_fit < best_fit
                best_fit = torch.where(better, gen_fit, best_fit)
                best_logits = torch.where(better.unsqueeze(-1), logits[rows, j], best_logits)
                best_p = torch.where(better.unsqueeze(-1), P[rows, j], best_p)

        if self.selection == "final_mean":
            # NON-PAPER ablation: return the CMA-ES mean's prediction (one extra head eval/sample).
            best_p = es.mean
            Z = z + torch.matmul(best_p.to(z.dtype), self.V.T)
            best_logits = self.classifier(Z)
            self._head_evals += B

        return best_logits, best_p

    # ---------------------------------------------------------------------------------------------
    # diagnostics (D9)
    # ---------------------------------------------------------------------------------------------

    def _log_diagnostics(self, src_logits, logits, p_star):
        if self._num_batches % self.log_every:
            return
        h_before = self._entropy(src_logits).mean().item()
        h_after = self._entropy(logits).mean().item()
        changed = (logits.argmax(1) != src_logits.argmax(1)).float().mean().item()
        p_norm = p_star.norm(dim=1).mean().item()
        # D8's assertion-style counter: exactly one encoder CALL per batch, i.e. one encoder pass
        # per sample and no re-encoding of candidates. Head evals per sample must be
        # 1 + lambda*n_iters (+1 for final_mean). Anything else breaks the efficiency claim.
        assert self._encoder_forwards == self._num_batches, (
            f"ELaTTA ran the encoder {self._encoder_forwards} times over {self._num_batches} "
            f"batches -- the paper's cost model (1 encoder pass/sample) is violated."
        )
        head_per_sample = self._head_evals / max(1, self._encoder_samples)
        logger.info(
            f"[ELaTTA batch {self._num_batches}] entropy {h_before:.4f} -> {h_after:.4f} nats "
            f"(ln C = {math.log(self.num_classes):.4f}), mean||p*|| = {p_norm:.4f}, "
            f"prediction changed on {changed * 100:.1f}% of samples | "
            f"encoder forwards {self._encoder_forwards} = batches (1 pass/sample), "
            f"head evals {self._head_evals} ({head_per_sample:.1f}/sample, "
            f"expected {1 + self.lam * self.n_iters + (1 if self.selection == 'final_mean' else 0)})"
            + (f", p_prev norm {self.p_prev.norm().item():.4f}"
               if self.init_mode == "continual" else "")
        )
