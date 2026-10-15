"""
CIFAR-10-C efficiency measurement: latency, peak GPU memory, FLOPs.

Sibling of ``scripts/measure_complexity.py`` (which is left untouched). Same
subprocess-per-cell CUDA-isolation contract -- run it through
``scripts/run_efficiency.sh``, never two cells at once.

What each cell measures, in this order, in one process:

  1. INFERENCE baseline on the *pristine* source model, built before the TTA
     method touches it: latency, peak memory, FLOPs.
  2. The TTA method is constructed (this may mutate/extend the model: EcoTTA
     splices meta networks in, TinyTTA adds early-exit heads).
  3. DEPLOYED-INFERENCE on the post-construction backbone (best effort; some
     methods split the model in place and this is skipped).
  4. ADAPT+PREDICT: the full ``forward_and_adapt`` path on one frame.

"Adaptation overhead" = (4) - (1) for latency and for peak memory.

FLOPs honesty: ``FlopCounterMode`` sees ATen matmul/conv dispatches. That is
faithful for plain ``nn.Module`` forward/backward, and NOT faithful for
``torch.func`` transforms (``jacrev``/``vmap``/``functional_call``), which AURA
uses -- so AURA's counted number is recorded but marked untrustworthy, and an
analytical estimate is built from the measured single-sample forward cost.

Usage::
    python scripts/measure_efficiency.py --method AURA
    python scripts/measure_efficiency.py --aggregate
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from statistics import mean, stdev

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conf import cfg, _C  # noqa: E402
from models.model import get_model  # noqa: E402
from methods import *  # noqa: F401,F403,E402  -- registers methods
from utils.registry import ADAPTATION_REGISTRY  # noqa: E402


DATASET = "cifar10_c"
SHAPE = (3, 32, 32)
NUM_CLASSES = 10
BATCH = 100

WARMUP_CALLS = 10
N_MEASUREMENTS = 20
N_MEM_CALLS = 3

CELL_DIR = "logs/efficiency_cells"

# method label -> (yaml, needs_labels, flop_counter_trustworthy, note)
METHODS = {
    "Source":  dict(yaml="source.yaml",    labels=False, flops_ok=True,
                    note="pure nn.Module forward"),
    "BN-Test": dict(yaml="norm_test.yaml", labels=False, flops_ok=True,
                    note="pure nn.Module forward; BN stat update is elementwise (no matmul/conv), "
                         "so it is invisible to the counter -- correctly ~0 FLOPs of the kind counted"),
    "EcoTTA":  dict(yaml="ecotta.yaml",    labels=False, flops_ok=True,
                    note="nn.Module forward + autograd backward; both are dispatched through ATen "
                         "and counted"),
    "TinyTTA": dict(yaml="tinytta.yaml",   labels=False, flops_ok=True,
                    note="nn.Module forward + autograd backward; both counted"),
    "AURA":    dict(yaml="aura.yaml",      labels=True,  flops_ok=False,
                    note="torch.func.jacrev + functional_call: FlopCounterMode does not see through "
                         "the functorch transform, undercounts by ~2-3x. Use the analytical estimate"),
    "ELaTTA":  dict(yaml="elatta.yaml",    labels=False, flops_ok=True,
                    note="gradient-free; the candidate evaluations are plain nn.Linear matmuls and "
                         "are counted. The CMA-ES bookkeeping (rank-mu update, eigendecomposition) "
                         "is O(k^3) with k=2 and is not counted -- negligible"),
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def reset_cfg():
    fresh = _C.clone()
    cfg.clear()
    for k, v in fresh.items():
        cfg[k] = v


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_ms(fn, n_calls):
    out = []
    for _ in range(n_calls):
        cuda_sync()
        t0 = time.perf_counter()
        fn()
        cuda_sync()
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def mem_mb(fn, n_calls=N_MEM_CALLS):
    """Peak allocated bytes attributable to one call, in MB.

    Returns (peak_mb, resident_before_mb). `peak` includes the resident model
    weights (that is what max_memory_allocated reports); `resident_before` lets
    the caller separate weights from transient activations if wanted.
    """
    if not torch.cuda.is_available():
        return None, None
    peaks, base = [], None
    for _ in range(n_calls):
        cuda_sync()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        fn()
        cuda_sync()
        peaks.append(torch.cuda.max_memory_allocated())
    return max(peaks) / 2 ** 20, base / 2 ** 20


def count_flops(fn):
    """Total counted FLOPs for one call, or an error string."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
        counter = FlopCounterMode(display=False)
        with counter:
            fn()
        return int(counter.get_total_flops()), None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def stats(xs):
    return {"mean": mean(xs), "std": stdev(xs) if len(xs) > 1 else 0.0,
            "min": min(xs), "max": max(xs), "n": len(xs)}


def real_frame(corruptions, severity, batch, device, seed=1):
    """One REAL mixed CIFAR-10-C frame: batch//len(corruptions) images per type.

    Required, not cosmetic: TinyTTA's early-exit gate and ELaTTA's entropy
    search are input-dependent, so their cost on synthetic noise is not their
    cost on data. Returns (imgs [B,3,32,32] in [0,1], labels [B]) or (None, None)
    if CIFAR-10-C is not present.
    """
    import numpy as np
    root = os.path.join(cfg.DATA_DIR, "CIFAR-10-C")
    lab_p = os.path.join(root, "labels.npy")
    if not os.path.exists(lab_p):
        return None, None
    all_lab = np.load(lab_p)
    rng = np.random.RandomState(seed)
    lo = (severity - 1) * 10000
    xs, ys = [], []
    per = batch // len(corruptions)
    counts = [per] * len(corruptions)
    for i in range(batch - per * len(corruptions)):
        counts[i] += 1
    for c, n in zip(corruptions, counts):
        p = os.path.join(root, f"{c}.npy")
        if not os.path.exists(p) or n == 0:
            return None, None
        arr = np.load(p, mmap_mode="r")
        idx = rng.choice(10000, size=n, replace=False) + lo
        xs.append(np.asarray(arr[idx]))
        ys.append(all_lab[idx])
    x = np.concatenate(xs).astype("float32") / 255.0        # [B,32,32,3] in [0,1]
    y = np.concatenate(ys).astype("int64")
    x = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous().to(device)
    return x, torch.from_numpy(y).to(device)


# --------------------------------------------------------------------------- #
# main measurement
# --------------------------------------------------------------------------- #
def measure_one(method_name: str) -> dict:
    spec = METHODS[method_name]
    reset_cfg()
    cfg_path = os.path.join("cfgs", DATASET, spec["yaml"])
    if not os.path.exists(cfg_path):
        return {"error": f"no cfg {cfg_path}"}
    cfg.merge_from_file(cfg_path)
    cfg.SETTING = "gradual_mixed_corruption"
    cfg.TEST.BATCH_SIZE = BATCH
    cfg.MIXED_GRADUAL.BATCH_SIZE = BATCH
    cfg.TIMING.ENABLED = False           # the method's own timer would only add sync overhead
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(cfg.MODEL.DEVICE)

    res = {
        "dataset": DATASET, "method": method_name, "cfg": cfg_path,
        "batch_size": BATCH, "arch": cfg.MODEL.ARCH,
        "warmup_calls": WARMUP_CALLS, "n_measurements": N_MEASUREMENTS,
        "flop_counter_trustworthy": spec["flops_ok"],
        "flop_counter_note": spec["note"],
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }

    # ---- checkpoint provenance ------------------------------------------- #
    ck = {}
    if method_name == "EcoTTA":
        p = cfg.ECOTTA.WARMUP_CKPT or (
            f"ckpt/warmup/ckpt_ecotta_{DATASET}_{cfg.MODEL.ARCH}_K{cfg.ECOTTA.K}"
            f"_bn{cfg.ECOTTA.BN_PLACEMENT}_seed{cfg.RNG_SEED}.pth")
        ck = {"kind": "ecotta_warmup", "path": p, "exists": os.path.exists(p)}
    elif method_name == "TinyTTA":
        p = cfg.TINYTTA.OFFLINE_CKPT or (
            f"ckpt/tinytta/ckpt_tinytta_{DATASET}_{cfg.MODEL.ARCH}_K{cfg.TINYTTA.K}"
            f"_ws{cfg.TINYTTA.WS_MODE}_seed{cfg.RNG_SEED}.pth")
        ck = {"kind": "tinytta_offline", "path": p, "exists": os.path.exists(p)}
    elif method_name == "ELaTTA":
        p = cfg.ELATTA.BASIS_CKPT or (
            f"ckpt/elatta/{DATASET}__{cfg.MODEL.ARCH}__k{cfg.ELATTA.K}"
            f"__N{cfg.ELATTA.N_SOURCE_SAMPLES}__seed{cfg.RNG_SEED}.pth")
        ck = {"kind": "elatta_basis", "path": p, "exists": os.path.exists(p)}
    elif method_name == "AURA":
        p = cfg.AURA.OFFLINE_CKPT
        ck = {"kind": "aura_offline", "path": p or "(cold-start)",
              "exists": bool(p) and os.path.exists(p),
              "cold_start": not bool(p)}
        if p and not os.path.exists(p):
            cfg.AURA.OFFLINE_CKPT = ""
            ck["fell_back_to_cold_start"] = True
    res["checkpoint"] = ck
    if ck and ck.get("kind") in {"ecotta_warmup", "tinytta_offline"} and not ck["exists"]:
        return {**res, "error": f"required checkpoint missing: {ck['path']}"}

    x, labels = real_frame(list(cfg.MIXED_GRADUAL.CORRUPTION_TYPES),
                           int(cfg.MIXED_GRADUAL.SEVERITY), BATCH, device,
                           seed=cfg.RNG_SEED)
    if x is None:
        x = torch.randn(BATCH, *SHAPE, device=device)
        labels = torch.randint(0, NUM_CLASSES, (BATCH,), device=device)
        res["input"] = "SYNTHETIC randn (CIFAR-10-C not found) -- input-dependent costs invalid"
    else:
        res["input"] = (f"real CIFAR-10-C mixed frame, severity {cfg.MIXED_GRADUAL.SEVERITY}, "
                        f"{len(cfg.MIXED_GRADUAL.CORRUPTION_TYPES)} types, seed {cfg.RNG_SEED}")
    x1 = x[:1].contiguous()

    # ---- 1) pristine source inference ------------------------------------ #
    try:
        base_model, _ = get_model(cfg, NUM_CLASSES, device)
    except Exception as e:  # noqa: BLE001
        return {**res, "error": f"build model: {type(e).__name__}: {e}"}

    base_model.eval()

    def infer():
        with torch.no_grad():
            return base_model(x)

    def infer1():
        with torch.no_grad():
            return base_model(x1)

    try:
        for _ in range(WARMUP_CALLS):
            infer()
        res["inference"] = {
            "latency_ms": stats(time_ms(infer, N_MEASUREMENTS)),
        }
        pk, bs_ = mem_mb(infer)
        res["inference"]["peak_mem_mb"] = pk
        res["inference"]["resident_mem_mb"] = bs_
        f, err = count_flops(infer)
        res["inference"]["flops_batch"] = f
        res["inference"]["flops_error"] = err

        for _ in range(3):
            infer1()
        f1, err1 = count_flops(infer1)
        res["inference"]["flops_single_sample"] = f1
        res["inference"]["flops_single_sample_error"] = err1
    except Exception as e:  # noqa: BLE001
        return {**res, "error": f"inference measurement: {type(e).__name__}: {e}"}

    res["source_model_params"] = sum(p.numel() for p in base_model.parameters())

    # ---- 2) build the TTA method ----------------------------------------- #
    try:
        adapt_cls = ADAPTATION_REGISTRY.get(cfg.MODEL.ADAPTATION)
        model = adapt_cls(cfg=cfg, model=base_model, num_classes=NUM_CLASSES)
    except Exception as e:  # noqa: BLE001
        return {**res, "error": f"build method: {type(e).__name__}: {e}"}

    res["trainable_params"] = int(getattr(model, "num_trainable_params", 0) or 0)
    res["total_params"] = int(getattr(model, "num_total_params", 0) or 0)

    extra = {}
    if method_name == "AURA":
        extra["latent_dim_cfg"] = cfg.AURA.LATENT_DIM
        extra["stage_selection"] = list(cfg.AURA.STAGE_SELECTION)
        extra["bn_grouping_mode"] = cfg.AURA.BN_GROUPING_MODE
        extra["num_adapt_samples"] = cfg.AURA.NUM_ADAPT_SAMPLES
        extra["can_slice"] = bool(getattr(model, "can_slice", False))
        for attr in ("latent_dims", "_meta_latent_dims_tuple"):
            v = getattr(model, attr, None)
            if v is not None:
                try:
                    extra[attr] = [int(i) for i in v]
                except Exception:  # noqa: BLE001
                    pass
        for attr in ("_meta_numels_tuple", "numels"):
            v = getattr(model, attr, None)
            if v is not None:
                try:
                    extra[attr] = [int(sum(i)) if hasattr(i, "__iter__") else int(i) for i in v]
                except Exception:  # noqa: BLE001
                    pass
        # adapted parameter count = sum over blocks of the lifted parameter dim
        try:
            extra["adapted_param_dim"] = int(sum(int(p.numel()) for p in model._phi_tuple))
        except Exception:  # noqa: BLE001
            pass
        try:
            extra["latent_dims_actual"] = [int(a.shape[0]) for a in model._A_tuple]
            extra["A_matrix_numel"] = int(sum(int(a.numel()) for a in model._A_tuple))
        except Exception:  # noqa: BLE001
            pass
    elif method_name == "ELaTTA":
        extra["k"] = cfg.ELATTA.K
        extra["n_iters"] = cfg.ELATTA.N_ITERS
        extra["population"] = int(getattr(model, "lam", 0))
        extra["feature_dim_D"] = int(getattr(model, "D", 0))
    elif method_name == "EcoTTA":
        extra["K_groups"] = cfg.ECOTTA.K
    elif method_name == "TinyTTA":
        extra["K_exits"] = cfg.TINYTTA.K
    res["method_config"] = extra

    # ---- 3) deployed-topology inference (best effort) --------------------- #
    # NOTE: use model.model, not base_model. EcoTTA's configure_model REPLACES
    # self.model with a meta-network-augmented wrapper, so base_model is still the
    # pristine WRN and would silently report the source cost here.
    deployed = getattr(model, "model", base_model)
    res["deployed_model_is_source_object"] = deployed is base_model

    def infer_post():
        with torch.no_grad():
            return deployed(x)

    try:
        for _ in range(3):
            infer_post()
        res["deployed_inference"] = {
            "latency_ms": stats(time_ms(infer_post, N_MEASUREMENTS)),
        }
        pk, _b = mem_mb(infer_post)
        res["deployed_inference"]["peak_mem_mb"] = pk
        f, err = count_flops(infer_post)
        res["deployed_inference"]["flops_batch"] = f
        res["deployed_inference"]["flops_error"] = err
    except Exception as e:  # noqa: BLE001
        res["deployed_inference"] = {"error": f"{type(e).__name__}: {e}"}

    # ---- 4) adapt + predict ---------------------------------------------- #
    if spec["labels"]:
        def adapt():
            return model.forward_and_adapt([x], labels=labels)
    else:
        def adapt():
            return model(x)

    try:
        for _ in range(WARMUP_CALLS):
            adapt()
    except Exception as e:  # noqa: BLE001
        return {**res, "error": f"adapt warmup: {type(e).__name__}: {e}"}

    try:
        lat = time_ms(adapt, N_MEASUREMENTS)
        res["adapt"] = {"latency_ms": stats(lat), "latency_ms_raw": lat}
        pk, bs_ = mem_mb(adapt)
        res["adapt"]["peak_mem_mb"] = pk
        res["adapt"]["resident_mem_mb"] = bs_
        f, err = count_flops(adapt)
        res["adapt"]["flops_batch"] = f
        res["adapt"]["flops_error"] = err
    except Exception as e:  # noqa: BLE001
        return {**res, "error": f"adapt measurement: {type(e).__name__}: {e}"}

    # TinyTTA's cost is input-dependent: record where the frame actually exited.
    if hasattr(model, "exit_stats"):
        try:
            res["exit_stats"] = model.exit_stats()
        except Exception:  # noqa: BLE001
            pass

    # ---- derived ---------------------------------------------------------- #
    inf_ms = res["inference"]["latency_ms"]["mean"]
    ad_ms = res["adapt"]["latency_ms"]["mean"]
    res["derived"] = {
        "per_frame_ms": ad_ms,
        "per_sample_ms": ad_ms / BATCH,
        "inference_only_per_frame_ms": inf_ms,
        "inference_only_per_sample_ms": inf_ms / BATCH,
        "adaptation_overhead_per_frame_ms": max(ad_ms - inf_ms, 0.0),
        "adaptation_overhead_per_sample_ms": max(ad_ms - inf_ms, 0.0) / BATCH,
        "peak_mem_mb": res["adapt"]["peak_mem_mb"],
        "inference_peak_mem_mb": res["inference"]["peak_mem_mb"],
        "mem_overhead_mb": (None if res["adapt"]["peak_mem_mb"] is None
                            else res["adapt"]["peak_mem_mb"] - res["inference"]["peak_mem_mb"]),
    }
    if res["adapt"]["flops_batch"] and res["inference"]["flops_batch"]:
        res["derived"]["flops_overhead"] = (res["adapt"]["flops_batch"]
                                            - res["inference"]["flops_batch"])
        res["derived"]["flops_ratio_vs_inference"] = (res["adapt"]["flops_batch"]
                                                      / res["inference"]["flops_batch"])
    return res


# --------------------------------------------------------------------------- #
def aggregate():
    import glob
    rows = []
    for p in sorted(glob.glob(f"{CELL_DIR}/*.json")):
        with open(p) as f:
            rows.append(json.load(f))
    os.makedirs("logs", exist_ok=True)
    with open("logs/efficiency_summary.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("Wrote logs/efficiency_summary.json")

    by = {}
    for r in rows:
        by.setdefault(r["method"], []).append(r)

    hdr = (f"{'method':<9}{'frame ms (across runs)':>26}{'/sample ms':>12}{'overhead ms':>13}"
           f"{'peak MB':>10}{'infer MB':>10}{'dMB':>9}{'GFLOPs':>12}{'trust':>7}{'params':>12}")
    print(hdr)
    for m in METHODS:
        rs = [r for r in by.get(m, []) if "error" not in r]
        if not rs:
            bad = by.get(m, [])
            print(f"{m:<9}  ERROR: {bad[0]['error'] if bad else 'not run'}")
            continue
        frames = [r["derived"]["per_frame_ms"] for r in rs]
        r0 = rs[0]
        d = r0["derived"]
        fm = mean(frames)
        spread = (max(frames) - min(frames)) / 2 if len(frames) > 1 else r0["adapt"]["latency_ms"]["std"]
        gf = r0["adapt"]["flops_batch"]
        gf = f"{gf/1e9:.2f}" if gf else "—"
        print(f"{m:<9}{fm:>17.2f}±{spread:<5.2f}(n={len(frames)})"
              f"{fm/BATCH:>12.4f}"
              f"{max(fm - d['inference_only_per_frame_ms'], 0):>13.2f}"
              f"{max(r['derived']['peak_mem_mb'] for r in rs):>10.1f}"
              f"{d['inference_peak_mem_mb']:>10.1f}"
              f"{max(r['derived']['mem_overhead_mb'] for r in rs):>9.1f}{gf:>12}"
              f"{('yes' if r0['flop_counter_trustworthy'] else 'NO'):>7}"
              f"{r0.get('trainable_params', 0):>12,}")


# --------------------------------------------------------------------------- #
# Analytical FLOPs, in the paper's convention (closed form, not a counter).
#
# scripts/plot_pareto_streaming.py's Fig.-2 x-axis is the POSTERIOR-UPDATE term
# only: AURA_FLOPS = m*m, EKF_FC = D_FC**2 (or D_FC diagonal). That term is
# reproduced verbatim below as its own line so the two are obviously the same
# metric. The cross-family total is a SUPERSET that adds forward / backward /
# Jacobian, because TinyTTA and EcoTTA have no posterior update at all and
# scoring them on the Fig.-2 axis alone would be meaningless in AURA's favour.
#
# Convention: 2 FLOPs per MAC (same as torch's FlopCounterMode).
# --------------------------------------------------------------------------- #
def analytical():
    F1 = 10_486_657_536          # measured: WRN-28-10, one 32x32 image, FlopCounterMode
    B = 100
    FB = F1 * B
    C = 10                       # classes
    G = 1e9

    def line(name, val, extra=""):
        print(f"    {name:<52}{val/G:>14.4f} GF   {extra}")

    print("Analytical FLOPs per frame (B=100, WRN-28-10, CIFAR-10-C).")
    print(f"  F_fwd(1 image)  = {F1/G:.4f} GF   (measured, pure nn.Module -> counter trustworthy)")
    print(f"  F_fwd(100)      = {FB/G:.4f} GF\n")

    print("Source")
    line("forward, B=100", FB)
    line("TOTAL", FB, "adaptation term = 0")

    print("\nBN-Test  (this repo: k=15 calibration subset, then eval-mode full batch)")
    line("phase 1 calibration forward, 15 images  = 15 * F1", 15 * F1)
    line("BN statistic update", 0, "elementwise over activations; no MAC -> not counted")
    line("phase 2 evaluation forward, B=100", FB)
    line("TOTAL", 15 * F1 + FB)
    line("  adaptation term", 15 * F1, "= 15.0% of a forward")

    print("\nEcoTTA  (K=5 meta groups, 3,484,192 trainable)")
    Feco = 1_194_942_105_600     # measured deployed forward (meta nets inserted)
    line("deployed forward (source 1048.67 GF + meta nets)", Feco,
         "= 1.139 x source forward")
    line("entropy backward ~ 2 x forward (reaches every group)", 2 * Feco)
    line("regularization pass fwd+bwd (Alg. 1, detached)", 6_120_718_899_200 - 3 * Feco,
         "= counted total - (fwd + 2x bwd)")
    line("SGD step, O(d), d=3,484,192", 5 * 3_484_192)
    line("TOTAL (analytical ~ 5-6 x forward)", 6_120_718_899_200,
         "counted 6120.72 GF = 5.12 x fwd -> use counted")
    line("  adaptation term", 6_120_718_899_200 - Feco)

    print("\nTinyTTA  (K=4 exits, 167,720 trainable; MEASURED exit dist on real data)")
    Ftiny = 529_148_672_000
    line("forward with early exit (49.4% of blocks skipped)", Ftiny,
         "= 0.505 x source forward")
    line("backward on the exited heads only", 536_333_395_200 - Ftiny, "~ 2 x head forward")
    line("SGD step, O(d), d=167,720", 5 * 167_720)
    line("TOTAL", 536_333_395_200, "counted 536.33 GF")
    line("  adaptation term", 536_333_395_200 - Ftiny)

    print("\nAURA  (m=500 per block, 2 blocks [fc d=6410, stage_1-BN d=2272], K=15 updates)")
    m, nb, K, d = 500, 2, 15, 8682
    per_call_lift = 2 * m * d                      # theta = phi + z A
    n_obs_calls = 2 + 2 * C                        # explicit fwd + jacrev(1 fwd + C vjp @ 2x)
    jac_update = n_obs_calls * F1
    post_block = 2 * m ** 3 + 2 * C * m ** 2 + 2 * m ** 2 * C + 2 * m * C * m \
        + 2 * m * C ** 2 + 2 * C ** 2 * m + C ** 3 + 2 * m * C
    post = nb * post_block
    lift = n_obs_calls * per_call_lift
    line("per update: forward + jacrev over C=10  = 22 * F1", jac_update,
         "vjp assumed 2 x fwd (grad-act + grad-param)")
    line("per update: posterior, 2 blocks", post,
         f"dominated by (I-KH)P = 2m^3 = {2*m**3/1e6:.0f} MF/block")
    line("per update: lifting z->theta", lift)
    line(f"x K={K} updates", K * (jac_update + post + lift))
    line("evaluation forward, B=100", FB)
    line("TOTAL", K * (jac_update + post + lift) + FB,
         "counter says 2945.66 GF -- UNDERCOUNT, see doc")
    line("  adaptation term", K * (jac_update + post + lift))
    print(f"    {'Fig.-2 proxy term only (m^2 per block-update)':<52}"
          f"{K * nb * m * m / G:>14.6f} GF   "
          f"= {K*nb*m*m:,} FLOPs, {K*nb*m*m/(K*(jac_update+post+lift)+FB)*100:.6f}% of the frame")
    print(f"    {'EKF-FC parameter-space counterpart, d=6410':<52}"
          f"{K * (2*6410**3) / G:>14.4f} GF   "
          f"= 2d^3 per update x {K}; Fig.-2 proxy is d^2 = {6410**2:,}")

    print("\nELaTTA  (k=2, n=2, lambda=6, D=640, C=10)")
    k, n, lam, D = 2, 2, 6, 640
    enc = FB
    src_head = B * 2 * D * C
    proj = n * B * lam * 2 * k * D
    cand = n * B * lam * 2 * D * C
    cma = n * B * (k ** 3 + lam * k ** 2)
    line("encoder forward, B=100 (identical to Source)", enc)
    line(f"source head eval  B*2*D*C", src_head)
    line(f"z + V_k p  n*B*lam*2*k*D", proj)
    line(f"candidate head evals  n*B*lam*2*D*C", cand)
    line("CMA-ES bookkeeping  n*B*(k^3 + lam*k^2)", cma)
    line("TOTAL", enc + src_head + proj + cand + cma)
    line("  adaptation term", src_head + proj + cand + cma,
         f"counted overhead 0.01845 GF")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=list(METHODS.keys()))
    p.add_argument("--aggregate", action="store_true")
    p.add_argument("--analytical", action="store_true",
                   help="print the closed-form FLOPs breakdown (the paper's convention)")
    p.add_argument("--run", type=int, default=0,
                   help="repeat index -- independent process, separate JSON. Latency has "
                        "process-to-process variance (GPU clock ramp, host jitter) that the "
                        "within-process std does not capture.")
    args = p.parse_args()

    if args.analytical:
        analytical()
        return
    if args.aggregate:
        aggregate()
        return
    if not args.method:
        p.error("need --method or --aggregate")

    os.makedirs(CELL_DIR, exist_ok=True)
    res = measure_one(args.method)
    res.setdefault("method", args.method)
    res["run"] = args.run
    suffix = "" if args.run == 0 else f"__run{args.run}"
    out = f"{CELL_DIR}/{DATASET}__{args.method.replace(' ', '_')}{suffix}.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    if "error" in res:
        print(f"{args.method:<9} ERROR: {res['error']}")
    else:
        d = res["derived"]
        print(f"{args.method:<9} frame={d['per_frame_ms']:.2f} ms | "
              f"sample={d['per_sample_ms']:.4f} ms | "
              f"overhead={d['adaptation_overhead_per_frame_ms']:.2f} ms | "
              f"peak={d['peak_mem_mb']:.1f} MB (infer {d['inference_peak_mem_mb']:.1f} MB) | "
              f"flops={res['adapt']['flops_batch']}")


if __name__ == "__main__":
    main()
