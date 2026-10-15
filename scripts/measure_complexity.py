"""
Per-(dataset, method) wall-clock complexity measurement, modeled after Guy's
``measure_runtime`` in the wireless-receiver repo (single_run.py:134).

Times the forward path (``base_model(x)``) and the full ``forward_and_adapt``
path separately; the adapt cost is the difference. Each cell is one subprocess
for CUDA isolation. No FLOPs counter -- ``FlopCounterMode`` under-counts
``torch.func`` paths by ~2-3x, so wall-clock is the truthful number.

Usage::
    python scripts/measure_complexity.py --dataset mnist_c --method AURA
    python scripts/measure_complexity.py --aggregate
"""
from __future__ import annotations
import sys
import os
import json
import time
from statistics import mean, stdev

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conf import cfg, _C  # noqa
from models.model import get_model  # noqa
from methods import *  # noqa: F401,F403  -- registers methods
from utils.registry import ADAPTATION_REGISTRY  # noqa


DATASETS = {
    "mnist_c":    {"shape": (1, 28, 28),  "num_classes": 10},
    "cifar10_c":  {"shape": (3, 32, 32),  "num_classes": 10},
    "cifar100_c": {"shape": (3, 32, 32),  "num_classes": 100},
}

METHODS = {
    "Source":   "source.yaml",
    "BN-Test":  "norm_test.yaml",
    "BN-Alpha": "norm_alpha.yaml",
    "BN-EMA":   "norm_ema.yaml",
    "Online GD": "sgd_base.yaml",
    "ROID":     "roid.yaml",
    "EKF-FC":   "ekf_fc.yaml",
    "AURA":     "aura.yaml",
}

# Per-dataset best AURA dim from the A1 sweep.
AURA_M = {"mnist_c": 500, "cifar10_c": 1000, "cifar100_c": 128}

WARMUP_CALLS = 50
N_MEASUREMENTS = 10
DEFAULT_BATCH = 100


def reset_cfg():
    fresh = _C.clone()
    cfg.clear()
    for k, v in fresh.items():
        cfg[k] = v


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_ms(fn, n_calls: int):
    """Return list of n per-call durations in ms (with cuda sync around each)."""
    out = []
    for _ in range(n_calls):
        cuda_sync()
        t0 = time.perf_counter()
        fn()
        cuda_sync()
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def measure_one(dataset: str, method_yaml: str, method_name: str) -> dict:
    reset_cfg()
    cfg_path = os.path.join("cfgs", dataset, method_yaml)
    if not os.path.exists(cfg_path):
        return {"error": f"no cfg {cfg_path}"}
    cfg.merge_from_file(cfg_path)
    cfg.SETTING = "gradual_mixed_corruption"
    # Force the published per-frame batch size for the gradual mixed protocol;
    # individual yamls don't all set this, falling back to the default of 6.
    cfg.MIXED_GRADUAL.BATCH_SIZE = DEFAULT_BATCH

    if method_name == "AURA":
        cfg.AURA.LATENT_DIM = AURA_M[dataset]
        cfg.AURA.OFFLINE_CKPT = ""

    if method_name == "ROID" and dataset == "cifar100_c":
        # Match the CIFAR-10 ROID cfg so the comparison is apples-to-apples.
        cfg.ROID.SUPERVISED = True
        cfg.ROID.ADAPT_SAMPLES = 15

    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(cfg.MODEL.DEVICE)
    info = DATASETS[dataset]

    try:
        base_model, _ = get_model(cfg, info["num_classes"], device)
        adapt_cls = ADAPTATION_REGISTRY.get(cfg.MODEL.ADAPTATION)
        model = adapt_cls(cfg=cfg, model=base_model, num_classes=info["num_classes"])
    except Exception as e:
        return {"error": f"build: {type(e).__name__}: {e}"}

    bs = cfg.MIXED_GRADUAL.BATCH_SIZE if cfg.MIXED_GRADUAL.BATCH_SIZE > 0 else DEFAULT_BATCH
    x = torch.randn(bs, *info["shape"], device=device)
    labels = torch.randint(0, info["num_classes"], (bs,), device=device)

    # Resolve how many of the B samples are actually used for adaptation.
    # Methods with NUM_ADAPT_SAMPLES <= 0 fall back to the full batch.
    def _resolve_adapt_count():
        if method_name == "AURA":
            n = getattr(cfg.AURA, "NUM_ADAPT_SAMPLES", -1)
        elif method_name == "Online GD":
            n = getattr(cfg.SGD_BASE, "NUM_ADAPT_SAMPLES", -1)
        elif method_name == "ROID":
            n = getattr(cfg.ROID, "ADAPT_SAMPLES", -1)
        elif method_name == "EKF-FC":
            n = getattr(cfg.EKF_FC, "NUM_ADAPT_SAMPLES", -1)
        elif method_name in {"BN-Test", "BN-Alpha", "BN-EMA"}:
            # Stat-only methods: adapt is the running-stat update on the full batch.
            return bs
        elif method_name == "Source":
            return 0
        else:
            n = -1
        if not n or n <= 0:
            return bs
        return min(int(n), bs)
    n_adapt = _resolve_adapt_count()

    supervised = method_name in {"Online GD", "ROID"}

    # Two callables: pure forward through the underlying model (no adaptation
    # state changes) and the full forward_and_adapt pass.
    def do_forward():
        with torch.no_grad():
            return base_model(x)

    if supervised:
        def do_total():
            return model.forward_and_adapt([x], labels=labels)
    else:
        def do_total():
            return model(x)

    # Warmup -- exercise both paths so caches/lazy state for both are primed.
    try:
        for _ in range(WARMUP_CALLS):
            do_forward()
        for _ in range(WARMUP_CALLS):
            do_total()
    except Exception as e:
        return {"error": f"warmup: {type(e).__name__}: {e}"}

    # Wall-clock
    try:
        t_forward_ms = time_ms(do_forward, N_MEASUREMENTS)
        t_total_ms = time_ms(do_total, N_MEASUREMENTS)
    except Exception as e:
        return {"error": f"timing: {type(e).__name__}: {e}"}

    fwd_mean, fwd_std = mean(t_forward_ms), stdev(t_forward_ms)
    tot_mean, tot_std = mean(t_total_ms), stdev(t_total_ms)

    t_adapt_total = max(tot_mean - fwd_mean, 0.0)
    n_adapt_safe = max(n_adapt, 1)

    return {
        "dataset": dataset,
        "method": method_name,
        "batch_size": bs,
        "n_adapt_samples": n_adapt,           # supervised samples used per batch
        "adapt_rate": n_adapt / bs,           # fraction of batch used for adaptation
        "warmup_calls": WARMUP_CALLS,
        "n_measurements": N_MEASUREMENTS,
        "t_forward_ms": t_forward_ms,
        "t_total_ms": t_total_ms,
        "t_forward_ms_mean": fwd_mean,
        "t_forward_ms_std": fwd_std,
        "t_total_ms_mean": tot_mean,
        "t_total_ms_std": tot_std,
        "t_adapt_ms_mean": t_adapt_total,
        # Per-test-sample (= total work / batch_size). The number every test
        # frame pays. Shrinks naturally when n_adapt_samples < batch_size.
        "per_sample_ms": {
            "forward": fwd_mean / bs,
            "adapt": t_adapt_total / bs,
            "total": tot_mean / bs,
        },
        # Per-supervised-sample (= adapt work / n_adapt_samples). The cost of
        # *one* labeled-sample update, comparable across methods that use
        # different adapt-rates.
        "per_adapt_ms": {
            "adapt": t_adapt_total / n_adapt_safe,
        },
    }


def fmt_ms(v):
    return f"{v:.2f}" if v is not None else "—"


def aggregate():
    import glob
    rows = []
    for p in sorted(glob.glob("logs/complexity_cells/*.json")):
        with open(p) as f:
            rows.append(json.load(f))
    out_path = "logs/complexity_summary.json"
    os.makedirs("logs", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote {out_path}")

    # Per-sample tables for adapt and total, both ms and GFLOPs.
    by = {}
    for r in rows:
        if "error" in r:
            continue
        by.setdefault(r["method"], {})[r["dataset"]] = r

    def print_table(metric_path: list, title: str, fmt):
        print(f"\n=== {title} ===")
        print(f"{'method':<10}", *[f"{ds:>14}" for ds in DATASETS])
        for method in METHODS:
            cells = []
            for ds in DATASETS:
                r = by.get(method, {}).get(ds)
                if r is None:
                    cells.append(f"{'—':>14}")
                    continue
                v = r
                for k in metric_path:
                    v = v[k]
                cells.append(f"{fmt(v):>14}")
            print(f"{method:<10}", *cells)

    print_table(["per_sample_ms", "total"], "per-test-sample TOTAL ms (forward + adapt)", fmt_ms)
    print_table(["per_sample_ms", "adapt"], "per-test-sample ADAPT ms (total - forward)", fmt_ms)
    print_table(["per_adapt_ms", "adapt"],
                "per-LABELED-sample ADAPT ms (cost of one supervised update)", fmt_ms)

    # And a tiny adapt-rate table so it's visible.
    print(f"\n=== n_adapt_samples per batch (B=100) ===")
    print(f"{'method':<10}", *[f"{ds:>14}" for ds in DATASETS])
    for method in METHODS:
        cells = []
        for ds in DATASETS:
            r = by.get(method, {}).get(ds)
            cells.append(f"{r['n_adapt_samples'] if r else '—':>14}")
        print(f"{method:<10}", *cells)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(DATASETS.keys()))
    p.add_argument("--method", choices=list(METHODS.keys()))
    p.add_argument("--aggregate", action="store_true")
    args = p.parse_args()

    if args.aggregate:
        aggregate()
        return

    if not (args.dataset and args.method):
        p.error("need both --dataset and --method (or --aggregate)")

    os.makedirs("logs/complexity_cells", exist_ok=True)
    res = measure_one(args.dataset, METHODS[args.method], args.method)
    res.setdefault("dataset", args.dataset)
    res.setdefault("method", args.method)
    out = f"logs/complexity_cells/{args.dataset}__{args.method.replace(' ', '_')}.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    if "error" in res:
        print(f"{args.dataset:12s} {args.method:10s} ERROR: {res['error']}")
    else:
        print(f"{args.dataset:12s} {args.method:10s} "
              f"forward={res['per_sample_ms']['forward']:.3f} ms | "
              f"adapt={res['per_sample_ms']['adapt']:.3f} ms | "
              f"total={res['per_sample_ms']['total']:.3f} ms")


if __name__ == "__main__":
    main()
