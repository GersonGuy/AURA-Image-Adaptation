import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def _load_dataset_from_meta(run_dir):
    meta_path = os.path.join(run_dir, "dataset_meta.json")
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    for k in ("dataset", "dataset_name", "base_dataset", "corruption_dataset"):
        if k in meta and meta[k]:
            return meta[k]
    return None


def _load_dataset_from_config(run_dir):
    cfg_path = os.path.join(run_dir, "config.txt")
    if not os.path.isfile(cfg_path):
        return None
    with open(cfg_path) as f:
        txt = f.read()
    for line in txt.splitlines():
        line = line.strip()
        if line.startswith("DATASET:"):
            val = line.split(":", 1)[1].strip().strip("'\"")
            if val:
                return val
    return None


def _infer_dataset(run_dir):
    return _load_dataset_from_meta(run_dir) or _load_dataset_from_config(run_dir) or "unknown"


def _collect_runs(root):
    runs = []
    for method_dir in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(method_dir):
            continue
        method = os.path.basename(method_dir)
        for run_dir in sorted(glob.glob(os.path.join(method_dir, "*"))):
            if not os.path.isdir(run_dir):
                continue
            acc_path = os.path.join(run_dir, "acc_per_batch.npy")
            timing_path = os.path.join(run_dir, "timing.json")
            if not os.path.isfile(acc_path):
                continue
            try:
                acc = np.load(acc_path)
            except (OSError, ValueError):
                continue
            if acc.size == 0:
                continue
            err_pct = (1.0 - float(np.mean(acc))) * 100.0
            ms = None
            if os.path.isfile(timing_path):
                try:
                    with open(timing_path) as f:
                        timing = json.load(f)
                    # Prefer the amortized per-frame latency: it uses the same divisor
                    # (frame size) for every method, so streaming and batch-wise points
                    # are comparable. mean_ms is the raw per-recorded-unit figure, whose
                    # divisor differs per method — only a fallback for older runs.
                    ms = timing.get("latency_ms")
                    if ms is None and timing.get("per_frame_ms") and timing.get("frame_size"):
                        ms = timing["per_frame_ms"] / timing["frame_size"]
                    if ms is None:
                        ms = float(timing.get("mean_ms", np.nan))
                except (OSError, json.JSONDecodeError, TypeError, ValueError):
                    ms = None
            dataset = _infer_dataset(run_dir)
            runs.append({"method": method, "dataset": dataset, "err": err_pct, "ms": ms})
    return runs


def _aggregate(runs):
    grouped = defaultdict(lambda: {"errs": [], "mss": []})
    for r in runs:
        key = (r["dataset"], r["method"])
        grouped[key]["errs"].append(r["err"])
        if r["ms"] is not None and np.isfinite(r["ms"]):
            grouped[key]["mss"].append(r["ms"])
    out = []
    for (ds, method), v in grouped.items():
        err_mean = float(np.mean(v["errs"]))
        err_std = float(np.std(v["errs"])) if len(v["errs"]) > 1 else 0.0
        ms_mean = float(np.mean(v["mss"])) if v["mss"] else float("nan")
        out.append({
            "dataset": ds,
            "method": method,
            "err_mean": err_mean,
            "err_std": err_std,
            "ms_mean": ms_mean,
            "n": len(v["errs"]),
        })
    return out


def _plot(points, out_path):
    datasets = sorted({p["dataset"] for p in points})
    n = len(datasets)
    fig, axes = plt.subplots(1, max(n, 1), figsize=(5.0 * max(n, 1), 4.5), squeeze=False)
    axes = axes[0]
    for ax, ds in zip(axes, datasets):
        pts = [p for p in points if p["dataset"] == ds]
        for p in pts:
            ms = p["ms_mean"]
            if not np.isfinite(ms) or ms <= 0:
                continue
            ax.errorbar(ms, p["err_mean"], yerr=p["err_std"], fmt="o", capsize=3)
            ax.annotate(p["method"], (ms, p["err_mean"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("adaptation time (ms/sample, log)")
        ax.set_ylabel("error (%)")
        ax.set_title(ds)
        ax.grid(True, which="both", alpha=0.3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="ckpt/gradual_mixed_results")
    ap.add_argument("--out", default="figures/pareto.pdf")
    args = ap.parse_args()
    runs = _collect_runs(args.root)
    if not runs:
        raise SystemExit(f"no runs with acc_per_batch.npy found under {args.root}")
    agg = _aggregate(runs)
    _plot(agg, args.out)
    for p in sorted(agg, key=lambda x: (x["dataset"], x["err_mean"])):
        print(f"{p['dataset']:16s} {p['method']:14s} err={p['err_mean']:6.2f}%"
              f" ±{p['err_std']:5.2f}  ms={p['ms_mean']:8.2f}  n={p['n']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
