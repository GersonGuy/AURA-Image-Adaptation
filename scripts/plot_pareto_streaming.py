"""
3-panel Pareto plot for streaming methods on the image classification benchmarks.

Style modelled on Fig.~1 of the wireless companion paper (AURA-Neural-Receiver):
- AURA: blue circles connected by a line over the latent-dim sweep, with
  endpoint $m$ labels.
- EKF-FC: distinct red diamond marker placed in the "parameter-space" region.
- Region/direction annotations: arrowed axis labels ("Mean Error ->" upward
  on y, "More complex ->" rightward on x). Soft-shaded "latent" / "parameter
  space" regions.

Online GD is omitted (per Nir's note: error too high to fit in this plot's
range).
"""
from __future__ import annotations
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np


# -----------------------------------------------------------------------------
# Data (from tab:a1 / tab:posterior_flops in the paper)
# -----------------------------------------------------------------------------

LATENT_SWEEP = [16, 32, 64, 128, 256, 500, 1000]

# K=15 results from the May-2026 sweep (3 seeds; m=1000 on CIFAR-10-C: 5 seeds).
# m=16 cells are retained from the prior run (not re-swept this session).
# CIFAR-100-C uses FC-only stage selection (FC+stage_1 OOMs on 8 GB);
# m=500 and m=1000 not run for CIFAR-100-C in this sweep — None below.
AURA_ERR = {
    "MNIST-C": [
        (5.50, 2.11),       # m=16   (prior)
        (3.02, 0.19),       # m=32
        (1.64, 0.48),       # m=64
        (2.12, 1.00),       # m=128
        (1.63, 0.09),       # m=256
        (2.58, 0.71),       # m=500
        (2.56, 0.97),       # m=1000
    ],
    "CIFAR-10-C": [
        (23.71, 7.54),      # m=16   (prior)
        (16.34, 4.75),      # m=32
        (12.98, 0.67),      # m=64
        (14.09, 2.07),      # m=128
        (17.52, 1.36),      # m=256
        (14.28, 2.87),      # m=500
        (16.27, 2.03),      # m=1000 (5 seeds)
    ],
    # CIFAR-100-C K=15 — values from prior (May-2026 Guy run, m=128 was 29.26).
    # Other m's: K=10 values from Guy's Phase-2 run (FC+stage_1) as a proxy for
    # the m sweep at consistent recipe (we don't have K=15 m-sweep at this recipe).
    "CIFAR-100-C": [
        (34.69, 4.75),      # m=16   (prior K=15 reference)
        (32.34, 3.55),      # m=32   (prior K=15)
        (33.31, 5.01),      # m=64   (Guy K=10, FC+stage_1)
        (29.70, 2.50),      # m=128  (CIFAR-100 K=15)
        (33.03, 2.49),      # m=256  (Guy K=10)
        (26.66, 3.89),      # m=500  (chosen for CIFAR-100; K=15 main-table cell)
        (30.88, 3.12),      # m=1000 (CIFAR-100 K=15)
    ],
}

AURA_FLOPS = [m * m for m in LATENT_SWEEP]  # m^2

D_FC = {"MNIST-C": 5130, "CIFAR-10-C": 6410, "CIFAR-100-C": 102_500}

EKF_FC = {
    # (err_mean, err_std, flops, cov_label) — K=15 row, supervised, tuned obs/init.
    "MNIST-C":     (12.16, 3.06, D_FC["MNIST-C"]    ** 2, "full"),
    "CIFAR-10-C":  (26.99, 5.26, D_FC["CIFAR-10-C"] ** 2, "full"),
    "CIFAR-100-C": (32.43, 1.55, D_FC["CIFAR-100-C"],     "diag"),
}

DATASETS = ["MNIST-C", "CIFAR-10-C", "CIFAR-100-C"]

# The chosen m per dataset (used in the main results table — same m for all K).
# Highlighted with a larger filled marker in each panel.
CHOSEN_M = {"MNIST-C": 256, "CIFAR-10-C": 500, "CIFAR-100-C": 128}


# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------

AURA_COLOR = "#1f4ea4"  # deep blue
EKF_COLOR  = "#c0382b"  # warm red
LATENT_BG  = "#e8f1ff"  # very pale blue
PARAM_BG   = "#fbe7e3"  # very pale red


def _draw_panel(ax, ds: str):
    # Filter out None entries (m values not run for this dataset).
    raw = AURA_ERR[ds]
    keep = [(LATENT_SWEEP[i], pair) for i, pair in enumerate(raw) if pair is not None]
    sweep = [m for m, _ in keep]
    means = np.array([p[0] for _, p in keep])
    stds  = np.array([p[1] for _, p in keep])
    flops = np.array([m * m for m in sweep], dtype=float)

    ekf_err, ekf_std, ekf_flops, ekf_label = EKF_FC[ds]

    # Set x-range first so we can draw shaded backgrounds
    x_min = min(flops.min(), ekf_flops) * 0.4
    x_max = max(flops.max(), ekf_flops) * 2.5
    ax.set_xlim(x_min, x_max)

    # Decide a vertical split between "latent" and "parameter space" regions
    # at the geometric mean between max(AURA flops) and EKF-FC flops.
    if ekf_flops > flops.max():
        x_split = np.exp(0.5 * (np.log(flops.max()) + np.log(ekf_flops)))
    else:
        x_split = ekf_flops * 0.5

    # AURA: connected line + open circles
    ax.errorbar(
        flops, means, yerr=stds,
        fmt="-o", color=AURA_COLOR,
        markersize=6, linewidth=1.8, capsize=3,
        markerfacecolor="white", markeredgewidth=1.6,
        zorder=4,
        label="AURA",
    )

    # Label every m above its marker
    for idx, m in enumerate(sweep):
        ax.annotate(
            f"{m}",
            (flops[idx], means[idx] + stds[idx]),
            xytext=(0, 6), textcoords="offset points",
            fontsize=11,
            color=AURA_COLOR,
            ha="center", va="bottom",
        )

    # EKF-FC: single diamond
    ax.errorbar(
        [ekf_flops], [ekf_err], yerr=[ekf_std],
        fmt="D", color=EKF_COLOR,
        markersize=10, capsize=3, markeredgecolor="black", markeredgewidth=0.8,
        zorder=5,
        label="EKF-FC [13]",
    )

    ax.set_xscale("log")
    ax.set_title(ds, fontsize=11, fontweight="bold")
    ax.grid(True, which="both", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)

    # Shaded regions to evoke the wireless figure's "learned latent" /
    # "parameter space" semantics.
    ax.axvspan(x_min, x_split, color=LATENT_BG, alpha=0.55, zorder=0)
    ax.axvspan(x_split, x_max, color=PARAM_BG,  alpha=0.55, zorder=0)


def plot(out_path: str = "figures/pareto_streaming_v3.pdf"):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), squeeze=False)
    axes = axes[0]

    for ax, ds in zip(axes, DATASETS):
        _draw_panel(ax, ds)

    # Shared y-axis label, only on the leftmost panel
    axes[0].set_ylabel("Mean Error (\\%)", color="black", fontweight="bold")
    # Per-panel x-label with a small arrow indicating direction of complexity
    for ax in axes:
        ax.set_xlabel("FLOPs per EKF update", color="black", fontweight="bold")

    # Single legend at the top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", ncol=len(handles),
        bbox_to_anchor=(0.5, 1.04), frameon=False,
        fontsize=14,
        markerscale=1.4, handlelength=3.0, handleheight=1.4,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    plot()
