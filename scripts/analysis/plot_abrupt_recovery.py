import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)

# Style settings for publication-quality figures
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'lines.linewidth': 2.0,
    'grid.alpha': 0.3
})

def compute_summary_tables(df, output_dir):
    """Computes and exports per-corruption and overall summary CSV tables."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Per-corruption summary table
    per_corr_rows = []
    grouped = df.groupby(["dataset", "latent_dim", "lifting_type", "cov_config_id", "current_corruption"])
    
    for (dataset, latent_dim, lifting_type, cov_id, corruption), group in grouped:
        # Step 0 error (u=0)
        u0_group = group[group["updates_since_shift"] == 0]
        u0_errs = u0_group["classification_error"].values
        u0_mean, u0_std = np.mean(u0_errs), np.std(u0_errs)
        
        # Max update step in this group
        max_u = group["updates_since_shift"].max()
        u_final_group = group[group["updates_since_shift"] == max_u]
        ufinal_errs = u_final_group["classification_error"].values
        ufinal_mean, ufinal_std = np.mean(ufinal_errs), np.std(ufinal_errs)
        
        abs_red_mean = u0_mean - ufinal_mean
        rel_red_mean = (abs_red_mean / u0_mean) * 100.0 if u0_mean > 0 else 0.0
        
        row = {
            "dataset": dataset,
            "latent_dim": latent_dim,
            "lifting_type": lifting_type,
            "cov_config_id": cov_id,
            "corruption": corruption,
            "error_u0_mean": round(u0_mean, 2),
            "error_u0_std": round(u0_std, 2),
            "final_error_mean": round(ufinal_mean, 2),
            "final_error_std": round(ufinal_std, 2),
            "abs_error_reduction_mean": round(abs_red_mean, 2),
            "rel_error_reduction_mean": round(rel_red_mean, 2),
            "mean_runtime_per_2_updates_ms": round(group["runtime_preceding_2_updates_ms"].mean(), 2)
        }
        
        # Record specific update horizons (u=2, 4, 6, 8, ...)
        for u_step in sorted(group["updates_since_shift"].unique()):
            if u_step > 0:
                u_step_errs = group[group["updates_since_shift"] == u_step]["classification_error"].values
                row[f"error_u{u_step}_mean"] = round(np.mean(u_step_errs), 2)
                row[f"error_u{u_step}_std"] = round(np.std(u_step_errs), 2)
                
        per_corr_rows.append(row)
        
    df_per_corr = pd.DataFrame(per_corr_rows)
    per_corr_path = os.path.join(output_dir, "summary_per_corruption.csv")
    df_per_corr.to_csv(per_corr_path, index=False)
    logger.info(f"Saved per-corruption summary table to {per_corr_path}")

    # 2. Compact overall summary table
    overall_rows = []
    grouped_overall = df.groupby(["dataset", "latent_dim", "lifting_type", "cov_config_id"])
    
    for (dataset, latent_dim, lifting_type, cov_id), group in grouped_overall:
        u0_group = group[group["updates_since_shift"] == 0]
        max_u = group["updates_since_shift"].max()
        ufinal_group = group[group["updates_since_shift"] == max_u]
        
        u0_mean = u0_group["classification_error"].mean()
        ufinal_mean = ufinal_group["classification_error"].mean()
        abs_red = u0_mean - ufinal_mean
        rel_red = (abs_red / u0_mean) * 100.0 if u0_mean > 0 else 0.0
        
        overall_rows.append({
            "dataset": dataset,
            "latent_dim": latent_dim,
            "lifting_type": lifting_type,
            "cov_config_id": cov_id,
            "overall_mean_error": round(group["classification_error"].mean(), 2),
            "mean_initial_shift_error": round(u0_mean, 2),
            "mean_final_block_error": round(ufinal_mean, 2),
            "mean_abs_error_reduction": round(abs_red, 2),
            "mean_rel_error_reduction": round(rel_red, 2),
            "mean_update_runtime_ms": round(group["runtime_preceding_2_updates_ms"].mean(), 2)
        })
        
    df_overall = pd.DataFrame(overall_rows)
    overall_path = os.path.join(output_dir, "summary_overall.csv")
    df_overall.to_csv(overall_path, index=False)
    logger.info(f"Saved overall summary table to {overall_path}")

def plot_main_4curve_recovery(df, dataset_name, output_dir, baseline_cov_id="cov5", source_baseline_dict=None):
    """
    Generates the main abrupt-recovery plot containing ONLY the 4 core combinations:
      (500d linear, 500d nonlinear, 1000d linear, 1000d nonlinear)
    under baseline covariance configuration (cov5).
    Includes a piecewise Source baseline step curve showing initial error per corruption block.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter dataset and baseline covariance config
    ds_df = df[(df["dataset"] == dataset_name) & (df["cov_config_id"] == baseline_cov_id)]
    if ds_df.empty:
        logger.warning(f"No data found for dataset {dataset_name} and cov_config_id {baseline_cov_id}")
        return

    plt.figure(figsize=(15, 6))
    ax = plt.gca()

    # Core 4 combinations
    combos = [
        (500, "linear", "#1f77b4", "-"),
        (500, "nonlinear", "#ff7f0e", "--"),
        (1000, "linear", "#2ca02c", "-."),
        (1000, "nonlinear", "#d62728", ":")
    ]

    # Get corruption block boundaries
    corruptions = ds_df["current_corruption"].unique()
    
    # Calculate step indices per corruption block
    sample_combo_df = ds_df[(ds_df["latent_dim"] == 500) & (ds_df["lifting_type"] == "linear") & (ds_df["seed"] == 1)]
    if sample_combo_df.empty:
        sample_combo_df = ds_df.drop_duplicates(subset=["global_online_step"]).sort_values("global_online_step")

    step_mapping = sample_combo_df.sort_values("global_online_step").reset_index(drop=True)
    
    # Draw vertical boundary lines and active corruption labels
    boundary_steps = []
    block_centers = []
    
    for block_idx, corruption in enumerate(corruptions):
        block_rows = step_mapping[step_mapping["current_corruption"] == corruption]
        if not block_rows.empty:
            start_step = block_rows.index[0]
            end_step = block_rows.index[-1]
            boundary_steps.append(start_step)
            block_centers.append((start_step + end_step) / 2.0)
            
            if block_idx > 0:
                ax.axvline(x=start_step, color="red", linestyle="--", alpha=0.6, linewidth=1.5)

    # Add corruption labels at top
    y_max = ds_df["classification_error"].max() * 1.15
    for center, corruption in zip(block_centers, corruptions):
        ax.text(center, y_max * 0.92, corruption, horizontalalignment='center', verticalalignment='top', fontsize=9, fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    # Plot piecewise Source baseline step curve (no-adaptation reference)
    if source_baseline_dict:
        source_steps = []
        source_errors = []
        for block_idx, corruption in enumerate(corruptions):
            block_rows = step_mapping[step_mapping["current_corruption"] == corruption]
            if not block_rows.empty:
                start_step = block_rows.index[0]
                end_step = block_rows.index[-1]
                err_val = source_baseline_dict.get(corruption, ds_df[ds_df["current_corruption"] == corruption]["classification_error"].max())
                source_steps.extend([start_step, end_step])
                source_errors.extend([err_val, err_val])
        ax.plot(source_steps, source_errors, color="black", linestyle="--", linewidth=2.0, label="Source Baseline (No Adapt)")

    # Plot the 4 core AURA combinations
    for dim, lifting, color, linestyle in combos:
        combo_data = ds_df[(ds_df["latent_dim"] == dim) & (ds_df["lifting_type"] == lifting)]
        if combo_data.empty:
            continue
            
        # Group by updates_since_shift and current_corruption to aggregate across seeds
        pivot = combo_data.pivot_table(index="global_online_step", columns="seed", values="classification_error")
        mean_curve = pivot.mean(axis=1).values
        std_curve = pivot.std(axis=1).values
        x_steps = np.arange(len(mean_curve))
        
        label = f"AURA d={dim} ({lifting})"
        ax.plot(x_steps, mean_curve, color=color, linestyle=linestyle, label=label)
        ax.fill_between(x_steps, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.15)

    ax.set_xlabel("Cumulative Online Steps (Evaluated Every 2 Updates)")
    ax.set_ylabel("Classification Error (%)")
    ax.set_title(f"Abrupt-Shift Recovery Trajectory — {dataset_name.upper()} (Baseline Covariance)")
    ax.set_ylim(0, y_max)
    ax.grid(True)
    ax.legend(loc="lower right")

    fig_path = os.path.join(output_dir, f"abrupt_recovery_main_{dataset_name}.pdf")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.savefig(fig_path.replace(".pdf", ".png"), dpi=300)
    plt.close()
    logger.info(f"Saved main 4-curve recovery plot to {fig_path}")

def plot_cov_sweeps(df, dataset_name, output_dir):
    """Generates separate covariance sweep figures for each (latent_dim, lifting_type) pair."""
    os.makedirs(output_dir, exist_ok=True)
    
    ds_df = df[df["dataset"] == dataset_name]
    for dim in [500, 1000]:
        for lifting in ["linear", "nonlinear"]:
            subset = ds_df[(ds_df["latent_dim"] == dim) & (ds_df["lifting_type"] == lifting)]
            if subset.empty:
                continue
                
            plt.figure(figsize=(14, 6))
            ax = plt.gca()
            
            cov_ids = sorted(subset["cov_config_id"].unique())
            colors = plt.cm.tab10(np.linspace(0, 1, len(cov_ids)))
            
            for cov_id, color in zip(cov_ids, colors):
                cov_data = subset[subset["cov_config_id"] == cov_id]
                pivot = cov_data.pivot_table(index="global_online_step", columns="seed", values="classification_error")
                mean_curve = pivot.mean(axis=1).values
                x_steps = np.arange(len(mean_curve))
                
                init_c = cov_data["init_cov"].iloc[0]
                obs_c = cov_data["obs_cov"].iloc[0]
                label = f"{cov_id} (init={init_c}, obs={obs_c})"
                ax.plot(x_steps, mean_curve, color=color, label=label)

            ax.set_xlabel("Cumulative Online Steps")
            ax.set_ylabel("Classification Error (%)")
            ax.set_title(f"Covariance Sweep — {dataset_name.upper()} (d={dim}, {lifting})")
            ax.grid(True)
            ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            
            fig_path = os.path.join(output_dir, f"cov_sweep_{dataset_name}_d{dim}_{lifting}.pdf")
            plt.tight_layout()
            plt.savefig(fig_path, dpi=300)
            plt.close()
            logger.info(f"Saved covariance sweep figure to {fig_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Abrupt Recovery Plotting & Summary Generator")
    parser.add_argument("--csv", type=str, required=True, help="Path to abrupt_recovery_detailed.csv")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save figures and summary tables")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv):
        logger.error(f"Input CSV file non-existent: {args.csv}")
        sys.exit(1)
        
    df = pd.read_csv(args.csv)
    
    # 1. Compute summary tables
    compute_summary_tables(df, args.output_dir)
    
    # 2. Generate plots per dataset
    datasets = df["dataset"].unique()
    for ds in datasets:
        plot_main_4curve_recovery(df, ds, args.output_dir)
        plot_cov_sweeps(df, ds, args.output_dir)
