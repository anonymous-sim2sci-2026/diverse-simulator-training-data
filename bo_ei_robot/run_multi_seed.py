#!/usr/bin/env python3
"""
Run bo_cdf_direct_gp_robot1 over multiple random seeds and summarize stability.

This repeats the direct-GP baseline experiment for several stochastic rollouts,
then aggregates direct-GP-vs-random success counts across seeds.
"""

import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import Config
from run_bo import run_single


def _parse_seeds(seed_text):
    """Parse comma-separated seeds or ranges like 1,2,5-8."""
    seeds = []
    for part in seed_text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, stop = part.split("-", 1)
            seeds.extend(range(int(start), int(stop) + 1))
        else:
            seeds.append(int(part))
    return seeds


def _plot_multi_seed_summary(metrics_df, output_root):
    """Plot mean +/- std performance across seeds."""
    grouped = metrics_df.groupby("iteration")
    iterations = np.array(sorted(metrics_df["iteration"].unique()))

    bo_cum_mean = grouped["bo_cumulative_positive"].mean().to_numpy()
    bo_cum_std = grouped["bo_cumulative_positive"].std(ddof=0).fillna(0).to_numpy()
    random_cum_mean = grouped["random_cumulative_positive"].mean().to_numpy()
    random_cum_std = grouped["random_cumulative_positive"].std(ddof=0).fillna(0).to_numpy()

    bo_rate_mean = grouped["bo_success_rate"].mean().to_numpy()
    bo_rate_std = grouped["bo_success_rate"].std(ddof=0).fillna(0).to_numpy()
    random_rate_mean = grouped["random_success_rate"].mean().to_numpy()
    random_rate_std = grouped["random_success_rate"].std(ddof=0).fillna(0).to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(iterations, bo_cum_mean, marker="o", linewidth=2.5, label="Direct GP EI")
    axes[0].fill_between(
        iterations, bo_cum_mean - bo_cum_std, bo_cum_mean + bo_cum_std,
        alpha=0.20
    )
    axes[0].plot(
        iterations, random_cum_mean, marker="s", linewidth=2.5,
        label="Random", color="tab:orange"
    )
    axes[0].fill_between(
        iterations, random_cum_mean - random_cum_std,
        random_cum_mean + random_cum_std, alpha=0.20, color="tab:orange"
    )
    axes[0].set_xlabel("Iteration / measured sample")
    axes[0].set_ylabel("Cumulative +1 labels")
    axes[0].set_title("Mean detections across random seeds")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(iterations, bo_rate_mean, marker="o", linewidth=2.5, label="Direct GP EI")
    axes[1].fill_between(
        iterations, bo_rate_mean - bo_rate_std, bo_rate_mean + bo_rate_std,
        alpha=0.20
    )
    axes[1].plot(
        iterations, random_rate_mean, marker="s", linewidth=2.5,
        label="Random", color="tab:orange"
    )
    axes[1].fill_between(
        iterations, random_rate_mean - random_rate_std,
        random_rate_mean + random_rate_std, alpha=0.20, color="tab:orange"
    )
    axes[1].set_xlabel("Iteration / measured sample")
    axes[1].set_ylabel("Success rate")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_title("Mean success rate across random seeds")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_root, "multi_seed_success_summary.png")
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def run_multi_seed(
    seeds,
    n_iterations=30,
    acquisition="ROBOT_DIRECT_EI",
    output_root=None,
    robot_distance_mode="tail_l1",
    robot_tau_quantile=85.0,
    robot_rarity_n_bins=20,
):
    """Run one closed-loop rollout per seed and aggregate benchmark metrics."""
    if acquisition != "ROBOT_DIRECT_EI":
        raise ValueError(
            "This paper-facing folder only supports --acquisition ROBOT_DIRECT_EI."
        )
    if robot_distance_mode != "tail_l1":
        raise ValueError(
            "This paper-facing folder only supports --robot-distance-mode tail_l1."
        )

    config = Config()
    if output_root is None:
        timestamp = datetime.now().strftime("%m%d%y_%H%M%S")
        output_root = os.path.join(config.OUTPUT_BASE, f"multi_seed_direct_gp_{timestamp}")
    os.makedirs(output_root, exist_ok=True)

    all_metrics = []
    cost_summaries = []
    summary_rows = []

    for seed in seeds:
        print("\n" + "=" * 80)
        print(f"MULTI-SEED DIRECT-GP RUN: seed={seed}")
        print("=" * 80)

        seed_output = os.path.join(output_root, f"seed_{seed}")
        result = run_single(
            seed=seed,
            n_iterations=n_iterations,
            acquisition=acquisition,
            output_folder=seed_output,
            make_plots=True,
            robot_distance_mode=robot_distance_mode,
            robot_tau_quantile=robot_tau_quantile,
            robot_rarity_n_bins=robot_rarity_n_bins,
        )

        metrics_path = os.path.join(result["plots_folder"], "benchmark_metrics.csv")
        if not os.path.exists(metrics_path):
            print(f"Warning: missing benchmark metrics for seed {seed}: {metrics_path}")
            continue

        metrics = pd.read_csv(metrics_path)
        metrics.insert(0, "seed", seed)
        metrics.insert(1, "output_folder", result["output_folder"])
        metrics.insert(2, "plots_folder", result["plots_folder"])
        all_metrics.append(metrics)

        cost_path = os.path.join(result["plots_folder"], "computational_benefit_summary.csv")
        cost_df = pd.DataFrame()
        if os.path.exists(cost_path):
            cost_df = pd.read_csv(cost_path)
            cost_df.insert(0, "seed", seed)
            cost_df.insert(1, "output_folder", result["output_folder"])
            cost_df.insert(2, "plots_folder", result["plots_folder"])
            cost_summaries.append(cost_df)
        else:
            print(f"Warning: missing computational benefit summary for seed {seed}: {cost_path}")

        last = metrics.iloc[-1]
        cost_last = cost_df.iloc[0] if not cost_df.empty else pd.Series(dtype=float)
        summary_rows.append({
            "seed": seed,
            "output_folder": result["output_folder"],
            "plots_folder": result["plots_folder"],
            "direct_gp_detections": int(last["bo_cumulative_positive"]),
            "random_detections": int(last["random_cumulative_positive"]),
            "direct_gp_success_rate": float(last["bo_success_rate"]),
            "random_success_rate": float(last["random_success_rate"]),
            "direct_gp_final_h_kl_sum": float(last.get("bo_h_kl_sum", np.nan)),
            "random_final_h_kl_sum": float(last.get("random_h_kl_sum", np.nan)),
            "direct_gp_final_distance_kl": float(last.get("bo_distance_kl", np.nan)),
            "random_final_distance_kl": float(last.get("random_distance_kl", np.nan)),
            "robot_distance_mode": robot_distance_mode,
            "robot_tau_quantile": robot_tau_quantile,
            "mean_simulation_seconds": float(cost_last.get("mean_simulation_seconds_X", np.nan)),
            "mean_gp_refit_seconds": float(cost_last.get("mean_gp_refit_seconds_Y", np.nan)),
            "mean_bo_overhead_seconds": float(cost_last.get("mean_acquisition_robot_overhead_seconds_Z", np.nan)),
            "direct_gp_total_measured_cost_seconds_X_plus_Y_plus_Z": float(
                cost_last.get("bo_total_measured_cost_seconds_X_plus_Y_plus_Z", np.nan)
            ),
            "random_simulations_needed_to_match_direct_gp_detections_estimate": float(
                cost_last.get("random_simulations_needed_to_match_bo_detections_estimate", np.nan)
            ),
            "random_detection_match_cost_seconds_estimate": float(
                cost_last.get("random_detection_match_cost_seconds_estimate", np.nan)
            ),
            "best_peak_magnitude": (
                None if result["best_peak_mag"] == np.inf else result["best_peak_mag"]
            ),
        })

    if not all_metrics:
        raise RuntimeError("No benchmark metrics were produced; cannot summarize seeds.")

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    metrics_path = os.path.join(output_root, "multi_seed_iteration_metrics.csv")
    summary_path = os.path.join(output_root, "multi_seed_summary.csv")
    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    cost_summary_path = None
    if cost_summaries:
        cost_summary_df = pd.concat(cost_summaries, ignore_index=True)
        cost_summary_path = os.path.join(output_root, "multi_seed_computational_benefit_summary.csv")
        cost_summary_df.to_csv(cost_summary_path, index=False)
    plot_path = _plot_multi_seed_summary(metrics_df, output_root)

    print("\n" + "=" * 80)
    print("MULTI-SEED DIRECT-GP SUMMARY")
    print("=" * 80)
    print(f"Seeds: {seeds}")
    print(f"Mean direct-GP detections: {summary_df['direct_gp_detections'].mean():.2f} "
          f"+/- {summary_df['direct_gp_detections'].std(ddof=0):.2f}")
    print(f"Mean random detections: {summary_df['random_detections'].mean():.2f} "
          f"+/- {summary_df['random_detections'].std(ddof=0):.2f}")
    print(f"Saved per-iteration metrics: {metrics_path}")
    print(f"Saved per-seed summary: {summary_path}")
    if cost_summary_path:
        print(f"Saved computational benefit summary: {cost_summary_path}")
    print(f"Saved aggregate plot: {plot_path}")

    return summary_df, metrics_df


def main():
    parser = argparse.ArgumentParser(description="Repeat bo_cdf_direct_gp_robot1 over multiple random seeds.")
    parser.add_argument(
        "--seeds",
        default="1-10",
        help="Comma-separated seeds and/or ranges, e.g. '1-10' or '1,2,5,20'.",
    )
    parser.add_argument("--n-iterations", type=int, default=30)
    parser.add_argument(
        "--acquisition",
        choices=["ROBOT_DIRECT_EI"],
        default="ROBOT_DIRECT_EI",
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--robot-distance-mode",
        choices=["tail_l1"],
        default="tail_l1",
        help="ROBOT h-space diversity distance mode for ROBOT_DIRECT_EI.",
    )
    parser.add_argument(
        "--robot-tau-quantile",
        type=float,
        default=85.0,
        help="Percentile of useful-event pairwise h-space distances used as tau.",
    )
    args = parser.parse_args()

    run_multi_seed(
        seeds=_parse_seeds(args.seeds),
        n_iterations=args.n_iterations,
        acquisition=args.acquisition,
        output_root=args.output_root,
        robot_distance_mode=args.robot_distance_mode,
        robot_tau_quantile=args.robot_tau_quantile,
    )


if __name__ == "__main__":
    main()
