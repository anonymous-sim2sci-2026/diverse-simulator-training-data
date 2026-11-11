"""
Visualization utilities
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional, List

from utils.h_columns import H_OUTPUT_COLUMNS, MAGNITUDE_COLUMNS


def plot_optimization_results(results, new_events_df, X_train, y_train_detection, 
                              output_folder='.', detection_filters=None):
    """Plot optimization results"""
    if detection_filters is None:
        detection_filters = ['g', 'r', 'i', 'u', 'z', 'y']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax1 = axes[0, 0]
    ax1.plot(results['probs_history'], 'b-', linewidth=2, label='Best detection probability')
    ax1.scatter(range(len(results['probs_all'])), results['probs_all'],
                alpha=0.3, s=20, label='All evaluated points')
    ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Detection threshold')
    ax1.set_xlabel('Iteration'); ax1.set_ylabel('Detection Probability')
    ax1.set_title('Bayesian Optimization Progress'); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ax2.hist(results['probs_all'], bins=30, alpha=0.7, edgecolor='black')
    ax2.axvline(x=0.5, color='r', linestyle='--', label='Threshold')
    ax2.set_xlabel('Detection Probability'); ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Detection Probabilities'); ax2.legend(); ax2.grid(True, alpha=0.3)

    ax3 = axes[0, 2]
    for filter_name in detection_filters:
        if filter_name in new_events_df.columns:
            ax3.hist(new_events_df[filter_name].dropna(), bins=20, alpha=0.5,
                      label=f'Filter {filter_name}', density=True)
    ax3.set_xlabel('Magnitude'); ax3.set_ylabel('Density')
    ax3.set_title('Magnitude Distribution of Generated Events'); ax3.legend(); ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 0]
    scatter = ax4.scatter(new_events_df['mass1'], new_events_df['mass2'],
                           c=new_events_df['detection_probability'], cmap='viridis', alpha=0.6, s=30)
    ax4.set_xlabel('Mass1'); ax4.set_ylabel('Mass2')
    ax4.set_title('Detection Probability in Mass Space')
    plt.colorbar(scatter, ax=ax4, label='Detection Probability'); ax4.grid(True, alpha=0.3)

    ax5 = axes[1, 1]
    ax5.scatter(new_events_df['inclination'], new_events_df['detection_probability'],
                alpha=0.5, s=20, c=new_events_df['detection_probability'], cmap='viridis')
    ax5.set_xlabel('Inclination (radians)'); ax5.set_ylabel('Detection Probability')
    ax5.set_title('Detection Probability vs Inclination'); ax5.grid(True, alpha=0.3)

    ax6 = axes[1, 2]
    train_detected = y_train_detection == 1
    train_undetected = y_train_detection == 0
    ax6.scatter(X_train[train_detected, 0], X_train[train_detected, 1],
                c='green', alpha=0.3, s=20, label='Training Detected')
    ax6.scatter(X_train[train_undetected, 0], X_train[train_undetected, 1],
                c='red', alpha=0.3, s=20, label='Training Undetected')
    new_detected = new_events_df[new_events_df['detected'] == 1]
    new_undetected = new_events_df[new_events_df['detected'] == 0]
    ax6.scatter(new_detected['mass1'], new_detected['mass2'],
                c='blue', alpha=0.7, s=30, marker='*', label='New Detected')
    ax6.scatter(new_undetected['mass1'], new_undetected['mass2'],
                c='gray', alpha=0.3, s=30, marker='x', label='New Undetected')
    ax6.set_xlabel('Mass1'); ax6.set_ylabel('Mass2')
    ax6.set_title('Comparison: Training vs New Events'); ax6.legend(); ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_folder, 'bayesian_optimization_results.png')
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  - Plot saved as: {plot_path}")


def plot_h_space_useful_samples(
    y_train_mags,
    y_train_detection,
    new_events_df,
    detection_calc,
    output_folder='.',
    detection_filters=None,
):
    """
    Plot useful samples in the full 11D h-space rather than x-space.

    The lower triangle shows pairwise projections of the h-vector. Training
    points are colored by continuous ZTF-CDF P_detect, and new BO samples are
    star markers with the same color scale.
    """
    h_cols = [col for col in H_OUTPUT_COLUMNS if col in y_train_mags and col in new_events_df.columns]
    if len(h_cols) < 2:
        print("  - Skipping h-space useful-samples plot: fewer than two shared h columns.")
        return

    train_h = {}
    n_train = None
    for col in h_cols:
        values = y_train_mags[col]
        arr = values.to_numpy(dtype=float) if hasattr(values, "to_numpy") else np.asarray(values, dtype=float)
        train_h[col] = arr
        n_train = len(arr) if n_train is None else min(n_train, len(arr))

    if n_train is None or n_train == 0:
        print("  - Skipping h-space useful-samples plot: empty training magnitudes.")
        return

    train_probs = np.zeros(n_train, dtype=float)
    for i in range(n_train):
        mags = {
            filt: float(train_h[filt][i])
            for filt in MAGNITUDE_COLUMNS
            if filt in train_h and np.isfinite(train_h[filt][i])
        }
        train_probs[i], _ = detection_calc.compute_detection_probability(mags)

    new_probs = new_events_df.get(
        "detection_probability",
        pd.Series(np.nan, index=new_events_df.index),
    ).to_numpy(dtype=float)

    n_dim = len(h_cols)
    fig, axes = plt.subplots(n_dim, n_dim, figsize=(2.2 * n_dim, 2.2 * n_dim))
    axes = np.asarray(axes)
    scatter = None

    for row, y_col in enumerate(h_cols):
        for col, x_col in enumerate(h_cols):
            ax = axes[row, col]
            if row < col:
                ax.axis('off')
                continue

            if row == col:
                valid = np.isfinite(train_h[x_col][:n_train])
                ax.hist(train_h[x_col][:n_train][valid], bins=24, color='0.75', alpha=0.8)
                if x_col in new_events_df:
                    new_vals = new_events_df[x_col].dropna().to_numpy(dtype=float)
                    if len(new_vals):
                        ax.scatter(new_vals, np.full(len(new_vals), ax.get_ylim()[1] * 0.08),
                                   marker='*', s=45, c='tab:blue', edgecolors='black', linewidths=0.3)
            else:
                valid_train = np.isfinite(train_h[x_col][:n_train]) & np.isfinite(train_h[y_col][:n_train])
                scatter = ax.scatter(
                    train_h[x_col][:n_train][valid_train],
                    train_h[y_col][:n_train][valid_train],
                    c=train_probs[valid_train],
                    cmap='YlOrRd',
                    vmin=0.0,
                    vmax=1.0,
                    s=8,
                    alpha=0.35,
                )

                new_valid = new_events_df[x_col].notna().to_numpy() & new_events_df[y_col].notna().to_numpy()
                if np.any(new_valid):
                    ax.scatter(
                        new_events_df.loc[new_valid, x_col],
                        new_events_df.loc[new_valid, y_col],
                        c=new_probs[new_valid],
                        cmap='YlOrRd',
                        vmin=0.0,
                        vmax=1.0,
                        s=65,
                        marker='*',
                        edgecolors='black',
                        linewidths=0.35,
                    )

            if x_col in MAGNITUDE_COLUMNS:
                ax.invert_xaxis()
            if row != col and y_col in MAGNITUDE_COLUMNS:
                ax.invert_yaxis()

            ax.grid(True, alpha=0.18)
            if row == n_dim - 1:
                ax.set_xlabel(x_col, fontsize=8, rotation=45, ha='right')
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel(y_col, fontsize=8)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=6)

    colorbar_axes = list(axes.flat)
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=colorbar_axes, shrink=0.65)
    else:
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(cmap='YlOrRd'),
            ax=colorbar_axes,
            shrink=0.65,
        )
    cbar.set_label('P_detect from ZTF CDF')

    fig.suptitle('Useful Samples in 11D h(x): training probability field + new BO samples', y=0.995)
    plt.subplots_adjust(left=0.06, right=0.90, bottom=0.08, top=0.96, wspace=0.08, hspace=0.08)
    plot_path = os.path.join(output_folder, 'h_space_useful_samples.png')
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  - h-space useful-samples plot saved as: {plot_path}")


def plot_parameter_exploration(results, best_X, output_folder='.', feature_columns=None):
    """Plot detection margin vs each parameter"""
    if feature_columns is None:
        feature_columns = ['mass1', 'mass2', 'spin1z', 'spin2z', 'inclination', 'distance']
    
    X_all = np.array(results['X_all'])
    margin_all = np.array(results['margin_all'], dtype=float)
    source = np.array(results['source'])
    is_train = source == 'training'
    is_bo = source == 'bo_candidate'

    n_feat = len(feature_columns)
    ncols = 3
    nrows = int(np.ceil(n_feat / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)

    for j, name in enumerate(feature_columns):
        ax = axes[j]
        ax.scatter(X_all[is_train, j], margin_all[is_train], s=15, alpha=0.35,
                   c='gray', label='Training events')
        ax.scatter(X_all[is_bo, j], margin_all[is_bo], s=28, alpha=0.75,
                   c='tab:blue', edgecolors='k', linewidths=0.3, label='BO-proposed candidates')
        ax.axvline(best_X[j], color='red', linestyle='--', linewidth=2, label='Final best point')
        ax.set_xlabel(name)
        ax.set_ylabel('Detection margin (sigma)')
        ax.set_title(f'{name}: where BO searched')
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8, loc='best')

    for k in range(n_feat, len(axes)):
        axes[k].axis('off')

    plt.tight_layout()
    plot_path = os.path.join(output_folder, 'parameter_exploration.png')
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  - Parameter exploration plot saved as: {plot_path}")


def plot_objective_improvement(results, output_folder='.'):
    """Plot objective improvement over iterations"""
    score_history = results['score_history']
    improved = results['improved_iteration']

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.step(range(len(score_history)), score_history, where='post',
            color='tab:blue', linewidth=2, label='Best detection margin so far')

    for k, it in enumerate(improved):
        ax.scatter(it, score_history[it], color='red', zorder=5, s=60,
                   label='New best accepted' if k == 0 else None)

    ax.set_xlabel('BO Iteration')
    ax.set_ylabel('Best detection margin (sigma)')
    n_improvements = len(improved)
    total_gain = score_history[-1] - score_history[0]
    ax.set_title(f'Objective improvement over BO iterations '
                  f'({n_improvements} improving steps, total gain {total_gain:+.3f} sigma)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_folder, 'objective_improvement.png')
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  - Objective improvement plot saved as: {plot_path}")


def plot_true_evaluation_history(true_eval_log, output_folder='.'):
    """Plot real simulator results across closed-loop iterations"""
    if not true_eval_log:
        return

    log_df = pd.DataFrame(true_eval_log)
    fig, ax = plt.subplots(figsize=(9, 5))

    detected_mask = log_df['true_detectable'] == 1
    ax.scatter(log_df.loc[detected_mask, 'iteration'], log_df.loc[detected_mask, 'true_peak_mag'],
               color='tab:green', s=60, label='True detection', zorder=5)
    ax.scatter(log_df.loc[~detected_mask, 'iteration'], log_df.loc[~detected_mask, 'true_peak_mag'],
               color='tab:red', s=60, marker='x', label='True non-detection', zorder=5)

    running_best = log_df['true_peak_mag'].where(log_df['true_detectable'] == 1).cummin()
    ax.plot(log_df['iteration'], running_best, color='tab:blue', linewidth=2,
            linestyle='--', label='Best TRUE peak mag so far')

    ax.set_xlabel('Closed-loop iteration')
    ax.set_ylabel('True peak magnitude (simulator)')
    ax.invert_yaxis()
    ax.set_title('Real simulator results across closed-loop iterations')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_folder, 'true_simulation_history.png')
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  - True simulation history plot saved as: {plot_path}")


def _kl_to_uniform(values, bin_edges):
    """Distance-bin coverage divergence from the empirical distribution to uniform."""
    counts, _ = np.histogram(values, bins=bin_edges)
    total = counts.sum()
    if total == 0:
        return np.nan, counts

    p = counts.astype(float) / total
    q = np.ones_like(p) / len(p)
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask]))), counts


def _format_bin_labels(bin_edges):
    return [f"{bin_edges[i]:.0f}-{bin_edges[i + 1]:.0f}" for i in range(len(bin_edges) - 1)]


def _simulate_random_baseline(
    n_new,
    bounds,
    gp_models,
    scaler_X,
    flat_limits,
    detection_calc,
    feature_columns,
    seed=12345,
):
    """Uniform-random candidate track with stochastic labels and predicted h."""
    from models import predict_magnitudes_batch
    from optimization.objectives import objective_function_probability

    rng = np.random.default_rng(seed)
    lows = np.array([b[0] for b in bounds], dtype=float)
    highs = np.array([b[1] for b in bounds], dtype=float)
    X_random = lows + rng.random((n_new, len(bounds))) * (highs - lows)

    probs = np.array([
        objective_function_probability(
            x, gp_models, scaler_X, flat_limits,
            detection_gp=None, use_gp_detection=False, detection_calc=detection_calc
        )
        for x in X_random
    ])
    labels = (rng.random(n_new) < probs).astype(int)

    random_df = pd.DataFrame(X_random, columns=feature_columns)
    random_df["detection_probability"] = probs
    random_df["detected"] = labels
    random_df["iteration"] = np.arange(n_new)

    # Random points are not sent through NMMA in this inexpensive baseline.
    # Save the same surrogate h outputs used to evaluate their detectability so
    # their predicted h-space coverage can be evaluated without a second run.
    predicted_h, _ = predict_magnitudes_batch(X_random, gp_models, scaler_X)
    for name, values in predicted_h.items():
        values = np.asarray(values, dtype=float)
        if len(values) == n_new:
            random_df[name] = values
    return random_df


def plot_active_search_benchmark(
    true_eval_log,
    new_events_df,
    X_train,
    y_train_detection,
    gp_models,
    scaler_X,
    flat_limits,
    bounds,
    feature_columns,
    detection_calc,
    output_folder='.',
    random_seed=12345,
):
    """
    Compare BO/active-search sampling against a uniform-random rollout baseline.

    Outputs:
      - active_search_success_comparison.png
      - distance_kl_comparison.png
      - input_space_new_points.png
      - random_baseline_samples.csv
      - benchmark_metrics.csv
    """
    if not true_eval_log:
        return

    os.makedirs(output_folder, exist_ok=True)

    bo_log = pd.DataFrame(true_eval_log).copy()
    n_new = len(bo_log)
    if n_new == 0:
        return

    random_df = _simulate_random_baseline(
        n_new, bounds, gp_models, scaler_X, flat_limits,
        detection_calc, feature_columns, seed=random_seed
    )

    bo_label_col = "active_search_detectable" if "active_search_detectable" in bo_log.columns else "true_detectable"
    bo_labels = bo_log[bo_label_col].fillna(0).astype(int).to_numpy()
    random_labels = random_df["detected"].astype(int).to_numpy()

    bo_probs = bo_log.get("active_search_detection_probability")
    if bo_probs is None:
        bo_probs = bo_log.get("detection_probability", pd.Series(np.nan, index=bo_log.index))
    bo_probs = np.asarray(bo_probs, dtype=float)
    random_probs = random_df["detection_probability"].to_numpy(dtype=float)

    iterations = np.arange(1, n_new + 1)
    bo_cum = np.cumsum(bo_labels)
    random_cum = np.cumsum(random_labels)
    bo_rate = bo_cum / iterations
    random_rate = random_cum / iterations

    metrics = pd.DataFrame({
        "iteration": iterations,
        "bo_label": bo_labels,
        "random_label": random_labels,
        "bo_cumulative_positive": bo_cum,
        "random_cumulative_positive": random_cum,
        "bo_success_rate": bo_rate,
        "random_success_rate": random_rate,
        "bo_detection_probability": bo_probs,
        "random_detection_probability": random_probs,
    })

    random_df.to_csv(os.path.join(output_folder, "random_baseline_samples.csv"), index=False)
    metrics.to_csv(os.path.join(output_folder, "benchmark_metrics.csv"), index=False)

    # Success-rate / positive-label plot.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].step(iterations, bo_cum, where="post", linewidth=2.5, label="BO / active search")
    axes[0].step(iterations, random_cum, where="post", linewidth=2.5, label="Random", color="tab:orange")
    axes[0].set_xlabel("Iteration (new measured sample)")
    axes[0].set_ylabel("Cumulative +1 labels")
    axes[0].set_title("How often each track finds detectable events")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(iterations, bo_rate, marker="o", linewidth=2, label="BO / active search")
    axes[1].plot(iterations, random_rate, marker="s", linewidth=2, label="Random", color="tab:orange")
    axes[1].set_xlabel("Iteration (new measured sample)")
    axes[1].set_ylabel("Cumulative success rate")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_title("Detection success rate vs budget")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_folder, "active_search_success_comparison.png")
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  - Active-search benchmark plot saved as: {plot_path}")

    # Distance-bin coverage diagnostic similar to the reference figure.
    if "distance" in feature_columns:
        dist_idx = feature_columns.index("distance")
        initial_distance = np.asarray(X_train)[:, dist_idx]
        bo_distance = bo_log["distance"].to_numpy(dtype=float) if "distance" in bo_log else new_events_df.get("distance", pd.Series(dtype=float)).to_numpy()
        random_distance = random_df["distance"].to_numpy(dtype=float)

        lo, hi = bounds[dist_idx]
        bin_edges = np.linspace(lo, hi, 14)
        labels = _format_bin_labels(bin_edges)
        xbins = np.arange(len(labels))

        initial_kl, initial_counts = _kl_to_uniform(initial_distance, bin_edges)
        bo_final_kl, bo_final_counts = _kl_to_uniform(np.concatenate([initial_distance, bo_distance]), bin_edges)
        random_final_kl, random_final_counts = _kl_to_uniform(np.concatenate([initial_distance, random_distance]), bin_edges)

        bo_kl_history = []
        random_kl_history = []
        for k in range(n_new + 1):
            bo_kl_history.append(_kl_to_uniform(np.concatenate([initial_distance, bo_distance[:k]]), bin_edges)[0])
            random_kl_history.append(_kl_to_uniform(np.concatenate([initial_distance, random_distance[:k]]), bin_edges)[0])

        bo_delta_kl = np.array(bo_kl_history[:-1]) - np.array(bo_kl_history[1:])
        random_delta_kl = np.array(random_kl_history[:-1]) - np.array(random_kl_history[1:])

        bo_new_counts, _ = np.histogram(bo_distance, bins=bin_edges)
        random_new_counts, _ = np.histogram(random_distance, bins=bin_edges)

        fig, axes = plt.subplots(2, 3, figsize=(19, 10))
        panels = [
            ("Initial", initial_counts, initial_kl, len(initial_distance)),
            ("BO Final", bo_final_counts, bo_final_kl, len(initial_distance) + len(bo_distance)),
            ("Random Final", random_final_counts, random_final_kl, len(initial_distance) + len(random_distance)),
        ]

        for ax, (title, counts, kl_val, n_total) in zip(axes[0], panels):
            ax.bar(xbins, counts, color="tab:blue", alpha=0.85, edgecolor="white")
            ax.axhline(np.mean(counts), color="tab:blue", linestyle="--", linewidth=1.8, alpha=0.8)
            empty_bins = int(np.sum(counts == 0))
            ax.text(0.01, 0.93, f"Empty bins: {empty_bins}", transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.5", alpha=0.9))
            ax.set_title(f"{title} (n={n_total})\nDistance coverage divergence={kl_val:.4f}")
            ax.set_xlabel("Distance bin")
            ax.set_ylabel("Count")
            ax.set_xticks(xbins)
            ax.set_xticklabels(labels, rotation=55, ha="right")
            ax.grid(True, axis="y", alpha=0.25)

        axes[1, 0].plot(range(n_new + 1), bo_kl_history, marker="o", linewidth=2, label="BO")
        axes[1, 0].plot(range(n_new + 1), random_kl_history, marker="s", linewidth=2, label="Random", color="tab:orange")
        axes[1, 0].axhline(initial_kl, color="0.5", linestyle=":", label="Initial coverage divergence")
        axes[1, 0].set_xlabel("Iteration (new points)")
        axes[1, 0].set_ylabel("Divergence from uniform distance bins")
        axes[1, 0].set_title("Distance coverage vs iteration (lower is more uniform)")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        width = 0.42
        axes[1, 1].bar(xbins - width / 2, bo_new_counts, width=width, label="BO", color="tab:blue", alpha=0.85)
        axes[1, 1].bar(xbins + width / 2, random_new_counts, width=width, label="Random", color="tab:orange", alpha=0.85)
        axes[1, 1].set_title("Where new points were added")
        axes[1, 1].set_xlabel("Distance bin")
        axes[1, 1].set_ylabel("# new points")
        axes[1, 1].set_xticks(xbins)
        axes[1, 1].set_xticklabels(labels, rotation=55, ha="right")
        axes[1, 1].legend()
        axes[1, 1].grid(True, axis="y", alpha=0.25)

        axes[1, 2].plot(iterations, bo_delta_kl, marker="o", linewidth=2, label="BO coverage improvement")
        axes[1, 2].plot(iterations, random_delta_kl, marker="s", linewidth=2, label="Random coverage improvement", color="tab:orange")
        axes[1, 2].axhline(0, color="0.3", linewidth=1)
        axes[1, 2].set_xlabel("Iteration (new points)")
        axes[1, 2].set_ylabel("Coverage improvement")
        axes[1, 2].set_title("Change in distance coverage per new point")
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(output_folder, "distance_kl_comparison.png")
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  - Distance coverage benchmark plot saved as: {plot_path}")

    # Input-space view of where the new samples were added.
    bo_points = bo_log[[c for c in feature_columns if c in bo_log.columns]].copy()
    random_points = random_df[feature_columns].copy()
    train_df = pd.DataFrame(np.asarray(X_train), columns=feature_columns)

    pairs = [
        ("distance", "inclination"),
        ("distance", "mass1"),
        ("distance", "mass2"),
        ("spin1z", "spin2z"),
    ]
    pairs = [(a, b) for a, b in pairs if a in feature_columns and b in feature_columns]
    if pairs:
        ncols = 2
        nrows = int(np.ceil(len(pairs) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows))
        axes = np.atleast_1d(axes).reshape(-1)

        for ax, (xcol, ycol) in zip(axes, pairs):
            ax.scatter(train_df[xcol], train_df[ycol], s=14, alpha=0.20, color="0.55", label="Initial")
            ax.scatter(random_points[xcol], random_points[ycol], s=55, alpha=0.75,
                       marker="x", color="tab:orange", label="Random new")
            ax.scatter(bo_points[xcol], bo_points[ycol], s=75, alpha=0.85,
                       marker="*", color="tab:blue", edgecolors="k", linewidths=0.3,
                       label="BO new")
            ax.set_xlabel(xcol)
            ax.set_ylabel(ycol)
            ax.set_title(f"New points in {xcol}-{ycol} space")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        for ax in axes[len(pairs):]:
            ax.axis("off")

        plt.tight_layout()
        plot_path = os.path.join(output_folder, "input_space_new_points.png")
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  - Input-space new-points plot saved as: {plot_path}")
