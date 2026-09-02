"""
Visualization utilities
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional, List

from utils.h_columns import H_OUTPUT_COLUMNS, H_PLOT_COLUMNS, H_PLOT_MAGNITUDE_COLUMNS, MAGNITUDE_COLUMNS


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
    Plot useful samples in h-space rather than x-space.

    Training points are split into detectable / non-detectable using the ZTF
    CDF probability, and new BO samples are overplotted with star/x markers.
    """
    excluded_h_cols = {"HasNS", "HasRemnant", "HasMassGap"}
    h_cols = [
        col
        for col in H_PLOT_COLUMNS
        if col not in excluded_h_cols and col in y_train_mags and col in new_events_df.columns
    ]
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

    detection_mag_arrays = {}
    for filt in MAGNITUDE_COLUMNS:
        if filt in y_train_mags:
            values = y_train_mags[filt]
            detection_mag_arrays[filt] = (
                values.to_numpy(dtype=float) if hasattr(values, "to_numpy") else np.asarray(values, dtype=float)
            )

    train_probs = np.zeros(n_train, dtype=float)
    for i in range(n_train):
        mags = {
            filt: float(arr[i])
            for filt, arr in detection_mag_arrays.items()
            if i < len(arr) and np.isfinite(arr[i])
        }
        train_probs[i], _ = detection_calc.compute_detection_probability(mags)
    train_detected = train_probs >= 0.5

    new_probs = new_events_df.get(
        "detection_probability",
        pd.Series(np.nan, index=new_events_df.index),
    ).to_numpy(dtype=float)
    if "detected" in new_events_df.columns:
        new_detected = new_events_df["detected"].astype(float).to_numpy() >= 0.5
    else:
        new_detected = new_probs >= 0.5

    n_dim = len(h_cols)
    fig, axes = plt.subplots(n_dim, n_dim, figsize=(2.75 * n_dim, 2.75 * n_dim))
    axes = np.asarray(axes)
    colors = {
        "train_no": "#b8b8b8",
        "train_yes": "#2374ab",
        "new_no": "#ff7f0e",
        "new_yes": "#d62728",
    }
    handles = None

    for row, y_col in enumerate(h_cols):
        for col, x_col in enumerate(h_cols):
            ax = axes[row, col]
            if row < col:
                ax.axis('off')
                continue

            if row == col:
                valid_no = np.isfinite(train_h[x_col][:n_train]) & ~train_detected
                valid_yes = np.isfinite(train_h[x_col][:n_train]) & train_detected
                ax.hist(train_h[x_col][:n_train][valid_no], bins=22, color=colors["train_no"], alpha=0.55)
                ax.hist(train_h[x_col][:n_train][valid_yes], bins=22, color=colors["train_yes"], alpha=0.60)
                if x_col in new_events_df:
                    new_vals = pd.to_numeric(new_events_df[x_col], errors="coerce").to_numpy(dtype=float)
                    y_level = ax.get_ylim()[1] * 0.08
                    valid_new_no = np.isfinite(new_vals) & ~new_detected
                    valid_new_yes = np.isfinite(new_vals) & new_detected
                    ax.scatter(new_vals[valid_new_no], np.full(valid_new_no.sum(), y_level),
                               marker='x', s=95, c=colors["new_no"], linewidths=2.2)
                    ax.scatter(new_vals[valid_new_yes], np.full(valid_new_yes.sum(), y_level),
                               marker='*', s=155, c=colors["new_yes"], edgecolors='black', linewidths=0.55)
            else:
                valid_train = np.isfinite(train_h[x_col][:n_train]) & np.isfinite(train_h[y_col][:n_train])
                no = valid_train & ~train_detected
                yes = valid_train & train_detected
                ax.scatter(train_h[x_col][:n_train][no], train_h[y_col][:n_train][no],
                           s=18, color=colors["train_no"], alpha=0.28)
                ax.scatter(train_h[x_col][:n_train][yes], train_h[y_col][:n_train][yes],
                           s=22, color=colors["train_yes"], alpha=0.55)

                new_valid = new_events_df[x_col].notna().to_numpy() & new_events_df[y_col].notna().to_numpy()
                if np.any(new_valid):
                    new_no = new_valid & ~new_detected
                    new_yes = new_valid & new_detected
                    h1 = ax.scatter(new_events_df.loc[new_no, x_col], new_events_df.loc[new_no, y_col],
                                    marker='x', s=110, c=colors["new_no"], linewidths=2.2)
                    h2 = ax.scatter(new_events_df.loc[new_yes, x_col], new_events_df.loc[new_yes, y_col],
                                    marker='*', s=170, c=colors["new_yes"], edgecolors='black', linewidths=0.55)
                    if handles is None:
                        h3 = ax.scatter([], [], s=56, color=colors["train_no"], alpha=0.65)
                        h4 = ax.scatter([], [], s=64, color=colors["train_yes"], alpha=0.75)
                        handles = [h4, h3, h2, h1]

            if x_col in H_PLOT_MAGNITUDE_COLUMNS:
                ax.invert_xaxis()
            if row != col and y_col in H_PLOT_MAGNITUDE_COLUMNS:
                ax.invert_yaxis()

            ax.grid(True, alpha=0.18)
            if row == n_dim - 1:
                ax.set_xlabel(x_col, fontsize=17, rotation=45, ha='right')
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel(y_col, fontsize=17)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=14)

    if handles is not None:
        fig.legend(
            handles,
            ["training detectable", "training non-detectable", "new detectable", "new non-detectable"],
            loc="upper right",
            bbox_to_anchor=(0.985, 0.985),
            fontsize=18,
            markerscale=1.25,
            frameon=True,
        )
        fig.text(
            0.985,
            0.875,
            (
                f"Detection summary: training detectable {int(train_detected.sum())}/{n_train} | "
                f"new detectable {int(new_detected.sum())}/{len(new_events_df)}"
            ),
            ha="right",
            va="top",
            fontsize=18,
            fontweight="bold",
        )

    plt.subplots_adjust(left=0.08, right=0.88, bottom=0.11, top=0.98, wspace=0.12, hspace=0.12)
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
    """Plot best observed scalar P_detect over iterations."""
    score_history = results['score_history']
    improved = results['improved_iteration']

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.step(range(len(score_history)), score_history, where='post',
            color='tab:blue', linewidth=2, label='Best observed P_detect so far')

    for k, it in enumerate(improved):
        ax.scatter(it, score_history[it], color='red', zorder=5, s=60,
                   label='New best accepted' if k == 0 else None)

    ax.set_xlabel('BO Iteration')
    ax.set_ylabel('Best observed P_detect')
    n_improvements = len(improved)
    total_gain = score_history[-1] - score_history[0]
    ax.set_title(f'Direct-GP objective over BO iterations '
                  f'({n_improvements} positive-label updates, total gain {total_gain:+.3f})')
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


def _numeric_array(values, n=None):
    if values is None:
        return None
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float) if hasattr(values, "to_numpy") else np.asarray(values, dtype=float)
    return arr[:n] if n is not None else arr


def _h_matrix_from_mapping(mapping, h_cols, n=None):
    cols = []
    used_cols = []
    for col in h_cols:
        if col not in mapping:
            continue
        arr = _numeric_array(mapping[col], n=n)
        if arr is None:
            continue
        cols.append(arr)
        used_cols.append(col)
    if not cols:
        return np.empty((0, 0)), []
    min_len = min(len(col) for col in cols)
    return np.column_stack([col[:min_len] for col in cols]), used_cols


def _compute_training_detection_probs(y_train_mags, detection_calc, n_train):
    probs = np.zeros(n_train, dtype=float)
    mag_arrays = {
        filt: _numeric_array(y_train_mags[filt], n=n_train)
        for filt in MAGNITUDE_COLUMNS
        if filt in y_train_mags
    }
    for i in range(n_train):
        mags = {
            filt: float(arr[i])
            for filt, arr in mag_arrays.items()
            if i < len(arr) and np.isfinite(arr[i])
        }
        probs[i], _ = detection_calc.compute_detection_probability(mags)
    return probs


def _make_h_bin_edges(reference_H, n_bins=12):
    edges = []
    valid_cols = []
    for j in range(reference_H.shape[1]):
        vals = reference_H[:, j]
        vals = vals[np.isfinite(vals)]
        if len(vals) < 2 or np.allclose(np.nanmin(vals), np.nanmax(vals)):
            continue
        lo, hi = np.nanmin(vals), np.nanmax(vals)
        pad = 1e-9 if np.isclose(lo, hi) else 0.02 * (hi - lo)
        edges.append(np.linspace(lo - pad, hi + pad, n_bins + 1))
        valid_cols.append(j)
    return edges, valid_cols


def _h_coverage_kl_to_uniform(H, bin_edges, valid_cols):
    """Average marginal h-space KL divergence to uniform bins."""
    H = np.asarray(H, dtype=float)
    if H.size == 0 or len(bin_edges) == 0:
        return np.nan, np.nan

    kls = []
    for edges, j in zip(bin_edges, valid_cols):
        vals = H[:, j]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        kl, _ = _kl_to_uniform(np.clip(vals, edges[0], edges[-1]), edges)
        if np.isfinite(kl):
            kls.append(kl)
    if not kls:
        return np.nan, np.nan
    return float(np.mean(kls)), float(np.sum(kls))


def _predict_h_for_points(X, gp_models, scaler_X, h_cols, feature_columns):
    """Predict h columns for random-baseline points using the h surrogate."""
    from models import predict_magnitudes_batch

    predicted, _ = predict_magnitudes_batch(X, gp_models, scaler_X)
    out = {}
    for col in h_cols:
        if col in predicted:
            out[col] = np.asarray(predicted[col], dtype=float)
    if "distance" in h_cols and "distance" in feature_columns:
        out["distance"] = np.asarray(X)[:, feature_columns.index("distance")]
    return out


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
    """Uniform-random candidate track with stochastic labels from P_detect."""
    from optimization.objectives import objective_function_probability

    rng = np.random.default_rng(seed)
    lows = np.array([b[0] for b in bounds], dtype=float)
    highs = np.array([b[1] for b in bounds], dtype=float)
    X_random = lows + rng.random((n_new, len(bounds))) * (highs - lows)

    probs = np.array([
        objective_function_probability(
            x, gp_models, scaler_X, flat_limits,
            detection_calc=detection_calc
        )
        for x in X_random
    ])
    labels = (rng.random(n_new) < probs).astype(int)

    random_df = pd.DataFrame(X_random, columns=feature_columns)
    random_df["detection_probability"] = probs
    random_df["detected"] = labels
    random_df["iteration"] = np.arange(n_new)
    return random_df


def _sum_numeric_col(df, col):
    if df is None or df.empty or col not in df:
        return 0.0
    return float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum())


def _mean_numeric_col(df, col):
    if df is None or df.empty or col not in df:
        return np.nan
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(vals.mean()) if len(vals) else np.nan


def _finite_or_nan(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def _first_iteration_at_or_below(series, threshold):
    if series is None:
        return np.nan
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    hits = np.where(vals <= threshold)[0]
    return int(hits[0] + 1) if len(hits) else np.nan


def _write_computational_benefit_summary(metrics, bo_log, output_folder):
    """
    Save cost accounting.

    X = expensive NMMA simulation time.
    Y = GP fitting/refit time.
    Z = BO acquisition/ROBOT/post-processing overhead.
    """
    if metrics is None or metrics.empty:
        return

    bo_log = pd.DataFrame(bo_log).copy()
    final = metrics.iloc[-1]
    n_bo = int(len(metrics))
    bo_hits = int(final.get("bo_cumulative_positive", 0))
    random_hits = int(final.get("random_cumulative_positive", 0))

    mean_sim = _mean_numeric_col(bo_log, "simulation_seconds")
    mean_gp = _mean_numeric_col(bo_log, "gp_refit_seconds")
    mean_acq = _mean_numeric_col(bo_log, "candidate_scoring_seconds")
    mean_robot = _mean_numeric_col(bo_log, "robot_selection_seconds")
    mean_post = _mean_numeric_col(bo_log, "postprocess_seconds")
    mean_overhead = _mean_numeric_col(bo_log, "bo_overhead_seconds")

    total_sim = _sum_numeric_col(bo_log, "simulation_seconds")
    total_gp = _sum_numeric_col(bo_log, "gp_refit_seconds")
    total_acq = _sum_numeric_col(bo_log, "candidate_scoring_seconds")
    total_robot = _sum_numeric_col(bo_log, "robot_selection_seconds")
    total_post = _sum_numeric_col(bo_log, "postprocess_seconds")
    total_overhead = _sum_numeric_col(bo_log, "bo_overhead_seconds")

    random_hit_rate = random_hits / n_bo if n_bo else np.nan
    if random_hits > 0 and bo_hits > 0:
        random_needed = bo_hits / random_hit_rate
        random_detection_match_cost = random_needed * mean_sim
        detection_note = "Estimated from random hit rate in this rollout."
    elif bo_hits > 0:
        random_needed = np.nan
        random_detection_match_cost = np.nan
        detection_note = (
            "Random had zero positives, so detection-match cost is not finite "
            "from this rollout; use as lower-bound evidence."
        )
    else:
        random_needed = 0.0
        random_detection_match_cost = 0.0
        detection_note = "BO found zero positives, so detection-match comparison is not informative."

    bo_total_measured_cost = total_sim + total_gp + total_overhead
    random_same_budget_cost = n_bo * mean_sim if np.isfinite(mean_sim) else np.nan

    h_threshold = final.get("bo_h_kl_sum", np.nan)
    dist_threshold = final.get("bo_distance_kl", np.nan)
    random_h_iters = _first_iteration_at_or_below(metrics.get("random_h_kl_sum"), h_threshold)
    random_dist_iters = _first_iteration_at_or_below(metrics.get("random_distance_kl"), dist_threshold)

    summary = pd.DataFrame([{
        "n_bo_iterations": n_bo,
        "bo_detectable_events": bo_hits,
        "random_detectable_events_same_budget": random_hits,
        "bo_success_rate": _finite_or_nan(final.get("bo_success_rate", np.nan)),
        "random_success_rate": _finite_or_nan(final.get("random_success_rate", np.nan)),
        "bo_final_h_kl_sum": _finite_or_nan(final.get("bo_h_kl_sum", np.nan)),
        "random_final_h_kl_sum": _finite_or_nan(final.get("random_h_kl_sum", np.nan)),
        "bo_final_distance_kl": _finite_or_nan(final.get("bo_distance_kl", np.nan)),
        "random_final_distance_kl": _finite_or_nan(final.get("random_distance_kl", np.nan)),
        "random_iterations_to_match_bo_h_kl_if_seen": random_h_iters,
        "random_iterations_to_match_bo_distance_kl_if_seen": random_dist_iters,
        "mean_simulation_seconds_X": mean_sim,
        "mean_gp_refit_seconds_Y": mean_gp,
        "mean_acquisition_robot_overhead_seconds_Z": mean_overhead,
        "total_simulation_seconds_X": total_sim,
        "total_gp_refit_seconds_Y": total_gp,
        "total_acquisition_robot_overhead_seconds_Z": total_overhead,
        "bo_total_measured_cost_seconds_X_plus_Y_plus_Z": bo_total_measured_cost,
        "random_same_budget_simulation_cost_seconds": random_same_budget_cost,
        "random_simulations_needed_to_match_bo_detections_estimate": random_needed,
        "random_detection_match_cost_seconds_estimate": random_detection_match_cost,
        "detection_cost_note": detection_note,
    }])
    summary.to_csv(os.path.join(output_folder, "computational_benefit_summary.csv"), index=False)

    components = pd.DataFrame([{
        "component": "X_simulation",
        "total_seconds": total_sim,
        "mean_seconds_per_iteration": mean_sim,
        "description": "Expensive NMMA injection/lightcurve-analysis simulation.",
    }, {
        "component": "Y_gp_refit",
        "total_seconds": total_gp,
        "mean_seconds_per_iteration": mean_gp,
        "description": "Direct scalar GP(g) refit after each new measured event.",
    }, {
        "component": "Z_candidate_scoring",
        "total_seconds": total_acq,
        "mean_seconds_per_iteration": mean_acq,
        "description": "Candidate acquisition scoring before the expensive simulation.",
    }, {
        "component": "Z_robot_selection",
        "total_seconds": total_robot,
        "mean_seconds_per_iteration": mean_robot,
        "description": "ROBOT h-space trust/diversity filtering.",
    }, {
        "component": "Z_postprocess",
        "total_seconds": total_post,
        "mean_seconds_per_iteration": mean_post,
        "description": "Parsing light curves, extracting h(x), and assigning labels.",
    }])
    components.to_csv(os.path.join(output_folder, "compute_cost_components.csv"), index=False)


def plot_active_search_benchmark(
    true_eval_log,
    new_events_df,
    X_train,
    y_train_mags,
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
      - h_space_kl_comparison.png
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

    # Full h-space coverage KL diagnostic. This uses only detectable/useful
    # events in the iteration histories, not all proposed events.
    h_cols = [col for col in H_PLOT_COLUMNS if col in y_train_mags]
    if h_cols:
        n_train_h = min(len(_numeric_array(y_train_mags[col])) for col in h_cols if col in y_train_mags)
        train_H, h_cols = _h_matrix_from_mapping(y_train_mags, h_cols, n=n_train_h)
        if train_H.size:
            train_probs = _compute_training_detection_probs(y_train_mags, detection_calc, len(train_H))
            train_detected_h = train_probs >= 0.5
            if not np.any(train_detected_h):
                train_detected_h = np.asarray(y_train_detection[:len(train_H)], dtype=float) >= 0.5

            bo_h_cols = [col for col in h_cols if col in new_events_df.columns]
            bo_H, bo_h_cols = _h_matrix_from_mapping(new_events_df, bo_h_cols, n=n_new)
            common_h_cols = [col for col in h_cols if col in bo_h_cols]
            if common_h_cols:
                random_h = _predict_h_for_points(
                    random_df[feature_columns].to_numpy(dtype=float),
                    gp_models=gp_models,
                    scaler_X=scaler_X,
                    h_cols=common_h_cols,
                    feature_columns=feature_columns,
                )
                common_h_cols = [col for col in common_h_cols if col in random_h]
                train_H, common_h_cols = _h_matrix_from_mapping(y_train_mags, common_h_cols, n=n_train_h)
                bo_H, _ = _h_matrix_from_mapping(new_events_df, common_h_cols, n=n_new)
                random_H, _ = _h_matrix_from_mapping(random_h, common_h_cols, n=n_new)

                train_probs = _compute_training_detection_probs(y_train_mags, detection_calc, len(train_H))
                train_detected_h = train_probs >= 0.5
                reference_H = train_H[train_detected_h]
                bin_edges, valid_cols = _make_h_bin_edges(reference_H)
                if len(reference_H) > 1 and len(bin_edges) > 0 and len(bo_H) == n_new and len(random_H) == n_new:
                    bo_h_mean_history = []
                    bo_h_sum_history = []
                    random_h_mean_history = []
                    random_h_sum_history = []
                    for k in range(n_new + 1):
                        bo_detected_H = bo_H[:k][bo_labels[:k] == 1]
                        random_detected_H = random_H[:k][random_labels[:k] == 1]
                        bo_current = np.vstack([reference_H, bo_detected_H]) if len(bo_detected_H) else reference_H
                        random_current = np.vstack([reference_H, random_detected_H]) if len(random_detected_H) else reference_H
                        bo_mean, bo_sum = _h_coverage_kl_to_uniform(bo_current, bin_edges, valid_cols)
                        random_mean, random_sum = _h_coverage_kl_to_uniform(random_current, bin_edges, valid_cols)
                        bo_h_mean_history.append(bo_mean)
                        bo_h_sum_history.append(bo_sum)
                        random_h_mean_history.append(random_mean)
                        random_h_sum_history.append(random_sum)

                    h_dim_names = [common_h_cols[j] for j in valid_cols]
                    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                    axes[0].plot(range(n_new + 1), bo_h_sum_history, marker="o", linewidth=2, label="BO")
                    axes[0].plot(range(n_new + 1), random_h_sum_history, marker="s", linewidth=2, color="tab:orange", label="Random")
                    axes[0].set_xlabel("Iteration (detectable new points only)")
                    axes[0].set_ylabel("Sum marginal h-space KL to uniform")
                    axes[0].set_title(f"Full h-space coverage KL ({len(h_dim_names)} dimensions)")
                    axes[0].legend()
                    axes[0].grid(True, alpha=0.3)

                    bo_delta_h = np.array(bo_h_sum_history[:-1]) - np.array(bo_h_sum_history[1:])
                    random_delta_h = np.array(random_h_sum_history[:-1]) - np.array(random_h_sum_history[1:])
                    axes[1].plot(iterations, bo_delta_h, marker="o", linewidth=2, label="BO KL improvement")
                    axes[1].plot(iterations, random_delta_h, marker="s", linewidth=2, color="tab:orange", label="Random KL improvement")
                    axes[1].axhline(0, color="0.3", linewidth=1)
                    axes[1].set_xlabel("Iteration")
                    axes[1].set_ylabel("KL improvement")
                    axes[1].set_title("Change in full h-space coverage KL")
                    axes[1].legend()
                    axes[1].grid(True, alpha=0.3)

                    plt.tight_layout()
                    plot_path = os.path.join(output_folder, "h_space_kl_comparison.png")
                    plt.savefig(plot_path, dpi=150)
                    plt.close(fig)
                    print(f"  - Full h-space KL benchmark plot saved as: {plot_path}")

                    metrics["bo_h_kl_sum"] = bo_h_sum_history[1:]
                    metrics["random_h_kl_sum"] = random_h_sum_history[1:]
                    metrics["bo_h_kl_mean"] = bo_h_mean_history[1:]
                    metrics["random_h_kl_mean"] = random_h_mean_history[1:]
                    metrics["bo_h_kl_improvement"] = bo_delta_h
                    metrics["random_h_kl_improvement"] = random_delta_h
                    metrics["h_kl_dimensions"] = len(h_dim_names)
                    metrics["h_kl_columns"] = ",".join(h_dim_names)

    # Distance-bin coverage diagnostic similar to the reference figure.
    # This version uses detectable/useful events for the distance histograms
    # and distance coverage curves.
    if "distance" in feature_columns:
        dist_idx = feature_columns.index("distance")
        initial_distance_all = np.asarray(X_train)[:, dist_idx]
        train_distance_probs = _compute_training_detection_probs(
            y_train_mags, detection_calc, len(initial_distance_all)
        )
        train_detected_distance = train_distance_probs >= 0.5
        if not np.any(train_detected_distance):
            train_detected_distance = np.asarray(y_train_detection[:len(initial_distance_all)], dtype=float) >= 0.5
        initial_distance = initial_distance_all[train_detected_distance]

        bo_distance_all = bo_log["distance"].to_numpy(dtype=float) if "distance" in bo_log else new_events_df.get("distance", pd.Series(dtype=float)).to_numpy()
        random_distance_all = random_df["distance"].to_numpy(dtype=float)
        bo_detected_distance = bo_distance_all[bo_labels == 1]
        random_detected_distance = random_distance_all[random_labels == 1]
        bo_distance = bo_detected_distance
        random_distance = random_detected_distance

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
            bo_current = np.concatenate([initial_distance, bo_distance_all[:k][bo_labels[:k] == 1]])
            random_current = np.concatenate([initial_distance, random_distance_all[:k][random_labels[:k] == 1]])
            bo_kl_history.append(_kl_to_uniform(bo_current, bin_edges)[0])
            random_kl_history.append(_kl_to_uniform(random_current, bin_edges)[0])

        bo_delta_kl = np.array(bo_kl_history[:-1]) - np.array(bo_kl_history[1:])
        random_delta_kl = np.array(random_kl_history[:-1]) - np.array(random_kl_history[1:])

        bo_new_counts, _ = np.histogram(bo_detected_distance, bins=bin_edges)
        random_new_counts, _ = np.histogram(random_detected_distance, bins=bin_edges)

        fig, axes = plt.subplots(2, 3, figsize=(19, 10))
        panels = [
            ("Initial detectable", initial_counts, initial_kl, len(initial_distance)),
            ("BO final detectable", bo_final_counts, bo_final_kl, len(initial_distance) + len(bo_distance)),
            ("Random final detectable", random_final_counts, random_final_kl, len(initial_distance) + len(random_distance)),
        ]

        for ax, (title, counts, kl_val, n_total) in zip(axes[0], panels):
            ax.bar(xbins, counts, color="tab:blue", alpha=0.85, edgecolor="white")
            ax.axhline(np.mean(counts), color="tab:blue", linestyle="--", linewidth=1.8, alpha=0.8)
            empty_bins = int(np.sum(counts == 0))
            ax.text(0.01, 0.93, f"Empty bins: {empty_bins}", transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.5", alpha=0.9))
            ax.set_title(f"{title} (n={n_total})\nDetectable distance divergence={kl_val:.4f}")
            ax.set_xlabel("Distance bin")
            ax.set_ylabel("Count")
            ax.set_xticks(xbins)
            ax.set_xticklabels(labels, rotation=55, ha="right")
            ax.grid(True, axis="y", alpha=0.25)

        axes[1, 0].plot(range(n_new + 1), bo_kl_history, marker="o", linewidth=2, label="BO")
        axes[1, 0].plot(range(n_new + 1), random_kl_history, marker="s", linewidth=2, label="Random", color="tab:orange")
        axes[1, 0].axhline(initial_kl, color="0.5", linestyle=":", label="Initial coverage divergence")
        axes[1, 0].set_xlabel("Iteration (new measured points)")
        axes[1, 0].set_ylabel("Divergence from uniform distance bins")
        axes[1, 0].set_title("Detectable-only distance coverage vs iteration")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        width = 0.42
        axes[1, 1].bar(xbins - width / 2, bo_new_counts, width=width, label="BO", color="tab:blue", alpha=0.85)
        axes[1, 1].bar(xbins + width / 2, random_new_counts, width=width, label="Random", color="tab:orange", alpha=0.85)
        axes[1, 1].set_title("Where detectable new points were added")
        axes[1, 1].set_xlabel("Distance bin")
        axes[1, 1].set_ylabel("# detectable new points")
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

        metrics["bo_distance_kl"] = bo_kl_history[1:]
        metrics["random_distance_kl"] = random_kl_history[1:]
        metrics["bo_distance_kl_improvement"] = bo_delta_kl
        metrics["random_distance_kl_improvement"] = random_delta_kl

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

    metrics.to_csv(os.path.join(output_folder, "benchmark_metrics.csv"), index=False)
    _write_computational_benefit_summary(metrics, bo_log, output_folder)
