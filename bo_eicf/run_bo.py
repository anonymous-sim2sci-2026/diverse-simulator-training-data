#!/usr/bin/env python3
"""
Closed-Loop Bayesian Optimization for Detectable BNS Events
Each iteration: GP proposes → NMMA simulates → Retrain → Repeat
"""

import os
import sys
import subprocess
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import joblib

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.environ.get("TMPDIR", "/tmp"), "matplotlib-cache"),
)

from config import Config
from data.loader import load_and_prepare_data, prepare_training_data
from models import build_gp_models, SimulationRunner
from models.gp_surrogate import MULTITASK_GP_KEY
from models.multitask_gp import build_multitask_magnitude_gp
from optimization import closed_loop_bayesian_optimization
from utils.detection import DetectionCalculator
from utils.h_columns import H_OUTPUT_COLUMNS
from utils.visualization import (
    plot_true_evaluation_history,
    plot_optimization_results,
    plot_h_space_useful_samples,
    plot_parameter_exploration,
    plot_objective_improvement,
    plot_active_search_benchmark,
)


def run_single(
    seed=None,
    n_iterations=10,
    acquisition='EI',
    output_folder=None,
    make_plots=True,
    candidate_samples=1500,
    candidate_uniform_samples=500,
    eicf_mc_samples=256,
    active_search=False,
):
    """Run one closed-loop BO rollout and return paths/results for aggregation."""
    if seed is not None:
        np.random.seed(seed)

    print("=" * 80)
    print("CLOSED-LOOP BAYESIAN OPTIMIZATION")
    print("GP proposes → NMMA simulates → Retrain → Repeat")
    if seed is not None:
        print(f"Random seed: {seed}")
    print("=" * 80)

    # Load configuration
    config = Config()
    project_dir = os.path.dirname(os.path.abspath(__file__))
    if output_folder is None:
        output_folder = config.output_dir
    else:
        os.makedirs(output_folder, exist_ok=True)
    print(f"\nOutput folder: {output_folder}")

    # Plots go in a separate directory but reuse the same run-name (e.g. 072126_1)
    run_name = os.path.basename(output_folder)
    plots_folder = os.path.join(project_dir, "plots", run_name)
    os.makedirs(plots_folder, exist_ok=True)
    print(f"Plots folder: {plots_folder}")

    # Check NMMA
    # print("\n[0] Checking NMMA installation...")
    # nmma_available = check_nmma_installation()

    # Load data
    print("\n[1] LOAD TRAINING DATA")
    print("-" * 40)
    df = load_and_prepare_data()
    print(f"  Total events loaded: {len(df)}")

    # Setup detection limits (do this before prepare_training_data)
    detection_calc = DetectionCalculator(
        detection_filters=config.detection_filters
    )
    limits = detection_calc.define_detection_limits()
    flat_limits = detection_calc.flatten_limits(limits)

    # Prepare training data
    print("\n[2] PREPARE TRAINING DATA")
    print("-" * 40)
    X_train, y_train_mags, y_train_detection = prepare_training_data(
        df, config.feature_columns, config.detection_filters, flat_limits=flat_limits
    )
    n_detectable = np.sum(y_train_detection)
    print(f"  Training events: {len(X_train)}")
    print(f"  Detectable events: {n_detectable} ({n_detectable/len(X_train)*100:.1f}%)")

    # Build (or load cached) initial GP models
    print("\n[3] BUILD INITIAL GP SURROGATE MODELS")
    print("-" * 40)

    gp_cache_dir = os.path.join(project_dir, "gp_cache")
    os.makedirs(gp_cache_dir, exist_ok=True)
    gp_cache_path = os.path.join(gp_cache_dir, "initial_gp_models_h11.joblib")

    if os.path.exists(gp_cache_path):
        print(f"  Found cached GP models at {gp_cache_path}, loading...")
        cached = joblib.load(gp_cache_path)
        gp_models = cached['gp_models']
        scaler_X = cached['scaler_X']
        detection_gp = cached['detection_gp']
        print(f"  Loaded {len(gp_models)} magnitude GPs + 1 detection GP (cached)")

        if MULTITASK_GP_KEY not in gp_models:
            print("  Cached models do not include MultiTaskGP; building it now...")
            multitask_filters = [f for f in H_OUTPUT_COLUMNS if f in y_train_mags]
            multitask_gp = build_multitask_magnitude_gp(
                X_train, y_train_mags, filters=multitask_filters
            )
            if multitask_gp.available:
                gp_models[MULTITASK_GP_KEY] = multitask_gp
                joblib.dump(
                    {'gp_models': gp_models, 'scaler_X': scaler_X, 'detection_gp': detection_gp},
                    gp_cache_path
                )
                print("  Added MultiTaskGP to cache")
            else:
                print("  Warning: MultiTaskGP unavailable; continuing with independent GPs")
                if multitask_gp.error_message:
                    print(f"    Reason: {multitask_gp.error_message}")
    else:
        print("  No cache found — building GP models from scratch...")
        gp_models, scaler_X, detection_gp = build_gp_models(
            X_train, y_train_mags, y_train_detection
        )
        joblib.dump(
            {'gp_models': gp_models, 'scaler_X': scaler_X, 'detection_gp': detection_gp},
            gp_cache_path
        )
        print(f"  Built {len(gp_models)} magnitude GPs + 1 detection GP")
        print(f"  Saved to cache: {gp_cache_path}")

    # Setup simulation runner
    print("\n[4] SETUP NMMA SIMULATOR")
    print("-" * 40)
    if not os.path.exists(config.SVD_PATH) or not os.path.exists(config.PRIOR_FILE):
        print("❌ ERROR: Required paths not found!")
        print(f"  SVD_PATH: {config.SVD_PATH} (exists: {os.path.exists(config.SVD_PATH)})")
        print(f"  PRIOR_FILE: {config.PRIOR_FILE} (exists: {os.path.exists(config.PRIOR_FILE)})")
        return

    detection_calc = DetectionCalculator(detection_filters=config.detection_filters)
    limits = detection_calc.define_detection_limits()
    flat_limits = detection_calc.flatten_limits(limits)

    runner = SimulationRunner(config, detection_calc=detection_calc)
    runner.validate_environment()

    print(f"  SVD_PATH: {config.SVD_PATH}")
    print(f"  PRIOR_FILE: {config.PRIOR_FILE}")
    print(f"  Filters: {config.FILTERS}")
    print(f"  Exposure times: {config.EXPOSURE_TIMES}s")

    # ============================================================
    # CLOSED-LOOP BAYESIAN OPTIMIZATION
    # ============================================================
    print("\n[5] RUN CLOSED-LOOP BAYESIAN OPTIMIZATION")
    print("=" * 80)
    print("  Iteration 0: Initial GP (trained on historical data)")
    print("  Each iteration:")
    print(f"    1. GP proposes candidate that maximizes {acquisition} acquisition")
    print("    2. NMMA simulates real lightcurve for candidate")
    print("    3. Add result (detectable/not) to training data")
    print("    4. Retrain detection GP with new data")
    print("    5. Repeat")
    print("=" * 80)

    sim_output_dir = output_folder

    (
        best_X, best_peak_mag, true_eval_log, gp_models, scaler_X, detection_gp,
        results, new_events_df
    ) = closed_loop_bayesian_optimization(
        X_train=X_train,
        y_train_mags=y_train_mags,
        y_train_detection=y_train_detection,
        flat_limits=flat_limits,
        runner=runner,
        base_sim_output_dir=sim_output_dir,
        gp_models=gp_models,
        scaler_X=scaler_X,
        detection_gp=detection_gp,
        n_iterations=n_iterations,
        acquisition=acquisition,
        kappa=2.0,
        exposure_time=180,
        bounds=config.bounds,
        feature_columns=config.feature_columns,
        detection_filters=config.detection_filters,
        candidate_samples=candidate_samples,
        candidate_uniform_samples=candidate_uniform_samples,
        eicf_mc_samples=eicf_mc_samples,
        active_search=active_search,
    )

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    print("\n[6] SAVE RESULTS")
    print("-" * 40)

    if true_eval_log:
        true_eval_df = pd.DataFrame(true_eval_log)
        true_eval_df.to_csv(
            os.path.join(output_folder, 'closed_loop_results.csv'),
            index=False
        )
        print(f"  Saved closed-loop results: closed_loop_results.csv")

    summary = {
        'run_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'initial_training_events': len(X_train),
        'initial_detectable_events': int(n_detectable),
        'closed_loop_iterations': len(true_eval_log) if true_eval_log else 0,
        'best_peak_magnitude': best_peak_mag if best_peak_mag != np.inf else None,
        'best_parameters': str(best_X) if best_X is not None else None,
    }

    if best_X is not None:
        for j, name in enumerate(config.feature_columns):
            if j < len(best_X):
                summary[f'best_{name}'] = best_X[j]

    pd.DataFrame([summary]).to_csv(
        os.path.join(output_folder, 'summary.csv'),
        index=False
    )
    print(f"  Saved summary: summary.csv")

    # ============================================================
    # CREATE VISUALIZATIONS (all 4 plots, saved under bo_cdf/plots/<run_name>)
    # ============================================================
    if make_plots:
        print("\n[7] CREATE VISUALIZATIONS")
        print("-" * 40)
    else:
        print("\n[7] SKIP VISUALIZATIONS")
        print("-" * 40)

    if make_plots and true_eval_log:
        plot_true_evaluation_history(true_eval_log, plots_folder)

    if make_plots and best_X is not None and results['X_all']:
        plot_parameter_exploration(results, best_X, plots_folder, feature_columns=config.feature_columns)

    if make_plots and results['score_history']:
        plot_objective_improvement(results, plots_folder)

    if make_plots and not new_events_df.empty:
        plot_optimization_results(
            results, new_events_df, X_train, y_train_detection,
            output_folder=plots_folder, detection_filters=config.detection_filters
        )
        plot_h_space_useful_samples(
            y_train_mags=y_train_mags,
            y_train_detection=y_train_detection,
            new_events_df=new_events_df,
            detection_calc=detection_calc,
            output_folder=plots_folder,
            detection_filters=config.detection_filters,
        )

    if make_plots and true_eval_log:
        plot_active_search_benchmark(
            true_eval_log=true_eval_log,
            new_events_df=new_events_df,
            X_train=X_train,
            y_train_detection=y_train_detection,
            gp_models=gp_models,
            scaler_X=scaler_X,
            flat_limits=flat_limits,
            bounds=config.bounds,
            feature_columns=config.feature_columns,
            detection_calc=detection_calc,
            output_folder=plots_folder,
            random_seed=(seed + 100000 if seed is not None else 12345),
        )

    # ============================================================
    # FINAL REPORT
    # ============================================================
    print("\n" + "=" * 80)
    print("CLOSED-LOOP BO COMPLETE!")
    print("=" * 80)

    if best_X is not None:
        print(f"\n✓ BEST DETECTABLE EVENT FOUND:")
        print(f"  Peak magnitude: {best_peak_mag:.3f}")
        print(f"  Parameters:")
        for j, name in enumerate(config.feature_columns):
            if j < len(best_X):
                print(f"    {name}: {best_X[j]:.4f}")
    else:
        print("\n⚠️ No detectable event found in any iteration")

    print(f"\n✓ Total closed-loop iterations: {len(true_eval_log) if true_eval_log else 0}")
    print(f"\nAll data saved to: {output_folder}/")
    print("  - closed_loop_results.csv (each iteration's simulation)")
    print("  - summary.csv (run summary)")
    print(f"\nAll plots saved to: {plots_folder}/")
    print("  - true_simulation_history.png")
    print("  - parameter_exploration.png")
    print("  - objective_improvement.png")
    print("  - bayesian_optimization_results.png")
    print("  - h_space_useful_samples.png")
    print("  - active_search_success_comparison.png")
    print("  - distance_kl_comparison.png")
    print("  - input_space_new_points.png")
    print("  - random_baseline_samples.csv")
    print("  - benchmark_metrics.csv")

    return {
        "seed": seed,
        "output_folder": output_folder,
        "plots_folder": plots_folder,
        "best_X": best_X,
        "best_peak_mag": best_peak_mag,
        "true_eval_log": true_eval_log,
        "results": results,
        "new_events_df": new_events_df,
    }


def main():
    parser = argparse.ArgumentParser(description="Run one bo_cdf_eicf1 closed-loop rollout.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible rollout labels/candidates.")
    parser.add_argument("--n-iterations", type=int, default=10, help="Number of closed-loop NMMA evaluations.")
    parser.add_argument("--acquisition", choices=["EI", "UCB"], default="EI", help="Acquisition function to use.")
    parser.add_argument("--output-folder", default=None, help="Optional explicit output folder for this run.")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    parser.add_argument("--candidate-samples", type=int, default=1500)
    parser.add_argument("--candidate-uniform-samples", type=int, default=500)
    parser.add_argument("--eicf-mc-samples", type=int, default=256)
    parser.add_argument(
        "--active-search",
        action="store_true",
        help="Score EI-CF candidates by expected detectability E[g(h(x))], matching the ROBOT active-search runs.",
    )
    args = parser.parse_args()

    run_single(
        seed=args.seed,
        n_iterations=args.n_iterations,
        acquisition=args.acquisition,
        output_folder=args.output_folder,
        make_plots=not args.no_plots,
        candidate_samples=args.candidate_samples,
        candidate_uniform_samples=args.candidate_uniform_samples,
        eicf_mc_samples=args.eicf_mc_samples,
        active_search=args.active_search,
    )


if __name__ == "__main__":
    main()
