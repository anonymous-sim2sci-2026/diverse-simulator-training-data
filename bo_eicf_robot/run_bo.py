#!/usr/bin/env python3
"""
Closed-Loop Bayesian Optimization for Detectable BNS Events
Each iteration: GP proposes → NMMA simulates → Retrain → Repeat
"""

import os
import sys
import subprocess
import time
import numpy as np
import pandas as pd
from datetime import datetime
import joblib

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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

def run_single(
    seed=None,
    n_iterations=30,
    acquisition='ROBOT_EICF',
    output_folder=None,
    make_plots=True,
    robot_distance_mode="tail_l1",
    robot_tau_quantile=85.0,
    robot_rarity_n_bins=20,
    eicf_mc_samples=16,
    candidate_samples=300,
    candidate_uniform_samples=150,
    scoring_progress_every=25,
    eicf_batch_size=128,
    use_multitask_gp=False,
):
    """Internal helper used by run_multi_seed.py for one seed."""
    if acquisition != "ROBOT_EICF":
        raise ValueError(
            "This paper-facing folder only supports --acquisition ROBOT_EICF."
        )
    if robot_distance_mode != "tail_l1":
        raise ValueError(
            "This paper-facing folder only supports --robot-distance-mode tail_l1."
        )

    if seed is not None:
        np.random.seed(seed)

    print("=" * 80)
    print("CLOSED-LOOP BAYESIAN OPTIMIZATION")
    print("EI-CF + ROBOT proposes -> NMMA simulates -> Retrain -> Repeat")
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

    # Plots go in a separate directory. For multi-seed runs, include the
    # method/output-root folder so seed_1 from different methods cannot collide.
    output_path = os.path.abspath(output_folder)
    run_name = os.path.basename(output_path)
    parent_name = os.path.basename(os.path.dirname(output_path))
    if run_name.startswith("seed_") and parent_name and parent_name != "out":
        plots_folder = os.path.join(project_dir, "plots", parent_name, run_name)
    else:
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
    cache_tag = "mtgp15" if use_multitask_gp else "sklearn_fast"
    gp_cache_path = os.path.join(gp_cache_dir, f"initial_gp_models_h11_{cache_tag}.joblib")

    initial_gp_seconds = np.nan
    initial_gp_source = None
    initial_multitask_gp_seconds = 0.0

    if os.path.exists(gp_cache_path):
        print(f"  Found cached GP models at {gp_cache_path}, loading...", flush=True)
        t_cache = time.time()
        cached = joblib.load(gp_cache_path)
        gp_models = cached['gp_models']
        scaler_X = cached['scaler_X']
        detection_gp = cached['detection_gp']
        initial_gp_seconds = time.time() - t_cache
        initial_gp_source = "cache"
        print(
            f"  Loaded {len(gp_models)} magnitude GPs + 1 detection GP (cached) "
            f"in {initial_gp_seconds:.1f} s",
            flush=True,
        )

        if use_multitask_gp and MULTITASK_GP_KEY not in gp_models:
            print("  Cached models do not include MultiTaskGP; building it now...", flush=True)
            t_mtgp = time.time()
            multitask_filters = [f for f in H_OUTPUT_COLUMNS if f in y_train_mags]
            multitask_gp = build_multitask_magnitude_gp(
                X_train, y_train_mags, filters=multitask_filters
            )
            initial_multitask_gp_seconds = time.time() - t_mtgp
            if multitask_gp.available:
                gp_models[MULTITASK_GP_KEY] = multitask_gp
                joblib.dump(
                    {'gp_models': gp_models, 'scaler_X': scaler_X, 'detection_gp': detection_gp},
                    gp_cache_path
                )
                print(f"  Added MultiTaskGP to cache in {initial_multitask_gp_seconds:.1f} s", flush=True)
            else:
                print("  Warning: MultiTaskGP unavailable; continuing with independent GPs", flush=True)
                if multitask_gp.error_message:
                    print(f"    Reason: {multitask_gp.error_message}")
    else:
        print("  No cache found — building GP models from scratch...", flush=True)
        t_gp = time.time()
        gp_models, scaler_X, detection_gp = build_gp_models(
            X_train, y_train_mags, y_train_detection,
            use_multitask_gp=use_multitask_gp,
        )
        initial_gp_seconds = time.time() - t_gp
        initial_gp_source = "built"
        print(f"  Initial GP build finished in {initial_gp_seconds:.1f} s; saving cache...", flush=True)
        joblib.dump(
            {'gp_models': gp_models, 'scaler_X': scaler_X, 'detection_gp': detection_gp},
            gp_cache_path
        )
        print(f"  Built {len(gp_models)} magnitude GPs + 1 detection GP", flush=True)
        print(f"  Saved to cache: {gp_cache_path}", flush=True)

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
    print(f"       ROBOT distance mode: {robot_distance_mode}")
    print(f"       ROBOT tau quantile: {robot_tau_quantile}")
    print("    2. NMMA simulates real lightcurve for candidate")
    print("    3. Add result (detectable/not) to training data")
    print("    4. Retrain detection GP with new data")
    print("    5. Repeat")
    print("=" * 80)

    sim_output_dir = output_folder

    closed_loop_t0 = time.time()
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
        robot_distance_mode=robot_distance_mode,
        robot_tau_quantile=robot_tau_quantile,
        robot_rarity_n_bins=robot_rarity_n_bins,
        eicf_mc_samples=eicf_mc_samples,
        candidate_samples=candidate_samples,
        candidate_uniform_samples=candidate_uniform_samples,
        scoring_progress_every=scoring_progress_every,
        eicf_batch_size=eicf_batch_size,
    )
    closed_loop_seconds = time.time() - closed_loop_t0

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

    if not new_events_df.empty:
        new_events_df.to_csv(
            os.path.join(output_folder, 'new_events_with_h.csv'),
            index=False
        )
        print(f"  Saved new-event h(x) table: new_events_with_h.csv")

    timing_df = pd.DataFrame(true_eval_log) if true_eval_log else pd.DataFrame()

    def _sum_timing(column):
        if timing_df.empty or column not in timing_df:
            return 0.0
        return float(pd.to_numeric(timing_df[column], errors="coerce").fillna(0.0).sum())

    def _mean_timing(column):
        if timing_df.empty or column not in timing_df:
            return np.nan
        values = pd.to_numeric(timing_df[column], errors="coerce")
        return float(values.mean()) if values.notna().any() else np.nan

    summary = {
        'run_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'initial_training_events': len(X_train),
        'initial_detectable_events': int(n_detectable),
        'closed_loop_iterations': len(true_eval_log) if true_eval_log else 0,
        'best_peak_magnitude': best_peak_mag if best_peak_mag != np.inf else None,
        'best_parameters': str(best_X) if best_X is not None else None,
        'acquisition': acquisition,
        'robot_distance_mode': robot_distance_mode,
        'robot_tau_quantile': robot_tau_quantile,
        'eicf_mc_samples': eicf_mc_samples,
        'candidate_samples': candidate_samples,
        'candidate_uniform_samples': candidate_uniform_samples,
        'eicf_batch_size': eicf_batch_size,
        'use_multitask_gp': use_multitask_gp,
        'initial_gp_seconds': initial_gp_seconds,
        'initial_gp_source': initial_gp_source,
        'initial_multitask_gp_seconds': initial_multitask_gp_seconds,
        'closed_loop_seconds': closed_loop_seconds,
        'total_candidate_scoring_seconds': _sum_timing('candidate_scoring_seconds'),
        'total_robot_selection_seconds': _sum_timing('robot_selection_seconds'),
        'total_simulation_seconds': _sum_timing('simulation_seconds'),
        'total_postprocess_seconds': _sum_timing('postprocess_seconds'),
        'total_gp_refit_seconds': _sum_timing('gp_refit_seconds'),
        'total_bo_overhead_seconds': _sum_timing('bo_overhead_seconds'),
        'mean_candidate_scoring_seconds': _mean_timing('candidate_scoring_seconds'),
        'mean_robot_selection_seconds': _mean_timing('robot_selection_seconds'),
        'mean_simulation_seconds': _mean_timing('simulation_seconds'),
        'mean_postprocess_seconds': _mean_timing('postprocess_seconds'),
        'mean_gp_refit_seconds': _mean_timing('gp_refit_seconds'),
        'mean_bo_overhead_seconds': _mean_timing('bo_overhead_seconds'),
        'mean_iteration_seconds': _mean_timing('iteration_seconds'),
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
        print("  Creating benchmark plots: active-search success, h-space KL, distance coverage, input-space points...")
        plot_active_search_benchmark(
            true_eval_log=true_eval_log,
            new_events_df=new_events_df,
            X_train=X_train,
            y_train_mags=y_train_mags,
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
    elif make_plots:
        print("  Skipping benchmark plots because true_eval_log is empty.")

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
    print("  - new_events_with_h.csv (new samples with predicted/true h columns)")
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
    print("  - computational_benefit_summary.csv")
    print("  - compute_cost_components.csv")

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
