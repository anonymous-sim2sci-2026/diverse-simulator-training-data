#!/usr/bin/env python3
"""
Closed-Loop Bayesian Optimization for Detectable BNS Events
Each iteration: GP proposes → NMMA simulates → Retrain → Repeat
"""

import os
import sys
import subprocess
import argparse
import time
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
from optimization.batch_bayesian_optimization import (
    closed_loop_batched_bayesian_optimization,
)
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
    robot_knn_k=5,
    robot_tau_quantile=25.0,
    robot_initial_length=0.6,
    robot_success_tolerance=10,
    robot_failure_tolerance=32,
    robot_rarity_n_bins=20,
    eicf_mc_samples=16,
    candidate_samples=1500,
    candidate_uniform_samples=600,
    robot_region_candidates=300,
    candidate_sampling_mode="mixed",
    candidate_detectability_threshold=0.5,
    robot_n_regions=10,
    robot_batch_regions=None,
    robot_candidates_per_region=1,
    parallel_simulations=1,
    trust_region_max_resamples=20,
    scoring_progress_every=25,
    eicf_batch_size=128,
    use_multitask_gp=False,
    skip_detection_gp_refit=False,
    checkpoint_every=1,
    resume=False,
    checkpoint_file=None,
    max_runtime_hours=None,
    debug_candidate_details=False,
    track_all_candidate_probs=False,
    refit_output_gps=False,
    output_gp_refit_every=1,
):
    """Run one closed-loop BO rollout and return paths/results for aggregation."""
    if acquisition != "ROBOT_EICF":
        raise ValueError(
            "This paper-facing folder only supports --acquisition ROBOT_EICF."
        )
    if robot_distance_mode not in {"tail_l1", "knn_l1"}:
        raise ValueError(
            "--robot-distance-mode must be tail_l1 or knn_l1."
        )
    if int(robot_knn_k) < 1:
        raise ValueError("--robot-knn-k must be at least 1")
    if not 0.05 <= float(robot_initial_length) <= 1.0:
        raise ValueError("--robot-initial-length must be between 0.05 and 1.0")
    if int(robot_success_tolerance) < 1 or int(robot_failure_tolerance) < 1:
        raise ValueError("ROBOT success/failure tolerances must be at least 1")
    if int(trust_region_max_resamples) < 1:
        raise ValueError("--trust-region-max-resamples must be at least 1")
    if not 0 <= int(robot_region_candidates) <= int(candidate_samples):
        raise ValueError(
            "--robot-region-candidates must be between 0 and --candidate-samples"
        )
    if candidate_sampling_mode not in {"mixed", "uniform_detectable"}:
        raise ValueError(
            "--candidate-sampling-mode must be mixed or uniform_detectable"
        )
    if not 0.0 <= float(candidate_detectability_threshold) <= 1.0:
        raise ValueError(
            "--candidate-detectability-threshold must be between 0 and 1"
        )
    if int(robot_n_regions) < 1:
        raise ValueError("--robot-n-regions must be at least 1")
    if robot_batch_regions is None:
        robot_batch_regions = int(robot_n_regions)
    if not 1 <= int(robot_batch_regions) <= int(robot_n_regions):
        raise ValueError(
            "--robot-batch-regions must be between 1 and --robot-n-regions"
        )
    if int(robot_candidates_per_region) < 1:
        raise ValueError("--robot-candidates-per-region must be at least 1")
    max_batch_events = int(robot_batch_regions) * int(robot_candidates_per_region)
    if not 1 <= int(parallel_simulations) <= max_batch_events:
        raise ValueError(
            "--parallel-simulations must be between 1 and "
            "--robot-batch-regions * --robot-candidates-per-region"
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
    output_folder = os.path.abspath(output_folder)
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
    batched_robot = int(robot_batch_regions) > 1 or int(robot_candidates_per_region) > 1
    if batched_robot:
        print(
            f"    1. GP proposes up to {max_batch_events} candidates, "
            f"up to {robot_candidates_per_region} from each ranked ROBOT region. "
            "Regions share one candidate "
            "pool and lower ranks exclude neighborhoods selected by higher ranks"
        )
        print(
            f"    2. NMMA simulates up to {parallel_simulations} events concurrently"
        )
    else:
        print(f"    1. GP proposes candidate that maximizes {acquisition} acquisition")
    print(f"       ROBOT distance mode: {robot_distance_mode}")
    print(f"       ROBOT tau quantile: {robot_tau_quantile}")
    print(f"       ROBOT initial trust-region length: {robot_initial_length}")
    print(
        "       ROBOT trust-region update tolerances: "
        f"success={robot_success_tolerance}, failure={robot_failure_tolerance}"
    )
    print("    2. NMMA simulates real lightcurve for candidate")
    print("    3. Add result (detectable/not) to training data")
    print("    4. Retrain detection GP with new data")
    print("    5. Repeat")
    print("=" * 80)

    sim_output_dir = output_folder

    closed_loop_t0 = time.time()
    optimization_function = (
        closed_loop_batched_bayesian_optimization
        if batched_robot
        else closed_loop_bayesian_optimization
    )
    (
        best_X, best_peak_mag, true_eval_log, gp_models, scaler_X, detection_gp,
        results, new_events_df
    ) = optimization_function(
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
        robot_knn_k=robot_knn_k,
        robot_tau_quantile=robot_tau_quantile,
        robot_initial_length=robot_initial_length,
        robot_rarity_n_bins=robot_rarity_n_bins,
        eicf_mc_samples=eicf_mc_samples,
        candidate_samples=candidate_samples,
        candidate_uniform_samples=candidate_uniform_samples,
        robot_region_candidates=robot_region_candidates,
        **(
            {
                "candidate_sampling_mode": candidate_sampling_mode,
                "candidate_detectability_threshold": (
                    candidate_detectability_threshold
                ),
            }
            if batched_robot
            else {}
        ),
        **(
            {
                "robot_n_regions": robot_n_regions,
                "robot_batch_regions": robot_batch_regions,
                "robot_candidates_per_region": robot_candidates_per_region,
                "parallel_simulations": parallel_simulations,
                "robot_success_tolerance": robot_success_tolerance,
                "robot_failure_tolerance": robot_failure_tolerance,
                "refit_output_gps": refit_output_gps,
                "output_gp_refit_every": output_gp_refit_every,
                "use_multitask_gp": use_multitask_gp,
            }
            if batched_robot
            else {}
        ),
        trust_region_max_resamples=trust_region_max_resamples,
        scoring_progress_every=scoring_progress_every,
        eicf_batch_size=eicf_batch_size,
        skip_detection_gp_refit=skip_detection_gp_refit,
        checkpoint_every=checkpoint_every,
        resume=resume,
        checkpoint_file=checkpoint_file,
        max_runtime_hours=max_runtime_hours,
        debug_candidate_details=debug_candidate_details,
        track_all_candidate_probs=track_all_candidate_probs,
    )
    last_job_closed_loop_seconds = time.time() - closed_loop_t0
    closed_loop_seconds = float(
        results.get("cumulative_runtime_seconds", last_job_closed_loop_seconds)
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
        'completed_batches': (
            int(timing_df['batch_iteration'].max()) + 1
            if not timing_df.empty and 'batch_iteration' in timing_df
            else len(true_eval_log)
        ),
        'best_peak_magnitude': best_peak_mag if best_peak_mag != np.inf else None,
        'best_parameters': str(best_X) if best_X is not None else None,
        'acquisition': acquisition,
        'robot_distance_mode': robot_distance_mode,
        'robot_knn_k': robot_knn_k,
        'robot_tau_quantile': robot_tau_quantile,
        'robot_initial_length': robot_initial_length,
        'robot_success_tolerance': robot_success_tolerance,
        'robot_failure_tolerance': robot_failure_tolerance,
        'eicf_mc_samples': eicf_mc_samples,
        'candidate_samples': candidate_samples,
        'candidate_uniform_samples': candidate_uniform_samples,
        'robot_region_candidates': robot_region_candidates,
        'robot_n_regions': robot_n_regions,
        'robot_batch_regions': robot_batch_regions,
        'robot_candidates_per_region': robot_candidates_per_region,
        'parallel_simulations': parallel_simulations,
        'trust_region_max_resamples': trust_region_max_resamples,
        'eicf_batch_size': eicf_batch_size,
        'use_multitask_gp': use_multitask_gp,
        'skip_detection_gp_refit': skip_detection_gp_refit,
        'checkpoint_every': checkpoint_every,
        'debug_candidate_details': debug_candidate_details,
        'track_all_candidate_probs': track_all_candidate_probs,
        'refit_output_gps': refit_output_gps,
        'output_gp_refit_every': output_gp_refit_every,
        'initial_gp_seconds': initial_gp_seconds,
        'initial_gp_source': initial_gp_source,
        'initial_multitask_gp_seconds': initial_multitask_gp_seconds,
        'closed_loop_seconds': closed_loop_seconds,
        'last_job_closed_loop_seconds': last_job_closed_loop_seconds,
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


def main():
    parser = argparse.ArgumentParser(description="Run one bo_cdf_eicf_robot1 closed-loop rollout.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible rollout labels/candidates.")
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=30,
        help=(
            "Number of BO iterations. In batched mode this is the number of "
            "batches, so the maximum event count is n_iterations multiplied "
            "by --robot-batch-regions."
        ),
    )
    parser.add_argument(
        "--acquisition",
        choices=["ROBOT_EICF"],
        default="ROBOT_EICF",
        help="Paper method: EI-CF plus h-space ROBOT filtering.",
    )
    parser.add_argument("--output-folder", default=None, help="Optional explicit output folder for this run.")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    parser.add_argument(
        "--robot-distance-mode",
        choices=["tail_l1", "knn_l1"],
        default="tail_l1",
        help=(
            "ROBOT h-space diversity distance: nearest-event L1 (tail_l1) or "
            "mean L1 to the k nearest selected events (knn_l1)."
        ),
    )
    parser.add_argument(
        "--robot-knn-k",
        type=int,
        default=5,
        help="Number of selected neighbors averaged by --robot-distance-mode knn_l1.",
    )
    parser.add_argument(
        "--robot-tau-quantile",
        type=float,
        default=25.0,
        help=(
            "Upper-tail percentage used once to fix tau from initial measurable-event "
            "distances. For example, 25 fixes tau at the 75th percentile."
        ),
    )
    parser.add_argument(
        "--robot-initial-length",
        type=float,
        default=0.6,
        help="Initial ROBOT trust-region side length in rank-scaled h-space (0.05 to 1.0).",
    )
    parser.add_argument(
        "--robot-success-tolerance",
        type=int,
        default=10,
        help="Consecutive successful region updates before doubling its length.",
    )
    parser.add_argument(
        "--robot-failure-tolerance",
        type=int,
        default=32,
        help="Consecutive failed region updates before halving its length.",
    )
    parser.add_argument(
        "--eicf-mc-samples",
        type=int,
        default=16,
        help="Monte Carlo samples per candidate for the composite active-search probability.",
    )
    parser.add_argument(
        "--candidate-samples",
        type=int,
        default=1500,
        help="Total number of candidate points scored per iteration.",
    )
    parser.add_argument(
        "--candidate-uniform-samples",
        type=int,
        default=600,
        help="Late-run uniform subset; the sampler uses up to twice this number early in the run.",
    )
    parser.add_argument(
        "--trust-region-max-resamples",
        type=int,
        default=20,
        help=(
            "Maximum candidate-pool draws used to find at least one point "
            "inside an active ROBOT trust region. The run stops instead of "
            "falling back to candidates outside the trust regions."
        ),
    )
    parser.add_argument(
        "--robot-region-candidates",
        type=int,
        default=300,
        help=(
            "Targeted candidates added around each ranked h-space trust-region "
            "center. These are combined with --candidate-samples into one shared "
            "pool, scored once, and filtered sequentially by rank."
        ),
    )
    parser.add_argument(
        "--candidate-sampling-mode",
        choices=("mixed", "uniform_detectable"),
        default="mixed",
        help=(
            "Use the legacy mixed local/global pool or draw the entire pool "
            "uniformly over the input bounds and retain candidates above a "
            "surrogate detectability threshold."
        ),
    )
    parser.add_argument(
        "--candidate-detectability-threshold",
        type=float,
        default=0.5,
        help=(
            "Minimum EI-CF active-search detectability score retained in "
            "uniform_detectable sampling mode."
        ),
    )
    parser.add_argument(
        "--robot-n-regions",
        type=int,
        default=10,
        help="Number M of ranked output-space ROBOT trust regions to maintain.",
    )
    parser.add_argument(
        "--robot-batch-regions",
        type=int,
        default=None,
        help=(
            "Number of ranked regions processed per BO step (default: all M). "
            "Set this equal to --robot-n-regions to follow the simultaneous "
            "ranked-region ROBOT schedule. "
            "The maximum batch size is this value multiplied by "
            "--robot-candidates-per-region."
        ),
    )
    parser.add_argument(
        "--robot-candidates-per-region",
        type=int,
        default=1,
        help=(
            "Maximum candidates selected from each ranked trust region in one "
            "BO batch (default: one, matching the paper-facing ROBOT schedule). "
            "Diversity is enforced hierarchically between regions."
        ),
    )
    parser.add_argument(
        "--parallel-simulations",
        type=int,
        default=1,
        help="Maximum NMMA event simulations run concurrently within a batch.",
    )
    parser.add_argument(
        "--scoring-progress-every",
        type=int,
        default=25,
        help="Print EI-CF scoring progress after this many candidates. Use 0 to disable.",
    )
    parser.add_argument(
        "--eicf-batch-size",
        type=int,
        default=128,
        help="Candidate batch size for vectorized EI-CF posterior scoring.",
    )
    parser.add_argument(
        "--use-multitask-gp",
        action="store_true",
        help="Use BoTorch MultiTaskGP for h(x). Default is fast independent sklearn GPs.",
    )
    parser.add_argument(
        "--skip-detection-gp-refit",
        action="store_true",
        help="Skip the scalar detection-GP refit inside the loop. EI-CF uses h-surrogate GPs, so this saves diagnostic overhead.",
    )
    parser.add_argument(
        "--refit-output-gps",
        action="store_true",
        help=(
            "Refit the h-output GP models after each completed BO batch. "
            "This is separate from the scalar detection GP."
        ),
    )
    parser.add_argument(
        "--output-gp-refit-every",
        type=int,
        default=1,
        help="Refit h-output GPs every N completed batches (default: 1).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Write CSV files and a restartable BO-state checkpoint every N iterations.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the BO checkpoint in the output folder.",
    )
    parser.add_argument(
        "--checkpoint-file",
        default=None,
        help="Optional checkpoint path. Default: OUTPUT_FOLDER/bo_state_checkpoint.joblib.",
    )
    parser.add_argument(
        "--max-runtime-hours",
        type=float,
        default=None,
        help="Stop cleanly after approximately this many hours and save a restart checkpoint.",
    )
    parser.add_argument(
        "--debug-candidate-details",
        action="store_true",
        help="Print expensive top-candidate diagnostics each iteration.",
    )
    parser.add_argument(
        "--track-all-candidate-probs",
        action="store_true",
        help="Store detection probabilities for every scored candidate for diagnostic plots. Slower and memory-heavy for long runs.",
    )
    args = parser.parse_args()

    run_single(
        seed=args.seed,
        n_iterations=args.n_iterations,
        acquisition=args.acquisition,
        output_folder=args.output_folder,
        make_plots=not args.no_plots,
        robot_distance_mode=args.robot_distance_mode,
        robot_knn_k=args.robot_knn_k,
        robot_tau_quantile=args.robot_tau_quantile,
        robot_initial_length=args.robot_initial_length,
        robot_success_tolerance=args.robot_success_tolerance,
        robot_failure_tolerance=args.robot_failure_tolerance,
        eicf_mc_samples=args.eicf_mc_samples,
        candidate_samples=args.candidate_samples,
        candidate_uniform_samples=args.candidate_uniform_samples,
        robot_region_candidates=args.robot_region_candidates,
        candidate_sampling_mode=args.candidate_sampling_mode,
        candidate_detectability_threshold=args.candidate_detectability_threshold,
        robot_n_regions=args.robot_n_regions,
        robot_batch_regions=args.robot_batch_regions,
        robot_candidates_per_region=args.robot_candidates_per_region,
        parallel_simulations=args.parallel_simulations,
        trust_region_max_resamples=args.trust_region_max_resamples,
        scoring_progress_every=args.scoring_progress_every,
        eicf_batch_size=args.eicf_batch_size,
        use_multitask_gp=args.use_multitask_gp,
        skip_detection_gp_refit=args.skip_detection_gp_refit,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        checkpoint_file=args.checkpoint_file,
        max_runtime_hours=args.max_runtime_hours,
        debug_candidate_details=args.debug_candidate_details,
        track_all_candidate_probs=args.track_all_candidate_probs,
        refit_output_gps=args.refit_output_gps,
        output_gp_refit_every=args.output_gp_refit_every,
    )


if __name__ == "__main__":
    main()
