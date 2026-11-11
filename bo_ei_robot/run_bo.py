#!/usr/bin/env python3
"""
Direct-GP Bayesian Optimization baseline for detectable BNS events.
Each iteration: scalar GP on g(x)=P_detect proposes -> NMMA simulates -> retrain.
ROBOT_DIRECT_EI adds predicted h-space trust/diversity filtering before simulation.
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
from models.gp_surrogate import MULTITASK_GP_KEY, build_fast_h_diagnostic_models
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


def run_single(
    seed=None,
    n_iterations=30,
    acquisition='ROBOT_DIRECT_EI',
    output_folder=None,
    make_plots=True,
    robot_distance_mode="tail_l1",
    robot_tau_quantile=85.0,
    robot_initial_length=0.6,
    robot_success_tolerance=10,
    robot_failure_tolerance=32,
    robot_rarity_n_bins=20,
    robot_n_regions=10,
    robot_batch_regions=None,
    robot_candidates_per_region=1,
    parallel_simulations=1,
    candidate_samples=1500,
    candidate_uniform_samples=500,
    robot_region_candidates=300,
    trust_region_max_resamples=20,
    scoring_progress_every=0,
    direct_gp_restarts=5,
    direct_gp_max_fit_points=None,
    checkpoint_every=1,
    resume=False,
    checkpoint_file=None,
    max_runtime_hours=None,
    debug_candidate_details=False,
    track_all_candidate_probs=False,
    save_direct_gp=False,
    refit_output_gps=False,
    output_gp_refit_every=1,
):
    """Run one closed-loop direct-GP rollout and return paths/results."""
    if acquisition != "ROBOT_DIRECT_EI":
        raise ValueError(
            "This paper-facing folder only supports --acquisition ROBOT_DIRECT_EI."
        )
    if robot_distance_mode != "tail_l1":
        raise ValueError(
            "This paper-facing folder only supports --robot-distance-mode tail_l1."
        )
    if not 0.05 <= float(robot_initial_length) <= 1.0:
        raise ValueError("--robot-initial-length must be between 0.05 and 1.0")
    if int(robot_n_regions) < 1:
        raise ValueError("--robot-n-regions must be at least 1")
    if robot_batch_regions is None:
        robot_batch_regions = int(robot_n_regions)
    if not 1 <= int(robot_batch_regions) <= int(robot_n_regions):
        raise ValueError("--robot-batch-regions must be between 1 and --robot-n-regions")
    if int(robot_candidates_per_region) < 1:
        raise ValueError("--robot-candidates-per-region must be at least 1")
    max_batch_events = int(robot_batch_regions) * int(robot_candidates_per_region)
    if not 1 <= int(parallel_simulations) <= max_batch_events:
        raise ValueError(
            "--parallel-simulations must be between 1 and the maximum batch size"
        )

    if seed is not None:
        np.random.seed(seed)

    print("=" * 80)
    print("DIRECT-GP / ROBOT-H CLOSED-LOOP BAYESIAN OPTIMIZATION")
    print("Scalar GP direct EI + ROBOT proposes -> NMMA -> retrain")
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

    # Direct-GP acquisition uses only the scalar GP on g(x)=P_detect. We still
    # build lightweight h(x) diagnostic surrogates so new-event rows include
    # snr, area90, HasNS, HasMassGap, HasRemnant, etc.
    print("\n[3] BUILD FAST DIAGNOSTIC h(x) SURROGATES")
    print("-" * 40)
    gp_cache_dir = os.path.join(project_dir, "gp_cache")
    os.makedirs(gp_cache_dir, exist_ok=True)
    gp_cache_path = os.path.join(gp_cache_dir, "fast_diagnostic_h_models_h11.joblib")
    initial_gp_seconds = np.nan
    initial_gp_source = None

    if os.path.exists(gp_cache_path):
        print(f"  Found cached fast h(x) diagnostic models at {gp_cache_path}, loading...")
        t_gp = time.time()
        cached = joblib.load(gp_cache_path)
        gp_models = cached['gp_models']
        scaler_X = cached['scaler_X']
        detection_gp = cached.get('detection_gp')
        initial_gp_seconds = time.time() - t_gp
        initial_gp_source = "cache"
        print(f"  Loaded {len(gp_models)} fast h(x) diagnostic models")
    else:
        t_gp = time.time()
        gp_models, scaler_X, detection_gp = build_fast_h_diagnostic_models(
            X_train, y_train_mags, max_fit_points=350, random_state=(seed or 12345)
        )
        initial_gp_seconds = time.time() - t_gp
        initial_gp_source = "built"
        joblib.dump(
            {'gp_models': gp_models, 'scaler_X': scaler_X, 'detection_gp': detection_gp},
            gp_cache_path
        )
        print(f"  Saved fast h(x) diagnostic models to cache: {gp_cache_path}")

    print("  ROBOT_DIRECT_EI uses scalar-GP DIRECT_EI plus predicted h-space trust/diversity filtering.")
    print(f"  ROBOT distance mode: {robot_distance_mode}")
    print(f"  ROBOT tau quantile: {robot_tau_quantile}")

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
    print("\n[5] RUN DIRECT-GP CLOSED-LOOP BAYESIAN OPTIMIZATION")
    print("=" * 80)
    print("  Iteration 0: Initial direct scalar GP (trained on historical g(x)=P_detect)")
    print("  Each iteration:")
    print(f"    1. Candidate scored with {acquisition}")
    print("    2. NMMA simulates real lightcurve for candidate")
    print("    3. Convert true g/r/i peak magnitudes to P_detect")
    print("    4. Sample Bernoulli active-search label from P_detect")
    print("    5. Retrain direct scalar GP with the new P_detect")
    print("    6. Repeat")
    print("=" * 80)

    sim_output_dir = output_folder
    batched_robot = int(robot_batch_regions) > 1 or int(robot_candidates_per_region) > 1

    closed_loop_t0 = time.time()
    optimization_function = (
        closed_loop_batched_bayesian_optimization
        if batched_robot
        else closed_loop_bayesian_optimization
    )
    (
        best_X, best_peak_mag, true_eval_log, gp_models, scaler_X, detection_gp,
        direct_gp, direct_scaler_X, results, new_events_df
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
        robot_tau_quantile=robot_tau_quantile,
        robot_rarity_n_bins=robot_rarity_n_bins,
        **(
            {
                "robot_n_regions": robot_n_regions,
                "robot_initial_length": robot_initial_length,
                "robot_success_tolerance": robot_success_tolerance,
                "robot_failure_tolerance": robot_failure_tolerance,
                "robot_batch_regions": robot_batch_regions,
                "robot_candidates_per_region": robot_candidates_per_region,
                "parallel_simulations": parallel_simulations,
                "robot_region_candidates": robot_region_candidates,
                "trust_region_max_resamples": trust_region_max_resamples,
                "scoring_progress_every": scoring_progress_every,
                "resume": resume,
                "checkpoint_file": checkpoint_file,
                "max_runtime_hours": max_runtime_hours,
                "refit_output_gps": refit_output_gps,
                "output_gp_refit_every": output_gp_refit_every,
            }
            if batched_robot
            else {}
        ),
        candidate_samples=candidate_samples,
        candidate_uniform_samples=candidate_uniform_samples,
        direct_gp_restarts=direct_gp_restarts,
        direct_gp_max_fit_points=direct_gp_max_fit_points,
        checkpoint_every=checkpoint_every,
        debug_candidate_details=debug_candidate_details,
        track_all_candidate_probs=track_all_candidate_probs,
    )
    closed_loop_seconds = time.time() - closed_loop_t0

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    print("\n[6] SAVE RESULTS")
    print("-" * 40)

    if save_direct_gp:
        direct_model_path = os.path.join(output_folder, "direct_gp_final.joblib")
        joblib.dump(
            {"direct_gp": direct_gp, "direct_scaler_X": direct_scaler_X},
            direct_model_path,
        )
        print(f"  Saved final direct scalar GP model: direct_gp_final.joblib")

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

    def _sum_timing(col):
        if timing_df.empty or col not in timing_df:
            return 0.0
        return float(pd.to_numeric(timing_df[col], errors="coerce").fillna(0.0).sum())

    def _mean_timing(col):
        if timing_df.empty or col not in timing_df:
            return np.nan
        vals = pd.to_numeric(timing_df[col], errors="coerce").dropna()
        return float(vals.mean()) if len(vals) else np.nan

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
        'robot_success_tolerance': robot_success_tolerance,
        'robot_failure_tolerance': robot_failure_tolerance,
        'candidate_samples': candidate_samples,
        'candidate_uniform_samples': candidate_uniform_samples,
        'direct_gp_restarts': direct_gp_restarts,
        'direct_gp_max_fit_points': direct_gp_max_fit_points,
        'checkpoint_every': checkpoint_every,
        'debug_candidate_details': debug_candidate_details,
        'track_all_candidate_probs': track_all_candidate_probs,
        'save_direct_gp': save_direct_gp,
        'refit_output_gps': refit_output_gps,
        'output_gp_refit_every': output_gp_refit_every,
        'initial_gp_seconds': initial_gp_seconds,
        'initial_gp_source': initial_gp_source,
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

    if make_plots and best_X is not None and results['X_all'] and np.isfinite(results['margin_all']).any():
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

    if make_plots and true_eval_log and gp_models is not None and scaler_X is not None:
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
    if save_direct_gp:
        print("  - direct_gp_final.joblib (final scalar GP on g(x)=P_detect)")
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
    parser = argparse.ArgumentParser(description="Run one bo_cdf_direct_gp_robot1 closed-loop rollout.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible rollout labels/candidates.")
    parser.add_argument("--n-iterations", type=int, default=30, help="Number of closed-loop NMMA evaluations.")
    parser.add_argument(
        "--acquisition",
        choices=["ROBOT_DIRECT_EI"],
        default="ROBOT_DIRECT_EI",
        help="Paper comparison method: scalar-GP direct EI plus h-space ROBOT filtering.",
    )
    parser.add_argument("--output-folder", default=None, help="Optional explicit output folder for this run.")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    parser.add_argument(
        "--robot-distance-mode",
        choices=["tail_l1"],
        default="tail_l1",
        help=(
            "ROBOT h-space diversity distance used in the paper: unweighted scaled L1 with "
            "tau from the empirical pairwise-distance CDF."
        ),
    )
    parser.add_argument(
        "--robot-tau-quantile",
        type=float,
        default=85.0,
        help="Percentile of useful-event pairwise h-space distances used as tau. 85 keeps the largest 15%% tail.",
    )
    parser.add_argument(
        "--robot-initial-length", type=float, default=0.6,
        help="Initial side length of each ranked h-space trust region.",
    )
    parser.add_argument(
        "--robot-success-tolerance", type=int, default=10,
        help="Consecutive successful region updates required before expansion.",
    )
    parser.add_argument(
        "--robot-failure-tolerance", type=int, default=32,
        help="Consecutive failed region updates required before contraction.",
    )
    parser.add_argument(
        "--robot-n-regions", type=int, default=10,
        help="Number of ranked ROBOT trust regions to maintain.",
    )
    parser.add_argument(
        "--robot-batch-regions", type=int, default=None,
        help=(
            "Number of ranked regions evaluated in each BO step (default: all M). "
            "Set this equal to --robot-n-regions for the simultaneous ROBOT schedule."
        ),
    )
    parser.add_argument(
        "--robot-candidates-per-region", type=int, default=1,
        help=(
            "Maximum candidates selected from each active region per batch "
            "(default: one). Diversity is enforced hierarchically between regions."
        ),
    )
    parser.add_argument(
        "--parallel-simulations", type=int, default=1,
        help="Maximum NMMA simulations run concurrently.",
    )
    parser.add_argument(
        "--robot-region-candidates", type=int, default=300,
        help="Targeted candidates sampled around each ranked region center.",
    )
    parser.add_argument(
        "--trust-region-max-resamples", type=int, default=20,
        help="Maximum candidate-pool redraws when no region is feasible.",
    )
    parser.add_argument(
        "--scoring-progress-every", type=int, default=0,
        help="Candidate-scoring progress interval; 0 disables it.",
    )
    parser.add_argument(
        "--candidate-samples",
        type=int,
        default=1500,
        help="Number of near-detectable candidate points to score per iteration.",
    )
    parser.add_argument(
        "--candidate-uniform-samples",
        type=int,
        default=500,
        help="Number of additional uniform candidate points per iteration.",
    )
    parser.add_argument(
        "--direct-gp-restarts",
        type=int,
        default=5,
        help="n_restarts_optimizer for the scalar direct GP. Use 0 for faster exploratory long runs.",
    )
    parser.add_argument(
        "--direct-gp-max-fit-points",
        type=int,
        default=None,
        help="Optional cap on scalar-GP training points for approximate long runs. Unset keeps the exact full GP.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Write closed_loop_results.csv and new_events_with_h.csv every N iterations. Use 0 to checkpoint only at the end.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume a batched run checkpoint.")
    parser.add_argument("--checkpoint-file", default=None, help="Optional batched checkpoint path.")
    parser.add_argument(
        "--max-runtime-hours", type=float, default=None,
        help="Stop cleanly near this runtime and save a checkpoint.",
    )
    # Accepted for command compatibility with EI-CF runs; direct EI does not use MC.
    parser.add_argument("--eicf-mc-samples", type=int, default=16, help=argparse.SUPPRESS)
    parser.add_argument("--eicf-batch-size", type=int, default=512, help=argparse.SUPPRESS)
    parser.add_argument("--skip-detection-gp-refit", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--refit-output-gps",
        action="store_true",
        help=(
            "Refit the h-output GP models after each completed BO batch. "
            "The scalar direct-EI GP is always refitted."
        ),
    )
    parser.add_argument(
        "--output-gp-refit-every",
        type=int,
        default=1,
        help="Refit h-output GPs every N completed batches (default: 1).",
    )
    parser.add_argument(
        "--debug-candidate-details",
        action="store_true",
        help="Print expensive top-candidate diagnostics each iteration.",
    )
    parser.add_argument(
        "--track-all-candidate-probs",
        action="store_true",
        help="Store direct-GP probabilities for every scored candidate for diagnostic plots.",
    )
    parser.add_argument(
        "--save-direct-gp",
        action="store_true",
        help="Save the final scalar GP joblib model. Off by default for long runs.",
    )
    args = parser.parse_args()

    run_single(
        seed=args.seed,
        n_iterations=args.n_iterations,
        acquisition=args.acquisition,
        output_folder=args.output_folder,
        make_plots=not args.no_plots,
        robot_distance_mode=args.robot_distance_mode,
        robot_tau_quantile=args.robot_tau_quantile,
        robot_initial_length=args.robot_initial_length,
        robot_success_tolerance=args.robot_success_tolerance,
        robot_failure_tolerance=args.robot_failure_tolerance,
        robot_n_regions=args.robot_n_regions,
        robot_batch_regions=args.robot_batch_regions,
        robot_candidates_per_region=args.robot_candidates_per_region,
        parallel_simulations=args.parallel_simulations,
        candidate_samples=args.candidate_samples,
        candidate_uniform_samples=args.candidate_uniform_samples,
        robot_region_candidates=args.robot_region_candidates,
        trust_region_max_resamples=args.trust_region_max_resamples,
        scoring_progress_every=args.scoring_progress_every,
        direct_gp_restarts=args.direct_gp_restarts,
        direct_gp_max_fit_points=args.direct_gp_max_fit_points,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        checkpoint_file=args.checkpoint_file,
        max_runtime_hours=args.max_runtime_hours,
        debug_candidate_details=args.debug_candidate_details,
        track_all_candidate_probs=args.track_all_candidate_probs,
        save_direct_gp=args.save_direct_gp,
        refit_output_gps=args.refit_output_gps,
        output_gp_refit_every=args.output_gp_refit_every,
    )


if __name__ == "__main__":
    main()
