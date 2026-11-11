"""
Closed-Loop Bayesian Optimization
GP proposes → NMMA simulates → Retrain → Repeat
"""

import json
import os
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.exceptions import ConvergenceWarning
from typing import Dict, Optional, Tuple

from models import build_gp_models
from models.simulator import build_gw_params_from_X
from optimization.acquisition import (
    expected_improvement, 
    sample_near_detectable,
    objective_with_uncertainty,
    expected_improvement_composite,
    expected_improvement_composite_batch,
    sample_around_robot_center,
)
from optimization.objectives import (
    objective_function_margin,
    objective_function_probability,
    objective_function_probability_batch,
)
from optimization.robot_h import (
    ROBOT_ACQUISITIONS,
    build_robot_state,
    rebuild_ranked_robot_centers,
    robot_ranked_region_mask,
    robot_region_input_center,
    select_robot_candidate,
    update_robot_state,
)
from utils.detection import DetectionCalculator
from utils.embright import compute_embright_classification
from utils.h_columns import H_OUTPUT_COLUMNS


def _true_peak_magnitudes_from_lc(output_dir: str, detection_filters: list) -> Dict[str, float]:
    """Extract brightest true peak magnitude per requested filter from lc.csv."""
    lc_file = os.path.join(output_dir, "lc.csv")
    if not os.path.exists(lc_file) or os.path.getsize(lc_file) == 0:
        return {}

    df = pd.read_csv(lc_file)
    if df.empty or "mag" not in df.columns or "filter" not in df.columns:
        return {}

    peaks = {}
    requested = set(detection_filters)
    valid = df[(df["mag"] != 99.0) & df["mag"].notna()].copy()
    for filt, sub_df in valid.groupby("filter"):
        filt_key = str(filt).replace("ztf", "")
        if filt_key in requested:
            peaks[filt_key] = float(sub_df["mag"].min())
    return peaks


def _sample_active_search_label(
    output_dir: str,
    detection_calc: DetectionCalculator,
    detection_filters: list,
) -> Tuple[int, float, Dict, Dict[str, float]]:
    """Sample the active-search 0/1 label from the true light-curve CDF probability."""
    peak_mags = _true_peak_magnitudes_from_lc(output_dir, detection_filters)
    prob_detect, filter_probs = detection_calc.compute_detection_probability(peak_mags)
    rollout_label = int(np.random.random() < prob_detect)
    return rollout_label, prob_detect, filter_probs, peak_mags


def _as_scalar(value, default=np.nan):
    """Read a scalar from NMMA json values that may be stored as one-item lists."""
    try:
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 0:
                return default
            return float(value[0])
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_injection_ejecta(output_dir: str) -> Dict[str, float]:
    """Read ejecta masses from the generated injection.json, if NMMA wrote them."""
    injection_file = os.path.join(output_dir, "injection.json")
    if not os.path.exists(injection_file):
        return {}

    try:
        with open(injection_file) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}

    if "injections" in payload and isinstance(payload["injections"], dict):
        content = payload["injections"].get("content", payload["injections"])
    else:
        content = payload.get("content", payload) if isinstance(payload, dict) else {}

    ejecta = {}
    for key in ("log10_mej_dyn", "log10_mej_wind"):
        value = _as_scalar(content.get(key, np.nan))
        if np.isfinite(value):
            ejecta[key] = value
    return ejecta


def closed_loop_bayesian_optimization(
    X_train: np.ndarray,
    y_train_mags: Dict,
    y_train_detection: np.ndarray,
    flat_limits: Dict,
    runner,
    base_sim_output_dir: str,
    gp_models: Optional[Dict] = None,
    scaler_X: Optional[StandardScaler] = None,
    detection_gp: Optional = None,
    n_iterations: int = 5,
    acquisition: str = 'ROBOT_EICF',
    xi: float = 0.01,
    kappa: float = 2.0,
    exposure_time: int = 180,
    bounds: Optional[list] = None,
    feature_columns: Optional[list] = None,
    detection_filters: Optional[list] = None,
    robot_distance_mode: str = "tail_l1",
    robot_knn_k: int = 5,
    robot_tau_quantile: float = 25.0,
    robot_initial_length: float = 0.6,
    robot_rarity_n_bins: int = 20,
    eicf_mc_samples: int = 16,
    candidate_samples: int = 1500,
    candidate_uniform_samples: int = 600,
    robot_region_candidates: int = 300,
    trust_region_max_resamples: int = 20,
    scoring_progress_every: int = 25,
    eicf_batch_size: int = 128,
    skip_detection_gp_refit: bool = False,
    checkpoint_every: int = 1,
    resume: bool = False,
    checkpoint_file: Optional[str] = None,
    max_runtime_hours: Optional[float] = None,
    debug_candidate_details: bool = False,
    track_all_candidate_probs: bool = False,
) -> Tuple:
    """
    CLOSED-LOOP BAYESIAN OPTIMIZATION

    STEP-BY-STEP:
    1. GP proposes candidate that maximizes ROBOT_EICF
    2. NMMA simulates real lightcurve for candidate
    3. Add result (detectable/not) to training data
    4. Retrain detection GP with new data
    5. Repeat until n_iterations complete

    Returns:
        best_X, best_true_peak_mag, true_eval_log, gp_models, scaler_X, detection_gp,
        results (dict for plot_optimization_results / plot_parameter_exploration / plot_objective_improvement),
        new_events_df (dataframe for plot_optimization_results)
    """

    if bounds is None:
        raise ValueError("bounds must be provided")

    if acquisition != "ROBOT_EICF":
        raise ValueError("Only ROBOT_EICF is supported in this paper-facing folder.")
    if robot_distance_mode not in {"tail_l1", "knn_l1"}:
        raise ValueError("robot_distance_mode must be tail_l1 or knn_l1.")
    if int(robot_knn_k) < 1:
        raise ValueError("robot_knn_k must be at least 1.")

    if feature_columns is None:
        feature_columns = ['mass1', 'mass2', 'spin1z', 'spin2z', 'inclination', 'distance']

    if detection_filters is None:
        detection_filters = ['g', 'r', 'i', 'u', 'z', 'y']

    detection_calc = DetectionCalculator(
        detection_filters=detection_filters,
        flat_limits=flat_limits
    )

    # STEP 0: Build initial GPs if not provided
    if gp_models is None or scaler_X is None:
        print("  Building initial GP models from training data...")
        gp_models, scaler_X, detection_gp = build_gp_models(
            X_train, y_train_mags, y_train_detection
        )

    os.makedirs(base_sim_output_dir, exist_ok=True)
    if checkpoint_file is None:
        checkpoint_file = os.path.join(base_sim_output_dir, "bo_state_checkpoint.joblib")
    checkpoint_file = os.path.abspath(checkpoint_file)

    # Initialize training pool
    X_arr = X_train if isinstance(X_train, np.ndarray) else np.asarray(X_train)
    X_true_pool = X_arr.copy()
    y_true_pool = np.asarray(y_train_detection, dtype=float).copy()
    n_train = len(X_arr)

    initial_probs = objective_function_probability_batch(
        X_arr, gp_models, scaler_X, flat_limits, detection_calc=detection_calc
    )
    observed_composite_scores = list(np.asarray(initial_probs, dtype=float))

    robot_state = None
    if acquisition in ROBOT_ACQUISITIONS:
        robot_state = build_robot_state(
            y_train_mags=y_train_mags,
            g_train=initial_probs,
            X_train=X_arr,
            n_centers=min(10, max(1, n_iterations)),
            useful_threshold=0.5,
            tau_quantile=robot_tau_quantile,
            distance_mode=robot_distance_mode,
            knn_k=robot_knn_k,
            rarity_n_bins=robot_rarity_n_bins,
            distance_bounds=bounds[-1],
            initial_trust_region_length=robot_initial_length,
        )
        if robot_state is None:
            print("  Warning: could not initialize ROBOT h-space state; using plain EI-CF selection.")
        else:
            print("  ROBOT h-space diversity initialized:")
            print(f"    h columns: {robot_state['h_columns']}")
            print(f"    centers: {len(robot_state['centers_z'])}")
            print(f"    h scaling: {robot_state.get('h_scaling', 'unknown')}")
            print(f"    distance mode: {robot_state.get('distance_mode', 'unknown')}")
            if robot_distance_mode == "knn_l1":
                print(f"    k nearest selected events: {robot_state['knn_k']}")
            print(f"    tau quantile: {robot_state.get('tau_quantile', robot_tau_quantile):.1f}")
            print(f"    diversity threshold tau: {robot_state['tau']:.4f}")
            print("    trust region: rectangular boxes in [0, 1] rank-scaled h-space")
            print(f"    initial length_i: {robot_state['trust_regions'][0].length:.4f}")
            print(f"    length_min: {robot_state['trust_regions'][0].length_min:.6f}")
            print(f"    length_max: {robot_state['trust_regions'][0].length_max:.4f}")

    best_true_peak_mag = np.inf
    best_X = None
    true_eval_log = []

    # ---- tracking structures for plotting ----
    results = {
        'probs_history': [],       # best CDF detection prob after each iteration
        'probs_all': [],           # CDF detection prob for every candidate proposed
        'score_history': [],       # best detection margin (sigma) after each iteration
        'improved_iteration': [],  # iteration indices where a new best TRUE detection was accepted
        'X_all': [],                # every point ever evaluated (training + BO candidates)
        'margin_all': [],          # detection margin (sigma) for every point in X_all
        'source': [],               # 'training' or 'bo_candidate' tag per point in X_all
    }
    new_events_records = []  # rows for new_events_df

    # Seed X_all/margin_all/source with the original training data
    initial_margins = []
    for x in X_arr:
        margin = objective_function_margin(x, gp_models, scaler_X, flat_limits, detection_calc)
        initial_margins.append(margin)
        results['X_all'].append(x)
        results['margin_all'].append(margin)
        results['source'].append('training')
    running_best_margin = float(np.nanmax(initial_margins)) if initial_margins else -np.inf

    start_iteration = 0
    elapsed_seconds_before = 0.0
    checkpoint_settings = {
        "acquisition": acquisition,
        "robot_distance_mode": robot_distance_mode,
        "robot_tau_quantile": float(robot_tau_quantile),
        "robot_initial_length": float(robot_initial_length),
        "eicf_mc_samples": int(eicf_mc_samples),
        "candidate_samples": int(candidate_samples),
        "candidate_uniform_samples": int(candidate_uniform_samples),
        "robot_region_candidates": int(robot_region_candidates),
        "eicf_batch_size": int(eicf_batch_size),
        "skip_detection_gp_refit": bool(skip_detection_gp_refit),
        "feature_columns": list(feature_columns),
    }
    if robot_distance_mode == "knn_l1":
        checkpoint_settings["robot_knn_k"] = int(robot_knn_k)

    if resume:
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(
                f"Cannot resume because checkpoint does not exist: {checkpoint_file}"
            )
        print(f"  Loading restart checkpoint: {checkpoint_file}", flush=True)
        state = joblib.load(checkpoint_file)
        if state.get("checkpoint_version") != 2:
            raise ValueError(
                "This checkpoint predates ranked output-space ROBOT. Start a new "
                "output folder so the changed algorithm is not mixed with the old run."
            )
        saved_settings = state.get("settings", {})
        mismatches = {
            key: (saved_settings.get(key), value)
            for key, value in checkpoint_settings.items()
            if saved_settings.get(key) != value
        }
        if mismatches:
            details = ", ".join(
                f"{key}: saved={old!r}, requested={new!r}"
                for key, (old, new) in mismatches.items()
            )
            raise ValueError(
                "Resume settings differ from the checkpoint. Use the same BO settings. "
                + details
            )

        start_iteration = int(state["completed_iterations"])
        if start_iteration > n_iterations:
            raise ValueError(
                f"Checkpoint already contains {start_iteration} iterations, which is "
                f"greater than requested target {n_iterations}."
            )
        X_true_pool = state["X_true_pool"]
        y_true_pool = state["y_true_pool"]
        observed_composite_scores = state["observed_composite_scores"]
        robot_state = state["robot_state"]
        best_true_peak_mag = state["best_true_peak_mag"]
        best_X = state["best_X"]
        true_eval_log = state["true_eval_log"]
        results = state["results"]
        new_events_records = state["new_events_records"]
        running_best_margin = state["running_best_margin"]
        detection_gp = state.get("detection_gp", detection_gp)
        elapsed_seconds_before = float(state.get("elapsed_seconds", 0.0))
        np.random.set_state(state["numpy_random_state"])
        print(
            f"  Resuming at iteration {start_iteration + 1}; "
            f"{start_iteration}/{n_iterations} iterations are complete.",
            flush=True,
        )

    run_segment_t0 = time.time()

    def _write_checkpoint(completed_iterations, force=False):
        if not true_eval_log:
            return
        if not force and (checkpoint_every is None or checkpoint_every <= 0):
            return
        pd.DataFrame(true_eval_log).to_csv(
            os.path.join(base_sim_output_dir, "closed_loop_results.csv"),
            index=False,
        )
        if new_events_records:
            pd.DataFrame(new_events_records).to_csv(
                os.path.join(base_sim_output_dir, "new_events_with_h.csv"),
                index=False,
            )
        state = {
            "checkpoint_version": 2,
            "completed_iterations": int(completed_iterations),
            "target_iterations": int(n_iterations),
            "settings": checkpoint_settings,
            "X_true_pool": X_true_pool,
            "y_true_pool": y_true_pool,
            "observed_composite_scores": observed_composite_scores,
            "robot_state": robot_state,
            "best_true_peak_mag": best_true_peak_mag,
            "best_X": best_X,
            "true_eval_log": true_eval_log,
            "results": results,
            "new_events_records": new_events_records,
            "running_best_margin": running_best_margin,
            "detection_gp": detection_gp,
            "numpy_random_state": np.random.get_state(),
            "elapsed_seconds": elapsed_seconds_before + (time.time() - run_segment_t0),
        }
        checkpoint_dir = os.path.dirname(checkpoint_file)
        os.makedirs(checkpoint_dir, exist_ok=True)
        temporary_file = checkpoint_file + ".tmp"
        joblib.dump(state, temporary_file)
        os.replace(temporary_file, checkpoint_file)

    # MAIN LOOP: Iterate n_iterations times
    for iteration in range(start_iteration, n_iterations):
        iteration_t0 = time.time()

        print(f"\n{'='*60}")
        print(f"CLOSED-LOOP ITERATION {iteration + 1}/{n_iterations}")
        print(f"{'='*60}")

        # ============================================================
        # STEP 1: PROPOSE CANDIDATE USING GP SURROGATE
        # ============================================================
        print(f"\n  [1] Proposing candidate from GP...")

        print(f"      Computing active-search utility from the composite h posterior...")
        best_score_for_acq = float(np.nanmax(observed_composite_scores))
        best_prob = best_score_for_acq
        
        print(f"      Current best observed composite value f*=P_detect: {best_prob:.4f}")
        print("      NOTE: Active-search score is E[g(h(x))], estimated from g/i/r posterior samples")
        print("      NOTE: ROBOT applies dynamic h-space diversity and distance-bin balancing")

        sampling_attempts = 0
        selected_region_rank = None
        robot_state = rebuild_ranked_robot_centers(robot_state)
        trust_regions = robot_state.get("trust_regions", [])
        first_rank = int(robot_state.get("next_region_rank", 0)) % len(trust_regions)
        region_order = [
            (first_rank + offset) % len(trust_regions)
            for offset in range(len(trust_regions))
        ]
        for region_rank in region_order:
            center_x = robot_region_input_center(robot_state, region_rank)
            for region_attempt in range(1, trust_region_max_resamples + 1):
                sampling_attempts += 1
                print(
                    f"      Sampling for ranked ROBOT region T_{region_rank + 1} "
                    f"(attempt {region_attempt}/{trust_region_max_resamples})..."
                )
                X_candidates = sample_near_detectable(
                    X_true_pool, y_true_pool, bounds,
                    n_samples=candidate_samples,
                    spread_factor=0.2,
                    n_uniform=candidate_uniform_samples,
                    iteration=iteration,
                    max_iterations=n_iterations,
                )
                n_targeted = min(int(robot_region_candidates), len(X_candidates))
                if center_x is not None and n_targeted > 0:
                    X_candidates[:n_targeted] = sample_around_robot_center(
                        center_x,
                        bounds,
                        n_samples=n_targeted,
                        spread_factor=max(
                            0.03,
                            0.15 * float(trust_regions[region_rank].length),
                        ),
                    )
                    np.random.shuffle(X_candidates)
                    print(
                        f"      Replaced {n_targeted} pool entries with inputs "
                        f"targeted around the T_{region_rank + 1} incumbent."
                    )
                feasible_mask = robot_ranked_region_mask(
                    X_candidates,
                    gp_models,
                    scaler_X,
                    robot_state,
                    region_rank,
                )
                n_inside = int(np.sum(feasible_mask))
                print(
                    f"      Candidates feasible for T_{region_rank + 1}: "
                    f"{n_inside}/{len(X_candidates)}"
                )
                if n_inside:
                    selected_region_rank = int(region_rank)
                    break
            if selected_region_rank is not None:
                break

            # Output-space boxes cannot be sampled directly. If repeated input
            # pools miss one, expand that region and let the next rank try.
            region = trust_regions[region_rank]
            old_length = float(region.length)
            region.length = min(2.0 * region.length, region.length_max)
            region.failure_counter = 0
            print(
                f"      Skipping empty T_{region_rank + 1}; expanded length "
                f"from {old_length:.4f} to {region.length:.4f}."
            )

        if selected_region_rank is None:
            for region in trust_regions:
                region.length = float(robot_state["initial_trust_region_length"])
                region.success_counter = 0
                region.failure_counter = 0
            raise RuntimeError(
                "All ranked ROBOT output-space regions were empty after targeted "
                "sampling. Their lengths were reset in memory; rerun with more "
                "--robot-region-candidates or a larger initial length."
            )

        print(f"      Scoring {len(X_candidates)} candidates with {acquisition} active search "
              f"(n_mc={eicf_mc_samples}, batch_size={eicf_batch_size})...", flush=True)
        t_score = time.time()
        eicf_scores = expected_improvement_composite_batch(
            X_candidates,
            gp_models,
            scaler_X,
            flat_limits,
            best_score_for_acq,
            detection_calc=detection_calc,
            n_mc=eicf_mc_samples,
            xi=xi,
            batch_size=eicf_batch_size,
            progress_every=max(1, scoring_progress_every // max(eicf_batch_size, 1)),
            active_search=True,
        )
        acq_values = -np.asarray(eicf_scores, dtype=float)
        candidate_scoring_seconds = time.time() - t_score
        print(f"      Candidate scoring finished in {candidate_scoring_seconds:.1f} s")
        robot_diagnostics = {}
        robot_selection_seconds = 0.0
        if acquisition in ROBOT_ACQUISITIONS and robot_state is not None:
            print("      Applying ROBOT h-space trust/diversity filter...")
            t_robot = time.time()
            eicf_scores = -acq_values
            best_idx, robot_diagnostics = select_robot_candidate(
                X_candidates=X_candidates,
                direct_scores=eicf_scores,
                gp_models=gp_models,
                scaler_X=scaler_X,
                robot_state=robot_state,
                region_index=selected_region_rank,
            )
            robot_selection_seconds = time.time() - t_robot
            print(f"      ROBOT filtering finished in {robot_selection_seconds:.1f} s")
            if robot_diagnostics.get("robot_used"):
                print(f"      ROBOT feasible candidates: {robot_diagnostics['feasible_candidates']}/{len(X_candidates)}")
                print(f"      ROBOT candidates inside h trust regions: {robot_diagnostics['trust_region_candidates']}/{len(X_candidates)}")
                print(
                    f"      ROBOT fixed tau used: "
                    f"{robot_diagnostics['robot_tau_used']:.4f}"
                )
                print(
                    f"      ROBOT selected ranked region: "
                    f"T_{int(robot_diagnostics['robot_region_rank']) + 1}"
                )
                if not robot_diagnostics.get("robot_tau_satisfied", True):
                    print(
                        "      ROBOT fixed diversity constraint unavailable; "
                        f"best available diversity={robot_diagnostics.get('robot_best_available_diversity')}, "
                        f"shortfall={robot_diagnostics.get('robot_tau_shortfall')}"
                    )
                print(f"      ROBOT distance mode: {robot_diagnostics.get('robot_distance_mode', 'unknown')}")
                if robot_diagnostics.get("robot_distance_bin") is not None:
                    print(
                        "      ROBOT selected distance bin: "
                        f"[{robot_diagnostics['robot_distance_bin_lower']:.1f}, "
                        f"{robot_diagnostics['robot_distance_bin_upper']:.1f}] Mpc "
                        f"(detectable count before selection="
                        f"{robot_diagnostics['robot_distance_bin_count_before']})"
                    )
                print(f"      ROBOT selected trust-region length: {robot_diagnostics['robot_length']:.4f} "
                      f"(min={robot_diagnostics['robot_length_min']:.6f}, "
                      f"max={robot_diagnostics['robot_length_max']:.4f})")
                print(f"      ROBOT selected h-center distance: {robot_diagnostics['robot_center_distance']:.4f}")
                if robot_diagnostics.get("robot_selected_distance") is not None:
                    print(f"      ROBOT selected distance from previous new h points: "
                          f"{robot_diagnostics['robot_selected_distance']:.4f}")
                print(f"      ROBOT blended score: {robot_diagnostics['robot_blended_score']:.4f} "
                      f"(acq={robot_diagnostics['robot_acquisition_score_norm']:.4f}, "
                      f"div={robot_diagnostics['robot_diversity_score_norm']:.4f}, "
                      f"w_div={robot_diagnostics['robot_diversity_weight']:.2f})")
            else:
                print(f"      ROBOT filter skipped: {robot_diagnostics.get('reason', 'unknown')}")
        else:
            best_idx = np.argmin(acq_values)
        candidate_X = X_candidates[best_idx]
        best_acq = -acq_values[best_idx]

        if debug_candidate_details:
            print(f"\n      Top 5 candidates by acquisition value:")
            top_indices = np.argsort(acq_values)[:5]
            for i, idx in enumerate(top_indices):
                acq_val = -acq_values[idx]
                X_top = X_candidates[idx]
                score, uncertainty, best_filter = objective_with_uncertainty(
                    X_top, gp_models, scaler_X, flat_limits, detection_calc, verbose=False
                )
                print(f"        #{i+1}: {acquisition}={acq_val:.6f}, mean P_detect={score:.4f}, "
                      f"limit_sigma={uncertainty:.4f}, filter={best_filter}")
                print(f"             params: {dict(zip(feature_columns, X_top))}")

        # Track CDF probability + margin for every scored candidate this iteration
        if track_all_candidate_probs:
            candidate_probs = objective_function_probability_batch(
                X_candidates, gp_models, scaler_X, flat_limits, detection_calc=detection_calc
            )
            results['probs_all'].extend(candidate_probs)

        formatted_candidate = ", ".join(f"{name}={val:.1f}" for name, val in zip(feature_columns, candidate_X))
        print(f"\n      Proposed candidate: [{formatted_candidate}]")
        print(f"      Acquisition value ({acquisition}): {best_acq:.6f}")
        
        # DEBUG: Check the proposed candidate's active-search probability
        score, uncertainty, best_filter = objective_with_uncertainty(
            candidate_X,
            gp_models,
            scaler_X,
            flat_limits,
            detection_calc,
            verbose=debug_candidate_details,
        )
        if debug_candidate_details:
            expected_improvement_composite(
                candidate_X, gp_models, scaler_X, flat_limits, best_score_for_acq,
                detection_calc=detection_calc, n_mc=eicf_mc_samples,
                xi=xi, verbose=True
            )
        print(f"      Candidate active-search P_detect: {score:.4f} (from {best_filter})")
        print(f"      Candidate limit uncertainty: {uncertainty:.4f}")

        # ============================================================
        # STEP 2: RUN REAL NMMA SIMULATION
        # ============================================================
        print(f"\n  [2] Running REAL NMMA simulation...")

        gw_params = build_gw_params_from_X(candidate_X, feature_columns)
        kilonova_params = runner.map_gw_to_kilonova(gw_params)

        for i, name in enumerate(feature_columns):
            if name == 'inclination' and i < len(candidate_X):
                kilonova_params['inclination_EM'] = float(np.clip(candidate_X[i], 0.0, np.pi / 2))
            elif name == 'distance' and i < len(candidate_X):
                kilonova_params['luminosity_distance'] = float(candidate_X[i])

        output_dir = os.path.join(base_sim_output_dir, f"i{iteration:02d}")
        t_sim = time.time()
        detectable, message, peak_mag = runner.run_kilonova_simulation(
            kilonova_params,
            exposure_time=exposure_time,
            output_dir=output_dir,
            gw_params=gw_params
        )
        simulation_seconds = time.time() - t_sim

        t_post = time.time()
        rollout_label, true_detection_prob, true_filter_probs, true_peak_mags_by_filter = _sample_active_search_label(
            output_dir, detection_calc, detection_filters
        )
        injection_ejecta = _read_injection_ejecta(output_dir)
        postprocess_seconds = time.time() - t_post

        print(f"      NMMA threshold result: {'DETECTABLE' if detectable else 'NOT DETECTABLE'}")
        print(f"      Active-search rollout P_detect: {true_detection_prob:.4f}")
        print(f"      Active-search rollout label: {rollout_label}")
        print(f"      Filter probabilities: {true_filter_probs}")
        print(f"      Peak magnitude: {peak_mag:.3f}")
        print(f"      Message: {message}")

        true_eval_log.append({
            'iteration': iteration,
            **{name: candidate_X[j] for j, name in enumerate(feature_columns)},
            'true_detectable': int(detectable),
            'active_search_detectable': int(rollout_label),
            'active_search_detection_probability': true_detection_prob,
            'true_peak_mag': peak_mag,
            'acquisition_value': best_acq,
            'candidate_count': int(len(X_candidates)),
            'candidate_sampling_attempts': int(sampling_attempts),
            'candidate_scoring_seconds': candidate_scoring_seconds,
            'robot_selection_seconds': robot_selection_seconds,
            'simulation_seconds': simulation_seconds,
            'postprocess_seconds': postprocess_seconds,
            'eicf_mc_samples': int(eicf_mc_samples),
            'eicf_batch_size': int(eicf_batch_size),
            'robot_used': bool(robot_diagnostics.get('robot_used', False)),
            'robot_distance_mode': robot_diagnostics.get('robot_distance_mode'),
            'robot_tau_quantile': robot_diagnostics.get('robot_tau_quantile'),
            'robot_tau': robot_diagnostics.get('robot_tau'),
            'robot_tau_used': robot_diagnostics.get('robot_tau_used'),
            'robot_dynamic_tau_percentile': robot_diagnostics.get('robot_dynamic_tau_percentile'),
            'robot_tail_fraction': robot_diagnostics.get('robot_tail_fraction'),
            'robot_tau_satisfied': robot_diagnostics.get('robot_tau_satisfied'),
            'robot_tau_fallback_reason': robot_diagnostics.get('robot_tau_fallback_reason'),
            'robot_best_available_diversity': robot_diagnostics.get('robot_best_available_diversity'),
            'robot_tau_shortfall': robot_diagnostics.get('robot_tau_shortfall'),
            'robot_length': robot_diagnostics.get('robot_length'),
            'robot_length_min': robot_diagnostics.get('robot_length_min'),
            'robot_length_max': robot_diagnostics.get('robot_length_max'),
            'robot_center_distance': robot_diagnostics.get('robot_center_distance'),
            'robot_selected_distance': robot_diagnostics.get('robot_selected_distance'),
            'robot_nearest_selected_index': robot_diagnostics.get('robot_nearest_selected_index'),
            'robot_neighbor_selected_indices': robot_diagnostics.get('robot_neighbor_selected_indices'),
            'robot_effective_neighbor_count': robot_diagnostics.get('robot_effective_neighbor_count'),
            'robot_knn_k': robot_diagnostics.get('robot_knn_k'),
            'robot_l1_contribution_sum': robot_diagnostics.get('robot_l1_contribution_sum'),
            'robot_selection_margin': robot_diagnostics.get('robot_selection_margin'),
            'robot_center_index': robot_diagnostics.get('robot_center_index'),
            'robot_region_rank': robot_diagnostics.get('robot_region_rank'),
            'robot_diversity_weight': robot_diagnostics.get('robot_diversity_weight'),
            'robot_blended_score': robot_diagnostics.get('robot_blended_score'),
            'robot_acquisition_score_norm': robot_diagnostics.get('robot_acquisition_score_norm'),
            'robot_diversity_score_norm': robot_diagnostics.get('robot_diversity_score_norm'),
            'robot_distance_balancing_used': robot_diagnostics.get('robot_distance_balancing_used'),
            'robot_distance_bin': robot_diagnostics.get('robot_distance_bin'),
            'robot_distance_bin_count_before': robot_diagnostics.get('robot_distance_bin_count_before'),
            'robot_distance_bin_lower': robot_diagnostics.get('robot_distance_bin_lower'),
            'robot_distance_bin_upper': robot_diagnostics.get('robot_distance_bin_upper'),
        })
        for h_col, contribution in robot_diagnostics.get(
            'robot_l1_contributions', {}
        ).items():
            true_eval_log[-1][f'robot_l1_contribution_{h_col}'] = contribution
        for filt in detection_filters:
            if filt in true_peak_mags_by_filter:
                true_eval_log[-1][f'true_peak_mag_{filt}'] = true_peak_mags_by_filter[filt]

        # Track this candidate for new_events_df (plot_optimization_results)
        candidate_prob = objective_function_probability(
            candidate_X, gp_models, scaler_X, flat_limits,
            detection_gp=None, use_gp_detection=False, detection_calc=detection_calc
        )
        candidate_mags, _ = None, None
        try:
            from models import predict_magnitudes_single
            candidate_mags, _ = predict_magnitudes_single(candidate_X, gp_models, scaler_X)
        except Exception:
            candidate_mags = {}

        event_record = {name: candidate_X[j] for j, name in enumerate(feature_columns)}
        event_record['detection_probability'] = candidate_prob
        event_record['detected'] = int(rollout_label)
        event_record['nmma_threshold_detected'] = int(detectable)
        event_record['robot_used'] = bool(robot_diagnostics.get('robot_used', False))
        event_record['candidate_sampling_attempts'] = int(sampling_attempts)
        event_record['robot_distance_mode'] = robot_diagnostics.get('robot_distance_mode')
        event_record['robot_tau_quantile'] = robot_diagnostics.get('robot_tau_quantile')
        event_record['robot_tau'] = robot_diagnostics.get('robot_tau')
        event_record['robot_tau_used'] = robot_diagnostics.get('robot_tau_used')
        event_record['robot_dynamic_tau_percentile'] = robot_diagnostics.get('robot_dynamic_tau_percentile')
        event_record['robot_tail_fraction'] = robot_diagnostics.get('robot_tail_fraction')
        event_record['robot_tau_satisfied'] = robot_diagnostics.get('robot_tau_satisfied')
        event_record['robot_tau_fallback_reason'] = robot_diagnostics.get('robot_tau_fallback_reason')
        event_record['robot_best_available_diversity'] = robot_diagnostics.get('robot_best_available_diversity')
        event_record['robot_tau_shortfall'] = robot_diagnostics.get('robot_tau_shortfall')
        event_record['robot_length'] = robot_diagnostics.get('robot_length')
        event_record['robot_length_min'] = robot_diagnostics.get('robot_length_min')
        event_record['robot_length_max'] = robot_diagnostics.get('robot_length_max')
        event_record['robot_center_distance'] = robot_diagnostics.get('robot_center_distance')
        event_record['robot_selected_distance'] = robot_diagnostics.get('robot_selected_distance')
        event_record['robot_nearest_selected_index'] = robot_diagnostics.get('robot_nearest_selected_index')
        event_record['robot_neighbor_selected_indices'] = robot_diagnostics.get('robot_neighbor_selected_indices')
        event_record['robot_effective_neighbor_count'] = robot_diagnostics.get('robot_effective_neighbor_count')
        event_record['robot_knn_k'] = robot_diagnostics.get('robot_knn_k')
        event_record['robot_l1_contribution_sum'] = robot_diagnostics.get('robot_l1_contribution_sum')
        event_record['robot_selection_margin'] = robot_diagnostics.get('robot_selection_margin')
        event_record['robot_center_index'] = robot_diagnostics.get('robot_center_index')
        event_record['robot_region_rank'] = robot_diagnostics.get('robot_region_rank')
        event_record['robot_diversity_weight'] = robot_diagnostics.get('robot_diversity_weight')
        event_record['robot_blended_score'] = robot_diagnostics.get('robot_blended_score')
        event_record['robot_acquisition_score_norm'] = robot_diagnostics.get('robot_acquisition_score_norm')
        event_record['robot_diversity_score_norm'] = robot_diagnostics.get('robot_diversity_score_norm')
        event_record['robot_distance_balancing_used'] = robot_diagnostics.get('robot_distance_balancing_used')
        event_record['robot_distance_bin'] = robot_diagnostics.get('robot_distance_bin')
        event_record['robot_distance_bin_count_before'] = robot_diagnostics.get('robot_distance_bin_count_before')
        event_record['robot_distance_bin_lower'] = robot_diagnostics.get('robot_distance_bin_lower')
        event_record['robot_distance_bin_upper'] = robot_diagnostics.get('robot_distance_bin_upper')
        for h_col, contribution in robot_diagnostics.get(
            'robot_l1_contributions', {}
        ).items():
            event_record[f'robot_l1_contribution_{h_col}'] = contribution
        event_record['log10_mej_dyn'] = float(injection_ejecta.get('log10_mej_dyn', kilonova_params['log10_mej_dyn']))
        event_record['log10_mej_wind'] = float(injection_ejecta.get('log10_mej_wind', kilonova_params['log10_mej_wind']))
        event_record['ejecta_source'] = 'injection.json' if injection_ejecta else 'map_gw_to_kilonova'
        event_record['mej_dyn'] = float(10 ** event_record['log10_mej_dyn'])
        event_record['mej_wind'] = float(10 ** event_record['log10_mej_wind'])
        event_record['distance'] = float(candidate_X[feature_columns.index('distance')]) if 'distance' in feature_columns else event_record.get('distance')
        for h_col, h_val in robot_diagnostics.get('predicted_h', {}).items():
            event_record[f'robot_predicted_{h_col}'] = h_val
            if h_col not in event_record or pd.isna(event_record.get(h_col)):
                event_record[h_col] = h_val
        for h_col in H_OUTPUT_COLUMNS:
            if h_col in true_peak_mags_by_filter:
                event_record[h_col] = true_peak_mags_by_filter[h_col]
                event_record[f'true_{h_col}'] = true_peak_mags_by_filter[h_col]
            elif h_col in event_record and event_record[h_col] is not None:
                continue
            elif candidate_mags and h_col in candidate_mags:
                event_record[h_col] = candidate_mags[h_col]
        for filt, true_peak_mag in true_peak_mags_by_filter.items():
            event_record[f'true_peak_mag_{filt}'] = true_peak_mag

        candidate_snr = event_record.get('snr')
        embright = compute_embright_classification(
            mass1=event_record.get('mass1'),
            mass2=event_record.get('mass2'),
            spin1z=event_record.get('spin1z'),
            spin2z=event_record.get('spin2z'),
            snr=candidate_snr,
        )
        event_record.update(embright)
        true_eval_log[-1].update({
            key: embright[key]
            for key in ("HasNS", "HasRemnant", "HasMassGap", "embright_status")
        })
        true_eval_log[-1].update(event_record)
        new_events_records.append(event_record)

        # Track margin + source for parameter_exploration plot
        candidate_margin = objective_function_margin(
            candidate_X, gp_models, scaler_X, flat_limits, detection_calc
        )
        running_best_margin = max(running_best_margin, float(candidate_margin))
        results['X_all'].append(candidate_X)
        results['margin_all'].append(candidate_margin)
        results['source'].append('bo_candidate')

        # Update best if this is the brightest TRUE detection so far
        if rollout_label == 1 and peak_mag < best_true_peak_mag:
            best_true_peak_mag = peak_mag
            best_X = candidate_X.copy()
            results['improved_iteration'].append(iteration)
            print(f"\n  *** NEW BEST ACTIVE-SEARCH DETECTION! Peak mag: {best_true_peak_mag:.3f} ***")

        # Track running-best margin/probability history for this iteration
        results['probs_history'].append(best_prob)
        results['score_history'].append(running_best_margin)

        # ============================================================
        # STEP 3: ADD NEW DATA TO TRAINING POOL
        # ============================================================
        print(f"\n  [3] Adding new data to training pool...")

        X_true_pool = np.vstack([X_true_pool, candidate_X])
        y_true_pool = np.append(y_true_pool, float(rollout_label))
        observed_composite_scores.append(float(true_detection_prob))

        if acquisition in ROBOT_ACQUISITIONS and robot_state is not None:
            observed_h = [
                event_record.get(
                    column,
                    robot_diagnostics.get("predicted_h", {}).get(column, np.nan),
                )
                for column in robot_state["h_columns"]
            ]
            robot_state = update_robot_state(
                robot_state,
                true_detection_prob,
                x_next=candidate_X,
                h_next=observed_h,
            )
            selected_center_index = robot_diagnostics.get('robot_center_index')
            if selected_center_index is not None:
                tr_state = robot_state["trust_regions"][int(selected_center_index)]
                print("      Updated ROBOT trust-region state:")
                print(f"        center index: {selected_center_index}")
                print(f"        length_i: {tr_state.length:.4f}")
                print(f"        success_counter: {tr_state.success_counter}")
                print(f"        failure_counter: {tr_state.failure_counter}")
                print(f"        restart_triggered: {tr_state.restart_triggered}")
                restart_info = robot_state.get("last_restart")
                if restart_info is not None:
                    print(
                        "        trust-region restart: "
                        f"length {restart_info['previous_length']:.6f} -> "
                        f"{restart_info['new_length']:.4f}, "
                        f"new selected-point center={restart_info['selected_point_index']}"
                    )
                    event_record["robot_restart_performed"] = True
                    event_record["robot_restart_previous_length"] = restart_info[
                        "previous_length"
                    ]
                    true_eval_log[-1].update(event_record)
                else:
                    event_record["robot_restart_performed"] = False
                    true_eval_log[-1]["robot_restart_performed"] = False

            robot_records = [
                row for row in true_eval_log
                if row.get("robot_tau_satisfied") is not None
            ]
            satisfied_count = sum(
                bool(row.get("robot_tau_satisfied")) for row in robot_records
            )
            satisfaction_rate = satisfied_count / max(len(robot_records), 1)
            print(
                "      ROBOT cumulative tau satisfaction: "
                f"{satisfied_count}/{len(robot_records)} "
                f"({satisfaction_rate:.1%})"
            )

        print(f"      Training pool size: {len(X_true_pool)} events")
        print(f"      Detectable in pool: {np.sum(y_true_pool)}/{len(y_true_pool)}")

        # ============================================================
        # STEP 4: RETRAIN DETECTION GP WITH NEW DATA
        # ============================================================
        t_gp_fit = time.time()
        if skip_detection_gp_refit:
            print(f"\n  [4] Skipping scalar detection-GP refit (--skip-detection-gp-refit)...")
            detection_gp_fit_seconds = 0.0
        else:
            print(f"\n  [4] Retraining detection GP with new data...")

            # Keep using the scaler that was fitted with the magnitude GPs.
            # Refitting it here would make later magnitude-GP predictions use a
            # different feature transform than the one those GPs were trained on.
            X_scaled_detection = scaler_X.transform(X_true_pool)

            detection_gp = GaussianProcessRegressor(
                kernel=(
                    ConstantKernel(1.0, (1e-4, 1e4))
                    * Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e4), nu=2.5)
                    + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e1))
                ),
                n_restarts_optimizer=1,
                alpha=1e-6,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                detection_gp.fit(X_scaled_detection, y_true_pool)
        detection_gp_fit_seconds = time.time() - t_gp_fit
        iteration_seconds = time.time() - iteration_t0
        bo_overhead_seconds = (
            candidate_scoring_seconds
            + robot_selection_seconds
            + postprocess_seconds
            + detection_gp_fit_seconds
        )
        true_eval_log[-1].update({
            'detection_gp_fit_seconds': detection_gp_fit_seconds,
            'gp_refit_seconds': detection_gp_fit_seconds,
            'bo_overhead_seconds': bo_overhead_seconds,
            'iteration_seconds': iteration_seconds,
        })

        if not skip_detection_gp_refit:
            print(f"      Detection GP retrained successfully")
        if checkpoint_every and (iteration + 1) % checkpoint_every == 0:
            _write_checkpoint(iteration + 1)
            print(
                f"      Restart checkpoint saved after iteration {iteration + 1}: "
                f"{checkpoint_file}"
            )
        print(
            "      Timing: "
            f"score={candidate_scoring_seconds:.1f}s, "
            f"ROBOT={robot_selection_seconds:.1f}s, "
            f"NMMA={simulation_seconds:.1f}s, "
            f"post={postprocess_seconds:.1f}s, "
            f"GP_refit={detection_gp_fit_seconds:.1f}s, "
            f"total={iteration_seconds:.1f}s"
        )

        # ============================================================
        # STEP 5: REPORT ITERATION SUMMARY
        # ============================================================
        print(f"\n  [5] Iteration {iteration + 1} summary:")
        print(f"      Best peak mag so far: {best_true_peak_mag:.3f}" if best_true_peak_mag != np.inf else "      Best peak mag so far: None")
        if best_X is not None:
            print(f"      Best parameters so far: {dict(zip(feature_columns, best_X))}")

        if max_runtime_hours is not None:
            segment_hours = (time.time() - run_segment_t0) / 3600.0
            if segment_hours >= max_runtime_hours:
                _write_checkpoint(iteration + 1, force=True)
                print(
                    f"\nReached the per-job runtime limit of {max_runtime_hours:.2f} hours. "
                    f"Stopped cleanly after iteration {iteration + 1}."
                )
                print(f"Resume from: {checkpoint_file}")
                break

    completed_iterations = len(true_eval_log)
    _write_checkpoint(completed_iterations, force=True)
    results["cumulative_runtime_seconds"] = (
        elapsed_seconds_before + (time.time() - run_segment_t0)
    )

    # ============================================================
    # FINAL RESULTS
    # ============================================================
    if best_X is None:
        print("\n⚠️ No detectable event found across all iterations.")
    else:
        print(f"\n{'='*60}")
        print(f"✓ BEST DETECTABLE EVENT FOUND!")
        print(f"{'='*60}")
        print(f"  Peak magnitude: {best_true_peak_mag:.3f}")
        print(f"  Parameters:")
        for j, name in enumerate(feature_columns):
            if j < len(best_X):
                print(f"    {name}: {best_X[j]:.4f}")

    new_events_df = pd.DataFrame(new_events_records) if new_events_records else pd.DataFrame(
        columns=feature_columns + ['detection_probability', 'detected']
    )

    return (
        best_X, best_true_peak_mag, true_eval_log, gp_models, scaler_X, detection_gp,
        results, new_events_df
    )
