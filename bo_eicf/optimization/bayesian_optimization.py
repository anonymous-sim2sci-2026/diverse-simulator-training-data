"""
Closed-Loop Bayesian Optimization
GP proposes → NMMA simulates → Retrain → Repeat
"""

import os
import time
import warnings
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
    upper_confidence_bound, 
    sample_near_detectable,
    objective_with_uncertainty,
    expected_improvement_composite,
)
from optimization.objectives import objective_function_probability, objective_function_margin
from utils.detection import DetectionCalculator
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
    acquisition: str = 'EI',
    xi: float = 0.01,
    kappa: float = 2.0,
    exposure_time: int = 180,
    bounds: Optional[list] = None,
    feature_columns: Optional[list] = None,
    detection_filters: Optional[list] = None,
    candidate_samples: int = 1500,
    candidate_uniform_samples: int = 500,
    eicf_mc_samples: int = 256,
    active_search: bool = False,
) -> Tuple:
    """
    CLOSED-LOOP BAYESIAN OPTIMIZATION

    STEP-BY-STEP:
    1. GP proposes candidate that maximizes acquisition function (EI/UCB)
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

    # Initialize training pool
    X_arr = X_train if isinstance(X_train, np.ndarray) else np.asarray(X_train)
    X_true_pool = X_arr.copy()
    y_true_pool = np.asarray(y_train_detection, dtype=float).copy()
    n_train = len(X_arr)

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
    for x in X_arr:
        margin = objective_function_margin(x, gp_models, scaler_X, flat_limits, detection_calc)
        results['X_all'].append(x)
        results['margin_all'].append(margin)
        results['source'].append('training')

    # MAIN LOOP: Iterate n_iterations times
    for iteration in range(n_iterations):
        iteration_started = time.perf_counter()

        print(f"\n{'='*60}")
        print(f"CLOSED-LOOP ITERATION {iteration + 1}/{n_iterations}")
        print(f"{'='*60}")

        # ============================================================
        # STEP 1: PROPOSE CANDIDATE USING GP SURROGATE
        # ============================================================
        print(f"\n  [1] Proposing candidate from GP...")
        proposal_started = time.perf_counter()

        print(f"      Computing best active-search utility from training pool...")
        pool_probs = [
            objective_function_probability(
                x, gp_models, scaler_X, flat_limits,
                detection_gp=None, use_gp_detection=False, detection_calc=detection_calc
            )
            for x in X_true_pool
        ]
        best_score_for_acq = float(np.max(pool_probs))
        best_prob = best_score_for_acq
        
        print(f"      Current best observed composite value f*=P_detect: {best_prob:.4f}")
        if acquisition == 'EI':
            print(f"      NOTE: Using EI-CF acquisition with 11D h(x); g(h) uses g/i/r peak magnitudes")
        else:
            print(f"      NOTE: Using {acquisition} acquisition")

        def acq_func(X_flat, _best_score=best_score_for_acq):
            if acquisition == 'EI':
                return -expected_improvement(
                    X_flat, gp_models, scaler_X, flat_limits,
                    _best_score, detection_gp, use_gp_detection=False, xi=xi,
                    detection_calc=detection_calc,  # This now has filter_errors
                    n_mc=eicf_mc_samples,
                    active_search=active_search,
                    verbose=False
                )
            else:
                return -upper_confidence_bound(
                    X_flat, gp_models, scaler_X, flat_limits,
                    detection_gp, use_gp_detection=False, kappa=kappa,
                    detection_calc=detection_calc,  # This now has filter_errors
                    verbose=False
                )

        print(f"      Sampling candidates near detectable events...")
        X_candidates = sample_near_detectable(
            X_true_pool, y_true_pool, bounds, 
            n_samples=candidate_samples,
            spread_factor=0.2, 
            n_uniform=candidate_uniform_samples,
            iteration=iteration,  # Pass iteration for adaptive sampling
            max_iterations=n_iterations
        )

        print(f"      Scoring {len(X_candidates)} candidates with {acquisition} acquisition...")
        acq_values = np.array([acq_func(x) for x in X_candidates])
        best_idx = np.argmin(acq_values)
        candidate_X = X_candidates[best_idx]
        best_acq = -acq_values[best_idx]

        # Preserve the surrogate h-vector used when this candidate was scored.
        # True light-curve outputs replace the corresponding predictions below.
        try:
            from models import predict_magnitudes_single
            candidate_h_pred, _ = predict_magnitudes_single(candidate_X, gp_models, scaler_X)
        except Exception:
            candidate_h_pred = {}
        proposal_seconds = time.perf_counter() - proposal_started

        # DEBUG: Check the top 5 candidates
        print(f"\n      Top 5 candidates by acquisition value:")
        top_indices = np.argsort(acq_values)[:5]
        for i, idx in enumerate(top_indices):
            acq_val = -acq_values[idx]
            X_top = X_candidates[idx]
            # Get score and uncertainty for this candidate
            score, uncertainty, best_filter = objective_with_uncertainty(
                X_top, gp_models, scaler_X, flat_limits, detection_calc, verbose=False
            )
            print(f"        #{i+1}: {acquisition}={acq_val:.6f}, mean P_detect={score:.4f}, "
                  f"limit_sigma={uncertainty:.4f}, filter={best_filter}")
            print(f"             params: {dict(zip(feature_columns, X_top))}")

        # Track CDF probability + margin for every scored candidate this iteration
        candidate_probs = [
            objective_function_probability(
                x, gp_models, scaler_X, flat_limits,
                detection_gp=None, use_gp_detection=False, detection_calc=detection_calc
            )
            for x in X_candidates
        ]
        results['probs_all'].extend(candidate_probs)

        formatted_candidate = ", ".join(f"{name}={val:.1f}" for name, val in zip(feature_columns, candidate_X))
        print(f"\n      Proposed candidate: [{formatted_candidate}]")
        print(f"      Acquisition value ({acquisition}): {best_acq:.6f}")
        
        # DEBUG: Check the proposed candidate's active-search probability
        score, uncertainty, best_filter = objective_with_uncertainty(
            candidate_X, gp_models, scaler_X, flat_limits, detection_calc, verbose=True
        )
        if acquisition == 'EI':
            expected_improvement(
                candidate_X, gp_models, scaler_X, flat_limits,
                best_score_for_acq, detection_gp,
                use_gp_detection=False, xi=xi,
                detection_calc=detection_calc,
                n_mc=eicf_mc_samples,
                active_search=active_search,
                verbose=True,
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
        simulation_started = time.perf_counter()
        detectable, message, peak_mag = runner.run_kilonova_simulation(
            kilonova_params,
            exposure_time=exposure_time,
            output_dir=output_dir,
            gw_params=gw_params
        )
        simulation_seconds = time.perf_counter() - simulation_started

        rollout_label, true_detection_prob, true_filter_probs, true_peak_mags_by_filter = _sample_active_search_label(
            output_dir, detection_calc, detection_filters
        )

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
            'proposal_seconds': proposal_seconds,
            'simulation_seconds': simulation_seconds,
        })
        for filt in detection_filters:
            if filt in true_peak_mags_by_filter:
                true_eval_log[-1][f'true_peak_mag_{filt}'] = true_peak_mags_by_filter[filt]

        for h_col, value in candidate_h_pred.items():
            if np.isscalar(value) and np.isfinite(value):
                true_eval_log[-1][h_col] = float(value)
        for filt, value in true_peak_mags_by_filter.items():
            true_eval_log[-1][filt] = float(value)
        true_eval_log[-1]['log10_mej_dyn'] = float(kilonova_params['log10_mej_dyn'])
        true_eval_log[-1]['log10_mej_wind'] = float(kilonova_params['log10_mej_wind'])
        if 'distance' in feature_columns:
            true_eval_log[-1]['distance'] = float(candidate_X[feature_columns.index('distance')])

        # Track this candidate for new_events_df (plot_optimization_results)
        candidate_prob = objective_function_probability(
            candidate_X, gp_models, scaler_X, flat_limits,
            detection_gp=None, use_gp_detection=False, detection_calc=detection_calc
        )
        candidate_mags = candidate_h_pred

        event_record = {name: candidate_X[j] for j, name in enumerate(feature_columns)}
        event_record['detection_probability'] = candidate_prob
        event_record['detected'] = int(rollout_label)
        event_record['nmma_threshold_detected'] = int(detectable)
        event_record['mej_dyn'] = float(10 ** kilonova_params['log10_mej_dyn'])
        event_record['mej_wind'] = float(10 ** kilonova_params['log10_mej_wind'])
        event_record['distance'] = float(candidate_X[feature_columns.index('distance')]) if 'distance' in feature_columns else event_record.get('distance')
        for h_col in H_OUTPUT_COLUMNS:
            if h_col in true_peak_mags_by_filter:
                event_record[h_col] = true_peak_mags_by_filter[h_col]
                event_record[f'true_{h_col}'] = true_peak_mags_by_filter[h_col]
            elif h_col in event_record and event_record[h_col] is not None:
                continue
            elif candidate_mags and h_col in candidate_mags:
                event_record[h_col] = candidate_mags[h_col]
        new_events_records.append(event_record)

        # Track margin + source for parameter_exploration plot
        candidate_margin = objective_function_margin(
            candidate_X, gp_models, scaler_X, flat_limits, detection_calc
        )
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
        current_margin_pool = [
            objective_function_margin(x, gp_models, scaler_X, flat_limits, detection_calc)
            for x in X_true_pool
        ]
        results['score_history'].append(float(np.max(current_margin_pool)))

        # ============================================================
        # STEP 3: ADD NEW DATA TO TRAINING POOL
        # ============================================================
        print(f"\n  [3] Adding new data to training pool...")

        X_true_pool = np.vstack([X_true_pool, candidate_X])
        y_true_pool = np.append(y_true_pool, float(rollout_label))

        print(f"      Training pool size: {len(X_true_pool)} events")
        print(f"      Detectable in pool: {np.sum(y_true_pool)}/{len(y_true_pool)}")

        # ============================================================
        # STEP 4: RETRAIN DETECTION GP WITH NEW DATA
        # ============================================================
        print(f"\n  [4] Retraining detection GP with new data...")
        gp_refit_started = time.perf_counter()

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

        gp_refit_seconds = time.perf_counter() - gp_refit_started
        iteration_seconds = time.perf_counter() - iteration_started
        true_eval_log[-1]['gp_refit_seconds'] = gp_refit_seconds
        true_eval_log[-1]['iteration_seconds'] = iteration_seconds
        true_eval_log[-1]['algorithm_overhead_seconds'] = max(
            0.0, iteration_seconds - simulation_seconds
        )

        print(f"      Detection GP retrained successfully")

        # ============================================================
        # STEP 5: REPORT ITERATION SUMMARY
        # ============================================================
        print(f"\n  [5] Iteration {iteration + 1} summary:")
        print(f"      Best peak mag so far: {best_true_peak_mag:.3f}" if best_true_peak_mag != np.inf else "      Best peak mag so far: None")
        if best_X is not None:
            print(f"      Best parameters so far: {dict(zip(feature_columns, best_X))}")

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
