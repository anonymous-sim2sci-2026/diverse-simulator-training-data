"""Batched ranked-region EI-CF + ROBOT optimization."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from models import build_gp_models, predict_magnitudes_single
from models.simulator import build_gw_params_from_X
from optimization.acquisition import (
    expected_improvement_composite_batch,
    objective_with_uncertainty,
    sample_around_robot_center,
    sample_near_detectable,
)
from optimization.bayesian_optimization import (
    _read_injection_ejecta,
    _sample_active_search_label,
)
from optimization.objectives import (
    objective_function_margin,
    objective_function_probability,
    objective_function_probability_batch,
)
from optimization.robot_h import (
    build_robot_state,
    predict_h_matrix,
    rebuild_ranked_robot_centers,
    robot_region_input_center,
    scale_h_to_unit,
    select_robot_candidate,
    update_robot_state,
)
from utils.detection import DetectionCalculator
from utils.embright import compute_embright_classification
from utils.h_columns import H_OUTPUT_COLUMNS


def rotating_region_ranks(robot_state, active_region_count, regions_per_batch):
    """Return the next ranked-region window while preserving rank order."""
    active_region_count = int(active_region_count)
    if active_region_count < 1:
        return [], 0
    regions_per_batch = min(int(regions_per_batch), active_region_count)
    # A full ROBOT step always visits T_1, ..., T_M in strict hierarchy order.
    # Rotation is retained only for the optional reduced-region schedule.
    if regions_per_batch == active_region_count:
        return list(range(active_region_count)), 0
    start = int(robot_state.get("next_region_rank", 0)) % active_region_count
    ranks = [
        (start + offset) % active_region_count
        for offset in range(regions_per_batch)
    ]
    # When the rotating window wraps, selection still proceeds from the
    # highest to the lowest rank.
    return sorted(ranks), start


def _prepare_simulation(runner, candidate_x, feature_columns, output_dir):
    gw_params = build_gw_params_from_X(candidate_x, feature_columns)
    kilonova_params = runner.map_gw_to_kilonova(gw_params)
    for index, name in enumerate(feature_columns):
        if name == "inclination":
            kilonova_params["inclination_EM"] = float(
                np.clip(candidate_x[index], 0.0, np.pi / 2)
            )
        elif name == "distance":
            kilonova_params["luminosity_distance"] = float(candidate_x[index])
    return {
        "candidate_x": np.asarray(candidate_x, dtype=float),
        "gw_params": gw_params,
        "kilonova_params": kilonova_params,
        "output_dir": output_dir,
    }


def _run_simulation(runner, task, exposure_time):
    started = time.time()
    detectable, message, peak_mag = runner.run_kilonova_simulation(
        task["kilonova_params"],
        exposure_time=exposure_time,
        output_dir=task["output_dir"],
        gw_params=task["gw_params"],
    )
    return {
        **task,
        "true_detectable": bool(detectable),
        "message": message,
        "peak_mag": float(peak_mag),
        "simulation_seconds": time.time() - started,
    }


def closed_loop_batched_bayesian_optimization(
    X_train,
    y_train_mags,
    y_train_detection,
    flat_limits: Dict,
    runner,
    base_sim_output_dir: str,
    gp_models=None,
    scaler_X=None,
    detection_gp=None,
    n_iterations: int = 1000,
    acquisition: str = "ROBOT_EICF",
    xi: float = 0.01,
    kappa: float = 2.0,
    exposure_time: int = 180,
    bounds=None,
    feature_columns=None,
    detection_filters=None,
    robot_distance_mode: str = "tail_l1",
    robot_knn_k: int = 5,
    robot_tau_quantile: float = 25.0,
    robot_initial_length: float = 0.6,
    robot_success_tolerance: int = 10,
    robot_failure_tolerance: int = 32,
    robot_rarity_n_bins: int = 20,
    robot_n_regions: int = 10,
    robot_batch_regions: int = 10,
    robot_candidates_per_region: int = 1,
    parallel_simulations: int = 10,
    eicf_mc_samples: int = 16,
    candidate_samples: int = 1500,
    candidate_uniform_samples: int = 600,
    robot_region_candidates: int = 300,
    candidate_sampling_mode: str = "mixed",
    candidate_detectability_threshold: float = 0.5,
    trust_region_max_resamples: int = 20,
    scoring_progress_every: int = 0,
    eicf_batch_size: int = 512,
    skip_detection_gp_refit: bool = True,
    checkpoint_every: int = 1,
    resume: bool = False,
    checkpoint_file: Optional[str] = None,
    max_runtime_hours: Optional[float] = None,
    debug_candidate_details: bool = False,
    track_all_candidate_probs: bool = False,
    refit_output_gps: bool = False,
    output_gp_refit_every: int = 1,
    use_multitask_gp: bool = False,
) -> Tuple:
    """Select one point per ranked region and evaluate the batch concurrently."""
    del debug_candidate_details, kappa
    if acquisition != "ROBOT_EICF":
        raise ValueError("Batched mode supports only ROBOT_EICF.")
    if not skip_detection_gp_refit:
        raise ValueError(
            "Batched mode currently requires --skip-detection-gp-refit. "
            "EI-CF uses the output GPs, so the scalar detection GP is unnecessary."
        )
    if bounds is None:
        raise ValueError("bounds must be provided")
    if feature_columns is None:
        feature_columns = [
            "mass1", "mass2", "spin1z", "spin2z", "inclination", "distance"
        ]
    if detection_filters is None:
        detection_filters = ["g", "r", "i"]
    if candidate_sampling_mode not in {"mixed", "uniform_detectable"}:
        raise ValueError(
            "candidate_sampling_mode must be mixed or uniform_detectable"
        )
    if not 0.0 <= float(candidate_detectability_threshold) <= 1.0:
        raise ValueError(
            "candidate_detectability_threshold must be between 0 and 1"
        )
    output_gp_refit_every = int(output_gp_refit_every)
    if output_gp_refit_every < 1:
        raise ValueError("output_gp_refit_every must be at least 1")
    robot_success_tolerance = int(robot_success_tolerance)
    robot_failure_tolerance = int(robot_failure_tolerance)
    if robot_success_tolerance < 1 or robot_failure_tolerance < 1:
        raise ValueError("ROBOT success/failure tolerances must be at least 1")

    robot_n_regions = int(robot_n_regions)
    robot_batch_regions = min(int(robot_batch_regions), robot_n_regions)
    robot_candidates_per_region = int(robot_candidates_per_region)
    max_batch_events = robot_batch_regions * robot_candidates_per_region
    parallel_simulations = min(int(parallel_simulations), max_batch_events)
    if min(
        robot_n_regions,
        robot_batch_regions,
        robot_candidates_per_region,
        parallel_simulations,
    ) < 1:
        raise ValueError("Region and parallelism settings must be positive.")

    detection_calc = DetectionCalculator(
        detection_filters=detection_filters,
        flat_limits=flat_limits,
    )
    if gp_models is None or scaler_X is None:
        gp_models, scaler_X, detection_gp = build_gp_models(
            X_train, y_train_mags, y_train_detection
        )

    os.makedirs(base_sim_output_dir, exist_ok=True)
    checkpoint_file = os.path.abspath(
        checkpoint_file
        or os.path.join(base_sim_output_dir, "bo_batch_state_checkpoint.joblib")
    )
    X_true_pool = np.asarray(X_train, dtype=float).copy()
    y_true_pool = np.asarray(y_train_detection, dtype=float).copy()
    initial_probs = objective_function_probability_batch(
        X_true_pool, gp_models, scaler_X, flat_limits, detection_calc=detection_calc
    )
    observed_scores = list(np.asarray(initial_probs, dtype=float))
    h_training_values = {
        column: pd.to_numeric(values, errors="coerce").tolist()
        for column, values in y_train_mags.items()
    }
    robot_state = build_robot_state(
        y_train_mags=y_train_mags,
        g_train=initial_probs,
        X_train=X_true_pool,
        n_centers=robot_n_regions,
        useful_threshold=0.5,
        tau_quantile=robot_tau_quantile,
        distance_mode=robot_distance_mode,
        knn_k=robot_knn_k,
        rarity_n_bins=robot_rarity_n_bins,
        distance_bounds=bounds[-1],
        initial_trust_region_length=robot_initial_length,
        success_tolerance=robot_success_tolerance,
        failure_tolerance=robot_failure_tolerance,
    )
    if robot_state is None:
        raise RuntimeError("Could not initialize ranked ROBOT regions.")

    true_eval_log = []
    new_events_records = []
    best_true_peak_mag = np.inf
    best_X = None
    results = {
        "probs_history": [], "probs_all": [], "score_history": [],
        "improved_iteration": [], "X_all": list(X_true_pool.copy()),
        "margin_all": [], "source": ["training"] * len(X_true_pool),
    }
    for x in X_true_pool:
        results["margin_all"].append(
            objective_function_margin(x, gp_models, scaler_X, flat_limits, detection_calc)
        )
    running_best_margin = float(np.nanmax(results["margin_all"]))
    start_batch = 0
    elapsed_before = 0.0
    settings = {
        "robot_algorithm": "rotating_ranked_fixed_tau_multi_candidate_v3",
        "acquisition": acquisition,
        "robot_distance_mode": robot_distance_mode,
        "robot_knn_k": int(robot_knn_k),
        "robot_tau_quantile": float(robot_tau_quantile),
        "robot_initial_length": float(robot_initial_length),
        "robot_success_tolerance": robot_success_tolerance,
        "robot_failure_tolerance": robot_failure_tolerance,
        "robot_n_regions": robot_n_regions,
        "robot_batch_regions": robot_batch_regions,
        "robot_candidates_per_region": robot_candidates_per_region,
        "parallel_simulations": parallel_simulations,
        "candidate_samples": int(candidate_samples),
        "candidate_uniform_samples": int(candidate_uniform_samples),
        "robot_region_candidates": int(robot_region_candidates),
        "candidate_sampling_mode": candidate_sampling_mode,
        "candidate_detectability_threshold": float(
            candidate_detectability_threshold
        ),
        "eicf_mc_samples": int(eicf_mc_samples),
        "eicf_batch_size": int(eicf_batch_size),
        "refit_output_gps": bool(refit_output_gps),
        "output_gp_refit_every": output_gp_refit_every,
        "use_multitask_gp": bool(use_multitask_gp),
        "feature_columns": list(feature_columns),
    }
    if resume:
        state = joblib.load(checkpoint_file)
        if state.get("checkpoint_version") != 7:
            raise ValueError(
                "This ranked-exclusion implementation requires a new output "
                "folder; checkpoints from the earlier ROBOT code are incompatible."
            )
        mismatches = {
            key: (state.get("settings", {}).get(key), value)
            for key, value in settings.items()
            if state.get("settings", {}).get(key) != value
        }
        if mismatches:
            raise ValueError(f"Resume settings differ from checkpoint: {mismatches}")
        start_batch = int(state["completed_batches"])
        X_true_pool = state["X_true_pool"]
        y_true_pool = state["y_true_pool"]
        observed_scores = state["observed_scores"]
        h_training_values = state.get("h_training_values", h_training_values)
        if refit_output_gps:
            gp_models = state["gp_models"]
            scaler_X = state["scaler_X"]
        robot_state = state["robot_state"]
        true_eval_log = state["true_eval_log"]
        new_events_records = state["new_events_records"]
        results = state["results"]
        running_best_margin = state["running_best_margin"]
        best_true_peak_mag = state["best_true_peak_mag"]
        best_X = state["best_X"]
        elapsed_before = float(state.get("elapsed_seconds", 0.0))
        np.random.set_state(state["numpy_random_state"])

    segment_started = time.time()

    def write_checkpoint(completed_batches):
        pd.DataFrame(true_eval_log).to_csv(
            os.path.join(base_sim_output_dir, "closed_loop_results.csv"), index=False
        )
        pd.DataFrame(new_events_records).to_csv(
            os.path.join(base_sim_output_dir, "new_events_with_h.csv"), index=False
        )
        state = {
            "checkpoint_version": 7,
            "completed_batches": int(completed_batches),
            "target_batches": int(n_iterations),
            "settings": settings,
            "X_true_pool": X_true_pool,
            "y_true_pool": y_true_pool,
            "observed_scores": observed_scores,
            "h_training_values": h_training_values,
            "gp_models": gp_models if refit_output_gps else None,
            "scaler_X": scaler_X if refit_output_gps else None,
            "robot_state": robot_state,
            "true_eval_log": true_eval_log,
            "new_events_records": new_events_records,
            "results": results,
            "running_best_margin": running_best_margin,
            "best_true_peak_mag": best_true_peak_mag,
            "best_X": best_X,
            "numpy_random_state": np.random.get_state(),
            "elapsed_seconds": elapsed_before + time.time() - segment_started,
        }
        temporary = checkpoint_file + ".tmp"
        joblib.dump(state, temporary)
        os.replace(temporary, checkpoint_file)

    for batch_iteration in range(start_batch, n_iterations):
        batch_started = time.time()
        robot_state = rebuild_ranked_robot_centers(robot_state)
        trust_regions = robot_state["trust_regions"]
        active_region_count = int(
            robot_state.get(
                "active_ranked_region_count",
                len(robot_state.get("ranked_center_history_indices", trust_regions)),
            )
        )
        available_region_count = min(
            robot_batch_regions, len(trust_regions), active_region_count
        )
        region_ranks, rotation_start = rotating_region_ranks(
            robot_state,
            active_region_count=min(len(trust_regions), active_region_count),
            regions_per_batch=available_region_count,
        )
        if len(region_ranks) < min(robot_batch_regions, len(trust_regions)):
            print(
                f"  ROBOT currently has {len(region_ranks)} genuinely diverse "
                f"ranked regions (requested {robot_batch_regions}).",
                flush=True,
            )
        best_observed = float(np.nanmax(observed_scores))
        selections = []
        print(
            f"\n{'=' * 72}\nBATCH {batch_iteration + 1}/{n_iterations}: "
            f"selecting from "
            f"{', '.join(f'T_{rank + 1}' for rank in region_ranks)}\n{'=' * 72}",
            flush=True,
        )

        empty_region_ranks = list(region_ranks)
        for attempt in range(1, trust_region_max_resamples + 1):
            if candidate_sampling_mode == "uniform_detectable":
                lows = np.asarray([bound[0] for bound in bounds], dtype=float)
                highs = np.asarray([bound[1] for bound in bounds], dtype=float)
                pool = lows + np.random.random(
                    (int(candidate_samples), len(bounds))
                ) * (highs - lows)
            else:
                pool_parts = [
                    sample_near_detectable(
                        X_true_pool,
                        y_true_pool,
                        bounds,
                        n_samples=candidate_samples,
                        spread_factor=0.2,
                        n_uniform=candidate_uniform_samples,
                        iteration=batch_iteration,
                        max_iterations=n_iterations,
                    )
                ]
                for region_rank in region_ranks:
                    center_x = robot_region_input_center(robot_state, region_rank)
                    if center_x is not None and int(robot_region_candidates) > 0:
                        pool_parts.append(
                            sample_around_robot_center(
                                center_x,
                                bounds,
                                n_samples=int(robot_region_candidates),
                                spread_factor=max(
                                    0.03,
                                    0.15 * float(trust_regions[region_rank].length),
                                ),
                            )
                        )
                pool = np.unique(np.vstack(pool_parts), axis=0)
            np.random.shuffle(pool)

            score_started = time.time()
            scores = expected_improvement_composite_batch(
                pool,
                gp_models,
                scaler_X,
                flat_limits,
                best_observed,
                detection_calc=detection_calc,
                n_mc=eicf_mc_samples,
                xi=xi,
                batch_size=eicf_batch_size,
                progress_every=(
                    max(1, scoring_progress_every // max(eicf_batch_size, 1))
                    if scoring_progress_every
                    else 0
                ),
                active_search=True,
            )
            scoring_seconds = time.time() - score_started

            if candidate_sampling_mode == "uniform_detectable":
                detectable_mask = (
                    np.isfinite(scores)
                    & (np.asarray(scores) >= candidate_detectability_threshold)
                )
                n_before_filter = len(pool)
                pool = pool[detectable_mask]
                scores = np.asarray(scores)[detectable_mask]
                print(
                    "  Uniform surrogate-detectability filter: "
                    f"{len(pool)}/{n_before_filter} candidates retained "
                    f"(threshold={candidate_detectability_threshold:.3f})",
                    flush=True,
                )
                if not len(pool):
                    continue

            H_pred = predict_h_matrix(
                pool, gp_models, scaler_X, robot_state["h_columns"]
            )
            Z_pred = scale_h_to_unit(
                H_pred, robot_state["references"], robot_state["fill_values"]
            )

            selected_h_z = []
            selected_h_raw = []
            selections = []
            empty_region_ranks = []
            for region_rank in region_ranks:
                # Candidates selected within one region share the same hierarchy
                # constraints. Fixed centers preserve T_1 > ... > T_M even
                # when only one rotating region is evaluated in this batch.
                higher_h_z = [
                    np.asarray(center, dtype=float).copy()
                    for center in robot_state["centers_z"][:region_rank]
                ]
                higher_h_raw = [
                    np.asarray(center, dtype=float).copy()
                    for center in robot_state["centers_h_raw"][:region_rank]
                ]
                # Selected points from active higher ranks add stricter,
                # batch-specific exclusions beyond the fixed centers.
                higher_h_z.extend(selected_h_z)
                higher_h_raw.extend(selected_h_raw)
                available_scores = np.asarray(scores, dtype=float).copy()
                region_selections = []
                last_error = None
                for region_slot in range(robot_candidates_per_region):
                    robot_started = time.time()
                    try:
                        index, diagnostics = select_robot_candidate(
                            X_candidates=pool,
                            direct_scores=available_scores,
                            gp_models=gp_models,
                            scaler_X=scaler_X,
                            robot_state=robot_state,
                            region_index=region_rank,
                            selected_h_z=higher_h_z,
                            selected_h_raw=higher_h_raw,
                            H_pred=H_pred,
                            Z_pred=Z_pred,
                        )
                    except RuntimeError as error:
                        last_error = error
                        break

                    available_scores[index] = -np.inf
                    diagnostics["robot_region_slot"] = region_slot
                    selected = {
                        "pool_index": int(index),
                        "candidate_x": np.asarray(pool[index], dtype=float),
                        "acquisition_value": float(scores[index]),
                        "diagnostics": diagnostics,
                        "candidate_count": len(pool),
                        "sampling_attempts": attempt,
                        "candidate_scoring_seconds": np.nan,
                        "shared_pool_scoring_seconds": scoring_seconds,
                        "robot_selection_seconds": time.time() - robot_started,
                    }
                    region_selections.append(selected)

                if not region_selections:
                    empty_region_ranks.append(region_rank)
                    print(f"  T_{region_rank + 1}: {last_error}", flush=True)
                    continue

                for selected in region_selections:
                    index = selected["pool_index"]
                    selected_h_z.append(Z_pred[index].copy())
                    selected_h_raw.append(H_pred[index].copy())
                    selections.append(selected)
                diagnostics = region_selections[0]["diagnostics"]
                print(
                    f"  T_{region_rank + 1}: selected {len(region_selections)}/"
                    f"{robot_candidates_per_region}; overlapping higher boxes="
                    f"{diagnostics['robot_overlap_candidates']}; excluded by tau="
                    f"{diagnostics['robot_similarity_excluded']}",
                    flush=True,
                )
            if (
                not selections
                and attempt == max(1, trust_region_max_resamples // 2)
            ):
                for region_rank in empty_region_ranks:
                    region = trust_regions[region_rank]
                    old_length = float(region.length)
                    region.length = min(region.length_max, 2.0 * region.length)
                    region.failure_counter = 0
                    print(
                        f"  T_{region_rank + 1}: expanding during resampling "
                        f"{old_length:.4f} -> {region.length:.4f}",
                        flush=True,
                    )
            if selections:
                amortized_scoring = scoring_seconds / len(selections)
                for selected in selections:
                    selected["candidate_scoring_seconds"] = amortized_scoring
                break

        for region_rank in empty_region_ranks:
            region = trust_regions[region_rank]
            old_length = float(region.length)
            region.length = min(region.length_max, 2.0 * region.length)
            region.failure_counter = 0
            print(
                f"  T_{region_rank + 1}: no feasible point after resampling; "
                f"expanded {old_length:.4f} -> {region.length:.4f}",
                flush=True,
            )

        if not selections:
            # A lower-ranked region can be empty even with a large candidate
            # pool: its predicted h values may fail the fixed-tau exclusions
            # from all higher-ranked centers. This is a local region failure,
            # not a reason to discard the rest of a long optimization run.
            robot_state["next_region_rank"] = (
                rotation_start + max(len(region_ranks), 1)
            ) % max(active_region_count, 1)
            print(
                "  No feasible candidate was found for this ranked region "
                "after resampling. Skipping this batch and rotating to the "
                "next region. If this happens often, weaken the fixed "
                "diversity threshold (use a larger --robot-tau-quantile in "
                "the current CLI convention) or refit the h surrogates.",
                flush=True,
            )
            if checkpoint_every and (batch_iteration + 1) % checkpoint_every == 0:
                write_checkpoint(batch_iteration + 1)
            continue

        first_event_index = len(true_eval_log)
        tasks = []
        for batch_index, selected in enumerate(selections):
            event_index = first_event_index + batch_index
            output_dir = os.path.join(base_sim_output_dir, f"i{event_index:05d}")
            task = _prepare_simulation(
                runner, selected["candidate_x"], feature_columns, output_dir
            )
            task.update(selected)
            task["event_index"] = event_index
            task["batch_index"] = batch_index
            tasks.append(task)

        print(
            f"  Simulating {len(tasks)} events with {parallel_simulations} workers...",
            flush=True,
        )
        completed = []
        with ThreadPoolExecutor(max_workers=parallel_simulations) as executor:
            futures = {
                executor.submit(_run_simulation, runner, task, exposure_time): task
                for task in tasks
            }
            for future in as_completed(futures):
                completed.append(future.result())
        completed.sort(key=lambda item: item["batch_index"])

        region_updates = {}
        for item in completed:
            post_started = time.time()
            rollout_label, true_probability, filter_probs, true_peaks = (
                _sample_active_search_label(
                    item["output_dir"], detection_calc, detection_filters
                )
            )
            injection_ejecta = _read_injection_ejecta(item["output_dir"])
            postprocess_seconds = time.time() - post_started
            candidate_x = item["candidate_x"]
            diagnostics = item["diagnostics"]
            try:
                predicted_h, _ = predict_magnitudes_single(
                    candidate_x, gp_models, scaler_X
                )
            except Exception:
                predicted_h = {}

            record = {
                "iteration": item["event_index"],
                "event_index": item["event_index"],
                "batch_iteration": batch_iteration,
                "batch_index": item["batch_index"],
                **{
                    name: float(candidate_x[index])
                    for index, name in enumerate(feature_columns)
                },
                "true_detectable": int(item["true_detectable"]),
                "active_search_detectable": int(rollout_label),
                "active_search_detection_probability": float(true_probability),
                "true_peak_mag": item["peak_mag"],
                "acquisition_value": item["acquisition_value"],
                "candidate_count": item["candidate_count"],
                "candidate_sampling_attempts": item["sampling_attempts"],
                "candidate_scoring_seconds": item["candidate_scoring_seconds"],
                "shared_pool_scoring_seconds": item[
                    "shared_pool_scoring_seconds"
                ],
                "robot_selection_seconds": item["robot_selection_seconds"],
                "simulation_seconds": item["simulation_seconds"],
                "postprocess_seconds": postprocess_seconds,
                "robot_used": True,
                "robot_region_rank": diagnostics.get("robot_region_rank"),
                "robot_region_slot": diagnostics.get("robot_region_slot"),
                "robot_center_index": diagnostics.get("robot_center_index"),
                "robot_tau_quantile": diagnostics.get("robot_tau_quantile"),
                "robot_tau": diagnostics.get("robot_tau"),
                "robot_tau_used": diagnostics.get("robot_tau_used"),
                "robot_tau_satisfied": diagnostics.get("robot_tau_satisfied"),
                "robot_length": diagnostics.get("robot_length"),
                "robot_length_min": diagnostics.get("robot_length_min"),
                "robot_length_max": diagnostics.get("robot_length_max"),
                "robot_center_distance": diagnostics.get("robot_center_distance"),
                "robot_selected_distance": diagnostics.get("robot_selected_distance"),
                "robot_region_candidates_before_hierarchy": diagnostics.get(
                    "robot_region_candidates_before_hierarchy"
                ),
                "robot_overlap_excluded": diagnostics.get("robot_overlap_excluded"),
                "robot_overlap_candidates": diagnostics.get(
                    "robot_overlap_candidates"
                ),
                "robot_similarity_excluded": diagnostics.get(
                    "robot_similarity_excluded"
                ),
                "robot_higher_rank_selections": diagnostics.get(
                    "robot_higher_rank_selections"
                ),
                "robot_selection_rule": diagnostics.get("robot_selection_rule"),
                "message": item["message"],
                "output_dir": item["output_dir"],
            }
            for filt, probability in filter_probs.items():
                record[f"true_detection_probability_{filt}"] = probability
            for filt, peak in true_peaks.items():
                record[f"true_peak_mag_{filt}"] = peak
                record[filt] = peak
            for column in H_OUTPUT_COLUMNS:
                if column not in record and column in predicted_h:
                    record[column] = predicted_h[column]
                record[f"robot_predicted_{column}"] = diagnostics.get(
                    "predicted_h", {}
                ).get(column, predicted_h.get(column))
            record["log10_mej_dyn"] = float(
                injection_ejecta.get(
                    "log10_mej_dyn", item["kilonova_params"]["log10_mej_dyn"]
                )
            )
            record["log10_mej_wind"] = float(
                injection_ejecta.get(
                    "log10_mej_wind", item["kilonova_params"]["log10_mej_wind"]
                )
            )
            record.update(
                compute_embright_classification(
                    mass1=record.get("mass1"),
                    mass2=record.get("mass2"),
                    spin1z=record.get("spin1z"),
                    spin2z=record.get("spin2z"),
                    snr=record.get("snr"),
                )
            )
            record["detection_probability"] = objective_function_probability(
                candidate_x,
                gp_models,
                scaler_X,
                flat_limits,
                detection_gp=None,
                use_gp_detection=False,
                detection_calc=detection_calc,
            )
            record["detected"] = int(rollout_label)
            record["nmma_threshold_detected"] = int(item["true_detectable"])
            true_eval_log.append(record.copy())
            new_events_records.append(record.copy())

            X_true_pool = np.vstack([X_true_pool, candidate_x])
            y_true_pool = np.append(y_true_pool, float(rollout_label))
            observed_scores.append(float(true_probability))
            for column in h_training_values:
                h_training_values[column].append(record.get(column, np.nan))
            observed_h = [
                record.get(
                    column, diagnostics.get("predicted_h", {}).get(column, np.nan)
                )
                for column in robot_state["h_columns"]
            ]
            region_rank = int(diagnostics.get("robot_region_rank"))
            update = region_updates.setdefault(
                region_rank, {"scores": [], "x": [], "h": []}
            )
            update["scores"].append(float(true_probability))
            update["x"].append(candidate_x)
            update["h"].append(observed_h)
            margin = objective_function_margin(
                candidate_x, gp_models, scaler_X, flat_limits, detection_calc
            )
            running_best_margin = max(running_best_margin, float(margin))
            results["X_all"].append(candidate_x)
            results["margin_all"].append(margin)
            results["source"].append("bo_candidate")
            results["probs_history"].append(float(np.nanmax(observed_scores)))
            results["score_history"].append(running_best_margin)
            if rollout_label and item["peak_mag"] < best_true_peak_mag:
                best_true_peak_mag = item["peak_mag"]
                best_X = candidate_x.copy()
                results["improved_iteration"].append(item["event_index"])

        for region_rank, update in sorted(region_updates.items()):
            robot_state = update_robot_state(
                robot_state,
                update["scores"],
                x_next=update["x"],
                h_next=update["h"],
                selected_center_index=region_rank,
                rebuild_centers=False,
            )

        gp_refit_seconds = 0.0
        if refit_output_gps and (batch_iteration + 1) % output_gp_refit_every == 0:
            refit_started = time.time()
            refit_targets = {
                column: pd.Series(values, dtype=float)
                for column, values in h_training_values.items()
            }
            gp_models, scaler_X, _ = build_gp_models(
                X_true_pool,
                refit_targets,
                y_train_detection=None,
                use_multitask_gp=use_multitask_gp,
            )
            gp_refit_seconds = time.time() - refit_started
            print(
                f"  Refit h-output GP models on {len(X_true_pool)} events "
                f"in {gp_refit_seconds:.1f}s",
                flush=True,
            )
        robot_state = rebuild_ranked_robot_centers(robot_state)
        robot_state["next_region_rank"] = (
            rotation_start + len(region_ranks)
        ) % max(active_region_count, 1)

        batch_seconds = time.time() - batch_started
        for record in true_eval_log[-len(completed):]:
            record["batch_seconds"] = batch_seconds
            record["gp_refit_seconds"] = gp_refit_seconds / max(len(completed), 1)
            record["parallel_simulations"] = parallel_simulations
            record["batch_size_actual"] = len(completed)
        print(
            f"  Batch complete: {len(completed)} events in {batch_seconds:.1f}s; "
            f"total events={len(true_eval_log)}",
            flush=True,
        )
        if checkpoint_every and (batch_iteration + 1) % checkpoint_every == 0:
            write_checkpoint(batch_iteration + 1)
        if max_runtime_hours is not None:
            elapsed_hours = (time.time() - segment_started) / 3600.0
            if elapsed_hours >= max_runtime_hours:
                write_checkpoint(batch_iteration + 1)
                print(
                    f"Reached runtime limit after batch {batch_iteration + 1}. "
                    f"Resume with --resume.",
                    flush=True,
                )
                break

    completed_batches = (
        int(true_eval_log[-1]["batch_iteration"]) + 1 if true_eval_log else start_batch
    )
    write_checkpoint(completed_batches)
    results["cumulative_runtime_seconds"] = (
        elapsed_before + time.time() - segment_started
    )
    new_events_df = pd.DataFrame(new_events_records)
    return (
        best_X,
        best_true_peak_mag,
        true_eval_log,
        gp_models,
        scaler_X,
        detection_gp,
        results,
        new_events_df,
    )
