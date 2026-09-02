"""ROBOT-style h-space diversity utilities.

This module keeps the expensive simulator unchanged. It uses the cheap
diagnostic h(x) surrogates to decide whether a proposed x is useful for
diversity before NMMA is run.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from models import predict_magnitudes_batch
from utils.h_columns import H_PLOT_COLUMNS


ROBOT_ACQUISITIONS = {"ROBOT_DIRECT_EI"}


CONTINUOUS_H_COLUMNS = {
    "snr",
    "area90",
    "log10_mej_dyn",
    "log10_mej_wind",
    "g",
    "r",
    "i",
    "distance",
}

ROBOT_DISTANCE_MODES = {"tail_l1"}


@dataclass
class TrustRegionState:
    """Paper-style trust-region state for one h-space center."""

    dim: int
    batch_size: int
    center_point: np.ndarray
    best_x: Any = ""
    best_value: float = -float("inf")
    length: float = 0.6
    length_min: float = 0.05
    length_max: float = 1.0
    failure_counter: int = 0
    failure_tolerance: int = 32
    success_counter: int = 0
    success_tolerance: int = 10
    restart_triggered: bool = False


def update_state(state, Y_next):
    """Update one trust-region length using the ROBOT paper code logic."""
    Y_next = np.atleast_1d(np.asarray(Y_next, dtype=float))
    if len(Y_next) == 0 or not np.any(np.isfinite(Y_next)):
        return state

    y_max = float(np.nanmax(Y_next))
    improvement_tol = 1e-3 * abs(state.best_value)
    if y_max > state.best_value + improvement_tol:
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += len(Y_next)

    if state.success_counter == state.success_tolerance:
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, y_max)
    if state.length < state.length_min:
        state.restart_triggered = True
    return state


def _as_float_array(values):
    if hasattr(values, "to_numpy"):
        return values.to_numpy(dtype=float)
    return np.asarray(values, dtype=float)


def h_matrix_from_dict(y_train_mags, h_columns=None):
    """Return an n x d h matrix from the training h dictionary."""
    if h_columns is None:
        h_columns = [col for col in H_PLOT_COLUMNS if col in y_train_mags]
    if not h_columns:
        return np.empty((0, 0)), []

    cols = []
    used_columns = []
    n_rows = None
    for col in h_columns:
        if col not in y_train_mags:
            continue
        arr = _as_float_array(y_train_mags[col])
        if n_rows is None:
            n_rows = len(arr)
        if len(arr) != n_rows:
            continue
        cols.append(arr)
        used_columns.append(col)
    if not cols:
        return np.empty((0, 0)), []

    return np.column_stack(cols), used_columns


def robot_distance_columns(h_columns):
    """Keep continuous h coordinates for trust regions/diversity distances."""
    return [col for col in h_columns if col in CONTINUOUS_H_COLUMNS]


def fit_h_rank_references(H, reference_rows):
    """Fit empirical-CDF references used to map each h dimension to [0, 1]."""
    H = np.asarray(H, dtype=float)
    reference_rows = np.asarray(reference_rows, dtype=bool)
    if len(reference_rows) != len(H) or not np.any(reference_rows):
        reference_rows = np.all(np.isfinite(H), axis=1)

    references = []
    fill_values = []
    for j in range(H.shape[1]):
        vals = H[reference_rows, j]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            vals = np.asarray([0.0])
        references.append(np.sort(vals))
        fill_values.append(float(np.nanmedian(vals)))
    return references, np.asarray(fill_values, dtype=float)


def scale_h_to_unit(H, references, fill_values):
    """Map h values to empirical percentiles in [0, 1]."""
    H = np.asarray(H, dtype=float)
    H = np.atleast_2d(H)
    H = np.where(np.isfinite(H), H, fill_values)
    Z = np.zeros_like(H, dtype=float)
    for j, ref in enumerate(references):
        if len(ref) <= 1 or np.allclose(ref[0], ref[-1]):
            Z[:, j] = 0.5
            continue
        ranks = np.searchsorted(ref, H[:, j], side="right")
        Z[:, j] = ranks / len(ref)
    return np.clip(np.nan_to_num(Z, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)


def fit_rarity_model(H, reference_rows, n_bins=20, clip=(0.5, 5.0), epsilon=1e-6):
    """Fit simple 1D histogram rarity weights on original h values."""
    H = np.asarray(H, dtype=float)
    reference_rows = np.asarray(reference_rows, dtype=bool)
    models = []
    for j in range(H.shape[1]):
        vals = H[reference_rows, j]
        vals = vals[np.isfinite(vals)]
        if len(vals) < 2 or np.allclose(vals[0], vals[-1]):
            models.append({
                "edges": np.asarray([-np.inf, np.inf], dtype=float),
                "density": np.asarray([1.0], dtype=float),
                "median_density": 1.0,
                "clip": clip,
                "epsilon": epsilon,
            })
            continue

        counts, edges = np.histogram(vals, bins=n_bins)
        density = counts.astype(float) / max(float(np.sum(counts)), 1.0)
        positive = density[density > 0]
        median_density = float(np.median(positive)) if len(positive) else 1.0
        models.append({
            "edges": edges,
            "density": density,
            "median_density": median_density,
            "clip": clip,
            "epsilon": epsilon,
        })
    return models


def rarity_weights_for_values(H, rarity_model):
    """Return inverse-density rarity weights for original h values."""
    H = np.atleast_2d(np.asarray(H, dtype=float))
    R = np.ones_like(H, dtype=float)
    if rarity_model is None:
        return R

    for j, model in enumerate(rarity_model):
        edges = model["edges"]
        density = model["density"]
        bins = np.searchsorted(edges, H[:, j], side="right") - 1
        bins = np.clip(bins, 0, len(density) - 1)
        local_density = density[bins]
        r = model["median_density"] / (local_density + model["epsilon"])
        lo, hi = model["clip"]
        R[:, j] = np.clip(r, lo, hi)
    return R


def weighted_l1_distances(
    A,
    B,
    weights,
    A_raw=None,
    B_raw=None,
    rarity_model=None,
    distance_mode="tail_l1",
):
    """Pairwise scaled L1 distances between rows of A and rows of B.

    For the paper-facing code path, weights are all ones, so this is the
    unweighted L1 distance in rank-scaled h-space.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    weights = np.asarray(weights, dtype=float)
    absdiff = np.abs(A[:, None, :] - B[None, :, :])
    factor = weights[None, None, :]

    return np.sum(absdiff * factor, axis=2)


def _pairwise_nonzero_distances(
    Z,
    weights,
    H_raw=None,
    rarity_model=None,
    distance_mode="tail_l1",
):
    if len(Z) < 2:
        return np.array([])
    D = weighted_l1_distances(
        Z,
        Z,
        weights,
        A_raw=H_raw,
        B_raw=H_raw,
        rarity_model=rarity_model,
        distance_mode=distance_mode,
    )
    tri = D[np.triu_indices_from(D, k=1)]
    return tri[tri > 1e-12]


def empirical_cdf_values(values, reference_values):
    """Evaluate the empirical CDF of reference_values at values."""
    values = np.asarray(values, dtype=float)
    cdf = np.zeros_like(values, dtype=float)
    reference_values = np.asarray(reference_values, dtype=float)
    reference_values = np.sort(reference_values[np.isfinite(reference_values)])
    finite = np.isfinite(values)
    if len(reference_values) == 0:
        return cdf
    ranks = np.searchsorted(reference_values, values[finite], side="right")
    cdf[finite] = ranks / float(len(reference_values))
    return np.clip(cdf, 0.0, 1.0)


def tail_l1_weights(distances, reference_distances, tau_quantile, tau):
    """Hard tail-L1 indicator from the empirical L1-distance CDF.

    tau is the requested percentile of the empirical pairwise scaled-L1
    distance distribution. Candidates with distance >= tau are in the tail.
    """
    distances = np.asarray(distances, dtype=float)
    cdf = empirical_cdf_values(distances, reference_distances)
    tau_cdf = float(np.clip(tau_quantile / 100.0, 0.0, 1.0))
    denom = max(1.0 - tau_cdf, 1e-12)
    tail_rank = np.clip((cdf - tau_cdf) / denom, 0.0, 1.0)
    in_tail = np.isfinite(distances) & (distances >= tau)
    weights = np.where(in_tail, 1.0, 0.0)
    return weights, cdf, tail_rank


def build_robot_state(
    y_train_mags,
    g_train,
    h_columns=None,
    useful_threshold=0.5,
    n_centers=10,
    tau_quantile=85.0,
    batch_size=1,
    distance_mode="tail_l1",
    rarity_n_bins=20,
    rarity_clip=(0.5, 5.0),
    diversity_weight=0.35,
):
    """
    Initialize ROBOT-style h-space trust centers from existing useful samples.

    Centers are chosen greedily from training events with P_detect above
    useful_threshold. For tail_l1, they are separated by an unweighted scaled
    L1 threshold tau, where tau is the requested percentile of the empirical
    pairwise-distance distribution among useful training events.
    """
    H_all, all_h_columns = h_matrix_from_dict(y_train_mags, h_columns=h_columns)
    if H_all.size == 0:
        return None

    distance_h_columns = robot_distance_columns(all_h_columns)
    if not distance_h_columns:
        return None

    H, h_columns = h_matrix_from_dict(y_train_mags, h_columns=distance_h_columns)
    if H.size == 0:
        return None

    if distance_mode not in ROBOT_DISTANCE_MODES:
        raise ValueError(
            f"Unknown ROBOT distance_mode={distance_mode!r}; choose one of {sorted(ROBOT_DISTANCE_MODES)}"
        )

    finite_rows = np.all(np.isfinite(H), axis=1)
    g_train = np.asarray(g_train, dtype=float)
    useful_rows = finite_rows & np.isfinite(g_train) & (g_train >= useful_threshold)
    if not np.any(useful_rows):
        useful_rows = finite_rows & np.isfinite(g_train)
    if not np.any(useful_rows):
        return None

    references, fill_values = fit_h_rank_references(H, useful_rows)
    Z = scale_h_to_unit(H, references, fill_values)
    weights = np.ones(len(h_columns), dtype=float)
    h_lb = np.zeros(len(h_columns), dtype=float)
    h_ub = np.ones(len(h_columns), dtype=float)
    rarity_model = None

    useful_idx = np.where(useful_rows)[0]
    useful_Z = Z[useful_idx]
    useful_H = H[useful_idx]
    nonzero = _pairwise_nonzero_distances(
        useful_Z,
        weights,
        H_raw=useful_H,
        rarity_model=rarity_model,
        distance_mode=distance_mode,
    )
    tau = float(np.percentile(nonzero, tau_quantile)) if len(nonzero) else 1.0
    tau = max(tau, 1e-6)
    tail_reference_distances = np.sort(nonzero[np.isfinite(nonzero)])

    order = useful_idx[np.argsort(-g_train[useful_idx])]
    center_indices = []
    centers = []
    for idx in order:
        z = Z[idx]
        if not centers:
            center_indices.append(int(idx))
            centers.append(z)
        else:
            center_raw = H[center_indices]
            min_dist = np.min(weighted_l1_distances(
                z,
                np.vstack(centers),
                weights,
                A_raw=H[idx],
                B_raw=center_raw,
                rarity_model=rarity_model,
                distance_mode=distance_mode,
            ))
            if min_dist >= tau:
                center_indices.append(int(idx))
                centers.append(z)
        if len(centers) >= n_centers:
            break

    if len(centers) < min(n_centers, len(order)):
        for idx in order:
            if int(idx) in center_indices:
                continue
            center_indices.append(int(idx))
            centers.append(Z[idx])
            if len(centers) >= n_centers:
                break

    centers_z = np.vstack(centers)
    trust_regions = [
        TrustRegionState(
            dim=centers_z.shape[1],
            batch_size=batch_size,
            center_point=centers_z[i].copy(),
            best_x=int(center_indices[i]),
            best_value=float(g_train[center_indices[i]]),
        )
        for i in range(len(center_indices))
    ]

    return {
        "h_columns": h_columns,
        "all_h_columns": all_h_columns,
        "references": references,
        "fill_values": fill_values,
        "h_scaling": "rank_percentile_useful_events",
        "distance_mode": distance_mode,
        "tau_quantile": float(tau_quantile),
        "rarity_model": rarity_model,
        "rarity_n_bins": int(rarity_n_bins),
        "rarity_clip": tuple(rarity_clip),
        "diversity_weight": float(diversity_weight),
        "weights": weights,
        "centers_z": centers_z,
        "centers_h_raw": H[center_indices].copy(),
        "center_indices": center_indices,
        "trust_regions": trust_regions,
        "h_lb": h_lb,
        "h_ub": h_ub,
        "tau": tau,
        "tail_reference_distances": tail_reference_distances,
        "tail_cdf_threshold": float(np.clip(tau_quantile / 100.0, 0.0, 1.0)),
        "useful_threshold": useful_threshold,
        "selected_h_z": [],
        "selected_h_raw": [],
        "last_selected_center_index": None,
    }


def predict_h_matrix(X, gp_models, scaler_X, h_columns):
    """Predict h(x) for many candidate x values using diagnostic h surrogates."""
    mags, _ = predict_magnitudes_batch(X, gp_models, scaler_X)
    cols = []
    for col in h_columns:
        if col not in mags:
            return None
        arr = np.asarray(mags[col], dtype=float)
        if len(arr) != len(X):
            return None
        cols.append(arr)
    H = np.column_stack(cols)
    return np.nan_to_num(H, nan=np.nanmedian(H, axis=0), posinf=0.0, neginf=0.0)


def select_robot_candidate(
    X_candidates,
    direct_scores,
    gp_models,
    scaler_X,
    robot_state,
):
    """
    Pick the highest-acquisition candidate satisfying ROBOT constraints.

    Trust region: paper-style rectangular box around each h-space center.
    Diversity: D(h_hat(x), h_new_j) >= tau from previously selected new points.
    The selected candidate is argmax acquisition inside the feasible set; the
    diversity distance is not blended into the acquisition score. If no candidate
    satisfies the fixed tau, choose from the most diverse available candidates
    and record the tau shortfall in the diagnostics.
    """
    if robot_state is None:
        best_idx = int(np.nanargmax(direct_scores))
        return best_idx, {"robot_used": False, "reason": "no_robot_state"}

    H_pred = predict_h_matrix(
        X_candidates, gp_models, scaler_X, robot_state["h_columns"]
    )
    if H_pred is None:
        best_idx = int(np.nanargmax(direct_scores))
        return best_idx, {"robot_used": False, "reason": "h_prediction_unavailable"}

    Z_pred = scale_h_to_unit(
        H_pred,
        robot_state["references"],
        robot_state["fill_values"],
    )
    distance_mode = robot_state.get("distance_mode", "tail_l1")
    rarity_model = robot_state.get("rarity_model")
    center_D = weighted_l1_distances(
        Z_pred,
        robot_state["centers_z"],
        robot_state["weights"],
        A_raw=H_pred,
        B_raw=robot_state.get("centers_h_raw"),
        rarity_model=rarity_model,
        distance_mode=distance_mode,
    )
    min_center_D = np.min(center_D, axis=1)
    center_index = np.argmin(center_D, axis=1)

    h_lb = robot_state["h_lb"]
    h_ub = robot_state["h_ub"]
    box_widths = h_ub - h_lb
    box_widths = np.where(np.isfinite(box_widths) & (box_widths > 1e-12), box_widths, 1.0)
    trust_regions = robot_state.get("trust_regions", [])
    in_trust_by_region = []
    for state in trust_regions:
        tr_lb = np.maximum(state.center_point - box_widths * state.length / 2.0, h_lb)
        tr_ub = np.minimum(state.center_point + box_widths * state.length / 2.0, h_ub)
        in_trust_by_region.append(np.all((Z_pred >= tr_lb) & (Z_pred <= tr_ub), axis=1))

    if in_trust_by_region:
        in_trust_matrix = np.column_stack(in_trust_by_region)
        in_trust = np.any(in_trust_matrix, axis=1)
    else:
        in_trust_matrix = np.zeros((len(X_candidates), 0), dtype=bool)
        in_trust = np.zeros(len(X_candidates), dtype=bool)

    selected = robot_state.get("selected_h_z", [])
    if selected:
        selected_D = weighted_l1_distances(
            Z_pred,
            np.vstack(selected),
            robot_state["weights"],
            A_raw=H_pred,
            B_raw=np.vstack(robot_state.get("selected_h_raw", [])),
            rarity_model=rarity_model,
            distance_mode=distance_mode,
        )
        min_selected_D = np.min(selected_D, axis=1)
    else:
        min_selected_D = np.full(len(X_candidates), np.inf)

    diversity_raw = min_selected_D.copy()
    if not np.any(np.isfinite(diversity_raw)):
        diversity_raw = min_center_D.copy()
    else:
        diversity_raw = np.where(np.isfinite(diversity_raw), diversity_raw, min_center_D)

    finite_score = np.isfinite(direct_scores)
    selected_tau = float(robot_state["tau"])
    tail_weights = np.zeros(len(X_candidates), dtype=float)
    tail_cdf = np.zeros(len(X_candidates), dtype=float)
    tail_rank = np.zeros(len(X_candidates), dtype=float)
    if distance_mode == "tail_l1":
        tail_weights, tail_cdf, tail_rank = tail_l1_weights(
            diversity_raw,
            robot_state.get("tail_reference_distances", []),
            float(robot_state.get("tau_quantile", 85.0)),
            selected_tau,
        )
    strict_feasible = finite_score & in_trust & (diversity_raw >= selected_tau)
    tau_satisfied = bool(np.any(strict_feasible))
    tau_fallback_reason = None
    best_available_diversity = None
    tau_shortfall = 0.0

    if tau_satisfied:
        feasible = strict_feasible
    else:
        fallback_pool = finite_score & in_trust
        if np.any(fallback_pool):
            tau_fallback_reason = "no_candidate_met_fixed_tau"
        else:
            fallback_pool = finite_score
            tau_fallback_reason = "no_candidate_in_trust_region_met_fixed_tau"

        if np.any(fallback_pool):
            pool_div = diversity_raw[fallback_pool]
            finite_pool_div = pool_div[np.isfinite(pool_div)]
            if finite_pool_div.size:
                best_available_diversity = float(np.nanmax(finite_pool_div))
                tau_shortfall = max(selected_tau - best_available_diversity, 0.0)
                feasible = fallback_pool & (diversity_raw >= best_available_diversity - 1e-12)
            else:
                feasible = fallback_pool
                tau_shortfall = selected_tau
        else:
            feasible = finite_score
            tau_fallback_reason = "no_finite_candidate"
            tau_shortfall = selected_tau

    score_norm = np.zeros(len(X_candidates), dtype=float)
    div_norm = np.zeros(len(X_candidates), dtype=float)
    feasible_scores = direct_scores[feasible]
    if len(feasible_scores):
        s_min = float(np.nanmin(feasible_scores))
        s_max = float(np.nanmax(feasible_scores))
        score_norm[feasible] = (feasible_scores - s_min) / max(s_max - s_min, 1e-12)

    feasible_div = diversity_raw[feasible]
    if len(feasible_div):
        d_min = float(np.nanmin(feasible_div))
        d_max = float(np.nanmax(feasible_div))
        div_norm[feasible] = (feasible_div - d_min) / max(d_max - d_min, 1e-12)

    masked_scores = np.where(feasible, direct_scores, -np.inf)
    best_idx = int(np.nanargmax(masked_scores))

    if in_trust_matrix.shape[1] and np.any(in_trust_matrix[best_idx]):
        selected_center_index = int(np.flatnonzero(in_trust_matrix[best_idx])[0])
    else:
        selected_center_index = int(center_index[best_idx])

    robot_state["selected_h_z"].append(Z_pred[best_idx].copy())
    robot_state["selected_h_raw"].append(H_pred[best_idx].copy())
    robot_state["last_selected_center_index"] = selected_center_index
    selected_state = trust_regions[selected_center_index] if trust_regions else None
    return best_idx, {
        "robot_used": True,
        "robot_distance_mode": distance_mode,
        "robot_tau_quantile": robot_state.get("tau_quantile"),
        "feasible_candidates": int(np.sum(feasible)),
        "trust_region_candidates": int(np.sum(in_trust)),
        "robot_tau": float(robot_state["tau"]),
        "robot_tau_used": float(selected_tau),
        "robot_tau_satisfied": tau_satisfied,
        "robot_tau_fallback_reason": tau_fallback_reason,
        "robot_best_available_diversity": best_available_diversity,
        "robot_tau_shortfall": float(tau_shortfall),
        "robot_selection_rule": "constrained_argmax_acquisition",
        "robot_diversity_weight": 0.0,
        "robot_blended_score": np.nan,
        "robot_acquisition_score": float(direct_scores[best_idx]),
        "robot_acquisition_score_norm": float(score_norm[best_idx]),
        "robot_diversity_score_norm": float(div_norm[best_idx]),
        "robot_tail_cdf_threshold": robot_state.get("tail_cdf_threshold"),
        "robot_tail_cdf": float(tail_cdf[best_idx]),
        "robot_tail_rank": float(tail_rank[best_idx]),
        "robot_tail_weight": float(tail_weights[best_idx]),
        "robot_length": None if selected_state is None else float(selected_state.length),
        "robot_length_min": None if selected_state is None else float(selected_state.length_min),
        "robot_length_max": None if selected_state is None else float(selected_state.length_max),
        "robot_center_distance": float(min_center_D[best_idx]),
        "robot_selected_distance": (
            None if not selected else float(min_selected_D[best_idx])
        ),
        "robot_center_index": selected_center_index,
        "predicted_h": {
            col: float(H_pred[best_idx, j])
            for j, col in enumerate(robot_state["h_columns"])
        },
    }


def update_robot_state(robot_state, y_next):
    """Update the selected trust-region state after the expensive evaluation."""
    if robot_state is None:
        return robot_state
    selected_center_index = robot_state.get("last_selected_center_index")
    trust_regions = robot_state.get("trust_regions", [])
    if selected_center_index is None or selected_center_index >= len(trust_regions):
        return robot_state
    update_state(trust_regions[selected_center_index], [y_next])
    return robot_state
