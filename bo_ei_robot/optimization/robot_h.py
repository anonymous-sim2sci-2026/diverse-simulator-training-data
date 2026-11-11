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


ROBOT_ACQUISITIONS = {"ROBOT_EICF"}


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

ROBOT_DISTANCE_MODES = {"tail_l1", "knn_l1"}
INITIAL_TRUST_REGION_LENGTH = 0.6


@dataclass
class TrustRegionState:
    """Paper-style trust-region state for one h-space center."""

    dim: int
    batch_size: int
    center_point: np.ndarray
    best_x: Any = ""
    best_value: float = -float("inf")
    length: float = INITIAL_TRUST_REGION_LENGTH
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
        state.failure_counter += 1

    if state.success_counter >= state.success_tolerance:
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter >= state.failure_tolerance:
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


def robot_trust_region_mask(
    X_candidates, gp_models, scaler_X, robot_state, region_index=None
):
    """Return candidates whose GP-predicted h values lie in a requested region."""
    if robot_state is None:
        return np.zeros(len(X_candidates), dtype=bool)

    H_pred = predict_h_matrix(
        X_candidates, gp_models, scaler_X, robot_state["h_columns"]
    )
    if H_pred is None:
        raise RuntimeError(
            "ROBOT cannot evaluate trust-region membership because h(x) "
            "predictions are unavailable."
        )
    Z_pred = scale_h_to_unit(
        H_pred, robot_state["references"], robot_state["fill_values"]
    )
    h_lb = robot_state["h_lb"]
    h_ub = robot_state["h_ub"]
    box_widths = h_ub - h_lb
    box_widths = np.where(
        np.isfinite(box_widths) & (box_widths > 1e-12), box_widths, 1.0
    )
    trust_regions = robot_state.get("trust_regions", [])
    if region_index is not None:
        if not 0 <= int(region_index) < len(trust_regions):
            return np.zeros(len(X_candidates), dtype=bool)
        trust_regions = [trust_regions[int(region_index)]]
    masks = []
    for state in trust_regions:
        tr_lb = np.maximum(
            state.center_point - box_widths * state.length / 2.0, h_lb
        )
        tr_ub = np.minimum(
            state.center_point + box_widths * state.length / 2.0, h_ub
        )
        masks.append(np.all((Z_pred >= tr_lb) & (Z_pred <= tr_ub), axis=1))
    if not masks:
        raise RuntimeError("ROBOT has no active h-space trust regions.")
    return np.any(np.column_stack(masks), axis=1)


def rebuild_ranked_robot_centers(robot_state):
    """Reconstruct rank-ordered centers from the shared evaluated history."""
    if robot_state is None:
        return robot_state
    H = np.asarray(robot_state.get("evaluated_h_raw", []), dtype=float)
    scores = np.asarray(robot_state.get("evaluated_scores", []), dtype=float)
    trust_regions = robot_state.get("trust_regions", [])
    if H.ndim != 2 or not len(H) or len(scores) != len(H) or not trust_regions:
        return robot_state

    Z = scale_h_to_unit(H, robot_state["references"], robot_state["fill_values"])
    finite = np.all(np.isfinite(Z), axis=1) & np.isfinite(scores)
    order = np.flatnonzero(finite)
    order = order[np.argsort(-scores[order])]
    tau = float(robot_state["tau"])
    chosen = []
    for idx in order:
        if not chosen:
            chosen.append(int(idx))
        else:
            distances = weighted_l1_distances(
                Z[idx], Z[chosen], robot_state["weights"]
            ).reshape(-1)
            if np.all(distances >= tau - 1e-12):
                chosen.append(int(idx))
        if len(chosen) == len(trust_regions):
            break

    for rank, history_idx in enumerate(chosen):
        state = trust_regions[rank]
        state.center_point = Z[history_idx].copy()
        state.best_x = int(history_idx)
        state.best_value = float(scores[history_idx])
        robot_state["centers_z"][rank] = state.center_point
        robot_state["centers_h_raw"][rank] = H[history_idx]
    robot_state["ranked_center_history_indices"] = chosen
    # Only centers that satisfy the fixed pairwise diversity rule are active.
    # Padding the hierarchy with non-diverse centers makes their later
    # feasibility constraints internally inconsistent.
    robot_state["active_ranked_region_count"] = len(chosen)
    return robot_state


def robot_region_input_center(robot_state, region_index):
    """Return the evaluated input associated with one ranked output center."""
    center_indices = robot_state.get("ranked_center_history_indices", [])
    X = np.asarray(robot_state.get("evaluated_x", []), dtype=float)
    if not 0 <= int(region_index) < len(center_indices):
        return None
    idx = int(center_indices[int(region_index)])
    if X.ndim != 2 or X.shape[1] == 0 or not 0 <= idx < len(X):
        return None
    return X[idx].copy()


def robot_ranked_region_mask(
    X_candidates,
    gp_models,
    scaler_X,
    robot_state,
    region_index,
    selected_h_z=None,
    H_pred=None,
    Z_pred=None,
):
    """Return region membership after ranked exclusion by earlier selections."""
    region_index = int(region_index)
    if H_pred is None:
        H_pred = predict_h_matrix(
            X_candidates, gp_models, scaler_X, robot_state["h_columns"]
        )
    if Z_pred is None:
        Z_pred = scale_h_to_unit(
            H_pred, robot_state["references"], robot_state["fill_values"]
        )
    trust_regions = robot_state.get("trust_regions", [])
    if not 0 <= region_index < len(trust_regions):
        raise IndexError(f"Invalid ROBOT region rank {region_index}")
    h_lb = robot_state["h_lb"]
    h_ub = robot_state["h_ub"]
    box_widths = np.where((h_ub - h_lb) > 1e-12, h_ub - h_lb, 1.0)
    memberships = []
    for state in trust_regions:
        lower = np.maximum(state.center_point - box_widths * state.length / 2.0, h_lb)
        upper = np.minimum(state.center_point + box_widths * state.length / 2.0, h_ub)
        memberships.append(np.all((Z_pred >= lower) & (Z_pred <= upper), axis=1))
    membership_matrix = np.column_stack(memberships)
    in_region = membership_matrix[:, region_index]

    selected_h_z = np.asarray(selected_h_z if selected_h_z is not None else [], dtype=float)
    if selected_h_z.size == 0 or not np.any(in_region):
        return in_region
    distances = weighted_l1_distances(
        Z_pred, selected_h_z, robot_state["weights"]
    )
    return in_region & np.all(distances >= float(robot_state["tau"]) - 1e-12, axis=1)


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
    X_train=None,
    h_columns=None,
    useful_threshold=0.5,
    n_centers=10,
    tau_quantile=85.0,
    batch_size=1,
    distance_mode="tail_l1",
    knn_k=5,
    rarity_n_bins=20,
    rarity_clip=(0.5, 5.0),
    diversity_weight=0.35,
    distance_bounds=(55.0, 722.0),
    n_distance_bins=12,
    initial_trust_region_length=INITIAL_TRUST_REGION_LENGTH,
    success_tolerance=10,
    failure_tolerance=32,
):
    """
    Initialize ROBOT-style h-space trust centers from existing useful samples.

    Centers are chosen greedily from training events with P_detect above
    useful_threshold. For tail_l1, tau_quantile is the retained upper-tail
    percentage, so 25 fixes tau at the 75th percentile of the initial empirical
    pairwise-distance distribution.
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
    knn_k = int(knn_k)
    if knn_k < 1:
        raise ValueError("knn_k must be at least 1")
    initial_trust_region_length = float(initial_trust_region_length)
    if not 0.05 <= initial_trust_region_length <= 1.0:
        raise ValueError("initial_trust_region_length must be between 0.05 and 1.0")
    success_tolerance = int(success_tolerance)
    failure_tolerance = int(failure_tolerance)
    if success_tolerance < 1 or failure_tolerance < 1:
        raise ValueError("ROBOT success/failure tolerances must be at least 1")

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
    fixed_tau_percentile = 100.0 - float(tau_quantile)
    tau = float(np.percentile(nonzero, fixed_tau_percentile)) if len(nonzero) else 1.0
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
            length=initial_trust_region_length,
            success_tolerance=success_tolerance,
            failure_tolerance=failure_tolerance,
        )
        for i in range(len(center_indices))
    ]

    distance_edges = np.linspace(
        float(distance_bounds[0]), float(distance_bounds[1]), int(n_distance_bins) + 1
    )
    if "distance" in h_columns:
        distance_col = h_columns.index("distance")
        initial_distance_counts, _ = np.histogram(
            useful_H[:, distance_col], bins=distance_edges
        )
    else:
        initial_distance_counts = np.zeros(int(n_distance_bins), dtype=int)

    return {
        "h_columns": h_columns,
        "all_h_columns": all_h_columns,
        "references": references,
        "fill_values": fill_values,
        "h_scaling": "rank_percentile_useful_events",
        "distance_mode": distance_mode,
        "knn_k": knn_k,
        "tau_quantile": float(tau_quantile),
        "success_tolerance": success_tolerance,
        "failure_tolerance": failure_tolerance,
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
        "tail_cdf_threshold": float(np.clip(fixed_tau_percentile / 100.0, 0.0, 1.0)),
        "useful_threshold": useful_threshold,
        "selected_h_z": [],
        "selected_h_raw": [],
        "last_selected_center_index": None,
        "distance_bin_edges": distance_edges,
        "distance_bin_counts": initial_distance_counts.astype(int),
        "last_selected_distance_bin": None,
        "initial_trust_region_length": initial_trust_region_length,
        "evaluated_x": (
            np.asarray(X_train, dtype=float).copy()
            if X_train is not None
            else np.empty((len(H), 0), dtype=float)
        ),
        "evaluated_h_raw": H.copy(),
        "evaluated_scores": g_train.copy(),
        "ranked_center_history_indices": [int(i) for i in center_indices],
        "next_region_rank": 0,
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
    region_index=0,
    selected_h_z=None,
    selected_h_raw=None,
    H_pred=None,
    Z_pred=None,
):
    """
    Pick one candidate from a rank-ordered output-space trust region.

    Region i is constrained by the fixed tau distance from higher-ranked
    centers and candidates selected earlier in the current ranked batch.
    Trust-region box overlap is diagnostic and is not an additional exclusion.
    """
    if robot_state is None:
        best_idx = int(np.nanargmax(direct_scores))
        return best_idx, {"robot_used": False, "reason": "no_robot_state"}

    if H_pred is None:
        H_pred = predict_h_matrix(
            X_candidates, gp_models, scaler_X, robot_state["h_columns"]
        )
    if H_pred is None:
        best_idx = int(np.nanargmax(direct_scores))
        return best_idx, {"robot_used": False, "reason": "h_prediction_unavailable"}

    if Z_pred is None:
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
    if not 0 <= int(region_index) < len(trust_regions):
        raise IndexError(f"Invalid ROBOT region rank {region_index}")
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

    higher_selected = np.asarray(
        selected_h_z if selected_h_z is not None else [], dtype=float
    )
    higher_selected_raw = np.asarray(
        selected_h_raw if selected_h_raw is not None else [], dtype=float
    )
    selected_D = None
    if higher_selected.size:
        selected_D = weighted_l1_distances(
            Z_pred,
            higher_selected,
            robot_state["weights"],
            A_raw=H_pred,
            B_raw=higher_selected_raw if higher_selected_raw.size else None,
            rarity_model=rarity_model,
            distance_mode=distance_mode,
        )
        if distance_mode == "knn_l1":
            k_eff = min(int(robot_state.get("knn_k", 5)), selected_D.shape[1])
            nearest_partition = np.partition(selected_D, kth=k_eff - 1, axis=1)
            min_selected_D = np.mean(nearest_partition[:, :k_eff], axis=1)
        else:
            min_selected_D = np.min(selected_D, axis=1)
    else:
        min_selected_D = np.full(len(X_candidates), np.inf)

    diversity_raw = min_selected_D.copy()
    if not np.any(np.isfinite(diversity_raw)):
        diversity_raw = min_center_D.copy()
    else:
        diversity_raw = np.where(np.isfinite(diversity_raw), diversity_raw, min_center_D)

    finite_score = np.isfinite(direct_scores)
    base_tau = float(robot_state["tau"])
    tail_weights = np.zeros(len(X_candidates), dtype=float)
    tail_cdf = np.zeros(len(X_candidates), dtype=float)
    tail_rank = np.zeros(len(X_candidates), dtype=float)
    if distance_mode == "tail_l1":
        tail_weights, tail_cdf, tail_rank = tail_l1_weights(
            diversity_raw,
            robot_state.get("tail_reference_distances", []),
            100.0 - float(robot_state.get("tau_quantile", 25.0)),
            base_tau,
        )
    region_membership = in_trust_matrix[:, int(region_index)].copy()
    overlap_with_higher_regions = np.zeros(len(X_candidates), dtype=bool)
    if int(region_index):
        overlap_with_higher_regions = np.any(
            in_trust_matrix[:, : int(region_index)], axis=1
        )
    # Overlapping boxes are diagnostic only. The ranked hierarchy is enforced
    # by the fixed-tau separation from higher-ranked centers/selections below.
    # Rejecting box overlap as well as enforcing tau can make every lower
    # region empty when the trust regions are intentionally broad.
    candidate_pool = finite_score & region_membership
    tau_fallback_reason = None
    if not np.any(candidate_pool):
        raise RuntimeError(
            "No finite-scored candidate lies inside an active ROBOT trust "
            "region. Candidate resampling must occur before ROBOT selection."
        )

    pool_distances = diversity_raw[candidate_pool]
    pool_distances = pool_distances[np.isfinite(pool_distances)]
    selected_tau = base_tau
    if not higher_selected.size:
        feasible = candidate_pool
    else:
        feasible = candidate_pool & (diversity_raw >= selected_tau - 1e-12)
    tau_satisfied = bool(np.any(feasible))
    best_available_diversity = (
        float(np.nanmax(pool_distances)) if len(pool_distances) else None
    )
    tau_shortfall = 0.0

    if not np.any(feasible):
        raise RuntimeError(
            f"No candidate in ranked ROBOT region {region_index} satisfies fixed tau={base_tau:.4f}."
        )
    candidate_bins = np.full(len(X_candidates), -1, dtype=int)
    distance_balancing_used = False

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

    # Preserve the exact selection-time L1 decomposition. These values use
    # the same CDF references, weights, and selected-event state as ROBOT.
    nearest_selected_index = None
    neighbor_selected_indices = []
    selected_l1_contributions = np.full(len(robot_state["h_columns"]), np.nan)
    selected_net_l1 = None
    if higher_selected.size:
        ordered_neighbors = np.argsort(selected_D[best_idx])
        k_eff = (
            min(int(robot_state.get("knn_k", 5)), len(higher_selected))
            if distance_mode == "knn_l1"
            else 1
        )
        neighbor_selected_indices = [
            int(index) for index in ordered_neighbors[:k_eff]
        ]
        nearest_selected_index = neighbor_selected_indices[0]
        neighbor_z = np.asarray(
            [higher_selected[index] for index in neighbor_selected_indices], dtype=float
        )
        per_neighbor_contributions = (
            np.abs(Z_pred[best_idx][None, :] - neighbor_z)
            * np.asarray(robot_state["weights"], dtype=float)[None, :]
        )
        selected_l1_contributions = np.mean(
            per_neighbor_contributions, axis=0
        )
        selected_net_l1 = float(np.sum(selected_l1_contributions))
        if not np.isclose(
            selected_net_l1,
            float(min_selected_D[best_idx]),
            rtol=1e-12,
            atol=1e-12,
        ):
            raise RuntimeError(
                "Saved ROBOT L1 contributions do not reproduce the selected distance."
            )
    selection_margin = (
        None if selected_net_l1 is None else float(selected_net_l1 - selected_tau)
    )

    selected_center_index = int(region_index)

    robot_state["selected_h_z"].append(Z_pred[best_idx].copy())
    robot_state["selected_h_raw"].append(H_pred[best_idx].copy())
    robot_state["last_selected_center_index"] = selected_center_index
    selected_distance_bin = None
    robot_state["last_selected_distance_bin"] = selected_distance_bin
    selected_state = trust_regions[selected_center_index] if trust_regions else None
    return best_idx, {
        "robot_used": True,
        "robot_distance_mode": distance_mode,
        "robot_tau_quantile": robot_state.get("tau_quantile"),
        "feasible_candidates": int(np.sum(feasible)),
        "trust_region_candidates": int(np.sum(in_trust)),
        "robot_region_candidates_before_hierarchy": int(np.sum(region_membership)),
        "robot_overlap_candidates": int(
            np.sum(region_membership & overlap_with_higher_regions)
        ),
        "robot_overlap_excluded": 0,
        "robot_similarity_excluded": int(
            np.sum(candidate_pool & (diversity_raw < selected_tau - 1e-12))
            if higher_selected.size
            else 0
        ),
        "robot_higher_rank_selections": int(len(higher_selected)),
        "robot_tau": base_tau,
        "robot_tau_used": float(selected_tau),
        "robot_tau_satisfied": tau_satisfied,
        "robot_tau_fallback_reason": tau_fallback_reason,
        "robot_best_available_diversity": best_available_diversity,
        "robot_tau_shortfall": float(tau_shortfall),
        "robot_selection_rule": "rotating_ranked_fixed_tau_then_argmax",
        "robot_dynamic_tau_percentile": None,
        "robot_tail_fraction": None,
        "robot_distance_balancing_used": distance_balancing_used,
        "robot_distance_bin": selected_distance_bin,
        "robot_distance_bin_count_before": (
            None if selected_distance_bin is None else int(distance_counts[selected_distance_bin])
        ),
        "robot_distance_bin_lower": (
            None if selected_distance_bin is None else float(distance_edges[selected_distance_bin])
        ),
        "robot_distance_bin_upper": (
            None if selected_distance_bin is None else float(distance_edges[selected_distance_bin + 1])
        ),
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
            selected_net_l1
        ),
        "robot_nearest_selected_index": nearest_selected_index,
        "robot_neighbor_selected_indices": neighbor_selected_indices,
        "robot_effective_neighbor_count": len(neighbor_selected_indices),
        "robot_knn_k": int(robot_state.get("knn_k", 5)),
        "robot_l1_contribution_sum": selected_net_l1,
        "robot_selection_margin": selection_margin,
        "robot_l1_contributions": {
            col: float(selected_l1_contributions[j])
            for j, col in enumerate(robot_state["h_columns"])
            if np.isfinite(selected_l1_contributions[j])
        },
        "robot_center_index": selected_center_index,
        "robot_region_rank": selected_center_index,
        "predicted_h": {
            col: float(H_pred[best_idx, j])
            for j, col in enumerate(robot_state["h_columns"])
        },
    }


def update_robot_state(
    robot_state,
    y_next,
    x_next=None,
    h_next=None,
    selected_center_index=None,
    rebuild_centers=True,
):
    """Update one region once and append one or more evaluated outcomes."""
    if robot_state is None:
        return robot_state
    robot_state["last_restart"] = None
    if selected_center_index is None:
        selected_center_index = robot_state.get("last_selected_center_index")
    else:
        selected_center_index = int(selected_center_index)
    trust_regions = robot_state.get("trust_regions", [])
    if selected_center_index is None or selected_center_index >= len(trust_regions):
        return robot_state
    selected_state = trust_regions[selected_center_index]
    y_values = np.atleast_1d(np.asarray(y_next, dtype=float))
    finite_y = y_values[np.isfinite(y_values)]
    if not len(finite_y):
        return robot_state
    region_best = float(np.max(finite_y))
    # Multiple candidates from one region constitute one region-level trial.
    update_state(selected_state, [region_best])

    if x_next is not None and h_next is not None:
        x_next = np.atleast_2d(np.asarray(x_next, dtype=float))
        h_next = np.atleast_2d(np.asarray(h_next, dtype=float))
        if (
            len(x_next) == len(h_next) == len(y_values)
            and h_next.shape[1] == len(robot_state["h_columns"])
        ):
            robot_state["evaluated_x"] = np.vstack(
                [np.asarray(robot_state["evaluated_x"], dtype=float), x_next]
            )
            robot_state["evaluated_h_raw"] = np.vstack(
                [np.asarray(robot_state["evaluated_h_raw"], dtype=float), h_next]
            )
            robot_state["evaluated_scores"] = np.append(
                np.asarray(robot_state["evaluated_scores"], dtype=float),
                y_values,
            )

    selected_distance_bin = robot_state.get("last_selected_distance_bin")
    if region_best >= float(robot_state.get("useful_threshold", 0.5)):
        counts = robot_state.get("distance_bin_counts")
        if counts is not None and selected_distance_bin is not None:
            if 0 <= int(selected_distance_bin) < len(counts):
                counts[int(selected_distance_bin)] += 1

    if selected_state.restart_triggered:
        selected_z = np.asarray(robot_state.get("selected_h_z", []), dtype=float)
        selected_raw = np.asarray(robot_state.get("selected_h_raw", []), dtype=float)

        if selected_z.ndim == 2 and len(selected_z):
            other_centers = [
                state.center_point
                for i, state in enumerate(trust_regions)
                if i != selected_center_index
            ]
            if other_centers:
                distances = weighted_l1_distances(
                    selected_z,
                    np.vstack(other_centers),
                    robot_state["weights"],
                )
                restart_point_index = int(np.argmax(np.min(distances, axis=1)))
            else:
                restart_point_index = len(selected_z) - 1

            selected_state.center_point = selected_z[restart_point_index].copy()
            robot_state["centers_z"][selected_center_index] = selected_state.center_point
            if selected_raw.ndim == 2 and len(selected_raw) > restart_point_index:
                robot_state["centers_h_raw"][selected_center_index] = selected_raw[
                    restart_point_index
                ].copy()
        else:
            restart_point_index = None

        previous_length = float(selected_state.length)
        selected_state.length = float(
            robot_state.get("initial_trust_region_length", INITIAL_TRUST_REGION_LENGTH)
        )
        selected_state.success_counter = 0
        selected_state.failure_counter = 0
        selected_state.restart_triggered = False
        selected_state.best_value = region_best
        robot_state["last_restart"] = {
            "center_index": int(selected_center_index),
            "selected_point_index": restart_point_index,
            "previous_length": previous_length,
            "new_length": float(selected_state.length),
        }
    robot_state["next_region_rank"] = (
        int(selected_center_index) + 1
    ) % max(len(trust_regions), 1)
    if rebuild_centers:
        return rebuild_ranked_robot_centers(robot_state)
    return robot_state
