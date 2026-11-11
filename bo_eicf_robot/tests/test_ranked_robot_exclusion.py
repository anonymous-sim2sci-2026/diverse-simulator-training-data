import numpy as np

from optimization.batch_bayesian_optimization import rotating_region_ranks
from optimization.robot_h import (
    TrustRegionState,
    select_robot_candidate,
    update_robot_state,
)


def test_single_region_rotates_through_ranked_hierarchy():
    state = {"next_region_rank": 0}
    observed = []
    for _ in range(6):
        ranks, start = rotating_region_ranks(
            state, active_region_count=3, regions_per_batch=1
        )
        observed.append(ranks[0])
        state["next_region_rank"] = (start + len(ranks)) % 3

    assert observed == [0, 1, 2, 0, 1, 2]


def test_wrapped_region_window_is_processed_in_hierarchy_order():
    state = {"next_region_rank": 3}
    ranks, start = rotating_region_ranks(
        state, active_region_count=4, regions_per_batch=2
    )

    assert start == 3
    assert ranks == [0, 3]


def test_all_regions_are_processed_in_strict_rank_order():
    state = {"next_region_rank": 3}
    ranks, start = rotating_region_ranks(
        state, active_region_count=5, regions_per_batch=5
    )

    assert start == 0
    assert ranks == [0, 1, 2, 3, 4]


def _state():
    return {
        "h_columns": ["h0", "h1"],
        "references": [np.array([0.0, 1.0]), np.array([0.0, 1.0])],
        "fill_values": np.array([0.5, 0.5]),
        "weights": np.ones(2),
        "centers_z": [np.array([0.2, 0.2]), np.array([0.8, 0.8])],
        "centers_h_raw": [np.array([0.2, 0.2]), np.array([0.8, 0.8])],
        "h_lb": np.zeros(2),
        "h_ub": np.ones(2),
        "tau": 0.5,
        "tau_quantile": 75.0,
        "tail_cdf_threshold": 0.75,
        "tail_reference_distances": np.array([0.2, 0.5, 0.8, 1.2]),
        "distance_mode": "tail_l1",
        "rarity_model": None,
        "knn_k": 5,
        "selected_h_z": [],
        "selected_h_raw": [],
        "trust_regions": [
            TrustRegionState(dim=2, batch_size=1, center_point=np.array([0.2, 0.2]), length=1.0),
            TrustRegionState(dim=2, batch_size=1, center_point=np.array([0.8, 0.8]), length=1.0),
        ],
    }


def test_lower_rank_excludes_overlap_and_selected_neighborhood():
    state = _state()
    X = np.arange(8, dtype=float).reshape(4, 2)
    Z = np.array(
        [
            [0.2, 0.2],  # T1 only
            [0.5, 0.5],  # overlap: owned by T1
            [0.71, 0.71],  # T2 only, but too close to T1 selection
            [0.95, 0.95],  # T2 only and sufficiently separated
        ]
    )
    scores = np.array([0.1, 1.0, 0.9, 0.8])

    first, _ = select_robot_candidate(
        X, scores, None, None, state, region_index=0, H_pred=Z, Z_pred=Z
    )
    assert first == 1

    second, diagnostics = select_robot_candidate(
        X,
        scores,
        None,
        None,
        state,
        region_index=1,
        selected_h_z=[Z[first]],
        selected_h_raw=[Z[first]],
        H_pred=Z,
        Z_pred=Z,
    )

    assert second == 3
    assert diagnostics["robot_overlap_candidates"] == 1
    assert diagnostics["robot_overlap_excluded"] == 0
    assert diagnostics["robot_similarity_excluded"] == 2
    assert diagnostics["robot_higher_rank_selections"] == 1
    assert diagnostics["robot_selected_distance"] >= state["tau"]


def test_region_batch_updates_success_counter_once():
    region = TrustRegionState(
        dim=2,
        batch_size=3,
        center_point=np.array([0.5, 0.5]),
        best_value=0.2,
    )
    state = {
        "trust_regions": [region],
        "h_columns": ["h0", "h1"],
        "evaluated_x": np.empty((0, 2)),
        "evaluated_h_raw": np.empty((0, 2)),
        "evaluated_scores": np.array([]),
        "useful_threshold": 0.5,
        "last_selected_distance_bin": None,
        "selected_h_z": [],
        "selected_h_raw": [],
        "centers_z": [np.array([0.5, 0.5])],
        "centers_h_raw": [np.array([0.5, 0.5])],
        "weights": np.ones(2),
        "initial_trust_region_length": 0.6,
    }

    update_robot_state(
        state,
        [0.3, 0.8, 0.7],
        x_next=[[1, 2], [3, 4], [5, 6]],
        h_next=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        selected_center_index=0,
        rebuild_centers=False,
    )

    assert region.success_counter == 1
    assert len(state["evaluated_scores"]) == 3
