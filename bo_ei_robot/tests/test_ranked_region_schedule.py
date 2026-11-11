from optimization.batch_bayesian_optimization import rotating_region_ranks


def test_all_regions_are_processed_in_strict_rank_order():
    state = {"next_region_rank": 3}
    ranks, start = rotating_region_ranks(
        state, active_region_count=5, regions_per_batch=5
    )

    assert start == 0
    assert ranks == [0, 1, 2, 3, 4]


def test_reduced_region_schedule_still_rotates():
    state = {"next_region_rank": 2}
    ranks, start = rotating_region_ranks(
        state, active_region_count=4, regions_per_batch=1
    )

    assert start == 2
    assert ranks == [2]
