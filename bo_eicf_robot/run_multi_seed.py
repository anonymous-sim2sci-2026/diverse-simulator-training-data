#!/usr/bin/env python3
"""Run the paper's ranked EI-CF+ROBOT experiment for multiple seeds."""

import argparse
import os

import pandas as pd

from run_bo import run_single


def parse_seeds(text):
    seeds = []
    for item in text.split(","):
        item = item.strip()
        if "-" in item:
            start, stop = (int(value) for value in item.split("-", 1))
            seeds.extend(range(start, stop + 1))
        elif item:
            seeds.append(int(item))
    return seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="1-3")
    parser.add_argument("--n-iterations", type=int, default=80)
    parser.add_argument("--output-root", default="out/paper_eicf_robot")
    parser.add_argument("--robot-tau-quantile", type=float, default=75.0)
    parser.add_argument("--robot-initial-length", type=float, default=0.6)
    parser.add_argument("--robot-success-tolerance", type=int, default=10)
    parser.add_argument("--robot-failure-tolerance", type=int, default=32)
    parser.add_argument("--robot-n-regions", type=int, default=5)
    parser.add_argument("--robot-batch-regions", type=int, default=5)
    parser.add_argument("--robot-candidates-per-region", type=int, default=1)
    parser.add_argument("--parallel-simulations", type=int, default=5)
    parser.add_argument("--candidate-samples", type=int, default=1500)
    parser.add_argument("--candidate-uniform-samples", type=int, default=1500)
    parser.add_argument("--robot-region-candidates", type=int, default=1500)
    parser.add_argument("--eicf-mc-samples", type=int, default=16)
    parser.add_argument("--eicf-batch-size", type=int, default=512)
    parser.add_argument("--output-gp-refit-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--max-runtime-hours", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-refit-output-gps", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    summary = []
    for seed in parse_seeds(args.seeds):
        result = run_single(
            seed=seed,
            n_iterations=args.n_iterations,
            acquisition="ROBOT_EICF",
            output_folder=os.path.join(args.output_root, f"seed_{seed}"),
            make_plots=not args.no_plots,
            robot_distance_mode="tail_l1",
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
            eicf_mc_samples=args.eicf_mc_samples,
            eicf_batch_size=args.eicf_batch_size,
            skip_detection_gp_refit=True,
            checkpoint_every=args.checkpoint_every,
            resume=args.resume,
            max_runtime_hours=args.max_runtime_hours,
            refit_output_gps=not args.no_refit_output_gps,
            output_gp_refit_every=args.output_gp_refit_every,
            scoring_progress_every=0,
        )
        events = result["new_events_df"]
        summary.append(
            {
                "seed": seed,
                "events": len(events),
                "measurable_fraction": (
                    events["active_search_detectable"].mean()
                    if "active_search_detectable" in events else float("nan")
                ),
                "output_folder": result["output_folder"],
            }
        )

    path = os.path.join(args.output_root, "multi_seed_summary.csv")
    pd.DataFrame(summary).to_csv(path, index=False)
    print(f"Saved multi-seed summary to {path}")


if __name__ == "__main__":
    main()
