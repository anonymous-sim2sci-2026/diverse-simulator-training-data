# Diverse simulator training data

This repository contains the three Bayesian-optimization methods compared in
the paper:

- `bo_eicf_robot`: EI-CF active search with ranked output-space ROBOT.
- `bo_eicf`: EI-CF active search without ROBOT (diversity ablation).
- `bo_ei_robot`: direct EI with ranked output-space ROBOT.

Each folder is self-contained and includes its initial O5a/O5b/O5c training
tables, command-line entry points, requirements, and method-specific README.
Generated NMMA products, GP caches, and checkpoints are intentionally omitted.

## Prerequisites

Before running these scripts, install the NMMA codebase and create/activate an
NMMA-compatible Python environment. The closed-loop simulations call NMMA's
`lightcurve-analysis` machinery and require access to the NMMA repository,
SVD model directory, and prior file.

For example:

```bash
git clone https://github.com/nuclear-multimessenger-astronomy/nmma.git
cd nmma
conda env create -f environment.yml
conda activate nmma_env
pip install -e .
```

## External dependencies

Install NMMA and set the external model paths before running a method:

```bash
export NMMA_WORKDIR=/path/to/nmma/source/checkout
export SVD_PATH=/path/to/svdmodels
export PRIOR_FILE=/path/to/priors/Bu2019lm.prior
export OUTPUT_BASE=./out
```

The legacy names `NMMA_SVD_PATH`, `NMMA_PRIOR_FILE`, and `BO_OUTPUT_BASE` are
also accepted.

## Paper runs

Each folder documents its complete single-seed and multi-seed commands. The
main entry points are:

```bash
(cd bo_eicf_robot && python run_multi_seed.py --seeds 1-3 --n-iterations 80 --no-plots)
(cd bo_eicf       && python run_multi_seed.py --seeds 1-3 --n-iterations 80 --no-plots)
(cd bo_ei_robot   && python run_multi_seed.py --seeds 1-3 --n-iterations 80 --no-plots)
```

For the ROBOT methods, an iteration is a BO batch and may contain multiple
simulations. For EI-CF without ROBOT, one iteration produces one selected
simulation.
