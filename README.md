# Sim2Science upload package

This package contains two paper-facing code folders:

- `bo_eicf_robot`: gray-box EI-CF + ROBOT code.
- `bo_ei_robot`: EI + ROBOT baseline.

The package includes the multi-seed result tables and plots used for the paper
comparison.

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

## External Paths

Set these environment variables before running on a cluster:

```bash
export NMMA_WORKDIR=/path/to/nmma1
export NMMA_SVD_PATH=/path/to/nmma1/svdmodels
export NMMA_PRIOR_FILE=/path/to/nmma1/priors/Bu2019lm.prior
export BO_OUTPUT_BASE=./out
```

## Example Multi-Seed Runs

EI-CF + ROBOT:

```bash
cd bo_eicf_robot
python run_multi_seed.py --seeds 1-5 --n-iterations 30 \
  --acquisition ROBOT_EICF \
  --robot-distance-mode tail_l1 \
  --robot-tau-quantile 75 \
  --candidate-samples 1500 \
  --eicf-mc-samples 16
```

EI + ROBOT:

```bash
cd bo_ei_robot
python run_multi_seed.py --seeds 1-5 --n-iterations 30 \
  --acquisition ROBOT_DIRECT_EI \
  --robot-distance-mode tail_l1 \
  --robot-tau-quantile 75
```

## Included Results

The uploaded result folders are:

- `bo_eicf_robot/results/multi_seed_083126_114517`
- `bo_eicf_robot/results/multi_seed_083126_114517plots`
- `bo_ei_robot/results/multi_seed_direct_gp`
- `bo_ei_robot/results/multi_seed_direct_gp_plots`

These folders contain processed multi-seed outputs and plots.