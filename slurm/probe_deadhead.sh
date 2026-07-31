#!/bin/bash
# =============================================================================
# Slurm: dead-head root-cause probe (iter-6).
# Measures per-logical label flip rate vs lz weight for BB-72 and BB-144 to
# confirm whether heads 8-11 die because their high-weight lz representatives
# make the observable marginal ~Bernoulli(0.5) (BCE floor = ln2).
# Pure stim + numpy, runs in seconds; dev partition.
#   sbatch slurm/probe_deadhead.sh
# =============================================================================

#SBATCH --job-name=dead_probe
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/dead_probe_%j.out
#SBATCH --error=logs/dead_probe_%j.err
#SBATCH --mail-type=FAIL

set -euo pipefail

WORKDIR="/work/u2467370/QEC/cascade"
cd "$WORKDIR"
mkdir -p logs

source "$WORKDIR/.venv/bin/activate"
export PYTHONUNBUFFERED=1

echo "========================================"
echo "Job ID : ${SLURM_JOB_ID}"
echo "Node   : $(hostname)"
echo "Start  : $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

python scripts/30_deadhead_label_probe.py

echo "========================================"
echo "End    : $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
