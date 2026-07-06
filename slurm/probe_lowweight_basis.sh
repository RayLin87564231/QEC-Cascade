#!/bin/bash
# =============================================================================
# Slurm: verify minimum-weight logical basis (iter-6 dead-head fix #1).
# Confirms BB-72/BB-144 lz/lx are now min-weight reps (all lz weight <= 16),
# that ker/symplectic/independence/min-weight invariants still hold, and prints
# marginal flip rates + minimiser wall-clock. Pure stim + numpy, seconds.
#   sbatch slurm/probe_lowweight_basis.sh
# =============================================================================

#SBATCH --job-name=lw_basis_verify
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/lw_basis_verify_%j.out
#SBATCH --error=logs/lw_basis_verify_%j.err
#SBATCH --mail-type=FAIL

set -euo pipefail

WORKDIR="/work/leo07010/Ray/QEC/cascade"
cd "$WORKDIR"
mkdir -p logs

source "$WORKDIR/.venv/bin/activate"
export PYTHONUNBUFFERED=1

echo "========================================"
echo "Job ID : ${SLURM_JOB_ID}"
echo "Node   : $(hostname)"
echo "Start  : $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

python scripts/31_lowweight_basis_verify.py

echo "========================================"
echo "End    : $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
