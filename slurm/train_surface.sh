#!/bin/bash
# =============================================================================
# Slurm: Cascade neural decoder — surface code Track 2 training
# NCHC nano5 cluster
#
# Submit:
#   sbatch slurm/train_surface.sh 5    # distance=5
#   sbatch slurm/train_surface.sh 7
#   sbatch slurm/train_surface.sh 9
# =============================================================================

#SBATCH --job-name=cascade_surf
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --output=logs/cascade_surf_%j.out
#SBATCH --error=logs/cascade_surf_%j.err
#SBATCH --mail-type=FAIL

set -euo pipefail

WORKDIR="/work/leo07010/Ray/QEC/cascade"
cd "$WORKDIR"

mkdir -p logs

DISTANCE="${1:-5}"

# Distance-dependent hyperparameters (rule: H=128 minimum to enable waterfall;
# L scales with distance for sufficient receptive field).
case "$DISTANCE" in
  3)  HIDDEN=64;  BLOCKS=4;  STEPS=20000;  BATCH=1024 ;;
  5)  HIDDEN=128; BLOCKS=6;  STEPS=40000;  BATCH=512  ;;
  7)  HIDDEN=128; BLOCKS=8;  STEPS=60000;  BATCH=384  ;;
  9)  HIDDEN=128; BLOCKS=10; STEPS=80000;  BATCH=256  ;;
  *)  echo "Unknown distance $DISTANCE"; exit 1 ;;
esac

# Activate venv (assumes Phase 0 setup already done)
source "$WORKDIR/.venv/bin/activate"

# Avoid per-job memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Stream stdout/stderr to slurm log immediately (avoid block-buffered pipe)
export PYTHONUNBUFFERED=1

echo "========================================"
echo "Job ID    : ${SLURM_JOB_ID}"
echo "Node      : $(hostname)"
echo "Distance  : ${DISTANCE}"
echo "Hidden    : ${HIDDEN}, Blocks: ${BLOCKS}, Steps: ${STEPS}, Batch: ${BATCH}"
echo "Start     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

python scripts/12_train_surface_v2.py \
    --distance "$DISTANCE" \
    --steps "$STEPS" \
    --batch "$BATCH" \
    --hidden "$HIDDEN" \
    --blocks "$BLOCKS" \
    --p-train 0.005 \
    --p-warmup 0.001 \
    --eval-every 1000 \
    --final-shots 200000 \
    --tag "v2_d${DISTANCE}" \
    --p-eval 0.001 0.002 0.003 0.004 0.005 0.006 0.007

echo "========================================"
echo "End       : $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
