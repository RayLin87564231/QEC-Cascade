#!/bin/bash
# =============================================================================
# Slurm: Surface code retrain at H=256 (iter-3, paper Λ-scaling).
#
# Iter-2 surface ckpts trained at H=128 → Λ ≈ 1.86. Paper Fig 4 shows m
# saturates at H ≥ 64 for surface, but circuit-level Λ grows substantially
# from H=128 to H=512 (paper's Λ ≈ 8.4 uses L=14, H=512). H=256 is the
# next step on the curve.
#
# Submit:
#   sbatch slurm/train_surface_v3.sh 5
#   sbatch slurm/train_surface_v3.sh 7
#   sbatch slurm/train_surface_v3.sh 9
#   sbatch slurm/train_surface_v3.sh 11   # new 4th distance for Λ fit
# =============================================================================

#SBATCH --job-name=cascade_surf_v3
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --output=logs/cascade_surf_v3_%j.out
#SBATCH --error=logs/cascade_surf_v3_%j.err
#SBATCH --mail-type=FAIL

set -euo pipefail

WORKDIR="/work/u2467370/QEC/cascade"
cd "$WORKDIR"
mkdir -p logs

DISTANCE="${1:-5}"

# H=256 — 4× the iter-2 hidden width, so micro-batch halved from iter-2 to
# stay under 80 GB. L matches iter-2 (kept fixed to isolate the H effect).
case "$DISTANCE" in
  5)   HIDDEN=256; BLOCKS=6;  STEPS=40000; BATCH=256 ;;
  7)   HIDDEN=256; BLOCKS=8;  STEPS=60000; BATCH=192 ;;
  9)   HIDDEN=256; BLOCKS=10; STEPS=80000; BATCH=128 ;;
  11)  HIDDEN=256; BLOCKS=12; STEPS=80000; BATCH=96  ;;
  *)   echo "Unknown distance $DISTANCE"; exit 1 ;;
esac

source "$WORKDIR/.venv/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
    --tag "v3_d${DISTANCE}" \
    --p-eval 0.001 0.002 0.003 0.004 0.005 0.006 0.007

echo "========================================"
echo "End       : $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
