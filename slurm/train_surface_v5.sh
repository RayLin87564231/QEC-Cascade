#!/bin/bash
# =============================================================================
# Slurm: Surface code retrain at H=512 (paper headline capacity) with the same
# paper effective batch (~3328) as v4. This is the "scale up the model" leg of
# the push toward the paper Lambda ~= 8.4. v4 (H=256) reproduced the claims but
# plateaued at Lambda_Cascade = 6.85 @ p=0.002; H=512 tests whether more model
# capacity lifts the waterfall curve.
#
# H=512 conv FLOPs are ~4x H=256, so micro-batch is halved (memory) and accum
# doubled to hold effective batch ~= 3328. Wall time grows accordingly; STEPS is
# overridable ($2) so d9 can be trimmed to fit the 24h wall (trainer has no
# resume yet).
#
# Submit (GPU is wide open; QOS caps 2 concurrent):
#   sbatch slurm/train_surface_v5.sh 5            # 40k default
#   sbatch slurm/train_surface_v5.sh 7
#   sbatch slurm/train_surface_v5.sh 9 30000      # trim d9 if smoke says >24h
#   sbatch slurm/train_surface_v5.sh 9 300        # smoke: measure steps/s + OOM
# =============================================================================

#SBATCH --job-name=cascade_surf_v5
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --output=logs/cascade_surf_v5_%j.out
#SBATCH --error=logs/cascade_surf_v5_%j.err
#SBATCH --mail-type=FAIL

set -euo pipefail

WORKDIR="/work/leo07010/Ray/QEC/cascade"
cd "$WORKDIR"
mkdir -p logs

DISTANCE="${1:-5}"
STEPS_OVERRIDE="${2:-}"

# H=512, L=d+1 (same L as v4 to isolate the capacity variable). Micro-batch
# halved vs v4; ACCUM doubled so micro x accum ~= 3328 (paper batch).
case "$DISTANCE" in
  5)   HIDDEN=512; BLOCKS=6;  STEPS=40000; BATCH=128; ACCUM=26 ;;  # 3328
  7)   HIDDEN=512; BLOCKS=8;  STEPS=40000; BATCH=96;  ACCUM=35 ;;  # 3360
  9)   HIDDEN=512; BLOCKS=10; STEPS=40000; BATCH=64;  ACCUM=52 ;;  # 3328
  11)  HIDDEN=512; BLOCKS=12; STEPS=30000; BATCH=48;  ACCUM=70 ;;  # 3360
  *)   echo "Unknown distance $DISTANCE"; exit 1 ;;
esac
[ -n "$STEPS_OVERRIDE" ] && STEPS="$STEPS_OVERRIDE"
EFF=$(( BATCH * ACCUM ))

source "$WORKDIR/.venv/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "========================================"
echo "Job ID     : ${SLURM_JOB_ID}"
echo "Node       : $(hostname)"
echo "Distance   : ${DISTANCE}"
echo "Hidden     : ${HIDDEN}, Blocks: ${BLOCKS}, Steps: ${STEPS}"
echo "Micro-batch: ${BATCH}, Accum: ${ACCUM}, Effective batch: ${EFF}"
echo "Start      : $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

python scripts/12_train_surface_v2.py \
    --distance "$DISTANCE" \
    --steps "$STEPS" \
    --batch "$BATCH" \
    --accum-steps "$ACCUM" \
    --hidden "$HIDDEN" \
    --blocks "$BLOCKS" \
    --p-train 0.005 \
    --p-warmup 0.001 \
    --eval-every 1000 \
    --final-shots 200000 \
    --tag "v5_d${DISTANCE}" \
    --p-eval 0.001 0.002 0.003 0.004 0.005 0.006 0.007

echo "========================================"
echo "End        : $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
