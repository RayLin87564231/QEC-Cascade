#!/bin/bash
# =============================================================================
# Slurm: Surface v4 deep-sub-threshold Lambda eval (Phase A3).
#
# Target-failures Monte-Carlo: for each p, sample until Cascade reaches
# TARGET_FAILURES failures or the per-distance shot cap is hit (self-sizing,
# see eval/decoder_compare.py). Cascade + MWPM decode the SAME shots. BP+OSD is
# NOT used for surface. The p=0.002 point is the paper-matched Lambda anchor;
# the full p curve feeds the waterfall fit (28_waterfall_fit.py).
#
# Per-distance shot caps are sized so the p=0.002 anchor clears >=200 Cascade
# failures for a clean CI; the deepest p (0.001) may saturate the cap and be
# auto-flagged `sufficient:false` (excluded from the Lambda fit) -- that is
# honest, not a bug.
#
# Submit AFTER training completes and the v4>v2 gate passes (QOS caps 2):
#   sbatch slurm/eval_surface_v4.sh 5
#   sbatch slurm/eval_surface_v4.sh 7
#   sbatch slurm/eval_surface_v4.sh 9
#   sbatch slurm/eval_surface_v4.sh 11
# =============================================================================

#SBATCH --job-name=cascade_eval_surf_v4
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=07:59:00
#SBATCH --output=logs/cascade_eval_surf_v4_%j.out
#SBATCH --error=logs/cascade_eval_surf_v4_%j.err
#SBATCH --mail-type=FAIL

set -euo pipefail

WORKDIR="/work/leo07010/Ray/QEC/cascade"
cd "$WORKDIR"
mkdir -p logs results

DISTANCE="${1:-5}"
TARGET_FAILURES="${2:-200}"

# Blocks match train_surface_v4.sh (H=256, L=d+1). Shot caps grow with d as
# the logical error rate drops ~10x per +2 distance sub-threshold.
case "$DISTANCE" in
  5)   BLOCKS=6;  SHOTS_CAP=3000000  ;;   # 3e6
  7)   BLOCKS=8;  SHOTS_CAP=15000000 ;;   # 1.5e7
  9)   BLOCKS=10; SHOTS_CAP=40000000 ;;   # 4e7
  11)  BLOCKS=12; SHOTS_CAP=60000000 ;;   # 6e7
  *)   echo "Unknown distance $DISTANCE"; exit 1 ;;
esac

CKPT="checkpoints/surface_d${DISTANCE}_v4_d${DISTANCE}/best.pt"
OUT="results/surface_d${DISTANCE}_v4.json"

if [ ! -f "$CKPT" ]; then
  echo "FATAL: checkpoint not found: $CKPT" >&2
  exit 1
fi

source "$WORKDIR/.venv/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "========================================"
echo "Job ID          : ${SLURM_JOB_ID}"
echo "Node            : $(hostname)"
echo "Distance        : ${DISTANCE} (blocks=${BLOCKS}, H=256)"
echo "Checkpoint      : ${CKPT}"
echo "Target failures : ${TARGET_FAILURES}   Shot cap: ${SHOTS_CAP}"
echo "Start           : $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

python scripts/20_eval_decoder.py \
    --code surface --distance "$DISTANCE" \
    --ckpt "$CKPT" \
    --prefer auto \
    --hidden 256 --blocks "$BLOCKS" \
    --p 0.001 0.002 0.003 0.004 0.005 \
    --shots "$SHOTS_CAP" \
    --target-failures "$TARGET_FAILURES" \
    --min-errors 100 \
    --batch 8192 \
    --out "$OUT"

echo "========================================"
echo "End             : $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
