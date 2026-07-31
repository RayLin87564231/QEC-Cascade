#!/bin/bash
# =============================================================================
# Slurm: Surface v5 (H=512) DEEP Lambda eval — the "measure deeper" leg.
#
# Goal: push the Lambda anchor below v4's p=0.002 toward the paper's deep
# sub-threshold regime (Lambda ~= 8.4). Self-sizing target-failures MC (stop at
# 200 Cascade failures or the shot cap). Adds a p=0.0015 intermediate anchor so
# the fit gains a deeper-than-0.002 point even if the p=0.001 monster point
# saturates its cap and is honestly flagged insufficient.
#
# Shot budget (d9, per point, ~200 block failures):
#   p=0.0015 -> ~4e7  (~2h at H=512)   p=0.001 -> ~4e8  (~19h, overnight)
# Wall raised to 24h (H=512 eval is ~2x slower than v4; no resume in eval).
#
# Submit AFTER the H=512 training completes (checkpoint present):
#   sbatch slurm/eval_surface_v5.sh 5
#   sbatch slurm/eval_surface_v5.sh 7
#   sbatch slurm/eval_surface_v5.sh 9
# =============================================================================

#SBATCH --job-name=cascade_eval_surf_v5
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --output=logs/cascade_eval_surf_v5_%j.out
#SBATCH --error=logs/cascade_eval_surf_v5_%j.err
#SBATCH --mail-type=FAIL

set -euo pipefail

WORKDIR="/work/u2467370/QEC/cascade"
cd "$WORKDIR"
mkdir -p logs results

DISTANCE="${1:-5}"
TARGET_FAILURES="${2:-200}"

# H=512, L=d+1 (match train_surface_v5.sh). Caps grow with d; the deepest p is
# the binding constraint. d9 cap 4e8 gives p=0.001 a real (if tight) chance.
case "$DISTANCE" in
  5)   BLOCKS=6;  HIDDEN=512; SHOTS_CAP=30000000  ;;   # 3e7
  7)   BLOCKS=8;  HIDDEN=512; SHOTS_CAP=100000000 ;;   # 1e8
  9)   BLOCKS=10; HIDDEN=512; SHOTS_CAP=400000000 ;;   # 4e8
  11)  BLOCKS=12; HIDDEN=512; SHOTS_CAP=600000000 ;;   # 6e8
  *)   echo "Unknown distance $DISTANCE"; exit 1 ;;
esac

CKPT="checkpoints/surface_d${DISTANCE}_v5_d${DISTANCE}/best.pt"
OUT="results/surface_d${DISTANCE}_v5.json"

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
echo "Distance        : ${DISTANCE} (blocks=${BLOCKS}, H=${HIDDEN})"
echo "Checkpoint      : ${CKPT}"
echo "Target failures : ${TARGET_FAILURES}   Shot cap: ${SHOTS_CAP}"
echo "Start           : $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

python scripts/20_eval_decoder.py \
    --code surface --distance "$DISTANCE" \
    --ckpt "$CKPT" \
    --prefer auto \
    --hidden "$HIDDEN" --blocks "$BLOCKS" \
    --p 0.001 0.0015 0.002 0.003 0.004 0.005 \
    --shots "$SHOTS_CAP" \
    --target-failures "$TARGET_FAILURES" \
    --min-errors 100 \
    --batch 8192 \
    --out "$OUT"

echo "========================================"
echo "End             : $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
