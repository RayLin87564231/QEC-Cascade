#!/bin/bash
# =============================================================================
# Slurm: BB-144 min-weight-basis FINAL Cascade eval (iter-6 v4)
#
# ⚠ Submit ONLY after the v4 chain (165337 -> 165338 -> 165339) reaches 40k
#   steps AND the convergence / best.pt check passes (mind the EMA warmup
#   gotcha: EMA_DECAY=0.999, warmup 5k steps — best.pt at 40k should have a
#   warm EMA, so --prefer auto is correct; if best was saved pre-5k, rerun
#   with --prefer live).
# ⚠ Size SHOTS_CAP before submitting: measure Cascade eval throughput from
#   the baseline job log (results/bb144_mw_bposd_baseline.json run), then
#   pick a cap that fits the wall time at p=0.002.
#
# Self-sizing target-failures MC over the paper headline range. BP+OSD
# baseline comes from eval_bb144_mw_bposd.sh (separate JSON) — merge in the
# paper table. Per-head numbers are NOT comparable to v2/v3 runs (GF(2)
# basis change); block-level P_L is basis-invariant.
#   sbatch slurm/eval_bb144_mw.sh [TARGET_FAILURES=200] [SHOTS_CAP=2000000]
# =============================================================================

#SBATCH --job-name=eval_bb144_mw
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --output=logs/eval_bb144_mw_%j.out
#SBATCH --error=logs/eval_bb144_mw_%j.err
#SBATCH --mail-type=FAIL

set -euo pipefail

WORKDIR="/work/leo07010/Ray/QEC/cascade"
cd "$WORKDIR"
mkdir -p logs results

TARGET_FAILURES="${1:-200}"
SHOTS_CAP="${2:-2000000}"

CKPT="checkpoints/bb_144_12_12_v6_bb144_mw/best.pt"
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
echo "Checkpoint      : ${CKPT}"
echo "Target failures : ${TARGET_FAILURES}   Shot cap: ${SHOTS_CAP}"
echo "Start           : $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

python scripts/20_eval_decoder.py \
    --code bb --bb-variant 144 \
    --ckpt "$CKPT" \
    --prefer auto \
    --p 0.002 0.003 0.004 0.005 0.0055 \
    --shots "$SHOTS_CAP" \
    --target-failures "$TARGET_FAILURES" \
    --min-errors 100 \
    --batch 128 \
    --no-mwpm \
    --out results/bb144_mw_v4.json

echo "========================================"
echo "End             : $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
