#!/bin/bash
# =============================================================================
# Slurm: BB-144 mixed-p (iter-7) FINAL Cascade eval (reports/iteration_7_plan.md
# §3d).
#
# ⚠ Submit ONLY after the v7_bb144_mixp 40k (or extended 60k, plan §5 decision
#   rule) training chain finishes AND best.pt looks converged (mind the EMA
#   warmup gotcha: EMA_DECAY=0.999, warmup 5k steps — --prefer auto is
#   correct as long as best.pt was saved past the EMA warmup window).
#
# scripts/20_eval_decoder.py needs NO code change (plan §3d) — it already
# sweeps --p and prefers EMA (--prefer auto). This script only repoints the
# checkpoint at the new v7_bb144_mixp tag dir and writes a separate results
# file so v6's results/bb144_mw_v4.json is never touched.
#
# SHOTS_CAP / TARGET_FAILURES sizing is REUSED unchanged from
# eval_bb144_mw.sh (plan §6: same sizing, cap 1e7 → worst case p=0.002 runs
# in ~15.4h, other points hit target-failures early, total ≲17h < 24h wall).
#
# BP+OSD baseline is REUSED unchanged from iter-6 (results/bb144_mw_bposd_p*.json)
# — same min-weight circuit, so the comparison stays fair (plan §3d). No new
# baseline eval is run here.
#
#   sbatch slurm/eval_bb144_v7_mixp.sh [TARGET_FAILURES=200] [SHOTS_CAP=10000000]
# =============================================================================

#SBATCH --job-name=eval_bb144_v7mixp
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --output=logs/eval_bb144_v7mixp_%j.out
#SBATCH --error=logs/eval_bb144_v7mixp_%j.err
#SBATCH --mail-type=FAIL

set -euo pipefail

WORKDIR="/work/u2467370/QEC/cascade"
cd "$WORKDIR"
mkdir -p logs results

TARGET_FAILURES="${1:-200}"
SHOTS_CAP="${2:-10000000}"

CKPT="checkpoints/bb_144_12_12_v7_bb144_mixp/best.pt"
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
    --out results/bb144_v7_mixp.json

echo "========================================"
echo "End             : $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
