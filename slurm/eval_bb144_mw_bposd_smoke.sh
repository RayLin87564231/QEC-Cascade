#!/bin/bash
# =============================================================================
# Slurm: BB-144 min-weight-basis BP+OSD SMOKE (iter-6 v4 eval prep)
#
# Two goals, both checkpoint-independent for the part that matters:
#   1. Validate scripts/20_eval_decoder.py end-to-end on BB-144 mw: ckpt
#      config resolution, BBCascadeModel build, and the NEW min-weight lz
#      circuit (bb.py _min_weight_logical_basis, seed 20260705).
#   2. Time BP+OSD per shot on BB-144 (12 rounds) to size the full baseline
#      job: run A (Cascade only) vs run B (+100 BP+OSD shots); the wall-time
#      difference isolates DEM build + BP+OSD decode cost.
#
# Cascade numbers here come from a MID-TRAINING best.pt (--prefer live: EMA
# is still warming up, see memory cascade-ema-warmup-gotcha) — progress probe
# only, NOT paper numbers. BP+OSD numbers depend only on the circuit.
#   sbatch slurm/eval_bb144_mw_bposd_smoke.sh
# =============================================================================

#SBATCH --job-name=bposd_smoke_bb144
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/bposd_smoke_bb144_%j.out
#SBATCH --error=logs/bposd_smoke_bb144_%j.err
#SBATCH --mail-type=FAIL

set -euo pipefail

WORKDIR="/work/leo07010/Ray/QEC/cascade"
cd "$WORKDIR"
mkdir -p logs results

CKPT="checkpoints/bb_144_12_12_v6_bb144_mw/best.pt"
if [ ! -f "$CKPT" ]; then
  echo "FATAL: checkpoint not found: $CKPT" >&2
  exit 1
fi

source "$WORKDIR/.venv/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "========================================"
echo "Job ID : ${SLURM_JOB_ID}"
echo "Node   : $(hostname)"
echo "Ckpt   : ${CKPT} (mid-training probe, live weights)"
echo "Start  : $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

echo "--- run A: Cascade only (2000 shots, p=0.005) ---"
time python scripts/20_eval_decoder.py \
    --code bb --bb-variant 144 \
    --ckpt "$CKPT" \
    --prefer live \
    --p 0.005 \
    --shots 2000 \
    --batch 128 \
    --no-mwpm \
    --out results/bb144_mw_smoke_cascade_only.json

echo "--- run B: Cascade + BP+OSD (100 bposd shots, p=0.005) ---"
time python scripts/20_eval_decoder.py \
    --code bb --bb-variant 144 \
    --ckpt "$CKPT" \
    --prefer live \
    --p 0.005 \
    --shots 2000 \
    --batch 128 \
    --bposd --bposd-shots 100 \
    --no-mwpm \
    --out results/bb144_mw_smoke_bposd.json

echo "========================================"
echo "End : $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
