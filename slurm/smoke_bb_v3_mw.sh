#!/bin/bash
# =============================================================================
# Slurm: BB-144 smoke on the min-weight logical basis (iter-6 fix #1), v3
# trainer. Dev partition, short, FRESH start (--no-resume), separate smoke
# checkpoint dir so it never collides with the real chain. Goal: confirm the
# pipeline trains without crashing on the new basis and that eval prints
# per-logical std (heads may not fully wake within dev time — that is fine).
#   sbatch slurm/smoke_bb_v3_mw.sh
# =============================================================================

#SBATCH --job-name=smoke_bb_mw
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/smoke_bb_mw_%j.out
#SBATCH --error=logs/smoke_bb_mw_%j.err
#SBATCH --mail-type=FAIL

set -euo pipefail

WORKDIR="/work/u2467370/QEC/cascade"
cd "$WORKDIR"
mkdir -p logs

source "$WORKDIR/.venv/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# Reduced config vs the full chain (HIDDEN=256 BLOCKS=12 STEPS=40000): small
# enough to reach several evals inside the dev window while still exercising
# BB-144 (rounds=12) with the new min-weight basis.
TAG="v6_bb144_mw"

echo "========================================"
echo "Job ID : ${SLURM_JOB_ID}"
echo "Node   : $(hostname)"
echo "Smoke  : BB-144 rounds=12, min-weight basis, FRESH, tag=${TAG}"
echo "Start  : $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

python scripts/14_train_bb_v3.py \
    --code 144 \
    --rounds 12 \
    --steps 400 \
    --batch 48 \
    --accum-steps 4 \
    --hidden 128 \
    --blocks 6 \
    --p-train 0.0055 \
    --p-warmup 0.001 \
    --ema-decay 0.999 \
    --head-reweight-alpha 1.0 \
    --head-reweight-clamp 4.0 \
    --head-reweight-ema 0.98 \
    --eval-every 50 \
    --eval-batch 256 \
    --final-shots 20000 \
    --p-eval 0.003 \
    --no-resume \
    --out "${WORKDIR}/.smoke_ckpt_v3_mw" \
    --tag "$TAG"

echo "========================================"
echo "End    : $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
