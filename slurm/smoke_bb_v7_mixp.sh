#!/bin/bash
# =============================================================================
# Slurm: BB-144 mixed-p (iter-7) smoke test (reports/iteration_7_plan.md §9).
# Dev partition, short, FRESH start (--no-resume), separate smoke checkpoint
# dir so it never collides with the real v7_bb144_mixp chain or with the v3
# smoke (.smoke_ckpt_v3_mw). Goal: prove the mixed-p pipeline (log-uniform
# per-micro-batch p draw + cached sampler + low-p monitor eval + new best.pt
# selection metric, plan §3a/§3b) runs end-to-end on dev before spending the
# ~106 GPU-h 40k from-scratch chain. No convergence is expected in the dev
# window — this validates plumbing, not accuracy (plan §9).
#
# FIXED WORKDIR BUG: the v3 template (slurm/smoke_bb_v3_mw.sh:25, :65) points
# at the OLD, pre-handoff account's /work path (see that file for the exact
# string — not repeated here on purpose). That path is dead under the
# current account and would be a cross-account write if it somehow resolved.
# This script uses /work/u2467370/QEC/cascade throughout (WORKDIR below and
# --out).
#
# Pass checks — read from the smoke log (logs/smoke_bb_v7mixp_<jobid>.out)
# after the run, per plan §9:
#   1. runs to completion (400 steps), no crash on the min-weight basis +
#      mixed-p path.
#   2. per-batch drawn p varies across the log-spaced grid, and the sampler
#      cache holds <= --p-grid-points (16) entries. Requires the §3a/§3b
#      debug print of drawn p / cache size to have landed in
#      scripts/14_train_bb_v3.py / src/cascade/data/stim_dataset.py — this
#      script does not add that print itself, it only exercises the path.
#   3. the eval block prints p_block at BOTH p_max=0.0055 and p_min=0.001
#      (the new low-p monitor eval, plan §3b).
#   4. all 12 per-logical std > 0 (no dead-head regression).
#   5. best.pt is written using the new generalization-weighted selection
#      metric (plan §3b), not p_train-only.
#
# Gate: only launch slurm/train_bb_v7_mixp.sh (the 40k from-scratch chain)
# after ALL of the above pass. Every sbatch — this smoke, the training
# chain, and the eval — is ask-first; this script submits nothing itself.
#   sbatch slurm/smoke_bb_v7_mixp.sh
# =============================================================================

#SBATCH --job-name=smoke_bb_v7mixp
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/smoke_bb_v7mixp_%j.out
#SBATCH --error=logs/smoke_bb_v7mixp_%j.err
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
# BB-144 (rounds=12) with the min-weight basis and the mixed-p path.
TAG="v7_bb144_mixp"
SMOKE_OUT="${WORKDIR}/.smoke_ckpt_v7_mixp"

echo "========================================"
echo "Job ID : ${SLURM_JOB_ID}"
echo "Node   : $(hostname)"
echo "Smoke  : BB-144 rounds=12, min-weight basis, mixed-p, FRESH, tag=${TAG}"
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
    --p-warmup 0.001 \
    --p-min 0.001 \
    --p-max 0.0055 \
    --p-sampling log-uniform \
    --p-grid-points 16 \
    --p-train 0.0055 \
    --ema-decay 0.999 \
    --head-reweight-alpha 1.0 \
    --head-reweight-clamp 4.0 \
    --head-reweight-ema 0.98 \
    --eval-every 50 \
    --eval-batch 256 \
    --final-shots 20000 \
    --p-eval 0.001 0.0055 \
    --no-resume \
    --out "$SMOKE_OUT" \
    --tag "$TAG"

echo "========================================"
echo "End    : $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
