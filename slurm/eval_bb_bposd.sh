#!/bin/bash
# =============================================================================
# Slurm: BB-72 BP+OSD baseline eval — iter-2 final point at p=0.005
#
# The H=256/L=8 Cascade model needs --batch 512 on H100 80GB to avoid OOM
# (8192-batch einsum b,k,f,t,s → b,k,f,t,n is 24 GiB).
#
# BP+OSD (osd_order=60, osd_cs) on BB-72 is CPU-bound; cap at 5000 shots.
# =============================================================================

#SBATCH --job-name=eval_bb_bposd
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=03:59:00
#SBATCH --output=logs/eval_bb_bposd_%j.out
#SBATCH --error=logs/eval_bb_bposd_%j.err
#SBATCH --mail-type=FAIL

set -euo pipefail

WORKDIR="/work/u2467370/QEC/cascade"
cd "$WORKDIR"
mkdir -p logs results

source "$WORKDIR/.venv/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "========================================"
echo "Job ID : ${SLURM_JOB_ID}"
echo "Node   : $(hostname)"
echo "Start  : $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

python scripts/20_eval_decoder.py \
    --code bb --bb-variant 72 \
    --ckpt checkpoints/bb_72_12_6_v3_bb72/best.pt \
    --p 0.005 \
    --shots 200000 \
    --batch 512 \
    --bposd --bposd-shots 2000 \
    --no-mwpm \
    --out results/bb72_v3_iter3_bposd.json

echo "========================================"
echo "End : $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
