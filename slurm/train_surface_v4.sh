#!/bin/bash
# =============================================================================
# Slurm: Surface code retrain at H=256 with PAPER EFFECTIVE BATCH (~3328).
#
# Root cause of the flat Λ≈1.86 (v2, H=128) and the WORSE v3 (H=256): the
# surface runs never used the paper's large batch. v3 even *halved* the batch
# to fit H=256 and kept steps fixed → undertrained (regressed below v2). This
# is the same small-batch bug already fixed for BB-72 via gradient
# accumulation to effective batch ≈3328. v4 ports that fix to surface:
#   effective_batch = micro-batch × accum ≈ 3328   (micro sized to fit 80 GB)
#
# Steps kept at 40k (30k for d11): with ~13–26× the effective batch, each
# opt-step is a far cleaner gradient than v3's, and 40k×3328 ≈ 133M samples is
# ~13× the *total* data v3 saw — while every job stays well under the 24h wall
# (surface trainer has no resume yet, so we do not risk the wall).
#
# Submit (QOS caps 2 concurrent):
#   sbatch slurm/train_surface_v4.sh 5
#   sbatch slurm/train_surface_v4.sh 7
#   sbatch slurm/train_surface_v4.sh 9
#   sbatch slurm/train_surface_v4.sh 11   # optional 4th distance for Λ fit
# =============================================================================

#SBATCH --job-name=cascade_surf_v4
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --output=logs/cascade_surf_v4_%j.out
#SBATCH --error=logs/cascade_surf_v4_%j.err
#SBATCH --mail-type=FAIL

set -euo pipefail

WORKDIR="/work/leo07010/Ray/QEC/cascade"
cd "$WORKDIR"
mkdir -p logs

DISTANCE="${1:-5}"

# H=256, L=d+1 (as v3). Micro-batch matches v3's known-fitting sizes; ACCUM
# chosen so micro×accum ≈ 3328 (paper batch). L=14/H=512 headline is deferred.
case "$DISTANCE" in
  5)   HIDDEN=256; BLOCKS=6;  STEPS=40000; BATCH=256; ACCUM=13 ;;  # 3328
  7)   HIDDEN=256; BLOCKS=8;  STEPS=40000; BATCH=192; ACCUM=17 ;;  # 3264
  9)   HIDDEN=256; BLOCKS=10; STEPS=40000; BATCH=128; ACCUM=26 ;;  # 3328
  11)  HIDDEN=256; BLOCKS=12; STEPS=30000; BATCH=96;  ACCUM=35 ;;  # 3360
  *)   echo "Unknown distance $DISTANCE"; exit 1 ;;
esac
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
    --tag "v4_d${DISTANCE}" \
    --p-eval 0.001 0.002 0.003 0.004 0.005 0.006 0.007

echo "========================================"
echo "End        : $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
