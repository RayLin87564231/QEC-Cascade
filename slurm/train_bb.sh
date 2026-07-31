#!/bin/bash
# =============================================================================
# Slurm: Cascade neural decoder — BB code Track 2 training
# NCHC nano5 cluster
#
# Submit:
#   sbatch slurm/train_bb.sh 72   # [[72,12,6]]
#   sbatch slurm/train_bb.sh 144  # [[144,12,12]] Gross code
#
# BB-144 exceeds the 48h wall (~67-100h at effective batch 3328). The trainer
# now writes last.pt every eval and auto-resumes, so chain allocations:
#   J=$(sbatch --parsable slurm/train_bb.sh 144)
#   sbatch --dependency=afterany:$J slurm/train_bb.sh 144   # resumes last.pt
# =============================================================================

#SBATCH --job-name=cascade_bb
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=47:59:00
#SBATCH --output=logs/cascade_bb_%j.out
#SBATCH --error=logs/cascade_bb_%j.err
#SBATCH --mail-type=FAIL

set -euo pipefail

WORKDIR="/work/u2467370/QEC/cascade"
cd "$WORKDIR"
mkdir -p logs

CODE="${1:-72}"

# Code-dependent hyperparameters
# Iteration-3 (BB-72): paper-aligned recipe — gradient accumulation to reach
# paper batch=3328, curriculum re-enabled (3-stage low→target). See
# reports/paper_review.md and ~/.claude/plans/concurrent-zooming-papert.md.
case "$CODE" in
  72)
    HIDDEN=256; BLOCKS=8;  STEPS=40000; BATCH=256; ROUNDS=6
    ACCUM_STEPS=13                      # micro 256 × 13 ≈ paper batch 3328
    P_TRAIN=0.0055; P_WARMUP=0.001      # paper-style 3-stage curriculum
    TAG="v3_bb${CODE}"                  # iter-3; preserves iter-2 ckpt
    EVAL_BATCH=4096
    EMA_DECAY=0.9998                    # legacy value; matches shipped v3 ckpt
    P_EVAL=(0.001 0.002 0.003 0.004 0.005)
    ;;
  144)
    # Fixed from the parked iter-2 recipe: curriculum ON + gradient accumulation
    # to the paper batch. n=144/rounds=12 make per-sample cost ~3.7-5.6× BB-72,
    # so micro-batch is dropped to 48 (256 OOMs at 80 GB) and ACCUM raised to
    # reach ~3328. EVAL_BATCH=256 avoids the scatter-einsum OOM (24 GiB @ 8192
    # for BB-72; ~3.7× worse here). If the first eval OOMs, drop BATCH to 32.
    HIDDEN=256; BLOCKS=12; STEPS=40000; BATCH=48; ROUNDS=12
    ACCUM_STEPS=69                      # micro 48 × 69 ≈ 3312 ≈ paper batch
    P_TRAIN=0.0055; P_WARMUP=0.001      # curriculum ON (was p1==p2 → off)
    TAG="v6_bb${CODE}"                  # iter-6; new lineage
    EVAL_BATCH=256
    EMA_DECAY=0.999                     # warmup 5k = steps/8; 0.9998 needs 25k
                                        # (62% of run) and cost iter-5 ~0.65 Λ
    P_EVAL=(0.001 0.002 0.003 0.004 0.005)
    ;;
  *)
    echo "Unknown BB code variant '$CODE'"; exit 1
    ;;
esac

source "$WORKDIR/.venv/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "========================================"
echo "Job ID         : ${SLURM_JOB_ID}"
echo "Node           : $(hostname)"
echo "Code           : BB-${CODE} (rounds=${ROUNDS})"
echo "Hidden         : ${HIDDEN}, Blocks: ${BLOCKS}, Steps: ${STEPS}"
echo "Micro batch    : ${BATCH}, Accum: ${ACCUM_STEPS}, Effective: $((BATCH * ACCUM_STEPS))"
echo "p_train        : ${P_TRAIN} (warmup ${P_WARMUP})"
echo "ema_decay      : ${EMA_DECAY}"
echo "Tag            : ${TAG}"
echo "Start          : $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

python scripts/14_train_bb_v2.py \
    --code "$CODE" \
    --rounds "$ROUNDS" \
    --steps "$STEPS" \
    --batch "$BATCH" \
    --accum-steps "$ACCUM_STEPS" \
    --hidden "$HIDDEN" \
    --blocks "$BLOCKS" \
    --p-train "$P_TRAIN" \
    --p-warmup "$P_WARMUP" \
    --ema-decay "$EMA_DECAY" \
    --eval-every 1000 \
    --eval-batch "$EVAL_BATCH" \
    --final-shots 200000 \
    --p-eval ${P_EVAL[@]} \
    --tag "$TAG"

echo "========================================"
echo "End       : $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
