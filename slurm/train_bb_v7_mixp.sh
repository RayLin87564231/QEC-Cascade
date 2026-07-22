#!/bin/bash
# =============================================================================
# Slurm: Cascade neural decoder — BB-144 Track 2 training, iter-7 MIXED-P
# retraining (reports/iteration_7_plan.md §3c). NCHC nano4/nano5 cluster.
#
# Single-variable change vs v6: the training noise recipe moves from a single
# p=0.0055 to a log-uniform mixed-p range [0.001, 0.0055] drawn per
# micro-batch off a 16-point cached grid (plan §2, §3a/§3b). Everything else
# (min-weight logical basis, model, optimizer, EMA, per-head reweighting,
# hyper-parameters) is held fixed so the mixed-p effect is cleanly
# attributable against v6_bb144_mw (iteration_6_status.md,
# results/bb144_mw_v4.json: flat ~0.40 p_block floor at low p — the
# generalization failure this run is meant to fix).
#
# Depends on the new CLI flags --p-min/--p-max/--p-sampling/--p-grid-points
# (plan §3b, scripts/14_train_bb_v3.py) landing separately. This launcher
# does not implement them; it only wires the SLURM invocation.
#
# TAG is v7_bb144_mixp — DELIBERATELY NEW, never v6_bb144_mw. Resume keys off
# the tag dir (14_train_bb_v3.py:189,212), so a fresh tag guarantees this
# from-scratch run cannot resume v6's last.pt and cannot touch v6's
# checkpoints/results (plan §5). Do NOT edit train_bb_v6_mw.sh in place;
# this is a new, separate file (plan §3c).
#
# Cost (plan §6): ~0.105 steps/s measured on v6 → 40k steps ≈ 106 GPU-h ≈
# 4.4 days. Submit as a 3-segment 48h afterany chain (each segment ~18k
# steps); the trainer's atomic save_last + auto-resume (14_train_bb_v3.py
# lines 191-227, 336) survive segment rollovers, same pattern as v6.
# Decision rule at 40k (plan §5): extend to a 4th segment (60k total) iff
# p_block@p_min is still trending down over the last ~5k steps.
#
# Submit (ask-first — every sbatch below is the user's explicit call; this
# plan/script submits nothing on its own):
#   J=$(sbatch --parsable slurm/train_bb_v7_mixp.sh 144)
#   J2=$(sbatch --parsable --dependency=afterany:$J slurm/train_bb_v7_mixp.sh 144)
#   sbatch --dependency=afterany:$J2 slurm/train_bb_v7_mixp.sh 144
#
# Gate: run slurm/smoke_bb_v7_mixp.sh (dev partition) first and confirm all
# §9 pass checks before submitting this chain.
# =============================================================================

#SBATCH --job-name=cascade_bb_mixp
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=47:59:00
#SBATCH --output=logs/cascade_bb_v7mixp_%j.out
#SBATCH --error=logs/cascade_bb_v7mixp_%j.err
#SBATCH --mail-type=FAIL

set -euo pipefail

WORKDIR="/work/u2467370/QEC/cascade"
cd "$WORKDIR"
mkdir -p logs

CODE="${1:-144}"

case "$CODE" in
  144)
    # Same architecture/schedule as v6_bb144_mw (train_bb_v6_mw.sh:56-67);
    # iter-7 only changes the p recipe (plan §3c), so this launcher is
    # BB-144-only — the plan does not define a mixed-p BB-72 variant.
    HIDDEN=256; BLOCKS=12; STEPS=40000; BATCH=48; ROUNDS=12
    ACCUM_STEPS=69                      # micro 48 × 69 ≈ 3312 ≈ paper batch
    P_WARMUP=0.001                      # stages 1-2 single-p curriculum warmup, unchanged (plan §3b)
    P_MIN=0.001; P_MAX=0.0055           # mixed-p range, plan §2 primary recommendation
    P_SAMPLING="log-uniform"            # plan §2: log-uniform, not uniform-in-p
    P_GRID_POINTS=16                    # cached snap grid, plan §2/§3a
    P_TRAIN="$P_MAX"                    # --p-train still names the high-p monitor eval (plan §3b)
    TAG="v7_bb${CODE}_mixp"             # NEW tag (plan §5) — must never be v6_bb144_mw
    EVAL_BATCH=256
    EMA_DECAY=0.999                     # warmup 5k = steps/8 (cascade-ema-warmup-gotcha), unchanged from v6
    P_EVAL=(0.001 0.002 0.003 0.004 0.005 0.0055)   # final sweep grid, plan §3c
    ;;
  *)
    echo "train_bb_v7_mixp.sh only supports BB-144 (iteration_7_plan.md §3c/§5); got CODE='$CODE'" >&2
    exit 1
    ;;
esac

# Reweighting knobs (see scripts/14_train_bb_v3.py): chance-level heads get up
# to CLAMPx the mean loss weight. Kept identical to v6 (plan §4: do NOT retune
# this in iter-7 — isolate the mixed-p variable).
RW_ALPHA=1.0
RW_CLAMP=4.0
RW_EMA=0.98

source "$WORKDIR/.venv/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "========================================"
echo "Job ID         : ${SLURM_JOB_ID}"
echo "Node           : $(hostname)"
echo "Code           : BB-${CODE} (rounds=${ROUNDS})"
echo "Hidden         : ${HIDDEN}, Blocks: ${BLOCKS}, Steps: ${STEPS}"
echo "Micro batch    : ${BATCH}, Accum: ${ACCUM_STEPS}, Effective: $((BATCH * ACCUM_STEPS))"
echo "p range        : [${P_MIN}, ${P_MAX}] (${P_SAMPLING}, ${P_GRID_POINTS} grid points), warmup ${P_WARMUP}"
echo "p_train (mon.) : ${P_TRAIN}"
echo "ema_decay      : ${EMA_DECAY}"
echo "head reweight  : alpha=${RW_ALPHA} clamp=${RW_CLAMP} bce_ema=${RW_EMA}"
echo "Tag            : ${TAG}"
echo "Start          : $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

python scripts/14_train_bb_v3.py \
    --code "$CODE" \
    --rounds "$ROUNDS" \
    --steps "$STEPS" \
    --batch "$BATCH" \
    --accum-steps "$ACCUM_STEPS" \
    --hidden "$HIDDEN" \
    --blocks "$BLOCKS" \
    --p-train "$P_TRAIN" \
    --p-warmup "$P_WARMUP" \
    --p-min "$P_MIN" \
    --p-max "$P_MAX" \
    --p-sampling "$P_SAMPLING" \
    --p-grid-points "$P_GRID_POINTS" \
    --ema-decay "$EMA_DECAY" \
    --head-reweight-alpha "$RW_ALPHA" \
    --head-reweight-clamp "$RW_CLAMP" \
    --head-reweight-ema "$RW_EMA" \
    --eval-every 1000 \
    --eval-batch "$EVAL_BATCH" \
    --final-shots 200000 \
    --p-eval "${P_EVAL[@]}" \
    --out "${WORKDIR}/checkpoints" \
    --tag "$TAG"

echo "========================================"
echo "End       : $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
