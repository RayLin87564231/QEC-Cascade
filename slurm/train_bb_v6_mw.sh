#!/bin/bash
# =============================================================================
# Slurm: Cascade neural decoder — BB code Track 2 training, v3 trainer on the
# iter-6 MIN-WEIGHT logical basis (fix #1). NCHC nano5 cluster.
#
# Same v3 recipe (Muon+Lion+curriculum+EMA+bf16+per-head reweighting) as
# train_bb_v3.sh; the ONLY change is that src/cascade/codes/bb.py now returns
# minimum-weight logical representatives (all BB-144 lz weight 12 = distance,
# vs the old 12..38). Root cause of the heads-8-11 death was high-weight lz ×
# mean-pool readout (reports/iteration_6_deadhead_rootcause.md); low weight
# lowers the per-head XOR-parity order so the pooled head can represent it.
# Verified by job 165329 (all weight<=12, ker/symplectic/independence PASS) and
# smoke job 165332 (all 12 heads std>0, no longer pinned at 0.000).
#
# NOTE for the fair comparison: any BP+OSD baseline MUST be evaluated on this
# SAME min-weight circuit (code.make_circuit reads the new code.lz), else the
# decoder and baseline face different observables.
#
# Submit (fresh chain — TAG v6_bb144_mw starts fresh, no old ckpt to resume):
#   J=$(sbatch --parsable slurm/train_bb_v6_mw.sh 144)
#   J2=$(sbatch --parsable --dependency=afterany:$J slurm/train_bb_v6_mw.sh 144)
#   sbatch --dependency=afterany:$J2 slurm/train_bb_v6_mw.sh 144
# =============================================================================

#SBATCH --job-name=cascade_bb_mw
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

CODE="${1:-144}"

case "$CODE" in
  72)
    # Smoke/ablation variant only — BB-72 already shipped on trainer v2/v3 recipe.
    HIDDEN=256; BLOCKS=8;  STEPS=40000; BATCH=256; ROUNDS=6
    ACCUM_STEPS=13
    P_TRAIN=0.0055; P_WARMUP=0.001
    TAG="v6_bb${CODE}_mw"
    EVAL_BATCH=4096
    EMA_DECAY=0.999
    P_EVAL=(0.001 0.002 0.003 0.004 0.005)
    ;;
  144)
    # Identical to the v6_bb144 recipe (train_bb.sh case 144) except the
    # trainer gains per-head loss reweighting and the tag/lineage is new, so
    # the run starts fresh instead of resuming the dead-head checkpoint.
    HIDDEN=256; BLOCKS=12; STEPS=40000; BATCH=48; ROUNDS=12
    ACCUM_STEPS=69                      # micro 48 × 69 ≈ 3312 ≈ paper batch
    P_TRAIN=0.0055; P_WARMUP=0.001      # 3-stage curriculum ON
    TAG="v6_bb${CODE}_mw"               # iter-6 fix #1: min-weight logical basis
    EVAL_BATCH=256
    EMA_DECAY=0.999                     # warmup 5k = steps/8 (cascade-ema-warmup-gotcha)
    P_EVAL=(0.001 0.002 0.003 0.004 0.005)
    ;;
  *)
    echo "Unknown BB code variant '$CODE'"; exit 1
    ;;
esac

# Reweighting knobs (see scripts/14_train_bb_v3.py): chance-level heads get up
# to CLAMPx the mean loss weight; alpha=0 would reproduce trainer v2 exactly.
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
echo "p_train        : ${P_TRAIN} (warmup ${P_WARMUP})"
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
    --ema-decay "$EMA_DECAY" \
    --head-reweight-alpha "$RW_ALPHA" \
    --head-reweight-clamp "$RW_CLAMP" \
    --head-reweight-ema "$RW_EMA" \
    --eval-every 1000 \
    --eval-batch "$EVAL_BATCH" \
    --final-shots 200000 \
    --p-eval ${P_EVAL[@]} \
    --out "${WORKDIR}/checkpoints" \
    --tag "$TAG"

echo "========================================"
echo "End       : $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
