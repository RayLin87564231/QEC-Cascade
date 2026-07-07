#!/bin/bash
# =============================================================================
# Slurm: BB-144 min-weight-basis BP+OSD baseline TOP-UP for p=0.003
#
# The main baseline (job 166180, slurm/eval_bb144_mw_bposd.sh) left p=0.003
# with only 47 BP+OSD failures / 1200 shots (rate ~3.9%) — under the ~100
# failures wanted for a tight CI. BP+OSD sampling is NOT fixed-seed across
# invocations (verified), so these extra shots are statistically independent
# of the original and MERGE by summing (n_failures, n_total) per point, then
# recomputing P_L/CI with cascade.eval.pblock_to_pl.per_cycle_pl_with_ci.
#
# Adds 5 workers x 480 = 2400 extra shots at p=0.003 (expected ~+90 failures
# -> ~130 total, comfortably >=100). Each worker is a single-core BP+OSD
# process (~22 s/shot) writing its OWN JSON so the existing p=0.003 result is
# never overwritten. Wall bound: 480 x ~22 s ~= 2.9 h.
#
# The Cascade column in these JSONs is a mid-training probe (best.pt is the
# still-training v4 checkpoint) — IGNORE it. Only the BP+OSD column matters;
# BP+OSD is classical and depends only on the (min-weight-basis) circuit.
#
#   sbatch slurm/eval_bb144_mw_bposd_topup_p003.sh
# Merge afterwards with scripts/merge_bposd_points.py (see that file).
# =============================================================================

#SBATCH --job-name=bposd_topup_p003
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:30:00
#SBATCH --output=logs/eval_bb144_bposd_topup_p003_%j.out
#SBATCH --error=logs/eval_bb144_bposd_topup_p003_%j.err
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
export OMP_NUM_THREADS=1

echo "========================================"
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : $(hostname)"
echo "Ckpt (probe) : ${CKPT}"
echo "Start        : $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

P=0.003
NWORKERS=5
SHOTS_PER=480

pids=()
for w in $(seq 0 $((NWORKERS - 1))); do
  OUT="results/bb144_mw_bposd_p${P}_topup_w${w}.json"
  echo "[launch] worker=${w} p=${P} shots=${SHOTS_PER} -> ${OUT}"
  python scripts/20_eval_decoder.py \
      --code bb --bb-variant 144 \
      --ckpt "$CKPT" \
      --prefer live \
      --p "$P" \
      --shots "$SHOTS_PER" \
      --batch 128 \
      --bposd \
      --no-mwpm \
      --out "$OUT" \
      > "logs/eval_bb144_bposd_topup_p003_${SLURM_JOB_ID}_w${w}.log" 2>&1 &
  pids+=($!)
done

fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[done]  worker=${i} at $(date '+%H:%M:%S')"
  else
    echo "[FAIL]  worker=${i} at $(date '+%H:%M:%S') — see logs/eval_bb144_bposd_topup_p003_${SLURM_JOB_ID}_w${i}.log"
    fail=1
  fi
done

echo "========================================"
echo "End          : $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
exit "$fail"
