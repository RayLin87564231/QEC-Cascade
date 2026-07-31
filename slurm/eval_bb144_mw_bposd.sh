#!/bin/bash
# =============================================================================
# Slurm: BB-144 min-weight-basis BP+OSD BASELINE (iter-6 v4 eval prep)
#
# The paper-facing BP+OSD baseline on the NEW min-weight lz circuit
# (bb.py _min_weight_logical_basis, seed 20260705). Old BB-144 baseline
# numbers (Gaussian basis) are NOT comparable — this replaces them.
# BP+OSD depends only on the circuit, so this runs while the v4 training
# chain (165337-339) is still going.
#
# DESIGN (from smoke 166164, logs/bposd_smoke_bb144_166164.{out,err}):
#   BP+OSD on BB-144 (12 rounds) is ~22 s/shot, SINGLE-CORE (user==real).
#   A serial 5-point sweep would take days, so each p point runs as its own
#   single-core python process, all 5 in parallel in this one job. Wall is
#   bound by the p=0.002 point (2400 shots x ~22 s = ~15 h). Shot counts
#   target >=100-200 BP+OSD failures per point (p_block guesses from the
#   smoke: 0.48 @ p=0.005).
#   NOTE --shots must be >= the wanted BP+OSD shots: decoder_compare.py
#   feeds BP+OSD the FIRST min(shots, bposd-shots) sampled shots.
#
# Cascade numbers in the outputs come from a MID-TRAINING best.pt with live
# weights (EMA still warming, memory cascade-ema-warmup-gotcha) — progress
# probe only, NOT paper numbers. The paper Cascade eval is
# slurm/eval_bb144_mw.sh, after the chain reaches 40k.
#
# Output: results/bb144_mw_bposd_p{P}.json, one per point (a per-point
# process writes its JSON when it finishes, so a straggler hitting the wall
# only loses its own point).
#   sbatch slurm/eval_bb144_mw_bposd.sh
# =============================================================================

#SBATCH --job-name=eval_bb144_bposd
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --output=logs/eval_bb144_bposd_%j.out
#SBATCH --error=logs/eval_bb144_bposd_%j.err
#SBATCH --mail-type=FAIL

set -euo pipefail

WORKDIR="/work/u2467370/QEC/cascade"
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

# p_phys : shots  (shots sized for >=100-200 BP+OSD failures at ~22 s/shot)
POINTS=(
  "0.002  2400"
  "0.003  1200"
  "0.004  800"
  "0.005  600"
  "0.0055 500"
)

pids=()
for entry in "${POINTS[@]}"; do
  read -r P N <<< "$entry"
  echo "[launch] p=${P} shots=${N} -> results/bb144_mw_bposd_p${P}.json"
  python scripts/20_eval_decoder.py \
      --code bb --bb-variant 144 \
      --ckpt "$CKPT" \
      --prefer live \
      --p "$P" \
      --shots "$N" \
      --batch 128 \
      --bposd \
      --no-mwpm \
      --out "results/bb144_mw_bposd_p${P}.json" \
      > "logs/eval_bb144_bposd_${SLURM_JOB_ID}_p${P}.log" 2>&1 &
  pids+=($!)
done

fail=0
for i in "${!pids[@]}"; do
  read -r P N <<< "${POINTS[$i]}"
  if wait "${pids[$i]}"; then
    echo "[done]  p=${P} at $(date '+%H:%M:%S')"
  else
    echo "[FAIL]  p=${P} at $(date '+%H:%M:%S') — see logs/eval_bb144_bposd_${SLURM_JOB_ID}_p${P}.log"
    fail=1
  fi
done

echo "========================================"
echo "End          : $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
exit "$fail"
