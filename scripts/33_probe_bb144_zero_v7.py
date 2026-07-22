"""Zero-syndrome probe for BB-144 iter-7 (v7_bb144_mixp) — closes acceptance criterion 5.

v7 analogue of probe_bb144_zero_syndrome.py (v6 diagnosis, job 185295).
Loads checkpoints/bb_144_12_12_v7_bb144_mixp/best.pt the way
scripts/20_eval_decoder.py does (--prefer auto -> EMA weights), then:

  1. Runs a batch of 64 ALL-ZERO-syndrome inputs (the p->0 limit) through
     the model and reports per-head sigmoid outputs / flip fraction.
     Criterion 5 requires 0/12 heads firing.
  2. Samples 64 shots at p=0.0005 via the exact eval code path and decodes.
     v6 measured p_block=0.42 here (the generalization failure); v7 is
     expected near 0.
  3. Sampler determinism check (unchanged from v6 probe): also relevant to
     the 2026-07-22 finding that results/bb144_mw_v4.json has byte-identical
     p=0.002 and p=0.003 entries.

Read-only wrt repo/checkpoints (load only, CPU, <=256 samples total).
Run via SLURM dev partition — never the login node.
"""

import sys

sys.path.insert(0, "/work/u2467370/QEC/cascade/src")

import numpy as np
import torch

from cascade.codes.bb import BBCode
from cascade.data.tensorize import make_grid_indexer, detection_events_to_grid
from cascade.models.cascade_bb import BBCascadeModel

CKPT_PATH = "/work/u2467370/QEC/cascade/checkpoints/bb_144_12_12_v7_bb144_mixp/best.pt"

device = torch.device("cpu")

print("=== loading checkpoint (CPU) ===")
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
cfg = ckpt["config"]
hidden = cfg["hidden"]
blocks = cfg["blocks"]
p_keys = {k: cfg[k] for k in ("p_train", "p_min", "p_max", "p_sampling") if k in cfg}
print(f"hidden={hidden} blocks={blocks} rounds={cfg.get('rounds')} p-config={p_keys}")
print(f"ckpt step={ckpt.get('step')} best_metric={ckpt.get('best_p_block', ckpt.get('best_metric'))}")

code = BBCode.code_144_12_12()
rounds = 12
layout = code.detector_layout(rounds=rounds)
flat_idx, grid_shape = make_grid_indexer(layout)
model = BBCascadeModel(code=code, layout=layout, hidden=hidden, num_blocks=blocks)

# Replicate scripts/20_eval_decoder.py::_load_ckpt_into with prefer="auto"
has_ema = ckpt.get("model_ema") is not None
prefer = "auto"
if prefer == "ema" or (prefer == "auto" and has_ema):
    assert has_ema
    missing = model.load_state_dict(ckpt["model_ema"])
    src = "ema"
else:
    missing = model.load_state_dict(ckpt["model"])
    src = "live"
print(f"loaded weights: {src}  load_state_dict result: {missing}")
model.to(device).eval()

T, ell, m, C = [int(v) for v in grid_shape.tolist()]
print(f"grid_shape=(T={T}, ell={ell}, m={m}, C={C})  num_logicals={code.num_logical_observables}")

# ---------------------------------------------------------------------
# 1. Zero-syndrome probe (p -> 0 limit): all detectors silent.
# ---------------------------------------------------------------------
print("\n=== 1. zero-syndrome probe (B=64, all detectors=0) ===")
B = 64
zero_grid = torch.zeros(B, T, ell, m, C, dtype=torch.float32)
with torch.inference_mode():
    logits0 = model(zero_grid)
sig0 = torch.sigmoid(logits0)
# All 64 rows are identical (deterministic model, identical input) -- report row 0.
row = sig0[0]
flips = (row > 0.5)
print("per-head sigmoid (zero syndrome, one representative row):")
print(np.array2string(row.numpy(), precision=4, suppress_small=True))
print(f"heads with sigmoid>0.5 on ALL-ZERO syndrome: {int(flips.sum())} / {code.num_logical_observables}")
print(f"all 64 rows identical: {bool(torch.allclose(sig0, sig0[0:1].expand_as(sig0)))}")
any_flip_per_shot = flips.any().item()
print(f"any head flips on zero syndrome (should be False for a correct decoder): {any_flip_per_shot}")

# ---------------------------------------------------------------------
# 2. Real sampling probe at p=0.0005 via the exact eval code path.
# ---------------------------------------------------------------------
print("\n=== 2. p=0.0005, 64 shots, via evaluate_decoders' own sampler path ===")
from cascade.eval.decoder_compare import evaluate_decoders  # noqa: E402

results = evaluate_decoders(
    code=code, rounds=rounds, p=0.0005, n_shots=64,
    model=model, flat_idx=flat_idx, grid_shape=grid_shape,
    device=device, batch=64, include_mwpm=False, include_bposd=False,
)
r = results["Cascade"]
print(f"Cascade @ p=0.0005: n_failures={r.n_failures} n_total={r.n_total} p_block={r.p_block:.5f}")
print("(v6 at this point measured p_block=0.42; v7 expected near 0)")

# ---------------------------------------------------------------------
# 3. Determinism check of circuit.compile_detector_sampler() with NO seed
#    (also relevant to the byte-identical p=0.002/0.003 entries found in
#    results/bb144_mw_v4.json on 2026-07-22).
# ---------------------------------------------------------------------
print("\n=== 3. sampler determinism check (no explicit seed) ===")
c2 = code.make_circuit(p=0.002, rounds=rounds)
c3 = code.make_circuit(p=0.003, rounds=rounds)
s2a = c2.compile_detector_sampler()
s2b = c2.compile_detector_sampler()  # second sampler, SAME circuit (p=0.002)
s3 = c3.compile_detector_sampler()   # different circuit (p=0.003)

det2a, obs2a = s2a.sample(shots=256, separate_observables=True, bit_packed=False)
det2b, obs2b = s2b.sample(shots=256, separate_observables=True, bit_packed=False)
det3, obs3 = s3.sample(shots=256, separate_observables=True, bit_packed=False)

print(f"two fresh samplers, SAME circuit (p=0.002), identical detection events?  "
      f"{bool(np.array_equal(det2a, det2b))}")
print(f"two fresh samplers, SAME circuit (p=0.002), identical observables?      "
      f"{bool(np.array_equal(obs2a, obs2b))}")
print(f"samplers from DIFFERENT circuits (p=0.002 vs p=0.003), identical det?   "
      f"{bool(np.array_equal(det2a, det3))}")
print(f"det2a mean click rate: {det2a.mean():.5f}   det3 mean click rate: {det3.mean():.5f}")

print("\n=== done ===")
