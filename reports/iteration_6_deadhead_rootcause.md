# Iteration 6 — BB-144 dead-head root cause (heads 8-11)

**Status:** root cause identified and confirmed by probe (job 165319,
`logs/dead_probe_165319.out`).
**Question:** why do heads 8-11 die *identically* in both trainer v2 (no
reweighting, job 163114) and v3 (adaptive per-head BCE reweighting, job 164476)
— `std=0.000`, `BCE=0.693147 (=ln2)`, permanent from step ~4000 — when the
iter-3 recipe (curriculum + batch 3328) revived all 12 heads on BB-72?

---

## 1. Verdict

**The dead heads are a `lz`-weight × mean-pool-readout interaction, not a
gradient bug and not a reweighting failure.** The per-logical target is the
XOR-parity of the final data-qubit Z-measurements over the support of the
logical representative `code.lz[i]`. `_logical_basis`
(`src/cascade/codes/bb.py:278`) returns **arbitrary, high-weight** coset
representatives — BB-72 weights `[6,6,6,6,6,6,14,8,16,18,20,20]`, BB-144 weights
`[12,12,12,12,12,12,24,12,36,34,28,38]` (probe, roughly doubled). The per-logical
head then reads a **mean pool** of features over those `w` support qubits
(`cascade_bb.py:282`), and a mean pool cannot represent the high-order XOR-parity
needed to extract the observable's conditional signal once `w` is large. Heads
6, 8-11 (weights 24-38) sit above that threshold at rounds=12 and collapse to the
only thing a saturated head can output — the constant `logit=0` → `std=0`,
`BCE=ln2` exactly.

**The probe overturns the simple "labels are 50/50" reading (iter-2).** At the
training noise p=0.0055, *every* BB-144 observable — including the light heads
0-5,7 (weight 12) that train fine — has marginal flip ≈ 0.50, BCE ≈ ln2. So a
0.5 marginal is **not** the death criterion: light heads share it yet decode
because their weight-12 parity is low-enough order for the pooled head to
resolve the *conditional* signal from the syndrome. The discriminator is
**weight** (→ parity order under the mean pool), amplified by **rounds** (12 vs
6 lowers the conditional signal-to-noise). This is why the iter-3 recipe revived
BB-72 (heavy heads weight 14-20 at rounds=6 — probe shows those *were* just
learnable) but cannot revive BB-144 (heavy heads weight 24-38 at rounds=12).

**Consequence for the fix:** scaling steps / batch / model / reweighting cannot
help — the readout is structurally unable to compute these parities and the
gradient at the saturated optimum is ~0 (so v3's 1.82× up-weighting is
multiplying ~0, exactly the observed no-op; whereas head 6, still partly
signalled, does recover faster under reweighting: v3 std 3.43 vs v2 2.29 at step
7000). The fix must **lower the parity order** (low-weight representatives) or
**change the readout** (dense per-qubit target that never computes a wide
parity). Hypotheses (b) gradient-blockage and (c) normalise-away are ruled out:
heads start non-zero (std 0.1-0.9 at step 1000) and collapse to *exactly* the
chance optimum — convergence onto a saturated readout, not stuck-at-init drift.

---

## 2. Key evidence (file:line)

- `src/cascade/data/stim_dataset.py:76-78` — labels are stim
  `sample(..., separate_observables=True)`; the per-head target *is* the raw
  logical observable, nothing else.
- `src/cascade/codes/bb_circuit.py:242-248` — `OBSERVABLE_INCLUDE` for logical
  `i` = parity of final data-qubit Z measurements over `code.lz[i]` support.
  Representative-dependent: a heavier `lz[i]` = XOR of more noisy bits.
- `src/cascade/codes/bb.py:278-309` — `_logical_basis` picks coset
  representatives by greedy Gaussian elimination over `ker(hx)`; **no
  weight minimisation**, so high-index logicals accrue high weight.
- `src/cascade/models/cascade_bb.py:282` —
  `pooled = einsum("bkfn,kn->bkf", data_feat, support)/denom`: the per-logical
  head sees the **mean** of features over the `w` support qubits. A mean pool
  destroys the per-qubit sign information a `w`-way XOR-parity needs, so even a
  *predictable* high-weight parity is unrepresentable by this head. Second,
  reinforcing root cause.
- `src/cascade/models/cascade_bb.py:171-178` — heads are independent MLPs with
  bias; the final `Linear(2*hidden, 1)` has a bias, so a head *can* learn any
  base rate. It settling at `logit=0` ⇒ the base rate it is matching is 0.5.
- iter-2 §3.6 (`reports/iteration_2_status.md:170-193, 264`) — already measured
  the weight→death correlation on BB-72 and attributed it to
  "marginal parity over 14–20 noisy data qubits is essentially Bernoulli(0.5)".
- iter-3 (`reports/iteration_3_status.md:20,58,88`) — same architecture/head,
  recipe-only change, revived all 12 BB-72 heads (heavy heads std 6–7,
  err 3.7–4.4%). Proves the heavy heads are *not* unconditionally dead — they
  die only once `w·q` pushes the marginal to 0.5, which BB-144 (rounds=12,
  bigger weights) does and BB-72 (rounds=6) does not.
- `src/cascade/models/cascade_bb.py:190-232` — `per_logical_mask` is built but
  **never referenced in `forward`** (dead buffer). Not the cause; noted so v4
  doesn't chase it.

---

## 3. Probe results (job 165319, `logs/dead_probe_165319.out`)

`scripts/30_deadhead_label_probe.py`, 200k shots per (code, p).

**BB-144 (rounds=12), marginal flip rate P(obs=1):**

| head | w_lz | w_greedy | p=0.001 | p=0.003 | p=0.0055 (train) | trains? |
|------|------|----------|---------|---------|------------------|---------|
| 0-5  | 12   | 12       | 0.42-0.43 | 0.497 | 0.50 | yes |
| 6    | 24   | 24       | 0.489   | 0.500   | 0.50 | **dead** |
| 7    | 12   | 12       | 0.417   | 0.497   | 0.50 | yes |
| 8    | 36   | 32       | 0.498   | 0.499   | 0.50 | **dead** |
| 9    | 34   | 24       | 0.496   | 0.500   | 0.50 | **dead** |
| 10   | 28   | 16       | 0.493   | 0.500   | 0.50 | **dead** |
| 11   | 38   | 24       | 0.498   | 0.500   | 0.50 | **dead** |

**BB-72 (rounds=6):** weights `[6,6,6,6,6,6,14,8,16,18,20,20]`; at p=0.0055 heavy
heads 8-11 also read flip ≈ 0.499-0.500, yet iter-3 revived them — i.e. weight
14-20 at rounds=6 was still resolvable by the pooled head; weight 24-38 at
rounds=12 is not.

**What this proves:**
1. Marginal flip ≈ 0.5 at train p holds for *all* BB-144 heads, so it is **not**
   the death criterion (light heads 0-5,7 share it and train). The
   discriminator is **`lz` weight** — dead set = exactly the high-weight rows
   {6,8,9,10,11}.
2. The greedy reduction lowers the heavy weights (e.g. head 10: 28→16, head 9:
   34→24) but **cannot get all of them low** (head 8 only 36→32). So fix #1 with
   a *greedy* reducer would likely revive the lighter of the heavy heads but not
   head 8 — a stronger minimiser (ILP / IBM-reference canonical logicals) or the
   dense-target fix #2 is needed for full coverage.

---

## 4. v4 fixes, ranked by confidence

1. **Low-weight logical representatives (highest confidence, smallest change).**
   Add a weight-reduction pass to `_logical_basis` (or post-process `code.lz`):
   reduce each logical modulo the Z-stabiliser rows `hz` to minimise Hamming
   weight. Apply the **same** reduced `lz` to both the observable
   (`bb_circuit.py:244`) and the model's pooling support
   (`cascade_bb.py:153,201`). *Treats the root cause:* a lower weight shrinks the
   parity order the mean-pool must represent. Two representatives differ only by
   measured stabilisers, so this is an equivalent, valid decoding target.
   *Caveat from the probe:* the *greedy* reducer only gets BB-144 head 8 to
   weight 32 (still heavy), so use a stronger minimiser — ILP over the coset, a
   lattice/MWPM-style search, or the IBM reference's canonical minimum-weight
   logicals (target ≈ d=12). *Risk:* must keep the 12 rows independent and apply
   the identical `lz` in observable + model; verify with a probe re-run
   (rounds=12 marginal + weight) before a full training run.

2. **Auxiliary per-qubit Z-error head + deterministic logical parity (high
   confidence, larger change).** Add an `n_data`-dim per-qubit error-prediction
   head with dense BCE (labels from stim's actual error record), then compute
   the 12 logical parities deterministically from the per-qubit predictions.
   Gives every logical a dense, weight-independent gradient (standard
   surface-code decoder recipe; iter-2 proposal #1). *Treats root cause* by
   bypassing the mean-pool parity bottleneck. *Risk:* need per-qubit ground
   truth wiring, larger output, more code; heavier lift than #1.

3. **Parity-capable per-logical readout (medium confidence).** Replace the mean
   pool at `cascade_bb.py:282` with an aggregation that can represent XOR-parity
   — e.g. predict a per-support-qubit logit then combine via a differentiable
   soft-XOR / product-of-tanh. *Treats the architecture half of the cause* but
   not the ~0.5-marginal information limit, so it likely needs #1 alongside it.
   *Risk:* parity is notoriously hard to optimise; uncertain payoff.

4. **Do NOT rely on more steps / bigger model / more reweighting (rejected).**
   iter-2 already showed scaling H/blocks/steps does not break these heads, and
   v2↔v3 show reweighting is a no-op on a zero-gradient optimum. Any v4 that only
   turns these knobs will reproduce the identical death.

**Recommendation:** ship fix #1 first (cheap, root-cause, testable via the same
probe re-run on the reduced `lz`); keep #2 as the fallback if the greedy
reduction cannot get all 12 weights low enough to clear the 0.5 marginal at
rounds=12.
