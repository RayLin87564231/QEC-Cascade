# Paper Review — Andi Gu et al. *Cascade: Scalable Neural Decoders* (arXiv:2604.08358)

Read date: 2026-05-04
Source: `/home/leo07010/Ray/QEC/paper/Scalable Neural Decoders for Practical Fault-Tolerant Quantum Computation.pdf` (18 pages: 8 main + 2 refs + 4 Methods + 4 Supplementary)

Goal of review: figure out why iter-2 BB-72 has 5/12 dead-head logicals (L6, L8–L11), and what the paper does differently. Specifically, is there a structural choice we missed (head architecture, auxiliary loss, etc.) or is it a hyperparameter mismatch.

---

## 1. Architecture (Methods, p.13)

### Backbone — matches our `cascade_bb.py`

Per Methods § Architecture and Extended Data Fig. 1:

> "The network backbone consists of L identically-structured processing blocks stacked sequentially, each with independent learned parameters, following a bottleneck residual design. In each block, the H-dimensional representation is first projected down to H/4 dimensions, processed by the code-specific convolution, and then projected back to H dimensions. … Each projection and convolution is preceded by batch normalization (BN), which standardizes activations to zero mean and unit variance, stabilizing training, and a SiLU activation function … A residual connection adds each block's input directly to the output."

Our `_BBBottleneckBlock` matches this: BN → SiLU → project H→H/4 → BN → SiLU → conv → BN → SiLU → project H/4→H → residual. ✅

### Code-specific convolution for BB (p.12–13)

> "For BB codes on a torus Z_ℓ × Z_m, the spatial neighborhood of each stabilizer is defined by the Tanner graph rather than a grid stencil, while the temporal direction uses a standard 1D convolution over adjacent time steps. … For BB codes, the savings are substantial: each check has 22 spatial neighbors in the check-to-check graph across 3 temporal offsets, giving 66 distinct relations per layer, while each bipartite step involves only 6 spatial neighbors across 2 temporal offsets (12 relations) — an over 5× reduction in kernel size. We implement these bipartite convolutions with custom Triton kernels …"

We factor the same way (check→data→check) via `_BBSpatialWrap`. We do not use Triton; PyTorch ops are slower but correct. This is a perf gap, not an accuracy gap. ✅ functionally equivalent.

### Head — exactly what we have

> "After the final convolutional block, a convolution scatters the check-node representations to data qubits. We then aggregate information for each logical observable by **average pooling over the data qubits in that observable's support**. The pooled representation is passed through a **two-layer multilayer perceptron (with hidden dimension 2H)** to produce a logit for each logical observable."

Our `cascade_bb.py:170–178`:

```python
self.heads = nn.ModuleList([
    nn.Sequential(
        nn.Linear(hidden, 2 * hidden),
        nn.SiLU(inplace=False),
        nn.Linear(2 * hidden, 1),
    )
    for _ in range(self.num_logicals)
])
```

Per-logical 2-layer MLP with hidden = 2H. **Exact match.** ✅ The head architecture is *not* the cause of dead logicals.

### What the paper does NOT do

Methods § Training, p.13:

> "We train with binary cross-entropy loss on the logical error prediction. For codes with multiple logical observables, we average the cross-entropy losses across all observables."

And in main text, p.3:

> "trained end-to-end with binary cross-entropy loss at a high physical error rate, requiring **no auxiliary losses, multi-stage fine-tuning, or labeled data at low noise levels**."

❌ **This rules out iter-3 option A.1** (per-qubit Z-error auxiliary loss) — paper explicitly states it is unnecessary. Our hypothesis that the heavy heads need auxiliary supervision is *inconsistent with paper's claim that pure BCE on logicals is sufficient*. So the gap must be elsewhere — in training setup, not architecture.

---

## 2. Training setup — where iter-2 actually deviates

| Knob | Paper | Iter-2 BB-72 | Delta | Plausible impact on dead heads |
|---|---|---|---|---|
| Optimizer (matrix params) | Muon, lr=3e-3 | Muon, lr=3e-3 | 0 | — |
| Optimizer (scalar/embed/readout) | Lion, lr=2e-4 | Lion, lr=2e-4 | 0 | — |
| Schedule | cosine over 50000, 1000 warmup, decay to 1/10 peak | (assumed similar) | 0 | — |
| Weight decay | 3e-3 | 3e-3 | 0 | — |
| EMA decay | 0.9998 | 0.9998 | 0 | — |
| Precision | bf16 + grad clip | bf16 + grad clip | 0 | — |
| **Batch size** | **3328** | **256** | **13× smaller** | **HIGH** (see §3) |
| **Steps** | **80000** | **40000** | **2× shorter** | medium |
| **Curriculum** | 3-stage, ~2% of total steps | **disabled** in `train_bb.sh` | removed | medium-high |
| **MuP** | yes | no | absent | low at fixed H, high if changing H |
| **Largest model** | L=14, H=512 | L=8, H=256 | smaller | medium |
| BB train p | 0.55% | 0.55% | 0 | — |
| Loss | BCE averaged over K logicals | BCE averaged over K logicals | 0 | — |

The optimizer / schedule / regularization stack is **already a 1:1 match**. The four mismatches are all in the training-data-budget axis: batch, steps, curriculum, capacity.

---

## 3. Why the dead heads, mechanistically

### The signal-to-noise problem on heavy logicals

For logical i with support weight w_i, the parity that the head must predict on a single shot is `Σ_{q ∈ supp(lz[i])} e_q (mod 2)` where e_q is the (unobserved) per-qubit Z error. Under p=0.55% over 6 rounds, the marginal flip rate per data qubit (`1 - (1 - p_eff)^R` for some effective per-cycle p_eff ≈ p) is ≈ 3–4%. For:

* w=6 (L0–L5): marginal `P(parity = 1) ≈ 0.5 - 0.5·(1-2·0.035)^6 ≈ 0.18`. **Skewed**, so a "predict 0" baseline gets 82% right and the head gets dense gradient.
* w=8 (L7): `P ≈ 0.23`. Still skewed enough for signal.
* w=14 (L6): `P ≈ 0.37`. Marginal already close to random.
* w=16 (L8): `P ≈ 0.40`.
* w=18 (L9): `P ≈ 0.42`.
* w=20 (L10–L11): `P ≈ 0.44`.

So for the dead heads the **base rate the head is trying to model is within 6–13 pp of 50%**. This is exactly the regime where a per-shot logistic regression with batch=256 sees the per-batch parity oscillate around 0.5 with a √(0.25 / 256) ≈ 3 pp standard deviation per batch — comparable to the signal we are asking the head to learn. The optimizer's averaged gradient gets pushed to 0, the head outputs 0, BCE settles at a constant.

### What 13× batch size does

With batch 3328, the per-batch standard deviation drops to √(0.25 / 3328) ≈ 0.9 pp — well below the 6–13 pp signal. The head can resolve it. **This is the most plausible single fix.**

### What curriculum does

The paper says (p.13):

> "training from random initialization directly at high p leads to prolonged periods where the network fails to learn better than random — a phenomenon reminiscent of 'grokking'. We address this with a simple three-stage curriculum (similar to the one used in [80]) that bootstraps the network from easier to harder problems."

Iter-2 explicitly disabled curriculum (`train_bb.sh:39`, `P_TRAIN=0.0055; P_WARMUP=0.0055`) because in iter-1 curriculum + the broken global-pool architecture together produced flat ln(2) BCE. After fixing the architecture in iter-2, the curriculum-off setup got 7/12 alive — but it's plausible that turning curriculum back on, with the fixed architecture and a larger batch, would also unfreeze the heavy heads (it gives them an early window where parities are sparse and signal dominates noise).

---

## 4. Implications for iter-3

The paper's training recipe is fully specified in Methods, and we are off on 3 axes that all point in the same direction (more gradient, more steps, easier early-training landscape):

### Iter-3 plan, ranked by ROI / risk

1. **(P0) Re-train BB-72 at paper's batch size and steps.** Restore curriculum. No architecture change.
   - `BATCH=3328` (was 256), `STEPS=80000` (was 40000), curriculum re-enabled (`P_WARMUP=0.001` say, `P_TRAIN=0.0055`)
   - Memory check: at H=256 L=8 batch 3328 the activations alone are roughly 13× iter-2's 256-batch run, which trained at ~75% of 80GB. 13× would OOM. Need to either (a) drop to H=128 or L=6 to fit batch 3328, or (b) gradient accumulation: micro-batch 256 × 13 accumulation steps, equivalent gradient.
   - Decision: **gradient accumulation** is the lower-risk path — keeps the architecture identical, only changes the optimizer's effective batch.
   - Risk: still doesn't unfreeze heavy heads → escalate to (2).

2. **(P1) Scale to paper's larger BB model.** L=12, H=256 first; L=14, H=512 if budget permits.
   - Per Fig 1 caption, paper's BB models span L=10–14 and H=128–512 across the 3 codes.
   - L=14 H=512 needs much more memory and time (paper says "the largest BB code model converges in under 100 [GPU hours]" on H200; we have H100 so similar).

3. **(P2) Add MuP.** Once we want to compare across H, MuP is essentially required to share hyperparameters. For a single (L, H) point it doesn't matter.

4. **(P3) Triton kernels for BB convolution.** Performance only — paper gets ~5× from custom Triton. Not on the critical path for accuracy.

### Iter-3 NOT-to-do list, based on paper

- ❌ A.1 (per-qubit auxiliary loss) — paper explicitly disclaims.
- ❌ A.3 (BPOSD warm-start as soft labels) — same reason: paper trains with pure BCE on logicals at high p and reaches all 12 logicals with no external labels. The fact that they get there with this recipe means the recipe is sufficient — we are missing a setup detail, not a supervision signal.
- ❌ Architecture redesign — head, pool, scatter, backbone all match paper. Don't fix what's not broken.

---

## 5. Source code availability

Methods § Data Availability (p.14):

> "The data that supports the findings of this study are available from the corresponding author on request."

No public repository linked. The companion paper *Neural Decoders for Universal Quantum Algorithms* (Bonilla Ataides, Gu, Yelin, Lukin, arXiv:2509.11370, 2025) by overlapping authors may have additional code/algorithm details. Worth a follow-up review if iter-3 P0 fails.

---

## 6. Hand-off to iter-3

Concrete next experiment:

```
# Paper-aligned BB-72 retrain
HIDDEN=256          # iter-2 ✓
BLOCKS=8            # iter-2 ✓ (paper uses 10–14; start with 8 to isolate batch effect)
STEPS=80000         # paper ✓ (iter-2 was 40000)
MICRO_BATCH=256     # H100 fit, iter-2 ✓
ACCUM_STEPS=13      # gives effective batch ≈ 3328 ≈ paper
P_TRAIN=0.0055      # paper ✓
P_WARMUP=0.001      # paper-style curriculum ✓ (iter-2 disabled)
```

Decision criterion: per-logical std (`scripts/24_bb_per_logical.py`) at end of training:
* If L8–L11 std > 0.5 → batch + curriculum was the fix; ship Track-2 BB.
* If still all dead → escalate to (P1) bigger model.
* If 1–2 of L8–L11 alive → directional success; consider (P1) or paper-exact 80k @ batch 3328 without accumulation (drop H to 128).
