"""Dead-head root-cause probe: per-logical label flip rate vs lz weight.

Iter-6 investigation (reports/iteration_6_deadhead_rootcause.md).

Hypothesis under test: heads 8-11 (and 6) die because their `code.lz`
representatives are HIGH WEIGHT. The Z-memory observable for logical i is the
XOR-parity of the final data-qubit Z measurements over lz[i]'s support. For a
weight-w representative, that parity is a XOR of ~w independently-noisy bits, so
by the piling-up lemma its marginal flip rate -> 0.5 as w * q grows (q =
effective per-qubit flip prob, which grows with `rounds`). A ~0.5 marginal with
no extractable signal in the mean-pooled per-logical feature pins the head at
BCE = ln2, std = 0 (Bayes-optimal constant = logit 0).

This script measures, for BB-72 (rounds=6, revived in iter-3) and BB-144
(rounds=12, dead in iter-6), and prints per-observable:
  * lz row weight (current _logical_basis representative)
  * a greedily weight-reduced representative weight (upper bound on min weight;
    demonstrates a lighter, equivalent target exists -> fix #1 feasibility)
  * marginal flip rate P(obs_i = 1) at p in {0.001, 0.003, 0.0055}
  * per-observable BCE floor H2(flip_rate) = entropy of a constant predictor
    (ln2 = 0.6931 <=> flip rate 0.5 <=> dead head)

No training, no gradients: pure stim sampling + numpy. Runs in seconds.
"""

from __future__ import annotations

import numpy as np

from cascade.codes.bb import BBCode


def h2_nats(pflip: float) -> float:
    """Binary entropy in nats (== per-observable BCE floor of a constant head)."""
    p = min(max(pflip, 1e-12), 1.0 - 1e-12)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def greedy_reduce_weight(lz_rows: np.ndarray, hz: np.ndarray) -> np.ndarray:
    """Greedily XOR Z-stabiliser rows into each logical to reduce Hamming weight.

    Returns the reduced-weight per-logical vector (upper bound on the minimum
    weight over the coset lz[i] + rowspace(hz)). This is NOT a full minimiser;
    it just shows a substantially lighter, equivalent representative exists.
    """
    hz = hz % 2
    out = []
    for i in range(lz_rows.shape[0]):
        v = lz_rows[i].copy() % 2
        improved = True
        while improved:
            improved = False
            w = int(v.sum())
            for s in hz:
                cand = (v + s) % 2
                if int(cand.sum()) < w:
                    v = cand
                    w = int(cand.sum())
                    improved = True
        out.append(int(v.sum()))
    return np.array(out, dtype=int)


def probe_code(code: BBCode, rounds: int, ps, n_shots: int, seed: int = 0) -> None:
    k = code.num_logical_observables
    lz = code.lz.astype(np.int64)
    hz = code.hz.astype(np.int64)
    weights = lz.sum(axis=1).astype(int)
    reduced = greedy_reduce_weight(lz, hz)

    print(f"\n{'='*78}")
    print(f"{code.name}  (k={k}, n_data={code.num_data_qubits}, rounds={rounds})")
    print(f"{'='*78}")
    print(f"lz row weights (current _logical_basis reps): {weights.tolist()}")
    print(f"greedy-reduced weights (feasible lighter reps): {reduced.tolist()}")

    # Sample marginal flip rates at each p.
    flip = {}
    for p in ps:
        sampler = code.make_circuit(p=p, rounds=rounds).compile_detector_sampler(seed=seed)
        _, obs = sampler.sample(shots=n_shots, separate_observables=True, bit_packed=False)
        flip[p] = obs.mean(axis=0)  # (k,)

    header = ("k  w_lz  w_red | " +
              " | ".join(f"p={p:<6.4f} flip  BCE" for p in ps))
    print("\n" + header)
    print("-" * len(header))
    for i in range(k):
        row = f"{i:<2d} {weights[i]:>4d} {reduced[i]:>5d} | "
        cells = []
        for p in ps:
            fr = float(flip[p][i])
            cells.append(f"        {fr:5.3f} {h2_nats(fr):5.3f}")
        print(row + " | ".join(cells))
    print(f"\n(ln2 = {np.log(2):.4f} nats. flip -> 0.5 and BCE -> ln2 == dead head.)")


def main() -> None:
    ps = [0.001, 0.003, 0.0055]
    n_shots = 200_000
    # BB-72: rounds=6 (iter-3 revived all 12 heads on this code).
    probe_code(BBCode.code_72_12_6(), rounds=6, ps=ps, n_shots=n_shots)
    # BB-144: rounds=12 (iter-6 dead heads 8-11).
    probe_code(BBCode.code_144_12_12(), rounds=12, ps=ps, n_shots=n_shots)


if __name__ == "__main__":
    main()
