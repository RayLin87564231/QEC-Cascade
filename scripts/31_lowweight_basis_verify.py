"""Verify the minimum-weight logical basis (iter-6 dead-head fix #1).

Confirms, for BB-72 and BB-144, that `code.lz`/`code.lx` are now
minimum-weight coset representatives and that all algebraic invariants the
observable + decoder rely on still hold:

  * new lz/lx Hamming weights (old Gaussian weights printed alongside)
  * all 12 lz representatives weight <= 16                     (acceptance bar)
  * lz in ker(hx), lx in ker(hz)                               (valid logicals)
  * rank(lz @ lx^T) == k                                       (symplectic OK)
  * lz rows independent mod hz; lx rows independent mod hx     (a real basis)
  * BB-72: min lz weight == 6, min lx weight == 6              (test_bb.py)
  * marginal flip rate P(obs=1) per head at train p            (root-cause map)
  * minimiser wall-clock (bounds per-process training startup cost)

Pure stim + numpy. Runs in seconds; dev partition. No training, no gradients.
"""

from __future__ import annotations

import time

import numpy as np

from cascade.codes.bb import (
    BBCode,
    _logical_basis_gaussian,
    _min_weight_logical_basis,
    _rank_f2,
)


def h2_nats(pflip: float) -> float:
    p = min(max(pflip, 1e-12), 1.0 - 1e-12)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def _independent_mod(logicals: np.ndarray, stab: np.ndarray) -> bool:
    """rank([stab; logicals]) - rank(stab) == #logical rows."""
    k = logicals.shape[0]
    r_stab = _rank_f2(stab % 2)
    r_all = _rank_f2(np.vstack([stab % 2, logicals % 2]))
    return (r_all - r_stab) == k


def verify_code(code: BBCode, rounds: int, ps, n_shots: int, seed: int = 0) -> bool:
    hx = code.hx.astype(np.int64)
    hz = code.hz.astype(np.int64)
    k = code.num_logical_observables

    # Old (Gaussian) weights for comparison.
    old_lz = _logical_basis_gaussian(hx, hz, kind="z")
    old_lx = _logical_basis_gaussian(hx, hz, kind="x")
    w_old_lz = old_lz.sum(axis=1).astype(int)
    w_old_lx = old_lx.sum(axis=1).astype(int)

    # New (minimum-weight) — time the minimiser as it runs at model build.
    t0 = time.time()
    new_lz = _min_weight_logical_basis(hx, hz, kind="z")
    t_lz = time.time() - t0
    t0 = time.time()
    new_lx = _min_weight_logical_basis(hx, hz, kind="x")
    t_lx = time.time() - t0
    # Sanity: cached property returns the same thing the circuit/model consume.
    assert np.array_equal(new_lz % 2, code.lz % 2), "lz property != minimiser output"
    assert np.array_equal(new_lx % 2, code.lx % 2), "lx property != minimiser output"

    w_new_lz = new_lz.sum(axis=1).astype(int)
    w_new_lx = new_lx.sum(axis=1).astype(int)

    print(f"\n{'='*78}")
    print(f"{code.name}  (k={k}, n_data={code.num_data_qubits}, rounds={rounds})")
    print(f"{'='*78}")
    print(f"minimiser wall-clock:  lz {t_lz:.2f}s   lx {t_lx:.2f}s")
    print(f"lz weight  old(Gaussian): {w_old_lz.tolist()}")
    print(f"lz weight  new(min-weight): {w_new_lz.tolist()}  "
          f"max={int(w_new_lz.max())} min={int(w_new_lz.min())}")
    print(f"lx weight  old(Gaussian): {w_old_lx.tolist()}")
    print(f"lx weight  new(min-weight): {w_new_lx.tolist()}  "
          f"max={int(w_new_lx.max())} min={int(w_new_lx.min())}")

    # --- Invariant checks ---
    checks: list[tuple[str, bool]] = []
    checks.append(("all lz weight <= 16", bool(w_new_lz.max() <= 16)))
    checks.append(("lz in ker(hx)  (hx @ lz^T == 0)",
                   bool(np.all((hx @ new_lz.T) % 2 == 0))))
    checks.append(("lx in ker(hz)  (hz @ lx^T == 0)",
                   bool(np.all((hz @ new_lx.T) % 2 == 0))))
    sympl = (new_lz @ new_lx.T) % 2
    checks.append((f"symplectic rank(lz @ lx^T) == k({k})",
                   _rank_f2(sympl) == k))
    checks.append(("lz rows independent mod hz",
                   _independent_mod(new_lz, hz)))
    checks.append(("lx rows independent mod hx",
                   _independent_mod(new_lx, hx)))
    checks.append(("lz.shape == (k, n_data)",
                   new_lz.shape == (k, code.num_data_qubits)))
    checks.append(("lx.shape == (k, n_data)",
                   new_lx.shape == (k, code.num_data_qubits)))
    if code.num_data_qubits == 72:  # BB-72 test_bb.py exact-6 requirements
        checks.append(("BB-72 min lz weight == 6", int(w_new_lz.min()) == 6))
        checks.append(("BB-72 min lx weight == 6", int(w_new_lx.min()) == 6))

    print("\ninvariant checks:")
    all_ok = True
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        all_ok = all_ok and ok

    # --- Marginal flip rates (root-cause map; not a pass/fail) ---
    print("\nmarginal flip P(obs=1) at train p (weight -> parity order under mean-pool):")
    flip = {}
    for p in ps:
        sampler = code.make_circuit(p=p, rounds=rounds).compile_detector_sampler(seed=seed)
        _, obs = sampler.sample(shots=n_shots, separate_observables=True, bit_packed=False)
        flip[p] = obs.mean(axis=0)
    header = "k  w_lz | " + " | ".join(f"p={p:<6.4f} flip  BCE" for p in ps)
    print(header)
    print("-" * len(header))
    for i in range(k):
        row = f"{i:<2d} {int(w_new_lz[i]):>4d} | "
        cells = [f"        {float(flip[p][i]):5.3f} {h2_nats(float(flip[p][i])):5.3f}"
                 for p in ps]
        print(row + " | ".join(cells))

    print(f"\n{code.name}: {'ALL INVARIANTS PASS' if all_ok else 'SOME INVARIANTS FAIL'}")
    return all_ok


def main() -> None:
    ps = [0.001, 0.003, 0.0055]
    n_shots = 200_000
    ok72 = verify_code(BBCode.code_72_12_6(), rounds=6, ps=ps, n_shots=n_shots)
    ok144 = verify_code(BBCode.code_144_12_12(), rounds=12, ps=ps, n_shots=n_shots)
    print(f"\n{'#'*78}")
    if ok72 and ok144:
        print("VERIFY_ALL_OK")
    else:
        print(f"VERIFY_FAILED  (bb72={'ok' if ok72 else 'FAIL'}, "
              f"bb144={'ok' if ok144 else 'FAIL'})")
    print(f"{'#'*78}")


if __name__ == "__main__":
    main()
