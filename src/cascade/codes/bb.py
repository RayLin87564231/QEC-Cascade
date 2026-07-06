"""Bivariate bicycle (BB) codes.

Defined by a pair of polynomials over Z_ell × Z_m

    A = x^a1 + y^a2 + y^a3
    B = y^b1 + x^b2 + x^b3

The check matrices are H_X = [A | B] and H_Z = [B^T | A^T] over F_2.
We follow the conventions of Bravyi et al. 2024 (arXiv:2308.07915) and
the IBM reference implementation `sbravyi/BivariateBicycleCodes`.

Two pre-set codes are exposed:

* ``BBCode.code_72_12_6()`` — [[72, 12, 6]]  (ell=6, m=6)
* ``BBCode.code_144_12_12()`` — [[144, 12, 12]] Gross code (ell=12, m=6)

The Stim circuit follows IBM's depth-7 ZX schedule (sX, sZ in
``decoder_setup.py``) translated to Stim's instruction set. Validation
that the constructed circuit reproduces the expected code distance is
left to ``tests/test_bb.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Sequence

import numpy as np
import stim

from .base import Code, DetectorLayout


@dataclass
class BBCodeParams:
    ell: int
    m: int
    a: tuple[int, int, int]  # (a1, a2, a3)
    b: tuple[int, int, int]  # (b1, b2, b3)

    @property
    def n_data(self) -> int:
        return 2 * self.ell * self.m

    @property
    def n_check(self) -> int:
        return self.ell * self.m  # number of X-checks (== Z-checks)


# IBM's depth-7 syndrome cycle schedule. Index by t=0..6.
# 'idle' means the ancilla is idle that step.
SCHEDULE_X = ("idle", 1, 4, 3, 5, 0, 2)
SCHEDULE_Z = (3, 5, 0, 1, 2, 4, "idle")


def _cyclic_shift(n: int, shift: int) -> np.ndarray:
    """Return the n×n cyclic shift matrix, shifting indices by ``shift`` (mod n)."""
    out = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        out[i, (i + shift) % n] = 1
    return out


@dataclass
class BBCode(Code):
    """Bivariate bicycle code with depth-7 syndrome cycle."""

    params: BBCodeParams
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            object.__setattr__(
                self, "name",
                f"bb_{self.params.n_data}_{self.k}_d",  # d unknown w/o ILP
            )

    # --- pre-set factories ---
    @classmethod
    def code_72_12_6(cls) -> "BBCode":
        return cls(params=BBCodeParams(
            ell=6, m=6, a=(3, 1, 2), b=(3, 1, 2),
        ), name="bb_72_12_6")

    @classmethod
    def code_144_12_12(cls) -> "BBCode":
        return cls(params=BBCodeParams(
            ell=12, m=6, a=(3, 1, 2), b=(3, 1, 2),
        ), name="bb_144_12_12")

    # --- check matrix constructions ---
    @cached_property
    def x_op(self) -> np.ndarray:
        """``ell·m × ell·m`` cyclic shift representing x (multiplication by x)."""
        ell, m = self.params.ell, self.params.m
        return np.kron(_cyclic_shift(ell, 1), np.eye(m, dtype=np.int64))

    @cached_property
    def y_op(self) -> np.ndarray:
        ell, m = self.params.ell, self.params.m
        return np.kron(np.eye(ell, dtype=np.int64), _cyclic_shift(m, 1))

    def _power(self, op: np.ndarray, k: int) -> np.ndarray:
        out = np.eye(op.shape[0], dtype=np.int64)
        for _ in range(k):
            out = (out @ op) % 2
        return out

    @cached_property
    def A(self) -> np.ndarray:
        a1, a2, a3 = self.params.a
        return (self._power(self.x_op, a1)
                + self._power(self.y_op, a2)
                + self._power(self.y_op, a3)) % 2

    @cached_property
    def B(self) -> np.ndarray:
        b1, b2, b3 = self.params.b
        return (self._power(self.y_op, b1)
                + self._power(self.x_op, b2)
                + self._power(self.x_op, b3)) % 2

    @cached_property
    def hx(self) -> np.ndarray:
        return np.hstack([self.A, self.B]) % 2

    @cached_property
    def hz(self) -> np.ndarray:
        return np.hstack([self.B.T, self.A.T]) % 2

    @cached_property
    def k(self) -> int:
        """Number of logical qubits."""
        n = self.params.n_data
        return n - _rank_f2(self.hx) - _rank_f2(self.hz)

    # --- Tanner graph ---
    @cached_property
    def tanner_x_to_data(self) -> np.ndarray:
        """``(n_check, 6)`` array: data-qubit index for each X-check's 6 neighbours.

        Order: 0, 1, 2 → A1, A2, A3 (data_left side);
              3, 4, 5 → B1, B2, B3 (data_right side).
        """
        return self._build_tanner(check_kind="x")

    @cached_property
    def tanner_z_to_data(self) -> np.ndarray:
        """Same structure for Z-checks (order: B1.T, B2.T, B3.T then A1.T..A3.T)."""
        return self._build_tanner(check_kind="z")

    def _build_tanner(self, *, check_kind: str) -> np.ndarray:
        a1, a2, a3 = self.params.a
        b1, b2, b3 = self.params.b
        x = self.x_op
        y = self.y_op
        if check_kind == "x":
            ops_left = (self._power(x, a1), self._power(y, a2), self._power(y, a3))
            ops_right = (self._power(y, b1), self._power(x, b2), self._power(x, b3))
        else:
            ops_left = (self._power(y, b1).T, self._power(x, b2).T, self._power(x, b3).T)
            ops_right = (self._power(x, a1).T, self._power(y, a2).T, self._power(y, a3).T)
        n2 = self.params.ell * self.params.m
        out = np.zeros((n2, 6), dtype=np.int64)
        n_data_left = n2  # data_left qubits indexed 0..n2-1
        for i in range(n2):
            for d, op in enumerate(ops_left):
                row_nz = np.flatnonzero(op[i, :])
                assert len(row_nz) == 1, f"unexpected weight at row {i}, op idx {d}"
                out[i, d] = row_nz[0]  # data_left index
            for d, op in enumerate(ops_right):
                row_nz = np.flatnonzero(op[i, :])
                assert len(row_nz) == 1
                out[i, 3 + d] = n_data_left + row_nz[0]  # data_right index
        return out

    # --- logical operators (matrix in F_2) ---
    @cached_property
    def lz(self) -> np.ndarray:
        """``(k, n_data)`` matrix; row i is the support of the i-th Z̄ logical.

        Returns *minimum-weight* coset representatives (iter-6 fix #1). The
        Z-memory observable and the decoder's per-logical pooling both read
        this matrix, so a low Hamming weight keeps the per-head XOR-parity
        low-order enough for the mean-pool readout to represent it (see
        ``reports/iteration_6_deadhead_rootcause.md``). Any two representatives
        of a logical differ only by measured Z-stabilisers, so reducing weight
        yields an equivalent, valid decoding target. The basis is a GF(2)
        change of basis of the old (Gaussian) one — block-level P_L is invariant
        under it; per-logical head indices now predict the new basis bits.
        """
        return _min_weight_logical_basis(self.hx, self.hz, kind="z")

    @cached_property
    def lx(self) -> np.ndarray:
        """``(k, n_data)`` minimum-weight X̄ representatives (see ``lz``)."""
        return _min_weight_logical_basis(self.hx, self.hz, kind="x")

    # --- Code interface implementation ---
    @property
    def num_data_qubits(self) -> int:
        return self.params.n_data

    @property
    def num_logical_observables(self) -> int:
        return int(self.k)

    def make_circuit(self, *, p: float, rounds: int) -> stim.Circuit:
        from cascade.codes.bb_circuit import build_memory_circuit
        return build_memory_circuit(self, p=p, rounds=rounds, basis="z")

    def detector_layout(self, *, rounds: int) -> DetectorLayout:
        # Layout will be derived from circuit detector coordinates set by the
        # circuit builder. We parse them analogously to surface code.
        circuit = self.make_circuit(p=0.001, rounds=rounds)
        return _layout_from_circuit(circuit, self.params, rounds=rounds)

    def logical_supports(self) -> np.ndarray:
        return self.lz.astype(bool)


# ---------------------------------------------------------------------------
# F2 helpers
# ---------------------------------------------------------------------------

def _rank_f2(A: np.ndarray) -> int:
    """Rank over GF(2) via Gaussian elimination."""
    A = A.copy() % 2
    rows, cols = A.shape
    rank = 0
    pivot_col = 0
    for r in range(rows):
        while pivot_col < cols and A[r:, pivot_col].sum() == 0:
            pivot_col += 1
        if pivot_col == cols:
            break
        # Find a row at or below r with 1 in pivot_col
        nz = np.flatnonzero(A[r:, pivot_col])
        if len(nz) == 0:
            pivot_col += 1
            continue
        swap = nz[0] + r
        if swap != r:
            A[[r, swap]] = A[[swap, r]]
        # Eliminate
        for rr in range(rows):
            if rr != r and A[rr, pivot_col]:
                A[rr, :] = (A[rr, :] + A[r, :]) % 2
        rank += 1
        pivot_col += 1
    return rank


def _kernel_f2(A: np.ndarray) -> np.ndarray:
    """Return columns spanning the null space of ``A`` over GF(2)."""
    A = A.copy() % 2
    rows, n = A.shape
    pivot_cols: list[int] = []
    R = A.copy()
    r = 0
    for c in range(n):
        nz = np.flatnonzero(R[r:, c])
        if len(nz) == 0:
            continue
        swap = nz[0] + r
        if swap != r:
            R[[r, swap]] = R[[swap, r]]
        for rr in range(rows):
            if rr != r and R[rr, c]:
                R[rr, :] = (R[rr, :] + R[r, :]) % 2
        pivot_cols.append(c)
        r += 1
        if r == rows:
            break
    free_cols = [c for c in range(n) if c not in pivot_cols]
    if not free_cols:
        return np.zeros((n, 0), dtype=np.int64)
    basis = np.zeros((n, len(free_cols)), dtype=np.int64)
    for j, fc in enumerate(free_cols):
        v = np.zeros(n, dtype=np.int64)
        v[fc] = 1
        for pi, pc in enumerate(pivot_cols):
            if R[pi, fc]:
                v[pc] = 1
        basis[:, j] = v
    return basis


def _logical_basis_gaussian(hx: np.ndarray, hz: np.ndarray, *, kind: str) -> np.ndarray:
    """Compute *arbitrary* (Gaussian-elimination) logical representatives.

    For ``kind='z'``: rows of the result span ker(hx) / image(hz^T).
    For ``kind='x'``: rows span ker(hz) / image(hx^T).

    These reps are **not** weight-minimised (high-index logicals accrue large
    Hamming weight). Used as the initial basis / span generator for
    ``_min_weight_logical_basis`` and kept for provenance / probing.
    """
    if kind == "z":
        ker = _kernel_f2(hx)  # columns are vectors v with hx v = 0
        img_T = hz  # rows of hz are stabilizers, image is span of rows -> v in row space
        # Find v in ker that is independent of rows of hz.
        candidates = []
        rows = list(img_T)
        for j in range(ker.shape[1]):
            v = ker[:, j]
            new_rows = rows + candidates + [v]
            mat = np.array(new_rows) % 2
            if _rank_f2(mat) > _rank_f2(np.array(rows + candidates)):
                candidates.append(v)
        return np.array(candidates, dtype=np.int64) % 2
    elif kind == "x":
        ker = _kernel_f2(hz)
        rows = list(hx)
        candidates = []
        for j in range(ker.shape[1]):
            v = ker[:, j]
            new_rows = rows + candidates + [v]
            mat = np.array(new_rows) % 2
            if _rank_f2(mat) > _rank_f2(np.array(rows + candidates)):
                candidates.append(v)
        return np.array(candidates, dtype=np.int64) % 2
    else:
        raise ValueError(kind)


# ---------------------------------------------------------------------------
# Minimum-weight logical basis (iter-6 dead-head fix #1)
#
# The Gaussian representatives above can have Hamming weight far above the code
# distance (BB-144: up to 38). The Z-memory observable and the decoder's
# per-logical mean-pool both read `code.lz`, and a mean pool cannot represent a
# wide XOR-parity, so high-weight rows produce permanently-dead heads
# (reports/iteration_6_deadhead_rootcause.md). We replace each representative by
# a minimum-weight element of its coset (logical + stabiliser rowspace) — an
# equivalent, valid decoding target of lower parity order.
#
# Method: randomised information-set search. G = [stabilisers ; logicals] spans
# the CSS sub-code C (ker(hx) for Z, ker(hz) for X). For many random column
# orders we row-reduce G; every reduced row is a codeword of C, and random
# information sets surface low-weight ones. We pool the candidates, then take a
# matroid-greedy minimum-weight basis (independence = linear independence mod
# stabilisers). Greedy-by-weight over a matroid minimises the *maximum* weight
# element (bottleneck property), so the pooled search converges to the lowest
# achievable per-head weights. A fixed seed makes the basis deterministic, so
# the observable (circuit) and the model support always agree — across
# processes and across the BP+OSD baseline circuit.
# ---------------------------------------------------------------------------

def _greedy_reduce_vec(v: np.ndarray, stab: np.ndarray) -> np.ndarray:
    """Greedily XOR stabiliser rows into ``v`` to lower its Hamming weight.

    Only adds stabilisers, so the logical class (coset) is preserved. Returns a
    representative of weight <= weight(v). Local minimum, not global.
    """
    v = (v % 2).astype(np.uint8).copy()
    stab = (stab % 2).astype(np.uint8)
    improved = True
    while improved:
        improved = False
        w = int(v.sum())
        for s in stab:
            cand = v ^ s
            cw = int(cand.sum())
            if cw < w:
                v, w, improved = cand, cw, True
    return v


def _rref_rows_f2(G: np.ndarray, col_order: np.ndarray) -> np.ndarray:
    """Row-reduce ``G`` over GF(2), choosing pivots along ``col_order``.

    Returns the nonzero reduced rows; each row is a codeword of rowspace(G),
    expressed in the original column coordinates. Different ``col_order``s
    (random information sets) surface different low-weight codewords.
    """
    R = (G % 2).astype(np.uint8).copy()
    nrows = R.shape[0]
    pr = 0
    for c in col_order:
        if pr >= nrows:
            break
        sub = R[pr:, c]
        nz = np.nonzero(sub)[0]
        if nz.size == 0:
            continue
        piv = pr + int(nz[0])
        if piv != pr:
            R[[pr, piv]] = R[[piv, pr]]
        col = R[:, c].astype(bool).copy()
        col[pr] = False
        if col.any():
            R[col] ^= R[pr]
        pr += 1
    return R[:pr]


class _F2Span:
    """Incremental GF(2) row space with O(#rows * n) independence test."""

    def __init__(self) -> None:
        self.rows: list[np.ndarray] = []   # reduced echelon rows
        self.piv: list[int] = []           # pivot column per row

    def add(self, v: np.ndarray) -> bool:
        """Add ``v`` if independent of the current span. Returns True if added."""
        v = (v % 2).astype(np.uint8).copy()
        for e, p in zip(self.rows, self.piv):
            if v[p]:
                v ^= e
        nz = np.nonzero(v)[0]
        if nz.size == 0:
            return False
        self.rows.append(v)
        self.piv.append(int(nz[0]))
        return True


def _select_min_weight_basis(
    cand_vecs: list[tuple[int, np.ndarray]], stab: np.ndarray, k: int,
) -> list[np.ndarray] | None:
    """Matroid-greedy minimum-weight basis of ``k`` logicals from candidates.

    Independence = linear independence modulo the stabiliser rowspace. Candidates
    are consumed in ascending Hamming weight, so the result minimises the maximum
    per-head weight. Returns ``None`` if fewer than ``k`` independent logicals are
    present in the pool.
    """
    order = sorted(range(len(cand_vecs)), key=lambda i: cand_vecs[i][0])
    span = _F2Span()
    for s in stab:
        span.add(s)
    selected: list[np.ndarray] = []
    for i in order:
        _, v = cand_vecs[i]
        if span.add(v):
            selected.append(v)
            if len(selected) == k:
                return selected
    return None


def _min_weight_logical_basis(
    hx: np.ndarray, hz: np.ndarray, *, kind: str,
    seed: int = 20260705, max_trials: int = 8000, chunk: int = 500,
    target_max_weight: int = 12,
) -> np.ndarray:
    """Minimum-weight logical basis for a CSS code (iter-6 fix #1).

    ``kind='z'`` → Z̄ representatives (reduce mod hz, search ker(hx));
    ``kind='x'`` → X̄ representatives (reduce mod hx, search ker(hz)).

    Deterministic given ``seed`` (identical basis in every process / run). Falls
    back to the Gaussian basis if the search cannot assemble ``k`` independent
    logicals within ``max_trials`` (guaranteed full rank, never fails).
    """
    hx = hx % 2
    hz = hz % 2
    if kind == "z":
        stab = hz
        seed_log = _logical_basis_gaussian(hx, hz, kind="z")
    elif kind == "x":
        stab = hx
        seed_log = _logical_basis_gaussian(hx, hz, kind="x")
    else:
        raise ValueError(kind)

    seed_log = (seed_log % 2).astype(np.uint8)
    k, n = seed_log.shape
    stab = stab.astype(np.uint8)
    G = np.vstack([stab, seed_log]) % 2  # spans the CSS sub-code C

    rng = np.random.default_rng(seed)
    cand_keys: set[bytes] = set()
    cand_vecs: list[tuple[int, np.ndarray]] = []

    def _consider(v: np.ndarray) -> None:
        v = (v % 2).astype(np.uint8)
        w = int(v.sum())
        if w == 0:
            return
        key = v.tobytes()
        if key in cand_keys:
            return
        cand_keys.add(key)
        cand_vecs.append((w, v))

    # Seed the pool with the Gaussian basis and its greedy reductions so a
    # full-rank basis is always assemblable, even before any random trial.
    for row in seed_log:
        _consider(row)
        _consider(_greedy_reduce_vec(row, stab))

    best_basis: list[np.ndarray] | None = None
    trials = 0
    while trials < max_trials:
        for _ in range(chunk):
            perm = rng.permutation(n)
            for row in _rref_rows_f2(G, perm):
                _consider(row)
        trials += chunk
        basis = _select_min_weight_basis(cand_vecs, stab, k)
        if basis is not None:
            best_basis = basis  # monotone: pool only grows, greedy only improves
            if max(int(v.sum()) for v in basis) <= target_max_weight:
                break

    if best_basis is None:
        best_basis = [row for row in seed_log]

    # Polish each selected representative with greedy stabiliser reduction.
    out = np.array(
        [_greedy_reduce_vec(v, stab) for v in best_basis], dtype=np.int64
    ) % 2
    return out


# Backwards-compatible alias: some probes / callers import ``_logical_basis``.
_logical_basis = _logical_basis_gaussian


# ---------------------------------------------------------------------------
# Detector layout helper (used by BBCode.detector_layout)
# ---------------------------------------------------------------------------

def _layout_from_circuit(
    circuit: stim.Circuit, params: BBCodeParams, *, rounds: int
) -> DetectorLayout:
    """Convert a BB circuit's detector coordinates to a (T, ell, m, C) tensor.

    The circuit builder annotates each detector with coordinate
    ``(check_index, time, type)`` where ``type ∈ {0, 1}`` (Z-check, X-check).
    """
    coords = circuit.get_detector_coordinates()
    n_det = circuit.num_detectors
    ell, m = params.ell, params.m

    # Layout: (T = rounds + 1, ell, m, C = 2)
    T = rounds + 1
    grid_shape = (T, ell, m, 2)
    det_to_grid = np.zeros((n_det, 4), dtype=np.int64)
    grid_mask = np.zeros(grid_shape, dtype=bool)

    for d_idx in range(n_det):
        c = coords[d_idx]
        # we use convention (check_idx, t, type)
        check_idx, t, ctype = int(c[0]), int(c[1]), int(c[2])
        i = check_idx // m
        j = check_idx % m
        det_to_grid[d_idx] = (t, i, j, ctype)
        grid_mask[t, i, j, ctype] = True

    return DetectorLayout(
        grid_shape=grid_shape,
        detector_to_grid=det_to_grid,
        grid_mask=grid_mask,
    )
