"""Stim circuit builder for BB code memory experiments.

Translates IBM's reference depth-7 ZX schedule (`sX`, `sZ` from
``decoder_setup.py``) into a Stim circuit with circuit-level depolarising
noise. Detectors are annotated with coordinates ``(check_index, t, ctype)``
where ``ctype`` is 0 for Z-check and 1 for X-check, so the detector layout
helper can recover a ``(T, ell, m, C)`` tensor.

Currently only the Z-memory experiment is implemented (the X variant is a
trivial swap of basis and bookkeeping).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import stim

if TYPE_CHECKING:
    from cascade.codes.bb import BBCode

from cascade.codes.bb import SCHEDULE_X, SCHEDULE_Z


# Qubit index layout: data_left | data_right | X-checks | Z-checks
def _qubit_indices(n2: int) -> dict[str, range]:
    return {
        "data_left": range(0, n2),
        "data_right": range(n2, 2 * n2),
        "x_check": range(2 * n2, 3 * n2),
        "z_check": range(3 * n2, 4 * n2),
    }


def _data_qubit_linear(name: str, j: int, n2: int) -> int:
    """Map ('data_left'/'data_right', j) → linear data-qubit index in [0, 2*n2)."""
    if name == "data_left":
        return j
    if name == "data_right":
        return n2 + j
    raise ValueError(name)


def _data_qubit_phys(linear: int, n2: int) -> int:
    """Map data-qubit linear index → physical qubit index."""
    return linear  # data qubits live at physical 0..2*n2-1 by construction


def build_memory_circuit(
    code: "BBCode",
    *,
    p: float,
    rounds: int,
    basis: str = "z",
) -> stim.Circuit:
    """Build a Z-memory Stim circuit for the given BB code at noise level ``p``."""
    if basis != "z":
        raise NotImplementedError("Only Z-memory is implemented")

    p_init = p
    p_idle = p
    p_cnot = p
    p_meas = p

    n2 = code.params.ell * code.params.m
    qrange = _qubit_indices(n2)
    x_check = qrange["x_check"]
    z_check = qrange["z_check"]

    # Pre-compute Tanner adjacency in (X-check index, direction) → physical-qubit index
    tx = code.tanner_x_to_data  # (n2, 6) data-linear indices
    tz = code.tanner_z_to_data  # (n2, 6)

    circuit = stim.Circuit()

    # 1. Coordinates (helps debugging; not strictly required for correctness)
    for j in range(n2):
        circuit.append("QUBIT_COORDS", [j], (j // code.params.m, j % code.params.m, 0))
        circuit.append("QUBIT_COORDS", [n2 + j], (j // code.params.m, j % code.params.m, 1))
        circuit.append("QUBIT_COORDS", [2 * n2 + j], (j // code.params.m, j % code.params.m, 2))
        circuit.append("QUBIT_COORDS", [3 * n2 + j], (j // code.params.m, j % code.params.m, 3))

    # 2. Initialise data qubits in |0⟩ for memory_z; ancillas are reset within the cycle.
    all_data = list(range(0, 2 * n2))
    circuit.append("R", all_data)
    if p_init > 0:
        circuit.append("X_ERROR", all_data, p_init)

    # 3. Build one syndrome cycle. We track the measurement record positions of
    # each ancilla so we can wire detectors round-by-round.
    #
    # Within one cycle, the order of MeasZ Z-checks then later MeasX X-checks
    # matters: we use rec[-N] indexing relative to the end of the cycle.
    #
    # Measurement-record bookkeeping:
    #   * Within each cycle we measure n2 Z-checks then n2 X-checks (in that order).
    #   * Number of measurements per cycle = 2 * n2.
    #   * X-check measurement index in the rec stream = i (0-indexed from start of X-block) ;
    #     Z-check measurement index = i (0-indexed from start of Z-block).
    #   * Both per-cycle measurement orders are aligned with check index 0..n2-1.

    def append_cycle(round_t: int, with_detectors: bool) -> None:
        """Append a full 8-sub-round cycle and (optionally) detectors."""
        # --- Sub-round 0: Reset X-checks; CNOT data → Z-checks (sZ[0]=3) ---
        circuit.append("RX", list(x_check))
        if p_init > 0:
            circuit.append("Z_ERROR", list(x_check), p_init)

        for i in range(n2):
            ctrl = tz[i, SCHEDULE_Z[0]]  # data linear index
            tgt = z_check[i]
            circuit.append("CX", [ctrl, tgt])
            if p_cnot > 0:
                circuit.append("DEPOLARIZE2", [ctrl, tgt], p_cnot)

        # data qubits not CNOTed in this sub-round are idle (subset of all data).
        # IBM's reference puts them as IDLE; we approximate by depolarising every
        # data qubit not used. For circuit-level noise we just depolarise all
        # data qubits (a slight over-noise acceptable at this level).
        idle_data = sorted(set(all_data) - {tz[i, SCHEDULE_Z[0]] for i in range(n2)})
        if p_idle > 0 and idle_data:
            circuit.append("DEPOLARIZE1", idle_data, p_idle)

        # --- Sub-rounds 1-5: parallel CNOT(Xcheck → data) and CNOT(data → Zcheck) ---
        for t in range(1, 6):
            sx = SCHEDULE_X[t]
            sz = SCHEDULE_Z[t]
            for i in range(n2):
                ctrl = x_check[i]
                tgt = tx[i, sx]
                circuit.append("CX", [ctrl, tgt])
                if p_cnot > 0:
                    circuit.append("DEPOLARIZE2", [ctrl, tgt], p_cnot)
            for i in range(n2):
                ctrl = tz[i, sz]
                tgt = z_check[i]
                circuit.append("CX", [ctrl, tgt])
                if p_cnot > 0:
                    circuit.append("DEPOLARIZE2", [ctrl, tgt], p_cnot)

        # --- Sub-round 6: CNOT(Xcheck → data) for sX[6]=2; MeasZ Z-checks ---
        sx = SCHEDULE_X[6]
        for i in range(n2):
            ctrl = x_check[i]
            tgt = tx[i, sx]
            circuit.append("CX", [ctrl, tgt])
            if p_cnot > 0:
                circuit.append("DEPOLARIZE2", [ctrl, tgt], p_cnot)
        # Idle data qubits not touched in this sub-round
        idle_data = sorted(set(all_data) - {int(tx[i, sx]) for i in range(n2)})
        if p_idle > 0 and idle_data:
            circuit.append("DEPOLARIZE1", idle_data, p_idle)
        if p_meas > 0:
            circuit.append("X_ERROR", list(z_check), p_meas)
        circuit.append("MR", list(z_check))  # measure-and-reset Z-checks

        # --- Sub-round 7: data idle; MeasX X-checks; PrepZ already handled via MR for Z ---
        if p_idle > 0:
            circuit.append("DEPOLARIZE1", all_data, p_idle)
        if p_meas > 0:
            circuit.append("Z_ERROR", list(x_check), p_meas)
        circuit.append("MRX", list(x_check))  # measure-and-reset X-checks

        # --- Detectors ---
        if with_detectors:
            # Z-check measurement record indices: most recent batch is X-checks at -1..-n2 ;
            # before that the Z-checks at -n2-1..-2n2.
            # Layout: meas record (in chronological order, last to first):
            #   X-check[n2-1], ..., X-check[0],          (-1, ..., -n2)
            #   Z-check[n2-1], ..., Z-check[0],          (-n2-1, ..., -2n2)
            #   ... previous cycle's X then Z
            for i in range(n2):
                # Z-check i: most recent at -2n2 + i
                rec_now = -2 * n2 + i
                if round_t == 0:
                    # First round: data initialised in |0⟩ deterministic;
                    # so the Z-check measurement should be 0 in the absence
                    # of errors. Detector = current Z-check measurement.
                    circuit.append("DETECTOR", [stim.target_rec(rec_now)],
                                   (i, round_t, 0))
                else:
                    # XOR with previous round's Z-check measurement
                    rec_prev = rec_now - 2 * n2
                    circuit.append("DETECTOR",
                                   [stim.target_rec(rec_now),
                                    stim.target_rec(rec_prev)],
                                   (i, round_t, 0))
            for i in range(n2):
                rec_now = -n2 + i
                if round_t == 0:
                    # X-check first round: random; cannot form detector
                    continue
                rec_prev = rec_now - 2 * n2
                circuit.append("DETECTOR",
                               [stim.target_rec(rec_now),
                                stim.target_rec(rec_prev)],
                               (i, round_t, 1))

    # 3a. Initial cycle (with detectors at t=0 — only Z type).
    # Some implementations do an initial "perfect" round; we just do a noisy
    # round and form the t=0 Z-detectors against the |0⟩ initialisation.
    append_cycle(0, with_detectors=True)
    # 3b. Remaining cycles
    for r in range(1, rounds):
        append_cycle(r, with_detectors=True)

    # 4. Final destructive measurement of all data qubits in Z basis.
    if p_meas > 0:
        circuit.append("X_ERROR", all_data, p_meas)
    circuit.append("M", all_data)

    # Compute Z-stabiliser values from the data measurements and form
    # final Z-detectors against the most recent Z-check measurements.
    # The data measurement records are the very last 2n2 entries; record index
    # for data qubit j is rec[-(2n2 - j)].
    n_data = 2 * n2
    for i in range(n2):
        # Final Z-detector for Z-check i:
        # parity of data qubits in support of hz[i, :]
        # XOR with last Z-check measurement (which was the second-to-last set
        # of measurements before the data destructive measurement: rec at
        # position -(n_data + n2) + ... wait, careful)
        # After the last cycle: layout is [Z-checks (n2), X-checks (n2), data (n_data)]
        # most recent first: rec[-1] = last data; rec[-(n_data)] = first data;
        # rec[-(n_data + 1)] = last X-check; rec[-(n_data + n2)] = first X-check;
        # rec[-(n_data + n2 + 1)] = last Z-check; rec[-(n_data + 2n2)] = first Z-check;
        rec_targets = []
        # Z-stabiliser i support over data qubits
        support = code.hz[i, :]  # shape (n_data,)
        for j, bit in enumerate(support):
            if bit:
                # data qubit j → rec position -(n_data - j)
                rec_targets.append(stim.target_rec(-(n_data - j)))
        # XOR with last Z-check measurement for i
        # Z-check[i] last measurement is at -(n_data + n2 + (n2 - i))
        last_zcheck_rec = -(n_data + 2 * n2 - i)
        rec_targets.append(stim.target_rec(last_zcheck_rec))
        circuit.append("DETECTOR", rec_targets, (i, rounds, 0))

    # 5. Logical observables. For Z-memory we observe each logical Z̄_i.
    # The observable is the parity of data qubits in lz[i, :].
    for i in range(code.k):
        rec_targets = []
        support = code.lz[i, :]
        for j, bit in enumerate(support):
            if bit:
                rec_targets.append(stim.target_rec(-(n_data - j)))
        circuit.append("OBSERVABLE_INCLUDE", rec_targets, i)

    return circuit
