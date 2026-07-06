"""Rotated surface code (memory_z) using Stim's built-in generator.

The detector layout maps each detector index to a position in a
``(T, H, W, C)`` tensor where:

* ``T = rounds + 1`` time slices (initial measurement + R rounds + final
  destructive measurement; some slices only carry one stabilizer type)
* ``(H, W)`` is the rotated 2D spatial grid containing both X- and
  Z-stabilizer positions on a checkerboard
* ``C = 2`` separates Z-stabilizers (channel 0) from X-stabilizers (channel 1)

Stim emits detector coordinates ``(x, y, t)``; we discover the
checkerboard automatically by which positions carry a detector at ``t=0``
(those are Z, since the experiment is ``memory_z``).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import stim

from .base import Code, DetectorLayout


@dataclass
class SurfaceCode(Code):
    """Rotated surface code memory_z experiment."""

    distance: int
    name: str = ""

    def __post_init__(self) -> None:
        if self.distance < 3 or self.distance % 2 == 0:
            raise ValueError("distance must be an odd integer ≥ 3")
        if not self.name:
            object.__setattr__(self, "name", f"surface_d{self.distance}")

    @property
    def num_data_qubits(self) -> int:
        return self.distance ** 2

    @property
    def num_logical_observables(self) -> int:
        return 1  # single Z̄ logical for memory_z

    def make_circuit(self, *, p: float, rounds: int) -> stim.Circuit:
        return stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=self.distance,
            rounds=rounds,
            after_clifford_depolarization=p,
            after_reset_flip_probability=p,
            before_measure_flip_probability=p,
            before_round_data_depolarization=p,
        )

    def detector_layout(self, *, rounds: int) -> DetectorLayout:
        # Use a tiny noise to keep `Circuit.generated` happy; coords don't depend on p
        circuit = self.make_circuit(p=0.001, rounds=rounds)
        coords = circuit.get_detector_coordinates()
        n = circuit.num_detectors

        # Discover the spatial grid: unique (x, y) values become axes
        xs = sorted({c[0] for c in coords.values()})
        ys = sorted({c[1] for c in coords.values()})
        x_to_w = {x: w for w, x in enumerate(xs)}
        y_to_h = {y: h for h, y in enumerate(reversed(ys))}  # row 0 is top

        H, W = len(ys), len(xs)
        T = rounds + 1
        C = 2  # 0 = Z stabiliser, 1 = X stabiliser

        # Z-stabilisers are exactly the positions whose detector exists at t=0
        z_positions = {
            (c[0], c[1]) for c in coords.values() if c[2] == 0
        }

        det_to_grid = np.zeros((n, 4), dtype=np.int64)
        grid_mask = np.zeros((T, H, W, C), dtype=bool)

        for d_idx in range(n):
            x, y, t = coords[d_idx]
            h, w = y_to_h[y], x_to_w[x]
            c = 0 if (x, y) in z_positions else 1
            t_int = int(round(t))
            det_to_grid[d_idx] = (t_int, h, w, c)
            grid_mask[t_int, h, w, c] = True

        return DetectorLayout(
            grid_shape=(T, H, W, C),
            detector_to_grid=det_to_grid,
            grid_mask=grid_mask,
        )

    @cached_property
    def _logical_supports(self) -> np.ndarray:
        """Crude logical support placeholder.

        For surface_d, the Z̄ observable runs along one column of d data
        qubits. We don't actually need data-qubit indices for the model
        currently (it pools over the check-node grid), so this is a
        placeholder for interface compliance.
        """
        d = self.distance
        sup = np.zeros((1, self.num_data_qubits), dtype=bool)
        # A column of d qubits forms a representative Z̄. The exact column
        # is irrelevant for accuracy at the model level — we pool over the
        # entire grid in the current model — but we keep this for future
        # support-aware pooling.
        sup[0, ::d] = True  # diagonal of d qubits
        return sup

    def logical_supports(self) -> np.ndarray:
        return self._logical_supports
