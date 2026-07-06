"""Abstract code interface for the Cascade decoder.

A `Code` provides everything the data pipeline and the model need:
  * A `stim.Circuit` describing a memory experiment at a chosen noise level.
  * A spatial layout that maps each detector index to a position in a tensor
    grid suitable for convolution.
  * Logical observable supports for the final pooling stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import stim


@dataclass(frozen=True)
class DetectorLayout:
    """How detector indices map onto a structured spatial tensor.

    `grid_shape` is the shape excluding the batch dimension. For surface code
    it is `(T, H, W, C)` where `C=2` stands for X/Z stabilizer channels. For
    BB code it is `(T, ell, m, C)` with `C=2` for X-check/Z-check.

    `detector_to_grid` is `(num_detectors, len(grid_shape))` ints; row `i` is
    the multi-index of detector `i` in the grid.

    `grid_mask` is a bool array of shape `grid_shape`; True where a detector
    actually exists. Cells without a detector are zero-padded by the data
    pipeline and ignored by the convolutional backbone via the mask.
    """

    grid_shape: tuple[int, ...]
    detector_to_grid: np.ndarray  # (num_detectors, ndim) int64
    grid_mask: np.ndarray  # bool of shape `grid_shape`

    @property
    def num_detectors(self) -> int:
        return int(self.detector_to_grid.shape[0])


@runtime_checkable
class Code(Protocol):
    """Stabilizer code providing memory experiments + structural metadata."""

    name: str
    num_data_qubits: int
    num_logical_observables: int

    def make_circuit(self, *, p: float, rounds: int) -> stim.Circuit:
        """Build a circuit-level depolarizing memory experiment."""
        ...

    def detector_layout(self, *, rounds: int) -> DetectorLayout:
        """Spatial layout of detectors for the chosen number of rounds."""
        ...

    def logical_supports(self) -> np.ndarray:
        """Bool array `(num_logical_observables, num_data_qubits)`.

        Row `i` is True at data-qubit indices in the support of the i-th
        logical observable. Used for the final average-pool that produces
        per-observable logits.
        """
        ...
