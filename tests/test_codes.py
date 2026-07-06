"""Tests for the Code interface and concrete implementations."""

from __future__ import annotations

import numpy as np
import pytest

from cascade.codes.surface import SurfaceCode


@pytest.mark.parametrize("d", [3, 5, 7])
def test_surface_circuit_basic(d):
    code = SurfaceCode(distance=d)
    circuit = code.make_circuit(p=0.001, rounds=d)
    assert circuit.num_observables == 1
    # Total detectors = 2 * (d^2-1)/2 (boundary) + (R-1) * (d^2-1) (bulk)
    expected = (d * d - 1) // 2 * 2 + (d - 1) * (d * d - 1)
    assert circuit.num_detectors == expected, (circuit.num_detectors, expected)


@pytest.mark.parametrize("d", [3, 5, 7])
def test_surface_layout_grid(d):
    code = SurfaceCode(distance=d)
    layout = code.detector_layout(rounds=d)
    T, H, W, C = layout.grid_shape
    assert T == d + 1
    assert H == d + 1
    assert W == d + 1
    assert C == 2

    # Each detector index maps to a unique grid cell
    flat = np.ravel_multi_index(layout.detector_to_grid.T, dims=layout.grid_shape)
    assert len(set(flat.tolist())) == layout.num_detectors

    # Mask True count equals number of detectors
    assert int(layout.grid_mask.sum()) == layout.num_detectors

    # First and last time slices have only one stabiliser type each
    assert layout.grid_mask[0, :, :, 1].sum() == 0  # no X at t=0
    assert layout.grid_mask[T - 1, :, :, 1].sum() == 0  # no X at t=T-1
    assert layout.grid_mask[0, :, :, 0].sum() > 0  # Z exists at t=0


def test_surface_invalid_distance():
    with pytest.raises(ValueError):
        SurfaceCode(distance=4)
    with pytest.raises(ValueError):
        SurfaceCode(distance=2)
