"""Tests for the Cascade model wiring."""

from __future__ import annotations

import torch

from cascade.codes.surface import SurfaceCode
from cascade.data.tensorize import detection_events_to_grid, make_grid_indexer
from cascade.models.cascade import CascadeModel


def test_forward_backward_d3():
    code = SurfaceCode(distance=3)
    layout = code.detector_layout(rounds=3)
    flat_idx, grid_shape = make_grid_indexer(layout)
    model = CascadeModel(layout=layout, num_logicals=1, hidden=32, num_blocks=2)
    events = torch.zeros(8, layout.num_detectors)
    events[0, 0] = 1
    events[1, 5] = 1
    grid = detection_events_to_grid(events, flat_idx, grid_shape)
    logits = model(grid)
    assert logits.shape == (8, 1)
    assert torch.isfinite(logits).all()
    loss = logits.sum()
    loss.backward()
    nz_grads = sum(1 for p in model.parameters() if p.grad is not None and p.grad.any())
    assert nz_grads > 0


def test_grid_mask_buffer_shape():
    code = SurfaceCode(distance=3)
    layout = code.detector_layout(rounds=3)
    model = CascadeModel(layout=layout, num_logicals=1, hidden=32, num_blocks=2)
    T, H, W, C = layout.grid_shape
    assert tuple(model.input_mask.shape) == (1, C, T, H, W)
    assert tuple(model.reduced_mask.shape) == (1, 1, T, H, W)
