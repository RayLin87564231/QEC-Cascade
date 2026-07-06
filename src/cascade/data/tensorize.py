"""Convert raw Stim detection events into the model's spatial input tensor."""

from __future__ import annotations

import numpy as np
import torch

from cascade.codes.base import DetectorLayout


def make_grid_indexer(layout: DetectorLayout) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute index tensors that scatter ``(B, num_detectors)`` events
    into ``(B, *grid_shape)`` zero-padded tensors.

    Returns ``(flat_index, grid_shape_tensor)`` where ``flat_index`` is a
    1-D long tensor of length ``num_detectors``: the ravelled grid index
    of each detector. The data pipeline scatters with ``index_add``-like
    semantics, but each detector hits a unique cell so a plain assignment
    via ``flat_index`` is sufficient.
    """
    grid_shape = layout.grid_shape
    flat_idx = np.ravel_multi_index(
        layout.detector_to_grid.T, dims=grid_shape
    )
    return torch.from_numpy(flat_idx).long(), torch.tensor(grid_shape, dtype=torch.long)


def detection_events_to_grid(
    events: torch.Tensor, flat_idx: torch.Tensor, grid_shape: torch.Tensor
) -> torch.Tensor:
    """Scatter ``(B, num_detectors)`` binary events to ``(B, *grid_shape)``.

    The grid is zero-padded; cells without a detector remain zero. The
    permutation produced is consumed by the model after a leading channel
    permutation (the model expects ``(B, C_in, T, H, W)``).
    """
    B = events.shape[0]
    flat = torch.zeros(B, int(np.prod([int(s) for s in grid_shape.tolist()])),
                       dtype=events.dtype, device=events.device)
    flat[:, flat_idx] = events
    return flat.view(B, *[int(s) for s in grid_shape.tolist()])
