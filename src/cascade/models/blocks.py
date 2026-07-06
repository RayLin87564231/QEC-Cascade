"""Bottleneck residual block (Cascade Extended Data Fig. 1).

The block's spatial operator is code-specific; surface uses a standard
3-D convolution, BB uses a generalised torus convolution. The block
itself is geometry-agnostic and just supplies the BN/SiLU/projection
machinery and the ``1/sqrt(2L)`` residual scaling.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class BottleneckBlock(nn.Module):
    """Bottleneck residual block.

    Args:
        hidden: residual stream dimension ``H``.
        spatial_op: callable ``Tensor -> Tensor`` that runs in the
            ``H/4``-dimensional bottleneck space. Output shape must equal
            input shape.
        num_blocks: total ``L`` for the residual scaling ``1/sqrt(2L)``.
        bottleneck_factor: defaults to 4 per the paper.
    """

    def __init__(self, hidden: int, spatial_op: nn.Module, num_blocks: int,
                 bottleneck_factor: int = 4) -> None:
        super().__init__()
        if hidden % bottleneck_factor:
            raise ValueError(
                f"hidden={hidden} not divisible by bottleneck_factor={bottleneck_factor}"
            )
        self.b = hidden // bottleneck_factor
        self.bn1 = nn.BatchNorm3d(hidden)
        self.proj_down = nn.Conv3d(hidden, self.b, kernel_size=1)
        self.bn2 = nn.BatchNorm3d(self.b)
        self.spatial = spatial_op
        self.bn3 = nn.BatchNorm3d(self.b)
        self.proj_up = nn.Conv3d(self.b, hidden, kernel_size=1)
        self.act = nn.SiLU(inplace=False)
        self._scale = 1.0 / math.sqrt(2.0 * num_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.proj_down(self.act(self.bn1(x)))
        x = self.spatial(self.act(self.bn2(x)))
        x = self.proj_up(self.act(self.bn3(x)))
        return residual + self._scale * x
