"""Surface code spatial operator: a vanilla 3-D convolution.

Acts on tensors of shape ``(B, F, T, H, W)`` with ``F = hidden / 4`` and
zero padding at boundaries (the paper's exact spec).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SurfaceConv(nn.Module):
    def __init__(self, channels: int, kernel: int = 3) -> None:
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError("kernel must be odd")
        self.conv = nn.Conv3d(
            channels, channels, kernel_size=kernel, padding=kernel // 2,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
