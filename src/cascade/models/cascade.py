"""Cascade neural decoder model.

The model is structured as:
    embed (1x1x1 conv) → L bottleneck blocks → final 1x1x1 conv → masked
    average pool → 2-layer MLP head per logical observable

The grid mask zeroes out positions without detectors at every conv
boundary so spurious zero entries cannot bias the activations or the
final pool.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cascade.codes.base import Code, DetectorLayout
from cascade.models.blocks import BottleneckBlock
from cascade.models.conv_surface import SurfaceConv


def _build_spatial_op(code_kind: str, channels: int, kernel: int) -> nn.Module:
    if code_kind == "surface":
        return SurfaceConv(channels, kernel=kernel)
    raise NotImplementedError(f"Spatial op for code_kind={code_kind!r} not implemented")


class CascadeModel(nn.Module):
    """End-to-end neural decoder for surface code (BB to be added)."""

    def __init__(
        self,
        layout: DetectorLayout,
        num_logicals: int,
        hidden: int = 128,
        num_blocks: int = 8,
        kernel: int = 3,
        code_kind: str = "surface",
    ) -> None:
        super().__init__()
        T, H, W, C = layout.grid_shape
        if code_kind != "surface":
            raise NotImplementedError("BB code not yet wired in this model")

        self.input_channels = C
        self.hidden = hidden
        self.num_logicals = num_logicals

        # Persistent grid mask (T, H, W, C) → broadcast to (1, C, T, H, W) by
        # permuting and adding a leading 1 for the channel dimension. We
        # keep a single-channel reduction (any C True) for use after the
        # backbone where the C axis has been merged into hidden.
        mask = torch.from_numpy(layout.grid_mask).bool()
        # input mask: (1, C, T, H, W)
        in_mask = mask.permute(3, 0, 1, 2).unsqueeze(0)
        # reduced mask for backbone/pool: (1, 1, T, H, W) — True if any
        # stabiliser type has a detector at that (t, h, w)
        reduced_mask = mask.any(dim=-1).unsqueeze(0).unsqueeze(0)
        self.register_buffer("input_mask", in_mask, persistent=False)
        self.register_buffer("reduced_mask", reduced_mask, persistent=False)

        self.embedding = nn.Conv3d(C, hidden, kernel_size=1, bias=True)

        self.blocks = nn.ModuleList([
            BottleneckBlock(
                hidden,
                spatial_op=_build_spatial_op(code_kind, hidden // 4, kernel),
                num_blocks=num_blocks,
            )
            for _ in range(num_blocks)
        ])

        self.final_conv = nn.Conv3d(hidden, hidden * num_logicals,
                                    kernel_size=kernel, padding=kernel // 2,
                                    bias=True)

        # 2-layer MLP head per logical observable (shared weights would
        # also work but per-observable matches the paper).
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, 2 * hidden),
                nn.SiLU(inplace=False),
                nn.Linear(2 * hidden, 1),
            )
            for _ in range(num_logicals)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: ``(B, T, H, W, C)`` binary detection events (float).

        Returns:
            ``(B, num_logicals)`` logits.
        """
        # Permute to (B, C, T, H, W) for Conv3d.
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = x * self.input_mask  # zero non-detector positions

        h = self.embedding(x)
        for blk in self.blocks:
            h = blk(h)

        # Final scatter convolution.
        h = self.final_conv(h)  # (B, hidden * num_logicals, T, H, W)
        B = h.shape[0]
        h = h.view(B, self.num_logicals, self.hidden, *h.shape[2:])

        # Masked average pool over (T, H, W).
        mask = self.reduced_mask  # (1, 1, T, H, W)
        # Expand to (1, num_logicals, 1, T, H, W) for broadcast against h.
        m = mask.unsqueeze(1).to(h.dtype)
        h = h * m
        denom = m.sum(dim=(2, 3, 4, 5)).clamp(min=1.0)  # (1, num_logicals, 1)
        # h sum: (B, num_logicals, hidden)
        pooled = h.sum(dim=(3, 4, 5)) / denom

        # Per-observable MLP head.
        outs = []
        for i, head in enumerate(self.heads):
            outs.append(head(pooled[:, i, :]).squeeze(-1))
        return torch.stack(outs, dim=1)
