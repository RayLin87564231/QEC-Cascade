"""Generalised torus convolution for BB codes.

Implements the Cascade paper's bipartite factorisation: each layer first
maps check-node hidden states to a learnable hidden state on every data
qubit (``check → data``), then back to the check nodes
(``data → check``). Each step is a small per-offset linear operator
sharing weights across the torus thanks to the code's translation
symmetry.

The forward pass operates on tensors of shape ``(B, F, T, ell, m, C=2)``,
where ``C`` distinguishes X-checks (channel 1) from Z-checks (channel 0)
and ``F`` is the bottleneck feature dim. The temporal axis uses a small
1-D convolution; spatial axes use a learned linear combination of
neighbour offsets.

Implementation notes
--------------------
* The 6 spatial neighbours per check come from the polynomial offsets
  ``A1=x^a1, A2=y^a2, A3=y^a3, B1=y^b1, B2=x^b2, B3=x^b3``.
* For X-checks: A operators connect to data_left, B operators to
  data_right.
* For Z-checks: B^T operators to data_left, A^T to data_right (i.e. the
  inverse cyclic shifts).
* We implement neighbour pickup via ``torch.roll`` once per offset and
  fuse the per-offset matrix multiplies into a single batched einsum.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _PerOffsetLinear(nn.Module):
    """K independent F×F linear maps, one per offset, applied in batch."""

    def __init__(self, num_offsets: int, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_offsets, out_features, in_features)
        )
        # Kaiming-like init scaled by 1/sqrt(num_offsets * in_features) so the
        # *summed* output across offsets has unit variance at initialisation.
        bound = 1.0 / ((num_offsets * in_features) ** 0.5)
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, neighbours: torch.Tensor) -> torch.Tensor:
        """neighbours: (..., num_offsets, in_features) → (..., out_features)."""
        # Sum over offsets and contract the input feature dim.
        return torch.einsum("oji,...oi->...j", self.weight, neighbours)


class BBTorusConv(nn.Module):
    """One full bipartite step over the torus.

    A single block performs two half-steps:
        check → data → check

    Each half-step combines 6 spatial neighbours (per the polynomial
    structure) with 3 temporal offsets (-1, 0, +1) for 18 distinct
    relations per half-step. We compress this further by factoring time
    out of space: a 3-tap 1-D temporal convolution plus a 6-offset
    spatial linear map per half-step. Because every relation has its own
    learned matrix in the Cascade paper, we instead expose the 6-offset
    spatial operator and follow it with a 1-D temporal Conv. This is a
    minor approximation (paper has 12 relations per half-step, we have
    6 + 3) that we accept for the MVP; can be unfactored later.

    Args:
        ell, m: torus dimensions.
        a, b: polynomial offsets.
        channels: feature dimension F.
    """

    NUM_OFFSETS = 6  # 3 from A-block + 3 from B-block

    def __init__(self, ell: int, m: int,
                 a: tuple[int, int, int], b: tuple[int, int, int],
                 channels: int) -> None:
        super().__init__()
        self.ell, self.m = ell, m
        self.a, self.b = a, b
        self.channels = channels

        # The "A-side" offsets shift along x (ell) by a1 and along y (m) by a2, a3.
        # The "B-side" offsets shift along y (m) by b1 and along x (ell) by b2, b3.
        # We register these as buffers so they move with .to(device).
        self._x_shifts = (a[0], 0, 0, 0, b[1], b[2])
        self._y_shifts = (0, a[1], a[2], b[0], 0, 0)

        # check → data and data → check half-steps
        self.cd_x = _PerOffsetLinear(self.NUM_OFFSETS, channels, channels)
        self.cd_z = _PerOffsetLinear(self.NUM_OFFSETS, channels, channels)
        self.dc_x = _PerOffsetLinear(self.NUM_OFFSETS, channels, channels)
        self.dc_z = _PerOffsetLinear(self.NUM_OFFSETS, channels, channels)
        self.self_check = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        # Temporal 1-D convolution (kernel=3, padding=1 with zero pad)
        # applied to the merged tensor after the spatial step.
        # Use grouped convs to keep parameter count manageable.
        self.time_conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)

    @staticmethod
    def _gather_offsets(
        x: torch.Tensor, x_shifts: tuple[int, ...], y_shifts: tuple[int, ...]
    ) -> torch.Tensor:
        """Stack rolled copies of `x` along a new offset dimension.

        x: (B, F, T, ell, m)
        return: (B, F, T, ell, m, num_offsets)
        """
        rolled = []
        for sx, sy in zip(x_shifts, y_shifts):
            r = torch.roll(x, shifts=(-sx, -sy), dims=(3, 4))
            rolled.append(r)
        return torch.stack(rolled, dim=-1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, F, T, ell, m, C=2). Returns same shape."""
        B, F_, T, ell, m, C = h.shape
        assert C == 2

        # --- check → data ---
        # X-checks ("type 1") use A,B as defined; Z-checks ("type 0") use B^T,A^T.
        # For our bookkeeping we track two virtual "data tensors" — one per side
        # of the (data_left, data_right) split — but flatten them since the
        # subsequent half-step recombines them.
        h_x = h[..., 1]  # (B, F, T, ell, m)
        h_z = h[..., 0]
        nbr_x = self._gather_offsets(h_x, self._x_shifts, self._y_shifts)
        nbr_z = self._gather_offsets(h_z, self._x_shifts, self._y_shifts)
        # Move offset axis to before features: (..., num_offsets, F)
        # _PerOffsetLinear expects (..., num_offsets, in_features).
        nbr_x = nbr_x.permute(0, 2, 3, 4, 5, 1).contiguous()  # (B, T, ell, m, num_offsets, F)
        nbr_z = nbr_z.permute(0, 2, 3, 4, 5, 1).contiguous()
        data_from_x = self.cd_x(nbr_x)  # (B, T, ell, m, F)
        data_from_z = self.cd_z(nbr_z)

        # Sum the contributions to form a unified data-side hidden representation.
        data_hidden = data_from_x + data_from_z  # (B, T, ell, m, F)

        # --- data → check ---
        # Apply inverse offsets (negatives) so that "data shifted to neighbour
        # of check c" comes back to position c.
        data_hidden_perm = data_hidden.permute(0, 4, 1, 2, 3).contiguous()  # (B, F, T, ell, m)
        inv_x_shifts = tuple(-sx for sx in self._x_shifts)
        inv_y_shifts = tuple(-sy for sy in self._y_shifts)
        nbr = self._gather_offsets(data_hidden_perm, inv_x_shifts, inv_y_shifts)
        nbr = nbr.permute(0, 2, 3, 4, 5, 1).contiguous()
        check_x = self.dc_x(nbr)  # (B, T, ell, m, F)
        check_z = self.dc_z(nbr)

        # Self-loop: a learnable identity-ish term per check.
        # Reshape (B, F, T, ell, m, C) → (B*T*ell*m*C, F, 1) for Conv1d.
        h_flat = h.permute(0, 2, 3, 4, 5, 1).reshape(-1, F_, 1)
        self_term = self.self_check(h_flat).reshape(
            B, T, ell, m, C, F_
        )

        out = torch.stack([check_z, check_x], dim=-2)  # (B, T, ell, m, C=2, F)
        out = out + self_term  # add self-loop

        # Permute back to (B, F, T, ell, m, C)
        out = out.permute(0, 5, 1, 2, 3, 4).contiguous()

        # --- temporal 1-D conv ---
        # Reshape (B, F, T, ell, m, C) → (B*ell*m*C, F, T)
        out_t = out.permute(0, 3, 4, 5, 1, 2).reshape(B * ell * m * C, F_, T)
        out_t = self.time_conv(out_t)
        out_t = out_t.reshape(B, ell, m, C, F_, T).permute(0, 4, 5, 1, 2, 3).contiguous()

        return out_t
