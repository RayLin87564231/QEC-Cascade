"""Cascade model for bivariate bicycle codes.

The internal hidden representation is

    (B, F, T, ell, 2*m)

where the trailing dimension packs the two check types as
``[Z-checks (m positions) | X-checks (m positions)]``. This shape is
compatible with standard ``nn.Conv3d`` and ``nn.BatchNorm3d`` ops
(treating ``2*m`` as just another spatial axis), while the
``BBTorusConv`` block reshapes back to ``(B, F, T, ell, m, C=2)`` for
the actual generalised convolution before flattening once more.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from cascade.codes.bb import BBCode
from cascade.codes.base import DetectorLayout
from cascade.models.conv_bb import BBTorusConv


class _BBSpatialWrap(nn.Module):
    """Adapter: ``(B, F, T, ell, 2m)`` ⇄ ``(B, F, T, ell, m, C=2)``.

    Layout convention: first ``m`` columns of ``2m`` are Z-checks, last
    ``m`` columns are X-checks. This must match
    ``BBCascadeModel._scatter_input``.
    """

    def __init__(self, ell: int, m: int,
                 a: tuple[int, int, int], b: tuple[int, int, int],
                 channels: int) -> None:
        super().__init__()
        self.ell, self.m = ell, m
        self.conv = BBTorusConv(ell, m, a, b, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, F, T, ell, m2 = x.shape
        assert ell == self.ell and m2 == 2 * self.m
        # Split into (Z-chunk, X-chunk) along the last dim, then stack on a new C axis.
        z_chunk = x[..., : self.m]
        x_chunk = x[..., self.m :]
        h = torch.stack([z_chunk, x_chunk], dim=-1)  # (B, F, T, ell, m, 2)
        h = self.conv(h)
        z_out, x_out = h[..., 0], h[..., 1]
        return torch.cat([z_out, x_out], dim=-1)  # (B, F, T, ell, 2m)


class _BBBottleneckBlock(nn.Module):
    """BottleneckBlock specialised for BB hidden layout.

    Differences from the surface ``BottleneckBlock``: spatial op is the
    BB torus wrapper. Everything else (BN3d, SiLU, 1×1 conv, residual
    scaling) is identical.
    """

    def __init__(self, hidden: int, spatial_op: nn.Module, num_blocks: int,
                 bottleneck_factor: int = 4) -> None:
        super().__init__()
        if hidden % bottleneck_factor:
            raise ValueError(f"hidden={hidden} not divisible by {bottleneck_factor}")
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


class BBCascadeModel(nn.Module):
    """Cascade decoder for bivariate bicycle codes."""

    def __init__(
        self,
        code: BBCode,
        layout: DetectorLayout,
        hidden: int = 128,
        num_blocks: int = 8,
    ) -> None:
        super().__init__()
        self.code = code
        self.hidden = hidden
        ell, m = code.params.ell, code.params.m
        T, _ell, _m, C = layout.grid_shape
        assert _ell == ell and _m == m and C == 2
        self.ell, self.m, self.T = ell, m, T
        self.num_logicals = code.num_logical_observables

        # Per-type embeddings: each (cell, c, value) → F. We use two
        # separate Conv3d(1, F, 1) layers, one for Z-checks and one for
        # X-checks. Output combined into (B, F, T, ell, 2m).
        self.embed_z = nn.Conv3d(1, hidden, kernel_size=1, bias=True)
        self.embed_x = nn.Conv3d(1, hidden, kernel_size=1, bias=True)

        # Backbone
        self.blocks = nn.ModuleList([
            _BBBottleneckBlock(
                hidden,
                spatial_op=_BBSpatialWrap(
                    ell, m, code.params.a, code.params.b, hidden // 4,
                ),
                num_blocks=num_blocks,
            )
            for _ in range(num_blocks)
        ])

        # Final scatter conv: produce a per-data-qubit hidden vector by
        # scattering check-node features through the Tanner graph. The
        # scatter is fixed (not learned) as a sparse buffer; the *content*
        # of each (B, F, T, ell, 2m) check-node feature gets averaged over
        # the checks that include each data qubit in their support.
        # Output of the scatter: (B, F, T, n_data).
        self.n_data = code.num_data_qubits

        # Build scatter weight: for each data qubit q, list of (flat check
        # index, channel) pairs that include q. The flat layout is
        # [Z-checks, X-checks] each of size n_check, totalling 2*n_check
        # positions per (t, i, j) cell. We collapse (i, j) to a flat
        # spatial axis then express scatter as a sparse (n_data, 2*n_check)
        # matrix and apply via dense matmul (n_check small enough).
        scatter_z = torch.from_numpy(code.hz.astype(np.float32))  # (n_check, n_data)
        scatter_x = torch.from_numpy(code.hx.astype(np.float32))  # (n_check, n_data)
        # Normalise per data qubit so scatter is an *average* not sum.
        zx_count = (scatter_z.sum(dim=0) + scatter_x.sum(dim=0)).clamp(min=1.0)
        # Combined: (2*n_check, n_data) where rows 0..n_check are Z-checks
        # and rows n_check..2*n_check are X-checks.
        n_check_total = ell * m
        scatter = torch.cat([scatter_z, scatter_x], dim=0)  # (2*n_check, n_data)
        scatter = scatter / zx_count[None, :]
        # We want to apply scatter across the (ell, m, C=2)→(2*n_check) axis.
        # Reshape: input feature (B, F, T, ell, m, C) → (B, F, T, 2*n_check)
        # by ordering as [Z-flatten over (i,j)] ++ [X-flatten over (i,j)].
        self.register_buffer("scatter_matrix", scatter, persistent=False)

        # Per-logical support pool: precompute boolean mask of which data
        # qubits belong to each logical's lz[i] support, for averaging.
        lz_support = torch.from_numpy(code.lz.astype(bool))  # (k, n_data)
        self.register_buffer(
            "lz_support_float", lz_support.to(torch.float32), persistent=False,
        )
        # And per-logical denominator (count of qubits in each logical support)
        self.register_buffer(
            "lz_support_count",
            lz_support.sum(dim=1).clamp(min=1).to(torch.float32),
            persistent=False,
        )

        # Final scatter conv: produces per-logical features at every check.
        # We keep a smaller projection: (hidden) → (hidden * num_logicals),
        # then group by logical so each logical channel gets its own scatter.
        self.final_conv = nn.Conv3d(hidden, hidden * self.num_logicals,
                                    kernel_size=1, bias=True)

        # Per-observable head.
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, 2 * hidden),
                nn.SiLU(inplace=False),
                nn.Linear(2 * hidden, 1),
            )
            for _ in range(self.num_logicals)
        ])

        # Reduced grid mask: True wherever any detector exists.
        # layout.grid_mask is (T, ell, m, 2). Convert to (T, ell, 2m) layout.
        mask = torch.from_numpy(layout.grid_mask).bool()  # (T, ell, m, 2)
        z_mask = mask[..., 0]  # (T, ell, m)
        x_mask = mask[..., 1]
        flat_mask = torch.cat([z_mask, x_mask], dim=-1)  # (T, ell, 2m)
        # Add a leading (1, 1) for broadcast against (B, F, T, ell, 2m) features.
        self.register_buffer("flat_mask", flat_mask.unsqueeze(0).unsqueeze(0),
                             persistent=False)

        # ------------------------------------------------------------------
        # Per-logical detector relevance mask.
        #
        # For Z-memory experiment, logical Z̄_i flips iff X errors land in
        # support `lz[i, :]`. A check at flat data-qubit support `s` is
        # informationally relevant to logical i iff their supports overlap.
        # We build a boolean mask of shape (num_logicals, T, ell, 2m) that
        # is True at (t, i_check, j_check, c) when the corresponding
        # stabiliser intersects `code.lz[logical, :]`. The temporal axis is
        # uniformly active where the underlying detector exists.
        # ------------------------------------------------------------------
        lz = torch.from_numpy(code.lz.astype(np.int64))      # (k, n_data)
        hx_t = torch.from_numpy(code.hx.astype(np.int64))    # (n_check, n_data)
        hz_t = torch.from_numpy(code.hz.astype(np.int64))    # (n_check, n_data)
        n_check = ell * m
        # X-check c is relevant to logical i iff (hx[c] · lz[i]).sum() > 0
        # (overlap count, regardless of mod-2 value — even overlap still
        # carries information through individual error events).
        x_rel = (hx_t.float() @ lz.t().float()).long() > 0   # (n_check, k) → bool
        z_rel = (hz_t.float() @ lz.t().float()).long() > 0   # (n_check, k)
        # Reshape n_check = ell * m to spatial grid
        x_rel = x_rel.view(ell, m, self.num_logicals)        # (ell, m, k)
        z_rel = z_rel.view(ell, m, self.num_logicals)        # (ell, m, k)
        # Move logical axis first
        x_rel = x_rel.permute(2, 0, 1)                       # (k, ell, m)
        z_rel = z_rel.permute(2, 0, 1)                       # (k, ell, m)
        # AND with grid_mask per channel (so we never count non-existent detectors)
        z_rel = z_rel.unsqueeze(0).expand(T, -1, -1, -1)     # (T, k, ell, m)
        x_rel = x_rel.unsqueeze(0).expand(T, -1, -1, -1)
        z_grid = mask[..., 0].unsqueeze(1)                   # (T, 1, ell, m)
        x_grid = mask[..., 1].unsqueeze(1)
        z_full = z_rel & z_grid                              # (T, k, ell, m)
        x_full = x_rel & x_grid
        # Reshape to (k, T, ell, 2m) matching the [Z|X] flat layout
        z_full = z_full.permute(1, 0, 2, 3)                  # (k, T, ell, m)
        x_full = x_full.permute(1, 0, 2, 3)
        per_logical_mask = torch.cat([z_full, x_full], dim=-1)  # (k, T, ell, 2m)
        # Broadcast shape for forward: (1, k, 1, T, ell, 2m)
        self.register_buffer(
            "per_logical_mask",
            per_logical_mask.unsqueeze(0).unsqueeze(2),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: ``(B, T, ell, m, C=2)`` binary detection events.

        Returns ``(B, num_logicals)`` logits.
        """
        B, T, ell, m, C = x.shape
        assert (T, ell, m, C) == (self.T, self.ell, self.m, 2)

        # Per-type embed.
        z_in = x[..., 0].unsqueeze(1)  # (B, 1, T, ell, m)
        x_in = x[..., 1].unsqueeze(1)  # (B, 1, T, ell, m)
        z_h = self.embed_z(z_in)        # (B, F, T, ell, m)
        x_h = self.embed_x(x_in)
        h = torch.cat([z_h, x_h], dim=-1)  # (B, F, T, ell, 2m)
        h = h * self.flat_mask  # zero out positions without detectors

        for blk in self.blocks:
            h = blk(h)

        # Final scatter
        h = self.final_conv(h)  # (B, F * num_logicals, T, ell, 2m)
        F_ = self.hidden
        K = self.num_logicals
        # h: (B, K * F, T, ell, 2m). Reshape so per-logical channel-block
        # is its own slot.
        h = h.view(B, K, F_, T, ell, 2 * m)

        # Scatter check-node features → data-qubit features.
        # First flatten the (ell, m, C=2) axis into 2*n_check (Z then X).
        # Layout is already [Z-cols (m), X-cols (m)] across the 2m axis,
        # arranged in (ell, 2m). We reshape to (B, K, F, T, ell*m, C=2)
        # and concat along (channel × spatial) to match scatter_matrix rows.
        # Easier: directly reshape (ell, 2m) → split into two (ell, m)
        # halves [Z, X], flatten each separately, then concatenate.
        h_z = h[..., :m].reshape(B, K, F_, T, ell * m)         # (B, K, F, T, n_check)
        h_x = h[..., m:].reshape(B, K, F_, T, ell * m)
        h_flat = torch.cat([h_z, h_x], dim=-1)                  # (B, K, F, T, 2*n_check)
        # Apply scatter: data_feat = h_flat @ scatter_matrix
        scatter = self.scatter_matrix.to(h.dtype)               # (2*n_check, n_data)
        data_feat = torch.einsum("bkfts,sn->bkftn", h_flat, scatter)
        # Average over time axis
        data_feat = data_feat.mean(dim=3)                       # (B, K, F, n_data)

        # Per-logical pool over lz[i] support
        # lz_support_float: (K, n_data); for logical i, mask = lz_support_float[i]
        support = self.lz_support_float.to(h.dtype)             # (K, n_data)
        denom = self.lz_support_count.to(h.dtype)               # (K,)
        # einsum: pool data_feat over n_data, weighted by support[i]
        pooled = torch.einsum("bkfn,kn->bkf", data_feat, support) / denom[None, :, None]

        outs = [head(pooled[:, i, :]).squeeze(-1) for i, head in enumerate(self.heads)]
        return torch.stack(outs, dim=1)
