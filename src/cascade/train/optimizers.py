"""Build Muon (matrix-valued params) + Lion (scalar params) optimisers.

The KellerJordan ``muon`` package provides ``SingleDeviceMuonWithAuxAdam``
combining Muon for hidden-conv weights with AdamW for everything else.
We use Lion for the "auxiliary" scalars instead, as in the paper.

Fallback: a vanilla ``AdamW`` optimiser for the MVP track (no Muon, no
Lion). Selected by ``optimizer="adamw"`` in the trainer config.
"""

from __future__ import annotations

import torch
from torch import nn

try:  # KellerJordan/Muon standalone package (single-device variant)
    from muon import SingleDeviceMuon
    HAS_MUON = True
except Exception:  # pragma: no cover
    try:  # torch >= 2.12 ships the same algorithm natively as torch.optim.Muon
        from torch.optim import Muon as SingleDeviceMuon
        HAS_MUON = True
    except Exception:
        HAS_MUON = False

try:
    from lion_pytorch import Lion
    HAS_LION = True
except Exception:  # pragma: no cover
    HAS_LION = False


def _is_matrix_param(p: torch.Tensor) -> bool:
    """Muon wants 2-D-or-higher (matrix-valued) parameters."""
    return p.dim() >= 2


def split_params(model: nn.Module) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return ``(matrix_params, scalar_params)`` excluding requires_grad=False
    entries. Matrix params include conv weights, linear weights, and
    BN/LN affine if any. Scalar params include biases."""
    matrix, scalar = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # We treat 2D+ as "matrix"; 1D (biases, BN gain/bias) as "scalar".
        if _is_matrix_param(p):
            matrix.append(p)
        else:
            scalar.append(p)
    return matrix, scalar


def build_muon_lion(
    model: nn.Module,
    *,
    muon_lr: float = 3e-3,
    muon_weight_decay: float = 3e-3,
    muon_momentum: float = 0.95,
    lion_lr: float = 2e-4,
    lion_weight_decay: float = 0.0,
) -> dict:
    """Construct (muon, lion) optimisers for the Cascade Track 2 recipe.

    Returns a dict ``{"muon": optim, "lion": optim, "matrix": [...], "scalar": [...]}``
    so the caller can step both in lockstep.
    """
    if not (HAS_MUON and HAS_LION):
        missing = "muon" if not HAS_MUON else "lion-pytorch"
        raise RuntimeError(
            f"Track 2 optimiser missing dependency: {missing}. "
            "Install via pyproject deps."
        )
    matrix, scalar = split_params(model)
    muon_opt = SingleDeviceMuon(
        params=matrix,
        lr=muon_lr,
        weight_decay=muon_weight_decay,
        momentum=muon_momentum,
    )
    lion_opt = Lion(
        params=scalar,
        lr=lion_lr,
        weight_decay=lion_weight_decay,
    )
    return {"muon": muon_opt, "lion": lion_opt, "matrix": matrix, "scalar": scalar}


def step_muon_lion(opts: dict) -> None:
    opts["muon"].step()
    opts["lion"].step()


def zero_grad_muon_lion(opts: dict, set_to_none: bool = True) -> None:
    opts["muon"].zero_grad(set_to_none=set_to_none)
    opts["lion"].zero_grad(set_to_none=set_to_none)
