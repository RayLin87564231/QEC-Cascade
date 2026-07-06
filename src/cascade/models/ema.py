"""Exponential moving average of model parameters.

The Cascade paper uses ``β = 0.9998`` and reports all final metrics from
the EMA weights. We swap the live and EMA weights with a context manager
during evaluation to avoid copying the entire state dict.
"""

from __future__ import annotations

import contextlib
from typing import Iterator

import torch
import torch.nn as nn


class ParamEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9998) -> None:
        if not 0 < decay < 1:
            raise ValueError("decay must be in (0, 1)")
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            name: p.detach().clone() for name, p in model.named_parameters()
            if p.requires_grad
        }
        # Track buffers we want consistent at eval-time too (BN running stats).
        self.shadow_buffers: dict[str, torch.Tensor] = {
            name: b.detach().clone() for name, b in model.named_buffers()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[name].mul_(d).add_(p.detach(), alpha=1.0 - d)
        for name, b in model.named_buffers():
            # Direct copy for buffers (BN running stats); they are already
            # exponential moving averages internally. We just keep the
            # latest training-time value so eval mode is consistent.
            self.shadow_buffers[name].copy_(b.detach())

    @contextlib.contextmanager
    def applied(self, model: nn.Module) -> Iterator[None]:
        """Temporarily swap live params with EMA weights."""
        backup_p: dict[str, torch.Tensor] = {}
        backup_b: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            backup_p[name] = p.detach().clone()
            p.data.copy_(self.shadow[name])
        for name, b in model.named_buffers():
            backup_b[name] = b.detach().clone()
            b.data.copy_(self.shadow_buffers[name])
        try:
            yield
        finally:
            for name, p in model.named_parameters():
                if name in backup_p:
                    p.data.copy_(backup_p[name])
            for name, b in model.named_buffers():
                b.data.copy_(backup_b[name])

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "shadow": {k: v.cpu() for k, v in self.shadow.items()},
            "shadow_buffers": {k: v.cpu() for k, v in self.shadow_buffers.items()},
        }

    def load_state_dict(self, state: dict, model: nn.Module) -> None:
        self.decay = state["decay"]
        device = next(model.parameters()).device
        self.shadow = {k: v.to(device) for k, v in state["shadow"].items()}
        self.shadow_buffers = {k: v.to(device) for k, v in state["shadow_buffers"].items()}
