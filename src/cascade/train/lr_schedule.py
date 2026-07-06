"""Cosine LR schedule with linear warmup, matching the Cascade paper."""

from __future__ import annotations

import math


def cosine_with_warmup(
    step: int, *, total_steps: int, warmup_steps: int = 1000,
    decay_floor: float = 0.1,
) -> float:
    """Multiplier in ``[decay_floor, 1.0]`` for the peak LR.

    * Linear warmup from 0 → 1 over the first ``warmup_steps``.
    * Cosine decay from 1 → ``decay_floor`` over the remaining steps.
    """
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return decay_floor + (1.0 - decay_floor) * cos
