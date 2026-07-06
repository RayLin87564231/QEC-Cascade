"""Three-stage noise curriculum from the Cascade paper Methods section.

* Stage 1 (≤ 2% of steps): warm up at a lower noise ``p1``.
* Stage 2 (≤ 2% of steps): linear anneal from ``p1`` → ``p2``.
* Stage 3 (the bulk): train at the target noise ``p2``.

We push the schedule into the ``StimMemoryDataset`` via ``set_noise``;
the dataset recompiles its sampler lazily on the next iteration.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CurriculumConfig:
    p1: float
    p2: float
    warmup_frac: float = 0.02   # fraction of steps spent at p1
    anneal_frac: float = 0.02   # fraction of steps spent annealing p1 → p2

    def p_at(self, step: int, total_steps: int) -> float:
        warmup_end = int(self.warmup_frac * total_steps)
        anneal_end = warmup_end + int(self.anneal_frac * total_steps)
        if step < warmup_end:
            return self.p1
        if step < anneal_end:
            t = (step - warmup_end) / max(1, anneal_end - warmup_end)
            return self.p1 + t * (self.p2 - self.p1)
        return self.p2
