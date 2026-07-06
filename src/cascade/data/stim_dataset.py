"""Online sampling of Stim memory experiments for training the decoder.

The dataset is a `torch.utils.data.IterableDataset` so multi-worker
DataLoader can saturate sampling throughput. Each worker compiles its
own `CompiledDetectorSampler` and re-samples on-the-fly. When the
curriculum changes the noise level, the training loop swaps the sampler
via ``set_noise``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import stim
import torch
from torch.utils.data import IterableDataset

from cascade.codes.base import Code


@dataclass
class StimSampleSpec:
    """Describes how a worker should produce one batch.

    The worker yields ``(detection_events, observable_flips)`` tensors of
    shape ``(batch, num_detectors)`` and ``(batch, num_observables)``. We
    stream these as torch tensors (CPU) and let the trainer move them to
    GPU.
    """

    code: Code
    rounds: int
    p: float
    batch_size: int


class StimMemoryDataset(IterableDataset):
    """Infinite stream of Stim detection events at a given noise level.

    Set ``set_noise`` from the trainer to drive a noise curriculum. Each
    DataLoader worker maintains its own compiled sampler so changes only
    take effect after the next worker recompile (which happens lazily on
    the next ``__iter__`` invocation, or eagerly via ``recompile``).
    """

    def __init__(self, code: Code, rounds: int, p: float, batch_size: int,
                 seed: int | None = None) -> None:
        super().__init__()
        self.code = code
        self.rounds = rounds
        self._p = p
        self.batch_size = batch_size
        self._seed = seed

    @property
    def p(self) -> float:
        return self._p

    def set_noise(self, p: float) -> None:
        """Update the target noise level. Effective on next worker iter."""
        self._p = p

    def _make_sampler(self) -> stim.CompiledDetectorSampler:
        circuit = self.code.make_circuit(p=self._p, rounds=self.rounds)
        return circuit.compile_detector_sampler(seed=self._seed)

    def __iter__(self):
        sampler = self._make_sampler()
        last_p = self._p
        while True:
            if last_p != self._p:
                sampler = self._make_sampler()
                last_p = self._p
            det, obs = sampler.sample(
                shots=self.batch_size, separate_observables=True, bit_packed=False
            )
            det_t = torch.from_numpy(det.astype(np.float32))
            obs_t = torch.from_numpy(obs.astype(np.float32))
            yield det_t, obs_t
