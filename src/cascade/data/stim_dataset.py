"""Online sampling of Stim memory experiments for training the decoder.

The dataset is a `torch.utils.data.IterableDataset` so multi-worker
DataLoader can saturate sampling throughput. Each worker compiles its
own `CompiledDetectorSampler` and re-samples on-the-fly. When the
curriculum changes the noise level, the training loop swaps the sampler
via ``set_noise``. For mixed-p training (iter-7), ``set_mixed`` instead
flips the dataset into a mode where a fresh p is drawn per yielded batch
from a small cache of compiled samplers (see ``PDist``).
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


@dataclass
class PDist:
    """Spec for mixed-p sampling: draw a fresh noise level per yielded batch.

    ``grid`` is the fixed list of noise levels each draw snaps to; each grid
    point gets its own cached compiled sampler with a seed derived from the
    dataset's base seed (see ``StimMemoryDataset.set_mixed``), so the
    distinct-p streams are independent rather than phase-locked. ``mode``
    controls how a draw is produced:

    - ``"log-uniform"``: draw ``log(p) ~ Uniform(log(p_min), log(p_max))``,
      snap to the nearest grid point in log-space.
    - ``"uniform"``: draw ``p ~ Uniform(p_min, p_max)``, snap to the nearest
      grid point in linear space.
    - ``"list"``: ignore ``p_min``/``p_max``; sample an index directly from
      ``grid`` (uniformly, or weighted by ``weights`` if given).
    """

    mode: str
    grid: tuple[float, ...]
    p_min: float | None = None
    p_max: float | None = None
    weights: tuple[float, ...] | None = None
    seed: int | None = None


class StimMemoryDataset(IterableDataset):
    """Infinite stream of Stim detection events at a given noise level.

    Set ``set_noise`` from the trainer to drive a noise curriculum. Each
    DataLoader worker maintains its own compiled sampler so changes only
    take effect after the next worker recompile (which happens lazily on
    the next ``__iter__`` invocation, or eagerly via ``recompile``).

    ``set_mixed`` (or passing ``p_dist`` at construction) instead switches
    to mixed-p mode: every yielded batch draws its own p from ``p_dist``,
    using a small dict cache of compiled samplers keyed by (snapped) p so
    the number of stim compiles is bounded by the grid size, not the step
    count. When ``p_dist`` is ``None`` (the default) the class behaves
    exactly as before mixed-p support was added.
    """

    def __init__(self, code: Code, rounds: int, p: float, batch_size: int,
                 seed: int | None = None, p_dist: PDist | None = None) -> None:
        super().__init__()
        self.code = code
        self.rounds = rounds
        self._p = p
        self.batch_size = batch_size
        self._seed = seed
        self._cache: dict[float, stim.CompiledDetectorSampler] = {}
        self._p_dist: PDist | None = None
        self._grid: tuple[float, ...] = ()
        self._grid_index: dict[float, int] = {}
        self._grid_weights: list[float] | None = None
        self._p_rng: np.random.Generator | None = None
        self.last_p: float | None = None
        if p_dist is not None:
            self.set_mixed(p_dist)

    @property
    def p(self) -> float:
        return self._p

    @property
    def sampler_cache_size(self) -> int:
        """Number of distinct compiled samplers currently cached."""
        return len(self._cache)

    def set_noise(self, p: float) -> None:
        """Update the target noise level. Effective on next worker iter."""
        if p != self._p:
            # Single-p callers (the anneal loop) change p once per distinct
            # step value; dropping the stale sampler keeps the cache O(1),
            # matching the pre-cache behaviour of recompiling on change.
            self._cache.clear()
        self._p = p

    def set_mixed(self, p_dist: PDist) -> None:
        """Switch the dataset into mixed-p mode (sibling to ``set_noise``).

        Effective on the next ``__iter__`` call. Each grid point gets its
        own cached compiled sampler, seeded ``base_seed + grid_index`` so
        the distinct-p streams are independent, not phase-locked.
        """
        grid = tuple(float(x) for x in p_dist.grid)
        if not grid:
            raise ValueError("p_dist.grid must be non-empty")
        self._p_dist = p_dist
        self._grid = grid
        self._grid_index = {v: i for i, v in enumerate(grid)}
        if p_dist.weights is not None:
            w = np.asarray(p_dist.weights, dtype=np.float64)
            if len(w) != len(grid):
                raise ValueError("p_dist.weights must match p_dist.grid length")
            self._grid_weights = (w / w.sum()).tolist()
        else:
            self._grid_weights = None
        self._p_rng = np.random.default_rng(p_dist.seed)
        # Root-cause fix (iter-7): samplers compiled during the single-p
        # warmup/anneal stage (one per distinct annealed p) must not linger
        # in the cache after switching to mixed mode, or sampler_cache_size
        # would exceed len(grid) and violate the "cache size <= p-grid-points"
        # smoke acceptance check. Recompiling the grid's samplers afterward
        # costs seconds — negligible against the run's total throughput.
        self._cache.clear()

    def _seed_for(self, p: float) -> int | None:
        if self._seed is None:
            return None
        idx = self._grid_index.get(p)
        if idx is not None:
            return self._seed + idx
        return self._seed

    def _get_sampler(self, p: float) -> stim.CompiledDetectorSampler:
        """Return the cached compiled sampler for noise level ``p``,
        compiling and storing it on a cache miss.
        """
        cached = self._cache.get(p)
        if cached is not None:
            return cached
        circuit = self.code.make_circuit(p=p, rounds=self.rounds)
        sampler = circuit.compile_detector_sampler(seed=self._seed_for(p))
        self._cache[p] = sampler
        return sampler

    def _snap(self, value: float, log_space: bool) -> float:
        """Snap a continuous draw to the nearest grid point."""
        if log_space:
            target = math.log(value)
            idx = min(range(len(self._grid)),
                      key=lambda i: abs(math.log(self._grid[i]) - target))
        else:
            idx = min(range(len(self._grid)), key=lambda i: abs(self._grid[i] - value))
        return self._grid[idx]

    def _draw_p(self) -> float:
        dist = self._p_dist
        assert dist is not None and self._p_rng is not None
        if dist.mode == "list":
            idx = int(self._p_rng.choice(len(self._grid), p=self._grid_weights))
            return self._grid[idx]
        if dist.mode == "log-uniform":
            draw = math.exp(self._p_rng.uniform(math.log(dist.p_min), math.log(dist.p_max)))
            return self._snap(draw, log_space=True)
        if dist.mode == "uniform":
            draw = self._p_rng.uniform(dist.p_min, dist.p_max)
            return self._snap(draw, log_space=False)
        raise ValueError(f"unknown p_dist mode: {dist.mode!r}")

    def _sample_batch(self, sampler: stim.CompiledDetectorSampler):
        det, obs = sampler.sample(
            shots=self.batch_size, separate_observables=True, bit_packed=False
        )
        det_t = torch.from_numpy(det.astype(np.float32))
        obs_t = torch.from_numpy(obs.astype(np.float32))
        return det_t, obs_t

    def __iter__(self):
        if self._p_dist is not None:
            while True:
                p_now = self._draw_p()
                self.last_p = p_now
                sampler = self._get_sampler(p_now)
                yield self._sample_batch(sampler)

        # Single-p branch (untouched behaviour): recompile only on set_noise.
        sampler = self._get_sampler(self._p)
        last_p = self._p
        while True:
            if last_p != self._p:
                sampler = self._get_sampler(self._p)
                last_p = self._p
            yield self._sample_batch(sampler)
