"""Sampling throughput benchmark.

Times Stim's CompiledDetectorSampler at the noise levels and batch sizes we
actually train at. Run on the login node before launching slurm jobs to
verify multi-worker DataLoader can saturate the GPU rather than starve it.
"""

from __future__ import annotations

import time

from cascade.codes.surface import SurfaceCode


def bench(distance: int, rounds: int, p: float, batch: int, n_iters: int = 50) -> None:
    code = SurfaceCode(distance=distance)
    sampler = code.make_circuit(p=p, rounds=rounds).compile_detector_sampler(seed=0)

    # Warm-up
    for _ in range(3):
        sampler.sample(shots=batch, separate_observables=True, bit_packed=False)

    t0 = time.time()
    total = 0
    for _ in range(n_iters):
        sampler.sample(shots=batch, separate_observables=True, bit_packed=False)
        total += batch
    dt = time.time() - t0
    print(f"surface d={distance:2d}, R={rounds:2d}, p={p:.4f}, batch={batch:4d}: "
          f"{total / dt:8.0f} shots/s   ({dt*1000/n_iters:.1f} ms/batch)")


def main() -> None:
    print("Throughput benchmark — single-thread, login node\n")
    for d, R in [(3, 3), (5, 5), (7, 7), (9, 9)]:
        for batch in [256, 512, 1024]:
            bench(distance=d, rounds=R, p=0.003, batch=batch)
        print()


if __name__ == "__main__":
    main()
