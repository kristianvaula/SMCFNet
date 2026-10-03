import os
import statistics
import time
import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from smcfnet.dataset.dsec.dsec_collate import dsec_events_collate
from smcfnet.dataset.samplers import InfiniteSampler


class _SyntheticEventsDataset(Dataset):
    def __init__(self, length: int, n_events: int):
        self._len = int(length)
        self._n = int(n_events)

        # Pre-generate one tensor and clone on access to simulate per-sample output
        # without triggering heavy allocations.
        base = torch.zeros((self._n, 4), dtype=torch.int64)
        base[:, 0] = torch.arange(self._n, dtype=torch.int64) % 640
        base[:, 1] = torch.arange(self._n, dtype=torch.int64) % 480
        base[:, 2] = torch.arange(self._n, dtype=torch.int64)
        base[:, 3] = 1
        self._base = base

    def __len__(self):
        return self._len

    def __getitem__(self, idx: int):
        # Mimic mapper output keys used by collate
        return {
            "seq_idx": 0,
            "frame_index": int(idx),
            "t_ref_us": 1000,
            "t_start_us": 0,
            "t_end_us": 1000,
            "events": self._base,  # already contiguous int64
        }


class TestBatchNextFetchingSynthetic(unittest.TestCase):
    def test_next_batch_latency_ms(self):
        # This is a stable micro-benchmark for *our* batching stack.
        # It excludes HDF5 and filesystem jitter.
        batch_size = int(os.environ.get("SYNTH_BENCH_BATCH", "16"))
        n_events = int(os.environ.get("SYNTH_BENCH_EVENTS", "2000"))
        num_workers = int(os.environ.get("SYNTH_BENCH_WORKERS", "0"))
        warmup = int(os.environ.get("SYNTH_BENCH_WARMUP", "20"))
        iters = int(os.environ.get("SYNTH_BENCH_ITERS", "200"))

        ds = _SyntheticEventsDataset(length=10_000, n_events=n_events)
        sampler = InfiniteSampler(len(ds), shuffle=True, seed=123)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=True,
            collate_fn=dsec_events_collate,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
        )

        it = iter(loader)
        for _ in range(max(warmup, 0)):
            _ = next(it)

        times_ms = []
        for _ in range(max(iters, 1)):
            t0 = time.perf_counter()
            _ = next(it)
            times_ms.append((time.perf_counter() - t0) * 1000.0)

        mean_ms = statistics.mean(times_ms)
        p95_ms = statistics.quantiles(times_ms, n=20)[18]
        print(
            f"\n[SYNTH next() latency] mean={mean_ms:.3f}ms p95≈{p95_ms:.3f}ms "
            f"(B={batch_size}, N={n_events}, workers={num_workers})"
        )

        # Optional enforcement (for regression guarding on a fixed machine).
        limit = os.environ.get("SYNTH_BENCH_ASSERT_P95_MS")
        if limit is not None:
            self.assertLessEqual(p95_ms, float(limit))


if __name__ == "__main__":
    unittest.main()