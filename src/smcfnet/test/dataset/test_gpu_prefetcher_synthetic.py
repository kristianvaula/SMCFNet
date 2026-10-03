import os
import statistics
import time
import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from smcfnet.dataset.dsec.dsec_collate import dsec_events_collate
from smcfnet.dataset.dsec.dsec_gpu_prefetcher import DSECGPUPrefetcher
from smcfnet.dataset.samplers import InfiniteSampler


class _SyntheticEventsDataset(Dataset):
    def __init__(self, length: int, n_events: int):
        self._len = int(length)
        self._n = int(n_events)

        base = torch.zeros((self._n, 4), dtype=torch.int64)
        base[:, 0] = torch.arange(self._n, dtype=torch.int64) % 640
        base[:, 1] = torch.arange(self._n, dtype=torch.int64) % 480
        base[:, 2] = torch.arange(self._n, dtype=torch.int64)
        base[:, 3] = 1
        self._base = base

    def __len__(self):
        return self._len

    def __getitem__(self, idx: int):
        return {
            "seq_idx": 0,
            "frame_index": int(idx),
            "t_ref_us": 1000,
            "t_start_us": 0,
            "t_end_us": 1000,
            "events": self._base,
        }


class TestGPUPrefetcherSynthetic(unittest.TestCase):
    def test_prefetcher_schema_and_latency(self):
        batch_size = int(os.environ.get("SYNTH_PREFETCH_BATCH", "16"))
        n_events = int(os.environ.get("SYNTH_PREFETCH_EVENTS", "2000"))
        num_workers = int(os.environ.get("SYNTH_PREFETCH_WORKERS", "0"))
        warmup = int(os.environ.get("SYNTH_PREFETCH_WARMUP", "20"))
        iters = int(os.environ.get("SYNTH_PREFETCH_ITERS", "200"))

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

        device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
        prefetch = DSECGPUPrefetcher(
            loader,
            device=device,
            num_bins=5,
            height=480,
            width=640,
            require_targets=False,
        )

        # Warmup
        for _ in range(max(warmup, 0)):
            _ = prefetch.next()

        times_ms = []
        for _ in range(max(iters, 1)):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            batch = prefetch.next()
            if device.type == "cuda":
                torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000.0)

            self.assertIn("events_voxel", batch)
            vox = batch["events_voxel"]
            self.assertIsInstance(vox, torch.Tensor)
            self.assertEqual(tuple(vox.shape), (batch_size, 5, 480, 640))
            self.assertEqual(vox.device.type, device.type)

        mean_ms = statistics.mean(times_ms)
        p95_ms = statistics.quantiles(times_ms, n=20)[18]
        print(
            f"\n[SYNTH prefetch.next() latency] mean={mean_ms:.3f}ms p95≈{p95_ms:.3f}ms "
            f"(B={batch_size}, N={n_events}, workers={num_workers}, device={device.type})"
        )


if __name__ == "__main__":
    unittest.main()