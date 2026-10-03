import statistics
import time
import unittest

import torch

from smcfnet.dataset.dsec.dsec_collate import (
    dsec_events_collate,
    dsec_streaming_events_collate,
)


class TestDSECCollateFast(unittest.TestCase):
    def _make_sample(self, n_events: int, seq_idx: int = 0, frame_index: int = 0):
        # int64 matches current slicer output
        events = torch.zeros((n_events, 4), dtype=torch.int64)
        events[:, 0] = torch.arange(n_events, dtype=torch.int64) % 640  # x
        events[:, 1] = torch.arange(n_events, dtype=torch.int64) % 480  # y
        events[:, 2] = torch.arange(n_events, dtype=torch.int64)  # t_ms
        events[:, 3] = 1  # p
        return {
            "seq_idx": int(seq_idx),
            "frame_index": int(frame_index),
            "t_ref_us": 1000,
            "t_start_us": 0,
            "t_end_us": 1000,
            "events": events,
        }

    def test_collate_schema_and_offsets(self):
        batch = [self._make_sample(3, frame_index=0), self._make_sample(0, frame_index=1), self._make_sample(5, frame_index=2)]
        out = dsec_events_collate(batch)

        self.assertIn("events", out)
        self.assertIn("events_offsets", out)
        self.assertIn("events_batch_idx", out)

        self.assertEqual(tuple(out["events"].shape), (8, 4))
        self.assertTrue(torch.equal(out["events_offsets"], torch.tensor([0, 3, 3, 8], dtype=torch.int64)))
        self.assertEqual(out["events_batch_idx"].numel(), 8)
        self.assertEqual(int(out["events_batch_idx"][:3].max().item()), 0)
        self.assertEqual(int(out["events_batch_idx"][3:].min().item()), 2)

    def test_streaming_collate_preserves_microsecond_timestamps(self):
        batch = [self._make_sample(2, frame_index=0), self._make_sample(3, frame_index=1)]
        batch[0]["events"][:, 2] = torch.tensor([1001, 1017], dtype=torch.int64)
        batch[1]["events"][:, 2] = torch.tensor([2003, 2009, 2011], dtype=torch.int64)

        out = dsec_streaming_events_collate(batch)

        self.assertTrue(
            torch.equal(
                out["events"][:, 2],
                torch.tensor([1001, 1017, 2003, 2009, 2011], dtype=torch.int64),
            )
        )
        self.assertTrue(torch.equal(out["events_offsets"], torch.tensor([0, 2, 5], dtype=torch.int64)))

    def test_collate_latency_ms(self):
        # Synthetic perf sanity: collate should be comfortably sub-ms to a few ms.
        # Keep this lenient to avoid flakiness on slower CI nodes.
        batch_size = 16
        n_events_each = 2000
        batch = [self._make_sample(n_events_each, frame_index=i) for i in range(batch_size)]

        # warmup
        for _ in range(10):
            _ = dsec_events_collate(batch)

        times_ms = []
        for _ in range(50):
            t0 = time.perf_counter()
            _ = dsec_events_collate(batch)
            times_ms.append((time.perf_counter() - t0) * 1000.0)

        mean_ms = statistics.mean(times_ms)
        p95_ms = statistics.quantiles(times_ms, n=20)[18]
        print(f"\n[dsec_events_collate latency] mean={mean_ms:.3f}ms p95≈{p95_ms:.3f}ms (B={batch_size}, N={n_events_each})")

        self.assertLess(mean_ms, 50.0)


if __name__ == "__main__":
    unittest.main()
