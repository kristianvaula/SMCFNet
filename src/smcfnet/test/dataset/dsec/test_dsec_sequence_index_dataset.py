import random
import statistics
import tempfile
import time
import unittest
from pathlib import Path

import yaml

from smcfnet.config.config import Config
from smcfnet.dataset.dsec import DSECDataset, build_dsec_sequence_index_datasets


def _resolve_dataset_config_path(cfg_path: str) -> Path:
    p = Path(cfg_path)
    if p.exists():
        return p.resolve()

    for parent in Path(__file__).resolve().parents:
        candidate = (parent / p).resolve()
        if candidate.exists():
            return candidate

    return p.resolve()


def _resolve_local_data_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "data"
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate.resolve()
    return None


def _ensure_accessible_dsec_config(cfg_path: Path) -> Path:
    local_root = _resolve_local_data_root()
    if local_root is None:
        return cfg_path

    with cfg_path.open("r") as fh:
        cfg = yaml.safe_load(fh)

    if not isinstance(cfg, dict):
        return cfg_path

    dataset_cfg = cfg.setdefault("dataset", {})
    if not isinstance(dataset_cfg, dict):
        return cfg_path

    dataset_cfg["root"] = str(local_root)

    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    try:
        yaml.safe_dump(cfg, tmp, sort_keys=False)
    finally:
        tmp.close()
    return Path(tmp.name)


class TestDSECSequenceIndexDataset(unittest.TestCase):
    """Extensive correctness + latency checks for the lightweight index dataset.

    These tests are designed to validate the *metadata-only* dataset stage.
    They do not read HDF5 event payloads or decode images.
    """

    @classmethod
    def setUpClass(cls):
        cls.cfg = Config()
        cls.dsec_yaml = _resolve_dataset_config_path(cls.cfg.data.dataset_config)
        cls.dsec_yaml = _ensure_accessible_dsec_config(cls.dsec_yaml)

        if not cls.dsec_yaml.exists():
            raise unittest.SkipTest(f"DSEC dataset config not found: {cls.dsec_yaml}")

        # Build per-sequence datasets for the configured training split.
        try:
            cls.per_sequence = build_dsec_sequence_index_datasets(
                dsec_yaml_path=cls.dsec_yaml,
                split="train",
                dt_ms=float(cls.cfg.data.event_window_ms),
            )
        except (FileNotFoundError, PermissionError, OSError, ValueError) as exc:
            raise unittest.SkipTest(f"DSEC dataset unavailable for integration test: {exc}")

        if len(cls.per_sequence) == 0:
            raise unittest.SkipTest("No train sequences configured in DSEC YAML")

        cls.concat = DSECDataset(cls.per_sequence)

    def test_basic_lengths(self):
        self.assertGreater(len(self.per_sequence), 0)
        self.assertGreater(len(self.concat), 0)

        expected = sum(len(ds) for ds in self.per_sequence)
        self.assertEqual(len(self.concat), expected)

    def test_meta_paths_exist(self):
        for ds in self.per_sequence:
            meta = ds.meta
            self.assertTrue(meta.seq_path.exists())
            self.assertTrue(meta.events_h5_path.exists())
            self.assertTrue(meta.timestamps_path.exists())
            self.assertIsInstance(meta.num_frames, int)
            self.assertGreater(meta.num_frames, 0)

    def test_item_schema_and_window(self):
        ds = self.per_sequence[0]
        item0 = ds[0]
        self.assertSetEqual(
            set(item0.keys()),
            {"seq_idx", "frame_index", "t_ref_us", "t_start_us", "t_end_us"},
        )
        for k in item0:
            self.assertIsInstance(item0[k], int)

        self.assertEqual(item0["t_end_us"], item0["t_ref_us"])
        self.assertLessEqual(item0["t_start_us"], item0["t_end_us"])

        # Check dt_us matches configured window (within rounding to integer us).
        dt_us = int(round(float(self.cfg.data.event_window_ms) * 1000.0))
        self.assertIn(item0["t_end_us"] - item0["t_start_us"], {dt_us, item0["t_end_us"]})

        item_last = ds[-1]
        self.assertEqual(item_last["frame_index"], len(ds) - 1)

    def test_random_access_is_stable(self):
        ds = self.per_sequence[0]
        n = len(ds)
        indices = [0, n // 2, n - 1]
        indices += [random.randrange(0, n) for _ in range(min(10, n))]

        for idx in indices:
            item = ds[idx]
            self.assertEqual(item["frame_index"], idx)
            self.assertGreaterEqual(item["t_end_us"], 0)
            self.assertGreaterEqual(item["t_start_us"], 0)
            self.assertLessEqual(item["t_start_us"], item["t_end_us"])

    def test_latency_per_getitem_ms(self):
        """Measure __getitem__ latency; prints summary and asserts it's not pathological."""
        ds = self.per_sequence[0]
        n = len(ds)
        if n <= 1:
            self.skipTest("Sequence too small to measure latency")

        # Warmup
        for _ in range(20):
            _ = ds[random.randrange(0, n)]

        # Timed
        samples = min(500, n)
        durations_ms = []
        for _ in range(samples):
            idx = random.randrange(0, n)
            t0 = time.perf_counter()
            _ = ds[idx]
            durations_ms.append((time.perf_counter() - t0) * 1000.0)

        mean_ms = statistics.mean(durations_ms)
        p95_ms = statistics.quantiles(durations_ms, n=20)[18]  # approx 95th percentile

        print(
            f"\n[DSECSequenceIndexDataset latency] sequence={ds.sequence} "
            f"mean={mean_ms:.4f}ms p95≈{p95_ms:.4f}ms samples={samples}"
        )

        # Keep threshold very lenient; this dataset should be extremely fast.
        self.assertLess(mean_ms, 20.0)


if __name__ == "__main__":
    unittest.main()