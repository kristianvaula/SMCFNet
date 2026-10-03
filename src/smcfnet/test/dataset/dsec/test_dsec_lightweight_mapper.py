import random
import statistics
import tempfile
import time
import unittest
from pathlib import Path

import yaml

from smcfnet.config.config import Config
from smcfnet.dataset.dsec import DSECDataset, DSECLightweightMapper, build_dsec_sequence_index_datasets


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


class TestDSECLightweightMapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = Config()
        cls.dsec_yaml = _resolve_dataset_config_path(cfg.data.dataset_config)
        cls.dsec_yaml = _ensure_accessible_dsec_config(cls.dsec_yaml)
        if not cls.dsec_yaml.exists():
            raise unittest.SkipTest(f"DSEC dataset config not found: {cls.dsec_yaml}")

        try:
            cls.per_sequence = build_dsec_sequence_index_datasets(
                dsec_yaml_path=cls.dsec_yaml,
                split="train",
                dt_ms=float(cfg.data.event_window_ms),
            )
        except (FileNotFoundError, PermissionError, OSError, ValueError) as exc:
            raise unittest.SkipTest(f"DSEC dataset unavailable for integration test: {exc}")

        if len(cls.per_sequence) == 0:
            raise unittest.SkipTest("No train sequences configured")

        cls.dataset = DSECDataset(cls.per_sequence)
        cls.mapper = DSECLightweightMapper(cls.per_sequence, load_events=True, load_tracks=False)

    def test_mapper_returns_tensor_events(self):
        idx = random.randrange(0, len(self.dataset))
        item = self.dataset[idx]
        mapped = self.mapper(item)
        self.assertIn("events", mapped)
        ev = mapped["events"]
        self.assertTrue(hasattr(ev, "shape"))
        self.assertEqual(ev.ndim, 2)
        self.assertEqual(ev.shape[1], 4)

    def test_mapper_latency_ms(self):
        n = len(self.dataset)
        if n <= 1:
            self.skipTest("Dataset too small")

        # warmup
        for _ in range(10):
            _ = self.mapper(self.dataset[random.randrange(0, n)])

        samples = min(200, n)
        durations_ms = []
        for _ in range(samples):
            t0 = time.perf_counter()
            _ = self.mapper(self.dataset[random.randrange(0, n)])
            durations_ms.append((time.perf_counter() - t0) * 1000.0)

        mean_ms = statistics.mean(durations_ms)
        p95_ms = statistics.quantiles(durations_ms, n=20)[18]
        print(f"\n[DSECLightweightMapper latency] mean={mean_ms:.2f}ms p95≈{p95_ms:.2f}ms samples={samples}")

        # Very lenient: HDF5 slicing may vary widely on shared filesystems.
        self.assertLess(mean_ms, 500.0)


if __name__ == "__main__":
    unittest.main()