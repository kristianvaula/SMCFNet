import time
import unittest
from pathlib import Path

import tempfile

import yaml

import torch

from smcfnet.config.config import Config
from smcfnet.model.build import build_train_loader_cfg


def _resolve_local_data_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "data"
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate.resolve()
    return None


def _ensure_accessible_dsec_config(cfg_path: str) -> str:
    path = Path(cfg_path)
    if not path.exists():
        return cfg_path

    local_root = _resolve_local_data_root()
    if local_root is None:
        return str(path.resolve())

    with path.open("r") as fh:
        cfg = yaml.safe_load(fh)

    if not isinstance(cfg, dict):
        return str(path.resolve())

    dataset_cfg = cfg.setdefault("dataset", {})
    if not isinstance(dataset_cfg, dict):
        return str(path.resolve())

    dataset_cfg["root"] = str(local_root)

    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    try:
        yaml.safe_dump(cfg, tmp, sort_keys=False)
    finally:
        tmp.close()
    return tmp.name


class TestDSECTrainLoaderE2E(unittest.TestCase):
    def test_next_batch_latency_and_schema(self):
        cfg = Config()
        cfg.data.dataset_config = _ensure_accessible_dsec_config(cfg.data.dataset_config)

        # Keep the test lightweight and less flaky in CI.
        cfg.training.batch_size = 2
        cfg.data.num_workers = 2
        cfg.data.persistent_workers = False
        cfg.data.prefetch_factor = 2
        cfg.data.drop_last = True

        try:
            loader = build_train_loader_cfg(cfg)
        except (FileNotFoundError, PermissionError, OSError, ValueError) as exc:
            raise unittest.SkipTest(f"DSEC loader unavailable in this environment: {exc}")

        it = iter(loader)

        # Warmup: first batch may include HDF5 open.
        t0 = time.perf_counter()
        batch = next(it)
        t1 = time.perf_counter()

        self.assertIn("events", batch)
        self.assertIn("events_offsets", batch)
        self.assertIn("events_batch_idx", batch)

        self.assertIsInstance(batch["events"], torch.Tensor)
        self.assertEqual(batch["events"].ndim, 2)
        self.assertEqual(batch["events"].shape[1], 4)

        self.assertIsInstance(batch["events_offsets"], torch.Tensor)
        self.assertEqual(batch["events_offsets"].dtype, torch.int64)
        self.assertEqual(batch["events_offsets"].numel(), cfg.training.batch_size + 1)

        self.assertIsInstance(batch["events_batch_idx"], torch.Tensor)
        self.assertEqual(batch["events_batch_idx"].dtype, torch.int64)
        self.assertEqual(batch["events_batch_idx"].numel(), batch["events"].shape[0])

        # Basic sanity: offsets last equals packed length.
        self.assertEqual(int(batch["events_offsets"][-1].item()), int(batch["events"].shape[0]))

        # Time a couple more batches (should be steadier than warmup).
        times = [t1 - t0]
        for _ in range(2):
            t0 = time.perf_counter()
            _ = next(it)
            times.append(time.perf_counter() - t0)

        # Print a compact timing summary for local runs.
        times_ms = [t * 1000.0 for t in times]
        print(
            f"DSEC loader next() latencies: "
            f"{', '.join(f'{t:.2f}ms' for t in times_ms)}"
        )

        # Don’t assert on absolute timing (machine-dependent), just record it.
        self.assertGreaterEqual(len(times), 3)


if __name__ == "__main__":
    unittest.main()