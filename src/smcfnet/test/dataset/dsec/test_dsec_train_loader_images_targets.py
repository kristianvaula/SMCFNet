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


class TestDSECTrainLoaderImagesTargets(unittest.TestCase):
    def test_batch_contains_image_and_targets_when_enabled(self):
        cfg = Config()
        cfg.data.dataset_config = _ensure_accessible_dsec_config(cfg.data.dataset_config)
        cfg.training.batch_size = 2
        cfg.data.num_workers = 0
        cfg.data.persistent_workers = False
        cfg.data.drop_last = True

        cfg.data.mapper_load_images = True
        cfg.data.mapper_prefer_distorted = True
        cfg.data.mapper_load_tracks = True
        cfg.data.mapper_build_targets = True

        try:
            loader = build_train_loader_cfg(cfg)
        except (FileNotFoundError, PermissionError, OSError, ValueError) as exc:
            raise unittest.SkipTest(f"DSEC loader unavailable in this environment: {exc}")


        batch = next(iter(loader))
        self.assertIn("image", batch)
        self.assertIn("targets", batch)
        self.assertIn("events", batch)
        self.assertIn("events_offsets", batch)

        img = batch["image"]
        targets = batch["targets"]

        self.assertIsInstance(img, torch.Tensor)
        self.assertEqual(tuple(img.shape[:2]), (cfg.training.batch_size, 3))
        self.assertEqual(tuple(img.shape[-2:]), (cfg.data.image_height, cfg.data.image_width))

        self.assertIsInstance(targets, torch.Tensor)
        self.assertEqual(targets.ndim, 3)
        self.assertEqual(tuple(targets.shape[:1]), (cfg.training.batch_size,))
        self.assertEqual(int(targets.shape[-1]), 5)


if __name__ == "__main__":
    unittest.main()