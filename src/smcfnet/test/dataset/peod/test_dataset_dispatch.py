from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from smcfnet.config.config import Config, update_config_from_dict
from smcfnet.model.build import build_train_loader_cfg

from ._fixtures import write_peod_coco_json, write_peod_timestamps


class TestPEODDatasetDispatch(unittest.TestCase):
    def test_build_train_loader_uses_peod_components(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "peod"
            root.mkdir(parents=True, exist_ok=True)
            seq_name = "sequence_001"
            ann_path = root / "annotations" / "coco_0111_last" / "train" / "challenge" / f"{seq_name}.json"
            ts_path = root / "timestamp" / "train" / f"{seq_name}.csv"
            write_peod_coco_json(ann_path, file_names=[f"{seq_name}_000001.png"])
            write_peod_timestamps(ts_path, [100000])

            dataset_cfg = root / "dataset.yaml"
            dataset_cfg.write_text(
                "\n".join(
                    [
                        "dataset:",
                        "  name: PEOD",
                        f"  root: {root}",
                        "  image_height: 720",
                        "  image_width: 1280",
                        "  train_subsets:",
                        "    - challenge",
                        "  train_sequences:",
                        f"    - {seq_name}",
                    ]
                ),
                encoding="utf-8",
            )

            cfg = Config()
            update_config_from_dict(
                cfg,
                {
                    "data": {
                        "dataset_config": str(dataset_cfg),
                        "event_window_ms": 50,
                        "image_height": 720,
                        "image_width": 1280,
                        "num_workers": 0,
                        "pin_memory": False,
                        "drop_last": False,
                        "mapper_load_events": False,
                        "mapper_load_tracks": True,
                        "mapper_load_images": True,
                        "mapper_prefer_distorted": False,
                        "mapper_build_targets": True,
                        "mapper_load_flow": False,
                    },
                    "model": {
                        "input_mode": "rgb",
                        "size": "s",
                        "num_classes": 6,
                        "pretrained": False,
                        "mcm": {"enabled": False},
                        "edum": {"enabled": False},
                        "cmm": {"enabled": False},
                    },
                    "training": {
                        "batch_size": 1,
                        "num_epochs": 1,
                        "max_total_steps": 1,
                        "use_amp": False,
                        "use_mosaic": False,
                        "use_mixup": False,
                    },
                },
            )
            components = build_train_loader_cfg(cfg, return_components=True)

            self.assertEqual(components["dataset"].__class__.__name__, "PEODDataset")
            self.assertEqual(components["per_sequence"][0].meta.root, root)
            self.assertEqual(components["mapper"].__class__.__name__, "PEODLightweightMapper")
            self.assertFalse(components["event_pipeline"]["mapper_load_events"])