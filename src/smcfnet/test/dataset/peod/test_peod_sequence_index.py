from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from smcfnet.dataset.peod import build_peod_sequence_index_datasets

from ._fixtures import write_peod_coco_json, write_peod_timestamps


class TestPEODSequenceIndex(unittest.TestCase):
    def test_index_dataset_uses_frame_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "peod"
            root.mkdir(parents=True, exist_ok=True)
            ann_path = root / "annotations" / "coco_0111_last" / "train" / "challenge" / "sequence_001.json"
            write_peod_coco_json(
                ann_path,
                file_names=["sequence_001_000001.png", "sequence_001_000002.png"],
            )
            write_peod_timestamps(
                root / "timestamp" / "train" / "sequence_001.csv",
                [100000, 133333],
            )

            dataset_yaml = root / "peod.yaml"
            dataset_yaml.write_text(
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
                        "    - sequence_001",
                    ]
                ),
                encoding="utf-8",
            )

            datasets = build_peod_sequence_index_datasets(
                peod_yaml_path=dataset_yaml,
                split="train",
                dt_ms=50.0,
            )
            self.assertEqual(len(datasets), 1)
            ds = datasets[0]
            self.assertEqual(len(ds), 2)

            item = ds[0]
            self.assertEqual(set(item.keys()), {"seq_idx", "frame_index", "t_ref_us", "t_start_us", "t_end_us"})
            self.assertEqual(item["seq_idx"], 0)
            self.assertEqual(item["frame_index"], 0)
            self.assertEqual(item["t_ref_us"], 100000)
            self.assertEqual(item["t_end_us"], 100000)
            self.assertEqual(item["t_start_us"], 50000)

            last = ds.pull_item(1)
            self.assertEqual(last["frame_index"], 1)
            self.assertEqual(last["t_ref_us"], 133333)