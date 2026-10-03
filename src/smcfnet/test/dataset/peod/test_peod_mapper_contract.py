from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from smcfnet.dataset.peod import PEODLightweightMapper, build_peod_sequence_index_datasets

from ._fixtures import write_peod_coco_json, write_peod_event_dat, write_peod_image, write_peod_timestamps


class TestPEODMapperContract(unittest.TestCase):
    def test_mapper_emits_expected_batch_contract(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "peod"
            root.mkdir(parents=True, exist_ok=True)
            seq_name = "sequence_001"
            ann_path = root / "annotations" / "coco_0111_last" / "train" / "challenge" / f"{seq_name}.json"
            rgb_dir = root / "rgb" / "train" / seq_name
            event_path = root / "events" / "train" / f"{seq_name}.dat"
            ts_path = root / "timestamp" / "train" / f"{seq_name}.csv"

            write_peod_coco_json(
                ann_path,
                file_names=[f"{seq_name}_000001.png", f"{seq_name}_000002.png"],
            )
            write_peod_timestamps(ts_path, [100000, 133333])
            write_peod_image(rgb_dir / f"{seq_name}_000001.png", color=(12, 34, 56))
            write_peod_image(rgb_dir / f"{seq_name}_000002.png", color=(90, 80, 70))
            write_peod_event_dat(
                event_path,
                [
                    (60000, 10, 11, 1),
                    (70000, 20, 21, -1),
                    (90000, 30, 31, 1),
                ],
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
                        f"    - {seq_name}",
                    ]
                ),
                encoding="utf-8",
            )

            per_sequence = build_peod_sequence_index_datasets(
                peod_yaml_path=dataset_yaml,
                split="train",
                dt_ms=50.0,
            )
            mapper = PEODLightweightMapper(
                per_sequence,
                load_events=True,
                load_images=True,
                build_targets=True,
            )

            mapped = mapper(per_sequence[0][0])
            self.assertEqual(
                set(mapped.keys()),
                {"seq_idx", "frame_index", "t_start_us", "t_end_us", "t_ref_us", "sequence_reset", "events", "image", "targets"},
            )
            self.assertTrue(mapped["sequence_reset"])
            self.assertIsInstance(mapped["events"], torch.Tensor)
            self.assertEqual(mapped["events"].shape[1], 4)
            self.assertGreater(mapped["events"].shape[0], 0)
            self.assertTrue(torch.all(mapped["events"][:, 0].ge(0)))
            self.assertTrue(torch.all(mapped["events"][:, 0].lt(1280)))
            self.assertTrue(torch.all(mapped["events"][:, 1].ge(0)))
            self.assertTrue(torch.all(mapped["events"][:, 1].lt(720)))
            self.assertTrue(torch.all(mapped["events"][:, 3].abs().eq(1)))

            self.assertEqual(tuple(mapped["image"].shape), (3, 720, 1280))
            self.assertEqual(mapped["image"].dtype, torch.uint8)
            self.assertEqual(tuple(mapped["targets"].shape), (2, 5))

    def test_mapper_runtime_caches_are_recreated_when_missing(self):
        mapper = PEODLightweightMapper([], load_events=False, load_images=False)
        delattr(mapper, "_flow_cache")

        mapper.reset_caches()
        mapper._reset_if_needed()

        self.assertTrue(hasattr(mapper, "_flow_cache"))
        self.assertEqual(mapper._flow_cache, {})
