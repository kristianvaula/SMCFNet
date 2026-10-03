from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from smcfnet.dataset.peod import PEOD_CATEGORIES, load_peod_coco_annotations

from ._fixtures import write_peod_coco_json


class TestPEODAnnotations(unittest.TestCase):
    def test_load_and_convert_frame_targets(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = Path(tmp_dir) / "sequence_001.json"
            write_peod_coco_json(
                json_path,
                file_names=["sequence_001_000002.png", "sequence_001_000001.png"],
            )

            frames = load_peod_coco_annotations(json_path)
            self.assertEqual(PEOD_CATEGORIES, ("car", "person", "bus", "truck", "2-wheeler", "3-wheeler"))
            self.assertEqual([frame.file_name for frame in frames], ["sequence_001_000001.png", "sequence_001_000002.png"])

            first_targets = frames[0].targets
            self.assertEqual(tuple(first_targets.shape), (1, 5))
            self.assertEqual(first_targets[0, 0].item(), 3.0)
            self.assertAlmostEqual(first_targets[0, 1].item(), 87.5)
            self.assertAlmostEqual(first_targets[0, 2].item(), 100.0)
            self.assertAlmostEqual(first_targets[0, 3].item(), 75.0)
            self.assertAlmostEqual(first_targets[0, 4].item(), 80.0)

            self.assertEqual(tuple(frames[1].targets.shape), (2, 5))
            self.assertListEqual(frames[1].targets[:, 0].tolist(), [0.0, 1.0])