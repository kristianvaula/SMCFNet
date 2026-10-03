import unittest

import torch

from smcfnet.dataset.dsec.dsec_collate import (
    DSECTrainCollate,
    dsec_rgb_train_collate,
    dsec_streaming_train_collate,
)


class TestDSECCollateAdditional(unittest.TestCase):
    def _base_sample(self, frame_index: int = 0):
        return {
            "seq_idx": 1,
            "frame_index": frame_index,
            "t_ref_us": 2000,
            "t_start_us": 1000,
            "t_end_us": 2000,
            "events": torch.tensor([[0, 1, 1500, 1]], dtype=torch.int64),
        }

    def test_train_collate_rejects_invalid_timestamp_unit(self):
        collate = DSECTrainCollate(use_events=True, timestamp_unit="sec")
        with self.assertRaises(ValueError):
            collate([self._base_sample(0)])

    def test_rgb_train_collate_stacks_image_and_pads_targets(self):
        batch = [
            {
                "seq_idx": 0,
                "frame_index": 3,
                "t_ref_us": 2000,
                "t_start_us": 1000,
                "t_end_us": 2000,
                "image": torch.ones((3, 8, 8), dtype=torch.float32),
                "targets": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float32),
            },
            {
                "seq_idx": 0,
                "frame_index": 4,
                "t_ref_us": 3000,
                "t_start_us": 2000,
                "t_end_us": 3000,
                "image": torch.zeros((3, 8, 8), dtype=torch.float32),
                "targets": torch.zeros((0, 5), dtype=torch.float32),
            },
        ]

        out = dsec_rgb_train_collate(batch)
        self.assertEqual(tuple(out["image"].shape), (2, 3, 8, 8))
        self.assertEqual(tuple(out["targets"].shape), (2, 1, 5))
        self.assertTrue(torch.equal(out["frame_index"], torch.tensor([3, 4], dtype=torch.int64)))

    def test_streaming_train_collate_stacks_flow_gt_and_preserves_timestamps(self):
        batch = [
            {
                **self._base_sample(0),
                "events": torch.tensor([[0, 1, 1001, 1]], dtype=torch.int64),
                "flow_gt": torch.ones((3, 4, 4), dtype=torch.float32),
            },
            {
                **self._base_sample(1),
                "events": torch.tensor([[1, 2, 2002, -1]], dtype=torch.int64),
                "flow_gt": torch.zeros((3, 4, 4), dtype=torch.float32),
            },
        ]

        out = dsec_streaming_train_collate(batch)
        self.assertTrue(torch.equal(out["events"][:, 2], torch.tensor([1001, 2002], dtype=torch.int64)))
        self.assertEqual(tuple(out["flow_gt"].shape), (2, 3, 4, 4))

    def test_train_collate_without_events_for_rgb_only_path(self):
        collate = DSECTrainCollate(use_events=False, timestamp_unit="ms")
        batch = [
            {
                "seq_idx": 0,
                "frame_index": 0,
                "t_ref_us": 1000,
                "t_start_us": 0,
                "t_end_us": 1000,
                "image": torch.ones((3, 4, 4), dtype=torch.float32),
                "targets": torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]], dtype=torch.float32),
            }
        ]

        out = collate(batch)
        self.assertNotIn("events", out)
        self.assertIn("image", out)
        self.assertIn("targets", out)
        self.assertEqual(tuple(out["targets"].shape), (1, 1, 5))


if __name__ == "__main__":
    unittest.main()