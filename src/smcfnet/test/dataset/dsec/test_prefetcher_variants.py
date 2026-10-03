import unittest

import pytest
import torch

from smcfnet.dataset.dsec.prefetcher import (
    DSECGPUPrefetcher,
    DSECRGBGPUPrefetcher,
    DSECStreamingGPUPrefetcher,
)


class TestPrefetcherVariants(unittest.TestCase):
    def test_rgb_prefetcher_moves_tensors_on_cpu_path(self):
        batch = {
            "image": torch.randn(2, 3, 8, 8),
            "targets": torch.zeros(2, 1, 5),
            "frame_index": torch.tensor([0, 1], dtype=torch.int64),
        }
        prefetch = DSECRGBGPUPrefetcher([batch], device=torch.device("cpu"), require_targets=True)
        out = prefetch.next()
        self.assertEqual(out["image"].device.type, "cpu")
        self.assertEqual(out["targets"].device.type, "cpu")
        self.assertEqual(prefetch.prepared_representation, "image")
        with self.assertRaises(StopIteration):
            prefetch.next()

    def test_streaming_prefetcher_moves_expected_tensor_keys(self):
        batch = {
            "events": torch.tensor([[0, 1, 1000, 1]], dtype=torch.int64),
            "events_offsets": torch.tensor([0, 1], dtype=torch.int64),
            "events_batch_idx": torch.tensor([0], dtype=torch.int64),
            "seq_idx": torch.tensor([0], dtype=torch.int64),
            "frame_index": torch.tensor([0], dtype=torch.int64),
            "t_ref_us": torch.tensor([1000], dtype=torch.int64),
            "t_start_us": torch.tensor([0], dtype=torch.int64),
            "t_end_us": torch.tensor([1000], dtype=torch.int64),
            "sequence_reset": torch.tensor([True], dtype=torch.bool),
            "targets": torch.zeros((1, 1, 5), dtype=torch.float32),
        }
        prefetch = DSECStreamingGPUPrefetcher([batch], device=torch.device("cpu"), require_targets=True)
        out = prefetch.next()
        self.assertEqual(out["events"].device.type, "cpu")
        self.assertEqual(out["targets"].device.type, "cpu")
        self.assertEqual(prefetch.prepared_representation, "streaming_events")

    def test_streaming_prefetcher_requires_events(self):
        batch = {"events_offsets": torch.tensor([0, 0], dtype=torch.int64)}
        with self.assertRaises(ValueError):
            DSECStreamingGPUPrefetcher([batch], device=torch.device("cpu"), require_targets=False)


if __name__ == "__main__":
    unittest.main()


def _packed_event_batch():
    return {
        "events": torch.tensor(
            [
                [1, 2, 1000.0, 1.0],
                [1, 2, 1010.0, -1.0],
                [3, 0, 1020.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        "events_offsets": torch.tensor([0, 3], dtype=torch.int64),
        "t_start_us": torch.tensor([1000000], dtype=torch.int64),
        "t_end_us": torch.tensor([1020000], dtype=torch.int64),
    }


def test_voxel_prefetcher_builds_only_voxel_by_default():
    prefetch = DSECGPUPrefetcher(
        [_packed_event_batch()],
        device=torch.device("cpu"),
        num_bins=3,
        height=4,
        width=5,
        require_targets=False,
        estimator_representation="voxel",
        output_representation="voxel",
        build_voxel=True,
    )

    out = prefetch.next()

    assert out["events_voxel"].shape == (1, 3, 4, 5)
    assert "estimator_input" not in out
    assert float(out["events_voxel"].abs().sum()) > 0.0


def test_cuadrado_iwe_prefetcher_builds_histogram_and_placeholder_without_voxelizing():
    prefetch = DSECGPUPrefetcher(
        [_packed_event_batch()],
        device=torch.device("cpu"),
        num_bins=3,
        height=4,
        width=5,
        require_targets=False,
        estimator_representation="cuadrado_histogram",
        output_representation="iwe",
        output_channels=3,
        build_voxel=False,
    )

    out = prefetch.next()

    assert out["events_voxel"].shape == (1, 3, 4, 5)
    assert out["events_voxel"].sum().item() == 0.0
    assert out["estimator_input"].shape == (1, 2, 3, 4, 5)
    assert out["estimator_input"].sum().item() == 3.0


def test_cuadrado_histogram_rejects_gpu_voxel_augmentation_mismatch():
    with pytest.raises(ValueError, match="cuadrado_histogram"):
        DSECGPUPrefetcher(
            [_packed_event_batch()],
            device=torch.device("cpu"),
            num_bins=3,
            height=4,
            width=5,
            require_targets=False,
            augmentation_backend="gpu_voxel",
            use_mosaic=True,
            estimator_representation="cuadrado_histogram",
            output_representation="iwe",
            build_voxel=False,
        )
