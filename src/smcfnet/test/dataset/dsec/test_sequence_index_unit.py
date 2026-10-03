import tempfile
import unittest
from pathlib import Path

from smcfnet.dataset.dsec.dsec_sequence_index import (
    DSECSequenceIndexDataset,
    build_dsec_sequence_index_datasets,
)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


class TestDSECSequenceIndexUnit(unittest.TestCase):
    def test_negative_index_and_dt_window(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seq = "toy_seq"
            yaml_path = root / "dsec.yaml"
            _write_text(
                yaml_path,
                "\n".join(
                    [
                        "dataset:",
                        f"  root: {root}",
                        "  camera: left",
                        "  train_sequences:",
                        f"    - {seq}",
                        "  val_sequences: []",
                        "  test_sequences: []",
                    ]
                ),
            )

            seq_path = root / "train" / seq
            _touch(seq_path / "events" / "left" / "events.h5")
            _write_text(seq_path / "images" / "timestamps.txt", "1000\n2000\n3000\n")

            datasets = build_dsec_sequence_index_datasets(
                dsec_yaml_path=yaml_path,
                split="train",
                dt_ms=1.0,
            )
            ds: DSECSequenceIndexDataset = datasets[0]
            self.assertEqual(len(ds), 3)

            item_last = ds[-1]
            self.assertEqual(item_last["frame_index"], 2)
            self.assertEqual(item_last["t_end_us"], 3000)
            self.assertEqual(item_last["t_start_us"], 2000)

    def test_use_rectified_tracks_falls_back_to_tracks_npy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seq = "toy_seq"
            yaml_path = root / "dsec.yaml"
            _write_text(
                yaml_path,
                "\n".join(
                    [
                        "dataset:",
                        f"  root: {root}",
                        "  camera: left",
                        "  train_sequences:",
                        f"    - {seq}",
                        "  val_sequences: []",
                        "  test_sequences: []",
                    ]
                ),
            )
            seq_path = root / "train" / seq
            _touch(seq_path / "events" / "left" / "events.h5")
            _write_text(seq_path / "images" / "timestamps.txt", "1000\n")
            _touch(seq_path / "object_detections" / "left" / "tracks.npy")

            datasets = build_dsec_sequence_index_datasets(
                dsec_yaml_path=yaml_path,
                split="train",
                dt_ms=1.0,
                use_rectified_tracks=True,
            )
            self.assertEqual(len(datasets), 1)
            self.assertTrue(datasets[0].meta.has_labels)
            self.assertTrue(str(datasets[0].meta.tracks_npy_path).endswith("tracks.npy"))

    def test_val_split_uses_val_sequences_and_split_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_seq = "train_seq"
            val_seq = "val_seq"
            yaml_path = root / "dsec.yaml"
            _write_text(
                yaml_path,
                "\n".join(
                    [
                        "dataset:",
                        f"  root: {root}",
                        "  camera: left",
                        "  val_split_dir: train",
                        "  train_sequences:",
                        f"    - {train_seq}",
                        "  val_sequences:",
                        f"    - {val_seq}",
                        "  test_sequences: []",
                    ]
                ),
            )

            for seq in (train_seq, val_seq):
                seq_path = root / "train" / seq
                _touch(seq_path / "events" / "left" / "events.h5")
                _write_text(seq_path / "images" / "timestamps.txt", "1000\n")

            datasets = build_dsec_sequence_index_datasets(
                dsec_yaml_path=yaml_path,
                split="val",
                dt_ms=1.0,
            )
            self.assertEqual(len(datasets), 1)
            self.assertEqual(datasets[0].sequence, val_seq)


if __name__ == "__main__":
    unittest.main()
