import tempfile
import unittest
from pathlib import Path

from smcfnet.dataset.dsec.dsec_flow_index import (
    DSECFlowIndexDataset,
    build_dsec_flow_index_datasets,
)
from smcfnet.dataset.dsec.dsec_sequence_index import build_dsec_sequence_metas


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


class TestDSECFlowIndexDataset(unittest.TestCase):
    def test_build_and_getitem(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seq = "toy_seq"

            # Minimal dsec YAML.
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

            flow_dir = seq_path / "flow"
            _write_text(flow_dir / "forward_timestamps.txt", "1000,2000\n2000,3000\n")
            _touch(flow_dir / "forward" / "000000.png")
            _touch(flow_dir / "forward" / "000001.png")

            metas = build_dsec_sequence_metas(
                dsec_yaml_path=yaml_path,
                split="train",
            )

            ds = DSECFlowIndexDataset(metas[0], seq_idx=0, flow_direction="forward")
            self.assertEqual(len(ds), 2)

            sample = ds[0]
            self.assertEqual(sample["seq_idx"], 0)
            self.assertEqual(sample["frame_index"], 0)
            self.assertEqual(sample["t_start_us"], 1000)
            self.assertEqual(sample["t_end_us"], 2000)
            self.assertTrue(sample["flow_png_path"].endswith("000000.png"))

            sample_last = ds[-1]
            self.assertEqual(sample_last["frame_index"], 1)

            built = build_dsec_flow_index_datasets(
                dsec_yaml_path=yaml_path,
                split="train",
                flow_direction="forward",
            )
            self.assertEqual(len(built), 1)
            self.assertEqual(len(built[0]), 2)

    def test_deterministic_local_window_selection(self):
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
            _write_text(seq_path / "images" / "timestamps.txt", "1000\n2000\n3000\n4000\n5000\n")

            flow_dir = seq_path / "flow"
            _write_text(
                flow_dir / "forward_timestamps.txt",
                "1000,2000\n2000,3000\n3000,4000\n4000,5000\n",
            )
            for i in range(4):
                _touch(flow_dir / "forward" / f"{i:06d}.png")

            metas = build_dsec_sequence_metas(dsec_yaml_path=yaml_path, split="train")
            ds = DSECFlowIndexDataset(
                metas[0],
                seq_idx=0,
                flow_direction="forward",
                index_start=1,
                index_count=2,
                index_stride=2,
            )

            self.assertEqual(len(ds), 2)
            self.assertEqual(ds[0]["frame_index"], 1)
            self.assertEqual(ds[0]["t_start_us"], 2000)
            self.assertTrue(ds[0]["flow_png_path"].endswith("000001.png"))
            self.assertEqual(ds[1]["frame_index"], 3)
            self.assertEqual(ds[1]["t_end_us"], 5000)

            built = build_dsec_flow_index_datasets(
                dsec_yaml_path=yaml_path,
                split="train",
                flow_direction="forward",
                index_start=1,
                index_count=2,
                index_stride=2,
            )
            self.assertEqual([built[0][i]["frame_index"] for i in range(len(built[0]))], [1, 3])


if __name__ == "__main__":
    unittest.main()
