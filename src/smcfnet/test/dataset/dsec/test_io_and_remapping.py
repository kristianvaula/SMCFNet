import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from smcfnet.dataset.dsec.io import (
    compare_dirs,
    extract_from_h5_by_index,
    h5_file_to_dict,
    yaml_file_to_dict,
)
from smcfnet.dataset.dsec.remapping import compute_remapping, conf_to_K


class TestIOAndRemapping(unittest.TestCase):
    def test_compare_dirs_by_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            d1 = root / "a"
            d2 = root / "b"
            (d1 / "sub").mkdir(parents=True)
            (d2 / "sub").mkdir(parents=True)
            (d1 / "sub" / "x.txt").write_text("same")
            (d2 / "sub" / "x.txt").write_text("same")
            self.assertTrue(compare_dirs(d1, d2))

            (d2 / "sub" / "x.txt").write_text("different")
            self.assertFalse(compare_dirs(d1, d2))

    def test_h5_extract_helpers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = Path(tmpdir) / "events.h5"
            with h5py.File(h5_path, "w") as h5f:
                ev = h5f.create_group("events")
                ev.create_dataset("x", data=np.array([1, 2, 3], dtype=np.int32))
                ev.create_dataset("y", data=np.array([4, 5, 6], dtype=np.int32))
                ev.create_dataset("p", data=np.array([1, 0, 1], dtype=np.int8))
                ev.create_dataset("t", data=np.array([100, 200, 300], dtype=np.int64))
                h5f.create_dataset("t_offset", data=np.array(10, dtype=np.int64))
                h5f.create_dataset("ms_to_idx", data=np.array([0, 1, 2, 3], dtype=np.int64))

            out = extract_from_h5_by_index(h5_path, 1, 3)
            self.assertEqual(out["x"].tolist(), [2, 3])
            self.assertEqual(out["t"].tolist(), [210, 310])

            h5_dict = h5_file_to_dict(h5_path)
            self.assertIn("ms_to_idx", h5_dict)

    def test_yaml_loader(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ypath = Path(tmpdir) / "cfg.yaml"
            ypath.write_text("a: 1\nb: test\n")
            out = yaml_file_to_dict(ypath)
            self.assertEqual(out["a"], 1)
            self.assertEqual(out["b"], "test")

    def test_compute_remapping_identity(self):
        mapping = np.zeros((2, 2, 2), dtype=np.float32)
        mapping[..., 0] = np.array([[0, 1], [0, 1]], dtype=np.float32)
        mapping[..., 1] = np.array([[0, 0], [1, 1]], dtype=np.float32)

        calibration = {
            "intrinsics": {
                "camRect0": {"camera_matrix": [1.0, 1.0, 0.0, 0.0]},
                "camRect1": {"camera_matrix": [1.0, 1.0, 0.0, 0.0]},
            },
            "extrinsics": {
                "R_rect0": np.eye(3).tolist(),
                "R_rect1": np.eye(3).tolist(),
                "T_10": np.eye(4).tolist(),
            },
        }

        out = compute_remapping(calibration, {"rectify_map": mapping})
        self.assertEqual(out.shape, mapping.shape)
        self.assertTrue(np.allclose(out, mapping, atol=1e-6))

        k = conf_to_K([2.0, 3.0, 4.0, 5.0])
        self.assertEqual(k[0, 0], 2.0)
        self.assertEqual(k[1, 1], 3.0)
        self.assertEqual(k[0, 2], 4.0)
        self.assertEqual(k[1, 2], 5.0)


if __name__ == "__main__":
    unittest.main()