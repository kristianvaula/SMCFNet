import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from smcfnet.dataset.dsec.dsec_slicer import DSECEventSlicer


class TestDSECEventSlicerTimestampUnit(unittest.TestCase):
    def test_get_window_us_supports_ms_and_us(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = Path(tmpdir) / "events.h5"
            with h5py.File(h5_path, "w") as h5f:
                evgrp = h5f.create_group("events")
                evgrp.create_dataset("x", data=np.array([10, 20, 30], dtype=np.int32))
                evgrp.create_dataset("y", data=np.array([1, 2, 3], dtype=np.int32))
                evgrp.create_dataset("t", data=np.array([100, 1100, 2100], dtype=np.int64))
                evgrp.create_dataset("p", data=np.array([1, 0, 1], dtype=np.int8))
                h5f.create_dataset("ms_to_idx", data=np.array([0, 1, 2, 3], dtype=np.int64))
                h5f.create_dataset("t_offset", data=np.array(0, dtype=np.int64))

            with h5py.File(h5_path, "r") as h5f:
                slicer = DSECEventSlicer(h5f)
                ms_window = slicer.get_window_us(1000, 2500, timestamp_unit="ms")
                us_window = slicer.get_window_us(1000, 2500, timestamp_unit="us")

            self.assertIsNotNone(ms_window)
            self.assertIsNotNone(us_window)
            self.assertEqual(ms_window.timestamp_unit, "ms")
            self.assertEqual(us_window.timestamp_unit, "us")
            self.assertEqual(ms_window.events.shape, us_window.events.shape)
            self.assertEqual(ms_window.events[:, 2].tolist(), [1, 2])
            self.assertEqual(us_window.events[:, 2].tolist(), [1100, 2100])


if __name__ == "__main__":
    unittest.main()
