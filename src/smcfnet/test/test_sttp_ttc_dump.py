from types import SimpleNamespace
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))

from sttp_ttc_dump import frame_record


def test_frame_record_preserves_full_trajectory():
    detections = np.array([[10, 20, 4, 6, 0.9, 2]], dtype=np.float32)
    pred_steps = np.arange(20, dtype=np.float32).reshape(1, 5, 4)
    oracle_steps = pred_steps + 1.0
    valid_steps = np.array([[True, True, True, True, True]])
    bundle = SimpleNamespace(
        seq_name="seq_a",
        frame_index=7,
        detections=detections,
        pred_future=pred_steps[:, -1],
        pred_future_steps=pred_steps,
        gt_future=oracle_steps[:, -1],
        gt_future_steps=oracle_steps,
        valid=np.array([True]),
        valid_steps=valid_steps,
    )
    record = frame_record(bundle, 5)
    np.testing.assert_array_equal(record["sttp_future_steps"], pred_steps)
    np.testing.assert_array_equal(record["oracle_steps"], oracle_steps)
    np.testing.assert_array_equal(record["oracle_valid_steps"], valid_steps)
    np.testing.assert_array_equal(record["sttp_future"], pred_steps[:, -1])
