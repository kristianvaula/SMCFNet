import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))

from audit_sttp_visual_candidates import (
    _jsonable,
    _review_tile,
    box_inside,
    box_iou,
    match_by_center,
    rank_figure,
)


def test_box_inside_respects_margin():
    assert box_inside(np.array([100.0, 100.0, 40.0, 20.0]), margin=8.0)
    assert not box_inside(np.array([20.0, 100.0, 30.0, 20.0]), margin=8.0)


def test_box_iou_handles_overlap_and_disjoint_boxes():
    a = np.array([50.0, 50.0, 20.0, 20.0])
    assert box_iou(a, a) == 1.0
    assert box_iou(a, np.array([100.0, 100.0, 10.0, 10.0])) == 0.0


def test_match_by_center_handles_reordered_detections():
    reference = np.array([[10.0, 20.0, 4.0, 5.0], [50.0, 60.0, 8.0, 9.0]])
    candidate = reference[[1, 0]].copy()
    indices, distances = match_by_center(reference, candidate)
    np.testing.assert_array_equal(indices, [1, 0])
    np.testing.assert_allclose(distances, 0.0)


def test_rank_figure_deduplicates_frames_and_preserves_metric_rank():
    rows = [
        {"seq": "a", "frame": 1, "metric": 9.0, "visual_score": 0.2},
        {"seq": "a", "frame": 1, "metric": 8.0, "visual_score": 1.0},
        {"seq": "b", "frame": 2, "metric": 7.0, "visual_score": 0.9},
    ]
    ranked, eligible, distinct = rank_figure(
        rows, "metric", descending=True, metric_pool=10, top=10, task_weight=1.0,
    )
    assert eligible == 2
    assert distinct == 2
    assert ranked[0]["seq"] == "a"
    assert ranked[0]["metric"] == 9.0
    assert ranked[0]["metric_rank"] == 1


def test_rank_figure_suppresses_nearby_frames_within_a_sequence():
    rows = [
        {"seq": "a", "frame": 10, "metric": 9.0, "visual_score": 0.5},
        {"seq": "a", "frame": 14, "metric": 8.0, "visual_score": 0.5},
        {"seq": "a", "frame": 25, "metric": 7.0, "visual_score": 0.5},
        {"seq": "b", "frame": 14, "metric": 6.0, "visual_score": 0.5},
    ]
    ranked, eligible, distinct = rank_figure(
        rows, "metric", descending=True, metric_pool=10, top=10,
        task_weight=1.0, min_frame_gap=10,
    )
    assert eligible == 4
    assert distinct == 3
    assert [(row["seq"], row["frame"]) for row in ranked] == [
        ("a", 10), ("a", 25), ("b", 14),
    ]


def test_jsonable_converts_nested_trajectory_arrays():
    row = {"prediction_steps": {"cuadrado": np.zeros((5, 4), dtype=np.float32)}}
    converted = _jsonable(row)
    assert converted["prediction_steps"]["cuadrado"] == [[0.0] * 4] * 5


def test_review_tile_draws_full_prediction_trajectory():
    row = {
        "current_box": [200.0, 200.0, 40.0, 30.0],
        "oracle_box": [260.0, 205.0, 45.0, 35.0],
        "prediction_steps": {
            "cuadrado": [[210.0 + 10.0 * step, 200.0, 40.0, 30.0] for step in range(5)]
        },
    }
    tile = _review_tile(Image.new("RGB", (640, 480), "gray"), row, (480, 360), "cuadrado")
    assert tile.size == (320, 240)
