from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS_DIR = REPO_ROOT / "tools"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(TOOLS_DIR / "thesis_figs"))

from pick_sttp_frames import match_by_center  # noqa: E402
from tools.pick_ttc_event import visual_hazard_metrics  # noqa: E402
from fig_sttp_qualitative import (  # noqa: E402
    _crop_window,
    _draw_error_connector,
    _draw_focus_panel,
    select_focus,
)


def _frame(boxes: np.ndarray, errors: list[float], order: list[int] | None = None) -> dict:
    boxes = np.asarray(boxes, dtype=np.float32)
    oracle = boxes.copy()
    oracle[:, 0] += 5.0
    pred = oracle.copy()
    pred[:, 0] += np.asarray(errors, dtype=np.float32)
    pred_steps = np.stack([
        boxes + (pred - boxes) * (step / 5.0) for step in range(1, 6)
    ], axis=1)
    valid = np.ones(len(boxes), dtype=bool)
    if order is not None:
        order_array = np.asarray(order, dtype=np.int64)
        boxes = boxes[order_array]
        oracle = oracle[order_array]
        pred = pred[order_array]
        pred_steps = pred_steps[order_array]
        valid = valid[order_array]
    return {
        "det_cxcywh": boxes,
        "sttp_future": pred,
        "sttp_future_steps": pred_steps,
        "oracle_future": oracle,
        "oracle_valid": valid,
    }


def _matches(frames: dict[str, dict], reference: str = "eraft") -> dict[str, np.ndarray]:
    ref_boxes = frames[reference]["det_cxcywh"]
    return {
        method: match_by_center(ref_boxes, frame["det_cxcywh"])[0]
        for method, frame in frames.items()
    }


def test_match_by_center_restores_reference_order() -> None:
    reference = np.asarray([
        [10.0, 20.0, 4.0, 6.0],
        [30.0, 40.0, 5.0, 7.0],
        [50.0, 60.0, 6.0, 8.0],
    ])
    candidate = reference[[2, 0, 1]].copy()
    candidate[:, :2] += np.asarray([[0.02, 0.01], [-0.01, 0.02], [0.01, -0.01]])

    indices, distances = match_by_center(reference, candidate)

    assert indices.tolist() == [1, 2, 0]
    assert float(distances.max()) < 0.03


@pytest.mark.parametrize(
    "candidate, message",
    [
        (np.asarray([[10.0, 20.0, 4.0, 6.0]]), "detection-count mismatch"),
        (
            np.asarray([
                [10.0, 20.0, 4.0, 6.0],
                [100.0, 100.0, 4.0, 6.0],
            ]),
            "not one-to-one",
        ),
        (
            np.asarray([
                [11.0, 20.0, 4.0, 6.0],
                [31.0, 40.0, 5.0, 7.0],
            ]),
            "exceeds",
        ),
    ],
)
def test_match_by_center_rejects_invalid_shared_sets(candidate: np.ndarray, message: str) -> None:
    reference = np.asarray([
        [10.0, 20.0, 4.0, 6.0],
        [30.0, 40.0, 5.0, 7.0],
    ])
    with pytest.raises(ValueError, match=message):
        match_by_center(reference, candidate)


def test_focus_selectors_are_invariant_to_method_row_order() -> None:
    boxes = np.asarray([
        [10.0, 20.0, 4.0, 6.0],
        [30.0, 40.0, 5.0, 7.0],
        [50.0, 60.0, 6.0, 8.0],
    ])
    frames = {
        "eraft": _frame(boxes, [1.0, 3.0, 2.0]),
        "identity": _frame(boxes, [5.0, 10.0, 2.0], order=[2, 0, 1]),
        "cuadrado": _frame(boxes, [2.0, 4.0, 3.0], order=[1, 2, 0]),
        "b": _frame(boxes, [9.0, 2.0, 4.0], order=[2, 1, 0]),
        "convgru": _frame(boxes, [4.0, 5.0, 8.0], order=[1, 0, 2]),
    }
    matches = _matches(frames)

    hero_success, _ = select_focus(
        "hero", {"note": "success"}, {"eraft": frames["eraft"]},
        {"eraft": matches["eraft"]}, "eraft",
    )
    hero_failure, _ = select_focus(
        "hero", {"note": "failure"}, {"eraft": frames["eraft"]},
        {"eraft": matches["eraft"]}, "eraft",
    )
    flow_focus, _ = select_focus(
        "flowsource", {},
        {method: frames[method] for method in ("identity", "cuadrado", "eraft")},
        {method: matches[method] for method in ("identity", "cuadrado", "eraft")},
        "eraft",
    )
    recurrence_focus, _ = select_focus(
        "recurrence", {},
        {method: frames[method] for method in ("b", "eraft", "convgru")},
        {method: matches[method] for method in ("b", "eraft", "convgru")},
        "eraft",
    )

    assert hero_success == 0
    assert hero_failure == 1
    assert flow_focus == 1
    assert recurrence_focus == 0


def test_hero_min_displacement_excludes_static_low_error_track() -> None:
    boxes = np.asarray([
        [10.0, 20.0, 4.0, 6.0],
        [30.0, 40.0, 5.0, 7.0],
    ])
    frame = _frame(boxes, [0.1, 2.0])
    frame["oracle_future"][1, 0] += 20.0
    frame["sttp_future"][1, 0] = frame["oracle_future"][1, 0] + 2.0
    frames = {"eraft": frame}
    matches = {"eraft": np.arange(2, dtype=np.int64)}

    focus, _ = select_focus(
        "hero", {"note": "success", "min_disp_px": 10.0},
        frames, matches, "eraft",
    )

    assert focus == 1


def test_pinned_focus_center_overrides_frame_level_error_extreme() -> None:
    boxes = np.asarray([
        [10.0, 20.0, 4.0, 6.0],
        [30.0, 40.0, 5.0, 7.0],
    ])
    frame = _frame(boxes, [50.0, 10.0])
    frames = {"eraft": frame}
    matches = {"eraft": np.arange(2, dtype=np.int64)}

    focus, _ = select_focus(
        "hero", {"note": "failure", "focus_center": [30.0, 40.0]},
        frames, matches, "eraft",
    )

    assert focus == 1


def test_crop_window_is_fixed_size_and_clamped_to_frame() -> None:
    image_shape = (480, 640, 3)
    assert _crop_window(np.asarray([67.0, 235.0]), image_shape, (480, 360)) == (
        0.0, 55.0, 480.0, 415.0,
    )
    assert _crop_window(np.asarray([600.0, 450.0]), image_shape, (480, 360)) == (
        160, 120, 640, 480,
    )


def test_focus_panel_draws_current_plus_five_true_prediction_centers() -> None:
    import matplotlib.pyplot as plt

    frame = _frame(np.asarray([[200.0, 200.0, 40.0, 30.0]]), [10.0])
    fig, ax = plt.subplots()
    _draw_focus_panel(
        ax, np.zeros((480, 640, 3), dtype=np.uint8), frame, 0, 10.0,
        (0.0, 0.0, 480.0, 360.0),
    )
    assert len(ax.lines) == 1
    assert len(ax.lines[0].get_xdata()) == 6
    plt.close(fig)


def test_error_connector_uses_line_without_overlapping_arrowheads() -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    _draw_error_connector(ax, np.asarray([10.0, 10.0]), np.asarray([13.0, 11.0]))

    assert len(ax.lines) == 1
    assert len(ax.patches) == 0
    assert np.allclose(ax.lines[0].get_xdata(), [10.0, 13.0])
    assert np.allclose(ax.lines[0].get_ydata(), [10.0, 11.0])
    plt.close(fig)


def test_visual_hazard_metrics_reports_occupancy_and_growth() -> None:
    boxes = {
        ("seq", 10): {"det_cxcywh": np.asarray([[100.0, 100.0, 40.0, 30.0]])},
        ("seq", 12): {"det_cxcywh": np.asarray([[105.0, 105.0, 80.0, 60.0]])},
    }
    track = {
        "seq": "seq",
        "frames": [
            {"frame_index": 10, "curr_center": [100.0, 100.0]},
            {"frame_index": 12, "curr_center": [105.0, 105.0]},
        ],
    }

    metrics = visual_hazard_metrics(
        boxes, track, {"sttp": 10, "cascade": 12}, pad=0,
    )

    assert metrics["warning_area_fraction"] == pytest.approx(1200.0 / (640.0 * 480.0))
    assert metrics["end_area_fraction"] == pytest.approx(4800.0 / (640.0 * 480.0))
    assert metrics["area_growth"] == pytest.approx(4.0)
