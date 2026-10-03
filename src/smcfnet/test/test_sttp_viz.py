# src/smcfnet/test/test_sttp_viz.py
import pytest
from smcfnet.tools.sttp_viz import registry


def test_register_and_get_diagnostic():
    @registry.register_diagnostic("dummy_t1")
    class Dummy:
        pass
    assert registry.get_diagnostic("dummy_t1") is Dummy
    assert Dummy.name == "dummy_t1"
    assert "dummy_t1" in registry.available_diagnostics()


def test_get_unknown_diagnostic_raises():
    with pytest.raises(ValueError, match="Unknown diagnostic"):
        registry.get_diagnostic("does_not_exist_xyz")


import numpy as np
from smcfnet.tools.sttp_viz.bundle import FrameBundle
from smcfnet.tools.sttp_viz.base import Diagnostic


def _synthetic_bundle(n=2):
    return FrameBundle(
        seq_name="seq_x", frame_index=7,
        rgb=np.zeros((48, 64, 3), dtype=np.uint8),
        detections=np.array([[20, 24, 10, 8, 0.9, 0], [40, 30, 12, 10, 0.8, 1]][:n], dtype=np.float32),
        pred_future=np.array([[24, 25, 10, 8], [44, 31, 12, 10]][:n], dtype=np.float32),
        gt_future=np.array([[25, 25, 10, 8], [np.nan] * 4][:n], dtype=np.float32),
        gt_current=np.array([[20, 24, 10, 8], [40, 30, 12, 10]][:n], dtype=np.float32),
        valid=np.array([True, False][:n], dtype=bool),
        per_obj_ade=np.array([1.4, np.nan][:n], dtype=np.float32),
        meta={"k": 5, "frame_ade": 1.4},
    )


def test_framebundle_fields_and_defaults():
    b = _synthetic_bundle()
    assert b.seq_name == "seq_x" and b.frame_index == 7
    assert b.detections.shape == (2, 6)
    assert b.flow is None and b.stem_features is None
    assert b.meta["k"] == 5


def test_diagnostic_base_defaults_noop():
    d = Diagnostic(out_dir="/tmp/x")
    # base methods exist and are no-ops (don't raise)
    d.on_frame(_synthetic_bundle())
    d.on_run(object())
    d.finalize()
    assert d.LEVEL == "frame"


from smcfnet.tools.sttp_viz import decode as dec


def test_cxcywh_to_xywh():
    x, y, w, h = dec.cxcywh_to_xywh(np.array([10.0, 20.0, 4.0, 6.0]))
    assert (x, y, w, h) == (8.0, 17.0, 4.0, 6.0)


def test_pair_ade_center_distance_and_validity():
    pred = np.array([[10, 10, 4, 4], [20, 20, 4, 4]], dtype=np.float32)
    gt = np.array([[13, 14, 4, 4], [np.nan] * 4], dtype=np.float32)
    valid = np.array([True, False])
    ade = dec.pair_ade(pred, gt, valid)
    assert abs(ade[0] - 5.0) < 1e-5     # sqrt(3^2 + 4^2)
    assert np.isnan(ade[1])             # invalid -> NaN


def test_decode_future_boxes_matches_decode_absolute_math():
    import torch
    # one scale, 1x1 grid, stride 8, pred (tx,ty,tw,th)=(0.5,0.5,0,0)
    fut = [torch.tensor([[[[0.5]], [[0.5]], [[0.0]], [[0.0]]]])]  # [1,4,1,1]
    grids = torch.tensor([[[2.0, 3.0]]])          # [1,1,2]  (gx,gy)
    strides = torch.tensor([[8.0]])                # [1,1]
    out = dec.decode_future_boxes(fut, strides, grids)  # [1,1,4]
    cx, cy, w, h = out[0, 0].tolist()
    assert abs(cx - (0.5 + 2.0) * 8) < 1e-4        # 20.0
    assert abs(cy - (0.5 + 3.0) * 8) < 1e-4        # 28.0
    assert abs(w - np.exp(0.0) * 8) < 1e-4         # 8.0
    assert abs(h - 8.0) < 1e-4


from pathlib import Path


def test_predict_renders_png_and_gif(tmp_path):
    from smcfnet.tools.sttp_viz.diagnostics.predict import PredictDiagnostic
    d = PredictDiagnostic(out_dir=tmp_path, style="A", fps=5)
    b = _synthetic_bundle(2)
    d.on_frame(b)
    b2 = _synthetic_bundle(2); b2.frame_index = 8
    d.on_frame(b2)
    d.finalize()
    assert (tmp_path / "seq_x" / "frame_000007.png").exists()
    assert (tmp_path / "seq_x" / "frame_000008.png").exists()
    assert (tmp_path / "seq_x.gif").exists()


# --- Task 5: SttpVizCore (cluster-guarded integration smoke) ---
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="core forward needs GPU + DSEC dataset (cluster only)")
def test_core_yields_bundles(tmp_path):
    from smcfnet.tools.sttp_viz.core import SttpVizCore
    core = SttpVizCore(
        config_path="src/smcfnet/config/runs/sttp/sttp_flowinj_eraft_v4.yaml",
        checkpoint="outputs/sttp/v4/sttp_flowinj_eraft_v4/best.pth",
        device="cuda",
    )
    bundles = list(core.iterate_sequence(seq_name=None, max_frames=3))  # first val seq, 3 frames
    assert len(bundles) >= 1
    b = bundles[0]
    assert b.rgb.ndim == 3 and b.rgb.shape[2] == 3
    assert b.detections.shape[1] == 6
    assert b.pred_future.shape[1] == 4
    assert b.meta["k"] >= 1


# --- Task 6: CLI + entry point ---
def test_cli_parses_and_rejects_unknown_diagnostic():
    from smcfnet.tools.sttp_viz import cli
    args = cli.parse_args([
        "--config", "c.yaml", "--checkpoint", "b.pth",
        "--diagnostic", "predict", "--sequence", "seq_x", "--out", "/tmp/o",
    ])
    assert args.diagnostic == "predict" and args.sequence == "seq_x"
    with pytest.raises(ValueError, match="Unknown diagnostic"):
        cli.resolve_diagnostic("nope_xyz", out_dir="/tmp/o", style="A", fps=5)


def test_cli_resolves_predict():
    from smcfnet.tools.sttp_viz import cli
    d = cli.resolve_diagnostic("predict", out_dir="/tmp/o", style="A", fps=5)
    assert d.LEVEL == "frame" and d.name == "predict"


# --- thesis-figure additions: frame list, conf, full-trajectory fields ---
def test_cli_parse_frame_list():
    from smcfnet.tools.sttp_viz import cli
    fl = cli.parse_frame_list("zurich_city_16_a:12,340;zurich_city_19_a:77")
    assert fl == {"zurich_city_16_a": [12, 340], "zurich_city_19_a": [77]}
    assert cli.parse_frame_list(None) is None
    assert cli.parse_frame_list("") is None
    # duplicate sequence entries merge
    assert cli.parse_frame_list("a:1;a:2") == {"a": [1, 2]}
    with pytest.raises(ValueError, match="bad --frame-list"):
        cli.parse_frame_list("no_colon_here")


def test_cli_conf_default_and_frame_list_args():
    from smcfnet.tools.sttp_viz import cli
    args = cli.parse_args([
        "--config", "c.yaml", "--checkpoint", "b.pth",
        "--diagnostic", "predict", "--out", "/tmp/o",
        "--frame-list", "seq_x:1,2", "--conf", "0.5",
    ])
    assert args.conf == 0.5
    assert cli.parse_frame_list(args.frame_list) == {"seq_x": [1, 2]}
    default = cli.parse_args([
        "--config", "c.yaml", "--checkpoint", "b.pth",
        "--diagnostic", "predict", "--out", "/tmp/o",
    ])
    assert default.conf == 0.3 and default.frame_list is None


def _steps_bundle(n=2, k=3):
    b = _synthetic_bundle(n)
    b.pred_future_steps = np.tile(b.pred_future[:, None, :], (1, k, 1)).astype(np.float32)
    b.gt_future_steps = np.tile(b.gt_future[:, None, :], (1, k, 1)).astype(np.float32)
    b.valid_steps = np.tile(b.valid[:, None], (1, k))
    b.meta["k"] = k
    return b


def test_framebundle_step_fields_default_none():
    b = _synthetic_bundle()
    assert b.pred_future_steps is None and b.gt_future_steps is None and b.valid_steps is None


def test_predict_renders_polyline_and_paper_style(tmp_path):
    from smcfnet.tools.sttp_viz.diagnostics.predict import PredictDiagnostic
    d = PredictDiagnostic(out_dir=tmp_path / "paper", style="paper", fps=5)
    d.on_frame(_steps_bundle())
    d.finalize()
    assert (tmp_path / "paper" / "seq_x" / "frame_000007.png").exists()
    # legacy endpoint-only bundle still renders under paper style
    d2 = PredictDiagnostic(out_dir=tmp_path / "legacy", style="paper", fps=5)
    d2.on_frame(_synthetic_bundle())
    d2.finalize()
    assert (tmp_path / "legacy" / "seq_x" / "frame_000007.png").exists()
