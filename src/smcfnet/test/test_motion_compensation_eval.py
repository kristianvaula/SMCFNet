from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path

import torch


def _load_eval_module():
    root = Path(__file__).resolve().parents[3]
    path = root / "experiments" / "p1" / "motion_compensation_eval.py"
    spec = importlib.util.spec_from_file_location("motion_compensation_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _peod_eval_args(root: Path, **overrides):
    values = {
        "config": str(
            root
            / "outputs"
            / "sofe"
            / "peod"
            / "firenet_lif_peod_fixed_time_dt500_g11_4_r2_pretrain"
            / "firenet_lif_peod_fixed_time_dt500_g11_4_r2_pretrain"
            / "config.yaml"
        ),
        "dataset_config": None,
        "batch_size": 1,
        "num_workers": 0,
        "output_representation": "config",
        "iwe_normalize": "none",
        "no_flow_gt": False,
        "viz_batches": 2,
    }
    values.update(overrides)
    return Namespace(**values)


def test_iwe_sharpness_gain_is_positive_for_sharper_output():
    mod = _load_eval_module()
    before = torch.zeros((1, 3, 4, 4))
    before[:, 1] = 0.25
    after = torch.zeros((1, 3, 4, 4))
    after[:, 1, 1, 1] = 4.0

    metrics = mod.image_sharpness_metrics(before, after, output_type="iwe_liu_rgb")

    assert metrics["iwe_var_after"] > metrics["iwe_var_before"]
    assert metrics["iwe_var_gain"] > 0.0
    assert metrics["iwe_positive_gain_frac"] == 1.0


def test_flow_statistics_exposes_zero_flow_collapse_flags():
    mod = _load_eval_module()
    flow = torch.zeros((2, 2, 3, 4))

    metrics = mod.flow_statistics(flow, large_threshold=10.0)

    assert metrics["flow_mag_mean"] == 0.0
    assert metrics["flow_mag_lt_0_1_frac"] == 1.0
    assert metrics["flow_spatial_std_mean"] == 0.0


def test_epe_metrics_uses_valid_mask_channel():
    mod = _load_eval_module()
    flow = torch.zeros((1, 2, 2, 2))
    flow_gt = torch.zeros((1, 3, 2, 2))
    flow_gt[:, 0] = 3.0
    flow_gt[:, 1] = 4.0
    flow_gt[:, 2] = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    metrics = mod.epe_metrics(flow, flow_gt)

    assert metrics["epe_valid_ratio"] == 0.5
    assert abs(metrics["epe_mean"] - 5.0) < 1e-5


def test_timestamp_concentration_accepts_packed_events_and_zero_flow():
    mod = _load_eval_module()
    events = torch.tensor(
        [
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, -1.0],
            [2.0, 2.0, 2.0, 1.0],
        ]
    )
    offsets = torch.tensor([0, 3], dtype=torch.long)
    flow = torch.zeros((1, 2, 4, 4))

    metrics = mod.timestamp_concentration_metrics(
        events=events,
        events_offsets=offsets,
        flow=flow,
        flow_scaling=1.0,
    )

    assert metrics["timestamp_occupied_px"] > 0.0
    assert metrics["timestamp_dispersion_mean"] >= 0.0


def test_events_per_occupied_pixel_reports_concentration():
    mod = _load_eval_module()
    representation = torch.zeros((2, 1, 3, 3))
    representation[0, 0, 0, 0] = 1.0
    representation[0, 0, 0, 1] = 1.0
    representation[1, 0, 1, 1] = 1.0
    offsets = torch.tensor([0, 4, 7], dtype=torch.long)

    metrics = mod.events_per_occupied_pixel_metrics(
        representation,
        events_offsets=offsets,
    )

    assert metrics["events_per_occupied_px_mean"] == 2.5
    assert metrics["events_per_occupied_px_max"] == 3.0
    assert metrics["occupied_px_for_events_mean"] == 1.5


def test_peod_eval_forces_no_flow_gt_and_loads_rgb_for_visualization():
    mod = _load_eval_module()
    root = Path(__file__).resolve().parents[3]

    cfg = mod.load_eval_config(_peod_eval_args(root))

    assert mod.resolve_dataset_name(cfg) == "PEOD"
    assert cfg.data.mapper_load_flow is False
    assert cfg.data.mapper_load_images is True
    assert cfg.model.mcm.output.iwe_normalize == "none"


def test_batch_sample_metadata_and_image_preserve_qualitative_identity():
    mod = _load_eval_module()
    image = torch.full((3, 4, 5), 127, dtype=torch.uint8)
    batch = {
        "frame_index": torch.tensor([23]),
        "t_ref_us": torch.tensor([456000]),
        "image": [image],
    }

    metadata = mod.batch_sample_metadata(batch, sample_idx=0, sequence_label="sequence_003")

    assert metadata["sequence"] == "sequence_003"
    assert metadata["frame_index"] == 23
    assert metadata["t_ref_us"] == 456000
    assert mod.batch_sample_image(batch, 0) is image


def test_load_mcm_weights_does_not_serialize_optimizer_state(monkeypatch):
    mod = _load_eval_module()
    module = torch.nn.Linear(2, 2)
    checkpoint = {
        "mcm": module.state_dict(),
        "epoch": 30,
        "best_metric": 0.25,
        "optimizer": {"state": {"large": torch.ones(1024)}},
    }
    monkeypatch.setattr(mod, "load_mcm_checkpoint", lambda *_args, **_kwargs: checkpoint)

    metadata = mod.load_mcm_weights(module, "unused.pth", strict=True)

    assert metadata == {"epoch": 30, "best_metric": 0.25}


def test_state_reset_policy_matches_training_representation_lifetime():
    mod = _load_eval_module()

    assert mod.resolve_state_reset_policy("events_voxel") == "per_batch"
    assert mod.resolve_state_reset_policy("streaming_events") == "per_sequence"


def test_amp_policy_honors_saved_training_config():
    mod = _load_eval_module()
    cfg = mod.Config()
    cuda = torch.device("cuda", 0)

    cfg.training.use_amp = False
    assert mod.resolve_amp_enabled(cfg, cuda) is False

    cfg.training.use_amp = True
    assert mod.resolve_amp_enabled(cfg, cuda) is True
    assert mod.resolve_amp_enabled(cfg, torch.device("cpu")) is False
