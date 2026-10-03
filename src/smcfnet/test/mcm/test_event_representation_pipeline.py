import torch

from smcfnet.config.config import Config, load_config_from_yaml, update_config_from_dict
from smcfnet.core.mcm_trainer import build_mcm_pretrain_forward_call
from smcfnet.model.build import (
    _build_mcm,
    build_dsec_prefetcher,
    _normalize_estimator_representation,
    _normalize_output_representation,
    _resolve_mcm_event_resolution,
    resolve_dsec_event_pipeline,
)
from smcfnet.model.mcm.mcm import MotionCompensationModule
from smcfnet.model.ofe import FlowEstimator


class _RecordingEstimator(FlowEstimator):
    def __init__(self):
        super().__init__()
        self.last_input_shape = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input_shape = tuple(x.shape)
        return torch.zeros((x.shape[0], 2, x.shape[-2], x.shape[-1]), dtype=x.dtype, device=x.device)

    def reset_state(self) -> None:
        pass

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def get_mode(self) -> str:
        return getattr(self, "mode", "ann")


class _IdentityWarper(torch.nn.Module):
    def forward(self, voxel: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        return voxel


def test_representation_normalization_supports_new_cuadrado_contract():
    cfg = Config()
    cfg.model.mcm.enabled = True
    cfg.model.mcm.architecture = "cuadrado_unet"
    cfg.model.mcm.estimator_representation = "cuadrado_histogram"
    cfg.model.mcm.output_representation = "iwe"

    assert _normalize_estimator_representation(cfg) == "cuadrado_histogram"
    assert _normalize_output_representation(cfg) == "iwe"

    pipeline = resolve_dsec_event_pipeline(cfg)
    assert pipeline["prefetcher_type"] == "voxel"
    assert pipeline["event_timestamp_unit"] == "ms"
    assert pipeline["input_representation"] == "cuadrado_histogram"


def test_legacy_cuadrado_native_histogram_alias_still_resolves():
    cfg = Config()
    cfg.model.mcm.enabled = True
    cfg.model.mcm.architecture = "cuadrado_unet"
    cfg.model.mcm.cuadrado_native_histogram = True

    assert _normalize_estimator_representation(cfg) == "cuadrado_histogram"


def test_cuadrado_histogram_accepts_experimental_cuadrado_architectures():
    import pytest

    pytest.importorskip("spikingjelly")

    for arch in ("cuadrado_unet_bottom_sep", "cuadrado_unet_factorized"):
        cfg = Config()
        cfg.data.event_bins = 21
        cfg.data.image_height = 32
        cfg.data.image_width = 32
        cfg.model.mcm.enabled = True
        cfg.model.mcm.architecture = arch
        cfg.model.mcm.estimator_representation = "cuadrado_histogram"
        cfg.model.mcm.output_representation = "iwe"
        cfg.model.mcm.flow_scaling = 1.0

        mcm = _build_mcm(cfg)

        assert mcm.estimator.image_height == 32
        assert mcm.estimator.image_width == 32
        assert mcm.estimator.get_mode() == cfg.model.mcm.mode


def test_output_representation_iwe_builds_three_channel_renderer():
    cfg = Config()
    cfg.data.event_bins = 5
    cfg.model.mcm.enabled = True
    cfg.model.mcm.architecture = "ecm"
    cfg.model.mcm.estimator_representation = "voxel"
    cfg.model.mcm.output_representation = "iwe"
    cfg.model.mcm.flow_scaling = 1.0

    mcm = _build_mcm(cfg)

    assert getattr(mcm, "output_type") == "iwe_liu_rgb"
    assert getattr(mcm, "output_channels") == 3


def test_mcm_pretrain_forward_call_forwards_estimator_input_separately_from_primary_input():
    voxel = torch.zeros((1, 3, 4, 5))
    estimator_input = torch.ones((1, 2, 3, 4, 5))
    batch = {
        "events_voxel": voxel,
        "estimator_input": estimator_input,
        "events": torch.zeros((0, 4)),
        "events_offsets": torch.tensor([0, 0], dtype=torch.int64),
    }

    primary, kwargs, rep = build_mcm_pretrain_forward_call(
        batch=batch,
        prepared_representation="events_voxel",
        mcm_module=torch.nn.Identity(),
    )

    assert rep == "events_voxel"
    assert primary is voxel
    assert kwargs["estimator_input"] is estimator_input


def test_motion_compensation_module_uses_estimator_input_but_warps_primary_voxel():
    estimator = _RecordingEstimator()
    mcm = MotionCompensationModule(estimator=estimator, warper=_IdentityWarper(), flow_scaling=1.0)
    voxel = torch.zeros((1, 3, 4, 5))
    estimator_input = torch.ones((1, 2, 3, 4, 5))

    corrected, loss = mcm(voxel, estimator_input=estimator_input)

    assert estimator.last_input_shape == tuple(estimator_input.shape)
    assert corrected is voxel
    assert loss.shape == torch.Size([])


def test_cuadrado_configs_load_new_representation_fields():
    for path in [
        "src/smcfnet/config/runs/mcm/cuadrado/cuadrado_unet_supervised.yaml",
        "src/smcfnet/config/runs/mcm/cuadrado/cuadrado_unet_finetune_selfsupervised.yaml",
    ]:
        cfg = Config()
        update_config_from_dict(cfg, load_config_from_yaml(path))
        assert cfg.model.mcm.estimator_representation == "cuadrado_histogram"
        assert cfg.model.mcm.output_representation == "iwe"


def test_streaming_mcm_uses_native_event_resolution_with_orig_rectified_events():
    cfg = Config()
    cfg.data.image_height = 1080
    cfg.data.image_width = 1440
    cfg.data.mapper_prefer_distorted = False
    cfg.data.mapper_rectified_events_resolution = "orig"
    cfg.model.mcm.enabled = True
    cfg.model.mcm.architecture = "sofe"
    cfg.model.mcm.estimator_representation = "streaming_events"
    cfg.model.mcm.output_representation = "iwe"
    cfg.model.mcm.input_representation = "streaming_events"
    cfg.model.mcm.prefetcher_type = "streaming"
    cfg.model.mcm.loss_type = "event_warping_supervised"
    cfg.model.mcm.sofe_architecture = {
        "family": "firenet_lif",
        "base_channels": 4,
        "step_encoding": "hist2",
    }

    assert _resolve_mcm_event_resolution(cfg) == (480, 640)

    mcm = _build_mcm(cfg)

    assert mcm.estimator.image_height == 480
    assert mcm.estimator.image_width == 640



def test_build_dsec_prefetcher_for_cuadrado_iwe_avoids_voxelization():
    cfg = Config()
    cfg.data.event_bins = 3
    cfg.data.image_height = 4
    cfg.data.image_width = 5
    cfg.model.mcm.enabled = True
    cfg.model.mcm.architecture = "cuadrado_unet"
    cfg.model.mcm.estimator_representation = "cuadrado_histogram"
    cfg.model.mcm.output_representation = "iwe"

    batch = {
        "events": torch.tensor(
            [[1, 2, 1000.0, 1.0], [1, 2, 1010.0, -1.0], [3, 0, 1020.0, 1.0]],
            dtype=torch.float32,
        ),
        "events_offsets": torch.tensor([0, 3], dtype=torch.int64),
        "t_start_us": torch.tensor([1000000], dtype=torch.int64),
        "t_end_us": torch.tensor([1020000], dtype=torch.int64),
    }

    prefetch = build_dsec_prefetcher(
        [batch],
        config=cfg,
        device=torch.device("cpu"),
        require_targets=False,
    )
    out = prefetch.next()

    assert prefetch.estimator_representation == "cuadrado_histogram"
    assert prefetch.output_representation == "iwe"
    assert prefetch.build_voxel is False
    assert out["events_voxel"].shape == (1, 3, 4, 5)
    assert out["events_voxel"].sum().item() == 0.0
    assert out["estimator_input"].shape == (1, 2, 3, 4, 5)
    assert out["estimator_input"].sum().item() == 3.0
