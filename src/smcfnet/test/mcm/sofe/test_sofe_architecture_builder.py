from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from smcfnet.model.ofe.sofe import StreamingSpikingFlowEstimator
from smcfnet.model.ofe.firenet_lif import SOFEFireNetLIFReadout
from smcfnet.model.ofe.sofe_builder import (
    SOFEMultiScaleReadout,
    SOFEPersistentDeltaReadout,
    build_sofe_modules,
    expand_sofe_search_space,
    normalize_sofe_architecture_spec,
)


def _make_packed_batch(device: torch.device):
    events = torch.tensor(
        [
            [1.0, 1.0, 0.0, 1.0],
            [2.0, 2.0, 200.0, 0.0],
            [3.0, 3.0, 900.0, 1.0],
            [3.0, 4.0, 1200.0, 1.0],
            [4.0, 4.0, 1900.0, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    offsets = torch.tensor([0, 5], dtype=torch.int64, device=device)
    t_start = torch.tensor([0], dtype=torch.int64, device=device)
    t_end = torch.tensor([2000], dtype=torch.int64, device=device)
    t_ref = torch.tensor([2000], dtype=torch.int64, device=device)
    return events, offsets, t_start, t_end, t_ref


def test_normalize_sofe_architecture_spec_uses_legacy_channels():
    spec = normalize_sofe_architecture_spec(
        {"family": "fire", "stage_depths": 2},
        encoder_channels=[16, 32, 16],
        num_res_blocks=3,
    )

    assert spec["family"] == "fire"
    assert spec["stage_channels"] == [16, 32, 16]
    assert spec["stage_depths"] == [2, 2, 2]
    assert spec["num_res_blocks"] == 3
    assert spec["readout_type"] == "single_scale"
    assert spec["flow_head_init"] == "zero"


def test_expand_sofe_search_space_cartesian_product():
    specs = expand_sofe_search_space(
        base_spec={"family": "fire", "stage_channels": [16, 32, 16]},
        search_space={
            "family": ["fire", "unet"],
            "recurrent_cell": ["convgru", "convrnn"],
            "readout_type": ["single_scale", "multi_scale"],
        },
    )

    assert len(specs) == 8
    families = {spec["family"] for spec in specs}
    assert families == {"fire", "unet"}
    assert all(spec["recurrent_cell"] in {"convgru", "convrnn"} for spec in specs)


@pytest.mark.parametrize(
    ("override", "field_fragment"),
    [
        ({"step_encoding": "mini_voxel"}, "step_encoding"),
        ({"family": "hybrid_unet"}, "family"),
        ({"recurrent_backend": "jax"}, "recurrent_backend"),
        ({"recurrent_cell": "convplif"}, "recurrent_cell"),
        ({"spiking_neuron": "adlif"}, "spiking_neuron"),
        ({"surrogate_function": "super_spike"}, "surrogate_function"),
        ({"spiking_backend": "cupy"}, "spiking_backend"),
        ({"readout_type": "recurrent_refine"}, "readout_type"),
        ({"flow_head_init": "orthogonal"}, "flow_head_init"),
    ],
)
def test_normalize_sofe_architecture_spec_rejects_outside_v1_matrix(override, field_fragment):
    with pytest.raises(NotImplementedError, match=field_fragment):
        normalize_sofe_architecture_spec(override)


def test_normalize_sofe_architecture_spec_rejects_unsupported_spikingjelly_cell_combo():
    with pytest.raises(NotImplementedError, match="spikingjelly"):
        normalize_sofe_architecture_spec(
            {
                "family": "unet",
                "recurrent_backend": "spikingjelly",
                "recurrent_cell": "convgru",
            }
        )


def test_normalize_sofe_architecture_spec_accepts_hist4_legacy():
    normalized = normalize_sofe_architecture_spec(
        {
            "family": "fire",
            "step_encoding": "hist4_legacy",
            "stage_channels": [16, 32, 16],
        }
    )

    assert normalized["step_encoding"] == "hist4_legacy"


def test_normalize_sofe_architecture_spec_accepts_firenet_lif():
    normalized = normalize_sofe_architecture_spec(
        {
            "family": "firenet_lif",
            "step_encoding": "hist2",
            "normalize_counts": False,
            "count_encoding": "raw",
            "base_channels": 8,
            "kernel_size": 3,
            "surrogate_function": "atan",
            "spiking_neuron": {
                "leak": [-4.0, 0.1],
                "thresh": [0.8, 0.1],
                "learn_leak": True,
                "learn_thresh": True,
                "hard_reset": True,
            },
            "flow_activation": "softsign",
            "flow_head_init": "firenet_small_uniform",
            "flow_head_init_std": 0.01,
        }
    )

    assert normalized["family"] == "firenet_lif"
    assert normalized["count_encoding"] == "raw"
    assert normalized["spiking_neuron"] == "lif"
    assert normalized["firenet_leak"] == pytest.approx([-4.0, 0.1])
    assert normalized["firenet_thresh"] == pytest.approx([0.8, 0.1])
    assert normalized["flow_head_init"] == "firenet_small_uniform"


def test_normalize_sofe_architecture_spec_accepts_hagenaars_firenet_lif():
    normalized = normalize_sofe_architecture_spec(
        {
            "family": "hagenaars_firenet_lif",
            "step_encoding": "hist2",
            "normalize_counts": False,
            "count_encoding": "raw",
            "base_channels": 8,
            "kernel_size": 3,
            "surrogate_function": "atan",
            "surrogate_alpha": 1.0,
            "spiking_v_threshold": 1.0,
            "detach_reset": True,
            "flow_activation": "tanh",
            "flow_head_init_std": 0.01,
        }
    )

    assert normalized["family"] == "hagenaars_firenet_lif"
    assert normalized["count_encoding"] == "raw"
    assert normalized["recurrent_backend"] == "spikingjelly"
    assert normalized["recurrent_cell"] == "convlif"
    assert normalized["surrogate_function"] == "atan"
    assert normalized["spiking_v_threshold"] == pytest.approx(1.0)


def test_normalize_sofe_architecture_spec_accepts_persistent_delta_readout():
    normalized = normalize_sofe_architecture_spec(
        {
            "family": "fire",
            "stage_channels": [16, 32, 16],
            "readout_type": "persistent_delta",
            "readout_update_rule": "ema",
            "readout_alpha_init": 0.5,
        }
    )

    assert normalized["readout_type"] == "persistent_delta"
    assert normalized["readout_update_rule"] == "ema"
    assert normalized["readout_alpha_init"] == pytest.approx(0.5)


def test_normalize_sofe_architecture_spec_stflow_lite_defaults_alignment_enabled():
    normalized = normalize_sofe_architecture_spec(
        {
            "family": "stflow_lite",
            "stage_channels": [8, 16, 32, 64],
            "stage_depths": [1, 1, 1, 1],
        }
    )

    assert normalized["family"] == "stflow_lite"
    assert normalized["stflow_alignment_enabled"] is True
    assert normalized["stflow_feedback_enabled"] is False


def test_normalize_sofe_architecture_spec_rejects_stflow_feedback_without_alignment():
    with pytest.raises(ValueError, match="stflow_feedback_enabled"):
        normalize_sofe_architecture_spec(
            {
                "family": "stflow_lite",
                "stage_channels": [8, 16, 32, 64],
                "stage_depths": [1, 1, 1, 1],
                "stflow_alignment_enabled": False,
                "stflow_feedback_enabled": True,
            }
        )


def test_sofe_enumerates_candidates_from_search_space_cpu():
    est = StreamingSpikingFlowEstimator(
        image_height=8,
        image_width=8,
        architecture_spec={"family": "fire", "stage_channels": [16, 32, 16]},
        architecture_search_space={
            "family": ["fire", "unet"],
            "readout_type": ["single_scale", "multi_scale"],
        },
    )

    candidates = est.enumerate_architecture_candidates()
    assert len(candidates) == 4
    assert {spec["family"] for spec in candidates} == {"fire", "unet"}


def test_forward_streaming_builds_fire_family_from_architecture_spec_cpu():
    device = torch.device("cpu")
    est = StreamingSpikingFlowEstimator(
        partition_mode="hybrid",
        events_per_step_target=2,
        step_dt_max_us=700,
        decay_tau_us=1000.0,
        image_height=8,
        image_width=8,
        architecture_spec={
            "family": "fire",
            "step_encoding": "hist4",
            "stage_channels": [16, 32, 16],
            "stage_depths": [1, 1, 1],
            "recurrent_cell": "convgru",
            "readout_type": "single_scale",
            "flow_activation": "tanh",
        },
    ).to(device)

    events, offsets, t_start, t_end, t_ref = _make_packed_batch(device)
    flow, state, debug = est.forward_streaming(
        events=events,
        events_offsets=offsets,
        t_start_us=t_start,
        t_end_us=t_end,
        t_ref_us=t_ref,
        return_debug=True,
    )

    assert flow.shape == (1, 2, 8, 8)
    assert state is not None
    assert debug["sofe_family"] == "fire"
    assert debug["step_encoding"] == "hist4"


def test_forward_streaming_builds_fire_family_convlif_cpu():
    device = torch.device("cpu")
    est = StreamingSpikingFlowEstimator(
        partition_mode="hybrid",
        events_per_step_target=2,
        step_dt_max_us=700,
        decay_tau_us=1000.0,
        image_height=8,
        image_width=8,
        alpha=0.85,
        theta=1.0,
        mode="ann",
        architecture_spec={
            "family": "fire",
            "step_encoding": "hist2",
            "stage_channels": [8, 8],
            "stage_depths": [1, 1],
            "recurrent_cell": "convlif",
            "readout_type": "single_scale",
            "flow_activation": "identity",
        },
    ).to(device)

    events, offsets, t_start, t_end, t_ref = _make_packed_batch(device)
    flow, state, debug = est.forward_streaming(
        events=events,
        events_offsets=offsets,
        t_start_us=t_start,
        t_end_us=t_end,
        t_ref_us=t_ref,
        return_debug=True,
    )

    assert flow.shape == (1, 2, 8, 8)
    assert state is not None
    assert isinstance(state.hidden, dict)
    assert "0" in state.hidden
    assert debug["sofe_family"] == "fire"
    assert est._last_flow_logits is not None
    assert est._last_flow_logits.shape == flow.shape


def test_forward_streaming_builds_firenet_lif_family_cpu():
    device = torch.device("cpu")
    est = StreamingSpikingFlowEstimator(
        partition_mode="hybrid",
        events_per_step_target=2,
        step_dt_max_us=700,
        decay_tau_us=None,
        image_height=8,
        image_width=8,
        mode="snn",
        architecture_spec={
            "family": "firenet_lif",
            "step_encoding": "hist2",
            "normalize_counts": False,
            "count_encoding": "raw",
            "base_channels": 4,
            "kernel_size": 3,
            "surrogate_function": "atan",
            "spiking_neuron": {
                "leak": [-4.0, 0.1],
                "thresh": [0.8, 0.1],
                "learn_leak": True,
                "learn_thresh": True,
                "hard_reset": True,
            },
            "detach_reset": True,
            "readout_type": "single_scale",
            "flow_activation": "softsign",
            "flow_head_init": "firenet_small_uniform",
            "flow_head_init_std": 0.01,
        },
    ).to(device)

    events, offsets, t_start, t_end, t_ref = _make_packed_batch(device)
    flow, state, debug = est.forward_streaming(
        events=events,
        events_offsets=offsets,
        t_start_us=t_start,
        t_end_us=t_end,
        t_ref_us=t_ref,
        return_debug=True,
    )

    assert flow.shape == (1, 2, 8, 8)
    assert torch.isfinite(flow).all()
    assert state is not None
    assert isinstance(state.hidden, dict)
    assert set(state.hidden) == {"head", "G1", "R1a", "R1b", "G2", "R2a", "R2b"}
    assert state.hidden["G1"]["v"].shape == (1, 4, 8, 8)
    assert debug["sofe_family"] == "firenet_lif"
    assert debug["step_encoding"] == "hist2"
    assert "head_spike_rate_final" in debug
    assert "head_spike_rate_active_mean" in debug
    assert "head_spike_rate_active_max" in debug
    assert "head_spike_rate_readout_mean" in debug
    assert "head_spike_rate_readout_last" in debug
    assert "head_v_minus_thresh_active_max" in debug
    assert "head_v_minus_thresh_active_p99" in debug
    assert est._last_flow_logits is not None
    assert est._last_flow_logits.shape == flow.shape
    assert est._last_step_readout_aux is not None
    assert est._last_step_readout_aux["flow_low"].shape == flow.shape
    assert est._last_step_readout_aux["flow_full"].shape == flow.shape
    assert est._last_step_readout_aux["flow_upsample_scale_x"] == 1.0
    assert est._last_step_readout_aux["flow_upsample_scale_y"] == 1.0


def test_forward_streaming_builds_hagenaars_firenet_lif_family_cpu():
    pytest.importorskip("spikingjelly.activation_based")

    device = torch.device("cpu")
    est = StreamingSpikingFlowEstimator(
        partition_mode="hybrid",
        events_per_step_target=2,
        step_dt_max_us=700,
        decay_tau_us=1000.0,
        image_height=8,
        image_width=8,
        mode="snn",
        architecture_spec={
            "family": "hagenaars_firenet_lif",
            "step_encoding": "hist2",
            "normalize_counts": False,
            "count_encoding": "raw",
            "base_channels": 4,
            "kernel_size": 3,
            "surrogate_function": "atan",
            "surrogate_alpha": 1.0,
            "spiking_v_threshold": 1.0,
            "detach_reset": True,
            "firenet_spike_decay_mode": "hold",
            "readout_type": "single_scale",
            "flow_activation": "tanh",
            "flow_head_init_std": 0.01,
        },
    ).to(device)

    events, offsets, t_start, t_end, t_ref = _make_packed_batch(device)
    flow, state, debug = est.forward_streaming(
        events=events,
        events_offsets=offsets,
        t_start_us=t_start,
        t_end_us=t_end,
        t_ref_us=t_ref,
        return_debug=True,
    )

    assert flow.shape == (1, 2, 8, 8)
    assert torch.isfinite(flow).all()
    assert state is not None
    assert isinstance(state.hidden, dict)
    assert set(state.hidden) == {"E1", "G1", "E2", "E3", "G2", "E4", "E5"}
    assert state.hidden["G1"]["mem"].shape == (1, 4, 8, 8)
    assert state.hidden["G1"]["spike"].shape == (1, 4, 8, 8)
    assert debug["sofe_family"] == "hagenaars_firenet_lif"
    assert debug["recurrent_backend"] == "spikingjelly"
    assert debug["recurrent_cell"] == "convlif"
    assert debug["step_encoding"] == "hist2"
    assert est._last_flow_logits is not None
    assert est._last_flow_logits.shape == flow.shape
    assert est._last_step_readout_aux is not None
    assert est._last_step_readout_aux["flow_low"].shape == flow.shape
    assert est._last_step_readout_aux["flow_full"].shape == flow.shape

    flow.square().mean().backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in est.parameters()
        if p.requires_grad
    )


def test_firenet_lif_readout_scales_upsampled_flow_vectors_cpu():
    readout = SOFEFireNetLIFReadout(in_channels=4, flow_head_init_std=0.0)
    with torch.no_grad():
        readout.pred.conv2d.weight.zero_()
        readout.pred.conv2d.bias.copy_(torch.tensor([0.5, -0.25]))

    features = {"latent": torch.ones(1, 4, 4, 5)}
    flow, aux = readout(features, output_size=(8, 10))

    low = aux["flow_low"]
    assert low.shape == (1, 2, 4, 5)
    assert flow.shape == (1, 2, 8, 10)
    assert aux["flow_upsample_scale_x"] == 2.0
    assert aux["flow_upsample_scale_y"] == 2.0
    assert torch.allclose(flow[:, 0].mean(), low[:, 0].mean() * 2.0)
    assert torch.allclose(flow[:, 1].mean(), low[:, 1].mean() * 2.0)


def test_forward_streaming_builds_unet_family_multiscale_cpu():
    device = torch.device("cpu")
    est = StreamingSpikingFlowEstimator(
        step_time_us=1000,
        partition_mode="fixed_time",
        decay_tau_us=1000.0,
        image_height=16,
        image_width=16,
        architecture_spec={
            "family": "unet",
            "step_encoding": "hist2",
            "stage_channels": [8, 16, 32],
            "stage_depths": [1, 1, 1],
            "recurrent_cell": "convrnn",
            "recurrent_placement": "encoder_only",
            "readout_type": "multi_scale",
            "readout_scales": [0, 1],
            "skip_mode": "concat",
            "upsample_mode": "nearest_conv",
        },
    ).to(device)

    events, offsets, t_start, t_end, t_ref = _make_packed_batch(device)
    flow, state, debug = est.forward_streaming(
        events=events,
        events_offsets=offsets,
        t_start_us=t_start,
        t_end_us=t_end,
        t_ref_us=t_ref,
        return_debug=True,
    )

    assert flow.shape == (1, 2, 16, 16)
    assert state is not None
    assert debug["sofe_family"] == "unet"
    assert debug["readout_type"] == "multi_scale"


def test_forward_streaming_builds_unet_family_spikingjelly_lif_cpu():
    pytest.importorskip("spikingjelly.activation_based")

    device = torch.device("cpu")
    est = StreamingSpikingFlowEstimator(
        step_time_us=1000,
        partition_mode="fixed_time",
        decay_tau_us=1000.0,
        image_height=16,
        image_width=16,
        architecture_spec={
            "family": "unet",
            "step_encoding": "hist2",
            "stage_channels": [8, 16, 32],
            "stage_depths": [1, 1, 1],
            "num_res_blocks": 1,
            "recurrent_backend": "spikingjelly",
            "recurrent_cell": "convlif",
            "recurrent_placement": "every_stage",
            "spiking_neuron": "lif",
            "surrogate_function": "sigmoid",
            "spiking_backend": "torch",
            "readout_type": "multi_scale",
            "readout_scales": [0, 1],
            "skip_mode": "concat",
            "upsample_mode": "nearest_conv",
            "flow_activation": "identity",
        },
    ).to(device)

    events, offsets, t_start, t_end, t_ref = _make_packed_batch(device)
    flow, state, debug = est.forward_streaming(
        events=events,
        events_offsets=offsets,
        t_start_us=t_start,
        t_end_us=t_end,
        t_ref_us=t_ref,
        return_debug=True,
    )

    assert flow.shape == (1, 2, 16, 16)
    assert state is not None
    assert isinstance(state.hidden, dict)
    assert "enc_0" in state.hidden
    assert isinstance(state.hidden["enc_0"], dict)
    assert {"mem", "spike"} <= set(state.hidden["enc_0"].keys())
    assert state.hidden["enc_0"]["mem"].shape[0] == 1
    assert debug["sofe_family"] == "unet"
    assert debug["recurrent_backend"] == "spikingjelly"
    assert debug["recurrent_cell"] == "convlif"
    assert debug["spiking_neuron"] == "lif"
    assert debug["spiking_backend"] == "torch"
    assert debug["readout_type"] == "multi_scale"

    assert est._state is not None
    assert est._state.hidden["enc_0"]["mem"].grad_fn is not None
    est.detach_state()
    assert est._state is not None
    assert est._state.hidden["enc_0"]["mem"].grad_fn is None


def test_forward_streaming_builds_stateless_separable_unet_cpu():
    device = torch.device("cpu")
    est = StreamingSpikingFlowEstimator(
        step_time_us=9000,
        partition_mode="fixed_time",
        decay_tau_us=9000.0,
        image_height=16,
        image_width=16,
        architecture_spec={
            "family": "unet",
            "step_encoding": "hist2",
            "stage_channels": [16, 32, 64, 128],
            "stage_depths": [1, 1, 1, 1],
            "num_res_blocks": 1,
            "recurrent_placement": "none",
            "downsample_mode": "maxpool",
            "upsample_mode": "nearest_conv",
            "skip_mode": "concat",
            "separable_conv": True,
            "readout_type": "multi_scale",
            "readout_scales": [0, 1, 2],
            "flow_activation": "identity",
        },
    ).to(device)

    events, offsets, t_start, t_end, t_ref = _make_packed_batch(device)
    flow, state, debug = est.forward_streaming(
        events=events,
        events_offsets=offsets,
        t_start_us=t_start,
        t_end_us=t_end,
        t_ref_us=t_ref,
        return_debug=True,
    )

    assert flow.shape == (1, 2, 16, 16)
    assert state is not None
    assert debug["sofe_family"] == "unet"
    assert debug["readout_type"] == "multi_scale"


def test_unet_pyramid_channel_metadata_matches_runtime_order_cpu():
    spec = normalize_sofe_architecture_spec(
        {
            "family": "unet",
            "stage_channels": [32, 64, 128, 256],
            "stage_depths": [1, 1, 1, 1],
            "num_res_blocks": 1,
            "recurrent_placement": "none",
            "downsample_mode": "maxpool",
            "upsample_mode": "nearest_conv",
            "skip_mode": "concat",
            "separable_conv": True,
            "readout_type": "multi_scale",
            "readout_scales": [0, 1, 2],
        }
    )
    modules = build_sofe_modules(spec, input_channels=2, act="silu")
    core = modules["core"]

    x = torch.randn(1, 2, 480, 640)
    features, _ = core(x, state=None, decay=None)

    runtime_channels = [int(feat.shape[1]) for feat in features["pyramid"]]
    assert runtime_channels == list(core.pyramid_channels)


def test_multiscale_readout_uses_successive_coarse_updates_cpu():
    readout = SOFEMultiScaleReadout(
        in_channels=[1, 1],
        mid_channels=1,
        flow_activation="identity",
        readout_scales=[0, 1],
    )
    readout.pre_blocks = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])

    with torch.no_grad():
        for head in readout.heads:
            head.weight.zero_()
            head.bias.zero_()
        # Runtime pyramid order is finest -> coarsest.
        # Coarse update should be applied first, then refined by the fine scale.
        readout.heads[0].bias.copy_(torch.tensor([2.0, 20.0]))
        readout.heads[1].bias.copy_(torch.tensor([1.0, 10.0]))

    features = {
        "pyramid": [
            torch.zeros(1, 1, 8, 8),
            torch.zeros(1, 1, 4, 4),
        ]
    }
    flow, aux = readout(features, output_size=(8, 8))

    expected = torch.empty_like(flow)
    expected[:, 0].fill_(3.0)
    expected[:, 1].fill_(30.0)
    assert torch.allclose(flow, expected)
    assert torch.allclose(aux["flow_logits"], expected)


def test_build_sofe_modules_supports_persistent_delta_readout_cpu():
    spec = normalize_sofe_architecture_spec(
        {
            "family": "unet",
            "stage_channels": [8, 16, 32],
            "stage_depths": [1, 1, 1],
            "readout_type": "persistent_delta",
            "readout_update_rule": "residual",
            "readout_alpha_init": 1.0,
        }
    )
    modules = build_sofe_modules(spec, input_channels=2, act="silu")
    assert isinstance(modules["readout"], SOFEPersistentDeltaReadout)


def test_build_sofe_modules_supports_stflow_lite_core_cpu():
    spec = normalize_sofe_architecture_spec(
        {
            "family": "stflow_lite",
            "stage_channels": [8, 16, 32, 64],
            "stage_depths": [1, 1, 1, 1],
            "num_res_blocks": 2,
            "readout_type": "single_scale",
        }
    )
    modules = build_sofe_modules(spec, input_channels=2, act="silu")
    core = modules["core"]

    x = torch.randn(1, 2, 32, 32)
    features, hidden = core(x, state=None, decay=None)
    assert features["latent"].shape == (1, 8, 32, 32)
    assert isinstance(features.get("pyramid"), list)
    assert len(features["pyramid"]) == 1
    assert isinstance(hidden, dict)
