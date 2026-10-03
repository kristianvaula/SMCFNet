from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from smcfnet.model.ofe.sofe import (
    HistogramStepInputEncoder,
    SOFEState,
    StreamingSpikingFlowEstimator,
    StreamingStepBatch,
)
from smcfnet.model.ofe.firenet_lif import _apply_decay_to_hidden


def _make_packed_batch(device):
    # Two samples packed together.
    # sample 0 times: 0, 200, 900, 1200, 1900
    # sample 1 times: 100, 1100, 2100
    events = torch.tensor(
        [
            [1.0, 1.0, 0.0, 1.0],
            [2.0, 2.0, 200.0, 0.0],
            [3.0, 3.0, 900.0, 1.0],
            [3.0, 4.0, 1200.0, 1.0],
            [4.0, 4.0, 1900.0, 0.0],
            [1.0, 2.0, 100.0, 1.0],
            [2.0, 3.0, 1100.0, 0.0],
            [3.0, 2.0, 2100.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    offsets = torch.tensor([0, 5, 8], dtype=torch.int64, device=device)
    t_start = torch.tensor([0, 0], dtype=torch.int64, device=device)
    t_end = torch.tensor([2000, 3000], dtype=torch.int64, device=device)
    t_ref = torch.tensor([2000, 3000], dtype=torch.int64, device=device)
    return events, offsets, t_start, t_end, t_ref


def test_partition_steps_fixed_time_baseline_cpu():
    device = torch.device("cpu")
    est = StreamingSpikingFlowEstimator(
        step_time_us=1000,
        partition_mode="fixed_time",
        image_height=8,
        image_width=8,
    )

    events, offsets, t_start, t_end, t_ref = _make_packed_batch(device)
    steps = est.partition_steps(
        events=events,
        events_offsets=offsets,
        t_start_us=t_start,
        t_end_us=t_end,
        t_ref_us=t_ref,
    )

    assert len(steps) == 3

    assert torch.equal(steps[0].events_offsets, torch.tensor([0, 3, 4], dtype=torch.int64))
    assert torch.equal(steps[1].events_offsets, torch.tensor([0, 2, 3], dtype=torch.int64))
    assert torch.equal(steps[2].events_offsets, torch.tensor([0, 0, 1], dtype=torch.int64))

    # For empty intervals, step_ref falls back to interval end.
    assert int(steps[2].step_ref_us[0].item()) == 2000


def test_forward_streaming_fixed_time_returns_flow_and_debug_cpu():
    device = torch.device("cpu")
    est = StreamingSpikingFlowEstimator(
        step_time_us=1000,
        partition_mode="fixed_time",
        decay_tau_us=1000.0,
        image_height=8,
        image_width=8,
    )

    events, offsets, t_start, t_end, t_ref = _make_packed_batch(device)
    flow, state, debug = est.forward_streaming(
        events=events,
        events_offsets=offsets,
        t_start_us=t_start,
        t_end_us=t_end,
        t_ref_us=t_ref,
        return_debug=True,
    )

    assert flow.shape == (2, 2, 8, 8)
    assert state is not None
    assert isinstance(state.last_step_ref_us, torch.Tensor)
    assert tuple(state.last_step_ref_us.shape) == (2,)
    assert debug["partition_mode"] == "fixed_time"
    assert debug["num_steps"] == 3
    assert debug["empty_step_policy"] == "explicit_no_event_step"
    assert isinstance(est._last_flow_history, list)
    assert len(est._last_flow_history) >= 1


def test_forward_streaming_fixed_time_honors_readout_interval_us_cpu():
    device = torch.device("cpu")
    est = StreamingSpikingFlowEstimator(
        step_time_us=1000,
        partition_mode="fixed_time",
        readout_interval_us=1500,
        image_height=8,
        image_width=8,
    )

    events, offsets, t_start, t_end, t_ref = _make_packed_batch(device)
    _, _, debug = est.forward_streaming(
        events=events,
        events_offsets=offsets,
        t_start_us=t_start,
        t_end_us=t_end,
        t_ref_us=t_ref,
        return_debug=True,
    )

    assert debug["readout_interval_us"] == 1500
    assert debug["num_steps"] == 3
    assert debug["num_readouts"] == 2


def test_partition_steps_hybrid_respects_event_target_and_time_cap_cpu():
    device = torch.device("cpu")
    est = StreamingSpikingFlowEstimator(
        partition_mode="hybrid",
        events_per_step_target=2,
        step_dt_max_us=700,
        image_height=8,
        image_width=8,
    )

    events, offsets, t_start, t_end, t_ref = _make_packed_batch(device)
    steps = est.partition_steps(
        events=events,
        events_offsets=offsets,
        t_start_us=t_start,
        t_end_us=t_end,
        t_ref_us=t_ref,
    )

    assert len(steps) == 4

    assert torch.equal(steps[0].events_offsets, torch.tensor([0, 2, 3], dtype=torch.int64))
    assert torch.equal(steps[1].events_offsets, torch.tensor([0, 2, 3], dtype=torch.int64))
    assert torch.equal(steps[2].events_offsets, torch.tensor([0, 1, 2], dtype=torch.int64))
    assert torch.equal(steps[3].events_offsets, torch.tensor([0, 0, 0], dtype=torch.int64))

    assert torch.equal(steps[0].step_ref_us, torch.tensor([200, 100], dtype=torch.int64))
    assert torch.equal(steps[1].step_ref_us, torch.tensor([1200, 1100], dtype=torch.int64))
    assert torch.equal(steps[2].step_ref_us, torch.tensor([1900, 2100], dtype=torch.int64))
    assert torch.equal(steps[3].step_ref_us, torch.tensor([2000, 3000], dtype=torch.int64))
    assert torch.equal(steps[3].active_mask, torch.tensor([False, False], dtype=torch.bool))


def test_partition_steps_hybrid_merges_tiny_intervals_with_min_dt_cpu():
    device = torch.device("cpu")
    est = StreamingSpikingFlowEstimator(
        partition_mode="hybrid",
        events_per_step_target=1,
        step_dt_min_us=80,
        image_height=8,
        image_width=8,
    )

    events = torch.tensor(
        [
            [1.0, 1.0, 100.0, 1.0],
            [2.0, 2.0, 150.0, 1.0],
            [3.0, 3.0, 200.0, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    offsets = torch.tensor([0, 3], dtype=torch.int64, device=device)
    t_start = torch.tensor([100], dtype=torch.int64, device=device)
    t_end = torch.tensor([250], dtype=torch.int64, device=device)
    t_ref = torch.tensor([250], dtype=torch.int64, device=device)

    steps = est.partition_steps(
        events=events,
        events_offsets=offsets,
        t_start_us=t_start,
        t_end_us=t_end,
        t_ref_us=t_ref,
    )

    assert len(steps) == 2
    assert torch.equal(steps[0].events_offsets, torch.tensor([0, 3], dtype=torch.int64))
    assert torch.equal(steps[0].step_ref_us, torch.tensor([200], dtype=torch.int64))
    assert torch.equal(steps[1].step_ref_us, torch.tensor([250], dtype=torch.int64))
    assert torch.equal(steps[1].active_mask, torch.tensor([False], dtype=torch.bool))


def test_forward_streaming_hybrid_returns_flow_and_debug_cpu():
    device = torch.device("cpu")
    est = StreamingSpikingFlowEstimator(
        partition_mode="hybrid",
        events_per_step_target=2,
        step_dt_max_us=700,
        decay_tau_us=1000.0,
        image_height=8,
        image_width=8,
    )

    events, offsets, t_start, t_end, t_ref = _make_packed_batch(device)
    flow, state, debug = est.forward_streaming(
        events=events,
        events_offsets=offsets,
        t_start_us=t_start,
        t_end_us=t_end,
        t_ref_us=t_ref,
        return_debug=True,
    )

    assert flow.shape == (2, 2, 8, 8)
    assert state is not None
    assert torch.equal(state.last_step_ref_us, torch.tensor([2000, 3000], dtype=torch.int64))
    assert debug["partition_mode"] == "hybrid"
    assert debug["num_steps"] == 4
    assert debug["delta_t_us_min"] == pytest.approx(100.0)
    assert debug["delta_t_us_max"] == pytest.approx(1000.0)
    assert debug["decay_min"] == pytest.approx(float(torch.exp(torch.tensor(-1.0)).item()))
    assert debug["decay_max"] == pytest.approx(float(torch.exp(torch.tensor(-0.1)).item()))
    assert debug["step_event_count_max"] == pytest.approx(2.0)
    assert debug["step_duration_us_max"] == pytest.approx(900.0)


def test_hybrid_decay_uses_variable_step_duration_and_trailing_quiet_step_cpu():
    device = torch.device("cpu")
    est = StreamingSpikingFlowEstimator(
        partition_mode="hybrid",
        events_per_step_target=2,
        decay_tau_us=1000.0,
        image_height=8,
        image_width=8,
    )

    events, offsets, t_start, t_end, t_ref = _make_packed_batch(device)
    steps = est.partition_steps(
        events=events,
        events_offsets=offsets,
        t_start_us=t_start,
        t_end_us=t_end,
        t_ref_us=t_ref,
    )

    assert len(steps) == 4
    assert torch.equal(steps[-1].active_mask, torch.tensor([False, False], dtype=torch.bool))

    last_ref = t_start.clone()
    delta_history = []
    decay_history = []
    for step in steps:
        delta_t = est.compute_step_delta_t_us(
            step_ref_us=step.step_ref_us,
            last_step_ref_us=last_ref,
            t_start_us=t_start,
        )
        decay = est.compute_decay(
            step_ref_us=step.step_ref_us,
            last_step_ref_us=last_ref,
            t_start_us=t_start,
        )
        delta_history.append(delta_t)
        decay_history.append(decay)
        last_ref = step.step_ref_us

    expected_delta = torch.tensor(
        [
            [200.0, 1100.0],
            [1000.0, 1000.0],
            [700.0, 900.0],
            [100.0, 0.0],
        ],
        dtype=torch.float32,
    )
    observed_delta = torch.stack(delta_history, dim=0)
    observed_decay = torch.stack(decay_history, dim=0)

    assert torch.equal(observed_delta, expected_delta)
    assert torch.allclose(observed_decay, torch.exp(-expected_delta / 1000.0))
    assert observed_decay[-1, 0].item() < 1.0
    assert observed_decay[-1, 1].item() == pytest.approx(1.0)


def test_hybrid_forward_debug_reports_decay_and_step_statistics_cpu():
    device = torch.device("cpu")
    est = StreamingSpikingFlowEstimator(
        partition_mode="hybrid",
        events_per_step_target=2,
        decay_tau_us=1000.0,
        image_height=8,
        image_width=8,
    )

    events, offsets, t_start, t_end, t_ref = _make_packed_batch(device)
    _, state, debug = est.forward_streaming(
        events=events,
        events_offsets=offsets,
        t_start_us=t_start,
        t_end_us=t_end,
        t_ref_us=t_ref,
        return_debug=True,
    )

    assert torch.equal(state.last_step_ref_us, t_ref)
    assert debug["delta_t_us_min"] == pytest.approx(0.0)
    assert debug["delta_t_us_max"] == pytest.approx(1100.0)
    assert debug["step_event_count_min"] == pytest.approx(0.0)
    assert debug["step_event_count_max"] == pytest.approx(2.0)
    assert debug["step_duration_us_min"] == pytest.approx(0.0)
    assert debug["step_duration_us_max"] == pytest.approx(1100.0)
    assert 0.0 < debug["decay_min"] < debug["decay_mean"] < debug["decay_max"] <= 1.0


def test_firenet_hidden_decay_can_hold_decay_or_zero_previous_spikes_cpu():
    device = torch.device("cpu")
    hidden = {
        "v": torch.ones((2, 1, 1, 1), dtype=torch.float32, device=device),
        "spike": torch.ones((2, 1, 1, 1), dtype=torch.float32, device=device),
    }
    decay = torch.tensor([0.25, 0.5], dtype=torch.float32, device=device)

    held = _apply_decay_to_hidden(hidden, decay, spike_decay_mode="hold")
    decayed = _apply_decay_to_hidden(hidden, decay, spike_decay_mode="decay")
    zeroed = _apply_decay_to_hidden(hidden, decay, spike_decay_mode="zero")

    expected = decay.view(2, 1, 1, 1)
    assert torch.allclose(held["v"], expected)
    assert torch.allclose(held["spike"], torch.ones_like(expected))
    assert torch.allclose(decayed["v"], expected)
    assert torch.allclose(decayed["spike"], expected)
    assert torch.allclose(zeroed["v"], expected)
    assert torch.allclose(zeroed["spike"], torch.zeros_like(expected))


def test_histogram_step_input_encoder_count_encodings_cpu():
    device = torch.device("cpu")
    step = StreamingStepBatch(
        events=torch.tensor(
            [
                [1.0, 1.0, 10.0, 1.0],
                [1.0, 1.0, 20.0, 1.0],
                [2.0, 2.0, 30.0, 0.0],
                [2.0, 2.0, 40.0, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        ),
        events_offsets=torch.tensor([0, 4], dtype=torch.int64, device=device),
    )

    normalized = HistogramStepInputEncoder(
        encoding="hist2",
        image_height=4,
        image_width=4,
        count_encoding="normalized",
    )(step)
    raw = HistogramStepInputEncoder(
        encoding="hist2",
        image_height=4,
        image_width=4,
        count_encoding="raw",
    )(step)
    log_scaled = HistogramStepInputEncoder(
        encoding="hist2",
        image_height=4,
        image_width=4,
        count_encoding="log",
        count_scale=2.0,
    )(step)

    assert normalized.shape == (1, 2, 4, 4)
    assert raw.shape == (1, 2, 4, 4)

    assert normalized[0, 0, 1, 1].item() == pytest.approx(0.5)
    assert normalized[0, 1, 2, 2].item() == pytest.approx(0.5)

    assert raw[0, 0, 1, 1].item() == pytest.approx(2.0)
    assert raw[0, 1, 2, 2].item() == pytest.approx(2.0)

    expected_log = 2.0 * torch.log1p(torch.tensor(2.0)).item()
    assert log_scaled[0, 0, 1, 1].item() == pytest.approx(expected_log)
    assert log_scaled[0, 1, 2, 2].item() == pytest.approx(expected_log)


def test_histogram_step_input_encoder_hist4_legacy_preserves_event_order_split_cpu():
    device = torch.device("cpu")
    step = StreamingStepBatch(
        events=torch.tensor(
            [
                [0.0, 0.0, 10.0, 1.0],
                [1.0, 0.0, 20.0, 0.0],
                [0.0, 1.0, 90.0, 1.0],
                [1.0, 1.0, 95.0, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        ),
        events_offsets=torch.tensor([0, 4], dtype=torch.int64, device=device),
        step_start_us=torch.tensor([0], dtype=torch.int64, device=device),
        step_end_us=torch.tensor([100], dtype=torch.int64, device=device),
    )

    encoded = HistogramStepInputEncoder(
        encoding="hist4_legacy",
        image_height=4,
        image_width=4,
        count_encoding="raw",
    )(step)

    assert encoded.shape == (1, 4, 4, 4)
    assert encoded[0, 0, 0, 0].item() == pytest.approx(1.0)
    assert encoded[0, 1, 0, 1].item() == pytest.approx(1.0)
    assert encoded[0, 2, 1, 0].item() == pytest.approx(1.0)
    assert encoded[0, 3, 1, 1].item() == pytest.approx(1.0)


def test_histogram_step_input_encoder_hist4_uses_step_time_split_cpu():
    device = torch.device("cpu")
    step = StreamingStepBatch(
        events=torch.tensor(
            [
                [0.0, 0.0, 10.0, 1.0],
                [1.0, 0.0, 20.0, 0.0],
                [0.0, 1.0, 90.0, 1.0],
                [1.0, 1.0, 95.0, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        ),
        events_offsets=torch.tensor([0, 4], dtype=torch.int64, device=device),
        step_start_us=torch.tensor([0], dtype=torch.int64, device=device),
        step_end_us=torch.tensor([100], dtype=torch.int64, device=device),
    )

    encoded = HistogramStepInputEncoder(
        encoding="hist4",
        image_height=4,
        image_width=4,
        count_encoding="raw",
    )(step)

    assert encoded.shape == (1, 4, 4, 4)
    assert encoded[0, 0, 0, 0].item() == pytest.approx(1.0)
    assert encoded[0, 1, 0, 1].item() == pytest.approx(1.0)
    assert encoded[0, 2, 1, 0].item() == pytest.approx(1.0)
    assert encoded[0, 3, 1, 1].item() == pytest.approx(1.0)
    assert encoded[0, 0, 1, 0].item() == pytest.approx(0.0)
    assert encoded[0, 1, 1, 1].item() == pytest.approx(0.0)


def test_streaming_debug_reports_count_encoding_cpu():
    device = torch.device("cpu")
    est = StreamingSpikingFlowEstimator(
        step_time_us=1000,
        partition_mode="fixed_time",
        image_height=8,
        image_width=8,
        architecture_spec={
            "count_encoding": "raw",
            "count_scale": 4.0,
        },
    )

    events, offsets, t_start, t_end, t_ref = _make_packed_batch(device)
    _, _, debug = est.forward_streaming(
        events=events,
        events_offsets=offsets,
        t_start_us=t_start,
        t_end_us=t_end,
        t_ref_us=t_ref,
        return_debug=True,
    )

    assert debug["count_encoding"] == "raw"
    assert debug["count_scale"] == pytest.approx(4.0)


def test_forward_streaming_persistent_delta_populates_readout_and_flow_state_cpu():
    device = torch.device("cpu")
    est = StreamingSpikingFlowEstimator(
        step_time_us=1000,
        partition_mode="fixed_time",
        image_height=8,
        image_width=8,
        architecture_spec={
            "family": "fire",
            "stage_channels": [8, 8],
            "stage_depths": [1, 1],
            "readout_type": "persistent_delta",
            "readout_update_rule": "residual",
            "readout_alpha_init": 1.0,
            "flow_head_init": "small_normal",
            "flow_head_init_std": 1e-3,
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

    assert flow.shape == (2, 2, 8, 8)
    assert state is not None
    assert isinstance(state.readout_state, dict)
    assert isinstance(state.flow_warm, torch.Tensor)
    assert tuple(state.flow_warm.shape) == tuple(flow.shape)
    assert debug["readout_type"] == "persistent_delta"
    assert debug["readout_update_rule"] == "residual"


def test_readout_flow_emits_maintained_flow_hypothesis_cpu():
    device = torch.device("cpu")
    est = StreamingSpikingFlowEstimator(
        step_time_us=1000,
        partition_mode="fixed_time",
        image_height=6,
        image_width=7,
        architecture_spec={
            "readout_type": "persistent_delta",
        },
    ).to(device)

    t_ref = torch.tensor([123, 456], dtype=torch.int64, device=device)
    flow_hypothesis = torch.full((2, 2, 6, 7), 0.25, dtype=torch.float32, device=device)
    state = SOFEState(
        hidden=None,
        readout_state={"aux": {}},
        flow_warm=flow_hypothesis,
        last_step_ref_us=t_ref.clone(),
    )

    step = StreamingStepBatch(
        step_ref_us=t_ref.clone(),
        step_end_us=t_ref.clone(),
    )
    flow = est.readout_flow(state=state, step=step, t_ref_us=t_ref)
    assert torch.allclose(flow, flow_hypothesis)


def test_forward_streaming_stflow_lite_uses_readout_alignment_state_cpu():
    device = torch.device("cpu")
    est = StreamingSpikingFlowEstimator(
        step_time_us=1000,
        partition_mode="fixed_time",
        image_height=16,
        image_width=16,
        architecture_spec={
            "family": "stflow_lite",
            "stage_channels": [8, 16, 32, 64],
            "stage_depths": [1, 1, 1, 1],
            "readout_type": "persistent_delta",
            "readout_update_rule": "residual",
            "readout_alpha_init": 0.5,
            "stflow_alignment_enabled": True,
            "stflow_feedback_enabled": False,
            "flow_head_init": "small_normal",
            "flow_head_init_std": 1e-3,
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

    assert flow.shape == (2, 2, 16, 16)
    assert isinstance(state.readout_state, dict)
    assert isinstance(state.readout_state.get("align_hidden"), torch.Tensor)
    assert state.readout_state["align_hidden"].shape[0] == 2
    assert isinstance(state.flow_warm, torch.Tensor)
    assert debug["sofe_family"] == "stflow_lite"
    assert debug["stflow_alignment_enabled"] is True
    assert debug["stflow_feedback_enabled"] is False
    assert isinstance(est._last_step_readout_aux, dict)
    assert "alignment_delta_flow" in est._last_step_readout_aux
