"""Synthetic CUDA stress tests for SOFE streaming partition/forward latency.

Run on GPU with:
    pytest -q -s src/smcfnet/test/mcm/sofe/test_sofe_streaming_stress.py

Useful environment variables:
  SOFE_STRESS_BATCH=1
  SOFE_STRESS_EVENTS=100000
  SOFE_STRESS_HEIGHT=480
  SOFE_STRESS_WIDTH=640
  SOFE_STRESS_WINDOW_US=50000
  SOFE_STRESS_WARMUP=10
  SOFE_STRESS_ITERS=50
  SOFE_STRESS_STEP_TIME_US=5000
  SOFE_STRESS_HYBRID_TARGET=10000
  SOFE_STRESS_HYBRID_MAX_DT_US=5000
  SOFE_STRESS_HYBRID_MIN_DT_US=0
  SOFE_STRESS_MAX_PARTITION_MS=0
  SOFE_STRESS_MAX_FORWARD_MS=0
"""

from __future__ import annotations

import os
import statistics
import time

import pytest

torch = pytest.importorskip("torch")

from smcfnet.model.ofe.sofe import StreamingSpikingFlowEstimator


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _p95(times: list[float]) -> float:
    return statistics.quantiles(times, n=20)[18] if len(times) >= 20 else max(times)


def _time_ms(fn, *, iters: int, warmup: int, device: torch.device) -> list[float]:
    sync = device.type == "cuda"
    for _ in range(max(warmup, 0)):
        fn()

    times: list[float] = []
    for _ in range(max(iters, 1)):
        if sync:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def _make_streaming_batch(
    *,
    batch_size: int,
    events_per_sample: int,
    height: int,
    width: int,
    window_us: int,
    device: torch.device,
):
    per_sample_events = []
    offsets = [0]

    for sample_idx in range(batch_size):
        n = int(events_per_sample)
        idx = torch.arange(n, device=device, dtype=torch.int64)

        events = torch.empty((n, 4), device=device, dtype=torch.float32)
        events[:, 0] = (idx % width).to(torch.float32)
        events[:, 1] = ((idx * 3 + sample_idx * 17) % height).to(torch.float32)

        # Alternate density profiles so hybrid mode exercises both target-driven
        # and time-cap-driven closures under the same synthetic workload.
        t = torch.linspace(0.0, 1.0, steps=n, device=device, dtype=torch.float32)
        if sample_idx % 3 == 0:
            ratios = t * t
        elif sample_idx % 3 == 1:
            ratios = 1.0 - (1.0 - t) * (1.0 - t)
        else:
            ratios = 0.5 * t + 0.5 * (t * t)
        events[:, 2] = torch.round(ratios * float(max(window_us - 1, 0)))

        pol = ((idx + sample_idx) & 1).to(torch.float32)
        events[:, 3] = pol * 2.0 - 1.0

        per_sample_events.append(events)
        offsets.append(offsets[-1] + n)

    packed = torch.cat(per_sample_events, dim=0)
    offsets_t = torch.as_tensor(offsets, dtype=torch.int64, device=device)
    t_start = torch.zeros((batch_size,), dtype=torch.int64, device=device)
    t_end = torch.full((batch_size,), int(window_us), dtype=torch.int64, device=device)
    t_ref = t_end.clone()
    return packed, offsets_t, t_start, t_end, t_ref


def _build_estimator(mode: str, *, height: int, width: int) -> StreamingSpikingFlowEstimator:
    common = dict(
        image_height=height,
        image_width=width,
        decay_tau_us=float(_env_float("SOFE_STRESS_DECAY_TAU_US", 10000.0)),
    )
    if mode == "fixed_time":
        return StreamingSpikingFlowEstimator(
            partition_mode="fixed_time",
            step_time_us=_env_int("SOFE_STRESS_STEP_TIME_US", 5000),
            **common,
        )
    if mode == "hybrid":
        return StreamingSpikingFlowEstimator(
            partition_mode="hybrid",
            events_per_step_target=_env_int("SOFE_STRESS_HYBRID_TARGET", 10000),
            step_dt_max_us=_env_int("SOFE_STRESS_HYBRID_MAX_DT_US", 5000),
            step_dt_min_us=_env_int("SOFE_STRESS_HYBRID_MIN_DT_US", 0),
            **common,
        )
    raise ValueError(f"Unsupported mode: {mode}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SOFE stress test is intended for CUDA/A100 runs")
@pytest.mark.parametrize("mode", ["fixed_time", "hybrid"])
def test_sofe_partition_latency_cuda(mode: str):
    device = torch.device("cuda", 0)
    batch_size = _env_int("SOFE_STRESS_BATCH", 1)
    events_per_sample = _env_int("SOFE_STRESS_EVENTS", 100000)
    height = _env_int("SOFE_STRESS_HEIGHT", 480)
    width = _env_int("SOFE_STRESS_WIDTH", 640)
    window_us = _env_int("SOFE_STRESS_WINDOW_US", 50000)
    warmup = _env_int("SOFE_STRESS_WARMUP", 10)
    iters = _env_int("SOFE_STRESS_ITERS", 50)

    estimator = _build_estimator(mode, height=height, width=width).to(device)
    events, offsets, t_start, t_end, t_ref = _make_streaming_batch(
        batch_size=batch_size,
        events_per_sample=events_per_sample,
        height=height,
        width=width,
        window_us=window_us,
        device=device,
    )

    with torch.inference_mode():
        sample_steps = estimator.partition_steps(
            events=events,
            events_offsets=offsets,
            t_start_us=t_start,
            t_end_us=t_end,
            t_ref_us=t_ref,
        )

        times = _time_ms(
            lambda: estimator.partition_steps(
                events=events,
                events_offsets=offsets,
                t_start_us=t_start,
                t_end_us=t_end,
                t_ref_us=t_ref,
            ),
            iters=iters,
            warmup=warmup,
            device=device,
        )

    mean_ms = statistics.mean(times)
    p95_ms = _p95(times)
    sample_rate = batch_size * 1000.0 / mean_ms
    active_steps_per_batch = sum(int(step.active_mask.sum().item()) for step in sample_steps)
    avg_steps_per_sample = active_steps_per_batch / float(batch_size)

    print(
        f"\n[SOFE partition latency][mode={mode}] "
        f"mean={mean_ms:.3f}ms p95≈{p95_ms:.3f}ms "
        f"samples/s≈{sample_rate:.2f} avg_steps/sample≈{avg_steps_per_sample:.2f} "
        f"(B={batch_size}, N={events_per_sample}, window_us={window_us}, device={device.type})"
    )

    self_max = _env_float("SOFE_STRESS_MAX_PARTITION_MS", 0.0)
    if self_max > 0.0:
        assert mean_ms < self_max
    assert len(sample_steps) >= 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SOFE stress test is intended for CUDA/A100 runs")
@pytest.mark.parametrize("mode", ["fixed_time", "hybrid"])
def test_sofe_forward_streaming_latency_cuda(mode: str):
    device = torch.device("cuda", 0)
    batch_size = _env_int("SOFE_STRESS_BATCH", 1)
    events_per_sample = _env_int("SOFE_STRESS_EVENTS", 100000)
    height = _env_int("SOFE_STRESS_HEIGHT", 480)
    width = _env_int("SOFE_STRESS_WIDTH", 640)
    window_us = _env_int("SOFE_STRESS_WINDOW_US", 50000)
    warmup = _env_int("SOFE_STRESS_WARMUP", 10)
    iters = _env_int("SOFE_STRESS_ITERS", 50)

    estimator = _build_estimator(mode, height=height, width=width).to(device)
    estimator.eval()
    events, offsets, t_start, t_end, t_ref = _make_streaming_batch(
        batch_size=batch_size,
        events_per_sample=events_per_sample,
        height=height,
        width=width,
        window_us=window_us,
        device=device,
    )

    with torch.inference_mode():
        estimator.reset_state()
        flow, state, debug = estimator.forward_streaming(
            events=events,
            events_offsets=offsets,
            t_start_us=t_start,
            t_end_us=t_end,
            t_ref_us=t_ref,
            return_debug=True,
        )

        times = _time_ms(
            lambda: _forward_once(
                estimator=estimator,
                events=events,
                events_offsets=offsets,
                t_start_us=t_start,
                t_end_us=t_end,
                t_ref_us=t_ref,
            ),
            iters=iters,
            warmup=warmup,
            device=device,
        )

    mean_ms = statistics.mean(times)
    p95_ms = _p95(times)
    sample_rate = batch_size * 1000.0 / mean_ms
    realtime_margin_20hz = sample_rate / 20.0

    print(
        f"\n[SOFE forward_streaming latency][mode={mode}] "
        f"mean={mean_ms:.3f}ms p95≈{p95_ms:.3f}ms "
        f"samples/s≈{sample_rate:.2f} 20Hz-margin≈{realtime_margin_20hz:.2f}x "
        f"num_steps={debug['num_steps']} num_readouts={debug['num_readouts']} "
        f"(B={batch_size}, N={events_per_sample}, HxW={height}x{width}, device={device.type})"
    )

    self_max = _env_float("SOFE_STRESS_MAX_FORWARD_MS", 0.0)
    if self_max > 0.0:
        assert mean_ms < self_max
    assert tuple(flow.shape) == (batch_size, 2, height, width)
    assert state is not None


def _forward_once(
    *,
    estimator: StreamingSpikingFlowEstimator,
    events: torch.Tensor,
    events_offsets: torch.Tensor,
    t_start_us: torch.Tensor,
    t_end_us: torch.Tensor,
    t_ref_us: torch.Tensor,
):
    estimator.reset_state()
    return estimator.forward_streaming(
        events=events,
        events_offsets=events_offsets,
        t_start_us=t_start_us,
        t_end_us=t_end_us,
        t_ref_us=t_ref_us,
        return_debug=False,
    )
