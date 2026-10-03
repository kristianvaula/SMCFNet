"""FLOPs, latency, and SynOps profiling for MCM variants and full model.

Measures:
  - Parameter counts for various ECM flow-estimator configurations
  - FLOPs (via thop) at standard resolutions
  - Forward-pass latency with proper warmup and CUDA sync
  - Sub-component breakdown (estimator vs warping vs full MCM)
  - Full-model overhead: MCM-on vs MCM-off (FLOPs and latency)
  - SynOps, spike rates, and SNN vs CNN energy comparison (RQ4)

Run:  python -m pytest src/smcfnet/test/test_profiling.py -v -s
      (-s required to see profiling tables)

Note: thop counts nn.Module-based ops (Conv2d, BN, etc.).  Grid-sample
and manual tensor ops inside warping / loss are NOT counted,
so reported FLOPs reflect the flow-predictor convolutions only.
Latency measurements capture everything.
"""

from __future__ import annotations

import statistics
import time

import pytest

torch = pytest.importorskip("torch")

from smcfnet.model.ofe.ecm_ofe import ECMFlowEstimator
from smcfnet.model.mcm import MotionCompensationModule, EventWarpingModule
from smcfnet.model.losses import CMaxLoss


# ── Helpers ──────────────────────────────────────────────────────────────


def _time_ms(fn, *, iters: int = 50, warmup: int = 10, device=None) -> list[float]:
    """Time *fn* in milliseconds with warmup and optional CUDA sync."""
    sync = device is not None and device.type == "cuda"
    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(iters):
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def _p95(times: list[float]) -> float:
    return sorted(times)[int(len(times) * 0.95)]


def _fmt_flops(n: float) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f} G"
    if n >= 1e6:
        return f"{n / 1e6:.2f} M"
    if n >= 1e3:
        return f"{n / 1e3:.2f} K"
    return f"{n:.0f}"


def _fmt_params(n: float) -> str:
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return str(int(n))


def _make_simple_mcm(num_bins=5, mid_channels=32):
    """Build MCM with the lightweight ECM estimator."""
    estimator = ECMFlowEstimator(num_bins=num_bins, mid_channels=mid_channels)
    loss_fn = CMaxLoss(contrast_weight=1.0, smooth_weight=0.01)
    return MotionCompensationModule(estimator=estimator, loss_fn=loss_fn)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


# ── 1. ECM parameter count ─────────────────────────────────────────────


def test_ecm_param_count():
    """Report ECM estimator parameter count for various mid_channels."""
    configs = [
        (16, "mid=16"),
        (32, "mid=32 (default)"),
        (64, "mid=64"),
    ]
    print("\n\n── ECM Parameter Count ──")
    print(f"  {'Config':<22} {'Params':>10}")
    print(f"  {'─' * 34}")
    for mid, label in configs:
        estimator = ECMFlowEstimator(num_bins=5, mid_channels=mid)
        total = sum(p.numel() for p in estimator.parameters())
        print(f"  {label:<22} {_fmt_params(total):>10}")
        assert total > 0


# ── 2. ECM FLOPs at multiple resolutions ──────────────────────────────


@pytest.mark.parametrize("h,w", [(64, 80), (240, 320), (480, 640)])
def test_ecm_flops(h, w):
    """Measure ECM estimator FLOPs via thop at different resolutions."""
    thop = pytest.importorskip("thop")
    estimator = ECMFlowEstimator(num_bins=5, mid_channels=32)
    estimator.eval()
    voxel = torch.randn(1, 5, h, w)

    flops, params = thop.profile(estimator, inputs=(voxel,), verbose=False)

    print(f"\n  ECM @ {h}x{w}: {_fmt_flops(flops)} FLOPs, {_fmt_params(params)} params")
    assert flops > 0
    assert params > 0


# ── 3. MCM sub-component latency breakdown ───────────────────────────────


def test_mcm_component_latency(device):
    """Break down MCM forward latency into estimator, warping, and full MCM."""
    estimator = ECMFlowEstimator(num_bins=5, mid_channels=32).to(device)
    warper = EventWarpingModule().to(device)
    mcm = _make_simple_mcm().to(device)
    estimator.eval()
    warper.eval()
    mcm.eval()
    voxel = torch.randn(2, 5, 240, 320, device=device)

    with torch.no_grad():
        flow_ref = estimator(voxel)

        t_est = _time_ms(lambda: estimator(voxel), device=device)
        t_warp = _time_ms(lambda: warper(voxel, flow_ref), device=device)
        t_full = _time_ms(lambda: mcm(voxel), device=device)

    full_mean = statistics.mean(t_full)

    print(f"\n\n  ── MCM Component Latency @ 240x320 B=2 ({device}) ──")
    print(f"  {'Component':<18} {'Mean ms':>9} {'P95 ms':>9} {'% total':>9}")
    print(f"  {'─' * 47}")
    for name, times in [
        ("estimator", t_est),
        ("warping", t_warp),
        ("full MCM", t_full),
    ]:
        mean = statistics.mean(times)
        pct = f"{mean / full_mean * 100:.1f}%" if name != "full MCM" else "─"
        print(f"  {name:<18} {mean:>9.3f} {_p95(times):>9.3f} {pct:>9}")

    assert full_mean > 0


# ── 4. MCM latency at multiple resolutions ───────────────────────────────


@pytest.mark.parametrize("h,w", [(64, 80), (240, 320), (480, 640)])
def test_mcm_latency(h, w, device):
    """Measure MCM forward-pass latency at different resolutions."""
    mcm = _make_simple_mcm().to(device)
    mcm.eval()
    voxel = torch.randn(1, 5, h, w, device=device)

    with torch.no_grad():
        times = _time_ms(lambda: mcm(voxel), device=device)

    mean = statistics.mean(times)
    print(f"\n  MCM @ {h}x{w} B=1 ({device}): mean={mean:.3f}ms  p95={_p95(times):.3f}ms")
    assert mean > 0


# ── 5. Full model FLOPs: MCM on vs off ──────────────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Full-model profiling best on GPU")
def test_full_model_flops_mcm_overhead():
    """Compare DualStreamYOLOX FLOPs with and without MCM."""
    thop = pytest.importorskip("thop")
    from smcfnet.model.backbone import CSPDarknetDualStream
    from smcfnet.model.yolo_head import YOLOXHead
    from smcfnet.model.yolo_pafpn import YOLOPAFPN
    from smcfnet.model.yolox import DualStreamYOLOX

    depth, width = 0.33, 0.25  # nano
    in_channels = [256, 512, 1024]
    H, W = 64, 80

    results: dict[str, dict] = {}
    for use_mcm, label in [(False, "baseline"), (True, "MCM-on")]:
        mcm_instance = _make_simple_mcm() if use_mcm else None
        backbone = CSPDarknetDualStream(
            dep_mul=depth,
            wid_mul=width,
            out_features=("dark3", "dark4", "dark5"),
            use_mcm=use_mcm,
            mcm_instance=mcm_instance,
        )
        neck = YOLOPAFPN(depth=depth, width=width, in_channels=in_channels)
        head = YOLOXHead(num_classes=8, width=width, in_channels=in_channels)
        model = DualStreamYOLOX(backbone, neck, head)
        model.eval()

        rgb = torch.randn(1, 3, H, W)
        events = torch.randn(1, 5, H, W)

        flops, params = thop.profile(model, inputs=(rgb, events), verbose=False)
        results[label] = {"flops": flops, "params": params}

    base_f = results["baseline"]["flops"]
    mcm_f = results["MCM-on"]["flops"]
    overhead = (mcm_f - base_f) / base_f * 100 if base_f > 0 else 0

    print(f"\n\n  ── Full Model FLOPs (nano @ {H}x{W}) ──")
    print(f"  {'Config':<12} {'FLOPs':>12} {'Params':>10}")
    print(f"  {'─' * 36}")
    for label, r in results.items():
        print(f"  {label:<12} {_fmt_flops(r['flops']):>12} {_fmt_params(r['params']):>10}")
    print(f"  {'MCM overhead':<12} {overhead:>11.1f}%")

    assert mcm_f >= base_f, "MCM-on should have >= baseline FLOPs"


# ── 6. Full model latency: MCM on vs off ────────────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Latency comparison needs GPU")
def test_full_model_latency_mcm_overhead():
    """Compare DualStreamYOLOX latency with and without MCM on GPU."""
    from smcfnet.model.backbone import CSPDarknetDualStream
    from smcfnet.model.yolo_head import YOLOXHead
    from smcfnet.model.yolo_pafpn import YOLOPAFPN
    from smcfnet.model.yolox import DualStreamYOLOX

    device = torch.device("cuda")
    depth, width = 0.33, 0.25  # nano
    in_channels = [256, 512, 1024]
    H, W = 64, 80

    results: dict[str, list[float]] = {}
    for use_mcm, label in [(False, "baseline"), (True, "MCM-on")]:
        mcm_instance = _make_simple_mcm() if use_mcm else None
        backbone = CSPDarknetDualStream(
            dep_mul=depth,
            wid_mul=width,
            out_features=("dark3", "dark4", "dark5"),
            use_mcm=use_mcm,
            mcm_instance=mcm_instance,
        ).to(device)
        neck = YOLOPAFPN(depth=depth, width=width, in_channels=in_channels).to(device)
        head = YOLOXHead(num_classes=8, width=width, in_channels=in_channels).to(device)
        model = DualStreamYOLOX(backbone, neck, head).to(device)
        model.eval()

        rgb = torch.randn(1, 3, H, W, device=device)
        events = torch.randn(1, 5, H, W, device=device)

        with torch.no_grad():
            results[label] = _time_ms(
                lambda: model(rgb, events), device=device, iters=30, warmup=10
            )

    base_mean = statistics.mean(results["baseline"])
    mcm_mean = statistics.mean(results["MCM-on"])
    overhead = (mcm_mean - base_mean) / base_mean * 100 if base_mean > 0 else 0

    print(f"\n\n  ── Full Model Latency (nano @ {H}x{W}, GPU) ──")
    print(f"  {'Config':<12} {'Mean ms':>9} {'P95 ms':>9}")
    print(f"  {'─' * 32}")
    for label, times in results.items():
        print(f"  {label:<12} {statistics.mean(times):>9.3f} {_p95(times):>9.3f}")
    print(f"  {'MCM overhead':<12} {overhead:>8.1f}%")


# ── 7. MCM SynOps breakdown per estimator ────────────────────────────────


# Estimator configs: (registry_key, mode, label)
_MCM_VARIANTS = [
    ("st_flownet", "snn", "STFlowNet (SNN)"),
    ("st_flownet", "ann", "STFlowNet (ANN)"),
    ("cuadrado_unet", "snn", "CuadradoUNet (SNN)"),
    ("cnn_baseline", "ann", "CNNFlowNet"),
]


def _variant_num_bins(name: str) -> int:
    return 21 if name == "cuadrado_unet" else 5


def _make_estimator(name: str, mode: str):
    """Create a flow estimator and set its mode."""
    from smcfnet.model.ofe import create_flow_estimator

    estimator = create_flow_estimator(name, num_bins=_variant_num_bins(name), mode="ann")
    estimator.set_mode(mode)
    estimator.eval()
    return estimator


def test_mcm_synops():
    """SynOps breakdown for each MCM estimator variant at 480x640."""
    from smcfnet.utils.synops import SynOpsProfiler, format_report

    H, W = 480, 640

    for name, mode, label in _MCM_VARIANTS:
        estimator = _make_estimator(name, mode)
        voxel = torch.randn(1, _variant_num_bins(name), H, W)

        estimator.reset_state()
        profiler = SynOpsProfiler(estimator)
        with torch.no_grad(), profiler:
            _ = estimator(voxel)

        report = profiler.report()
        print(format_report(report, title=f"{label} @ {H}x{W}"))

        # Basic sanity: should have recorded layers
        assert len(report["layers"]) > 0, f"No layers profiled for {label}"
        assert report["total_macs"] > 0, f"Zero MACs for {label}"


# ── 8. SNN vs CNN energy comparison (RQ4 table) ─────────────────────────


def test_mcm_energy_comparison():
    """Compare SNN vs CNN energy across all estimator variants.

    Produces the RQ4 comparison table:
    Variant | Params | MACs | SynOps | Spike Rate | E_SNN | E_CNN | Ratio
    """
    from smcfnet.utils.synops import SynOpsProfiler, _fmt_ops

    H, W = 480, 640

    print(f"\n\n  ── MCM Energy Comparison @ {H}x{W} (RQ4) ──")
    print(
        f"  {'Variant':<22} {'Params':>8} {'MACs':>10} {'SynOps':>10} "
        f"{'Rate':>6} {'E_SNN mJ':>10} {'E_CNN mJ':>10} {'Ratio':>7}"
    )
    print(f"  {'─' * 90}")

    results: list[dict] = []
    for name, mode, label in _MCM_VARIANTS:
        estimator = _make_estimator(name, mode)
        params = sum(p.numel() for p in estimator.parameters())
        voxel = torch.randn(1, _variant_num_bins(name), H, W)

        estimator.reset_state()
        profiler = SynOpsProfiler(estimator)
        with torch.no_grad(), profiler:
            _ = estimator(voxel)

        r = profiler.report()
        results.append({"label": label, "report": r, "params": params})

        sr = f"{r['avg_spike_rate']:.3f}" if r['avg_spike_rate'] > 0 else "  -"
        print(
            f"  {label:<22} {_fmt_params(params):>8} "
            f"{_fmt_ops(r['total_macs']):>10} {_fmt_ops(r['total_synops']):>10} "
            f"{sr:>6} {r['energy_snn_mj']:>10.4f} {r['energy_cnn_mj']:>10.4f} "
            f"{r['energy_ratio']:>7.4f}"
        )

    # Verify profiler-recognized SNN models have energy savings vs CNN equivalent.
    snn_results = [r for r in results if "SNN" in r["label"]]
    profiled_spiking = 0
    for r in snn_results:
        report = r["report"]
        spiking_count = sum(
            1 for l in report["layers"] if l["layer_type"] == "spiking"
        )
        if spiking_count == 0:
            assert abs(report["energy_ratio"] - 1.0) < 0.01, (
                f"{r['label']}: expected neutral energy ratio when no "
                f"profiler-recognized spiking layers are detected, got "
                f"{report['energy_ratio']:.4f}"
            )
            continue
        profiled_spiking += 1
        assert report["energy_ratio"] < 1.0, (
            f"{r['label']}: expected energy_ratio < 1.0, "
            f"got {report['energy_ratio']:.4f}"
        )
    assert profiled_spiking > 0, "No profiler-recognized SNN layers were detected"

    # Verify CNN baseline has ratio ~1.0
    cnn_results = [r for r in results if r["label"] == "CNNFlowNet"]
    for r in cnn_results:
        report = r["report"]
        assert abs(report["energy_ratio"] - 1.0) < 0.01, (
            f"CNN baseline: expected ratio ~1.0, got {report['energy_ratio']:.4f}"
        )
        assert report["total_spikes"] == 0, "CNN baseline should have no spikes"


# ── 9. Spike rate by layer (heatmap data) ───────────────────────────────


def test_mcm_spike_rate_by_layer():
    """Per-layer spike rate for STFlowNet in SNN mode.

    Prints a table suitable for generating spike rate heatmaps.
    Uses multiple timesteps with positive-biased input to accumulate
    membrane potential past the firing threshold, ensuring spikes occur
    even with untrained weights.
    """
    from smcfnet.utils.synops import SynOpsProfiler

    H, W = 240, 320  # smaller for speed (multiple timesteps)
    estimator = _make_estimator("st_flownet", "snn")

    # Run several timesteps with positive-biased input to charge membranes.
    estimator.reset_state()
    with torch.no_grad():
        for _ in range(5):
            voxel = torch.rand(1, 5, H, W) * 3  # positive, above-average input
            _ = estimator(voxel)

    # Now profile the final timestep (membranes are charged)
    profiler = SynOpsProfiler(estimator)
    voxel = torch.rand(1, 5, H, W) * 3
    with torch.no_grad(), profiler:
        _ = estimator(voxel)

    report = profiler.report()

    print(f"\n\n  ── STFlowNet SNN Spike Rates @ {H}x{W} (after 5 warmup steps) ──")
    print(f"  {'Layer':<40} {'Shape':<30} {'Spike Rate':>10}")
    print(f"  {'─' * 82}")

    spiking_layers = [l for l in report["layers"] if l["layer_type"] == "spiking"]

    for layer in spiking_layers:
        print(
            f"  {layer['name']:<40} {layer['shape']:<30} "
            f"{layer['spike_rate']:>10.4f}"
        )

    assert len(spiking_layers) > 0, "No spiking layers found"

    # After warmup, at least the first encoder should fire
    total_spikes = sum(l["spike_count"] for l in spiking_layers)
    assert total_spikes > 0, (
        "No spikes after 5 warmup timesteps — LIF membranes never reached threshold"
    )

    # No layer should be 100% firing
    for layer in spiking_layers:
        assert layer["spike_rate"] < 1.0, (
            f"{layer['name']}: 100% spike rate (all-fire)"
        )

    print(f"\n  Spiking layers: {len(spiking_layers)}")
    print(f"  Non-spiking layers: {len(report['layers']) - len(spiking_layers)}")
    print(f"  Total spikes: {total_spikes:,}")


# ── CLI runner ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
