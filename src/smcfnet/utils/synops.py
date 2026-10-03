"""SynOps (synaptic operations) profiler for spiking neural networks.

Measures spike counts, spike rates, synaptic operations, and theoretical
energy consumption for SNN vs CNN comparison (RQ4).

Energy model (Horowitz 2014, 45nm CMOS):
  - SNN accumulate (AC): 0.9 pJ per operation
  - CNN multiply-accumulate (MAC): 4.6 pJ per operation

SynOps for a spiking Conv2d layer:
  Each non-zero output element (spike) required C_in * k_h * k_w accumulations
  to produce. SynOps = spike_count * C_in * k_h * k_w.
  Equivalently: SynOps = output_spike_rate * MACs.

Profiling is most meaningful in SNN mode (binary spike outputs). In ANN mode,
outputs are multi-level (ClipFloor quantization), so spike_rate reflects
non-zero fraction rather than true spiking activity.

Usage:
    profiler = SynOpsProfiler(model)
    with profiler:
        output = model(input)
    report = profiler.report()
    print(format_report(report))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from smcfnet.model.spiking_blocks import SpikingConvBlock


# ---------------------------------------------------------------------------
# Energy constants (Horowitz 2014, 45nm CMOS)
# ---------------------------------------------------------------------------

AC_ENERGY_PJ = 0.9    # Accumulate operation (SNN spike-triggered)
MAC_ENERGY_PJ = 4.6   # Multiply-accumulate operation (standard CNN)


# ---------------------------------------------------------------------------
# Per-layer statistics
# ---------------------------------------------------------------------------

@dataclass
class LayerStats:
    """Per-layer profiling statistics."""

    name: str
    layer_type: str        # "spiking" | "conv" | "conv_transpose"
    in_channels: int
    out_channels: int
    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    spatial: tuple[int, int]  # (H_out, W_out)
    spike_count: int = 0       # non-zero output elements (spiking layers only)
    total_elements: int = 0    # total output elements (spiking layers only)
    spike_rate: float = 0.0    # spike_count / total_elements
    synops: int = 0            # spike-aware ops (spiking) or MACs (non-spiking)
    macs: int = 0              # theoretical MACs if this were a CNN layer
    energy_snn_pj: float = 0.0  # synops * AC for spiking, macs * MAC for conv
    energy_cnn_pj: float = 0.0  # macs * MAC (CNN equivalent)


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------

class SynOpsProfiler:
    """Counts spikes and synaptic operations in spiking neural networks.

    Attaches PyTorch forward hooks to SpikingConvBlock and bare Conv2d/
    ConvTranspose2d layers. Non-invasive: hooks are removed on detach or
    context manager exit.

    Usage:
        profiler = SynOpsProfiler(model)
        with profiler:
            output = model(input)
        report = profiler.report()
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._stats: list[LayerStats] = []
        self._batch_size: int = 1

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, *args):
        self.detach()

    def attach(self) -> None:
        """Register forward hooks on all relevant layers."""
        self._stats.clear()
        self._hooks.clear()

        # Identify Conv2d modules owned by SpikingConvBlock (hooked separately)
        spiking_conv_ids: set[int] = set()
        for module in self.model.modules():
            if isinstance(module, SpikingConvBlock):
                spiking_conv_ids.add(id(module.conv))

        for name, module in self.model.named_modules():
            if isinstance(module, SpikingConvBlock):
                hook = module.register_forward_hook(
                    self._make_spiking_hook(name)
                )
                self._hooks.append(hook)
            elif isinstance(module, nn.Conv2d) and id(module) not in spiking_conv_ids:
                hook = module.register_forward_hook(
                    self._make_conv_hook(name)
                )
                self._hooks.append(hook)
            elif isinstance(module, nn.ConvTranspose2d):
                hook = module.register_forward_hook(
                    self._make_conv_hook(name, transpose=True)
                )
                self._hooks.append(hook)

    def detach(self) -> None:
        """Remove all hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    # -- Hook factories ----------------------------------------------------

    def _make_spiking_hook(self, name: str):
        """Create a forward hook for SpikingConvBlock layers."""
        def hook(module: SpikingConvBlock, input, output: torch.Tensor):
            B = output.shape[0]
            self._batch_size = B

            conv = module.conv
            C_in = conv.in_channels
            C_out = conv.out_channels
            k = conv.kernel_size   # always a tuple
            s = conv.stride
            H_out, W_out = output.shape[2], output.shape[3]

            # Spike statistics (across entire batch)
            spike_count = int(output.count_nonzero().item())
            total_elements = output.numel()
            spike_rate = spike_count / total_elements if total_elements > 0 else 0.0

            # SynOps: each output spike required C_in * k_h * k_w accumulates
            synops = spike_count * C_in * k[0] * k[1]

            # MACs: what a standard CNN Conv2d would need
            macs = B * C_out * C_in * k[0] * k[1] * H_out * W_out

            self._stats.append(LayerStats(
                name=name,
                layer_type="spiking",
                in_channels=C_in,
                out_channels=C_out,
                kernel_size=k,
                stride=s,
                spatial=(H_out, W_out),
                spike_count=spike_count,
                total_elements=total_elements,
                spike_rate=spike_rate,
                synops=synops,
                macs=macs,
                energy_snn_pj=synops * AC_ENERGY_PJ,
                energy_cnn_pj=macs * MAC_ENERGY_PJ,
            ))
        return hook

    def _make_conv_hook(self, name: str, transpose: bool = False):
        """Create a forward hook for standard Conv2d / ConvTranspose2d."""
        def hook(module, input, output: torch.Tensor):
            B = output.shape[0]
            self._batch_size = B

            C_in = module.in_channels
            C_out = module.out_channels
            k = module.kernel_size   # always a tuple
            s = module.stride
            H_out, W_out = output.shape[2], output.shape[3]
            groups = getattr(module, "groups", 1)

            # MACs = C_in * C_out * k_h * k_w * H_out * W_out / groups (per batch)
            macs = B * C_in * C_out * k[0] * k[1] * H_out * W_out // groups

            layer_type = "conv_transpose" if transpose else "conv"

            # Non-spiking: no sparsity benefit. Both SNN and CNN energy = MACs * MAC cost.
            self._stats.append(LayerStats(
                name=name,
                layer_type=layer_type,
                in_channels=C_in,
                out_channels=C_out,
                kernel_size=k,
                stride=s,
                spatial=(H_out, W_out),
                synops=macs,
                macs=macs,
                energy_snn_pj=macs * MAC_ENERGY_PJ,
                energy_cnn_pj=macs * MAC_ENERGY_PJ,
            ))
        return hook

    # -- Report generation -------------------------------------------------

    def report(self, per_sample: bool = True) -> dict[str, Any]:
        """Generate profiling report.

        Args:
            per_sample: If True, normalize all counts by batch size.

        Returns:
            Dict with per-layer breakdown and aggregate metrics.
        """
        B = self._batch_size if per_sample else 1

        layers: list[dict[str, Any]] = []
        total_synops = 0
        total_macs = 0
        total_spikes = 0
        total_spike_elements = 0

        for s in self._stats:
            layers.append({
                "name": s.name,
                "layer_type": s.layer_type,
                "shape": (
                    f"{s.in_channels}->{s.out_channels} "
                    f"k={s.kernel_size[0]}x{s.kernel_size[1]} "
                    f"@ {s.spatial[0]}x{s.spatial[1]}"
                ),
                "spike_count": s.spike_count // B,
                "total_elements": s.total_elements // B,
                "spike_rate": s.spike_rate,
                "synops": s.synops // B,
                "macs": s.macs // B,
                "energy_snn_pj": s.energy_snn_pj / B,
                "energy_cnn_pj": s.energy_cnn_pj / B,
            })
            total_synops += s.synops
            total_macs += s.macs
            if s.layer_type == "spiking":
                total_spikes += s.spike_count
                total_spike_elements += s.total_elements

        total_synops //= B
        total_macs //= B
        total_spikes //= B
        total_spike_elements //= B

        avg_spike_rate = (
            total_spikes / total_spike_elements
            if total_spike_elements > 0 else 0.0
        )

        energy_snn_pj = sum(s.energy_snn_pj for s in self._stats) / B
        energy_cnn_pj = sum(s.energy_cnn_pj for s in self._stats) / B

        energy_ratio = energy_snn_pj / energy_cnn_pj if energy_cnn_pj > 0 else 0.0

        return {
            "layers": layers,
            "total_synops": total_synops,
            "total_macs": total_macs,
            "total_spikes": total_spikes,
            "avg_spike_rate": avg_spike_rate,
            "energy_snn_mj": energy_snn_pj / 1e9,   # pJ -> mJ
            "energy_cnn_mj": energy_cnn_pj / 1e9,
            "energy_ratio": energy_ratio,
        }


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def _fmt_ops(n: int) -> str:
    """Format operation count with SI suffix."""
    if n >= 1e9:
        return f"{n / 1e9:.2f}G"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return str(n)


def format_report(report: dict[str, Any], title: str = "SynOps Report") -> str:
    """Format a SynOps profiling report as a human-readable table.

    Args:
        report: Output from ``SynOpsProfiler.report()``.
        title: Header for the table.

    Returns:
        Multi-line string suitable for printing.
    """
    lines: list[str] = []
    lines.append(f"\n  -- {title} --")
    hdr = (
        f"  {'Layer':<40} {'Type':<8} {'Rate':>6} "
        f"{'SynOps':>10} {'MACs':>10} {'E_SNN':>10} {'E_CNN':>10}"
    )
    lines.append(hdr)
    lines.append(f"  {'─' * (len(hdr) - 2)}")

    for layer in report["layers"]:
        sr = (
            f"{layer['spike_rate']:.3f}"
            if layer["layer_type"] == "spiking" else "  -"
        )
        e_snn = _fmt_ops(int(layer["energy_snn_pj"])) + "pJ"
        e_cnn = _fmt_ops(int(layer["energy_cnn_pj"])) + "pJ"
        lines.append(
            f"  {layer['name']:<40} {layer['layer_type']:<8} {sr:>6} "
            f"{_fmt_ops(layer['synops']):>10} {_fmt_ops(layer['macs']):>10} "
            f"{e_snn:>10} {e_cnn:>10}"
        )

    lines.append(f"  {'─' * (len(hdr) - 2)}")
    lines.append("")
    lines.append(f"  Total spikes:     {report['total_spikes']:,}")
    lines.append(f"  Avg spike rate:   {report['avg_spike_rate']:.4f}")
    lines.append(f"  Total SynOps:     {_fmt_ops(report['total_synops'])}")
    lines.append(f"  Total MACs:       {_fmt_ops(report['total_macs'])}")
    lines.append(f"  Energy SNN:       {report['energy_snn_mj']:.6f} mJ")
    lines.append(f"  Energy CNN:       {report['energy_cnn_mj']:.6f} mJ")
    lines.append(f"  Energy ratio:     {report['energy_ratio']:.4f} (E_SNN / E_CNN)")
    lines.append(f"  Energy savings:   {(1 - report['energy_ratio']) * 100:.1f}%")

    return "\n".join(lines)
