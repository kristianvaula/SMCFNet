"""Profile one real DSEC batch through the MCM/SOFE pretraining stack.

This tool is intentionally close to the real training path:

- it loads a normal YAML run config
- builds the real MCM module from that config
- builds the real DSEC train loader + GPU prefetcher
- fetches one real batch
- measures fetch time, forward/backward latency, and CUDA memory

Typical usage:

    PYTHONPATH=src conda run -n smcm-mcfnet \
    python -m smcfnet.utils.profile_mcm_batch \
      --config src/smcfnet/config/runs/sofe/smoke/fire_lite_fixed_time_smoke.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.amp import GradScaler, autocast

from smcfnet.config.config import Config, load_config_from_yaml, update_config_from_dict
from smcfnet.core.mcm_trainer import (
    _build_mcm_optimizer,
    build_mcm_pretrain_forward_call,
    configure_mcm_pretrain_data_flags,
)
from smcfnet.model.build import _build_mcm, build_dsec_prefetcher, build_train_loader_cfg
from smcfnet.model.mcm.weight_utils import load_mcm_weights_into_module

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile one real MCM/SOFE batch")
    parser.add_argument("--config", type=str, required=True, help="Run config YAML")
    parser.add_argument("--iters", type=int, default=5, help="Number of measured compute iterations")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup compute iterations")
    parser.add_argument("--device", type=str, default=None, help="Device override, e.g. cuda:0 or cpu")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch-size override")
    parser.add_argument("--num-workers", type=int, default=None, help="Optional num-workers override")
    parser.add_argument("--forward-only", action="store_true", help="Measure forward only")
    parser.add_argument(
        "--optimizer-step",
        action="store_true",
        help="Include optimizer.step() in measured iterations when backward is enabled",
    )
    parser.add_argument(
        "--keep-state",
        action="store_true",
        help="Do not reset streaming state between profiled iterations",
    )
    parser.add_argument("--output-json", type=str, default=None, help="Optional JSON output path")
    return parser.parse_args()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    idx = min(max(int(len(values) * 0.95), 0), len(values) - 1)
    return sorted(values)[idx]


def _scalarize(val: Any) -> Any:
    if isinstance(val, torch.Tensor):
        if val.numel() == 1:
            return float(val.detach().item())
        return [float(v) for v in val.detach().reshape(-1).tolist()]
    if isinstance(val, dict):
        return {str(k): _scalarize(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_scalarize(v) for v in val]
    if isinstance(val, (int, float, str, bool)) or val is None:
        return val
    return str(val)


def _collect_loss_and_flow_summary(mcm: torch.nn.Module) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}

    for attr_name, key in [
        ("_last_contrast_loss", "contrast"),
        ("_last_smooth_loss", "smooth"),
        ("_last_magnitude_loss", "magnitude"),
        ("_last_epe", "epe"),
    ]:
        val = getattr(mcm, attr_name, None)
        if isinstance(val, torch.Tensor):
            summary[key] = float(val.detach().item())

    flow = getattr(mcm, "_last_flow", None)
    if isinstance(flow, torch.Tensor) and flow.numel() > 0:
        flow_f = flow.detach().float()
        flow_mag = flow_f.norm(p=2, dim=1)
        summary["flow"] = {
            "mean": float(flow_f.mean().item()),
            "std": float(flow_f.std().item()),
            "min": float(flow_f.min().item()),
            "max": float(flow_f.max().item()),
            "mag_mean": float(flow_mag.mean().item()),
            "mag_max": float(flow_mag.max().item()),
        }

    stream_debug = getattr(mcm, "_last_stream_debug", None)
    if isinstance(stream_debug, dict):
        summary["stream_debug"] = _scalarize(stream_debug)

    return summary


def _build_batch_summary(batch: Dict[str, Any], *, representation: str) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"representation": str(representation)}

    if "events_offsets" in batch and isinstance(batch["events_offsets"], torch.Tensor):
        offsets = batch["events_offsets"].detach().to(torch.int64)
        if offsets.numel() >= 2:
            counts = offsets[1:] - offsets[:-1]
            summary["batch_size"] = int(counts.numel())
            summary["num_events_total"] = int(counts.sum().item())
            summary["events_per_sample_mean"] = float(counts.float().mean().item())
            summary["events_per_sample_max"] = int(counts.max().item())
            summary["events_per_sample_min"] = int(counts.min().item())

    if "t_start_us" in batch and "t_end_us" in batch:
        t_start = batch["t_start_us"].detach().to(torch.int64)
        t_end = batch["t_end_us"].detach().to(torch.int64)
        if t_start.numel() == t_end.numel() and t_start.numel() > 0:
            durations = (t_end - t_start).clamp(min=0)
            summary["window_us_mean"] = float(durations.float().mean().item())
            summary["window_us_max"] = int(durations.max().item())
            summary["window_us_min"] = int(durations.min().item())

    if "events_voxel" in batch and isinstance(batch["events_voxel"], torch.Tensor):
        voxel = batch["events_voxel"]
        summary["batch_size"] = int(voxel.shape[0])
        summary["voxel_shape"] = [int(v) for v in voxel.shape]

    return summary


def _load_config(config_path: str, *, batch_size: Optional[int], num_workers: Optional[int]) -> Config:
    cfg = Config()
    yaml_cfg = load_config_from_yaml(config_path)
    update_config_from_dict(cfg, yaml_cfg)

    cfg.training.mode = "mcm_pretrain"
    cfg.num_gpus = 1
    cfg.wandb.use_wandb = False
    cfg.wandb.mode = "disabled"
    cfg.log.save_all_checkpoints = False

    if batch_size is not None:
        cfg.training.batch_size = int(batch_size)
    if num_workers is not None:
        cfg.data.num_workers = int(num_workers)

    configure_mcm_pretrain_data_flags(cfg)
    return cfg


def _pick_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    return torch.device("cpu")


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")

    cfg = _load_config(args.config, batch_size=args.batch_size, num_workers=args.num_workers)
    device = _pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    logger.info("Building MCM module from %s", args.config)
    mcm = _build_mcm(cfg).to(device)
    if getattr(cfg.model.mcm, "pretrained_weights", None):
        load_mcm_weights_into_module(mcm, cfg.model.mcm.pretrained_weights)

    optimizer = None
    if not bool(args.forward_only):
        optimizer = _build_mcm_optimizer(cfg, mcm)

    logger.info("Building train loader and prefetcher")
    train_loader = build_train_loader_cfg(cfg)
    loader_iter = iter(train_loader)
    prefetcher = build_dsec_prefetcher(
        loader_iter,
        config=cfg,
        device=device,
        require_targets=False,
        voxel_height=int(getattr(cfg.data, "image_height", 480)),
        voxel_width=int(getattr(cfg.data, "image_width", 640)),
    )
    prepared_representation = str(prefetcher.prepared_representation)

    fetch_t0 = time.perf_counter()
    batch = prefetcher.next()
    _sync(device)
    fetch_ms = (time.perf_counter() - fetch_t0) * 1000.0

    primary_input, forward_kwargs, representation = build_mcm_pretrain_forward_call(
        batch=batch,
        prepared_representation=prepared_representation,
        mcm_module=mcm.module if hasattr(mcm, "module") else mcm,
        tbptt_steps=int(getattr(cfg.model.mcm, "bptt_steps", 0) or 0),
        iteration=0,
    )

    include_backward = not bool(args.forward_only)
    optimizer_step = bool(args.optimizer_step and include_backward)
    use_amp = bool(cfg.training.use_amp and device.type == "cuda")
    scaler = GradScaler("cuda") if use_amp and include_backward else None

    logger.info(
        "Profiling one real batch: rep=%s batch_size=%d fetch=%.3fms iters=%d warmup=%d backward=%s step=%s",
        representation,
        int(_build_batch_summary(batch, representation=representation).get("batch_size", 0)),
        fetch_ms,
        int(args.iters),
        int(args.warmup),
        str(include_backward),
        str(optimizer_step),
    )

    forward_times: list[float] = []
    backward_times: list[float] = []
    optimizer_times: list[float] = []
    total_times: list[float] = []
    peak_memory_mb: list[float] = []
    last_summary: Dict[str, Any] = {}

    def _run_profile_iteration() -> Dict[str, Any]:
        if hasattr(mcm, "reset_state") and not bool(args.keep_state):
            mcm.reset_state()

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _sync(device)

        t_total_0 = time.perf_counter()
        t_fwd_0 = time.perf_counter()
        iteration_forward_kwargs = dict(forward_kwargs)
        if str(representation) == "streaming_events":
            iteration_forward_kwargs["return_debug"] = True

        if use_amp:
            with autocast("cuda"):
                forward_out = mcm(primary_input, **iteration_forward_kwargs)
        else:
            forward_out = mcm(primary_input, **iteration_forward_kwargs)

        if isinstance(forward_out, tuple) and len(forward_out) == 3:
            _corrected, loss, _stream_debug = forward_out
        elif isinstance(forward_out, tuple) and len(forward_out) == 2:
            _corrected, loss = forward_out
        else:
            raise TypeError(
                "Expected MCM forward to return (corrected, loss) or "
                "(corrected, loss, debug) during profiling."
            )

        _sync(device)
        forward_ms = (time.perf_counter() - t_fwd_0) * 1000.0

        backward_ms = 0.0
        step_ms = 0.0
        if include_backward:
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss during profiling: {loss}")

            t_bwd_0 = time.perf_counter()
            if use_amp:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            _sync(device)
            backward_ms = (time.perf_counter() - t_bwd_0) * 1000.0

            if optimizer_step:
                t_step_0 = time.perf_counter()
                grad_clip = float(getattr(cfg.training, "grad_clip", 0.0) or 0.0)
                if grad_clip > 0 and optimizer is not None:
                    torch.nn.utils.clip_grad_norm_(mcm.parameters(), grad_clip)
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    assert optimizer is not None
                    optimizer.step()
                _sync(device)
                step_ms = (time.perf_counter() - t_step_0) * 1000.0

        total_ms = (time.perf_counter() - t_total_0) * 1000.0
        peak_mb = (
            float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
            if device.type == "cuda"
            else 0.0
        )

        summary = _collect_loss_and_flow_summary(mcm)
        summary["loss_total"] = float(loss.detach().item())
        return {
            "forward_ms": forward_ms,
            "backward_ms": backward_ms,
            "optimizer_ms": step_ms,
            "total_ms": total_ms,
            "peak_memory_mb": peak_mb,
            "summary": summary,
        }

    for _ in range(max(int(args.warmup), 0)):
        _run_profile_iteration()

    for _ in range(max(int(args.iters), 1)):
        out = _run_profile_iteration()
        forward_times.append(float(out["forward_ms"]))
        if include_backward:
            backward_times.append(float(out["backward_ms"]))
        if optimizer_step:
            optimizer_times.append(float(out["optimizer_ms"]))
        total_times.append(float(out["total_ms"]))
        peak_memory_mb.append(float(out["peak_memory_mb"]))
        last_summary = dict(out["summary"])

    report: Dict[str, Any] = {
        "config": str(args.config),
        "device": str(device),
        "batch": _build_batch_summary(batch, representation=representation),
        "fetch_ms": float(fetch_ms),
        "timing": {
            "forward_ms_mean": float(statistics.mean(forward_times)),
            "forward_ms_p95": float(_p95(forward_times)),
            "total_ms_mean": float(statistics.mean(total_times)),
            "total_ms_p95": float(_p95(total_times)),
            "peak_memory_mb_max": float(max(peak_memory_mb) if peak_memory_mb else 0.0),
        },
        "sanity": _scalarize(last_summary),
    }

    if include_backward:
        report["timing"]["backward_ms_mean"] = float(statistics.mean(backward_times))
        report["timing"]["backward_ms_p95"] = float(_p95(backward_times))
    if optimizer_step:
        report["timing"]["optimizer_ms_mean"] = float(statistics.mean(optimizer_times))
        report["timing"]["optimizer_ms_p95"] = float(_p95(optimizer_times))

    print(json.dumps(report, indent=2, sort_keys=False))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        logger.info("Saved profile report to %s", output_path)


if __name__ == "__main__":
    main()
