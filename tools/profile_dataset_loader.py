"""Profile and compare dataloader components across datasets (DSEC vs PEOD).

Two passes per config:
  1. Sequential mapper micro-benchmark (single process):
     times event read, image read, and target build per sample independently.
  2. DataLoader macro-benchmark with variable num_workers:
     times end-to-end batch wall time including worker dispatch and collate.

Usage:
  python tools/profile_dataset_loader.py \\
    --configs src/smcfnet/config/runs/baselines/peod/train/rgbe_cmm_m.yaml \\
              src/smcfnet/config/runs/baselines/train/rgbe_edum_cmm_m.yaml \\
    --warmup 5 --measured 60 \\
    --num-workers 1 2 4 \\
    --batch-size 1 \\
    --output-csv outputs/preliminary/loader_profile.csv
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ---------------------------------------------------------------------------
# Config helpers (reuse the same pattern as run_preliminary_ddp_benchmark.py)
# ---------------------------------------------------------------------------

def _load_config(base_config: Path, *, batch_size: int = 1, num_workers: int = 2) -> Any:
    from smcfnet.config.config import Config, load_config_from_yaml, update_config_from_dict

    cfg = Config()
    loaded = load_config_from_yaml(str(base_config))
    update_config_from_dict(cfg, loaded)

    cfg.num_gpus = 1
    cfg.training.num_epochs = 1
    cfg.training.batch_size = int(batch_size)
    cfg.training.max_steps_per_epoch = 9999
    cfg.training.max_total_steps = 0
    cfg.log.output_dir = "/tmp/profile_dataset_loader"
    cfg.log.experiment_name = "profile"
    cfg.log.save_all_checkpoints = False
    cfg.log.log_interval = 99999
    cfg.log.memory_log_interval = 0
    cfg.wandb.use_wandb = False
    cfg.wandb.mode = "disabled"
    cfg.eval.interval = 0
    cfg.eval.save_interval = 0

    cfg.data.num_workers = int(num_workers)
    cfg.data.pin_memory = True
    cfg.data.prefetch_factor = 2
    cfg.data.persistent_workers = True
    cfg.data.drop_last = True
    cfg.data.mapper_load_images = True
    cfg.data.mapper_load_tracks = True
    cfg.data.mapper_build_targets = True
    return cfg


def _dataset_label(cfg: Any) -> str:
    from smcfnet.model.build import _resolve_dataset_config_path, _dataset_name_from_config_path
    p = _resolve_dataset_config_path(cfg, None)
    return _dataset_name_from_config_path(p)


# ---------------------------------------------------------------------------
# Mapper micro-benchmark helpers
# ---------------------------------------------------------------------------

def _get_slicer_and_call(mapper: Any, seq_idx: int, t_start_us: int, t_end_us: int, frame_index: int):
    """Return (slicer_call_time_s, n_events) for one event window read."""
    mapper_cls = type(mapper).__name__

    if mapper_cls == "DSECLightweightMapper":
        mapper._ensure_events_open(seq_idx)
        slicer = mapper._slicer[seq_idx]
        seq_id = mapper.sequence_datasets[seq_idx].sequence
        ts_unit = getattr(mapper, "event_timestamp_unit", "ms")
        t0 = time.perf_counter()
        window = slicer.get_window_us(
            t_start_us, t_end_us,
            sequence_id=seq_id,
            frame_index=frame_index,
            timestamp_unit=ts_unit,
        )
        dt = time.perf_counter() - t0
        n = int(window.events.shape[0]) if window is not None else 0

    elif mapper_cls == "PEODLightweightMapper":
        ts_unit = getattr(mapper, "event_timestamp_unit", "ms")
        slicer = mapper._ensure_slicer(seq_idx)
        t0 = time.perf_counter()
        window = slicer.get_window_us(
            t_start_us, t_end_us,
            frame_index=frame_index,
            timestamp_unit=ts_unit,
        )
        dt = time.perf_counter() - t0
        n = int(window.events.shape[0])

    else:
        raise NotImplementedError(f"Unknown mapper class: {mapper_cls}")

    return dt, n


def _get_image_time(mapper: Any, seq_idx: int, frame_index: int) -> float:
    t0 = time.perf_counter()
    mapper._load_image_tensor(seq_idx, frame_index)
    return time.perf_counter() - t0


def _get_target_time(mapper: Any, seq_idx: int, frame_index: int, t_ref_us: int) -> float:
    mapper_cls = type(mapper).__name__
    img_h = int(getattr(mapper, "image_height", 480) or 480)
    img_w = int(getattr(mapper, "image_width", 640) or 640)
    t0 = time.perf_counter()
    if mapper_cls == "DSECLightweightMapper":
        mapper._targets_for_frame(seq_idx, t_ref_us, img_h=img_h, img_w=img_w)
    else:
        mapper._targets_for_frame(seq_idx, frame_index)
    return time.perf_counter() - t0


def _profile_mapper(mapper: Any, dataset: Any, *, n_warmup: int, n_measured: int, seed: int = 0) -> dict:
    """Time mapper components sequentially in the main process."""
    rng = np.random.default_rng(seed)
    n_total = len(dataset)
    indices = rng.integers(0, n_total, size=n_warmup + n_measured).tolist()

    ev_times, img_times, tgt_times, total_times, n_events_list = [], [], [], [], []

    for i, idx in enumerate(indices):
        item = dataset[idx]
        seq_idx = int(item["seq_idx"])
        frame_index = int(item["frame_index"])
        t_start_us = int(item["t_start_us"])
        t_end_us = int(item["t_end_us"])
        t_ref_us = int(item.get("t_ref_us", t_end_us))

        t_ev, n_ev = _get_slicer_and_call(mapper, seq_idx, t_start_us, t_end_us, frame_index)
        t_img = _get_image_time(mapper, seq_idx, frame_index)
        t_tgt = _get_target_time(mapper, seq_idx, frame_index, t_ref_us)

        if i < n_warmup:
            continue  # discard warmup

        ev_times.append(t_ev)
        img_times.append(t_img)
        tgt_times.append(t_tgt)
        total_times.append(t_ev + t_img + t_tgt)
        n_events_list.append(n_ev)

    def _stats(arr: list) -> dict:
        a = np.array(arr)
        return {
            "mean_ms": float(np.mean(a) * 1e3),
            "p50_ms": float(np.percentile(a, 50) * 1e3),
            "p90_ms": float(np.percentile(a, 90) * 1e3),
            "p95_ms": float(np.percentile(a, 95) * 1e3),
        }

    return {
        "event_read": _stats(ev_times),
        "image_read": _stats(img_times),
        "target_build": _stats(tgt_times),
        "total": _stats(total_times),
        "n_events": {
            "mean": float(np.mean(n_events_list)),
            "p50": float(np.percentile(n_events_list, 50)),
            "p95": float(np.percentile(n_events_list, 95)),
        },
    }


# ---------------------------------------------------------------------------
# DataLoader macro-benchmark
# ---------------------------------------------------------------------------

def _profile_dataloader(base_config: Path, *, batch_size: int, num_workers: int,
                        n_warmup: int, n_measured: int,
                        sequences: list[str] | None = None) -> dict:
    from smcfnet.model.build import build_dataset_loader_cfg

    cfg = _load_config(base_config, batch_size=batch_size, num_workers=num_workers)
    loader = build_dataset_loader_cfg(cfg, split="train", is_train=True, sequences=sequences)

    batch_times: list[float] = []
    it = iter(loader)

    for _ in range(n_warmup):
        next(it)

    for _ in range(n_measured):
        t0 = time.perf_counter()
        next(it)
        batch_times.append(time.perf_counter() - t0)

    a = np.array(batch_times)
    return {
        "batch_ms_mean": float(np.mean(a) * 1e3),
        "batch_ms_p50": float(np.percentile(a, 50) * 1e3),
        "batch_ms_p90": float(np.percentile(a, 90) * 1e3),
        "sample_ms_mean": float(np.mean(a) * 1e3 / batch_size),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_mapper_table(label: str, config_name: str, result: dict) -> None:
    r = result
    print(f"\n{'='*66}")
    print(f"  Mapper micro-benchmark  |  {label}  |  {config_name}")
    print(f"{'='*66}")
    print(f"  {'Component':<18}  {'mean ms':>8}  {'p50 ms':>8}  {'p90 ms':>8}  {'p95 ms':>8}")
    print(f"  {'-'*58}")
    for key, name in [("event_read", "event read"), ("image_read", "image read"),
                      ("target_build", "target build"), ("total", "TOTAL")]:
        s = r[key]
        print(f"  {name:<18}  {s['mean_ms']:>8.1f}  {s['p50_ms']:>8.1f}  {s['p90_ms']:>8.1f}  {s['p95_ms']:>8.1f}")
    n = r["n_events"]
    print(f"\n  Events/sample:  mean={n['mean']:.0f}  p50={n['p50']:.0f}  p95={n['p95']:.0f}")


def _print_loader_table(label: str, config_name: str, results_by_nw: dict) -> None:
    print(f"\n{'='*66}")
    print(f"  DataLoader macro-benchmark  |  {label}  |  {config_name}")
    print(f"{'='*66}")
    print(f"  {'num_workers':>11}  {'batch ms mean':>14}  {'batch ms p50':>13}  {'batch ms p90':>13}  {'ms/sample':>10}")
    print(f"  {'-'*60}")
    for nw, r in sorted(results_by_nw.items()):
        print(f"  {nw:>11}  {r['batch_ms_mean']:>14.1f}  {r['batch_ms_p50']:>13.1f}  {r['batch_ms_p90']:>13.1f}  {r['sample_ms_mean']:>10.1f}")


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def _save_csv(all_results: list[dict], path: Path) -> None:
    import csv
    rows = []
    for entry in all_results:
        label = entry["label"]
        config = entry["config"]
        mr = entry.get("mapper")
        if mr:
            for comp in ("event_read", "image_read", "target_build", "total"):
                s = mr[comp]
                rows.append({
                    "label": label, "config": config, "phase": "mapper",
                    "component": comp,
                    "mean_ms": round(s["mean_ms"], 3), "p50_ms": round(s["p50_ms"], 3),
                    "p90_ms": round(s["p90_ms"], 3), "p95_ms": round(s["p95_ms"], 3),
                    "num_workers": 0, "batch_size": "",
                    "n_events_mean": round(mr["n_events"]["mean"], 1),
                })
        for nw, dr in entry.get("dataloader", {}).items():
            rows.append({
                "label": label, "config": config, "phase": "dataloader",
                "component": "batch", "mean_ms": round(dr["batch_ms_mean"], 3),
                "p50_ms": round(dr["batch_ms_p50"], 3), "p90_ms": round(dr["batch_ms_p90"], 3),
                "p95_ms": "", "num_workers": nw, "batch_size": entry["batch_size"],
                "n_events_mean": "",
            })

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", nargs="+", required=True, metavar="YAML",
                   help="One or more run config YAML paths (DSEC or PEOD)")
    p.add_argument("--warmup", type=int, default=5, metavar="N",
                   help="Warmup samples/batches to discard (default: 5)")
    p.add_argument("--measured", type=int, default=60, metavar="N",
                   help="Measured samples/batches per variant (default: 60)")
    p.add_argument("--num-workers", type=int, nargs="+", default=[1, 2, 4],
                   metavar="W", help="num_workers values to sweep for DataLoader bench")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Batch size for DataLoader bench (default: 1)")
    p.add_argument("--sequences", nargs="+", default=None, metavar="SEQ",
                   help="Restrict to specific sequence names (default: all)")
    p.add_argument("--skip-mapper", action="store_true",
                   help="Skip the sequential mapper micro-benchmark")
    p.add_argument("--skip-dataloader", action="store_true",
                   help="Skip the DataLoader macro-benchmark")
    p.add_argument("--output-csv", type=Path, default=None,
                   help="Optional path to save results as CSV")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sample selection")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    import logging
    logging.basicConfig(level=logging.WARNING)

    all_results: list[dict] = []

    for config_path_str in args.configs:
        config_path = Path(config_path_str)
        if not config_path.is_absolute():
            config_path = (PROJECT_ROOT / config_path).resolve()
        if not config_path.exists():
            print(f"[ERROR] Config not found: {config_path}", file=sys.stderr)
            sys.exit(1)

        config_name = config_path.stem
        print(f"\n{'#'*66}")
        print(f"  Config: {config_path.name}")

        # Build config + components for the mapper bench
        cfg = _load_config(config_path, batch_size=args.batch_size, num_workers=2)
        dataset_label = _dataset_label(cfg)
        print(f"  Dataset: {dataset_label}")

        entry: dict = {
            "label": dataset_label,
            "config": config_name,
            "batch_size": args.batch_size,
        }

        if not args.skip_mapper:
            from smcfnet.model.build import build_dataset_loader_cfg
            print(f"\n  [mapper bench] warming up {args.warmup} + measuring {args.measured} samples ...")
            components = build_dataset_loader_cfg(
                cfg, split="train", is_train=True, return_components=True,
                sequences=args.sequences,
            )
            mapper = components["mapper"]
            dataset = components["dataset"]

            mapper_result = _profile_mapper(
                mapper, dataset,
                n_warmup=args.warmup,
                n_measured=args.measured,
                seed=args.seed,
            )
            entry["mapper"] = mapper_result
            _print_mapper_table(dataset_label, config_name, mapper_result)

        if not args.skip_dataloader:
            loader_results: dict[int, dict] = {}
            for nw in args.num_workers:
                print(f"\n  [dataloader bench] num_workers={nw} — "
                      f"warmup {args.warmup} + measured {args.measured} batches ...")
                loader_results[nw] = _profile_dataloader(
                    config_path,
                    batch_size=args.batch_size,
                    num_workers=nw,
                    n_warmup=args.warmup,
                    n_measured=args.measured,
                    sequences=args.sequences,
                )
            entry["dataloader"] = loader_results
            _print_loader_table(dataset_label, config_name, loader_results)

        all_results.append(entry)

    if args.output_csv and all_results:
        _save_csv(all_results, args.output_csv)

    print("\nDone.")


if __name__ == "__main__":
    main()
