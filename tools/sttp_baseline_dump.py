#!/usr/bin/env python3
"""Stage 1 of the classical trajectory baseline: dump frozen-detector outputs + GT.

Runs the frozen detector over the val set once per frame and writes, per (sequence,
frame), the post-NMS detections plus the GT future targets at every horizon k. The
stage-2 replay (tools/sttp_baseline_tracker.py) consumes this artifact on CPU with no
dataset access, and scores it with the SAME accumulator the learned head uses, so the
baseline and the learned head are compared on byte-identical detections + GT.

Reuses the exact eval forward path (build_model, the sttp_viz [0,255] RGB +
events_to_voxel_grid_torch convention, postprocess_predictions) so preprocessing cannot
diverge from the learned-head eval. The GT future targets reuse the mapper's own builder
(DSECLightweightMapper, build_future_targets=True) per k, so GT cannot diverge either.

  python -m tools.sttp_baseline_dump --config <v5 detector yaml> \
      --resume outputs/baselines/rgbe_s_halfres_sumfusion_v5_m03_na15/best.pth \
      --out outputs/sttp_results/baseline_dump/val.pt --k 3 --k 5 --k 10

Needs a GPU node + the DSEC dataset (cluster). CPU-only falls back automatically but is slow.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import torch

from smcfnet.config.config import Config, load_config_from_yaml, update_config_from_dict
from smcfnet.model.build import build_model
from smcfnet.dataset.dsec.dsec_lightweight_mapper import DSECLightweightMapper
from smcfnet.dataset.dsec.dsec_sequence_index import build_dsec_sequence_index_datasets
from smcfnet.utils.event_utils import events_to_voxel_grid_torch
from smcfnet.utils.validation import postprocess_predictions

KS_DEFAULT: Tuple[int, ...] = (3, 5, 10)


def _frame_inputs(cfg: Config, sample: dict, device: torch.device):
    """Mirror sttp_viz.core.SttpVizCore._frame_inputs exactly (same RGB/voxel convention)."""
    h, w = int(cfg.data.image_height), int(cfg.data.image_width)
    bins = int(cfg.data.event_bins)
    ev = sample["events"]
    ev = ev if isinstance(ev, torch.Tensor) else torch.as_tensor(ev)
    voxel = events_to_voxel_grid_torch(
        ev.to(device), num_bins=bins, width=w, height=h, source_width=w, source_height=h
    ).unsqueeze(0)
    img = sample["image"]
    img = img if isinstance(img, torch.Tensor) else torch.as_tensor(img)
    img = img.to(device).float()
    if img.dim() == 3:
        img = img.unsqueeze(0)
    if img.shape[-2:] != (h, w):
        img = torch.nn.functional.interpolate(img, size=(h, w), mode="bilinear", align_corners=False)
    return img, voxel


def dump(config_path: str, resume: str, out_path: str, ks: Tuple[int, ...],
         max_frames: int = 0, max_seqs: int = 0) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = Config()
    update_config_from_dict(cfg, load_config_from_yaml(config_path))
    cfg.model.sttp.enabled = False   # detector only; the baseline replaces the head
    cfg.model.mcm.enabled = False

    model = build_model(cfg)
    ckpt = torch.load(resume, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded {resume}: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval().to(device)

    h, w = int(cfg.data.image_height), int(cfg.data.image_width)
    num_classes = int(cfg.model.num_classes)
    conf = float(cfg.eval.conf_threshold)
    nms = float(cfg.eval.nms_threshold)
    split = str(getattr(cfg.eval, "split", "train") or "train")

    per_seq = build_dsec_sequence_index_datasets(
        dsec_yaml_path=Path(str(cfg.data.dataset_config)),
        split=split,
        dt_ms=float(cfg.data.event_window_ms),
        sequences=None,
        use_rectified_events=False, use_rectified_tracks=False,
    )
    if not per_seq:
        raise RuntimeError(f"no sequences for split={split!r}")

    prefer_distorted = bool(getattr(cfg.data, "mapper_prefer_distorted", True))
    # One mapper per k: the future-GT builder is k-specific. rgb/voxel/targets are
    # k-independent so we take them from the primary mapper and only pull each mapper's
    # future_targets, keeping a single detector forward per frame.
    mappers = {
        k: DSECLightweightMapper(
            per_seq, load_events=True, load_images=True, load_tracks=True,
            build_targets=True, build_future_targets=True,
            future_steps=k, future_target_mode="absolute",
            image_height=h, image_width=w, prefer_distorted=prefer_distorted,
        )
        for k in ks
    }
    primary = mappers[ks[0]]

    frames: List[dict] = []
    seqs = per_seq if max_seqs <= 0 else per_seq[:max_seqs]
    for si, seq_ds in enumerate(seqs):
        n = len(seq_ds) if max_frames <= 0 else min(max_frames, len(seq_ds))
        for fi in range(n):
            item = seq_ds[fi]
            sample = primary(item)
            rgb, voxel = _frame_inputs(cfg, sample, device)
            with torch.inference_mode():
                out = model(rgb, voxel)
            raw = out[0] if isinstance(out, tuple) else out  # [B, n_anchors, 5+ncls]
            preds = postprocess_predictions(raw.clone(), num_classes=num_classes,
                                            conf_thre=conf, nms_thre=nms)
            dets = preds[0]
            fts = {}
            for k, mp in mappers.items():
                s_k = mp(item)
                ft = s_k["future_targets"]
                fts[k] = {"boxes": ft["boxes"].cpu(), "validity": ft["validity"].cpu()}
            frames.append({
                "seq": seq_ds.sequence,
                "frame_index": fi,
                "dets_xyxy7": None if dets is None else dets.detach().cpu(),
                "gt_targets": sample["targets"].detach().cpu(),
                "future_targets": fts,
            })
        print(f"[{si + 1}/{len(seqs)}] {seq_ds.sequence}: {n} frames "
              f"(running total {len(frames)})")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(frames, out_path)
    print(f"wrote {len(frames)} frame records -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="detector / v5 run YAML")
    ap.add_argument("--resume", required=True, help="frozen detector checkpoint")
    ap.add_argument("--out", required=True, help="output .pt dump path")
    ap.add_argument("--k", type=int, action="append", default=None, help="horizon(s); repeatable")
    ap.add_argument("--max-frames", type=int, default=0, help="cap frames/sequence (smoke)")
    ap.add_argument("--max-seqs", type=int, default=0, help="cap sequences (smoke)")
    args = ap.parse_args()
    ks = tuple(args.k) if args.k else KS_DEFAULT
    dump(args.config, args.resume, args.out, ks, args.max_frames, args.max_seqs)


if __name__ == "__main__":
    main()
