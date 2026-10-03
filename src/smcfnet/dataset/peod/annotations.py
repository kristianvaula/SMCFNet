from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch


PEOD_CATEGORIES: tuple[str, ...] = (
    "car",
    "person",
    "bus",
    "truck",
    "2-wheeler",
    "3-wheeler",
)


@dataclass(frozen=True)
class PEODFrameAnnotation:
    image_id: int
    file_name: str
    width: int
    height: int
    targets: torch.Tensor


def _frame_sort_key(file_name: str) -> tuple[str, int]:
    stem = Path(str(file_name)).stem
    match = re.search(r"_(\d+)$", stem)
    if match is None:
        return stem, -1
    return stem[: match.start()], int(match.group(1))


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"PEOD annotation JSON not found: {path}")
    with path.open("r") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid PEOD annotation JSON, expected object: {path}")
    return data


def _validate_categories(categories: Sequence[Mapping[str, Any]], path: Path) -> None:
    ids_to_names = {int(cat["id"]): str(cat["name"]) for cat in categories}
    expected = {i: name for i, name in enumerate(PEOD_CATEGORIES)}
    if ids_to_names != expected:
        raise ValueError(
            f"Unexpected PEOD categories in {path}: {ids_to_names}. "
            f"Expected {expected}."
        )


def _targets_for_image(
    annotations: Iterable[Mapping[str, Any]],
    *,
    width: int,
    height: int,
) -> torch.Tensor:
    rows: List[List[float]] = []
    for ann in annotations:
        if int(ann.get("ignore", 0)) != 0:
            continue
        if int(ann.get("iscrowd", 0)) != 0:
            continue
        bbox = ann.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        x, y, w, h = [float(v) for v in bbox]
        if not all(torch.isfinite(torch.tensor([x, y, w, h])).tolist()):
            continue
        if w <= 0.0 or h <= 0.0:
            continue

        x1 = max(0.0, min(float(width), x))
        y1 = max(0.0, min(float(height), y))
        x2 = max(0.0, min(float(width), x + w))
        y2 = max(0.0, min(float(height), y + h))
        clipped_w = x2 - x1
        clipped_h = y2 - y1
        if clipped_w <= 0.0 or clipped_h <= 0.0:
            continue

        class_id = int(ann["category_id"])
        rows.append(
            [
                float(class_id),
                x1 + clipped_w / 2.0,
                y1 + clipped_h / 2.0,
                clipped_w,
                clipped_h,
            ]
        )

    if len(rows) == 0:
        return torch.zeros((0, 5), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


def load_peod_coco_annotations(path: str | Path) -> list[PEODFrameAnnotation]:
    """Load one PEOD per-sequence COCO JSON as frame-level YOLO targets."""
    path = Path(path)
    data = _load_json(path)

    categories = data.get("categories", [])
    if isinstance(categories, list):
        _validate_categories(categories, path)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError(f"Invalid PEOD annotation JSON structure: {path}")

    by_image_id: Dict[int, List[Mapping[str, Any]]] = {}
    for ann in annotations:
        if not isinstance(ann, Mapping):
            continue
        image_id = int(ann["image_id"])
        by_image_id.setdefault(image_id, []).append(ann)

    frames: List[PEODFrameAnnotation] = []
    for image in images:
        if not isinstance(image, Mapping):
            continue
        image_id = int(image["id"])
        file_name = str(image["file_name"])
        width = int(image["width"])
        height = int(image["height"])
        targets = _targets_for_image(
            by_image_id.get(image_id, []),
            width=width,
            height=height,
        )
        frames.append(
            PEODFrameAnnotation(
                image_id=image_id,
                file_name=file_name,
                width=width,
                height=height,
                targets=targets,
            )
        )

    frames.sort(key=lambda frame: _frame_sort_key(frame.file_name))
    return frames
