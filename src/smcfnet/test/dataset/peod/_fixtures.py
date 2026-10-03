from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def write_peod_image(path: Path, *, color: tuple[int, int, int], height: int = 720, width: int = 1280) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = np.array(color, dtype=np.uint8)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write image: {path}")


def write_peod_timestamps(path: Path, timestamps_us: Iterable[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for frame_index, timestamp_us in enumerate(timestamps_us, start=1):
            fh.write(f"{frame_index},{int(timestamp_us)}\n")


def write_peod_coco_json(
    path: Path,
    *,
    file_names: list[str],
    image_size: tuple[int, int] = (720, 1280),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = image_size
    payload = {
        "images": [
            {
                "id": idx + 1,
                "file_name": file_name,
                "height": height,
                "width": width,
            }
            for idx, file_name in enumerate(file_names)
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 0,
                "bbox": [123.0, 45.0, 100.0, 50.0],
                "area": 5000.0,
                "iscrowd": 0,
                "ignore": 0,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "bbox": [-10.0, -20.0, 25.0, 40.0],
                "area": 1000.0,
                "iscrowd": 0,
                "ignore": 0,
            },
            {
                "id": 3,
                "image_id": 1,
                "category_id": 2,
                "bbox": [10.0, 20.0, 0.0, 30.0],
                "area": 0.0,
                "iscrowd": 0,
                "ignore": 0,
            },
            {
                "id": 4,
                "image_id": 2,
                "category_id": 3,
                "bbox": [50.0, 60.0, 75.0, 80.0],
                "area": 6000.0,
                "iscrowd": 0,
                "ignore": 0,
            },
            {
                "id": 5,
                "image_id": 2,
                "category_id": 4,
                "bbox": [0.0, 0.0, 10.0, 10.0],
                "area": 100.0,
                "iscrowd": 1,
                "ignore": 0,
            },
        ],
        "categories": [
            {"id": idx, "name": name}
            for idx, name in enumerate(("car", "person", "bus", "truck", "2-wheeler", "3-wheeler"))
        ],
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def write_peod_event_dat(path: Path, records: list[tuple[int, int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = bytearray()
    payload.extend(b"% test dat\n")
    payload.extend(b"% end\n")
    payload.extend(bytes([12, 8]))
    for timestamp_us, x, y, polarity in records:
        packed = (int(x) & 0x3FFF) | ((int(y) & 0x3FFF) << 14) | ((1 if int(polarity) > 0 else 0) << 28)
        payload.extend(np.array([(int(timestamp_us), int(packed))], dtype=[("t", "<u4"), ("e", "<u4")]).tobytes())
    path.write_bytes(bytes(payload))