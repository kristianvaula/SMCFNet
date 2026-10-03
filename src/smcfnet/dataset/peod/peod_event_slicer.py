from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import logging


@dataclass
class PEODEventWindow:
    t_ref_us: int
    events: torch.Tensor
    timestamp_unit: str = "ms"
    sequence_id: Optional[str] = None
    frame_index: Optional[int] = None

    def is_empty(self) -> bool:
        return self.events.numel() == 0


class PEODEventSlicer:
    """Slicer for PEOD Prophesee/PSEE DAT event files.

    Extracted PEOD `.dat` files have an ASCII header made of ``%``-prefixed
    lines, followed by one byte event type, one byte event size, then fixed-size
    8-byte records. Some exports include an explicit ``% end`` line; others
    start the binary type/size bytes immediately after the last header line.

    - uint32 timestamp in microseconds
    - uint32 packed event: x in bits 0..13, y in bits 14..27, polarity bit 28
    """

    def __init__(self, event_path: str | Path):
        self.event_path = Path(event_path)
        if not self.event_path.exists():
            raise FileNotFoundError(f"PEOD event file not found: {self.event_path}")
        self.header: dict[str, str] = {}
        self.event_type: int
        self.event_size: int
        self._records_offset = self._read_header_offset()
        # Create a memmap for the fixed-size 8-byte records. Some DAT files
        # may contain trailing bytes (e.g. from incomplete extraction) which
        # make the available size not an exact multiple of the record size.
        # Use a defensive approach: compute how many full records are
        # available and map only that many, warning if trailing bytes exist.
        record_dtype = np.dtype([("t", "<u4"), ("e", "<u4")])
        file_size = self.event_path.stat().st_size
        remaining = file_size - int(self._records_offset)
        rec_size = int(record_dtype.itemsize)
        if remaining <= 0:
            raise ValueError(f"No event records found in {self.event_path} (offset={self._records_offset})")
        n_records = remaining // rec_size
        if n_records == 0:
            raise ValueError(f"No complete event records found in {self.event_path} (remaining={remaining} bytes)")
        if remaining % rec_size != 0:
            logging.getLogger(__name__).warning(
                "PEOD DAT file %s has %d trailing byte(s) after header; ignoring them",
                self.event_path,
                remaining % rec_size,
            )
        self._events = np.memmap(
            self.event_path,
            mode="r",
            dtype=record_dtype,
            offset=self._records_offset,
            shape=(n_records,),
        )
        self._ms_to_idx: Optional[np.ndarray] = None
        self._t_min_us: int = 0
        self._n_glitches: int = 0
        self._load_ms_index()

    def _load_ms_index(self) -> None:
        """Load the ms-level sidecar index if present and version-compatible."""
        idx_path = self.event_path.with_suffix(self.event_path.suffix + ".ms_idx.npz")
        if not idx_path.exists():
            return
        try:
            with np.load(idx_path) as data:
                version = int(data["version"]) if "version" in data else 0
                if version != 1:
                    logging.getLogger(__name__).warning(
                        "PEOD ms-index %s has unsupported version %d; falling back to full searchsorted",
                        idx_path.name, version,
                    )
                    return
                self._ms_to_idx = data["ms_to_idx"][:]
                self._t_min_us = int(data["t_min_us"])
                self._n_glitches = int(data["n_glitches"]) if "n_glitches" in data else 0
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Failed to load PEOD ms-index %s: %s; falling back to full searchsorted",
                idx_path.name, exc,
            )

    # Extra ms to search on each side when the sidecar recorded clock glitches.
    # Glitches ≤ 5 ms are tolerated during build; a 6 ms margin ensures events
    # that landed in a slightly-wrong bucket are still found by searchsorted.
    _GLITCH_MARGIN_MS: int = 6

    def _resolve_search_bounds(self, t_us: int) -> tuple[int, int]:
        """Return (lo, hi) record indices to narrow searchsorted for t_us.

        When the ms-index is loaded, uses a coarse O(1) lookup that covers a
        ~1 ms bracket, matching what DSECEventSlicer does. Falls back to the
        full array bounds when no index is available.
        For sequences with recorded clock glitches the bracket is widened by
        _GLITCH_MARGIN_MS on each side so misassigned events are still found.
        """
        n = len(self._events)
        if self._ms_to_idx is None:
            return 0, n
        rel_ms = (t_us - self._t_min_us) // 1000
        if rel_ms < 0:
            return 0, 0
        margin = self._GLITCH_MARGIN_MS if self._n_glitches > 0 else 0
        idx_len = len(self._ms_to_idx)
        lo = int(self._ms_to_idx[min(max(0, rel_ms - margin), idx_len - 1)])
        hi = int(self._ms_to_idx[min(rel_ms + 1 + margin, idx_len - 1)])
        return max(0, lo), min(hi, n)

    def _read_header_offset(self) -> int:
        header_lines: list[bytes] = []
        with self.event_path.open("rb") as fh:
            head = fh.read(8192)
            pos = 0
            while True:
                if pos >= len(head) or head[pos:pos + 1] != b"%":
                    break
                next_nl = head.find(b"\n", pos)
                if next_nl < 0:
                    raise ValueError(f"PEOD DAT header line exceeds 8192 bytes: {self.event_path}")
                line = head[pos:next_nl + 1]
                header_lines.append(line)
                if line.strip() == b"% end":
                    pos = next_nl + 1
                    break
                pos = next_nl + 1

            if pos >= len(head):
                raise ValueError(f"PEOD DAT header did not expose type/size bytes within 8192 bytes: {self.event_path}")
            type_size = head[pos:pos + 2]
            if len(type_size) != 2:
                raise ValueError(f"PEOD DAT file missing type/size bytes: {self.event_path}")
            self.event_type = int(type_size[0])
            self.event_size = int(type_size[1])
            if self.event_size != 8:
                raise ValueError(
                    f"Unsupported PEOD DAT event size {self.event_size} in {self.event_path}; expected 8"
                )
            records_offset = pos + 2

        for raw in header_lines:
            text = raw.decode("utf-8", errors="ignore").strip()
            if not text.startswith("%"):
                continue
            text = text[1:].strip()
            if text == "end" or " " not in text:
                continue
            key, value = text.split(" ", 1)
            self.header[key.lower()] = value.strip()
        return records_offset

    def get_window_us(
        self,
        t_start_us: int,
        t_end_us: int,
        *,
        sequence_id: Optional[str] = None,
        frame_index: Optional[int] = None,
        timestamp_unit: str = "ms",
    ) -> PEODEventWindow:
        if timestamp_unit not in {"ms", "us"}:
            raise ValueError(f"timestamp_unit must be 'ms' or 'us', got {timestamp_unit!r}")
        if t_end_us <= t_start_us:
            raise ValueError(f"Expected t_end_us > t_start_us, got {t_start_us}, {t_end_us}")

        t_arr = self._events["t"]
        lo_s, hi_s = self._resolve_search_bounds(int(t_start_us))
        lo_e, hi_e = self._resolve_search_bounds(int(t_end_us))
        start = lo_s + int(np.searchsorted(t_arr[lo_s:hi_s], int(t_start_us), side="left"))
        end = lo_e + int(np.searchsorted(t_arr[lo_e:hi_e], int(t_end_us), side="left"))
        if end <= start:
            events = torch.empty((0, 4), dtype=torch.int64)
            return PEODEventWindow(
                t_ref_us=int(t_end_us),
                events=events,
                timestamp_unit=timestamp_unit,
                sequence_id=sequence_id,
                frame_index=frame_index,
            )

        records = self._events[start:end]
        packed = np.asarray(records["e"], dtype=np.uint32)
        xs = (packed & np.uint32(0x3FFF)).astype(np.int64)
        ys = ((packed >> np.uint32(14)) & np.uint32(0x3FFF)).astype(np.int64)
        ps = np.where(((packed >> np.uint32(28)) & np.uint32(1)) > 0, 1, -1).astype(np.int64)
        ts_us = np.asarray(records["t"], dtype=np.int64)
        if timestamp_unit == "ms":
            ts = ts_us // 1000
        else:
            ts = ts_us

        ev_np = np.stack([xs, ys, ts, ps], axis=1).astype(np.int64, copy=False)
        return PEODEventWindow(
            t_ref_us=int(t_end_us),
            events=torch.from_numpy(ev_np).contiguous(),
            timestamp_unit=timestamp_unit,
            sequence_id=sequence_id,
            frame_index=frame_index,
        )
