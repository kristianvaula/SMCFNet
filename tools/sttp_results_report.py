"""Render the latest STTP eval per (design, flow_source, k) from the master CSV.

Reads ``outputs/sttp_results/results.csv`` (or ``--csv <path>``), groups rows by
``(design, flow_source, k)``, and keeps the row with the latest ``timestamp``
that has a numeric ``ade5_px``. Renders a plain-text table by default; pass
``--markdown`` for a thesis-ready table.

Examples:
    # Plain text summary
    python -m tools.sttp_results_report

    # Markdown table
    python -m tools.sttp_results_report --markdown

    # Filter to one design family
    python -m tools.sttp_results_report --design D
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

DESIGN_ORDER = {"B": 0, "C": 1, "D": 2, "trivial": 3}
FLOW_ORDER = {"none": 0, "r3_live": 1, "kristian": 2, "eraft": 3}


def _is_success(row: dict) -> bool:
    if row.get("exit_code") not in ("0",):
        return False
    return bool(row.get("ade5_px"))


def _sort_key(row: dict):
    return (
        DESIGN_ORDER.get(row["design"], 99),
        FLOW_ORDER.get(row["flow_source"], 99),
        int(row["k"]) if row["k"].isdigit() else 0,
    )


def latest_per_group(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    by_group: dict[tuple, dict] = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not _is_success(row):
                continue
            key = (row["design"], row["flow_source"], row["k"])
            prev = by_group.get(key)
            if prev is None or row["timestamp"] > prev["timestamp"]:
                by_group[key] = row
    return sorted(by_group.values(), key=_sort_key)


def render_text(rows: list[dict]) -> str:
    if not rows:
        return "(no successful rows yet)"
    headers = ["design", "flow", "k", "ADE@k px", "mAP50", "mAP", "config_sha", "ts"]
    body = []
    for r in rows:
        body.append(
            [
                r["design"],
                r["flow_source"],
                r["k"],
                f"{float(r['ade5_px']):.4f}",
                f"{float(r['map50']):.4f}",
                f"{float(r['map_full']):.4f}",
                r["config_sha256"][:8],
                r["timestamp"],
            ]
        )
    widths = [max(len(h), max(len(row[i]) for row in body)) for i, h in enumerate(headers)]
    sep = "  ".join("-" * w for w in widths)
    lines = [
        "  ".join(h.ljust(w) for h, w in zip(headers, widths)),
        sep,
        *("  ".join(c.ljust(w) for c, w in zip(row, widths)) for row in body),
    ]
    return "\n".join(lines)


def render_markdown(rows: list[dict]) -> str:
    if not rows:
        return "_(no successful rows yet)_"
    headers = ["Design", "Flow", "k", "ADE@k (px)", "mAP50", "mAP", "Config sha", "Timestamp"]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append(
            "| "
            + " | ".join(
                [
                    r["design"],
                    r["flow_source"],
                    r["k"],
                    f"{float(r['ade5_px']):.4f}",
                    f"{float(r['map50']):.4f}",
                    f"{float(r['map_full']):.4f}",
                    r["config_sha256"][:8],
                    r["timestamp"],
                ]
            )
            + " |"
        )
    return "\n".join(out)


def render_pending(rows: list[dict], all_csv_rows: list[dict]) -> Optional[str]:
    """Show MISSING_RESUME / non-zero-exit rows that haven't yet succeeded."""
    succeeded = {(r["design"], r["flow_source"], r["k"]) for r in rows}
    pending: dict[tuple, dict] = {}
    for r in all_csv_rows:
        key = (r["design"], r["flow_source"], r["k"]) 
        if key in succeeded:
            continue
        prev = pending.get(key)
        if prev is None or r["timestamp"] > prev["timestamp"]:
            pending[key] = r
    if not pending:
        return None
    lines = ["", "Pending / failed (no successful eval yet):"]
    for r in sorted(pending.values(), key=_sort_key):
        lines.append(
            f"  {r['design']:7} {r['flow_source']:8} k={r['k']:>2}  "
            f"exit_code={r['exit_code']}  resume={r['resume_path']}"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path("outputs/sttp_results/results.csv"),
    )
    ap.add_argument("--markdown", action="store_true", help="emit Markdown table")
    ap.add_argument("--design", help="filter to one design (B / C / D / trivial)")
    ap.add_argument(
        "--no-pending",
        action="store_true",
        help="suppress pending/failed summary",
    )
    args = ap.parse_args()

    rows = latest_per_group(args.csv)
    if args.design:
        rows = [r for r in rows if r["design"] == args.design]

    if args.markdown:
        print(render_markdown(rows))
    else:
        print(f"# STTP results — latest per (design, flow, k) — from {args.csv}\n")
        print(render_text(rows))

    if not args.no_pending:
        all_rows: list[dict] = []
        if args.csv.exists():
            with args.csv.open() as f:
                all_rows = list(csv.DictReader(f))
        pending = render_pending(rows, all_rows)
        if pending:
            print(pending)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
