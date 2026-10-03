# src/smcfnet/tools/sttp_viz/cli.py
"""CLI: pick a diagnostic by name, run it frame-level or run-level."""
from __future__ import annotations
import argparse

from . import diagnostics  # noqa: F401  (triggers @register_diagnostic)
from .registry import get_diagnostic, available_diagnostics


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="sttp_viz", description="STTP visualization diagnostics")
    p.add_argument("--config", required=True, help="run config YAML")
    p.add_argument("--checkpoint", required=True, help="run's best.pth (detector + trained STTP head)")
    p.add_argument("--diagnostic", required=True, help=f"one of: {available_diagnostics()}")
    p.add_argument("--sequence", default=None, help="sequence name; default = first/all val sequences")
    p.add_argument("--frames", type=int, default=0, help="max frames per sequence (0 = all)")
    p.add_argument("--frame-list", default=None,
                   help='exact frames to render: "seq_a:12,340;seq_b:77" (overrides --sequence/--frames)')
    p.add_argument("--conf", type=float, default=0.3,
                   help="detection confidence threshold for viz (eval default 0.001 floods panels)")
    p.add_argument("--style", default="A")
    p.add_argument("--fps", type=int, default=5)
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


def resolve_diagnostic(name, *, out_dir, style, fps):
    cls = get_diagnostic(name)  # raises ValueError on unknown
    return cls(out_dir=out_dir, style=style, fps=fps)


def parse_frame_list(spec):
    """'seq_a:12,340;seq_b:77' -> {'seq_a': [12, 340], 'seq_b': [77]}"""
    if not spec:
        return None
    out = {}
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        seq, _, idxs = part.partition(":")
        if not seq or not idxs:
            raise ValueError(f"bad --frame-list entry {part!r}; expected 'seq:i,j,...'")
        out.setdefault(seq.strip(), []).extend(int(i) for i in idxs.split(",") if i.strip())
    return out


def main(argv=None):
    args = parse_args(argv)
    diag = resolve_diagnostic(args.diagnostic, out_dir=args.out, style=args.style, fps=args.fps)
    if diag.LEVEL == "frame":
        from .core import SttpVizCore
        core = SttpVizCore(args.config, args.checkpoint, device=args.device, conf=args.conf)
        try:
            for bundle in core.iterate_sequence(args.sequence, max_frames=args.frames,
                                                frame_indices=parse_frame_list(args.frame_list)):
                diag.on_frame(bundle)
        finally:
            core.close()
    else:
        raise NotImplementedError(
            f"diagnostic '{args.diagnostic}' is run-level (LEVEL={diag.LEVEL!r}); "
            "run-level diagnostics are not implemented yet"
        )
    diag.finalize()
    print(f"[sttp_viz] {args.diagnostic} done -> {args.out}")
