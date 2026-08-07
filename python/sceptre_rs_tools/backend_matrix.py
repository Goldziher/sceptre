"""Aggregate `backend_matrix.rs` leg reports into one artifact and a step-summary table.

`crates/sceptre/tests/backend_matrix.rs` writes one JSON file per backend/accelerator leg to
`benchmark-results/backends/<leg>.json` (see ADR 0035 for the methodology: fixed
`canvas_size`, one `Reader` per leg, warm-up + repeats, median steady-state timing, and
`model_load_ms` kept separate). Each leg can come from a different CI job — Linux CPU, macOS,
or a GPU runner — so nothing assumes every leg is present in one directory at once; this
module simply combines whatever leg reports it finds.

This is deliberately **not** the ADR 0030 publish pipeline: there is no drift gate here and
no promotion to `benchmarks/published/latest.json`. `--out` writes the combined artifact
(mirroring `benchmarks/published/backends.json`'s schema) and `--summary` renders a Markdown
table suitable for `$GITHUB_STEP_SUMMARY`. Promoting a run's numbers into the committed
`benchmarks/published/backends.json` is a human decision, made by copying `--out`'s output
over it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

DEFAULT_INPUT_DIR = Path("benchmark-results/backends")
DEFAULT_OUTPUT_PATH = Path("benchmark-results/backends-combined.json")


def load_leg_reports(input_dir: Path) -> list[dict[str, Any]]:
    """Every `*.json` leg report under `input_dir`, sorted by leg name for stable output."""
    if not input_dir.is_dir():
        return []
    reports = []
    for path in sorted(input_dir.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            reports.append(json.load(handle))
    return sorted(reports, key=lambda report: report.get("leg", ""))


def combine(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """The combined artifact: `schema_version` plus every leg report, unmodified."""
    return {"schema_version": SCHEMA_VERSION, "legs": reports}


def render_summary(reports: list[dict[str, Any]]) -> str:
    """A Markdown table for `$GITHUB_STEP_SUMMARY`: one row per leg, one row per image."""
    if not reports:
        return "# Backend x accelerator benchmark matrix\n\nNo leg reports were found.\n"

    lines = [
        "# Backend x accelerator benchmark matrix",
        "",
        (
            "Median steady-state timings per leg (see ADR 0035 for methodology). "
            "This is a liveness/throughput measurement, not parity evidence — "
            "`backend_agreement.rs` remains the only correctness bar."
        ),
        "",
        (
            "| Leg | Backend | Accelerator requested | Accelerator registered | Model load (ms) "
            "| Image | Total (ms) | Detect (ms) | Recognize (ms) |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for report in reports:
        leg = report.get("leg", "?")
        backend = report.get("backend", "?")
        requested = report.get("accelerator_requested", "?")
        registered = report.get("accelerator_registered") or "undetermined"
        model_load_ms = report.get("model_load_ms", 0.0)
        images = report.get("images", [])
        if not images:
            lines.append(f"| {leg} | {backend} | {requested} | {registered} | {model_load_ms:.1f} | - | - | - | - |")
            continue
        for image in images:
            median = image.get("median", {})
            lines.append(
                f"| {leg} | {backend} | {requested} | {registered} | {model_load_ms:.1f} "
                f"| {image.get('image', '?')} | {median.get('total_ms', 0.0):.1f} "
                f"| {median.get('detect_ms', 0.0):.1f} | {median.get('recognize_ms', 0.0):.1f} |"
            )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory of per-leg JSON reports (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Combined artifact output path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Path to append the Markdown summary to (e.g. $GITHUB_STEP_SUMMARY)",
    )
    args = parser.parse_args(argv)

    reports = load_leg_reports(args.input_dir)
    if not reports:
        print(f"no leg reports found under {args.input_dir}; nothing to aggregate", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(combine(reports), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(reports)} leg report(s) to {args.out}")

    summary = render_summary(reports)
    if args.summary is not None:
        with args.summary.open("a", encoding="utf-8") as handle:
            handle.write(summary)
    else:
        print(summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
