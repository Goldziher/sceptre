"""Publish benchmark numbers from a measured run into the committed docs.

The head-to-head harness (:mod:`sceptre_rs_tools.benchmark`) writes
``benchmark-results/comparison.json``, which is gitignored and machine-local. Every
number quoted in the README and on the docs site used to be copied out of such a run by
hand, so nothing tied a published figure to the run that produced it, and nothing noticed
when the two drifted apart.

This module closes that loop:

* ``publish`` distils a comparison report into ``benchmarks/published/latest.json`` — a
  small, committed, stable-schema artifact carrying the headline figures *and* the
  ``environment`` provenance block naming the runtimes that produced them.
* It then renders the headline table from that artifact into every marked region in the
  README and the docs site, so the published tables are generated output, not prose.
* ``--check`` re-renders from the *committed artifact* and fails if the committed tables
  disagree with it. It never reads the gitignored measurement report, so it works on a
  fresh clone with no benchmark run behind it — no torch, no models, no release binary —
  and is cheap enough to run on every pull request.

A published number is only as good as its provenance: a report with no ``environment``
block is rejected rather than published, because the resulting table could not be
attributed to any particular runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

from sceptre_rs_tools.benchmark import (
    MIN_CLAIMABLE_RATIO,
    PUBLISHED_RELATIVE_PATH,
    HeadlineTotals,
    _batch_totals,
    _mean_of,
    _peak_rss,
    _throughput,
    _warm_speedup,
    paired_rss_ratio,
)

#: Bumped only when an artifact written by an older tool can no longer be rendered
#: correctly — a renamed or removed field, or one whose meaning changed. Adding an
#: *optional* field is not such a change: ``check_docs`` rejects any mismatch, so a bump
#: for a purely additive field would fail the pull-request gate against a perfectly
#: readable committed artifact and demand a full re-measurement to clear it.
#:
#: 2 — ``headline.rss_ratio`` went from the quotient of the two engines' corpus-wide peak-RSS
#: maxima to the median of the per-image ratios (ADR 0042). A version-1 artifact still has
#: every field this renderer reads, but its ``rss_ratio`` would be printed under the new
#: statistic's label and would be a different, unpaired number: rendered, not rendered
#: correctly. The bump forces the re-publish that makes the artifact and its label agree.
SCHEMA_VERSION = 2

MEGABYTES_PER_GIGABYTE = 1024.0

#: Generated prose is wrapped to stay under the repository's 120-column Markdown limit.
PROSE_LINE_WIDTH = 110

#: Provenance fields that describe the operator's machine rather than the runtime, and so
#: must not be committed. ``sceptre_binary`` is an absolute path into a home directory.
LOCAL_ONLY_ENVIRONMENT_FIELDS = frozenset({"sceptre_binary"})


@dataclass(frozen=True)
class MarkedRegion:
    """A generated block in a documentation file, addressed by its marker pair."""

    path: Path
    name: str
    comment_open: str
    comment_close: str

    def start(self) -> str:
        """The opening marker line."""
        return f"{self.comment_open} generated:{self.name}:start {self.comment_close}"

    def end(self) -> str:
        """The closing marker line."""
        return f"{self.comment_open} generated:{self.name}:end {self.comment_close}"


def marked_regions(root: Path) -> list[MarkedRegion]:
    """Every documentation region this module generates.

    The README is plain Markdown and takes HTML comments; the docs site is MDX, where an
    HTML comment is a parse error and the comment form is a JSX expression instead.
    """
    return [
        MarkedRegion(root / "README.md", "benchmark-headline", "<!--", "-->"),
        MarkedRegion(
            root / "docs-site/src/content/docs/reference/benchmarks.mdx",
            "benchmark-headline",
            "{/*",
            "*/}",
        ),
    ]


def totals_from_report(report: dict) -> HeadlineTotals:
    """Rebuild the corpus-wide totals from a serialized comparison report.

    Deserialization only — every figure is computed by the same helpers the benchmark
    itself uses, so a published number cannot drift from how the harness measured it.
    """
    metadata = report.get("metadata", {})
    sceptre_batch = metadata.get("sceptre_batch", {})
    easyocr_batch = metadata.get("easyocr_batch", {})
    sceptre_total, sceptre_images = _batch_totals(sceptre_batch)
    easyocr_total, easyocr_images = _batch_totals(easyocr_batch)
    scored = [record for record in report.get("records", []) if record.get("skipped") is None]
    cold_seconds = [
        record["sceptre_cold_seconds"]
        for record in scored
        if isinstance(record.get("sceptre_cold_seconds"), (int, float))
    ]
    return HeadlineTotals(
        scored_n=len(scored),
        sceptre_warm_total=sceptre_total,
        sceptre_warm_imgs=sceptre_images,
        sceptre_cold_total=sum(cold_seconds),
        easyocr_warm_total=easyocr_total,
        easyocr_warm_imgs=easyocr_images,
        sceptre_peak_rss=_peak_rss(sceptre_batch),
        easyocr_peak_rss=_peak_rss(easyocr_batch),
    )


def _cold_speedup(totals: HeadlineTotals) -> float | None:
    """Warm-EasyOCR versus cold-sceptre speedup, per image, or None."""
    if not (totals.sceptre_cold_total > 0 and totals.scored_n > 0):
        return None
    if not (totals.easyocr_warm_total > 0 and totals.easyocr_warm_imgs > 0):
        return None
    return (totals.easyocr_warm_total / totals.easyocr_warm_imgs) / (totals.sceptre_cold_total / totals.scored_n)


class MissingProvenanceError(RuntimeError):
    """Raised when a report carries no ``metadata.environment`` block."""


def published_payload(report: dict) -> dict:
    """Distil a full comparison report into the committed, stable-schema artifact."""
    metadata = report.get("metadata", {})
    environment = metadata.get("environment")
    if not isinstance(environment, dict) or not environment:
        raise MissingProvenanceError(
            "the report carries no metadata.environment block, so its numbers cannot be "
            "attributed to any runtime; re-run the benchmark with a current harness"
        )
    labeled = report.get("aggregates", {}).get("labeled", {})
    totals = totals_from_report(report)
    scored = [record for record in report.get("records", []) if record.get("skipped") is None]
    labeled_scored = metadata.get("labeled_scored")
    breadth_scored = metadata.get("breadth_scored")
    measured = None
    if isinstance(labeled_scored, int) and isinstance(breadth_scored, int):
        measured = labeled_scored + breadth_scored
    run: dict[str, object] = {
        "platform": metadata.get("platform"),
        "group": metadata.get("group"),
        "repeats": metadata.get("repeats"),
        "threads": metadata.get("threads"),
    }
    runner_label = environment.get("runner_label")
    if isinstance(runner_label, str) and runner_label:
        run["runner_label"] = runner_label
    return {
        "schema_version": SCHEMA_VERSION,
        "environment": {key: value for key, value in environment.items() if key not in LOCAL_ONLY_ENVIRONMENT_FIELDS},
        "run": run,
        "corpus": {
            "total": metadata.get("corpus_total"),
            "measured": measured,
            "labeled_scored": labeled_scored,
            "breadth_scored": breadth_scored,
            "capability_gaps": metadata.get("capability_gaps"),
        },
        "headline": {
            "easyocr_warm": {
                "throughput_img_s": _throughput(totals.easyocr_warm_imgs, totals.easyocr_warm_total),
                "peak_rss_mb": totals.easyocr_peak_rss,
                "mean_cer": _mean_of(labeled, "easyocr_cer"),
                "mean_token_f1": _mean_of(labeled, "easyocr_token_f1"),
            },
            "sceptre_warm": {
                "throughput_img_s": _throughput(totals.sceptre_warm_imgs, totals.sceptre_warm_total),
                "peak_rss_mb": totals.sceptre_peak_rss,
                "mean_cer": _mean_of(labeled, "sceptre_cer"),
                "mean_token_f1": _mean_of(labeled, "sceptre_token_f1"),
            },
            "sceptre_cold": {
                "throughput_img_s": _throughput(totals.scored_n, totals.sceptre_cold_total),
                "peak_rss_mb": totals.sceptre_peak_rss,
                "mean_cer": _mean_of(labeled, "sceptre_cer"),
                "mean_token_f1": _mean_of(labeled, "sceptre_token_f1"),
            },
            "warm_speedup": _warm_speedup(totals),
            "cold_speedup": _cold_speedup(totals),
            "rss_ratio": paired_rss_ratio(
                (record.get("sceptre_rss_mb"), record.get("easyocr_rss_mb")) for record in scored
            ),
        },
    }


def _number(value: object, digits: int) -> str:
    """Format an optional number for a Markdown cell, or an em dash if unmeasured."""
    if not isinstance(value, (int, float)):
        return "—"
    return f"{value:.{digits}f}"


def _rss_cell(megabytes: object, *, unit: str) -> str:
    """Render a peak-RSS figure in gigabytes or megabytes."""
    if not isinstance(megabytes, (int, float)):
        return "—"
    if unit == "GB":
        return f"{megabytes / MEGABYTES_PER_GIGABYTE:.1f} GB"
    return f"{megabytes:,.1f}"


def _ratio_parenthetical(value: object, suffix: str = "") -> str:
    """The ``(~2.3×)`` aside for a ratio, or nothing when there is no claim to make.

    Unmeasured, and too close to parity to distinguish from run-to-run noise
    (:data:`~sceptre_rs_tools.benchmark.MIN_CLAIMABLE_RATIO`), both render as an absent
    aside: a cell that says nothing beats one asserting a non-difference. One decimal
    place, never zero — rounding 1.08 to ``1×`` reads as "1× lower", which claims an
    advantage and states equality in the same breath.
    """
    if not isinstance(value, (int, float)) or value < MIN_CLAIMABLE_RATIO:
        return ""
    return f" (~{value:.1f}×{suffix})"


def render_headline_table(payload: dict, *, unit: str) -> str:
    """Render the headline comparison table in gigabytes (README) or megabytes (docs)."""
    headline = payload["headline"]
    easyocr = headline["easyocr_warm"]
    warm = headline["sceptre_warm"]
    cold = headline["sceptre_cold"]
    rss_header = "Peak RSS" if unit == "GB" else "Peak RSS (MB)"
    warm_speedup = headline.get("warm_speedup")
    cold_speedup = headline.get("cold_speedup")
    rss_ratio = headline.get("rss_ratio")
    warm_throughput = f"**{_number(warm['throughput_img_s'], 2)}**" + _ratio_parenthetical(warm_speedup)
    cold_throughput = _number(cold["throughput_img_s"], 2) + _ratio_parenthetical(cold_speedup)
    # The aside names its own statistic because it is not the quotient of the two cells beside
    # it: those are each engine's corpus-wide maximum, picked independently, while the ratio is
    # the median over images measured in the same batch (ADR 0042). ~keep
    warm_rss = f"**{_rss_cell(warm['peak_rss_mb'], unit=unit)}**" + _ratio_parenthetical(
        rss_ratio, " lower, per-image median"
    )
    rows = [
        [
            "EasyOCR (warm/batch)",
            _number(easyocr["throughput_img_s"], 2),
            _rss_cell(easyocr["peak_rss_mb"], unit=unit),
            _number(easyocr["mean_cer"], 3),
            _number(easyocr["mean_token_f1"], 3),
        ],
        [
            "**sceptre** (warm/batch)",
            warm_throughput,
            warm_rss,
            _number(warm["mean_cer"], 3),
            _number(warm["mean_token_f1"], 3),
        ],
        [
            "sceptre (cold CLI run)",
            cold_throughput,
            _rss_cell(cold["peak_rss_mb"], unit=unit),
            _number(cold["mean_cer"], 3),
            _number(cold["mean_token_f1"], 3),
        ],
    ]
    header = ["Engine", "Throughput (img/s)", rss_header, "Mean CER", "Mean token-F1"]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    lines.append("")
    lines.append(_provenance_line(payload))
    return "\n".join(lines)


def _host_description(payload: dict) -> str:
    """The host these numbers were measured on, named by runner label where there is one.

    ``Linux/x86_64`` cannot distinguish one CI runner class from another, and different
    classes do not produce comparable figures, so the label leads when the run recorded one
    (ADR 0035). A local run records none and reads exactly as it did before.
    """
    environment = payload["environment"]
    host = f"{environment.get('os') or 'unknown'}/{environment.get('arch') or 'unknown'}"
    run = payload.get("run")
    runner_label = run.get("runner_label") if isinstance(run, dict) else None
    if isinstance(runner_label, str) and runner_label:
        return f"{runner_label} ({host})"
    return host


def _provenance_line(payload: dict) -> str:
    """One line naming the runtimes and corpus behind the table above it."""
    environment = payload["environment"]
    reference = environment.get("reference")
    reference = reference if isinstance(reference, dict) else {}
    corpus = payload["corpus"]
    sceptre = environment.get("sceptre_version") or "unknown"
    backend = environment.get("backend") or "unknown"
    accelerator = environment.get("accelerator") or "unknown"
    onnxruntime = environment.get("onnxruntime")
    onnxruntime = onnxruntime.get("version") if isinstance(onnxruntime, dict) else onnxruntime
    onnxruntime = onnxruntime or "unknown"
    easyocr = reference.get("easyocr_version") or "unknown"
    torch = reference.get("torch_version") or "unknown"
    host = _host_description(payload)
    measured = corpus.get("measured")
    total = corpus.get("total")
    line = (
        f"Measured with sceptre {sceptre} ({backend}/{accelerator}, ONNX Runtime {onnxruntime}) "
        f"against EasyOCR {easyocr} (torch {torch}) on {host}, over "
        f"{measured if measured is not None else '—'} of {total if total is not None else '—'} "
        "corpus entries. Regenerate with `task python:benchmark` then `task python:publish`."
    )
    # break_on_hyphens would split a runner label such as `runner-medium` across lines. ~keep
    return "\n".join(textwrap.wrap(line, width=PROSE_LINE_WIDTH, break_long_words=False, break_on_hyphens=False))


def inject(text: str, region: MarkedRegion, body: str) -> str:
    """Replace the content between a region's markers, leaving the markers in place."""
    start = region.start()
    end = region.end()
    start_index = text.find(start)
    if start_index < 0:
        raise ValueError(f"{region.path}: missing marker {start!r}")
    end_index = text.find(end, start_index)
    if end_index < 0:
        raise ValueError(f"{region.path}: missing marker {end!r}")
    head = text[: start_index + len(start)]
    tail = text[end_index:]
    return f"{head}\n\n{body}\n\n{tail}"


def _render_for(region: MarkedRegion, payload: dict) -> str:
    """The generated body for a region, in the unit its file already uses."""
    unit = "GB" if region.path.name == "README.md" else "MB"
    return render_headline_table(payload, unit=unit)


def apply_to_docs(root: Path, payload: dict, *, check: bool) -> list[Path]:
    """Inject the generated tables, or in check mode report which files have drifted."""
    drifted: list[Path] = []
    for region in marked_regions(root):
        original = region.path.read_text(encoding="utf-8")
        updated = inject(original, region, _render_for(region, payload))
        if updated == original:
            continue
        drifted.append(region.path)
        if not check:
            region.path.write_text(updated, encoding="utf-8")
    return drifted


def check_docs(root: Path) -> int:
    """Verify the committed tables still match the committed artifact.

    Reads only files that are in git — never the gitignored measurement report — so this
    runs on a fresh clone with no benchmark run behind it.
    """
    artifact = root / PUBLISHED_RELATIVE_PATH
    if not artifact.is_file():
        print(
            f"sceptre_rs_tools.publish: no published artifact at {PUBLISHED_RELATIVE_PATH}. "
            "Run `task python:benchmark` then `task python:publish` and commit the result.",
            file=sys.stderr,
        )
        return 1
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        print(
            f"sceptre_rs_tools.publish: {PUBLISHED_RELATIVE_PATH} declares schema_version "
            f"{version!r}, but this tool writes {SCHEMA_VERSION}. Re-publish to migrate it.",
            file=sys.stderr,
        )
        return 1
    drifted = apply_to_docs(root, payload, check=True)
    if drifted:
        for path in drifted:
            print(f"sceptre_rs_tools.publish: {path.relative_to(root)} is out of date", file=sys.stderr)
        print(
            f"\nThe committed benchmark tables no longer match {PUBLISHED_RELATIVE_PATH}. "
            "Run `task python:publish` and commit the result.",
            file=sys.stderr,
        )
        return 1
    print("sceptre_rs_tools.publish: published benchmark numbers are up to date.", file=sys.stderr)
    return 0


def publish(root: Path, source: Path, *, check: bool) -> int:
    """Publish from a measured report, or verify the committed state. Returns an exit code."""
    if check:
        return check_docs(root)

    if not source.is_file():
        print(
            f"sceptre_rs_tools.publish: no benchmark report at {source}. Run `task python:benchmark` first.",
            file=sys.stderr,
        )
        return 1
    report = json.loads(source.read_text(encoding="utf-8"))
    try:
        payload = published_payload(report)
    except MissingProvenanceError as error:
        print(f"sceptre_rs_tools.publish: {error}", file=sys.stderr)
        return 1

    artifact = root / PUBLISHED_RELATIVE_PATH
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    drifted = apply_to_docs(root, payload, check=False)
    print(f"Wrote {artifact.relative_to(root)}", file=sys.stderr)
    for path in drifted:
        print(f"Updated {path.relative_to(root)}", file=sys.stderr)
    if not drifted:
        print("Documentation tables were already current.", file=sys.stderr)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the publish command line."""
    parser = argparse.ArgumentParser(
        prog="python -m sceptre_rs_tools.publish",
        description="Publish measured benchmark numbers into the committed docs.",
    )
    parser.add_argument(
        "--from",
        dest="source",
        type=Path,
        default=None,
        help="Comparison report to publish (default: benchmark-results/comparison.json).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the committed numbers match the artifact; write nothing and exit non-zero on drift.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (default: inferred from this file's location).",
    )
    return parser.parse_args(argv)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    root = args.repo_root or Path(__file__).resolve().parents[2]
    source = args.source or root / "benchmark-results" / "comparison.json"
    raise SystemExit(publish(root, source, check=args.check))


if __name__ == "__main__":
    main()
