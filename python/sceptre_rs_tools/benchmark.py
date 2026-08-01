"""Head-to-head benchmark: sceptre (release CLI) vs upstream Python EasyOCR.

Runs both OCR engines over a shared image corpus and reports cross-engine agreement,
absolute accuracy against ground truth (labeled images), speed, and best-effort peak
memory. Results are written to ``benchmark-results/comparison.{json,md}``.

Fairness notes baked into the harness:
  * sceptre is invoked as the RELEASE binary; a debug build would be unfairly slow.
  * EasyOCR is timed warm (Reader constructed once per language set, reused); sceptre
    is invoked as a fresh process each time (cold: includes ONNX init + model parse),
    so an "amortized inner" time subtracts a measured fixed startup/model-load cost.
  * Memory figures are not directly comparable: the EasyOCR RSS includes the whole
    Python + torch runtime, while sceptre's is a lean native process.

This is a dev-only tool; it needs the opt-in ``export`` dependency group (torch,
easyocr). Without it, the run exits cleanly with an explanatory message.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

from sceptre_rs_tools.corpus import CorpusEntry, build_corpus

# sceptre release binary produced by the ort-bundled CLI build.
SCEPTRE_BIN = Path("target/release/sceptre")
OUTPUT_DIR = Path("benchmark-results")
OVERHEAD_RUNS = 3  # process launches used to estimate sceptre's fixed startup cost

# Optional accelerated edit distance; the harness falls back to a stdlib DP if absent.
try:
    from rapidfuzz.distance import Levenshtein as _rapidfuzz_levenshtein
except ImportError:
    _rapidfuzz_levenshtein = None


def repo_root() -> Path:
    """Locate the repository root from this file's location."""
    return Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------------------
# Text metrics (mirror the definitions in crates/sceptre/tests/helpers/mod.rs)
# --------------------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Whitespace-tokenize and case-fold, matching the Rust ``word_f1`` tokenizer."""
    return [token.lower() for token in text.split()]


def _char_bag(text: str) -> list[str]:
    """Lowercased non-whitespace characters, matching the Rust ``char_f1`` bag."""
    return [character.lower() for character in text if not character.isspace()]


def _multiset_f1(hypothesis: list[str], reference: list[str]) -> float:
    """Multiset precision/recall F1 over two token bags (empty/empty scores 1.0)."""
    if not hypothesis and not reference:
        return 1.0
    if not hypothesis or not reference:
        return 0.0
    remaining = list(reference)
    matched = 0
    for token in hypothesis:
        if token in remaining:
            remaining.remove(token)
            matched += 1
    if matched == 0:
        return 0.0
    precision = matched / len(hypothesis)
    recall = matched / len(reference)
    return 2.0 * precision * recall / (precision + recall)


def word_f1(hypothesis: str, reference: str) -> float:
    """Bag-of-words F1 between two strings (whitespace-tokenized, case-folded)."""
    return _multiset_f1(_tokenize(hypothesis), _tokenize(reference))


def char_f1(hypothesis: str, reference: str) -> float:
    """Bag-of-characters F1 between two strings (whitespace ignored, case-folded)."""
    return _multiset_f1(_char_bag(hypothesis), _char_bag(reference))


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection-over-union of two axis-aligned boxes ``(x_min, y_min, x_max, y_max)``."""
    x_min = max(a[0], b[0])
    y_min = max(a[1], b[1])
    x_max = min(a[2], b[2])
    y_max = min(a[3], b[3])
    intersection = max(x_max - x_min, 0.0) * max(y_max - y_min, 0.0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def quad_bbox(quad: list[list[float]]) -> tuple[float, float, float, float]:
    """Axis-aligned bounds ``(x_min, y_min, x_max, y_max)`` of a four-corner quad."""
    xs = [point[0] for point in quad]
    ys = [point[1] for point in quad]
    return (min(xs), min(ys), max(xs), max(ys))


def mean_best_line_iou(reference_quads: list[list[list[float]]], hypothesis_quads: list[list[list[float]]]) -> float:
    """Mean over reference lines of the best box-IoU against any hypothesis line."""
    if not reference_quads:
        return 1.0 if not hypothesis_quads else 0.0
    if not hypothesis_quads:
        return 0.0
    hypothesis_boxes = [quad_bbox(quad) for quad in hypothesis_quads]
    total = 0.0
    for reference_quad in reference_quads:
        reference_box = quad_bbox(reference_quad)
        total += max(box_iou(reference_box, box) for box in hypothesis_boxes)
    return total / len(reference_quads)


# --------------------------------------------------------------------------------------
# Ground-truth accuracy (CER / WER / token-F1)
# --------------------------------------------------------------------------------------

_MARKDOWN_CHARS = re.compile(r"[#|*`_>]")
_LIST_BULLET = re.compile(r"^\s*(?:[-+•]|\d+[.)])\s+", re.MULTILINE)
_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Strip light Markdown, lowercase, and collapse whitespace to single spaces.

    Applied identically to reference and hypothesis before scoring so document markup
    and reading-order line breaks do not dominate the edit distance.
    """
    without_bullets = _LIST_BULLET.sub(" ", text)
    without_markup = _MARKDOWN_CHARS.sub(" ", without_bullets)
    collapsed = _WHITESPACE.sub(" ", without_markup)
    return collapsed.strip().lower()


def edit_distance(a: list[str], b: list[str]) -> int:
    """Levenshtein distance between two sequences (rapidfuzz if available, else DP)."""
    if _rapidfuzz_levenshtein is not None:
        return int(_rapidfuzz_levenshtein.distance(a, b))
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, item_a in enumerate(a, start=1):
        current = [i]
        for j, item_b in enumerate(b, start=1):
            cost = 0 if item_a == item_b else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float | None:
    """Order-sensitive CER = char edit distance / reference length (None if empty ref)."""
    reference_chars = list(reference)
    if not reference_chars:
        return None
    return edit_distance(reference_chars, list(hypothesis)) / len(reference_chars)


def word_error_rate(reference: str, hypothesis: str) -> float | None:
    """Order-sensitive WER = word edit distance / reference word count (None if empty)."""
    reference_words = reference.split()
    if not reference_words:
        return None
    return edit_distance(reference_words, hypothesis.split()) / len(reference_words)


# --------------------------------------------------------------------------------------
# Engine drivers
# --------------------------------------------------------------------------------------


@dataclass
class Detection:
    """One recognized line: text plus its four-corner quad."""

    text: str
    quad: list[list[float]]


@dataclass
class EngineOutput:
    """The result of running one engine over one image."""

    detections: list[Detection]
    seconds: float
    rss_mb: float | None = None

    @property
    def text(self) -> str:
        """All line texts joined with single spaces, in engine order."""
        return " ".join(detection.text for detection in self.detections)

    @property
    def quads(self) -> list[list[list[float]]]:
        """All line quads."""
        return [detection.quad for detection in self.detections]


def easyocr_detections(reader: object, image: Path) -> list[Detection]:
    """Run an EasyOCR reader over one image and normalize to ``Detection`` records."""
    raw = reader.readtext(str(image))  # type: ignore[attr-defined]
    detections: list[Detection] = []
    for box, text, _confidence in raw:
        quad = [[float(point[0]), float(point[1])] for point in box[:4]]
        detections.append(Detection(text=str(text), quad=quad))
    return detections


def run_easyocr(reader: object, image: Path) -> EngineOutput:
    """Time a warm EasyOCR ``readtext`` call and capture the process peak RSS."""
    import resource

    start = perf_counter()
    detections = easyocr_detections(reader, image)
    seconds = perf_counter() - start
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return EngineOutput(detections=detections, seconds=seconds, rss_mb=_maxrss_to_mb(peak))


def _maxrss_to_mb(value: int) -> float:
    """Normalize ``ru_maxrss`` to MB (bytes on darwin, KiB on linux)."""
    if sys.platform == "darwin":
        return value / 1_000_000.0
    return value * 1024 / 1_000_000.0


_TIME_RSS_DARWIN = re.compile(r"(\d+)\s+maximum resident set size")
_TIME_RSS_LINUX = re.compile(r"Maximum resident set size \(kbytes\):\s+(\d+)")


def _time_wrapper() -> list[str]:
    """The ``/usr/bin/time`` prefix that reports peak RSS, or empty if unavailable."""
    time_bin = Path("/usr/bin/time")
    if not time_bin.exists():
        return []
    return [str(time_bin), "-l" if sys.platform == "darwin" else "-v"]


def _parse_child_rss(stderr: str) -> float | None:
    """Extract child peak RSS in MB from ``/usr/bin/time`` stderr output."""
    match = _TIME_RSS_DARWIN.search(stderr)
    if match:  # darwin reports bytes
        return int(match.group(1)) / 1_000_000.0
    match = _TIME_RSS_LINUX.search(stderr)
    if match:  # linux reports KiB
        return int(match.group(1)) * 1024 / 1_000_000.0
    return None


def _parse_sceptre_json(payload: str) -> list[Detection]:
    """Parse sceptre ``run --format json`` stdout into ``Detection`` records."""
    data = json.loads(payload)
    detections: list[Detection] = []
    for line in data.get("lines", []):
        points = line.get("quad", {}).get("points", [])
        quad = [[float(point["x"]), float(point["y"])] for point in points]
        detections.append(Detection(text=str(line.get("text", "")), quad=quad))
    return detections


def sceptre_command(binary: Path, image: Path, sceptre_langs: list[str]) -> list[str]:
    """Build the sceptre CLI argument vector for one image."""
    command = [str(binary), "run", str(image)]
    for language in sceptre_langs:
        command += ["--lang", language]
    command += ["--format", "json"]
    return command


def run_sceptre(binary: Path, image: Path, sceptre_langs: list[str], root: Path) -> EngineOutput:
    """Invoke the sceptre CLI as a fresh process; time the full cold wall clock."""
    command = sceptre_command(binary, image, sceptre_langs)
    wrapped = _time_wrapper() + command
    start = perf_counter()
    completed = subprocess.run(wrapped, capture_output=True, text=True, cwd=root, check=False)
    seconds = perf_counter() - start
    if completed.returncode != 0:
        raise RuntimeError(f"sceptre exited {completed.returncode}: {completed.stderr.strip()[-500:]}")
    detections = _parse_sceptre_json(completed.stdout)
    return EngineOutput(detections=detections, seconds=seconds, rss_mb=_parse_child_rss(completed.stderr))


def measure_sceptre_overhead(binary: Path, entries: list[CorpusEntry], root: Path) -> float | None:
    """Estimate sceptre's fixed startup + model-load cost.

    Uses the smallest available English image, launched ``OVERHEAD_RUNS`` times, and
    returns the median cold wall time. This approximates the per-process fixed cost
    (ONNX init + english model parse) that the amortized-inner time subtracts.
    """
    candidates = [
        entry
        for entry in entries
        if entry.image is not None and entry.image.exists() and entry.sceptre_langs == ["english"]
    ]
    if not candidates:
        return None
    smallest = min(candidates, key=lambda entry: entry.image.stat().st_size)  # type: ignore[union-attr]
    timings: list[float] = []
    for _ in range(OVERHEAD_RUNS):
        try:
            timings.append(run_sceptre(binary, smallest.image, ["english"], root).seconds)  # type: ignore[arg-type]
        except (RuntimeError, ValueError):
            return None
    return statistics.median(timings)


# --------------------------------------------------------------------------------------
# Per-image benchmark record
# --------------------------------------------------------------------------------------


@dataclass
class ImageRecord:
    """All measured metrics for a single corpus image."""

    stem: str
    group: str
    languages: list[str]
    skipped: str | None = None
    easyocr_seconds: float | None = None
    sceptre_cold_seconds: float | None = None
    sceptre_amortized_seconds: float | None = None
    easyocr_rss_mb: float | None = None
    sceptre_rss_mb: float | None = None
    agreement_char_f1: float | None = None
    agreement_word_f1: float | None = None
    agreement_mean_iou: float | None = None
    easyocr_cer: float | None = None
    easyocr_wer: float | None = None
    easyocr_token_f1: float | None = None
    sceptre_cer: float | None = None
    sceptre_wer: float | None = None
    sceptre_token_f1: float | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize to a plain JSON-friendly dict, dropping unset fields."""
        return {key: value for key, value in self.__dict__.items() if value is not None or key == "skipped"}


def _score_agreement(record: ImageRecord, easyocr: EngineOutput, sceptre: EngineOutput) -> None:
    """Populate cross-engine agreement metrics (char/word F1, mean best-line IoU)."""
    record.agreement_char_f1 = char_f1(sceptre.text, easyocr.text)
    record.agreement_word_f1 = word_f1(sceptre.text, easyocr.text)
    record.agreement_mean_iou = mean_best_line_iou(easyocr.quads, sceptre.quads)


def _score_accuracy(record: ImageRecord, reference_raw: str, easyocr: EngineOutput, sceptre: EngineOutput) -> None:
    """Populate absolute CER/WER/token-F1 for both engines against ground truth."""
    reference = normalize_text(reference_raw)
    for output, cer_field, wer_field, token_field in (
        (easyocr, "easyocr_cer", "easyocr_wer", "easyocr_token_f1"),
        (sceptre, "sceptre_cer", "sceptre_wer", "sceptre_token_f1"),
    ):
        hypothesis = normalize_text(output.text)
        setattr(record, cer_field, character_error_rate(reference, hypothesis))
        setattr(record, wer_field, word_error_rate(reference, hypothesis))
        setattr(record, token_field, word_f1(hypothesis, reference))


def benchmark_entry(
    entry: CorpusEntry,
    reader: object,
    binary: Path,
    overhead: float | None,
    root: Path,
) -> ImageRecord:
    """Run both engines over one entry and assemble its metrics record."""
    record = ImageRecord(stem=entry.stem, group=entry.group, languages=list(entry.languages))
    if entry.image is None or not entry.image.exists():
        record.skipped = "image not found on disk"
        return record
    try:
        easyocr = run_easyocr(reader, entry.image)
        sceptre = run_sceptre(binary, entry.image, entry.sceptre_langs, root)
    except Exception as error:  # noqa: BLE001 - record the failure instead of aborting the run
        record.skipped = f"engine error: {type(error).__name__}: {str(error)[:200]}"
        return record

    record.easyocr_seconds = easyocr.seconds
    record.easyocr_rss_mb = easyocr.rss_mb
    record.sceptre_cold_seconds = sceptre.seconds
    record.sceptre_rss_mb = sceptre.rss_mb
    if overhead is not None:
        record.sceptre_amortized_seconds = max(sceptre.seconds - overhead, 0.0)

    _score_agreement(record, easyocr, sceptre)
    if entry.ground_truth is not None and entry.ground_truth.exists():
        _score_accuracy(record, entry.ground_truth.read_text(encoding="utf-8"), easyocr, sceptre)
    return record


# --------------------------------------------------------------------------------------
# Aggregation and reporting
# --------------------------------------------------------------------------------------


def _summary(values: list[float]) -> dict[str, float] | None:
    """Median / p95 / mean / count over a list of floats, or None if empty."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
    return {
        "median": statistics.median(ordered),
        "p95": ordered[index],
        "mean": statistics.fmean(ordered),
        "count": len(ordered),
    }


def _collect(records: list[ImageRecord], attribute: str) -> list[float]:
    """Gather the non-None values of one attribute across records."""
    return [value for value in (getattr(record, attribute) for record in records) if value is not None]


AGGREGATE_FIELDS = (
    "easyocr_seconds",
    "sceptre_cold_seconds",
    "sceptre_amortized_seconds",
    "easyocr_rss_mb",
    "sceptre_rss_mb",
    "agreement_char_f1",
    "agreement_word_f1",
    "agreement_mean_iou",
    "easyocr_cer",
    "easyocr_wer",
    "easyocr_token_f1",
    "sceptre_cer",
    "sceptre_wer",
    "sceptre_token_f1",
)


def aggregate(records: list[ImageRecord]) -> dict[str, dict[str, float]]:
    """Summarize every aggregate field over the scored (non-skipped) records."""
    scored = [record for record in records if record.skipped is None]
    result: dict[str, dict[str, float]] = {}
    for field_name in AGGREGATE_FIELDS:
        summary = _summary(_collect(scored, field_name))
        if summary is not None:
            result[field_name] = summary
    return result


@dataclass
class RunReport:
    """The full benchmark result: metadata, per-image records, and aggregates."""

    metadata: dict[str, object]
    records: list[ImageRecord]
    aggregates_labeled: dict[str, dict[str, float]] = field(default_factory=dict)
    aggregates_all: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialize the whole report for JSON output."""
        return {
            "metadata": self.metadata,
            "aggregates": {"labeled": self.aggregates_labeled, "all": self.aggregates_all},
            "records": [record.to_dict() for record in self.records],
        }


def _fmt(value: float | None, digits: int = 3) -> str:
    """Format an optional number for a Markdown cell."""
    return "-" if value is None else f"{value:.{digits}f}"


def _median_of(aggregates: dict[str, dict[str, float]], field_name: str) -> float | None:
    """Median of an aggregate field, or None if absent."""
    summary = aggregates.get(field_name)
    return summary["median"] if summary else None


def _mean_of(aggregates: dict[str, dict[str, float]], field_name: str) -> float | None:
    """Mean of an aggregate field, or None if absent."""
    summary = aggregates.get(field_name)
    return summary["mean"] if summary else None


def _throughput(seconds: float | None) -> str:
    """Images-per-second from a median seconds value."""
    if not seconds:
        return "-"
    return f"{1.0 / seconds:.2f}"


def headline_rows(report: RunReport) -> list[list[str]]:
    """Build the headline comparison table rows (one per engine)."""
    labeled = report.aggregates_labeled
    every = report.aggregates_all
    easy_median = _median_of(every, "easyocr_seconds")
    scep_median = _median_of(every, "sceptre_cold_seconds")
    return [
        [
            "EasyOCR (warm readtext)",
            _fmt(easy_median),
            _throughput(easy_median),
            _fmt(_median_of(every, "easyocr_rss_mb"), 1),
            _fmt(_mean_of(labeled, "easyocr_cer")),
            _fmt(_mean_of(labeled, "easyocr_wer")),
            _fmt(_mean_of(labeled, "easyocr_token_f1")),
        ],
        [
            "sceptre (cold CLI run)",
            _fmt(scep_median),
            _throughput(scep_median),
            _fmt(_median_of(every, "sceptre_rss_mb"), 1),
            _fmt(_mean_of(labeled, "sceptre_cer")),
            _fmt(_mean_of(labeled, "sceptre_wer")),
            _fmt(_mean_of(labeled, "sceptre_token_f1")),
        ],
    ]


def _markdown_table(header: list[str], rows: list[list[str]]) -> str:
    """Render a simple GitHub-flavored Markdown table."""
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


HEADLINE_HEADER = [
    "Engine",
    "Median s",
    "Throughput img/s",
    "Median RSS MB",
    "Mean CER",
    "Mean WER",
    "Mean token-F1",
]

PER_IMAGE_HEADER = [
    "Image",
    "Group",
    "Lang",
    "EasyOCR s",
    "sceptre cold s",
    "sceptre amort s",
    "char-F1",
    "word-F1",
    "mean IoU",
    "Easy CER",
    "Scep CER",
    "Note",
]


def per_image_rows(records: list[ImageRecord]) -> list[list[str]]:
    """Build the per-image detail table rows."""
    rows: list[list[str]] = []
    for record in records:
        rows.append(
            [
                record.stem,
                record.group,
                "+".join(record.languages),
                _fmt(record.easyocr_seconds),
                _fmt(record.sceptre_cold_seconds),
                _fmt(record.sceptre_amortized_seconds),
                _fmt(record.agreement_char_f1),
                _fmt(record.agreement_word_f1),
                _fmt(record.agreement_mean_iou),
                _fmt(record.easyocr_cer),
                _fmt(record.sceptre_cer),
                record.skipped or "",
            ]
        )
    return rows


def render_markdown(report: RunReport) -> str:
    """Render the full scannable Markdown report."""
    meta = report.metadata
    overhead = meta.get("sceptre_overhead_seconds")
    scep_amort = _median_of(report.aggregates_all, "sceptre_amortized_seconds")
    skips = [record for record in report.records if record.skipped is not None]
    sections = [
        "# sceptre vs EasyOCR benchmark",
        "",
        f"- Platform: `{meta['platform']}` | sceptre binary: `{meta['sceptre_binary']}`",
        (
            f"- Corpus: {meta['corpus_total']} entries "
            f"({meta['labeled_scored']} labeled scored, {meta['breadth_scored']} breadth scored, "
            f"{len(skips)} skipped)"
        ),
        (
            f"- sceptre fixed overhead (startup + english model load): "
            f"{_fmt(overhead if isinstance(overhead, float) else None)} s "
            f"(median of {OVERHEAD_RUNS} runs on the smallest English image)"
        ),
        "",
        "## Headline",
        "",
        _markdown_table(HEADLINE_HEADER, headline_rows(report)),
        "",
        (
            "CER / WER / token-F1 are averaged over labeled images only. Median seconds and "
            "throughput are over the full run set (labeled + breadth)."
        ),
        "",
        "### Cold vs amortized speed",
        "",
        (
            f"sceptre's median cold run is {_fmt(_median_of(report.aggregates_all, 'sceptre_cold_seconds'))} s, "
            f"which includes a fixed ~{_fmt(overhead if isinstance(overhead, float) else None)} s startup + "
            f"model-load cost paid once per process. Subtracting it yields an amortized-inner median of "
            f"{_fmt(scep_amort)} s — the figure to compare against EasyOCR's warm "
            f"{_fmt(_median_of(report.aggregates_all, 'easyocr_seconds'))} s readtext, which excludes the "
            "one-time Reader construction (recorded separately in the JSON metadata)."
        ),
        "",
        "### Memory caveat",
        "",
        (
            "RSS figures are NOT directly comparable. EasyOCR RSS is the peak of the whole Python + torch "
            "process (interpreter, torch, model weights), measured via `getrusage(RUSAGE_SELF)` and "
            "monotonic across the run. sceptre RSS is a lean native subprocess peak from `/usr/bin/time`. "
            "Treat them as order-of-magnitude context, not a like-for-like delta."
        ),
        "",
        "## Per-image detail",
        "",
        _markdown_table(PER_IMAGE_HEADER, per_image_rows(report.records)),
        "",
    ]
    if skips:
        sections.append("## Skipped")
        sections.append("")
        for record in skips:
            sections.append(f"- `{record.stem}`: {record.skipped}")
        sections.append("")
    return "\n".join(sections)


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


def _build_readers(easyocr_module: object, entries: list[CorpusEntry]) -> dict[tuple[str, ...], object]:
    """Construct one EasyOCR reader per unique language set and time each build."""
    readers: dict[tuple[str, ...], object] = {}
    build_seconds: dict[str, float] = {}
    for entry in entries:
        if entry.image is None:
            continue
        key = tuple(entry.easyocr_codes)
        if key in readers:
            continue
        start = perf_counter()
        # gpu=False keeps the benchmark reproducible across machines without a GPU.
        readers[key] = easyocr_module.Reader(list(key), gpu=False)  # type: ignore[attr-defined]
        build_seconds["+".join(key)] = perf_counter() - start
    _build_readers.build_seconds = build_seconds  # type: ignore[attr-defined]
    return readers


def run_benchmark(group: str, limit: int | None, output_dir: Path) -> int:
    """Run the full benchmark and write JSON + Markdown; returns a process exit code."""
    try:
        import easyocr
    except ImportError:
        print(
            "sceptre_rs_tools.benchmark: needs the optional 'export' dependency group "
            "(torch, easyocr). Install it with `uv sync --group export`, then re-run.",
            file=sys.stderr,
        )
        return 1

    root = repo_root()
    binary = root / SCEPTRE_BIN
    if not binary.exists():
        print(
            f"sceptre_rs_tools.benchmark: release binary not found at {binary}. Build it with "
            "`cargo build --release -p sceptre-cli --no-default-features --features ort-bundled,download`.",
            file=sys.stderr,
        )
        return 1

    entries = build_corpus(root, group=group)
    if limit is not None:
        entries = entries[:limit]

    overhead = measure_sceptre_overhead(binary, entries, root)
    readers = _build_readers(easyocr, entries)

    records: list[ImageRecord] = []
    for entry in entries:
        reader = readers.get(tuple(entry.easyocr_codes))
        print(f"sceptre_rs_tools.benchmark: {entry.stem} ({'+'.join(entry.easyocr_codes)})", file=sys.stderr)
        records.append(benchmark_entry(entry, reader, binary, overhead, root))

    labeled = [record for record in records if record.group == "labeled"]
    scored = [record for record in records if record.skipped is None]
    metadata = {
        "platform": platform.platform(),
        "sceptre_binary": str(SCEPTRE_BIN),
        "group": group,
        "corpus_total": len(records),
        "labeled_scored": sum(1 for record in labeled if record.skipped is None),
        "breadth_scored": sum(1 for record in scored if record.group == "breadth"),
        "sceptre_overhead_seconds": overhead,
        "easyocr_reader_build_seconds": getattr(_build_readers, "build_seconds", {}),
        "overhead_runs": OVERHEAD_RUNS,
    }
    report = RunReport(
        metadata=metadata,
        records=records,
        aggregates_labeled=aggregate(labeled),
        aggregates_all=aggregate(records),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown = render_markdown(report)
    (output_dir / "comparison.md").write_text(markdown + "\n", encoding="utf-8")

    print(_markdown_table(HEADLINE_HEADER, headline_rows(report)))
    print(f"\nWrote {output_dir / 'comparison.json'} and {output_dir / 'comparison.md'}", file=sys.stderr)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the benchmark CLI arguments."""
    parser = argparse.ArgumentParser(description="Benchmark sceptre against upstream EasyOCR.")
    parser.add_argument(
        "--group",
        choices=["labeled", "all"],
        default="all",
        help="Corpus subset: 'labeled' (ground-truth images) or 'all' (default).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of images (for smoke runs).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root() / OUTPUT_DIR,
        help="Directory for comparison.json and comparison.md.",
    )
    return parser.parse_args(argv)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    sys.exit(run_benchmark(group=args.group, limit=args.limit, output_dir=args.output_dir))


if __name__ == "__main__":
    main()
