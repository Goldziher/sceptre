"""Head-to-head development benchmark: sceptre vs upstream Python EasyOCR.

This is a development tool, not a one-off report: it drives sceptre toward release-quality
parity with EasyOCR at substantially better speed and memory, and is built for tight
iteration (fast ``--group`` / ``--limit`` subsets, ``--repeats`` for stable numbers, a
``--baseline`` delta view, and a ``--assert`` regression gate). Results are written to
``benchmark-results/comparison.{json,md}``.

Provenance — every report records the runtimes that produced it in ``metadata.environment``:
the sceptre binary's version, backend, accelerator, ONNX Runtime and model pins (probed from
the binary under test, not from this checkout), plus the reference EasyOCR/torch/Python
versions reported by the runner. A published number without its runtime is not reproducible.

Fairness — both engines are measured identically (see the benchmark-methodology ADR):
  * Each engine runs as a FRESH SUBPROCESS per language group, under ``/usr/bin/time``, so
    peak RSS is a whole-process figure captured the same way for both. The EasyOCR RSS
    therefore legitimately includes the Python + torch runtime — that is what EasyOCR costs
    to run.
  * Each subprocess loads its model/Reader once and processes every image in the group (the
    warm/batch axis), mirroring how a server reuses a warm engine. Group wall time includes
    the one-time load for both; ``build_seconds`` is recorded so it can be subtracted.
  * Both engines run at their native multi-threaded default (representative of real
    deployment); ``--threads N`` pins both for a controlled per-core comparison. Stability
    comes from ``--repeats`` (median wall time, max peak RSS), not from crippling parallelism.
  * sceptre is the RELEASE binary; a debug build would be unfairly slow. A secondary
    per-image "cold" sceptre figure (fresh process per image, pays model load every time)
    is reported for the real per-invocation CLI cost.

Needs the opt-in ``export`` dependency group (torch, easyocr) and a built release binary;
without either it exits cleanly with an explanatory message.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

from sceptre_rs_tools.corpus import CorpusEntry, build_corpus
from sceptre_rs_tools.stats import P95_MIN_SAMPLES, P99_MIN_SAMPLES, integrate_cpu_core_seconds, suppressed_percentile
from sceptre_rs_tools.text_metrics import f1_parts_from, greedy_match, reading_order_score
from sceptre_rs_tools.text_metrics import tokenize as _cjk_bigram_tokenize

# sceptre release binary produced by the ort-bundled CLI build.
SCEPTRE_BIN = Path("target/release/sceptre")
OUTPUT_DIR = Path("benchmark-results")

# The committed, published benchmark artifact, relative to the repository root. It is written
# by `sceptre_rs_tools.publish` and read back here as the baseline for the absolute peak-RSS
# ceiling, so the constant lives with the harness both sides share. ~keep
PUBLISHED_RELATIVE_PATH = Path("benchmarks/published/latest.json")
OVERHEAD_RUNS = 3  # process launches used to estimate sceptre's fixed startup cost
DEFAULT_REPEATS = 3  # per-measurement repeats: median wall time, max peak RSS

# Smallest ratio worth stating as a comparative claim. Below it, "~1.1x lower" reads as an
# advantage while being indistinguishable from parity and from run-to-run noise, so the claim
# is dropped rather than rounded. Presentation only — it feeds no regression gate. ~keep
MIN_CLAIMABLE_RATIO = 1.15

# Environment variable naming the CI runner that produced a run (see .github/workflows/
# benchmarks.yaml). Unset locally, in which case no runner label is recorded at all. ~keep
RUNNER_LABEL_ENV = "BENCHMARK_RUNNER_LABEL"

# sceptre's own peak RSS must stay within this multiple of the figure the committed baseline
# published for the same host (ADR 0042). The mirror image of GUARDRAIL_THRESHOLD_FACTOR's 10%
# slack, for the same reason: peak RSS is a max over repeats of a whole-process max, driven by
# allocator and canvas rounding rather than by timing noise, so it repeats to a few percent on a
# fixed host and corpus. 10% clears that without hiding the growth an extra resident model or a
# leaked buffer would cause. ~keep
SCEPTRE_RSS_CEILING_FACTOR = 1.1

# sceptre's own warm/batch throughput must stay at least this fraction of the figure the committed
# baseline published for the same host (ADR 0043). Symmetric with SCEPTRE_RSS_CEILING_FACTOR's 1.10:
# 10% of run-to-run degradation is tolerated, which covers a shared runner's timing noise without
# hiding the slowdown an extra pass over the image or a lost parallel stage would cause. ~keep
SCEPTRE_THROUGHPUT_FLOOR_FACTOR = 0.9

# Per-image quality floors (see GuardrailContract below) replace a single corpus-mean quality
# check: a corpus mean cannot catch a regression isolated to one language or one image. ~keep
GUARDRAILS_SCHEMA_VERSION = 1
GUARDRAIL_THRESHOLD_FACTOR = 0.9  # each image's floor is its baseline score times this factor

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
    """NFKC-normalize, case-fold, and CJK-bigram-expand; matches the Rust ``word_f1`` tokenizer.

    See ``sceptre_rs_tools.text_metrics.tokenize`` for the full contract.
    """
    return _cjk_bigram_tokenize(text)


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


# IoU floor for a hypothesis/reference line pair to count as a match, matching the
# Rust tier2_golden parity floor.
LINE_IOU_THRESHOLD = 0.5


def line_detection_scores(
    reference_quads: list[list[list[float]]], hypothesis_quads: list[list[list[float]]]
) -> tuple[float, float, float]:
    """Greedy IoU-matched (f1, precision, recall) of hypothesis lines against reference lines.

    Replaces `mean_best_line_iou`, which averaged over *reference* lines only and so never
    penalized a hypothesis line with no reference match (recall-only -- no fabrication
    penalty). Greedy matching (`greedy_match`) plus `f1_parts_from` reports precision and
    recall separately, so a low score reads as fabrication (low precision) vs omission
    (low recall) instead of collapsing both into one number.
    """
    reference_boxes = [quad_bbox(quad) for quad in reference_quads]
    hypothesis_boxes = [quad_bbox(quad) for quad in hypothesis_quads]
    matches = greedy_match(hypothesis_boxes, reference_boxes, box_iou, LINE_IOU_THRESHOLD)
    return f1_parts_from(float(len(matches)), len(hypothesis_boxes), len(reference_boxes))


# --------------------------------------------------------------------------------------
# Ground-truth accuracy (CER / WER / token-F1)
# --------------------------------------------------------------------------------------

_MARKDOWN_CHARS = re.compile(r"[#|*`_>]")
_LIST_BULLET = re.compile(r"^\s*(?:[-+•]|\d+[.)])\s+", re.MULTILINE)
_WHITESPACE = re.compile(r"\s+")
# A markdown table cell line break (e.g. "2025/26<br>Actual"), not visible text an OCR
# engine could ever emit -- distinct from a literal angle-bracket placeholder that IS
# printed in a source document (e.g. "lp://<dataset-name>/..."), which must survive
# normalization untouched. ~keep
_TABLE_LINE_BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)


def normalize_text(text: str) -> str:
    """Strip light Markdown, lowercase, and collapse whitespace to single spaces.

    Applied identically to reference and hypothesis before scoring so document markup
    and reading-order line breaks do not dominate the edit distance.
    """
    without_line_breaks = _TABLE_LINE_BREAK.sub(" ", text)
    without_bullets = _LIST_BULLET.sub(" ", without_line_breaks)
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
# Engine output model
# --------------------------------------------------------------------------------------


@dataclass
class Detection:
    """One recognized line: text plus its four-corner quad."""

    text: str
    quad: list[list[float]]


@dataclass
class ImageDetections:
    """One image's slice of a batch run: detections, warm inner time, or a failure.

    ``unsupported_format`` marks an image whose container the engine cannot decode (a
    capability gap surfaced by the report), distinct from a genuine runtime error.
    """

    detections: list[Detection] | None
    seconds: float | None = None
    error: str | None = None
    unsupported_format: bool = False

    @property
    def text(self) -> str:
        """All line texts joined with single spaces, in engine order."""
        return " ".join(detection.text for detection in self.detections or [])

    @property
    def quads(self) -> list[list[list[float]]]:
        """All line quads."""
        return [detection.quad for detection in self.detections or []]


@dataclass
class BatchResult:
    """One engine's warm/batch run over a whole per-language image group.

    ``total_seconds`` is the median whole-process wall time over repeats (model/Reader load
    once + every image); ``rss_mb`` is the max peak RSS over repeats; ``build_seconds`` is the
    one-time model/Reader load, so ``(total - build) / image_count`` is the amortized warm
    per-image inner time.
    """

    engine: str
    langs: tuple[str, ...]
    total_seconds: float
    image_count: int
    rss_mb: float | None
    build_seconds: float | None
    per_image: dict[str, ImageDetections]
    cpu_core_seconds: float | None = None
    """Median total CPU core-seconds (user + sys) over the repeats, or None if unmeasured."""
    versions: dict[str, str] = field(default_factory=dict)
    """Runtime versions reported by the engine's driver (EasyOCR only); provenance for the report."""

    def warm_per_image(self) -> float | None:
        """Amortized warm inner seconds per image: (total - build) / image_count."""
        if self.image_count <= 0:
            return None
        build = self.build_seconds or 0.0
        return max((self.total_seconds - build) / self.image_count, 0.0)


# --------------------------------------------------------------------------------------
# Subprocess plumbing (peak RSS, timing, JSON parsing) — shared by both engines
# --------------------------------------------------------------------------------------

_TIME_RSS_DARWIN = re.compile(r"(\d+)\s+maximum resident set size")
_TIME_RSS_LINUX = re.compile(r"Maximum resident set size \(kbytes\):\s+(\d+)")
_TIME_CPU_DARWIN = re.compile(r"([\d.]+)\s+real\s+([\d.]+)\s+user\s+([\d.]+)\s+sys")
_TIME_USER_LINUX = re.compile(r"User time \(seconds\):\s+([\d.]+)")
_TIME_SYS_LINUX = re.compile(r"System time \(seconds\):\s+([\d.]+)")


_warned_missing_time_binary: set[str] = set()


def _time_wrapper() -> list[str]:
    """The ``/usr/bin/time`` prefix that reports peak RSS, or empty if unavailable.

    Absence is reported once rather than passed over: without the wrapper every ``rss_mb``
    and ``cpu_core_seconds`` in the report is ``None``, for both engines, and the peak-RSS
    floor becomes unmeasurable. The gate does fail on that, but only the warning explains
    it -- the shell builtin ``time`` is not a substitute, the binary has to be installed.
    """
    time_bin = Path("/usr/bin/time")
    if not time_bin.exists():
        if not _warned_missing_time_binary:
            _warned_missing_time_binary.add(str(time_bin))
            print(
                f"sceptre_rs_tools.benchmark: {time_bin} not found; peak RSS and CPU "
                "core-seconds will be unmeasured for every engine (install GNU time)",
                file=sys.stderr,
            )
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


def _parse_child_cpu_seconds(stderr: str) -> float | None:
    """Extract total CPU core-seconds (user + sys) from ``/usr/bin/time`` stderr output.

    Reads the same fields ``/usr/bin/time -l``/``-v`` already reports alongside peak RSS,
    so no additional sampling is needed to compare a threaded Rust engine against
    single-threaded Python, or to show a GPU leg offloading CPU work.
    """
    match = _TIME_CPU_DARWIN.search(stderr)
    if match:
        return integrate_cpu_core_seconds(float(match.group(2)), float(match.group(3)))
    user_match = _TIME_USER_LINUX.search(stderr)
    sys_match = _TIME_SYS_LINUX.search(stderr)
    if user_match and sys_match:
        return integrate_cpu_core_seconds(float(user_match.group(1)), float(sys_match.group(1)))
    return None


def _strip_time_stats(stderr: str) -> str:
    """Drop the ``/usr/bin/time`` resource block, leaving only the child's own stderr.

    The wrapper appends a fixed block of ``N  label`` resource lines; keeping them would
    leak "5 page faults / 0 swaps ..." noise into any surfaced engine error message.
    """
    kept: list[str] = []
    for line in stderr.splitlines():
        stripped = line.strip()
        # /usr/bin/time -l/-v lines are "<number>  <words>" (darwin) or "Label: <number>" (linux). ~keep
        if re.match(r"^[\d,]+\s+[a-z]", stripped) or _TIME_RSS_LINUX.search(stripped):
            continue
        if re.match(r"^[A-Z][A-Za-z()/ ]+:\s", stripped) and re.search(r"\d", stripped):
            continue
        kept.append(line)
    return "\n".join(line for line in kept if line.strip())


def _is_unsupported_format(message: str) -> bool:
    """True when an engine error is a decode/format-support failure, not a runtime crash."""
    lowered = message.lower()
    return "not supported" in lowered or "failed to decode" in lowered or "unsupported" in lowered


def _median(values: list[float]) -> float:
    """Median of a non-empty list of floats."""
    return statistics.median(values)


# --------------------------------------------------------------------------------------
# sceptre drivers
# --------------------------------------------------------------------------------------


def _detections_from_lines(lines: list[dict[str, object]]) -> list[Detection]:
    """Convert a sceptre ``lines`` array into ``Detection`` records."""
    detections: list[Detection] = []
    for line in lines:
        points = line.get("quad", {}).get("points", [])
        quad = [[float(point["x"]), float(point["y"])] for point in points]
        detections.append(Detection(text=str(line.get("text", "")), quad=quad))
    return detections


def _sceptre_image_from_element(element: dict[str, object]) -> ImageDetections:
    """Convert one batch JSON element into an ``ImageDetections`` (detections or failure)."""
    if "error" in element:
        message = str(element["error"])
        return ImageDetections(detections=None, error=message, unsupported_format=_is_unsupported_format(message))
    return ImageDetections(detections=_detections_from_lines(element.get("lines", [])))


def _parse_sceptre_batch_json(payload: str, images: list[Path]) -> dict[str, ImageDetections]:
    """Parse batch ``sceptre run`` stdout, aligned to the group's input order.

    sceptre emits a JSON array (one element per image, each carrying an ``image`` key) for
    more than one image, and a single object ``{"lines": ...}`` for exactly one image.
    """
    data = json.loads(payload)
    results: dict[str, ImageDetections] = {}
    if isinstance(data, dict):
        key = data.get("image") or (str(images[0]) if images else "")
        results[str(key)] = _sceptre_image_from_element(data)
        return results
    for index, element in enumerate(data):
        key = element.get("image")
        if key is None:
            key = str(images[index]) if index < len(images) else str(index)
        results[str(key)] = _sceptre_image_from_element(element)
    return results


def _thread_args(threads: int | None) -> list[str]:
    """The sceptre ``--threads`` flag, or empty to use the engine default (all cores)."""
    return ["--threads", str(threads)] if threads is not None else []


def _sceptre_batch_command(
    binary: Path, images: list[Path], sceptre_langs: list[str], threads: int | None
) -> list[str]:
    """Build the sceptre CLI argument vector for a batch sharing one language set."""
    command = [str(binary), "run", *[str(image) for image in images]]
    for language in sceptre_langs:
        command += ["--lang", language]
    command += _thread_args(threads) + ["--format", "json"]
    return command


def _run_sceptre_batch_once(
    binary: Path, images: list[Path], sceptre_langs: list[str], root: Path, threads: int | None
) -> tuple[float, float | None, float | None, dict[str, ImageDetections]]:
    """One warm ``sceptre run`` over a group; return (seconds, rss_mb, cpu_core_seconds, per_image).

    A non-zero exit with per-image errors already encoded in the JSON is tolerated (sceptre
    exits non-zero when any image fails); only a run with no parseable JSON raises.
    """
    wrapped = _time_wrapper() + _sceptre_batch_command(binary, images, sceptre_langs, threads)
    start = perf_counter()
    completed = subprocess.run(wrapped, capture_output=True, text=True, cwd=root, check=False)
    seconds = perf_counter() - start
    rss = _parse_child_rss(completed.stderr)
    cpu_core_seconds = _parse_child_cpu_seconds(completed.stderr)
    try:
        per_image = _parse_sceptre_batch_json(completed.stdout, images)
    except json.JSONDecodeError as error:
        clean = _strip_time_stats(completed.stderr)[-300:]
        raise RuntimeError(f"sceptre batch produced no JSON (exit {completed.returncode}): {clean}") from error
    return seconds, rss, cpu_core_seconds, per_image


def run_sceptre_batch(
    binary: Path, images: list[Path], sceptre_langs: list[str], root: Path, repeats: int, threads: int | None
) -> BatchResult:
    """Warm/batch sceptre over a group, repeated for a stable median time and max RSS."""
    times: list[float] = []
    rss_values: list[float] = []
    cpu_values: list[float] = []
    per_image: dict[str, ImageDetections] = {}
    for _ in range(max(repeats, 1)):
        seconds, rss, cpu_core_seconds, per_image = _run_sceptre_batch_once(
            binary, images, sceptre_langs, root, threads
        )
        times.append(seconds)
        if rss is not None:
            rss_values.append(rss)
        if cpu_core_seconds is not None:
            cpu_values.append(cpu_core_seconds)
    return BatchResult(
        engine="sceptre",
        langs=tuple(sceptre_langs),
        total_seconds=_median(times),
        image_count=len(images),
        rss_mb=max(rss_values) if rss_values else None,
        build_seconds=None,
        per_image=per_image,
        cpu_core_seconds=_median(cpu_values) if cpu_values else None,
    )


def _sceptre_cold_command(binary: Path, image: Path, sceptre_langs: list[str], threads: int | None) -> list[str]:
    """Build the sceptre CLI argument vector for a single cold image run."""
    command = [str(binary), "run", str(image)]
    for language in sceptre_langs:
        command += ["--lang", language]
    command += _thread_args(threads) + ["--format", "json"]
    return command


def run_sceptre_cold(binary: Path, image: Path, sceptre_langs: list[str], root: Path, threads: int | None) -> float:
    """Time ONE fresh ``sceptre run`` over a single image (cold: pays model load)."""
    wrapped = _time_wrapper() + _sceptre_cold_command(binary, image, sceptre_langs, threads)
    start = perf_counter()
    completed = subprocess.run(wrapped, capture_output=True, text=True, cwd=root, check=False)
    seconds = perf_counter() - start
    if completed.returncode != 0:
        clean = _strip_time_stats(completed.stderr)[-300:]
        raise RuntimeError(f"sceptre exited {completed.returncode}: {clean}")
    return seconds


def measure_sceptre_overhead(binary: Path, entries: list[CorpusEntry], root: Path, threads: int | None) -> float | None:
    """Estimate sceptre's fixed startup + model-load cost from the smallest English image."""
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
            timings.append(run_sceptre_cold(binary, smallest.image, ["english"], root, threads))  # type: ignore[arg-type]
        except (RuntimeError, ValueError):
            return None
    return statistics.median(timings)


# --------------------------------------------------------------------------------------
# EasyOCR drivers (subprocess, symmetric to sceptre batch)
# --------------------------------------------------------------------------------------


def _easyocr_runner_command(
    images: list[Path], easyocr_codes: list[str], threads: int | None, output: Path
) -> list[str]:
    """Build the ``_easyocr_runner`` argument vector for a batch sharing one language set."""
    command = [sys.executable, "-m", "sceptre_rs_tools._easyocr_runner", "--output", str(output)]
    for code in easyocr_codes:
        command += ["--lang", code]
    if threads is not None:
        command += ["--threads", str(threads)]
    command += [str(image) for image in images]
    return command


def _parse_easyocr_runner_json(payload: str) -> tuple[float | None, dict[str, str], dict[str, ImageDetections]]:
    """Parse ``_easyocr_runner`` stdout into (reader_build_seconds, versions, per_image)."""
    data = json.loads(payload)
    build_seconds = data.get("reader_build_seconds")
    raw_versions = data.get("versions", {})
    versions = {str(key): str(value) for key, value in raw_versions.items()} if isinstance(raw_versions, dict) else {}
    per_image: dict[str, ImageDetections] = {}
    for element in data.get("images", []):
        key = str(element.get("image", ""))
        if "error" in element:
            message = str(element["error"])
            per_image[key] = ImageDetections(
                detections=None, error=message, unsupported_format=_is_unsupported_format(message)
            )
            continue
        detections = [Detection(text=str(d["text"]), quad=d["quad"]) for d in element.get("detections", [])]
        per_image[key] = ImageDetections(detections=detections, seconds=element.get("seconds"))
    return (float(build_seconds) if isinstance(build_seconds, (int, float)) else None, versions, per_image)


def _run_easyocr_batch_once(
    images: list[Path], easyocr_codes: list[str], root: Path, threads: int | None
) -> tuple[float, float | None, float | None, float | None, dict[str, str], dict[str, ImageDetections]]:
    """One warm ``_easyocr_runner`` subprocess; (seconds, rss, cpu_core_seconds, build, versions, per_image)."""
    with tempfile.TemporaryDirectory(prefix="sceptre-easyocr-") as tmp:
        payload_path = Path(tmp) / "result.json"
        wrapped = _time_wrapper() + _easyocr_runner_command(images, easyocr_codes, threads, payload_path)
        start = perf_counter()
        completed = subprocess.run(wrapped, capture_output=True, text=True, cwd=root, check=False)
        seconds = perf_counter() - start
        rss = _parse_child_rss(completed.stderr)
        cpu_core_seconds = _parse_child_cpu_seconds(completed.stderr)
        payload = payload_path.read_text(encoding="utf-8") if payload_path.is_file() else ""
        try:
            build_seconds, versions, per_image = _parse_easyocr_runner_json(payload)
        except json.JSONDecodeError as error:
            # Report both streams: the payload is what failed to parse, but the reason the
            # runner produced none is almost always in its stderr. ~keep
            out = _strip_time_stats(completed.stdout)[-300:]
            err = _strip_time_stats(completed.stderr)[-600:]
            raise RuntimeError(
                f"easyocr runner produced no JSON (exit {completed.returncode})\n"
                f"  payload ({len(payload)} chars): {payload[:200]!r}\n"
                f"  stdout tail: {out!r}\n"
                f"  stderr tail: {err!r}"
            ) from error
    return seconds, rss, cpu_core_seconds, build_seconds, versions, per_image


def run_easyocr_batch(
    images: list[Path], easyocr_codes: list[str], root: Path, repeats: int, threads: int | None
) -> BatchResult:
    """Warm/batch EasyOCR over a group, repeated for a stable median time and max RSS."""
    times: list[float] = []
    rss_values: list[float] = []
    cpu_values: list[float] = []
    build_values: list[float] = []
    per_image: dict[str, ImageDetections] = {}
    versions: dict[str, str] = {}
    for _ in range(max(repeats, 1)):
        seconds, rss, cpu_core_seconds, build_seconds, versions, per_image = _run_easyocr_batch_once(
            images, easyocr_codes, root, threads
        )
        times.append(seconds)
        if rss is not None:
            rss_values.append(rss)
        if cpu_core_seconds is not None:
            cpu_values.append(cpu_core_seconds)
        if build_seconds is not None:
            build_values.append(build_seconds)
    return BatchResult(
        engine="easyocr",
        langs=tuple(easyocr_codes),
        total_seconds=_median(times),
        image_count=len(images),
        rss_mb=max(rss_values) if rss_values else None,
        build_seconds=_median(build_values) if build_values else None,
        per_image=per_image,
        cpu_core_seconds=_median(cpu_values) if cpu_values else None,
        versions=versions,
    )


# --------------------------------------------------------------------------------------
# Environment provenance
#
# A published parity/benchmark number describes a runtime, so the report has to record
# which one. The sceptre side is probed from the binary under test rather than from this
# checkout's sources, because the binary is what produced the numbers.
# --------------------------------------------------------------------------------------

# Preferred probe: a single subcommand that reports version, backend, accelerator, ONNX
# Runtime build info and the model pins. Older binaries do not have it, so the probe
# degrades to `--version` plus `models list`. ~keep
SCEPTRE_ENV_ARGS = ("env", "--format", "json")
UNKNOWN = "unknown"


def _binary_json(binary: Path, args: list[str], root: Path) -> object | None:
    """Run the sceptre binary with ``args`` and parse stdout as JSON, or ``None`` on failure."""
    try:
        completed = subprocess.run(
            [str(binary), *args], capture_output=True, text=True, cwd=root, check=False, timeout=60
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None


def _binary_version(binary: Path, root: Path) -> str | None:
    """The version string from ``sceptre --version`` (``sceptre 0.3.0`` -> ``0.3.0``)."""
    try:
        completed = subprocess.run(
            [str(binary), "--version"], capture_output=True, text=True, cwd=root, check=False, timeout=60
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    parts = completed.stdout.strip().split()
    return parts[-1] if parts else None


def _models_from_list(binary: Path, root: Path, languages: list[str]) -> list[dict[str, object]]:
    """Model repos for ``languages`` from ``sceptre models list --format json``.

    This fallback cannot report the sha256 pins — ``models list`` does not emit them — so
    they are recorded as ``None`` rather than copied out of this checkout's registry, which
    may not be the registry the binary was built from.
    """
    args = ["models", "list", "--format", "json"]
    for language in languages:
        args += ["--lang", language]
    payload = _binary_json(binary, args, root)
    if not isinstance(payload, list):
        return []
    return [
        {"name": entry.get("name"), "repo": entry.get("repo"), "sha256": entry.get("sha256")}
        for entry in payload
        if isinstance(entry, dict)
    ]


def _sceptre_environment(binary: Path, root: Path, languages: list[str]) -> dict[str, object]:
    """Probe the sceptre binary under test for its build and runtime identity.

    Prefers ``sceptre env --format json``; falls back to ``--version`` plus
    ``models list --format json`` when the binary predates that subcommand. Fields the
    fallback cannot observe (backend, accelerator, ONNX Runtime version, model sha256)
    stay ``None`` rather than being guessed.
    """
    payload = _binary_json(binary, list(SCEPTRE_ENV_ARGS), root)
    if isinstance(payload, dict):
        onnxruntime = payload.get("onnxruntime")
        return {
            "sceptre_version": payload.get("version") or payload.get("sceptre_version"),
            "backend": payload.get("backend"),
            "accelerator": payload.get("accelerator"),
            "onnxruntime": onnxruntime if onnxruntime is not None else payload.get("onnxruntime_version"),
            "models": payload.get("models") or [],
            "probe": "sceptre env --format json",
        }
    return {
        "sceptre_version": _binary_version(binary, root),
        "backend": None,
        "accelerator": None,
        "onnxruntime": None,
        "models": _models_from_list(binary, root, languages),
        "probe": "sceptre --version + models list",
    }


def _reference_environment(versions: dict[str, str]) -> dict[str, object]:
    """The EasyOCR reference half of the environment block, from the runner's payload."""
    return {
        "easyocr_version": versions.get("easyocr"),
        "torch_version": versions.get("torch"),
        "python": versions.get("python"),
    }


# Path of the fixtures submodule's content-addressed lock file, relative to the repo root.
CORPUS_LOCK_PATH = Path("test_documents") / "corpus.lock.json"


def _load_corpus_lock(root: Path) -> dict[str, object] | None:
    """Load ``test_documents/corpus.lock.json`` if the fixtures submodule has published one.

    Returns ``None`` (not an error) until the fixture migration lands or the file is
    malformed -- callers fall back to unresolved provenance rather than fabricating one.
    """
    path = root / CORPUS_LOCK_PATH
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("objects"), dict):
        return None
    return payload


def _corpus_lock_hash_for(lock: dict[str, object], image: Path) -> str | None:
    """Look up an image's sha256 in the corpus lock by file name.

    The lock's keys are paths relative to the `test_documents` submodule root; matching by
    name (rather than a full relative path) is robust to the vendored copies under
    `crates/sceptre/tests/data/images/` not (yet) sharing that exact layout.
    """
    objects = lock.get("objects", {})
    if not isinstance(objects, dict):
        return None
    for key, meta in objects.items():
        if isinstance(meta, dict) and Path(key).name == image.name:
            sha256 = meta.get("sha256")
            return str(sha256) if sha256 else None
    return None


def _corpus_provenance(root: Path, entries: list[CorpusEntry]) -> dict[str, object]:
    """Provenance for this run's corpus images, citing `test_documents/corpus.lock.json`.

    The lock already carries a per-object sha256 for every fixture, so this cites those
    hashes directly instead of re-hashing files itself (see ADR 0021). Returns
    `resolved: False` (no fabricated hashes) until the lock file exists, or when an
    entry's image cannot be matched to a lock object.
    """
    lock = _load_corpus_lock(root)
    if lock is None:
        return {"resolved": False, "source": None, "images": {}}
    images: dict[str, str] = {}
    for entry in entries:
        if entry.image is None:
            continue
        sha256 = _corpus_lock_hash_for(lock, entry.image)
        if sha256 is not None:
            images[entry.stem] = sha256
    return {
        "resolved": True,
        "source": str(CORPUS_LOCK_PATH),
        "lock_schema": lock.get("schema"),
        "images": images,
    }


def environment_metadata(
    binary: Path,
    root: Path,
    languages: list[str],
    reference_versions: dict[str, str],
    entries: list[CorpusEntry] | None = None,
) -> dict[str, object]:
    """The ``metadata.environment`` block: which runtimes and corpus produced this report."""
    environment = _sceptre_environment(binary, root, languages)
    environment.update(
        {
            "sceptre_binary": str(binary),
            "os": platform.system(),
            "arch": platform.machine(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "reference": _reference_environment(reference_versions),
            "corpus": _corpus_provenance(root, entries or []),
        }
    )
    runner_label = runner_label_from_environment()
    if runner_label is not None:
        environment["runner_label"] = runner_label
    return environment


def runner_label_from_environment() -> str | None:
    """The CI runner label for this run, or None when it was not run on a labeled runner.

    ``os`` and ``arch`` cannot tell ``runner-medium`` from ``ubuntu-latest``, and the two
    do not produce comparable numbers, so a published figure needs the label to be
    attributable to a host (ADR 0035). Absent locally, where there is no runner to name.
    """
    label = os.environ.get(RUNNER_LABEL_ENV, "").strip()
    return label or None


# --------------------------------------------------------------------------------------
# Batch orchestration over language groups
# --------------------------------------------------------------------------------------


def _runnable_images(entries: list[CorpusEntry]) -> list[CorpusEntry]:
    """Entries whose image exists on disk."""
    return [entry for entry in entries if entry.image is not None and entry.image.exists()]


def _group_by(entries: list[CorpusEntry], key) -> dict[tuple[str, ...], list[CorpusEntry]]:
    """Group runnable entries by a language-tuple key, preserving input order."""
    groups: dict[tuple[str, ...], list[CorpusEntry]] = {}
    for entry in _runnable_images(entries):
        groups.setdefault(tuple(key(entry)), []).append(entry)
    return groups


def run_sceptre_pass(
    binary: Path, entries: list[CorpusEntry], root: Path, repeats: int, threads: int | None
) -> dict[tuple[str, ...], BatchResult]:
    """One warm sceptre batch per distinct sceptre-language group."""
    results: dict[tuple[str, ...], BatchResult] = {}
    for langs, group in _group_by(entries, lambda entry: entry.sceptre_langs).items():
        images = [entry.image for entry in group if entry.image is not None]
        print(f"sceptre_rs_tools.benchmark: sceptre batch [{'+'.join(langs)}] x{len(images)}", file=sys.stderr)
        results[langs] = run_sceptre_batch(binary, images, list(langs), root, repeats, threads)
    return results


def run_easyocr_pass(
    entries: list[CorpusEntry], root: Path, repeats: int, threads: int | None
) -> dict[tuple[str, ...], BatchResult]:
    """One warm EasyOCR batch per distinct EasyOCR-language group."""
    results: dict[tuple[str, ...], BatchResult] = {}
    for codes, group in _group_by(entries, lambda entry: entry.easyocr_codes).items():
        images = [entry.image for entry in group if entry.image is not None]
        print(f"sceptre_rs_tools.benchmark: easyocr batch [{'+'.join(codes)}] x{len(images)}", file=sys.stderr)
        results[codes] = run_easyocr_batch(images, list(codes), root, repeats, threads)
    return results


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
    unsupported_format: bool = False
    easyocr_warm_seconds: float | None = None
    sceptre_cold_seconds: float | None = None
    sceptre_amortized_seconds: float | None = None
    easyocr_rss_mb: float | None = None
    sceptre_rss_mb: float | None = None
    agreement_char_f1: float | None = None
    agreement_word_f1: float | None = None
    agreement_line_f1: float | None = None
    agreement_line_precision: float | None = None
    agreement_line_recall: float | None = None
    agreement_reading_order: float | None = None
    easyocr_cer: float | None = None
    easyocr_wer: float | None = None
    easyocr_token_f1: float | None = None
    sceptre_cer: float | None = None
    sceptre_wer: float | None = None
    sceptre_token_f1: float | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize to a plain JSON-friendly dict, dropping unset fields."""
        keep_always = {"skipped", "unsupported_format"}
        return {key: value for key, value in self.__dict__.items() if value is not None or key in keep_always}


def _score_agreement(record: ImageRecord, easyocr: ImageDetections, sceptre: ImageDetections) -> None:
    """Populate cross-engine agreement metrics: char/word F1, line detection P/R/F1, reading order."""
    record.agreement_char_f1 = char_f1(sceptre.text, easyocr.text)
    record.agreement_word_f1 = word_f1(sceptre.text, easyocr.text)
    line_f1, line_precision, line_recall = line_detection_scores(easyocr.quads, sceptre.quads)
    record.agreement_line_f1 = line_f1
    record.agreement_line_precision = line_precision
    record.agreement_line_recall = line_recall
    record.agreement_reading_order = reading_order_score(sceptre.text, easyocr.text)


def _score_accuracy(
    record: ImageRecord, reference_raw: str, easyocr: ImageDetections, sceptre: ImageDetections
) -> None:
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
    binary: Path,
    root: Path,
    overhead: float | None,
    repeats: int,
    threads: int | None,
    sceptre_slice: ImageDetections | None,
    easyocr_slice: ImageDetections | None,
    sceptre_rss: float | None,
    easyocr_rss: float | None,
) -> ImageRecord:
    """Assemble one entry's metrics from its resolved batch slices plus a cold sceptre timing.

    Slices are looked up by image path outside; ``sceptre_rss`` / ``easyocr_rss`` are the
    group-level peak RSS for the entry's measured language batches (``None`` for capability
    probes, which are excluded from the timed passes).
    """
    record = ImageRecord(stem=entry.stem, group=entry.group, languages=list(entry.languages))
    if entry.image is None or not entry.image.exists():
        record.skipped = "image not found on disk"
        return record

    if sceptre_slice is not None and sceptre_slice.unsupported_format:
        record.unsupported_format = True
        record.skipped = "sceptre cannot decode this format (capability gap)"
        # EasyOCR memory/time for this group is still meaningful, but per-image parity is not. ~keep
        return record
    if sceptre_slice is None or sceptre_slice.detections is None:
        detail = sceptre_slice.error if sceptre_slice else "missing from sceptre batch output"
        record.skipped = f"sceptre error: {detail}"
        return record
    if easyocr_slice is None or easyocr_slice.detections is None:
        detail = easyocr_slice.error if easyocr_slice else "missing from easyocr batch output"
        record.skipped = f"easyocr error: {detail}"
        return record

    # Warm figures come from the batch passes (group-level RSS, per-image warm time). ~keep
    record.easyocr_warm_seconds = easyocr_slice.seconds
    record.easyocr_rss_mb = easyocr_rss
    record.sceptre_rss_mb = sceptre_rss

    try:
        cold_times = [
            run_sceptre_cold(binary, entry.image, entry.sceptre_langs, root, threads) for _ in range(max(repeats, 1))
        ]
        record.sceptre_cold_seconds = _median(cold_times)
        if overhead is not None:
            record.sceptre_amortized_seconds = max(record.sceptre_cold_seconds - overhead, 0.0)
    except (RuntimeError, ValueError) as error:
        record.skipped = f"sceptre cold run failed: {str(error)[:200]}"
        return record

    _score_agreement(record, easyocr_slice, sceptre_slice)
    if entry.ground_truth is not None and entry.ground_truth.exists():
        _score_accuracy(record, entry.ground_truth.read_text(encoding="utf-8"), easyocr_slice, sceptre_slice)
    return record


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------


"""Key of the one `_summary` statistic that is a tally rather than a value in the metric's units."""
SAMPLE_COUNT_STAT = "count"


def _summary(values: list[float]) -> dict[str, float | None] | None:
    """Median / p95 / p99 / mean / count over a list of floats, or None if empty.

    p95 and p99 use R-7 interpolation (`percentile_r7`) and are suppressed to `None`
    below `P95_MIN_SAMPLES` / `P99_MIN_SAMPLES` samples respectively — a percentile from
    a handful of images is just the maximum wearing a statistical label.
    """
    if not values:
        return None
    return {
        "median": statistics.median(values),
        "p95": suppressed_percentile(values, 0.95, P95_MIN_SAMPLES),
        "p99": suppressed_percentile(values, 0.99, P99_MIN_SAMPLES),
        "mean": statistics.fmean(values),
        "count": len(values),
    }


def _collect(records: list[ImageRecord], attribute: str) -> list[float]:
    """Gather the non-None values of one attribute across records."""
    return [value for value in (getattr(record, attribute) for record in records) if value is not None]


AGGREGATE_FIELDS = (
    "easyocr_warm_seconds",
    "sceptre_cold_seconds",
    "sceptre_amortized_seconds",
    "easyocr_rss_mb",
    "sceptre_rss_mb",
    "agreement_char_f1",
    "agreement_word_f1",
    "agreement_line_f1",
    "agreement_line_precision",
    "agreement_line_recall",
    "agreement_reading_order",
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

    # Bump when a field is added, removed, or changes meaning in `to_dict()`'s output, so a
    # consumer (publish.py's drift check, a stored baseline) can tell reports apart. Mirrors
    # the published comparison artifact, which has carried a schema_version since inception.
    SCHEMA_VERSION = 1

    metadata: dict[str, object]
    records: list[ImageRecord]
    aggregates_labeled: dict[str, dict[str, float]] = field(default_factory=dict)
    aggregates_all: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialize the whole report for JSON output."""
        return {
            "schema_version": self.SCHEMA_VERSION,
            "metadata": self.metadata,
            "aggregates": {"labeled": self.aggregates_labeled, "all": self.aggregates_all},
            "records": [record.to_dict() for record in self.records],
        }


# Aggregate/record fields that are scores bounded to [0, 1]; anything else measured (seconds,
# RSS, CPU core-seconds) only needs to be finite. ~keep
_UNIT_INTERVAL_FIELDS = frozenset(
    {
        "agreement_char_f1",
        "agreement_word_f1",
        "agreement_line_f1",
        "agreement_line_precision",
        "agreement_line_recall",
        "agreement_reading_order",
        "easyocr_token_f1",
        "sceptre_token_f1",
    }
)


def validate_report(report: RunReport) -> list[str]:
    """Reject a malformed report -- NaN or an out-of-range score -- before it is written to disk.

    A regression *gate* failure (a real quality/speed drop) is expected to happen and still
    leaves a complete, inspectable report on disk; this catches a different kind of failure --
    a bug producing a NaN or an impossible score -- which must never land in comparison.json in
    the first place.
    """
    problems: list[str] = []
    for aggregate_name, aggregates in (("labeled", report.aggregates_labeled), ("all", report.aggregates_all)):
        for field_name, summary in aggregates.items():
            for stat_name, value in summary.items():
                label = f"aggregates.{aggregate_name}.{field_name}.{stat_name}"
                if stat_name == SAMPLE_COUNT_STAT:
                    problems.extend(_validate_sample_count(label, value))
                else:
                    problems.extend(_validate_score(label, value, field_name))
    for record in report.records:
        for field_name in _UNIT_INTERVAL_FIELDS:
            problems.extend(
                _validate_score(f"records[{record.stem}].{field_name}", getattr(record, field_name), field_name)
            )
    return problems


def _validate_sample_count(label: str, value: float | None) -> list[str]:
    """Validate a `_summary` sample count, which is a tally rather than a score."""
    if value is None:
        return []
    if not math.isfinite(value) or value < 0 or value != int(value):
        return [f"{label} = {value} is not a sample count"]
    return []


def _validate_score(label: str, value: float | None, field_name: str) -> list[str]:
    """Validate one numeric field, or skip it if unset."""
    if value is None:
        return []
    if not math.isfinite(value):
        return [f"{label} is not finite: {value}"]
    if field_name in _UNIT_INTERVAL_FIELDS and not (0.0 <= value <= 1.0):
        return [f"{label} = {value} is outside [0, 1]"]
    return []


# --------------------------------------------------------------------------------------
# Headline totals
# --------------------------------------------------------------------------------------


@dataclass
class HeadlineTotals:
    """Corpus-wide warm/cold totals and peak RSS for the honest headline figures."""

    scored_n: int
    sceptre_warm_total: float
    sceptre_warm_imgs: int
    sceptre_cold_total: float
    easyocr_warm_total: float
    easyocr_warm_imgs: int
    sceptre_peak_rss: float | None
    easyocr_peak_rss: float | None


def _batch_totals(batch: dict[str, object]) -> tuple[float, int]:
    """Sum per-group warm wall time and image count from a serialized batch summary."""
    groups = [group for group in batch.values() if isinstance(group, dict)]
    total = sum(float(g["total_seconds"]) for g in groups if isinstance(g.get("total_seconds"), (int, float)))
    images = sum(int(g["image_count"]) for g in groups if isinstance(g.get("image_count"), int))
    return total, images


def _peak_rss(batch: dict[str, object]) -> float | None:
    """Max peak RSS across the language groups of a serialized batch summary."""
    values = [g["rss_mb"] for g in batch.values() if isinstance(g, dict) and isinstance(g.get("rss_mb"), (int, float))]
    return max(values) if values else None


def headline_totals(report: RunReport) -> HeadlineTotals:
    """Corpus-total warm/cold figures and peak RSS from the batch metadata."""
    scored = [record for record in report.records if record.skipped is None]
    sceptre_batch = report.metadata.get("sceptre_batch", {})
    easyocr_batch = report.metadata.get("easyocr_batch", {})
    sceptre_batch = sceptre_batch if isinstance(sceptre_batch, dict) else {}
    easyocr_batch = easyocr_batch if isinstance(easyocr_batch, dict) else {}
    sceptre_total, sceptre_imgs = _batch_totals(sceptre_batch)
    easyocr_total, easyocr_imgs = _batch_totals(easyocr_batch)
    return HeadlineTotals(
        scored_n=len(scored),
        sceptre_warm_total=sceptre_total,
        sceptre_warm_imgs=sceptre_imgs,
        sceptre_cold_total=sum(_collect(scored, "sceptre_cold_seconds")),
        easyocr_warm_total=easyocr_total,
        easyocr_warm_imgs=easyocr_imgs,
        sceptre_peak_rss=_peak_rss(sceptre_batch),
        easyocr_peak_rss=_peak_rss(easyocr_batch),
    )


def _throughput(count: int, total_seconds: float) -> float | None:
    """Images-per-second, or None if not measurable."""
    if not total_seconds or count <= 0:
        return None
    return count / total_seconds


def ratio_claim(value: float | None) -> str | None:
    """Render a ratio as ``2.3x``, or None when it is too close to parity to be a claim.

    One decimal place, never zero: rounding 1.08 to ``1x`` produces "1x lower", which
    asserts an advantage and states equality at the same time.
    """
    if value is None or value < MIN_CLAIMABLE_RATIO:
        return None
    return f"{value:.1f}x"


def speedup_summary(report: RunReport) -> str:
    """One-line warm speedup and peak-RSS ratio versus EasyOCR."""
    totals = headline_totals(report)
    parts: list[str] = []
    warm_speedup = ratio_claim(_warm_speedup(totals))
    if warm_speedup is not None:
        parts.append(f"~{warm_speedup} faster warm")
    if totals.sceptre_cold_total > 0 and totals.easyocr_warm_total > 0:
        cold_speedup = ratio_claim(totals.easyocr_warm_total / totals.sceptre_cold_total)
        if cold_speedup is not None:
            parts.append(f"~{cold_speedup} faster cold")
    rss_ratio = ratio_claim(paired_rss_ratio(_record_rss_pairs(report.records)))
    if rss_ratio is not None:
        parts.append(f"~{rss_ratio} less peak RSS (per-image median)")
    if not parts:
        return ""
    return "sceptre is " + ", ".join(parts) + " vs EasyOCR (per-image normalized, both warm/batch subprocesses)."


def _warm_speedup(totals: HeadlineTotals) -> float | None:
    """Per-image-normalized warm speedup of sceptre over EasyOCR, or None."""
    if not (totals.sceptre_warm_total > 0 and totals.sceptre_warm_imgs > 0):
        return None
    if not (totals.easyocr_warm_total > 0 and totals.easyocr_warm_imgs > 0):
        return None
    return (totals.easyocr_warm_total / totals.easyocr_warm_imgs) / (
        totals.sceptre_warm_total / totals.sceptre_warm_imgs
    )


def paired_rss_ratio(pairs: Iterable[tuple[float | None, float | None]]) -> float | None:
    """Median of the per-image EasyOCR-over-sceptre peak-RSS ratios, or None if unmeasurable.

    Each pair is one image's ``(sceptre, easyocr)`` peak RSS, taken from the batch that image
    was actually measured in, so numerator and denominator describe the same page. The corpus
    maxima do not: each engine's max is picked independently, so their quotient can compare two
    measurements that never co-occurred, and it collapses toward 1 as the largest page grows
    because peak RSS is dominated by the CRAFT canvas both engines allocate (ADR 0042).

    The median is the central tendency because these are ratios: it is robust to a single
    outlying batch, and it is invariant under inversion, so the sceptre-over-EasyOCR median is
    exactly the reciprocal of this one. Taking it over images rather than over batches weights
    each language group by how much of the corpus it covers.
    """
    ratios = [
        easyocr / sceptre
        for sceptre, easyocr in pairs
        if isinstance(sceptre, (int, float)) and isinstance(easyocr, (int, float)) and sceptre > 0
    ]
    return statistics.median(ratios) if ratios else None


def _record_rss_pairs(records: list[ImageRecord]) -> list[tuple[float | None, float | None]]:
    """Each scored image's ``(sceptre, easyocr)`` peak RSS, as :func:`paired_rss_ratio` wants."""
    return [(record.sceptre_rss_mb, record.easyocr_rss_mb) for record in records if record.skipped is None]


# --------------------------------------------------------------------------------------
# Markdown report
# --------------------------------------------------------------------------------------


def _fmt(value: float | None, digits: int = 3) -> str:
    """Format an optional number for a Markdown cell."""
    return "-" if value is None else f"{value:.{digits}f}"


def _mean_of(aggregates: dict[str, dict[str, float]], field_name: str) -> float | None:
    """Mean of an aggregate field, or None if absent."""
    summary = aggregates.get(field_name)
    return summary["mean"] if summary else None


def _median_of(aggregates: dict[str, dict[str, float]], field_name: str) -> float | None:
    """Median of an aggregate field, or None if absent."""
    summary = aggregates.get(field_name)
    return summary["median"] if summary else None


def _markdown_table(header: list[str], rows: list[list[str]]) -> str:
    """Render a simple GitHub-flavored Markdown table."""
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


HEADLINE_HEADER = [
    "Engine",
    "Corpus total s",
    "Throughput img/s",
    "Peak RSS MB",
    "Mean CER",
    "Mean WER",
    "Mean token-F1",
]


def headline_rows(report: RunReport) -> list[list[str]]:
    """Build the headline comparison table rows (both engines warm/batch, sceptre cold)."""
    labeled = report.aggregates_labeled
    totals = headline_totals(report)
    return [
        [
            "EasyOCR (warm/batch)",
            _fmt(totals.easyocr_warm_total, 2),
            _fmt(_throughput(totals.easyocr_warm_imgs, totals.easyocr_warm_total), 2),
            _fmt(totals.easyocr_peak_rss, 1),
            _fmt(_mean_of(labeled, "easyocr_cer")),
            _fmt(_mean_of(labeled, "easyocr_wer")),
            _fmt(_mean_of(labeled, "easyocr_token_f1")),
        ],
        [
            "sceptre (warm/batch)",
            _fmt(totals.sceptre_warm_total, 2),
            _fmt(_throughput(totals.sceptre_warm_imgs, totals.sceptre_warm_total), 2),
            _fmt(totals.sceptre_peak_rss, 1),
            _fmt(_mean_of(labeled, "sceptre_cer")),
            _fmt(_mean_of(labeled, "sceptre_wer")),
            _fmt(_mean_of(labeled, "sceptre_token_f1")),
        ],
        [
            "sceptre (cold CLI run)",
            _fmt(totals.sceptre_cold_total, 2),
            _fmt(_throughput(totals.scored_n, totals.sceptre_cold_total), 2),
            _fmt(totals.sceptre_peak_rss, 1),
            _fmt(_mean_of(labeled, "sceptre_cer")),
            _fmt(_mean_of(labeled, "sceptre_wer")),
            _fmt(_mean_of(labeled, "sceptre_token_f1")),
        ],
    ]


PER_IMAGE_HEADER = [
    "Image",
    "Group",
    "Lang",
    "EasyOCR warm s",
    "sceptre cold s",
    "sceptre amort s",
    "char-F1",
    "word-F1",
    "line-F1",
    "line-P",
    "line-R",
    "read-order",
    "Easy CER",
    "Scep CER",
    "Note",
]


def _baseline_index(baseline: dict[str, object] | None) -> dict[str, dict[str, object]]:
    """Index a prior report's records by stem for delta lookups."""
    if not baseline:
        return {}
    records = baseline.get("records", [])
    return {str(record.get("stem")): record for record in records if isinstance(record, dict)}


def _delta(current: float | None, previous: object) -> str:
    """Signed delta ``current - previous`` for a Markdown cell, or '-' if unavailable."""
    if current is None or not isinstance(previous, (int, float)):
        return "-"
    difference = current - float(previous)
    return f"{difference:+.3f}"


def per_image_rows(records: list[ImageRecord], baseline: dict[str, dict[str, object]]) -> list[list[str]]:
    """Build the per-image detail table rows, with a baseline delta note when available."""
    rows: list[list[str]] = []
    for record in records:
        note = record.skipped or ""
        prior = baseline.get(record.stem)
        if prior is not None and record.sceptre_token_f1 is not None:
            delta = _delta(record.sceptre_token_f1, prior.get("sceptre_token_f1"))
            note = (f"ΔtokF1 {delta}; " + note).strip("; ").strip()
        rows.append(
            [
                record.stem,
                record.group,
                "+".join(record.languages),
                _fmt(record.easyocr_warm_seconds),
                _fmt(record.sceptre_cold_seconds),
                _fmt(record.sceptre_amortized_seconds),
                _fmt(record.agreement_char_f1),
                _fmt(record.agreement_word_f1),
                _fmt(record.agreement_line_f1),
                _fmt(record.agreement_line_precision),
                _fmt(record.agreement_line_recall),
                _fmt(record.agreement_reading_order),
                _fmt(record.easyocr_cer),
                _fmt(record.sceptre_cer),
                note,
            ]
        )
    return rows


def _environment_lines(meta: dict[str, object]) -> list[str]:
    """One Markdown provenance line naming both runtimes, or nothing when unrecorded."""
    environment = meta.get("environment")
    if not isinstance(environment, dict):
        return []
    reference = environment.get("reference")
    reference = reference if isinstance(reference, dict) else {}
    onnxruntime = environment.get("onnxruntime")
    onnxruntime = onnxruntime.get("version") if isinstance(onnxruntime, dict) else onnxruntime
    return [
        (
            f"- Runtime: sceptre {environment.get('sceptre_version') or UNKNOWN} "
            f"({environment.get('backend') or UNKNOWN}/{environment.get('accelerator') or UNKNOWN}, "
            f"ONNX Runtime {onnxruntime or UNKNOWN}) on {environment.get('os') or UNKNOWN}/"
            f"{environment.get('arch') or UNKNOWN}, {environment.get('cpu_count') or UNKNOWN} cores "
            f"| reference: EasyOCR {reference.get('easyocr_version') or UNKNOWN}, "
            f"torch {reference.get('torch_version') or UNKNOWN}"
        )
    ]


def render_markdown(report: RunReport, baseline: dict[str, object] | None = None) -> str:
    """Render the full scannable Markdown report."""
    meta = report.metadata
    capability = [record for record in report.records if record.unsupported_format]
    skips = [record for record in report.records if record.skipped is not None and not record.unsupported_format]
    sections = [
        "# sceptre vs EasyOCR benchmark",
        "",
        f"- Platform: `{meta['platform']}` | sceptre binary: `{meta['sceptre_binary']}`",
        *_environment_lines(meta),
        (
            f"- Corpus: {meta['corpus_total']} entries "
            f"({meta['labeled_scored']} labeled scored, {meta['breadth_scored']} breadth scored, "
            f"{len(capability)} capability gaps, {len(skips)} skipped)"
        ),
        (
            f"- Repeats: {meta.get('repeats')} (median wall time, max peak RSS) | "
            f"threads: {meta.get('threads')} (same for both engines)"
        ),
        "",
        "## Headline",
        "",
        _markdown_table(HEADLINE_HEADER, headline_rows(report)),
        "",
        speedup_summary(report),
        "",
        (
            "Both engines run as fresh subprocesses per language group, loading their model/Reader once "
            "and processing every image (warm/batch), with peak RSS captured identically via `/usr/bin/time`. "
            "EasyOCR peak RSS legitimately includes the Python + torch runtime — that is its real cost. "
            "CER / WER / token-F1 are averaged over labeled images only; cold is the per-invocation sceptre "
            "cost (fresh process per image, pays model load every time)."
        ),
        "",
        "## Per-image detail",
        "",
        _markdown_table(PER_IMAGE_HEADER, per_image_rows(report.records, _baseline_index(baseline))),
        "",
    ]
    if capability:
        sections += ["## Capability gaps (sceptre cannot decode; EasyOCR can)", ""]
        for record in capability:
            sections.append(f"- `{record.stem}` ({'+'.join(record.languages)})")
        sections.append("")
    if skips:
        sections += ["## Skipped", ""]
        for record in skips:
            sections.append(f"- `{record.stem}`: {record.skipped}")
        sections.append("")
    return "\n".join(sections)


# --------------------------------------------------------------------------------------
# Regression gate (--assert)
# --------------------------------------------------------------------------------------


def _host_identity(os_name: object, arch: object, runner_label: object) -> tuple[str, str, str]:
    """Normalize a host into the identity a floor is scoped to (ADR 0042).

    A runner class is part of the identity, not a detail of it: ``Linux/x86_64`` names two
    machines whose absolute memory and timing figures are not comparable (ADR 0035).
    """
    os_text, arch_text, label_text = (
        value.strip() if isinstance(value, str) and value.strip() else UNKNOWN
        for value in (os_name, arch, runner_label)
    )
    return (os_text, arch_text, label_text)


def _host_label(identity: tuple[str, str, str]) -> str:
    """A host identity as one readable phrase, naming the runner class when there is one."""
    os_name, arch, runner_label = identity
    host = f"{os_name}/{arch}"
    return host if runner_label == UNKNOWN else f"{runner_label} ({host})"


def run_host(report: RunReport) -> tuple[str, str, str]:
    """The host a measured report was produced on."""
    environment = report.metadata.get("environment")
    environment = environment if isinstance(environment, dict) else {}
    return _host_identity(environment.get("os"), environment.get("arch"), environment.get("runner_label"))


def published_host(published: dict[str, object]) -> tuple[str, str, str]:
    """The host a published baseline artifact was measured on.

    ``publish`` splits the provenance: the operating system and architecture stay in
    ``environment``, while the runner label is lifted into the ``run`` block.
    """
    environment = published.get("environment")
    environment = environment if isinstance(environment, dict) else {}
    run = published.get("run")
    run = run if isinstance(run, dict) else {}
    return _host_identity(environment.get("os"), environment.get("arch"), run.get("runner_label"))


def published_sceptre_peak_rss(published: dict[str, object]) -> float | None:
    """The sceptre warm/batch peak RSS a published baseline carries, or None if it has none."""
    headline = published.get("headline")
    headline = headline if isinstance(headline, dict) else {}
    warm = headline.get("sceptre_warm")
    warm = warm if isinstance(warm, dict) else {}
    value = warm.get("peak_rss_mb")
    return float(value) if isinstance(value, (int, float)) else None


def published_sceptre_warm_throughput(published: dict[str, object]) -> float | None:
    """The sceptre warm/batch throughput a published baseline carries, or None if it has none."""
    headline = published.get("headline")
    headline = headline if isinstance(headline, dict) else {}
    warm = headline.get("sceptre_warm")
    warm = warm if isinstance(warm, dict) else {}
    value = warm.get("throughput_img_s")
    return float(value) if isinstance(value, (int, float)) else None


def load_published_baseline(path: Path) -> dict[str, object] | None:
    """Read the committed published artifact, or None when it is absent or unreadable.

    Returning None is not a pass: :func:`check_rss_ceiling` turns it into a breach naming the
    path, because a gate that cannot be evaluated must not read as a gate that was cleared.
    """
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        print(f"sceptre_rs_tools.benchmark: could not read published baseline {path}: {error}", file=sys.stderr)
        return None
    return payload if isinstance(payload, dict) else None


def check_rss_ceiling(
    measured_mb: float | None,
    host: tuple[str, str, str],
    published: dict[str, object] | None,
    factor: float = SCEPTRE_RSS_CEILING_FACTOR,
) -> list[str]:
    """Breaches of the absolute, host-scoped ceiling on sceptre's own peak RSS (ADR 0042).

    Self-referential by construction: it compares this run's sceptre peak RSS against the
    figure the committed baseline published for the same host, so an EasyOCR release that
    changes torch's footprint cannot move it, and the corpus's own image sizes cancel out on
    both sides. Every way of not being able to evaluate it -- no baseline, no field in the
    baseline, a baseline from another host, or a run with no RSS measured -- is a breach.
    """
    ceiling_of = f"sceptre peak-RSS ceiling ({_host_label(host)})"
    if published is None:
        return [
            (
                f"{ceiling_of} unmeasurable: no committed baseline at {PUBLISHED_RELATIVE_PATH}; "
                "measure this host and run `task python:publish`"
            )
        ]
    baseline_host = published_host(published)
    if baseline_host != host:
        return [
            (
                f"{ceiling_of} unmeasurable: {PUBLISHED_RELATIVE_PATH} was measured on "
                f"{_host_label(baseline_host)}; absolute floors are scoped to a host (ADR 0042), so "
                "re-publish from a run on this one"
            )
        ]
    baseline_mb = published_sceptre_peak_rss(published)
    if baseline_mb is None or baseline_mb <= 0:
        return [
            (
                f"{ceiling_of} unmeasurable: {PUBLISHED_RELATIVE_PATH} carries no "
                "headline.sceptre_warm.peak_rss_mb to gate against"
            )
        ]
    if measured_mb is None:
        return [f"{ceiling_of} unmeasurable: this run measured no sceptre peak RSS (is /usr/bin/time present?)"]
    ceiling_mb = baseline_mb * factor
    if measured_mb > ceiling_mb:
        return [
            (
                f"sceptre peak RSS {measured_mb:,.1f} MB > ceiling {ceiling_mb:,.1f} MB "
                f"(published baseline {baseline_mb:,.1f} MB x {factor:.2f}, {_host_label(host)})"
            )
        ]
    return []


def check_warm_throughput_floor(
    measured_img_s: float | None,
    host: tuple[str, str, str],
    published: dict[str, object] | None,
    factor: float = SCEPTRE_THROUGHPUT_FLOOR_FACTOR,
) -> list[str]:
    """Breaches of the absolute, host-scoped floor on sceptre's own warm throughput (ADR 0043).

    Self-referential by construction: it compares this run's sceptre warm/batch throughput
    against the figure the committed baseline published for the same host, so a faster EasyOCR
    release cannot report itself as a sceptre regression, and the parallel scaling a host's core
    count allows is baked into both sides rather than into a cross-engine ratio. Every way of not
    being able to evaluate it -- no baseline, no field in the baseline, a baseline from another
    host, or a run with no timings -- is a breach.
    """
    floor_of = f"sceptre warm-throughput floor ({_host_label(host)})"
    if published is None:
        return [
            (
                f"{floor_of} unmeasurable: no committed baseline at {PUBLISHED_RELATIVE_PATH}; "
                "measure this host and run `task python:publish`"
            )
        ]
    baseline_host = published_host(published)
    if baseline_host != host:
        return [
            (
                f"{floor_of} unmeasurable: {PUBLISHED_RELATIVE_PATH} was measured on "
                f"{_host_label(baseline_host)}; absolute floors are scoped to a host (ADR 0043), so "
                "re-publish from a run on this one"
            )
        ]
    baseline_img_s = published_sceptre_warm_throughput(published)
    if baseline_img_s is None or baseline_img_s <= 0:
        return [
            (
                f"{floor_of} unmeasurable: {PUBLISHED_RELATIVE_PATH} carries no "
                "headline.sceptre_warm.throughput_img_s to gate against"
            )
        ]
    if measured_img_s is None:
        return [f"{floor_of} unmeasurable: this run measured no sceptre warm throughput (missing batch timings)"]
    floor_img_s = baseline_img_s * factor
    if measured_img_s < floor_img_s:
        return [
            (
                f"sceptre warm throughput {measured_img_s:,.3f} img/s < floor {floor_img_s:,.3f} img/s "
                f"(published baseline {baseline_img_s:,.3f} img/s x {factor:.2f}, {_host_label(host)})"
            )
        ]
    return []


def check_thresholds(report: RunReport, published: dict[str, object] | None = None) -> list[str]:
    """Return a list of threshold breaches (empty when the run clears the gate)."""
    totals = headline_totals(report)
    host = run_host(report)
    measured_throughput = _throughput(totals.sceptre_warm_imgs, totals.sceptre_warm_total)

    breaches = check_warm_throughput_floor(measured_throughput, host, published)
    breaches.extend(check_rss_ceiling(totals.sceptre_peak_rss, host, published))

    return breaches


def derive_guardrails(
    records: list[ImageRecord], threshold_factor: float = GUARDRAIL_THRESHOLD_FACTOR
) -> dict[str, object]:
    """Build a per-image guardrail contract from a baseline run's scored, labeled images.

    Ported from xberg's `split_benchmark.rs` boundary guardrails: each image's floor is its
    baseline `sceptre_token_f1` times `threshold_factor` (~0.9), so normal run-to-run
    variation does not trip the gate but a real regression isolated to one image does --
    something a single corpus-mean quality check cannot see.
    """
    contracts = [
        {"stem": record.stem, "min_token_f1": record.sceptre_token_f1 * threshold_factor}
        for record in records
        if record.skipped is None and record.group == "labeled" and record.sceptre_token_f1 is not None
    ]
    return {"schema_version": GUARDRAILS_SCHEMA_VERSION, "threshold_factor": threshold_factor, "contracts": contracts}


def load_guardrails(path: Path) -> dict[str, object] | None:
    """Load a guardrail contract file, or ``None`` if it does not exist.

    Raises ``ValueError`` for an existing file with an unsupported ``schema_version`` --
    a stale gate silently checking the wrong floors is worse than a loud failure.
    """
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != GUARDRAILS_SCHEMA_VERSION:
        raise ValueError(f"unsupported guardrails schema_version: {payload.get('schema_version')!r}")
    return payload


def check_guardrails(records: list[ImageRecord], guardrails: dict[str, object]) -> list[str]:
    """Per-image guardrail breaches (empty when every contract clears its floor).

    A missing record or a non-finite score is a failure, not a silently-skipped contract --
    an image that stopped being measured at all is exactly the kind of regression this gate
    exists to catch.
    """
    by_stem = {record.stem: record for record in records}
    failures: list[str] = []
    for contract in guardrails.get("contracts", []):
        stem = contract.get("stem")
        floor = contract.get("min_token_f1")
        if floor is None:
            continue
        record = by_stem.get(stem)
        score = record.sceptre_token_f1 if record is not None else None
        if score is None or not math.isfinite(score):
            failures.append(f"{stem}: sceptre token-F1 unmeasurable (guardrail requires >= {floor:.3f})")
        elif score < floor:
            failures.append(f"{stem}: sceptre token-F1 {score:.3f} < guardrail floor {floor:.3f}")
    return failures


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


def _batch_summary(results: dict[tuple[str, ...], BatchResult]) -> dict[str, dict[str, object]]:
    """Per-language-group warm/batch summary for the JSON metadata."""
    summary: dict[str, dict[str, object]] = {}
    for langs, result in results.items():
        summary["+".join(langs)] = {
            "total_seconds": result.total_seconds,
            "image_count": result.image_count,
            "throughput": _throughput(result.image_count, result.total_seconds),
            "warm_per_image_seconds": result.warm_per_image(),
            "build_seconds": result.build_seconds,
            "rss_mb": result.rss_mb,
            "cpu_core_seconds": result.cpu_core_seconds,
        }
    return summary


def _merge_per_image(results: list[BatchResult]) -> dict[str, ImageDetections]:
    """Flatten every batch's per-image slices into one path-keyed lookup."""
    merged: dict[str, ImageDetections] = {}
    for result in results:
        merged.update(result.per_image)
    return merged


def _load_baseline(path: Path | None) -> dict[str, object] | None:
    """Load a prior comparison.json for delta rendering, or None if absent/unreadable."""
    if path is None:
        return None
    if not path.exists():
        print(f"sceptre_rs_tools.benchmark: baseline {path} not found; skipping deltas.", file=sys.stderr)
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        print(f"sceptre_rs_tools.benchmark: could not read baseline {path}: {error}", file=sys.stderr)
        return None


def run_benchmark(
    group: str,
    limit: int | None,
    output_dir: Path,
    repeats: int,
    threads: int | None,
    baseline_path: Path | None,
    assert_gate: bool,
    guardrails_path: Path | None = None,
    write_guardrails_path: Path | None = None,
) -> int:
    """Run the full benchmark and write JSON + Markdown; returns a process exit code."""
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

    # Capability probes fail-fast on decode; excluding them from the timed passes keeps them
    # out of the speed/RSS headline (they are gaps, not wins). They still get a one-shot
    # detection probe so the report can list them. ~keep
    measured = [entry for entry in entries if entry.group != "capability"]
    capability = [entry for entry in entries if entry.group == "capability"]

    overhead = measure_sceptre_overhead(binary, measured, root, threads)
    sceptre_results = run_sceptre_pass(binary, measured, root, repeats, threads)
    easyocr_results = run_easyocr_pass(measured, root, repeats, threads)
    cap_sceptre = run_sceptre_pass(binary, capability, root, 1, threads) if capability else {}
    cap_easyocr = run_easyocr_pass(capability, root, 1, threads) if capability else {}

    sceptre_by_path = _merge_per_image([*sceptre_results.values(), *cap_sceptre.values()])
    easyocr_by_path = _merge_per_image([*easyocr_results.values(), *cap_easyocr.values()])

    records: list[ImageRecord] = []
    for entry in entries:
        key = str(entry.image) if entry.image is not None else ""
        sceptre_batch = sceptre_results.get(tuple(entry.sceptre_langs))
        easyocr_batch = easyocr_results.get(tuple(entry.easyocr_codes))
        print(f"sceptre_rs_tools.benchmark: score {entry.stem} ({'+'.join(entry.easyocr_codes)})", file=sys.stderr)
        records.append(
            benchmark_entry(
                entry,
                binary,
                root,
                overhead,
                repeats,
                threads,
                sceptre_by_path.get(key),
                easyocr_by_path.get(key),
                sceptre_batch.rss_mb if sceptre_batch else None,
                easyocr_batch.rss_mb if easyocr_batch else None,
            )
        )

    labeled = [record for record in records if record.group == "labeled"]
    scored = [record for record in records if record.skipped is None]
    reference_versions = next((batch.versions for batch in easyocr_results.values() if batch.versions), {})
    languages = sorted({language for entry in measured for language in entry.sceptre_langs})
    metadata = {
        "platform": platform.platform(),
        "sceptre_binary": str(SCEPTRE_BIN),
        "environment": environment_metadata(binary, root, languages, reference_versions, entries),
        "group": group,
        "repeats": repeats,
        "threads": threads if threads is not None else "engine default (all cores)",
        "corpus_total": len(records),
        "labeled_scored": sum(1 for record in labeled if record.skipped is None),
        "breadth_scored": sum(1 for record in scored if record.group == "breadth"),
        "capability_gaps": sum(1 for record in records if record.unsupported_format),
        "sceptre_overhead_seconds": overhead,
        "sceptre_batch": _batch_summary(sceptre_results),
        "easyocr_batch": _batch_summary(easyocr_results),
        "overhead_runs": OVERHEAD_RUNS,
    }
    report = RunReport(
        metadata=metadata,
        records=records,
        aggregates_labeled=aggregate(labeled),
        aggregates_all=aggregate(records),
    )

    problems = validate_report(report)
    if problems:
        print("sceptre_rs_tools.benchmark: report failed validation, not writing artifacts:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    baseline = _load_baseline(baseline_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "comparison.md").write_text(render_markdown(report, baseline) + "\n", encoding="utf-8")

    print(_markdown_table(HEADLINE_HEADER, headline_rows(report)))
    summary = speedup_summary(report)
    if summary:
        print(summary)
    print(f"\nWrote {output_dir / 'comparison.json'} and {output_dir / 'comparison.md'}", file=sys.stderr)

    if write_guardrails_path is not None:
        guardrails = derive_guardrails(records)
        write_guardrails_path.write_text(json.dumps(guardrails, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {write_guardrails_path} ({len(guardrails['contracts'])} contracts)", file=sys.stderr)

    if assert_gate:
        breaches = check_thresholds(report, load_published_baseline(root / PUBLISHED_RELATIVE_PATH))
        if guardrails_path is not None:
            guardrails = load_guardrails(guardrails_path)
            if guardrails is None:
                breaches.append(f"guardrails file not found: {guardrails_path}")
            else:
                breaches.extend(check_guardrails(report.records, guardrails))
        if breaches:
            print("\nBenchmark gate FAILED:", file=sys.stderr)
            for breach in breaches:
                print(f"  - {breach}", file=sys.stderr)
            return 1
        print("\nBenchmark gate passed.", file=sys.stderr)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the benchmark CLI arguments."""
    parser = argparse.ArgumentParser(description="Benchmark sceptre against upstream EasyOCR.")
    parser.add_argument(
        "--group",
        choices=["labeled", "breadth", "capability", "all"],
        default="all",
        help="Corpus subset to run (default: all).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of images (for smoke runs).")
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"Timing repeats per measurement (default: {DEFAULT_REPEATS}).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Pin both engines to N threads (default: each engine's native multi-threaded default).",
    )
    parser.add_argument(
        "--baseline", type=Path, default=None, help="Prior comparison.json to render per-image deltas against."
    )
    parser.add_argument(
        "--assert", dest="assert_gate", action="store_true", help="Fail (exit 1) if a regression floor is breached."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root() / OUTPUT_DIR,
        help="Directory for comparison.json and comparison.md.",
    )
    parser.add_argument(
        "--guardrails",
        type=Path,
        default=None,
        help="Per-image guardrail contract file (see derive_guardrails); checked with --assert.",
    )
    parser.add_argument(
        "--write-guardrails",
        type=Path,
        default=None,
        help="Derive a guardrail contract file from this run's scored, labeled images and write it here.",
    )
    return parser.parse_args(argv)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    sys.exit(
        run_benchmark(
            group=args.group,
            limit=args.limit,
            output_dir=args.output_dir,
            repeats=args.repeats,
            threads=args.threads,
            baseline_path=args.baseline,
            assert_gate=args.assert_gate,
            guardrails_path=args.guardrails,
            write_guardrails_path=args.write_guardrails,
        )
    )


if __name__ == "__main__":
    main()
