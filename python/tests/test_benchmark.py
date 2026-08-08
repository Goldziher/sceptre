"""Unit tests for the pure logic of the sceptre-vs-EasyOCR benchmark harness.

These cover the parts that do not need torch/easyocr or the release binary: metric parity,
subprocess stderr hygiene, capability-gap detection, report rendering (capability section
and baseline deltas), and the regression gate.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest
from sceptre_rs_tools import benchmark as b

# -- text/box metrics (mirror crates/sceptre/tests/helpers/mod.rs) ----------------------


def test_word_f1_is_one_for_identical_strings() -> None:
    assert b.word_f1("hello world", "hello world") == 1.0


def test_word_f1_is_zero_for_disjoint_strings() -> None:
    assert b.word_f1("alpha", "beta") == 0.0


def test_char_f1_ignores_whitespace() -> None:
    assert b.char_f1("a b c", "abc") == 1.0


def test_word_f1_bigram_expands_cjk_runs() -> None:
    # 6 CJK chars -> 5 bigrams each side; hypothesis is a strict subset (precision 1, recall 5/6).
    f1 = b.word_f1("ポイ橋て禁止", "ポイ橋て禁止』")
    assert f1 == pytest.approx(5 / 5.5)


def test_word_f1_keeps_latin_tokens_whole() -> None:
    assert b.word_f1("Easy OCR rocks", "easy ocr rocks") == 1.0


def test_word_f1_bigram_matches_char_f1_for_identical_cjk_lines() -> None:
    assert b.word_f1("清潔できれいな", "清潔できれいな") == 1.0


# -- line detection precision/recall/F1 --------------------------------------------------


def test_line_detection_scores_of_identical_lines_is_perfect() -> None:
    quad = [[0.0, 0.0], [10.0, 0.0], [10.0, 4.0], [0.0, 4.0]]
    f1, precision, recall = b.line_detection_scores([quad], [quad])
    assert (f1, precision, recall) == (1.0, 1.0, 1.0)


def test_line_detection_scores_penalizes_a_fabricated_hypothesis_line() -> None:
    reference_quad = [[0.0, 0.0], [10.0, 0.0], [10.0, 4.0], [0.0, 4.0]]
    fabricated_quad = [[100.0, 100.0], [110.0, 100.0], [110.0, 104.0], [100.0, 104.0]]
    f1, precision, recall = b.line_detection_scores([reference_quad], [reference_quad, fabricated_quad])
    assert recall == 1.0
    assert precision == pytest.approx(0.5)
    assert f1 < 1.0


def test_line_detection_scores_penalizes_an_omitted_reference_line() -> None:
    reference_quad_a = [[0.0, 0.0], [10.0, 0.0], [10.0, 4.0], [0.0, 4.0]]
    reference_quad_b = [[20.0, 0.0], [30.0, 0.0], [30.0, 4.0], [20.0, 4.0]]
    f1, precision, recall = b.line_detection_scores([reference_quad_a, reference_quad_b], [reference_quad_a])
    assert precision == 1.0
    assert recall == pytest.approx(0.5)
    assert f1 < 1.0


def test_line_detection_scores_of_empty_sides() -> None:
    assert b.line_detection_scores([], []) == (1.0, 1.0, 1.0)
    quad = [[0.0, 0.0], [10.0, 0.0], [10.0, 4.0], [0.0, 4.0]]
    assert b.line_detection_scores([quad], []) == (0.0, 0.0, 0.0)


def test_box_iou_of_identical_boxes_is_one() -> None:
    assert b.box_iou((0.0, 0.0, 2.0, 2.0), (0.0, 0.0, 2.0, 2.0)) == 1.0


def test_box_iou_of_disjoint_boxes_is_zero() -> None:
    assert b.box_iou((0.0, 0.0, 1.0, 1.0), (5.0, 5.0, 6.0, 6.0)) == 0.0


def test_character_error_rate_counts_substitutions() -> None:
    assert b.character_error_rate("cat", "car") == 1 / 3


def test_word_error_rate_is_none_for_empty_reference() -> None:
    assert b.word_error_rate("", "anything") is None


def test_normalize_text_strips_markdown_table_line_breaks() -> None:
    assert b.normalize_text("2025/26<br>Actual") == "2025/26 actual"
    assert b.normalize_text("Prior Year<BR/>Actual") == "prior year actual"


def test_normalize_text_does_not_treat_a_literal_placeholder_as_a_line_break() -> None:
    # "<dataset-name>" is real printed text (a code placeholder), not a markdown table line
    # break -- only the exact "<br>"/"<br/>" tag is stripped as a table artifact.
    assert b.normalize_text("lp://<dataset-name>/<model-name>") == "lp://<dataset-name /<model-name"


# -- percentile aggregation ---------------------------------------------------------------


def test_summary_reports_p95_and_p99_at_full_sample_counts() -> None:
    summary = b._summary([float(i) for i in range(120)])
    assert summary is not None
    assert summary["p95"] is not None
    assert summary["p99"] is not None
    assert summary["count"] == 120


def test_summary_suppresses_p95_below_min_samples() -> None:
    from sceptre_rs_tools.stats import P95_MIN_SAMPLES

    summary = b._summary([float(i) for i in range(P95_MIN_SAMPLES - 1)])
    assert summary is not None
    assert summary["p95"] is None


def test_summary_suppresses_p99_but_not_p95_between_the_two_floors() -> None:
    from sceptre_rs_tools.stats import P95_MIN_SAMPLES, P99_MIN_SAMPLES

    values = [float(i) for i in range(P95_MIN_SAMPLES, P99_MIN_SAMPLES - 1)]
    summary = b._summary(values)
    assert summary is not None
    assert summary["p95"] is not None
    assert summary["p99"] is None


def test_summary_of_empty_list_is_none() -> None:
    assert b._summary([]) is None


# -- subprocess stderr hygiene ----------------------------------------------------------


def test_strip_time_stats_removes_darwin_resource_block() -> None:
    stderr = (
        "Error: running OCR over sample.bmp\n"
        "    1: The image format Bmp is not supported\n"
        "        0.05 real         0.03 user         0.01 sys\n"
        "             10485760  maximum resident set size\n"
        "                    5  page reclaims\n"
        "                    0  swaps\n"
    )
    cleaned = b._strip_time_stats(stderr)
    assert "The image format Bmp is not supported" in cleaned
    assert "maximum resident set size" not in cleaned
    assert "page reclaims" not in cleaned
    assert "swaps" not in cleaned


def test_parse_child_cpu_seconds_reads_darwin_user_and_sys() -> None:
    stderr = "        0.05 real         0.30 user         0.12 sys\n"
    assert b._parse_child_cpu_seconds(stderr) == pytest.approx(0.42)


def test_parse_child_cpu_seconds_reads_linux_user_and_sys() -> None:
    stderr = "\tUser time (seconds): 0.30\n\tSystem time (seconds): 0.12\n"
    assert b._parse_child_cpu_seconds(stderr) == pytest.approx(0.42)


def test_time_wrapper_warns_once_when_gnu_time_is_missing(monkeypatch, capsys) -> None:
    """A missing /usr/bin/time silently nulls peak RSS and CPU core-seconds -- say so."""
    monkeypatch.setattr(b.Path, "exists", lambda _self: False)
    b._warned_missing_time_binary.clear()

    assert b._time_wrapper() == []
    first = capsys.readouterr().err
    assert "/usr/bin/time" in first
    assert "peak RSS" in first

    assert b._time_wrapper() == []
    assert capsys.readouterr().err == ""


def test_parse_child_cpu_seconds_is_none_without_a_time_wrapper() -> None:
    assert b._parse_child_cpu_seconds("no timing info here") is None


def test_parse_child_rss_reads_darwin_bytes() -> None:
    assert b._parse_child_rss("   10000000  maximum resident set size") == 10.0


def test_is_unsupported_format_matches_decode_errors() -> None:
    assert b._is_unsupported_format("The image format Bmp is not supported")
    assert b._is_unsupported_format("failed to decode image bytes")
    assert not b._is_unsupported_format("model session creation failed")


# -- batch parsing ----------------------------------------------------------------------


def test_parse_sceptre_batch_flags_unsupported_format() -> None:
    payload = '[{"image": "a.bmp", "error": "The image format Bmp is not supported"}]'
    parsed = b._parse_sceptre_batch_json(payload, [])
    slice_ = parsed["a.bmp"]
    assert slice_.detections is None
    assert slice_.unsupported_format is True


def test_parse_easyocr_runner_json_reads_detections_and_build() -> None:
    payload = (
        '{"reader_build_seconds": 1.5, "images": [{"image": "a.png", "seconds": 0.2, '
        '"detections": [{"text": "HI", "quad": [[0,0],[1,0],[1,1],[0,1]]}]}]}'
    )
    build, versions, per_image = b._parse_easyocr_runner_json(payload)
    assert build == 1.5
    assert versions == {}
    assert per_image["a.png"].seconds == 0.2
    assert per_image["a.png"].text == "HI"


def test_parse_easyocr_runner_json_reads_reference_versions() -> None:
    payload = '{"reader_build_seconds": 1.5, "versions": {"easyocr": "1.7.2", "torch": "2.5.1"}, "images": []}'
    _, versions, _ = b._parse_easyocr_runner_json(payload)
    assert versions == {"easyocr": "1.7.2", "torch": "2.5.1"}


def test_merge_per_image_flattens_batches() -> None:
    first = b.BatchResult("sceptre", ("english",), 1.0, 1, None, None, {"a": b.ImageDetections(detections=[])})
    second = b.BatchResult("sceptre", ("japanese",), 1.0, 1, None, None, {"b": b.ImageDetections(detections=[])})
    merged = b._merge_per_image([first, second])
    assert set(merged) == {"a", "b"}


def test_batch_summary_includes_cpu_core_seconds() -> None:
    result = b.BatchResult("sceptre", ("english",), 1.0, 2, 500.0, None, {}, cpu_core_seconds=0.42)
    summary = b._batch_summary({("english",): result})
    assert summary["english"]["cpu_core_seconds"] == pytest.approx(0.42)


# -- report rendering + regression gate -------------------------------------------------


def _report(
    sceptre_f1: float,
    easyocr_f1: float,
    sceptre_rss: float,
    easyocr_rss: float,
    environment: dict | None = None,
) -> b.RunReport:
    """Build a minimal report with one labeled record and both batch summaries.

    The record carries the same peak-RSS figures as the batch summary, which is how the harness
    fills it in: each image is stamped with the peak RSS of the batch it was measured in.
    """
    record = b.ImageRecord(
        stem="doc",
        group="labeled",
        languages=["english"],
        sceptre_token_f1=sceptre_f1,
        sceptre_cold_seconds=0.4,
        sceptre_rss_mb=sceptre_rss,
        easyocr_rss_mb=easyocr_rss,
    )
    capability = b.ImageRecord(
        stem="probe.avif",
        group="capability",
        languages=["english"],
        unsupported_format=True,
        skipped="sceptre cannot decode this format (capability gap)",
    )
    metadata = {
        "platform": "test",
        "sceptre_binary": "bin",
        "corpus_total": 2,
        "labeled_scored": 1,
        "breadth_scored": 0,
        "repeats": 1,
        "threads": "engine default (all cores)",
        "sceptre_batch": {"english": {"total_seconds": 1.0, "image_count": 2, "rss_mb": sceptre_rss}},
        "easyocr_batch": {"en": {"total_seconds": 5.0, "image_count": 2, "rss_mb": easyocr_rss}},
    }
    if environment is not None:
        metadata["environment"] = environment
    return b.RunReport(
        metadata=metadata,
        records=[record, capability],
        aggregates_labeled={
            "sceptre_token_f1": {"mean": sceptre_f1},
            "easyocr_token_f1": {"mean": easyocr_f1},
        },
        aggregates_all={},
    )


def test_run_report_to_dict_carries_a_schema_version() -> None:
    payload = _report(0.9, 0.8, 500.0, 11000.0).to_dict()
    assert payload["schema_version"] == b.RunReport.SCHEMA_VERSION
    assert isinstance(payload["schema_version"], int)


def test_render_markdown_lists_capability_gaps() -> None:
    markdown = b.render_markdown(_report(0.9, 0.8, 500.0, 11000.0))
    assert "Capability gaps" in markdown
    assert "`probe.avif`" in markdown


def test_render_markdown_shows_baseline_delta() -> None:
    baseline = {"records": [{"stem": "doc", "sceptre_token_f1": 0.7}]}
    markdown = b.render_markdown(_report(0.9, 0.8, 500.0, 11000.0), baseline)
    assert "ΔtokF1 +0.200" in markdown


_HOST = {"os": "Linux", "arch": "x86_64", "runner_label": "runner-medium"}


def _host(os_name: str = "Linux", arch: str = "x86_64", runner_label: str | None = "runner-medium") -> tuple:
    """The normalized host identity a floor is scoped to."""
    return b._host_identity(os_name, arch, runner_label)


def _published(peak_rss_mb: float | None = 500.0, throughput_img_s: float | None = 2.0) -> dict:
    """A committed published artifact, shaped like `publish.published_payload`'s output."""
    warm: dict = {"peak_rss_mb": peak_rss_mb}
    if throughput_img_s is not None:
        warm["throughput_img_s"] = throughput_img_s
    return {
        "schema_version": 2,
        "environment": {"os": _HOST["os"], "arch": _HOST["arch"]},
        "run": {"runner_label": _HOST["runner_label"]},
        "headline": {"sceptre_warm": warm},
    }


def test_check_thresholds_passes_a_run_within_the_published_rss_ceiling() -> None:
    # sceptre peak RSS 500 MB and 2 img/s (2 images / 1.0 s) against a 500 MB, 2 img/s baseline. ~keep
    report = _report(0.9, 0.8, 500.0, 11000.0, environment=_HOST)
    assert b.check_thresholds(report, _published(500.0)) == []


@pytest.mark.parametrize("easyocr_rss", [500.0, 11000.0, 40000.0])
def test_check_thresholds_ignores_an_easyocr_memory_change(easyocr_rss: float) -> None:
    """The gate is self-referential: only sceptre's own peak RSS can trip it.

    An EasyOCR or torch release that halves (or quadruples) the reference engine's footprint
    moves the old cross-engine ratio through any floor without sceptre changing at all.
    """
    report = _report(0.9, 0.8, 500.0, easyocr_rss, environment=_HOST)
    assert b.check_thresholds(report, _published(500.0)) == []


def test_check_rss_ceiling_flags_a_sceptre_memory_regression() -> None:
    breaches = b.check_rss_ceiling(700.0, _host(), _published(500.0))
    assert breaches == [
        "sceptre peak RSS 700.0 MB > ceiling 550.0 MB "
        "(published baseline 500.0 MB x 1.10, runner-medium (Linux/x86_64))"
    ]


def test_check_rss_ceiling_tolerates_growth_inside_the_factor() -> None:
    """Peak RSS wobbles a few percent run to run; the ceiling leaves room for that, and no more."""
    assert b.check_rss_ceiling(500.0 * b.SCEPTRE_RSS_CEILING_FACTOR, _host(), _published(500.0)) == []
    assert b.check_rss_ceiling(500.0 * b.SCEPTRE_RSS_CEILING_FACTOR + 0.1, _host(), _published(500.0)) != []


def test_check_rss_ceiling_flags_an_absent_baseline_rather_than_passing() -> None:
    breaches = b.check_rss_ceiling(500.0, _host(), None)
    assert any("unmeasurable" in breach and str(b.PUBLISHED_RELATIVE_PATH) in breach for breach in breaches)


def test_check_rss_ceiling_flags_a_baseline_that_carries_no_peak_rss() -> None:
    breaches = b.check_rss_ceiling(500.0, _host(), _published(None))
    assert any("headline.sceptre_warm.peak_rss_mb" in breach for breach in breaches)


def test_check_rss_ceiling_refuses_a_baseline_measured_on_another_host() -> None:
    """An absolute floor only means something on the host it was measured on (ADR 0042)."""
    breaches = b.check_rss_ceiling(500.0, _host("Darwin", "arm64", None), _published(500.0))
    assert any("scoped to a host" in breach and "runner-medium" in breach for breach in breaches)


def test_check_rss_ceiling_distinguishes_two_runner_classes_of_the_same_os() -> None:
    assert b.check_rss_ceiling(500.0, _host(runner_label="runner-large"), _published(500.0)) != []


def test_check_rss_ceiling_flags_a_run_that_measured_no_sceptre_rss() -> None:
    breaches = b.check_rss_ceiling(None, _host(), _published(500.0))
    assert any("no sceptre peak RSS" in breach for breach in breaches)


def test_check_warm_throughput_floor_passes_a_run_at_the_published_throughput() -> None:
    assert b.check_warm_throughput_floor(2.0, _host(), _published()) == []


def test_check_warm_throughput_floor_tolerates_degradation_inside_the_factor() -> None:
    """Wall-clock wobbles a few percent run to run; the floor leaves room for that, and no more."""
    assert b.check_warm_throughput_floor(2.0 * b.SCEPTRE_THROUGHPUT_FLOOR_FACTOR, _host(), _published()) == []
    assert b.check_warm_throughput_floor(2.0 * b.SCEPTRE_THROUGHPUT_FLOOR_FACTOR - 0.01, _host(), _published()) != []


def test_check_warm_throughput_floor_flags_a_sceptre_speed_regression() -> None:
    breaches = b.check_warm_throughput_floor(1.5, _host(), _published())
    assert breaches == [
        "sceptre warm throughput 1.500 img/s < floor 1.800 img/s "
        "(published baseline 2.000 img/s x 0.90, runner-medium (Linux/x86_64))"
    ]


def test_check_warm_throughput_floor_flags_an_absent_baseline_rather_than_passing() -> None:
    breaches = b.check_warm_throughput_floor(2.0, _host(), None)
    assert any("unmeasurable" in breach and str(b.PUBLISHED_RELATIVE_PATH) in breach for breach in breaches)


def test_check_warm_throughput_floor_flags_a_baseline_that_carries_no_throughput() -> None:
    breaches = b.check_warm_throughput_floor(2.0, _host(), _published(throughput_img_s=None))
    assert any("headline.sceptre_warm.throughput_img_s" in breach for breach in breaches)


def test_check_warm_throughput_floor_refuses_a_baseline_measured_on_another_host() -> None:
    """sceptre's warm advantage is parallel scaling, so it is a property of the host (ADR 0043)."""
    breaches = b.check_warm_throughput_floor(2.0, _host("Darwin", "arm64", None), _published())
    assert any("scoped to a host" in breach and "runner-medium" in breach for breach in breaches)


def test_check_warm_throughput_floor_distinguishes_two_runner_classes_of_the_same_os() -> None:
    assert b.check_warm_throughput_floor(2.0, _host(runner_label="runner-large"), _published()) != []


def test_check_warm_throughput_floor_flags_a_run_that_measured_no_throughput() -> None:
    breaches = b.check_warm_throughput_floor(None, _host(), _published())
    assert any("no sceptre warm throughput" in breach for breach in breaches)


def test_check_thresholds_flags_a_sceptre_warm_throughput_regression() -> None:
    report = _report(0.9, 0.8, 500.0, 11000.0, environment=_HOST)
    report.metadata["sceptre_batch"]["english"]["total_seconds"] = 2.0
    breaches = b.check_thresholds(report, _published(500.0))
    assert breaches == [
        "sceptre warm throughput 1.000 img/s < floor 1.800 img/s "
        "(published baseline 2.000 img/s x 0.90, runner-medium (Linux/x86_64))"
    ]


@pytest.mark.parametrize("easyocr_seconds", [1.0, 5.0, 50.0])
def test_check_thresholds_ignores_an_easyocr_speed_change(easyocr_seconds: float) -> None:
    """The speed gate is self-referential: only sceptre's own throughput can trip it (ADR 0043).

    A faster EasyOCR release lowers the cross-engine warm speedup through any floor without
    sceptre changing at all, and the old gate reported that as a sceptre regression.
    """
    report = _report(0.9, 0.8, 500.0, 11000.0, environment=_HOST)
    report.metadata["easyocr_batch"]["en"]["total_seconds"] = easyocr_seconds
    assert b.check_thresholds(report, _published(500.0)) == []


def test_load_published_baseline_returns_none_when_absent(tmp_path: Path) -> None:
    assert b.load_published_baseline(tmp_path / "latest.json") is None


def test_load_published_baseline_returns_none_for_unparseable_json(tmp_path: Path) -> None:
    path = tmp_path / "latest.json"
    path.write_text("{not json", encoding="utf-8")
    assert b.load_published_baseline(path) is None


def test_published_host_reads_the_runner_label_from_the_run_block() -> None:
    assert b.published_host(_published()) == ("Linux", "x86_64", "runner-medium")


def test_run_host_is_unknown_without_a_provenance_block() -> None:
    assert b.run_host(_report(0.9, 0.8, 500.0, 11000.0)) == ("unknown", "unknown", "unknown")


# -- reported peak-RSS ratio (a figure, not a gate) --------------------------------------


def test_paired_rss_ratio_is_the_median_of_the_per_image_ratios() -> None:
    assert b.paired_rss_ratio([(100.0, 200.0), (100.0, 300.0), (100.0, 900.0)]) == pytest.approx(3.0)


def test_paired_rss_ratio_is_none_without_a_paired_measurement() -> None:
    assert b.paired_rss_ratio([]) is None
    assert b.paired_rss_ratio([(100.0, None), (None, 200.0)]) is None


def test_paired_rss_ratio_ignores_a_zero_denominator() -> None:
    assert b.paired_rss_ratio([(0.0, 200.0), (100.0, 400.0)]) == pytest.approx(4.0)


def test_paired_rss_ratio_pairs_images_across_engine_specific_batch_keys() -> None:
    """sceptre keys its batches `english`/`english+korean`, EasyOCR keys the same images `en`/`en+ko`.

    Nothing in the report lines those two key spaces up, so a ratio computed by pairing batch
    keys would drop or mismatch groups. Pairing per image is what makes the figure well-defined.
    """
    records = [
        b.ImageRecord(stem="a", group="labeled", languages=["english"], sceptre_rss_mb=100.0, easyocr_rss_mb=400.0),
        b.ImageRecord(
            stem="b",
            group="breadth",
            languages=["english", "korean"],
            sceptre_rss_mb=200.0,
            easyocr_rss_mb=400.0,
        ),
        b.ImageRecord(stem="c", group="labeled", languages=["english"], skipped="no image"),
    ]
    assert b.paired_rss_ratio(b._record_rss_pairs(records)) == pytest.approx(3.0)


def test_paired_rss_ratio_does_not_collapse_the_way_the_max_of_maxes_does() -> None:
    """The runner-medium shape: one heavy shared-CRAFT batch drags max/max to ~1.08x.

    Both engines allocate the same CRAFT canvas for the largest page, so their corpus maxima
    converge and their quotient says nothing about sceptre. The paired median keeps saying what
    the typical image costs.
    """
    pairs = [(1000.0, 1080.0), (100.0, 325.0), (100.0, 266.0), (100.0, 200.0), (100.0, 137.0)]
    max_of_maxes = max(easyocr for _, easyocr in pairs) / max(sceptre for sceptre, _ in pairs)
    assert max_of_maxes == pytest.approx(1.08)
    assert b.paired_rss_ratio(pairs) == pytest.approx(2.0)


def test_speedup_summary_states_a_large_ratio_to_one_decimal() -> None:
    # warm speedup = (5/2)/(1/2) = 5x; per-image rss ratio = 11000/500 = 22x.
    summary = b.speedup_summary(_report(0.9, 0.8, 500.0, 11000.0))
    assert "~5.0x faster warm" in summary
    assert "~22.0x less peak RSS (per-image median)" in summary


def test_speedup_summary_omits_the_rss_claim_when_no_image_was_measured_on_both_engines() -> None:
    report = _report(0.9, 0.8, 500.0, 11000.0)
    report.records[0].easyocr_rss_mb = None
    assert "peak RSS" not in b.speedup_summary(report)


def test_speedup_summary_keeps_a_ratio_just_above_the_claim_threshold() -> None:
    summary = b.speedup_summary(_report(0.9, 0.8, 10000.0, 20000.0))
    assert "~2.0x less peak RSS" in summary


def test_speedup_summary_drops_a_ratio_too_close_to_parity_to_claim() -> None:
    # 10800/10000 = 1.08x, which must not be published as "~1x less peak RSS". ~keep
    summary = b.speedup_summary(_report(0.9, 0.8, 10000.0, 10800.0))
    assert "peak RSS" not in summary
    assert "1x" not in summary


def test_ratio_claim_renders_and_suppresses_around_the_threshold() -> None:
    assert b.ratio_claim(22.0) == "22.0x"
    assert b.ratio_claim(2.04) == "2.0x"
    assert b.ratio_claim(b.MIN_CLAIMABLE_RATIO) == f"{b.MIN_CLAIMABLE_RATIO:.1f}x"
    assert b.ratio_claim(1.08) is None
    assert b.ratio_claim(None) is None


# -- per-image guardrail contract (replaces the corpus-mean quality check) --------------


def _labeled_record(stem: str, token_f1: float) -> b.ImageRecord:
    return b.ImageRecord(stem=stem, group="labeled", languages=["english"], sceptre_token_f1=token_f1)


def test_derive_guardrails_applies_the_threshold_factor_per_image() -> None:
    records = [_labeled_record("a", 0.9), _labeled_record("b", 0.8)]
    guardrails = b.derive_guardrails(records, threshold_factor=0.9)
    assert guardrails == {
        "schema_version": b.GUARDRAILS_SCHEMA_VERSION,
        "threshold_factor": 0.9,
        "contracts": [
            {"stem": "a", "min_token_f1": pytest.approx(0.81)},
            {"stem": "b", "min_token_f1": pytest.approx(0.72)},
        ],
    }


def test_derive_guardrails_skips_unscored_and_unlabeled_records() -> None:
    skipped = b.ImageRecord(stem="skip", group="labeled", languages=["english"], skipped="no image")
    breadth = b.ImageRecord(stem="breadth", group="breadth", languages=["english"], sceptre_token_f1=0.9)
    guardrails = b.derive_guardrails([skipped, breadth])
    assert guardrails["contracts"] == []


def test_load_guardrails_returns_none_when_missing(tmp_path: Path) -> None:
    assert b.load_guardrails(tmp_path / "missing.json") is None


def test_load_guardrails_rejects_an_unknown_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "guardrails.json"
    path.write_text('{"schema_version": 99, "threshold_factor": 0.9, "contracts": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        b.load_guardrails(path)


def test_check_guardrails_passes_when_every_contract_clears_its_floor() -> None:
    guardrails = {"schema_version": 1, "threshold_factor": 0.9, "contracts": [{"stem": "a", "min_token_f1": 0.8}]}
    assert b.check_guardrails([_labeled_record("a", 0.85)], guardrails) == []


def test_check_guardrails_flags_a_score_below_its_floor() -> None:
    guardrails = {"schema_version": 1, "threshold_factor": 0.9, "contracts": [{"stem": "a", "min_token_f1": 0.8}]}
    failures = b.check_guardrails([_labeled_record("a", 0.5)], guardrails)
    assert any("a: sceptre token-F1 0.500 < guardrail floor 0.800" in failure for failure in failures)


def test_check_guardrails_treats_a_missing_record_as_a_failure() -> None:
    guardrails = {"schema_version": 1, "threshold_factor": 0.9, "contracts": [{"stem": "missing", "min_token_f1": 0.8}]}
    failures = b.check_guardrails([], guardrails)
    assert any("missing" in failure and "unmeasurable" in failure for failure in failures)


def test_check_guardrails_treats_a_nan_score_as_a_failure() -> None:
    guardrails = {"schema_version": 1, "threshold_factor": 0.9, "contracts": [{"stem": "a", "min_token_f1": 0.8}]}
    failures = b.check_guardrails([_labeled_record("a", float("nan"))], guardrails)
    assert any("unmeasurable" in failure for failure in failures)


# -- validate-before-write ---------------------------------------------------------------


def test_validate_report_passes_a_well_formed_report() -> None:
    assert b.validate_report(_report(0.9, 0.8, 500.0, 11000.0)) == []


def test_validate_report_flags_a_nan_aggregate_score() -> None:
    report = _report(0.9, 0.8, 500.0, 11000.0)
    report.aggregates_labeled["sceptre_token_f1"]["mean"] = float("nan")
    problems = b.validate_report(report)
    assert any("not finite" in problem for problem in problems)


def test_validate_report_flags_an_out_of_range_aggregate_score() -> None:
    report = _report(0.9, 0.8, 500.0, 11000.0)
    report.aggregates_labeled["sceptre_token_f1"]["mean"] = 1.5
    problems = b.validate_report(report)
    assert any("outside [0, 1]" in problem for problem in problems)


def test_validate_report_flags_an_out_of_range_record_score() -> None:
    report = _report(0.9, 0.8, 500.0, 11000.0)
    report.records[0].agreement_word_f1 = -0.1
    problems = b.validate_report(report)
    assert any("agreement_word_f1" in problem and "outside [0, 1]" in problem for problem in problems)


def test_validate_report_allows_a_sample_count_on_a_unit_interval_metric() -> None:
    """`count` is a sample count, not a score, so the [0, 1] bound must not apply to it."""
    report = _report(0.9, 0.8, 500.0, 11000.0)
    report.aggregates_labeled["sceptre_token_f1"] = b._summary([0.9] * 27)
    assert report.aggregates_labeled["sceptre_token_f1"]["count"] == 27
    assert b.validate_report(report) == []


def test_validate_report_still_flags_a_negative_sample_count() -> None:
    report = _report(0.9, 0.8, 500.0, 11000.0)
    report.aggregates_labeled["sceptre_token_f1"]["count"] = -1
    problems = b.validate_report(report)
    assert any("count" in problem for problem in problems)


def test_validate_report_allows_unbounded_timing_fields() -> None:
    report = _report(0.9, 0.8, 500.0, 11000.0)
    report.aggregates_labeled["easyocr_warm_seconds"] = {"mean": 123.4}
    assert b.validate_report(report) == []


# -- environment provenance -------------------------------------------------------------


def test_environment_metadata_records_host_and_reference_versions(tmp_path: Path) -> None:
    missing_binary = tmp_path / "sceptre"
    environment = b.environment_metadata(
        missing_binary, tmp_path, ["english"], {"easyocr": "1.7.2", "torch": "2.5.1", "python": "3.11.9"}
    )
    assert environment["os"] == platform.system()
    assert environment["arch"] == platform.machine()
    assert environment["cpu_count"] == os.cpu_count()
    assert environment["sceptre_binary"] == str(missing_binary)
    assert environment["reference"] == {
        "easyocr_version": "1.7.2",
        "torch_version": "2.5.1",
        "python": "3.11.9",
    }


@pytest.mark.skipif(os.name != "posix", reason="the stub binary is a POSIX shell script")
def test_environment_metadata_prefers_the_env_subcommand(tmp_path: Path) -> None:
    payload = (
        '{"version": "0.4.0", "backend": "ort", "accelerator": "coreml", '
        '"onnxruntime": {"version": "1.22.0"}, '
        '"models": [{"name": "craft_mlt_25k", "repo": "xberg-io/sceptre-craft_mlt_25k", "sha256": "159f5f"}]}'
    )
    binary = tmp_path / "sceptre"
    binary.write_text(f"#!/bin/sh\ncat <<'JSON'\n{payload}\nJSON\n", encoding="utf-8")
    binary.chmod(0o755)

    environment = b.environment_metadata(binary, tmp_path, ["english"], {})

    assert environment["sceptre_version"] == "0.4.0"
    assert environment["backend"] == "ort"
    assert environment["accelerator"] == "coreml"
    assert environment["onnxruntime"] == {"version": "1.22.0"}
    assert environment["models"][0]["sha256"] == "159f5f"
    assert environment["probe"] == "sceptre env --format json"


def test_environment_metadata_records_the_ci_runner_label_when_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(b.RUNNER_LABEL_ENV, "runner-medium")
    environment = b.environment_metadata(tmp_path / "sceptre", tmp_path, ["english"], {})
    assert environment["runner_label"] == "runner-medium"


def test_environment_metadata_omits_the_runner_label_off_ci(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(b.RUNNER_LABEL_ENV, raising=False)
    environment = b.environment_metadata(tmp_path / "sceptre", tmp_path, ["english"], {})
    assert "runner_label" not in environment


def test_environment_metadata_treats_a_blank_runner_label_as_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(b.RUNNER_LABEL_ENV, "   ")
    environment = b.environment_metadata(tmp_path / "sceptre", tmp_path, ["english"], {})
    assert "runner_label" not in environment


def test_environment_metadata_leaves_unprobeable_fields_none(tmp_path: Path) -> None:
    environment = b.environment_metadata(tmp_path / "sceptre", tmp_path, ["english"], {})
    assert environment["sceptre_version"] is None
    assert environment["backend"] is None
    assert environment["accelerator"] is None
    assert environment["onnxruntime"] is None
    assert environment["models"] == []


# -- corpus provenance (test_documents/corpus.lock.json) ---------------------------------


def _entry(stem: str, image: Path) -> b.CorpusEntry:
    return b.CorpusEntry(stem=stem, image=image, ground_truth=None, languages=("english",), group="labeled")


def test_corpus_provenance_is_unresolved_without_a_lock_file(tmp_path: Path) -> None:
    environment = b.environment_metadata(tmp_path / "sceptre", tmp_path, ["english"], {})
    assert environment["corpus"] == {"resolved": False, "source": None, "images": {}}


def test_corpus_provenance_cites_the_lock_hash_when_present(tmp_path: Path) -> None:
    lock_dir = tmp_path / "test_documents"
    lock_dir.mkdir()
    (lock_dir / "corpus.lock.json").write_text(
        '{"schema": 1, "objects": {"images/sample.png": {"sha256": "deadbeef", "size": 10}}}',
        encoding="utf-8",
    )
    image = tmp_path / "sample.png"
    image.write_bytes(b"")

    environment = b.environment_metadata(tmp_path / "sceptre", tmp_path, ["english"], {}, [_entry("sample", image)])

    assert environment["corpus"] == {
        "resolved": True,
        "source": "test_documents/corpus.lock.json",
        "lock_schema": 1,
        "images": {"sample": "deadbeef"},
    }


def test_corpus_provenance_omits_entries_missing_from_the_lock(tmp_path: Path) -> None:
    lock_dir = tmp_path / "test_documents"
    lock_dir.mkdir()
    (lock_dir / "corpus.lock.json").write_text('{"schema": 1, "objects": {}}', encoding="utf-8")
    image = tmp_path / "unmatched.png"
    image.write_bytes(b"")

    environment = b.environment_metadata(tmp_path / "sceptre", tmp_path, ["english"], {}, [_entry("unmatched", image)])

    assert environment["corpus"] == {
        "resolved": True,
        "source": "test_documents/corpus.lock.json",
        "lock_schema": 1,
        "images": {},
    }


def test_render_markdown_reports_the_runtime_line() -> None:
    report = _report(0.9, 0.8, 500.0, 11000.0)
    report.metadata["environment"] = {
        "sceptre_version": "0.3.0",
        "backend": "ort",
        "accelerator": "cpu",
        "onnxruntime": {"version": "1.22.0"},
        "os": "Darwin",
        "arch": "arm64",
        "cpu_count": 10,
        "reference": {"easyocr_version": "1.7.2", "torch_version": "2.5.1"},
    }
    markdown = b.render_markdown(report)
    assert "- Runtime: sceptre 0.3.0 (ort/cpu, ONNX Runtime 1.22.0) on Darwin/arm64, 10 cores" in markdown
    assert "reference: EasyOCR 1.7.2, torch 2.5.1" in markdown


def test_render_markdown_omits_the_runtime_line_without_an_environment() -> None:
    markdown = b.render_markdown(_report(0.9, 0.8, 500.0, 11000.0))
    assert "- Runtime:" not in markdown


def test_easyocr_runner_command_names_the_output_file(tmp_path: Path) -> None:
    out = tmp_path / "result.json"
    command = b._easyocr_runner_command([Path("a.png")], ["en"], None, out)
    assert "--output" in command
    assert command[command.index("--output") + 1] == str(out)


def test_easyocr_runner_writes_json_to_output_despite_stdout_noise(tmp_path: Path) -> None:
    """The runner's payload must survive a dependency writing to stdout.

    torch's DataLoader emits a `pin_memory` UserWarning on any accelerator-less host, and
    easyocr prints model-download progress. Both land on stdout, so a runner that returned
    its payload there produced unparseable output on exactly the CPU-only Linux runners the
    benchmark is measured on. This stubs `easyocr` with a module that pollutes stdout the
    same way and asserts the payload still arrives intact.
    """
    stub = tmp_path / "stub"
    stub.mkdir()
    (stub / "easyocr.py").write_text(
        "import sys\n"
        "__version__ = '1.7.2'\n"
        "sys.stdout.write('pinned memory will not be used.\\n')\n"
        "class Reader:\n"
        "    def __init__(self, languages, gpu=False):\n"
        "        sys.stdout.write('Progress: 100%\\n')\n"
        "    def readtext(self, image):\n"
        "        return [([[0, 0], [1, 0], [1, 1], [0, 1]], 'HI', 0.9)]\n",
        encoding="utf-8",
    )
    out = tmp_path / "result.json"
    package_root = Path(b.__file__).resolve().parent.parent
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(stub), str(package_root)]))
    completed = subprocess.run(
        [sys.executable, "-m", "sceptre_rs_tools._easyocr_runner", "--output", str(out), "a.png"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "pinned memory" in completed.stdout
    build, versions, per_image = b._parse_easyocr_runner_json(out.read_text(encoding="utf-8"))
    assert build is not None
    assert versions["easyocr"] == "1.7.2"
    assert per_image["a.png"].text == "HI"
