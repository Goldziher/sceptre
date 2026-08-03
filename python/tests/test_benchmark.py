"""Unit tests for the pure logic of the sceptre-vs-EasyOCR benchmark harness.

These cover the parts that do not need torch/easyocr or the release binary: metric parity,
subprocess stderr hygiene, capability-gap detection, report rendering (capability section
and baseline deltas), and the regression gate.
"""

from __future__ import annotations

from sceptre_rs_tools import benchmark as b

# -- text/box metrics (mirror crates/sceptre/tests/helpers/mod.rs) ----------------------


def test_word_f1_is_one_for_identical_strings() -> None:
    assert b.word_f1("hello world", "hello world") == 1.0


def test_word_f1_is_zero_for_disjoint_strings() -> None:
    assert b.word_f1("alpha", "beta") == 0.0


def test_char_f1_ignores_whitespace() -> None:
    assert b.char_f1("a b c", "abc") == 1.0


def test_box_iou_of_identical_boxes_is_one() -> None:
    assert b.box_iou((0.0, 0.0, 2.0, 2.0), (0.0, 0.0, 2.0, 2.0)) == 1.0


def test_box_iou_of_disjoint_boxes_is_zero() -> None:
    assert b.box_iou((0.0, 0.0, 1.0, 1.0), (5.0, 5.0, 6.0, 6.0)) == 0.0


def test_character_error_rate_counts_substitutions() -> None:
    assert b.character_error_rate("cat", "car") == 1 / 3


def test_word_error_rate_is_none_for_empty_reference() -> None:
    assert b.word_error_rate("", "anything") is None


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
    build, per_image = b._parse_easyocr_runner_json(payload)
    assert build == 1.5
    assert per_image["a.png"].seconds == 0.2
    assert per_image["a.png"].text == "HI"


def test_merge_per_image_flattens_batches() -> None:
    first = b.BatchResult("sceptre", ("english",), 1.0, 1, None, None, {"a": b.ImageDetections(detections=[])})
    second = b.BatchResult("sceptre", ("japanese",), 1.0, 1, None, None, {"b": b.ImageDetections(detections=[])})
    merged = b._merge_per_image([first, second])
    assert set(merged) == {"a", "b"}


# -- report rendering + regression gate -------------------------------------------------


def _report(sceptre_f1: float, easyocr_f1: float, sceptre_rss: float, easyocr_rss: float) -> b.RunReport:
    """Build a minimal report with one labeled record and both batch summaries."""
    record = b.ImageRecord(
        stem="doc", group="labeled", languages=["english"], sceptre_token_f1=sceptre_f1, sceptre_cold_seconds=0.4
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
    return b.RunReport(
        metadata=metadata,
        records=[record, capability],
        aggregates_labeled={
            "sceptre_token_f1": {"mean": sceptre_f1},
            "easyocr_token_f1": {"mean": easyocr_f1},
        },
        aggregates_all={},
    )


def test_render_markdown_lists_capability_gaps() -> None:
    markdown = b.render_markdown(_report(0.9, 0.8, 500.0, 11000.0))
    assert "Capability gaps" in markdown
    assert "`probe.avif`" in markdown


def test_render_markdown_shows_baseline_delta() -> None:
    baseline = {"records": [{"stem": "doc", "sceptre_token_f1": 0.7}]}
    markdown = b.render_markdown(_report(0.9, 0.8, 500.0, 11000.0), baseline)
    assert "ΔtokF1 +0.200" in markdown


def test_check_thresholds_passes_a_healthy_run() -> None:
    # warm speedup = (5/2)/(1/2) = 5x; rss ratio = 22x; f1 0.9 >= 0.8 - 0.05.
    assert b.check_thresholds(_report(0.9, 0.8, 500.0, 11000.0)) == []


def test_check_thresholds_flags_rss_regression() -> None:
    # rss ratio 1.1x is below the 5x floor.
    breaches = b.check_thresholds(_report(0.9, 0.8, 10000.0, 11000.0))
    assert any("peak-RSS ratio" in breach for breach in breaches)


def test_check_thresholds_flags_quality_regression() -> None:
    # sceptre token-F1 0.5 is far below EasyOCR 0.8 - 0.05.
    breaches = b.check_thresholds(_report(0.5, 0.8, 500.0, 11000.0))
    assert any("token-F1" in breach for breach in breaches)
