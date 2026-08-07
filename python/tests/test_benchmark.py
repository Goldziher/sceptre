"""Unit tests for the pure logic of the sceptre-vs-EasyOCR benchmark harness.

These cover the parts that do not need torch/easyocr or the release binary: metric parity,
subprocess stderr hygiene, capability-gap detection, report rendering (capability section
and baseline deltas), and the regression gate.
"""

from __future__ import annotations

import os
import platform
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


def test_box_iou_of_identical_boxes_is_one() -> None:
    assert b.box_iou((0.0, 0.0, 2.0, 2.0), (0.0, 0.0, 2.0, 2.0)) == 1.0


def test_box_iou_of_disjoint_boxes_is_zero() -> None:
    assert b.box_iou((0.0, 0.0, 1.0, 1.0), (5.0, 5.0, 6.0, 6.0)) == 0.0


def test_character_error_rate_counts_substitutions() -> None:
    assert b.character_error_rate("cat", "car") == 1 / 3


def test_word_error_rate_is_none_for_empty_reference() -> None:
    assert b.word_error_rate("", "anything") is None


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


def test_check_thresholds_passes_a_healthy_run() -> None:
    # warm speedup = (5/2)/(1/2) = 5x; rss ratio = 22x.
    assert b.check_thresholds(_report(0.9, 0.8, 500.0, 11000.0)) == []


def test_check_thresholds_flags_rss_regression() -> None:
    # rss ratio 1.1x is below the 5x floor.
    breaches = b.check_thresholds(_report(0.9, 0.8, 10000.0, 11000.0))
    assert any("peak-RSS ratio" in breach for breach in breaches)


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
        '"models": [{"name": "craft_mlt_25k", "repo": "sceptre-ocr/craft_mlt_25k", "sha256": "159f5f"}]}'
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
