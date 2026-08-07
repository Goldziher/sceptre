"""Unit tests for publishing measured benchmark numbers into the committed docs.

All of this is pure logic over JSON and Markdown — no torch, no easyocr, no release
binary — which is what lets the drift check run on every pull request.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sceptre_rs_tools import publish as p


def _report(**overrides: object) -> dict:
    """A minimal but complete comparison report, shaped like benchmark.py's output."""
    report = {
        "metadata": {
            "platform": "macOS-26.0-arm64",
            "group": "all",
            "repeats": 2,
            "threads": "native",
            "corpus_total": 43,
            "labeled_scored": 27,
            "breadth_scored": 13,
            "capability_gaps": 3,
            # Mirrors benchmark.py's environment_metadata exactly: onnxruntime is a
            # nested object, and the reference versions carry a _version suffix.
            "environment": {
                "sceptre_version": "0.4.0",
                "backend": "ort",
                "accelerator": "cpu",
                "onnxruntime": {"provisioning": "bundled", "version": "1.28.0"},
                "sceptre_binary": "/Users/someone/workspace/sceptre/target/release/sceptre",
                "os": "Darwin",
                "arch": "arm64",
                "cpu_count": 14,
                "reference": {
                    "easyocr_version": "1.7.2",
                    "torch_version": "2.13.0",
                    "python": "3.14.1",
                },
            },
            "sceptre_batch": {
                "english": {"total_seconds": 50.0, "image_count": 20, "rss_mb": 6624.0},
            },
            "easyocr_batch": {
                "en": {"total_seconds": 200.0, "image_count": 20, "rss_mb": 22626.0},
            },
        },
        "aggregates": {
            "labeled": {
                "easyocr_cer": {"mean": 0.554, "median": 0.5, "p95": 0.9, "count": 27},
                "easyocr_token_f1": {"mean": 0.348, "median": 0.3, "p95": 0.9, "count": 27},
                "sceptre_cer": {"mean": 0.568, "median": 0.5, "p95": 0.9, "count": 27},
                "sceptre_token_f1": {"mean": 0.356, "median": 0.3, "p95": 0.9, "count": 27},
            },
            "all": {},
        },
        "records": [
            {"stem": "a", "skipped": None, "sceptre_cold_seconds": 2.0},
            {"stem": "b", "skipped": None, "sceptre_cold_seconds": 2.0},
            {"stem": "c", "skipped": "unsupported", "sceptre_cold_seconds": None},
        ],
    }
    report["metadata"].update(overrides)
    return report


# -- payload -----------------------------------------------------------------------------


def test_should_reject_a_report_with_no_provenance_block() -> None:
    report = _report()
    del report["metadata"]["environment"]
    with pytest.raises(p.MissingProvenanceError):
        p.published_payload(report)


def test_should_reject_a_report_whose_provenance_block_is_empty() -> None:
    with pytest.raises(p.MissingProvenanceError):
        p.published_payload(_report(environment={}))


def test_should_carry_the_provenance_block_into_the_published_payload() -> None:
    payload = p.published_payload(_report())
    assert payload["environment"]["sceptre_version"] == "0.4.0"
    assert payload["environment"]["reference"]["easyocr_version"] == "1.7.2"


def test_should_derive_measured_count_from_the_scored_subsets() -> None:
    payload = p.published_payload(_report())
    assert payload["corpus"] == {
        "total": 43,
        "measured": 40,
        "labeled_scored": 27,
        "breadth_scored": 13,
        "capability_gaps": 3,
    }


def test_should_compute_throughput_from_batch_totals() -> None:
    payload = p.published_payload(_report())
    assert payload["headline"]["sceptre_warm"]["throughput_img_s"] == pytest.approx(20 / 50.0)
    assert payload["headline"]["easyocr_warm"]["throughput_img_s"] == pytest.approx(20 / 200.0)


def test_should_compute_warm_speedup_per_image() -> None:
    payload = p.published_payload(_report())
    assert payload["headline"]["warm_speedup"] == pytest.approx(4.0)


def test_should_exclude_skipped_records_from_the_cold_figure() -> None:
    payload = p.published_payload(_report())
    assert payload["headline"]["sceptre_cold"]["throughput_img_s"] == pytest.approx(2 / 4.0)


def test_should_compute_the_peak_rss_ratio() -> None:
    payload = p.published_payload(_report())
    assert payload["headline"]["rss_ratio"] == pytest.approx(22626.0 / 6624.0)


def test_should_pin_the_schema_version() -> None:
    assert p.published_payload(_report())["schema_version"] == p.SCHEMA_VERSION


# -- rendering ---------------------------------------------------------------------------


def test_should_render_peak_rss_in_gigabytes_for_the_readme() -> None:
    table = p.render_headline_table(p.published_payload(_report()), unit="GB")
    assert "22.1 GB" in table
    assert "Peak RSS |" in table


def test_should_render_peak_rss_in_megabytes_for_the_docs_site() -> None:
    table = p.render_headline_table(p.published_payload(_report()), unit="MB")
    assert "22,626.0" in table
    assert "Peak RSS (MB)" in table


def test_should_name_both_runtimes_in_the_provenance_line() -> None:
    table = p.render_headline_table(p.published_payload(_report()), unit="GB")
    assert "sceptre 0.4.0" in table
    assert "EasyOCR 1.7.2" in table
    assert "torch 2.13.0" in table
    assert "40 of 43" in table


def test_should_unwrap_the_nested_onnxruntime_version_rather_than_the_object() -> None:
    table = p.render_headline_table(p.published_payload(_report()), unit="GB")
    assert "ONNX Runtime 1.28.0" in table
    assert "provisioning" not in table


def test_should_strip_the_local_binary_path_from_the_published_artifact() -> None:
    payload = p.published_payload(_report())
    assert "sceptre_binary" not in payload["environment"]
    assert "/Users/someone" not in json.dumps(payload)


def test_should_keep_the_model_pins_and_runtime_fields_it_does_publish() -> None:
    payload = p.published_payload(_report())
    assert payload["environment"]["onnxruntime"]["version"] == "1.28.0"
    assert payload["environment"]["reference"]["easyocr_version"] == "1.7.2"


def test_should_render_an_em_dash_for_an_unmeasured_figure() -> None:
    report = _report()
    report["aggregates"]["labeled"].pop("sceptre_cer")
    table = p.render_headline_table(p.published_payload(report), unit="GB")
    assert "—" in table


# -- injection ---------------------------------------------------------------------------


def _region(tmp_path: Path) -> p.MarkedRegion:
    return p.MarkedRegion(tmp_path / "doc.md", "benchmark-headline", "<!--", "-->")


def test_should_replace_only_the_content_between_the_markers(tmp_path: Path) -> None:
    region = _region(tmp_path)
    text = f"before\n\n{region.start()}\n\nstale\n\n{region.end()}\n\nafter\n"
    updated = p.inject(text, region, "fresh")
    assert "stale" not in updated
    assert "fresh" in updated
    assert updated.startswith("before")
    assert updated.endswith("after\n")
    assert region.start() in updated
    assert region.end() in updated


def test_should_be_idempotent_when_injecting_the_same_body_twice(tmp_path: Path) -> None:
    region = _region(tmp_path)
    text = f"{region.start()}\n\nold\n\n{region.end()}\n"
    once = p.inject(text, region, "body")
    assert p.inject(once, region, "body") == once


def test_should_fail_when_the_opening_marker_is_absent(tmp_path: Path) -> None:
    region = _region(tmp_path)
    with pytest.raises(ValueError, match="missing marker"):
        p.inject(f"no markers here\n{region.end()}\n", region, "body")


def test_should_fail_when_the_closing_marker_is_absent(tmp_path: Path) -> None:
    region = _region(tmp_path)
    with pytest.raises(ValueError, match="missing marker"):
        p.inject(f"{region.start()}\nunterminated\n", region, "body")


def test_should_use_jsx_comments_for_mdx_and_html_comments_for_markdown(tmp_path: Path) -> None:
    regions = p.marked_regions(tmp_path)
    by_suffix = {region.path.suffix: region for region in regions}
    assert by_suffix[".md"].start().startswith("<!--")
    assert by_suffix[".mdx"].start().startswith("{/*")


# -- publish / check ---------------------------------------------------------------------


def _repo(tmp_path: Path) -> Path:
    """A miniature repository with both marked documents and a comparison report."""
    (tmp_path / "docs-site/src/content/docs/reference").mkdir(parents=True)
    (tmp_path / "benchmark-results").mkdir()
    for region in p.marked_regions(tmp_path):
        region.path.write_text(f"intro\n\n{region.start()}\n\n{region.end()}\n\noutro\n", encoding="utf-8")
    (tmp_path / "benchmark-results/comparison.json").write_text(json.dumps(_report()), encoding="utf-8")
    return tmp_path


def test_should_write_the_artifact_and_both_tables(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    assert p.publish(root, root / "benchmark-results/comparison.json", check=False) == 0
    artifact = json.loads((root / p.PUBLISHED_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert artifact["schema_version"] == p.SCHEMA_VERSION
    for region in p.marked_regions(root):
        assert "EasyOCR (warm/batch)" in region.path.read_text(encoding="utf-8")


def test_should_pass_the_drift_check_immediately_after_publishing(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    p.publish(root, root / "benchmark-results/comparison.json", check=False)
    assert p.publish(root, root / "benchmark-results/comparison.json", check=True) == 0


def test_should_fail_the_drift_check_when_a_table_was_edited_by_hand(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    p.publish(root, root / "benchmark-results/comparison.json", check=False)
    readme = root / "README.md"
    tampered = readme.read_text(encoding="utf-8").replace("EasyOCR (warm/batch)", "EasyOCR (hand-edited)")
    assert "hand-edited" in tampered, "the fixture must actually change the generated table"
    readme.write_text(tampered, encoding="utf-8")
    assert p.publish(root, root / "benchmark-results/comparison.json", check=True) == 1


def test_should_fail_the_drift_check_when_the_artifact_is_missing(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    assert p.publish(root, root / "benchmark-results/comparison.json", check=True) == 1


def test_should_check_without_any_measurement_report_present(tmp_path: Path) -> None:
    """The gate runs on a fresh clone: comparison.json is gitignored and never exists in CI."""
    root = _repo(tmp_path)
    p.publish(root, root / "benchmark-results/comparison.json", check=False)
    (root / "benchmark-results/comparison.json").unlink()
    assert p.check_docs(root) == 0


def test_should_still_detect_drift_without_a_measurement_report(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    p.publish(root, root / "benchmark-results/comparison.json", check=False)
    (root / "benchmark-results/comparison.json").unlink()
    readme = root / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8").replace("EasyOCR (warm/batch)", "EasyOCR (hand-edited)"),
        encoding="utf-8",
    )
    assert p.check_docs(root) == 1


def test_should_reject_an_artifact_written_by_a_different_schema_version(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    p.publish(root, root / "benchmark-results/comparison.json", check=False)
    artifact = root / p.PUBLISHED_RELATIVE_PATH
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["schema_version"] = p.SCHEMA_VERSION + 1
    artifact.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert p.check_docs(root) == 1


def test_should_not_write_anything_in_check_mode(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    before = (root / "README.md").read_text(encoding="utf-8")
    p.publish(root, root / "benchmark-results/comparison.json", check=True)
    assert (root / "README.md").read_text(encoding="utf-8") == before
    assert not (root / p.PUBLISHED_RELATIVE_PATH).exists()


def test_should_report_a_missing_report_rather_than_crashing(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    assert p.publish(root, root / "benchmark-results/absent.json", check=False) == 1


def test_should_refuse_to_publish_a_report_without_provenance(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    report = _report()
    del report["metadata"]["environment"]
    source = root / "benchmark-results/comparison.json"
    source.write_text(json.dumps(report), encoding="utf-8")
    assert p.publish(root, source, check=False) == 1
    assert not (root / p.PUBLISHED_RELATIVE_PATH).exists()
