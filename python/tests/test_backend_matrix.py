"""Unit tests for `sceptre_rs_tools.backend_matrix` (leg-report aggregation, no models needed)."""

from __future__ import annotations

import json
from pathlib import Path

from sceptre_rs_tools.backend_matrix import SCHEMA_VERSION, combine, load_leg_reports, main, render_summary

LEG_REPORT = {
    "leg": "ort-cpu",
    "canvas_size": 1024,
    "backend": "ort",
    "accelerator_requested": "cpu",
    "accelerator_registered": "cpu",
    "model_load_ms": 250.0,
    "images": [
        {
            "image": "english.png",
            "language": "English",
            "repeats": 5,
            "median": {"setup_ms": 10.0, "detect_ms": 20.0, "recognize_ms": 30.0, "total_ms": 60.0},
        }
    ],
}


def write_leg_report(directory: Path, name: str, report: dict) -> None:
    (directory / f"{name}.json").write_text(json.dumps(report), encoding="utf-8")


def test_load_leg_reports_returns_empty_list_for_missing_directory(tmp_path: Path) -> None:
    assert load_leg_reports(tmp_path / "does-not-exist") == []


def test_load_leg_reports_reads_every_json_file_sorted_by_leg(tmp_path: Path) -> None:
    write_leg_report(tmp_path, "z-leg", {**LEG_REPORT, "leg": "z-leg"})
    write_leg_report(tmp_path, "a-leg", {**LEG_REPORT, "leg": "a-leg"})

    reports = load_leg_reports(tmp_path)

    assert [report["leg"] for report in reports] == ["a-leg", "z-leg"]


def test_combine_wraps_reports_with_a_schema_version() -> None:
    combined = combine([LEG_REPORT])

    assert combined["schema_version"] == SCHEMA_VERSION
    assert combined["legs"] == [LEG_REPORT]


def test_render_summary_reports_no_leg_reports_found() -> None:
    summary = render_summary([])

    assert "No leg reports were found" in summary


def test_render_summary_includes_leg_and_image_figures() -> None:
    summary = render_summary([LEG_REPORT])

    assert "ort-cpu" in summary
    assert "english.png" in summary
    assert "60.0" in summary  # total_ms
    assert "250.0" in summary  # model_load_ms


def test_main_writes_combined_artifact_and_prints_summary(tmp_path: Path, capsys) -> None:
    input_dir = tmp_path / "backends"
    input_dir.mkdir()
    write_leg_report(input_dir, "ort-cpu", LEG_REPORT)
    out_path = tmp_path / "combined.json"

    exit_code = main(["--input-dir", str(input_dir), "--out", str(out_path)])

    assert exit_code == 0
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["schema_version"] == SCHEMA_VERSION
    assert len(written["legs"]) == 1
    assert "ort-cpu" in capsys.readouterr().out


def test_main_appends_summary_to_the_given_path(tmp_path: Path) -> None:
    input_dir = tmp_path / "backends"
    input_dir.mkdir()
    write_leg_report(input_dir, "ort-cpu", LEG_REPORT)
    out_path = tmp_path / "combined.json"
    summary_path = tmp_path / "summary.md"
    summary_path.write_text("existing content\n", encoding="utf-8")

    main(["--input-dir", str(input_dir), "--out", str(out_path), "--summary", str(summary_path)])

    content = summary_path.read_text(encoding="utf-8")
    assert content.startswith("existing content\n")
    assert "ort-cpu" in content
