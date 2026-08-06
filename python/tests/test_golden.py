"""Unit tests for the EasyOCR-reference golden generator's merge logic.

These cover the half of the dual-golden contract this tool owns: it writes the ``easyocr``
side and its provenance, and must never drop the ``sceptre`` side written by
``cargo run -p sceptre-tools -- snapshot``.
"""

from __future__ import annotations

import json

from sceptre_rs_tools import golden


def _sceptre_side() -> dict[str, object]:
    return {"lines": [{"text": "SNAPSHOT", "quad": [[0, 0], [1, 0], [1, 1], [0, 1]]}]}


def test_merge_preserves_the_sceptre_side_and_its_provenance() -> None:
    existing = {
        "placeholder": False,
        "metadata": {"sceptre": {"sceptre_version": "0.3.0", "git_commit": "abc123"}},
        "easyocr": {"lines": []},
        "sceptre": _sceptre_side(),
    }

    merged = golden.merge_easyocr_side(existing, [{"text": "REFERENCE", "quad": []}], ["en"])

    assert merged["sceptre"] == _sceptre_side()
    assert merged["metadata"]["sceptre"] == {"sceptre_version": "0.3.0", "git_commit": "abc123"}
    assert merged["easyocr"]["lines"][0]["text"] == "REFERENCE"
    assert merged["placeholder"] is False


def test_merge_writes_easyocr_provenance() -> None:
    merged = golden.merge_easyocr_side({"sceptre": {"lines": []}}, [], ["ja", "en"])

    provenance = merged["metadata"]["easyocr"]
    assert provenance["languages"] == ["ja", "en"]
    assert provenance["gpu"] is False
    assert set(provenance) == {"easyocr_version", "torch_version", "python", "gpu", "languages", "generated_at"}
    assert provenance["generated_at"].endswith("Z")


def test_merge_result_is_json_serializable() -> None:
    merged = golden.merge_easyocr_side({}, [{"text": "hi", "quad": [[0.0, 0.0]]}], ["en"])
    assert json.loads(json.dumps(merged))["metadata"]["easyocr"]["languages"] == ["en"]


def test_load_existing_returns_an_empty_dual_golden_for_a_missing_file(tmp_path) -> None:
    assert golden.load_existing(tmp_path / "absent.json") == {"metadata": {}, "sceptre": {"lines": []}}


def test_load_existing_returns_an_empty_dual_golden_for_invalid_json(tmp_path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert golden.load_existing(broken) == {"metadata": {}, "sceptre": {"lines": []}}
