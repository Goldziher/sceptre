"""Assert the Python metrics against the shared contract vectors.

``crates/sceptre/tests/data/metrics_vectors.json`` is the contract between the two
independent implementations of the parity metrics — this harness's
(``python/sceptre_rs_tools/benchmark.py``) and the Rust test harness's
(``crates/sceptre/tests/helpers/mod.rs``). Both sides assert against it, so a change to
one implementation that is not mirrored in the other fails here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sceptre_rs_tools import benchmark as b

TOLERANCE = 1e-6


def _vectors() -> dict[str, object]:
    path = Path(b.repo_root()) / "crates" / "sceptre" / "tests" / "data" / "metrics_vectors.json"
    return json.loads(path.read_text(encoding="utf-8"))


VECTORS = _vectors()


def test_vectors_use_the_known_schema_version() -> None:
    assert VECTORS["schema_version"] == 1


@pytest.mark.parametrize("case", VECTORS["text"], ids=lambda case: case["note"])
def test_text_metrics_match_the_contract(case: dict[str, object]) -> None:
    hypothesis, reference = case["hypothesis"], case["reference"]
    assert b.word_f1(hypothesis, reference) == pytest.approx(case["word_f1"], abs=TOLERANCE)
    assert b.char_f1(hypothesis, reference) == pytest.approx(case["char_f1"], abs=TOLERANCE)


@pytest.mark.parametrize("case", VECTORS["boxes"], ids=lambda case: case["note"])
def test_box_iou_matches_the_contract(case: dict[str, object]) -> None:
    a, box_b = tuple(case["a"]), tuple(case["b"])
    assert b.box_iou(a, box_b) == pytest.approx(case["iou"], abs=TOLERANCE)
