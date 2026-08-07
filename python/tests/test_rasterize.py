"""Unit tests for the PDF-rasterization script's pure logic: deterministic page naming,
per-run provenance shape, and the whole-document ground-truth concatenation resolution.

None of these need `pypdfium2` (the opt-in `corpus` dependency group) or the `test_documents`
corpus to be present.
"""

from __future__ import annotations

import pytest
from sceptre_rs_tools import rasterize


def test_page_filename_zero_pads_to_the_documents_own_page_count() -> None:
    assert rasterize.page_filename("report", 3, 12) == "report_p03.png"
    assert rasterize.page_filename("report", 3, 200) == "report_p003.png"


def test_page_filename_pads_to_at_least_two_digits_for_short_documents() -> None:
    assert rasterize.page_filename("memo", 1, 1) == "memo_p01.png"


def test_concatenate_page_texts_joins_non_empty_pages_in_order() -> None:
    joined = rasterize.concatenate_page_texts(["first page", "  ", "second page"])
    assert joined == "first page\n\nsecond page"


def test_concatenate_page_texts_of_no_pages_is_empty() -> None:
    assert rasterize.concatenate_page_texts([]) == ""


def test_provenance_records_the_pinned_dpi_and_a_timestamp() -> None:
    record = rasterize.provenance(300)
    assert record["dpi"] == 300
    assert isinstance(record["generated_at"], str) and record["generated_at"]


def test_rasterize_one_raises_for_a_stem_not_in_the_corpus() -> None:
    with pytest.raises(FileNotFoundError):
        rasterize.rasterize_one(
            "definitely-not-a-real-corpus-stem", rasterize.default_output_dir(), rasterize.DEFAULT_DPI
        )


def test_main_requires_at_least_one_pdf_stem() -> None:
    with pytest.raises(SystemExit):
        rasterize.main([])
