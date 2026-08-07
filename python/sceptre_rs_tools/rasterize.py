"""Rasterize `test_documents` corpus PDFs into deterministic per-page PNGs.

This is an offline dev script, not a runtime path: sceptre keeps decoding images only, so
nothing here adds a Rust decode dependency or changes the OCR pipeline. It turns
``test_documents/pdf/<stem>.pdf`` into rasterized pages a `Reader` can already read, using
``pypdfium2`` (the opt-in ``corpus`` dependency group).

Ground truth for a PDF (``test_documents/ground_truth/pdf/<stem>.md``) is a whole-document
transcript, but sceptre's recognizer runs per image. The chosen resolution — see
:func:`concatenate_page_texts` — is to score the concatenation of a PDF's rasterized pages, in
page order, against that whole-document transcript, rather than splitting the ground truth
per page.

Rendering is deterministic: the DPI is pinned (:data:`DEFAULT_DPI`) and every run's manifest
records the exact pdfium/pypdfium2 build that produced it, so re-running on the same pdfium
build reproduces byte-identical PNGs.

    uv run python -m sceptre_rs_tools.rasterize --pdf 10075815
    uv run python -m sceptre_rs_tools.rasterize --pdf 10075815 --pdf 103528129 --dpi 300

This script does not publish its output to the ``test_documents`` bucket — that is a
deliberate, separate step against that repository's own ``scripts/publish_corpus.py``, run by
a human once the rendered pages are reviewed.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

try:
    import pypdfium2 as pdfium
    from pypdfium2 import version as pdfium_version
except ImportError:  # pragma: no cover - exercised by should_raise_when_the_corpus_group_is_missing
    pdfium = None
    pdfium_version = None

# Deterministic rendering DPI. Pinned so re-running the script reproduces the same pixel
# dimensions for a given PDF; the exact figure is also recorded in each run's provenance. ~keep
DEFAULT_DPI = 200
POINTS_PER_INCH = 72.0
DEFAULT_OUTPUT_DIRNAME = "rasterized-pages"


def repo_root() -> Path:
    """Repository root, derived from this file's location."""
    return Path(__file__).resolve().parents[2]


def test_documents_dir() -> Path:
    """Root of the `test_documents` corpus: `TEST_DOCUMENTS_DIR` when set, otherwise the
    submodule checked out at the repository root.
    """
    override = os.environ.get("TEST_DOCUMENTS_DIR")
    return Path(override) if override else repo_root() / "test_documents"


def pdf_path(stem: str) -> Path:
    """Absolute path to a corpus PDF by stem, under `test_documents/pdf/`."""
    return test_documents_dir() / "pdf" / f"{stem}.pdf"


def ground_truth_path(stem: str) -> Path:
    """Absolute path to a PDF's whole-document ground-truth transcript."""
    return test_documents_dir() / "ground_truth" / "pdf" / f"{stem}.md"


def default_output_dir() -> Path:
    """Default local output directory for rendered pages (gitignored; not the corpus)."""
    return repo_root() / DEFAULT_OUTPUT_DIRNAME


@dataclass(frozen=True)
class RasterizedPage:
    """One rendered page: its 1-based page number and the PNG path it was written to."""

    page_number: int
    path: Path


def page_filename(stem: str, page_number: int, page_count: int) -> str:
    """`<stem>_p<NN>.png`, zero-padded to the document's own page count (at least two digits)
    so filenames for a single document always sort in page order.
    """
    width = max(2, len(str(page_count)))
    return f"{stem}_p{page_number:0{width}d}.png"


def render_pdf(pdf_file: Path, output_dir: Path, dpi: int = DEFAULT_DPI) -> list[RasterizedPage]:
    """Render every page of `pdf_file` to PNGs in `output_dir` at `dpi`, returning them in
    page order.
    """
    if pdfium is None:
        raise ImportError(
            "sceptre_rs_tools.rasterize needs the optional 'corpus' dependency group "
            "(pypdfium2, pillow). Install it with `uv sync --group corpus`, then re-run "
            "`task python:rasterize`."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    document = pdfium.PdfDocument(pdf_file)
    try:
        page_count = len(document)
        scale = dpi / POINTS_PER_INCH
        pages: list[RasterizedPage] = []
        for index in range(page_count):
            page_number = index + 1
            page = document[index]
            try:
                image = page.render(scale=scale).to_pil()
            finally:
                page.close()
            output_path = output_dir / page_filename(pdf_file.stem, page_number, page_count)
            image.save(output_path)
            pages.append(RasterizedPage(page_number=page_number, path=output_path))
        return pages
    finally:
        document.close()


def concatenate_page_texts(texts: list[str]) -> str:
    """Join per-page recognized text, in page order, into the single string a PDF's
    whole-document ground truth is scored against.

    Sceptre's recognizer runs per image, but a PDF's ground truth is one whole-document
    transcript; concatenating the pages in order (rather than trying to split the ground
    truth per page) is the chosen resolution for scoring a rasterized PDF — see the module
    docstring. Blank pages are dropped so they do not inject spurious paragraph breaks.
    """
    return "\n\n".join(text.strip() for text in texts if text.strip())


def provenance(dpi: int) -> dict[str, object]:
    """Per-run rendering provenance: the pinned DPI plus the exact pdfium/pypdfium2 build
    that produced the pages, so a re-run on the same pdfium build reproduces the same bytes.
    """
    pdfium_build = getattr(pdfium_version, "PDFIUM_INFO", None) if pdfium_version else None
    pypdfium2_build = getattr(pdfium_version, "PYPDFIUM_INFO", None) if pdfium_version else None
    return {
        "dpi": dpi,
        "pdfium_version": str(pdfium_build) if pdfium_build is not None else None,
        "pypdfium2_version": str(pypdfium2_build) if pypdfium2_build is not None else None,
        "python": platform.python_version(),
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def rasterize_one(stem: str, output_dir: Path, dpi: int) -> dict[str, object]:
    """Render one corpus PDF and build its manifest entry: pages, ground truth (if any), and
    provenance.
    """
    source = pdf_path(stem)
    if not source.exists():
        raise FileNotFoundError(
            f"{source} not found; fetch it first with `python3 "
            f"test_documents/scripts/fetch_corpus.py --include 'pdf/{stem}.pdf'`"
        )
    pages = render_pdf(source, output_dir, dpi)
    ground_truth = ground_truth_path(stem)
    return {
        "stem": stem,
        "source": str(source),
        "pages": [str(page.path) for page in pages],
        "ground_truth": str(ground_truth) if ground_truth.exists() else None,
        "provenance": provenance(dpi),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: rasterize one or more corpus PDFs by stem; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf",
        action="append",
        default=[],
        required=True,
        dest="stems",
        help="corpus PDF stem under test_documents/pdf/ (repeatable), e.g. --pdf 10075815",
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help=f"rendering DPI (default {DEFAULT_DPI})")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"output directory (default <repo root>/{DEFAULT_OUTPUT_DIRNAME})",
    )
    args = parser.parse_args(argv)

    output_dir = args.output_dir or default_output_dir()
    manifest: list[dict[str, object]] = []
    for stem in args.stems:
        try:
            entry = rasterize_one(stem, output_dir, args.dpi)
        except (ImportError, FileNotFoundError) as error:
            print(f"sceptre_rs_tools.rasterize: {error}", file=sys.stderr)
            return 1
        manifest.append(entry)
        print(
            f"sceptre_rs_tools.rasterize: wrote {len(entry['pages'])} page(s) for {stem} to {output_dir}",
            file=sys.stderr,
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"sceptre_rs_tools.rasterize: wrote {manifest_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
