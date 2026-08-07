"""Unit tests for the benchmark corpus manifest resolution."""

from __future__ import annotations

from pathlib import Path

import pytest
from sceptre_rs_tools import corpus


def _repo_root() -> Path:
    """Repository root (this file lives at ``python/tests/``)."""
    return Path(__file__).resolve().parents[2]


def _require_corpus_images(root: Path) -> None:
    """Skip when the `test_documents` image binaries are absent.

    The manifest itself is static, so most of this module runs anywhere; only the
    extension pins need the real files. CI's Python job checks out without the
    submodule and never runs `fetch_corpus.py`, so it must skip rather than fail —
    but it skips loudly and never falls back to a stand-in image.
    """
    images = corpus.test_documents_dir(root) / "images"
    if not images.is_dir() or not any(images.iterdir()):
        pytest.skip(f"test_documents image corpus not fetched at {images}")


def test_build_corpus_filters_by_group() -> None:
    root = _repo_root()
    labeled = corpus.build_corpus(root, "labeled")
    capability = corpus.build_corpus(root, "capability")
    assert labeled and all(entry.group == "labeled" for entry in labeled)
    assert capability and all(entry.group == "capability" for entry in capability)


def test_all_group_is_the_full_manifest() -> None:
    root = _repo_root()
    assert len(corpus.build_corpus(root, "all")) == len(corpus.MANIFEST)


def test_capability_probes_resolve_to_pinned_extensions() -> None:
    root = _repo_root()
    _require_corpus_images(root)
    for entry in corpus.build_corpus(root, "capability"):
        # These files exist in the submodule; each probe pins an exact non-png/jpg extension.
        assert entry.image is not None, f"{entry.stem} did not resolve"
        assert entry.image.suffix in {".bmp", ".heif", ".avif", ".jp2"}


def test_language_codes_dedupe_and_map() -> None:
    entry = corpus.CorpusEntry(stem="s", image=None, ground_truth=None, languages=("english", "latin"), group="breadth")
    # english + latin both map to EasyOCR "en"; dedupe collapses them, sceptre keeps both.
    assert entry.easyocr_codes == ["en"]
    assert entry.sceptre_langs == ["english", "latin"]
