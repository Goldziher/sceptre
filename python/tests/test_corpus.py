"""Unit tests for the benchmark corpus manifest resolution."""

from __future__ import annotations

from pathlib import Path

from sceptre_rs_tools import corpus


def _repo_root() -> Path:
    """Repository root (this file lives at ``python/tests/``)."""
    return Path(__file__).resolve().parents[2]


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
    for entry in corpus.build_corpus(root, "capability"):
        # These files exist in the submodule; each probe pins an exact non-png/jpg extension.
        assert entry.image is not None, f"{entry.stem} did not resolve"
        assert entry.image.suffix in {".bmp", ".heif", ".avif", ".jp2"}


def test_language_codes_dedupe_and_map() -> None:
    entry = corpus.CorpusEntry(stem="s", image=None, ground_truth=None, languages=("english", "latin"), group="breadth")
    # english + latin both map to EasyOCR "en"; dedupe collapses them, sceptre keeps both.
    assert entry.easyocr_codes == ["en"]
    assert entry.sceptre_langs == ["english", "latin"]
