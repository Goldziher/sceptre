"""Corpus manifest for the sceptre-vs-EasyOCR comparative benchmark.

Declares the image corpus and, for each entry, the abstract language names that are
mapped to both engines' language codes. Labeled entries carry a ground-truth path so
absolute CER/WER can be scored; breadth entries only support cross-engine agreement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Abstract language name -> EasyOCR language code. Latin maps to English per the
# benchmark spec (the itext latin_g2 recognizer has no dedicated EasyOCR analogue).
EASYOCR_CODES: dict[str, str] = {
    "english": "en",
    "latin": "en",
    "chinese": "ch_sim",
    "japanese": "ja",
    "korean": "ko",
}

# Abstract language name -> sceptre `--lang` value (kebab-case clap variants).
SCEPTRE_LANGS: dict[str, str] = {
    "english": "english",
    "latin": "latin",
    "chinese": "chinese-simplified",
    "japanese": "japanese",
    "korean": "korean",
}

# Image extensions both engines can decode, tried in order when resolving a stem.
IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp")

# Base directories, keyed by a short manifest tag, relative to the repo root.
IMAGE_BASES: dict[str, Path] = {
    "documents": Path("test_documents/images"),
    "examples": Path("crates/sceptre/tests/data/images"),
}

GROUND_TRUTH_BASE = Path("test_documents/ground_truth/images")
GROUND_TRUTH_EXTENSIONS: tuple[str, ...] = (".md", ".txt")


@dataclass(frozen=True)
class ManifestRecord:
    """One planned benchmark unit before path resolution."""

    stem: str
    base: str
    languages: tuple[str, ...]
    group: str  # "labeled" or "breadth"


@dataclass(frozen=True)
class CorpusEntry:
    """A resolved corpus entry ready to run through both engines."""

    stem: str
    image: Path | None
    ground_truth: Path | None
    languages: tuple[str, ...]
    group: str

    @property
    def easyocr_codes(self) -> list[str]:
        """EasyOCR language codes for this entry, de-duplicated in order."""
        return _dedupe(EASYOCR_CODES[name] for name in self.languages)

    @property
    def sceptre_langs(self) -> list[str]:
        """sceptre `--lang` values for this entry, de-duplicated in order."""
        return _dedupe(SCEPTRE_LANGS[name] for name in self.languages)


# Labeled corpus: images with committed ground truth under test_documents/ground_truth.
_LABELED: tuple[ManifestRecord, ...] = (
    ManifestRecord("balance_sheet_1", "documents", ("english",), "labeled"),
    ManifestRecord("financial_table_1", "documents", ("english",), "labeled"),
    ManifestRecord("invoice_image", "documents", ("english",), "labeled"),
    ManifestRecord("complex_document", "documents", ("english",), "labeled"),
    ManifestRecord("complex_document_rotated_90", "documents", ("english",), "labeled"),
    ManifestRecord("complex_document_rotated_180", "documents", ("english",), "labeled"),
    ManifestRecord("complex_document_rotated_270", "documents", ("english",), "labeled"),
    ManifestRecord("ocr_test_original", "documents", ("english",), "labeled"),
    ManifestRecord("ocr_test_rotated_90", "documents", ("english",), "labeled"),
    ManifestRecord("ocr_test_rotated_180", "documents", ("english",), "labeled"),
    ManifestRecord("ocr_test_rotated_270", "documents", ("english",), "labeled"),
    ManifestRecord("layout_parser_paper_with_table", "documents", ("english",), "labeled"),
    ManifestRecord("english_and_korean", "documents", ("english", "korean"), "labeled"),
)

# Breadth corpus: no ground truth, used for cross-engine agreement and speed only.
_BREADTH: tuple[ManifestRecord, ...] = (
    ManifestRecord("chi_sim_image", "documents", ("chinese",), "breadth"),
    ManifestRecord("jpn_vert", "documents", ("japanese",), "breadth"),
    ManifestRecord("ocr_image", "documents", ("english",), "breadth"),
    ManifestRecord("layout_parser_ocr", "documents", ("english",), "breadth"),
    ManifestRecord("sample_text", "documents", ("english",), "breadth"),
    ManifestRecord("test_hello_world", "documents", ("english",), "breadth"),
    ManifestRecord("sample", "documents", ("english",), "breadth"),
    ManifestRecord("simple_table", "documents", ("english",), "breadth"),
    ManifestRecord("english", "examples", ("english",), "breadth"),
    ManifestRecord("french", "examples", ("latin",), "breadth"),
    ManifestRecord("chinese", "examples", ("chinese",), "breadth"),
    ManifestRecord("japanese", "examples", ("japanese",), "breadth"),
    ManifestRecord("korean", "examples", ("korean",), "breadth"),
)

MANIFEST: tuple[ManifestRecord, ...] = _LABELED + _BREADTH


def _dedupe(values) -> list[str]:
    """Return values with duplicates removed while preserving first-seen order."""
    seen: list[str] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def _resolve_image(root: Path, record: ManifestRecord) -> Path | None:
    """Find the on-disk image for a manifest record, trying known extensions."""
    base = root / IMAGE_BASES[record.base]
    for extension in IMAGE_EXTENSIONS:
        candidate = base / f"{record.stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def _resolve_ground_truth(root: Path, record: ManifestRecord) -> Path | None:
    """Find the ground-truth transcript for a labeled record, if any exists."""
    if record.group != "labeled":
        return None
    base = root / GROUND_TRUTH_BASE
    for extension in GROUND_TRUTH_EXTENSIONS:
        candidate = base / f"{record.stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def build_corpus(root: Path, group: str = "all") -> list[CorpusEntry]:
    """Resolve the manifest into corpus entries filtered by ``group``.

    ``group`` is ``"labeled"`` (only ground-truth entries) or ``"all"`` (everything).
    An entry whose image cannot be found is returned with ``image=None`` so the caller
    can record it as skipped rather than crashing.
    """
    entries: list[CorpusEntry] = []
    for record in MANIFEST:
        if group == "labeled" and record.group != "labeled":
            continue
        entries.append(
            CorpusEntry(
                stem=record.stem,
                image=_resolve_image(root, record),
                ground_truth=_resolve_ground_truth(root, record),
                languages=record.languages,
                group=record.group,
            )
        )
    return entries
