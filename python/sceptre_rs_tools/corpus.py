"""Corpus manifest for the sceptre-vs-EasyOCR comparative benchmark.

Declares the image corpus and, for each entry, the abstract language names that are
mapped to both engines' language codes. Entries fall into three groups:

  * ``labeled`` — carry a ground-truth transcript, so absolute CER/WER/token-F1 are scored.
  * ``breadth`` — no ground truth; support cross-engine agreement and speed only.
  * ``capability`` — a format sceptre cannot yet decode (BMP/HEIF/AVIF/JP2); the benchmark
    expects sceptre to report an unsupported-format gap while EasyOCR decodes it. These
    probe input-format coverage and are excluded from timing/quality aggregates.
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
IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff")

# Base directories, keyed by a short manifest tag, relative to the repo root. Both tags
# resolve to the same vendored directory; "documents" and "examples" remain distinct keys
# because they group manifest records by provenance, not by location (see ATTRIBUTIONS.md
# in that directory for the "documents" entries' upstream licenses).
IMAGE_BASES: dict[str, Path] = {
    "documents": Path("crates/sceptre/tests/data/images"),
    "examples": Path("crates/sceptre/tests/data/images"),
}

# Transcripts for the `labeled` records, vendored alongside the images they describe. Upstream
# `test_documents` split these across four directories by source extension; manifest stems are
# unique, so one flat directory holds them without collision.
GROUND_TRUTH_BASES: tuple[Path, ...] = (Path("crates/sceptre/tests/data/ground_truth"),)
GROUND_TRUTH_EXTENSIONS: tuple[str, ...] = (".md", ".txt")


@dataclass(frozen=True)
class ManifestRecord:
    """One planned benchmark unit before path resolution.

    ``extension`` pins an exact image extension (used for capability probes whose stem is
    ambiguous or whose format is not in ``IMAGE_EXTENSIONS``); when ``None`` the resolver
    tries ``IMAGE_EXTENSIONS`` in order.
    """

    stem: str
    base: str
    languages: tuple[str, ...]
    group: str  # "labeled", "breadth", or "capability"
    extension: str | None = None


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


# Labeled corpus: images with a committed ground-truth transcript.
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
    # Scene text and dense documents with authoritative ground truth (ground_truth/jpg).
    ManifestRecord("textocr_scene_01", "documents", ("english",), "labeled"),
    ManifestRecord("textocr_scene_02", "documents", ("english",), "labeled"),
    ManifestRecord("textocr_scene_03", "documents", ("english",), "labeled"),
    ManifestRecord("doclaynet_page_01", "documents", ("english",), "labeled"),
    ManifestRecord("doclaynet_page_02", "documents", ("english",), "labeled"),
    ManifestRecord("cord_receipt_01", "documents", ("english",), "labeled"),
    ManifestRecord("cord_receipt_02", "documents", ("english",), "labeled"),
    ManifestRecord("cord_receipt_03", "documents", ("english",), "labeled"),
    ManifestRecord("cord_receipt_04", "documents", ("english",), "labeled"),
    # Vertical Japanese: dense, and a known-hard case for both engines.
    ManifestRecord("ndl_meiji_vertical_01", "documents", ("japanese",), "labeled"),
    ManifestRecord("ndl_meiji_vertical_02", "documents", ("japanese",), "labeled"),
    ManifestRecord("ndl_meiji_vertical_03", "documents", ("japanese",), "labeled"),
    ManifestRecord("ndl_meiji_vertical_04", "documents", ("japanese",), "labeled"),
    ManifestRecord("ndl_meiji_vertical_05", "documents", ("japanese",), "labeled"),
)

# Breadth corpus: no ground truth, used for cross-engine agreement and speed only.
_BREADTH: tuple[ManifestRecord, ...] = (
    ManifestRecord("chi_sim_image", "documents", ("chinese",), "breadth"),
    ManifestRecord("jpn_vert", "documents", ("japanese",), "breadth"),
    ManifestRecord("ocr_image", "documents", ("english",), "breadth"),
    ManifestRecord("layout_parser_ocr", "documents", ("english",), "breadth"),
    ManifestRecord("test_hello_world", "documents", ("english",), "breadth"),
    ManifestRecord("sample", "documents", ("english",), "breadth"),
    # BMP is now decodable (pure-Rust image feature), so this is a scored breadth entry.
    ManifestRecord("sample_text", "documents", ("english",), "breadth", ".bmp"),
    ManifestRecord("simple_table", "documents", ("english",), "breadth"),
    ManifestRecord("english", "examples", ("english",), "breadth"),
    ManifestRecord("french", "examples", ("latin",), "breadth"),
    ManifestRecord("chinese", "examples", ("chinese",), "breadth"),
    ManifestRecord("japanese", "examples", ("japanese",), "breadth"),
    ManifestRecord("korean", "examples", ("korean",), "breadth"),
)

# Capability probes: formats sceptre cannot decode yet (see Cargo.toml image features).
# EasyOCR decodes them; the benchmark reports each as an unsupported-format gap.
# Remaining gaps need C libraries (libheif / dav1d / openjpeg), which would break the
# pure-Rust WASM/Android build, so they stay out of scope (see ADR 0022).
_CAPABILITY: tuple[ManifestRecord, ...] = (
    ManifestRecord("alpha", "documents", ("english",), "capability", ".heif"),
    ManifestRecord("test", "documents", ("english",), "capability", ".avif"),
    ManifestRecord("Hadley_Crater", "documents", ("english",), "capability", ".jp2"),
)

MANIFEST: tuple[ManifestRecord, ...] = _LABELED + _BREADTH + _CAPABILITY


def _dedupe(values) -> list[str]:
    """Return values with duplicates removed while preserving first-seen order."""
    seen: list[str] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def _resolve_image(root: Path, record: ManifestRecord) -> Path | None:
    """Find the on-disk image for a manifest record.

    A pinned ``extension`` resolves exactly; otherwise the known decodable extensions are
    tried in order.
    """
    base = root / IMAGE_BASES[record.base]
    extensions = (record.extension,) if record.extension else IMAGE_EXTENSIONS
    for extension in extensions:
        candidate = base / f"{record.stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def _resolve_ground_truth(root: Path, record: ManifestRecord) -> Path | None:
    """Find the ground-truth transcript for a labeled record across the GT bases, if any."""
    if record.group != "labeled":
        return None
    for base in GROUND_TRUTH_BASES:
        for extension in GROUND_TRUTH_EXTENSIONS:
            candidate = root / base / f"{record.stem}{extension}"
            if candidate.exists():
                return candidate
    return None


def build_corpus(root: Path, group: str = "all") -> list[CorpusEntry]:
    """Resolve the manifest into corpus entries filtered by ``group``.

    ``group`` is ``"labeled"``, ``"breadth"``, ``"capability"``, or ``"all"`` (everything).
    An entry whose image cannot be found is returned with ``image=None`` so the caller can
    record it as skipped rather than crashing.
    """
    entries: list[CorpusEntry] = []
    for record in MANIFEST:
        if group != "all" and record.group != group:
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
