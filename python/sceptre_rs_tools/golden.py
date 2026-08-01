"""Generate the EasyOCR-reference side of the golden fixtures.

Runs upstream Python EasyOCR (torch) over the committed example images and writes the
recognized text plus per-line quad into ``crates/sceptre/tests/data/golden/<image>.json``.
Only the ``easyocr`` side of each dual golden is written; any existing ``sceptre``
snapshot side is preserved (see ``crates/sceptre/tests/data/golden/README.md`` and
ADR 0016). The heavy ``easyocr``/``torch`` dependencies live in the opt-in ``export``
group; without them this exits cleanly with an explanatory message.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Maps each example image to the EasyOCR language codes to load for it. English is the
# default corpus; the others exercise the per-language gen2 recognizers. ~keep
IMAGE_LANGUAGES: dict[str, list[str]] = {
    "english.png": ["en"],
    "example.png": ["en"],
    "french.jpg": ["fr", "en"],
    "chinese.jpg": ["ch_sim", "en"],
    "japanese.jpg": ["ja", "en"],
    "korean.png": ["ko", "en"],
    "cyrillic.png": ["ru"],
}


def repo_root() -> Path:
    """Locate the repository root from this file's location."""
    return Path(__file__).resolve().parents[2]


def images_dir() -> Path:
    """Directory holding the committed example images."""
    return repo_root() / "crates" / "sceptre" / "tests" / "data" / "images"


def golden_dir() -> Path:
    """Directory holding the golden fixtures."""
    return repo_root() / "crates" / "sceptre" / "tests" / "data" / "golden"


def quad_from_box(box: list[list[float]]) -> list[list[float]]:
    """Normalize an EasyOCR polygon into four ``[x, y]`` float corners."""
    return [[float(point[0]), float(point[1])] for point in box[:4]]


def easyocr_lines(reader: object, image_path: Path) -> list[dict[str, object]]:
    """Run an EasyOCR reader over one image and return golden line dicts."""
    detections = reader.readtext(str(image_path))  # type: ignore[attr-defined]
    lines: list[dict[str, object]] = []
    for box, text, _confidence in detections:
        lines.append({"text": text, "quad": quad_from_box(box)})
    return lines


def merge_easyocr_side(existing: dict[str, object], lines: list[dict[str, object]]) -> dict[str, object]:
    """Overwrite only the ``easyocr`` side, preserving any existing ``sceptre`` side."""
    sceptre_side = existing.get("sceptre", {"lines": []})
    merged: dict[str, object] = {
        "placeholder": False,
        "easyocr": {"lines": lines},
        "sceptre": sceptre_side,
    }
    return merged


def load_existing(path: Path) -> dict[str, object]:
    """Load an existing fixture, or an empty dual golden if none/invalid."""
    if not path.exists():
        return {"sceptre": {"lines": []}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"sceptre": {"lines": []}}


def generate() -> int:
    """Generate the EasyOCR reference goldens; returns a process exit code."""
    try:
        import easyocr
    except ImportError:
        print(
            "sceptre_rs_tools.golden: the reference generator needs the optional 'export' "
            "dependency group (torch, easyocr). Install it with `uv sync --group export`, "
            "then re-run `task py:golden`.",
            file=sys.stderr,
        )
        return 1

    images = images_dir()
    goldens = golden_dir()
    goldens.mkdir(parents=True, exist_ok=True)

    readers: dict[tuple[str, ...], object] = {}
    for image_name, languages in IMAGE_LANGUAGES.items():
        image_path = images / image_name
        if not image_path.exists():
            print(f"sceptre_rs_tools.golden: skipping missing image {image_path}", file=sys.stderr)
            continue

        key = tuple(languages)
        if key not in readers:
            # gpu=False keeps generation reproducible across machines without a GPU. ~keep
            readers[key] = easyocr.Reader(list(languages), gpu=False)
        lines = easyocr_lines(readers[key], image_path)

        fixture_path = goldens / f"{Path(image_name).stem}.json"
        merged = merge_easyocr_side(load_existing(fixture_path), lines)
        fixture_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"sceptre_rs_tools.golden: wrote easyocr side of {fixture_path} ({len(lines)} lines)", file=sys.stderr)

    return 0


def main() -> None:
    """CLI entry point."""
    sys.exit(generate())


if __name__ == "__main__":
    main()
