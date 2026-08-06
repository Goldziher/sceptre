"""Standalone EasyOCR batch driver, run as a fresh subprocess per language group.

The comparative benchmark launches this module once per language group under the same
``/usr/bin/time`` wrapper it uses for the sceptre CLI, so both engines' peak RSS is a
whole-process figure measured identically (see ADR on benchmark methodology). A single
``easyocr.Reader`` is constructed and reused across every image in the group, mirroring how
the sceptre batch process loads its model once — this is the honest warm/batch axis.

It emits one JSON object on stdout::

    {
        "reader_build_seconds": 1.83,
        "versions": {"easyocr": "1.7.2", "torch": "2.5.1", "python": "3.11.9 (...)"},
        "images": [
            {"image": "path", "seconds": 0.42, "detections": [{"text": "...", "quad": [[x, y], ...]}]},
            {"image": "path", "error": "..."},
        ],
    }

The ``versions`` block is the reference half of the benchmark report's ``environment``
provenance: published EasyOCR numbers are meaningless without the versions that produced them.

Timing is per-``readtext`` (warm inner time, model already loaded). Threads default to
torch's native setting; ``--threads N`` pins torch/OMP to match a ``sceptre --threads N`` run.
The heavy ``export`` dependency group (torch, easyocr) must be installed; without it the
module prints an explanatory message to stderr and exits non-zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from time import perf_counter


def _pin_threads(threads: int | None) -> None:
    """Pin torch/OMP to ``threads`` when given; otherwise leave torch's native default."""
    if threads is None:
        return
    import os

    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    try:
        import torch

        torch.set_num_threads(threads)
    except ImportError:
        pass


def _versions() -> dict[str, str]:
    """Report the reference runtime's versions so the benchmark can record provenance."""
    versions: dict[str, str] = {"python": sys.version}
    for name in ("easyocr", "torch"):
        try:
            module = __import__(name)
        except ImportError:
            continue
        version = getattr(module, "__version__", None)
        if version is not None:
            versions[name] = str(version)
    return versions


def _detections(reader: object, image: str) -> list[dict[str, object]]:
    """Run EasyOCR over one image, normalizing each result to ``{text, quad}``."""
    raw = reader.readtext(image)  # type: ignore[attr-defined]
    detections: list[dict[str, object]] = []
    for box, text, _confidence in raw:
        quad = [[float(point[0]), float(point[1])] for point in box[:4]]
        detections.append({"text": str(text), "quad": quad})
    return detections


def run(languages: list[str], images: list[str], threads: int | None) -> dict[str, object]:
    """Build one reader for ``languages`` and recognize every image, timing each call."""
    import easyocr

    _pin_threads(threads)

    build_start = perf_counter()
    # gpu=False keeps the benchmark reproducible across machines without a GPU. ~keep
    reader = easyocr.Reader(languages, gpu=False)
    reader_build_seconds = perf_counter() - build_start

    results: list[dict[str, object]] = []
    for image in images:
        start = perf_counter()
        try:
            detections = _detections(reader, image)
        except Exception as error:  # noqa: BLE001 - record the per-image failure, keep going
            results.append({"image": image, "error": f"{type(error).__name__}: {str(error)[:200]}"})
            continue
        results.append({"image": image, "seconds": perf_counter() - start, "detections": detections})
    return {"reader_build_seconds": reader_build_seconds, "versions": _versions(), "images": results}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse ``--lang`` (repeatable) plus the trailing image paths."""
    parser = argparse.ArgumentParser(description="Run EasyOCR over a batch of images (one Reader).")
    parser.add_argument(
        "--lang", action="append", dest="languages", default=[], help="EasyOCR language code (repeatable)."
    )
    parser.add_argument("--threads", type=int, default=None, help="Pin torch/OMP to N threads (default: torch native).")
    parser.add_argument("images", nargs="+", help="Image paths to recognize.")
    return parser.parse_args(argv)


def main() -> None:
    """CLI entry point: emit the batch result JSON on stdout."""
    args = parse_args()
    try:
        import easyocr  # noqa: F401 - probe availability before doing any work
    except ImportError:
        print(
            "sceptre_rs_tools._easyocr_runner: needs the optional 'export' dependency group "
            "(torch, easyocr). Install it with `uv sync --group export`, then re-run.",
            file=sys.stderr,
        )
        sys.exit(1)

    languages = args.languages or ["en"]
    result = run(languages, args.images, args.threads)
    json.dump(result, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
