"""Fallback model export path: convert EasyOCR ``.pth`` weights to ONNX/safetensors.

This is the FALLBACK export path. The preferred path is the Rust ``tools/`` crate
(``sceptre-tools``, candle-based); see ADR 0008. This module exists so the export can
run from Python via ``torch``/``easyocr`` when the Rust path is unavailable.

Requires the heavy, opt-in ``export`` dependency group (torch, easyocr, onnx, numpy).
The conversion logic is not yet implemented.
"""

from __future__ import annotations

import sys


def main() -> None:
    """Print an informative not-implemented message and exit non-zero."""
    print(
        "sceptre_rs_tools.export: model export not yet implemented — this is the "
        "fallback .pth->ONNX/safetensors export path; requires the 'export' "
        "dependency group (torch, easyocr, onnx, numpy). The preferred path is the "
        "Rust `sceptre-tools` crate (see ADR 0008).",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
