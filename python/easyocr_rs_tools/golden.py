"""Generate golden JSON fixtures from the reference Python EasyOCR pipeline.

Runs upstream EasyOCR over the example images and serializes its output to JSON, which
the Rust test suite validates against for parity (text equality per line plus a box-IoU
threshold). Requires the heavy, opt-in ``export`` dependency group (torch, easyocr).

The generation logic is not yet implemented.
"""

from __future__ import annotations

import sys


def main() -> None:
    """Print an informative not-implemented message and exit non-zero."""
    print(
        "easyocr_rs_tools.golden: golden-fixture generation not yet implemented — "
        "generates golden JSON fixtures from Python EasyOCR over the example images "
        "for parity testing; requires the 'export' dependency group (torch, easyocr).",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
