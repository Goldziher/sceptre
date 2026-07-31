"""Dev-only fallback tooling for easyocr-rs.

This package hosts the Python (uv, torch/easyocr) fallback model-export path and the
golden-fixture generator. The preferred model export/conversion path is the Rust
``tools/`` crate (``easyocr-tools``, candle-based); see ADR 0008.
"""

__version__ = "0.1.0"
