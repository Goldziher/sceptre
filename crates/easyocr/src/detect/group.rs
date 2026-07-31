//! Box grouping: merge character/word boxes into text lines.
//!
//! Reference: EasyOCR `utils.py` (`group_text_box`). Splits horizontal vs. free
//! (rotated) boxes via the slope threshold and merges using the y-center,
//! height, and width thresholds, applying `add_margin`.
