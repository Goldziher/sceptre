//! Low-confidence second-pass contrast adjustment.
//!
//! Reference: EasyOCR `recognition.py` (`adjust_contrast_grey`). When a crop's
//! contrast is below `contrast_ths`, it is re-normalized using the 10th/90th
//! percentile range and recognized again.
