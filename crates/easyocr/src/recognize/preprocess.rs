//! Recognition preprocessing: normalize to [-1, 1] and pad a batch to equal width.
//!
//! Reference: EasyOCR `recognition.py` (`NormalizePAD`, `AlignCollate`).
//! Grayscale pixels are scaled to [0, 1] then `(x - 0.5) / 0.5`; crops are
//! right-padded (edge-replicated) to the batch's maximum width, forming a
//! `[B, 1, 64, W]` tensor.
