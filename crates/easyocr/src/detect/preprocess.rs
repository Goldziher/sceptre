//! Detection preprocessing: aspect-ratio resize and mean/variance normalization.
//!
//! Reference: EasyOCR `imgproc.py` (`resize_aspect_ratio`, `normalizeMeanVariance`).
//! RGB is normalized with ImageNet mean/std (each scaled by 255) and laid out as
//! an NCHW tensor before the CRAFT forward pass.
