//! Image loading and colour-space preparation.
//!
//! Mirrors EasyOCR's `reformat_input`: every input is decoded into both an RGB
//! image (for CRAFT detection) and a grayscale image (for recognition).

use std::path::Path;

use crate::error::Result;

/// An image decoded into the two colour spaces the pipeline needs.
pub struct LoadedImage {
    /// RGB view, consumed by detection.
    pub rgb: image::RgbImage,
    /// Grayscale view, consumed by recognition.
    pub grey: image::GrayImage,
}

/// Load and prepare an image from a filesystem path.
pub fn load(_path: &Path) -> Result<LoadedImage> {
    todo!("decode the file into RGB and grayscale views")
}

/// Load and prepare an image from raw encoded bytes.
pub fn load_bytes(_bytes: &[u8]) -> Result<LoadedImage> {
    todo!("decode the bytes into RGB and grayscale views")
}
