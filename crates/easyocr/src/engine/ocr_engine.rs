//! The primary public extension seam: [`OcrEngine`].

use crate::error::Result;
use crate::types::{Image, OcrResult};

use super::ReadOptions;

/// The primary extension seam: an OCR engine turns an image into recognized text.
///
/// Implementors receive a fully decoded [`Image`] and per-call [`ReadOptions`] and
/// return an [`OcrResult`]. Custom engines can be injected through
/// [`ReaderBuilder::engine`](crate::ReaderBuilder::engine).
pub trait OcrEngine: Send + Sync {
    /// Recognize all text in `image`, honoring `options`.
    fn recognize(&self, image: &Image, options: &ReadOptions) -> Result<OcrResult>;
}
