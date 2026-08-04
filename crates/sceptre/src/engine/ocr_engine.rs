//! The primary public extension seam: [`OcrEngine`].

use crate::error::Result;
use crate::types::{Image, OcrResult, Point, Quad, TextLine};

use super::ReadOptions;

/// The primary extension seam: an OCR engine turns an image into recognized text.
///
/// Implementors receive a fully decoded [`Image`] and per-call [`ReadOptions`] and
/// return an [`OcrResult`]. Custom engines can be injected through
/// [`ReaderBuilder::engine`](crate::ReaderBuilder::engine).
pub trait OcrEngine: Send + Sync {
    /// Eagerly initialize reusable model/backend state.
    ///
    /// Custom engines that need no initialization may keep this default no-op.
    fn warm_up(&self) -> Result<()> {
        Ok(())
    }

    /// Recognize all text in `image`, honoring `options`.
    fn recognize(&self, image: &Image, options: &ReadOptions) -> Result<OcrResult>;

    /// Detect text regions only, returning each region's bounding quad in the order
    /// `recognize` would report them. The default derives quads from `recognize`
    /// (running the full pipeline); the built-in engine overrides this to skip
    /// recognition.
    fn detect(&self, image: &Image, options: &ReadOptions) -> Result<Vec<Quad>> {
        Ok(self
            .recognize(image, options)?
            .lines
            .into_iter()
            .map(|line| line.quad)
            .collect())
    }

    /// Recognize `image` as a single, already-cropped text line, skipping detection.
    /// The default runs `recognize` and merges the resulting lines into one line; the
    /// built-in engine overrides this to run the recognizer on the whole image as one
    /// crop.
    fn recognize_line(&self, image: &Image, options: &ReadOptions) -> Result<TextLine> {
        Ok(merge_lines_into_one(image, self.recognize(image, options)?))
    }
}

/// Collapse a multi-line [`OcrResult`] into a single line spanning the whole image.
///
/// The quad covers the entire image; the text joins the line texts with a single
/// ASCII space and the confidence is the arithmetic mean of the line confidences. An
/// empty result yields empty text at confidence `0.0`.
fn merge_lines_into_one(image: &Image, result: OcrResult) -> TextLine {
    let width = image.width() as f32;
    let height = image.height() as f32;
    let quad = Quad {
        points: [
            Point::new(0.0, 0.0),
            Point::new(width, 0.0),
            Point::new(width, height),
            Point::new(0.0, height),
        ],
    };
    if result.lines.is_empty() {
        return TextLine {
            quad,
            text: String::new(),
            confidence: 0.0,
        };
    }
    let count = result.lines.len() as f32;
    let confidence = result.lines.iter().map(|line| line.confidence).sum::<f32>() / count;
    let text = result
        .lines
        .into_iter()
        .map(|line| line.text)
        .collect::<Vec<_>>()
        .join(" ");
    TextLine { quad, text, confidence }
}
