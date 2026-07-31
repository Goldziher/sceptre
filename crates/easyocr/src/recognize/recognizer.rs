//! Internal text-recognition seam and its stage DTOs.
//!
//! The recognizer consumes already-cropped, grayscale regions and yields decoded
//! text with a confidence. Crops carry their source corners so the engine can map
//! each result back to a public [`Quad`](crate::types::Quad).

use crate::error::Result;

/// Number of corners describing a crop's source region.
const REGION_CORNERS: usize = 4;

/// Internal seam: turns cropped regions into recognized text.
// Seam method the engine invokes to run CRNN recognition. ~keep
#[allow(dead_code)]
pub(crate) trait TextRecognizer: Send + Sync {
    /// Recognize text for each crop, preserving order.
    fn recognize(&self, crops: &[RegionCrop]) -> Result<Vec<RecognizedText>>;
}

/// Internal DTO: one cropped region ready for the recognizer (grayscale, owned).
// Stage-boundary DTO the engine constructs when feeding the recognizer. ~keep
#[allow(dead_code)]
pub(crate) struct RegionCrop {
    /// Crop width in pixels.
    pub width: u32,
    /// Crop height in pixels.
    pub height: u32,
    /// Row-major grayscale pixels, length `width * height`.
    pub gray: Vec<u8>,
    /// Source location `[x, y]` corners, to map back to a public quad.
    pub corners: [[f32; 2]; REGION_CORNERS],
}

/// Internal DTO: recognizer output for one crop.
// Stage-boundary DTO the engine maps to a public text line. ~keep
#[allow(dead_code)]
pub(crate) struct RecognizedText {
    /// The decoded text.
    pub text: String,
    /// Recognition confidence in `[0.0, 1.0]`.
    pub confidence: f32,
}

/// The CRNN + CTC text recognizer.
pub(crate) struct CrnnRecognizer {}

impl CrnnRecognizer {
    /// Construct a CRNN recognizer.
    pub(crate) fn new() -> Self {
        Self {}
    }
}

impl TextRecognizer for CrnnRecognizer {
    fn recognize(&self, _crops: &[RegionCrop]) -> Result<Vec<RecognizedText>> {
        todo!("run the CRNN over each crop and CTC-decode the logits to text")
    }
}
