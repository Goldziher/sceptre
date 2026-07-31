//! CRNN recognition with CTC decoding.
//!
//! Stages: [`crop`] (perspective-crop each region, resize to height 64) →
//! [`preprocess`] (normalize + pad a batch) → [`crnn`] (run the recognizer) →
//! [`ctc`] (decode logits to text). [`charset`] supplies per-language alphabets;
//! [`contrast`] drives the low-confidence second pass. Crops are resized to a
//! height of 64 px (EasyOCR `imgH`).

mod charset;
mod contrast;
mod crnn;
mod crop;
mod ctc;
mod preprocess;
mod recognizer;

pub(crate) use recognizer::{CrnnRecognizer, TextRecognizer};

use crate::config::RecognitionConfig;
use crate::detect::Detection;
use crate::error::Result;
use crate::imaging::LoadedImage;
use crate::inference::ModelBackend;
use crate::types::TextLine;

/// Recognize text for every detected region.
// Legacy stage entry point, superseded by the `TextRecognizer` seam. ~keep
#[allow(dead_code)]
pub fn recognize(
    _backend: &dyn ModelBackend,
    _image: &LoadedImage,
    _detection: &Detection,
    _charset: &charset::Charset,
    _config: &RecognitionConfig,
) -> Result<Vec<TextLine>> {
    todo!("crop, preprocess, run the CRNN, then CTC-decode")
}
