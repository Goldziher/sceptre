//! CRNN recognition with CTC decoding.
//!
//! Stages: [`crop`] (perspective-crop each region, resize to height 64) →
//! [`preprocess`] (normalize + pad a batch) → [`crnn`] (run the recognizer) →
//! [`ctc`] (decode logits to text). [`charset`] supplies per-language alphabets;
//! [`contrast`] drives the low-confidence second pass. Crops are resized to a
//! height of 64 px (EasyOCR `imgH`).
//!
//! The recognizer seam and its stage DTOs live in [`recognizer`]; the individual
//! stages are module-private helpers wired together by
//! [`recognizer::CrnnRecognizer`].

mod charset;
mod contrast;
mod crnn;
mod crop;
mod ctc;
mod preprocess;
mod recognizer;

pub(crate) use charset::Charset;
pub(crate) use crop::crop_region;
pub(crate) use recognizer::{CrnnRecognizer, RecognizedText, RegionCrop, TextRecognizer};
