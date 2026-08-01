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

/// Crop one axis-aligned region and run recognition batch preprocessing,
/// returning the padded batch tensor. A crate-internal shim over the stage
/// entry points so the `bench` seam can drive them without widening the public
/// API (see ADR 0015).
#[cfg(feature = "bench")]
pub(crate) fn bench_crop_preprocess(
    gray: &image::GrayImage,
    corners: &[[f32; 2]; 4],
) -> crate::error::Result<crate::inference::Tensor> {
    let crop = crop::crop_region(gray, corners, true)?;
    preprocess::prepare_batch(std::slice::from_ref(&crop))
}

/// Greedy-decode recognizer logits with the English charset. Crate-internal
/// `bench` shim over `decode_greedy`.
#[cfg(feature = "bench")]
pub(crate) fn bench_ctc_decode(logits: ndarray::ArrayView2<f32>) -> RecognizedText {
    let charset = charset::Charset::for_language(crate::config::Language::English);
    ctc::decode_greedy(logits, &charset, &[])
}
