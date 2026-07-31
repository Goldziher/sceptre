//! # sceptre
//!
//! A Rust reimplementation of [EasyOCR](https://github.com/JaidedAI/EasyOCR)'s OCR
//! pipeline — CRAFT text detection followed by gen2 CRNN recognition with CTC
//! decoding — running the models over ONNX.
//!
//! ## Architecture
//!
//! The pipeline is three stages behind a single [`Reader`] handle:
//!
//! 1. **Detection** ([`detect`]) — CRAFT produces region/link heat-maps that are
//!    post-processed into text boxes.
//! 2. **Recognition** ([`recognize`]) — each box is cropped, normalized, and run
//!    through a gen2 CRNN; the logits are CTC-decoded into text.
//! 3. **Inference backend** ([`inference`]) — a runtime-neutral [`inference::ModelBackend`]
//!    seam abstracts over `ort` (native), `tract` (pure-Rust), and `candle` (deferred).
//!
//! Models are resolved and cached by [`models`]; behavior is configured through
//! [`config`].
#![doc(html_root_url = "https://docs.rs/sceptre")]

pub(crate) mod config;
pub(crate) mod detect;
pub(crate) mod engine;
pub(crate) mod error;
pub(crate) mod imaging;
pub(crate) mod inference;
pub(crate) mod models;
pub(crate) mod recognize;
pub(crate) mod types;

#[cfg(feature = "mcp")]
pub mod mcp;

pub use config::{
    Backend, ConcurrencyConfig, Decoder, DetectionConfig, Language, ModelConfig, OcrConfig, RecognitionConfig,
};
pub use engine::seams::{ModelProvider, ProgressSink};
pub use engine::{FallbackEngine, OcrEngine, ReadOptions, Reader, ReaderBuilder};
pub use error::{OcrError, Result};
pub use types::{BBox, Image, OcrResult, Point, Quad, TextLine};
