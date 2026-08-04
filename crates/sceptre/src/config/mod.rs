//! Configuration for the OCR pipeline.
//!
//! [`OcrConfig`] aggregates the per-stage config structs. It is
//! `#[serde(deny_unknown_fields)]` + `#[serde(default)]`, so it loads from
//! partial TOML/JSON and rejects typos. Layered precedence (applied by the CLI)
//! is: defaults < config file < environment < flags.

mod concurrency;
mod detection;
mod model;
mod recognition;

pub use concurrency::ConcurrencyConfig;
pub use detection::DetectionConfig;
pub use model::{Backend, Language, ModelConfig};
pub use recognition::{Decoder, RecognitionConfig};

pub(crate) use concurrency::{WorkerPool, build_thread_pool, resolve_thread_budget};

use serde::{Deserialize, Serialize};

/// Top-level OCR configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OcrConfig {
    /// Text-detection (CRAFT) parameters.
    pub detection: DetectionConfig,
    /// Text-recognition (CRNN + CTC) parameters.
    pub recognition: RecognitionConfig,
    /// Thread-budget / parallelism parameters.
    pub concurrency: ConcurrencyConfig,
    /// Model selection, backend, and cache location.
    pub model: ModelConfig,
}
