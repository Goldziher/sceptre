//! Download and cache model artifacts.
//!
//! Resolves a [`ModelEntry`] to a local path under the cache directory (default
//! `~/.cache/easyocr-rs`), downloading from Hugging Face and verifying the
//! SHA-256 when the `download` feature is enabled.

use std::path::{Path, PathBuf};

use crate::error::{OcrError, Result};
use crate::models::registry::ModelEntry;

/// Ensure a model artifact is present locally, returning its path.
#[cfg(feature = "download")]
pub fn ensure(_entry: &ModelEntry, _cache_dir: &Path) -> Result<PathBuf> {
    todo!("fetch from Hugging Face, cache the file, and verify its sha256")
}

/// Ensure a model artifact is present locally, returning its path.
#[cfg(not(feature = "download"))]
pub fn ensure(_entry: &ModelEntry, _cache_dir: &Path) -> Result<PathBuf> {
    Err(OcrError::model(
        "model download requires the `download` feature; provide a local model path instead",
    ))
}

/// Default cache directory: `~/.cache/easyocr-rs` (or the platform cache dir).
pub fn default_cache_dir() -> Result<PathBuf> {
    dirs::cache_dir()
        .map(|d| d.join("easyocr-rs"))
        .ok_or_else(|| OcrError::model("could not determine a cache directory"))
}
