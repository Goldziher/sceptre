//! Injectable extension points ("seams") with in-crate default implementations.

use std::path::PathBuf;

use crate::config::{Language, OcrConfig};
use crate::error::Result;
use crate::models::download::{self, default_cache_dir};
use crate::models::registry::{craft_entry, recognizer_entry};

/// Resolves model artifacts to local paths (download + cache by default).
pub trait ModelProvider: Send + Sync {
    /// Path to the CRAFT detector ONNX.
    fn detector(&self) -> Result<PathBuf>;
    /// Path to the recognizer ONNX for a language group.
    fn recognizer(&self, language: Language) -> Result<PathBuf>;
}

/// Receives coarse progress notifications from the pipeline.
pub trait ProgressSink: Send + Sync {
    /// Called when a pipeline stage begins.
    fn on_stage(&self, _stage: &str) {}
}

/// A progress sink that discards all notifications.
pub(crate) struct NoopProgress;

impl ProgressSink for NoopProgress {}

/// The default model provider: resolves via the cache dir, downloading on miss.
pub(crate) struct DefaultModelProvider {
    cache_dir: PathBuf,
    registry_owner: Option<String>,
}

impl DefaultModelProvider {
    /// Build from config, honoring `model.cache_dir` or the platform default and
    /// the optional `model.registry_owner` re-pointing override.
    pub(crate) fn from_config(config: &OcrConfig) -> Result<Self> {
        let cache_dir = match &config.model.cache_dir {
            Some(dir) => dir.clone(),
            None => default_cache_dir()?,
        };
        Ok(Self {
            cache_dir,
            registry_owner: config.model.registry_owner.clone(),
        })
    }
}

impl ModelProvider for DefaultModelProvider {
    fn detector(&self) -> Result<PathBuf> {
        download::ensure(&craft_entry(), &self.cache_dir, self.registry_owner.as_deref())
    }

    fn recognizer(&self, language: Language) -> Result<PathBuf> {
        download::ensure(
            &recognizer_entry(language),
            &self.cache_dir,
            self.registry_owner.as_deref(),
        )
    }
}
