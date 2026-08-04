//! Injectable extension points ("seams") with in-crate default implementations.

mod model_provider;

use std::path::PathBuf;

use crate::config::{Language, OcrConfig};
use crate::error::Result;
use crate::models::download;
use crate::models::registry::{craft_entry, recognizer_entry};

pub use model_provider::{ModelArtifact, ModelProvider, VerifiedModelProvider};

/// Receives coarse progress notifications from the pipeline.
pub trait ProgressSink: Send + Sync {
    /// Called when a pipeline stage begins.
    fn on_stage(&self, _stage: &str) {}
}

/// A progress sink that discards all notifications.
pub(crate) struct NoopProgress;

impl ProgressSink for NoopProgress {}

/// The default model provider: resolves through the Hugging Face hub cache,
/// downloading on miss.
pub(crate) struct DefaultModelProvider {
    detector_path: Option<PathBuf>,
    recognizer_path: Option<PathBuf>,
    cache_dir_override: Option<PathBuf>,
    registry_owner: Option<String>,
}

impl DefaultModelProvider {
    /// Build from config, honoring the optional `model.cache_dir` hub-cache-root
    /// override and the optional `model.registry_owner` re-pointing override.
    pub(crate) fn from_config(config: &OcrConfig) -> Result<Self> {
        config.model.validate()?;
        Ok(Self {
            detector_path: config.model.detector_path.clone(),
            recognizer_path: config.model.recognizer_path.clone(),
            cache_dir_override: config.model.cache_dir.clone(),
            registry_owner: config.model.registry_owner.clone(),
        })
    }
}

impl ModelProvider for DefaultModelProvider {
    fn detector(&self) -> Result<ModelArtifact> {
        if let Some(path) = &self.detector_path {
            return Ok(ModelArtifact::Path(path.clone()));
        }
        download::ensure(
            &craft_entry(),
            self.cache_dir_override.as_deref(),
            self.registry_owner.as_deref(),
        )
        .map(ModelArtifact::Path)
    }

    fn recognizer(&self, language: Language) -> Result<ModelArtifact> {
        if let Some(path) = &self.recognizer_path {
            return Ok(ModelArtifact::Path(path.clone()));
        }
        download::ensure(
            &recognizer_entry(language),
            self.cache_dir_override.as_deref(),
            self.registry_owner.as_deref(),
        )
        .map(ModelArtifact::Path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn should_resolve_explicit_model_paths_without_cache_or_download() {
        let mut config = OcrConfig::default();
        config.model.detector_path = Some("assets/craft.onnx".into());
        config.model.recognizer_path = Some("assets/english.onnx".into());
        let provider = DefaultModelProvider::from_config(&config).expect("paired paths are valid");

        assert!(matches!(
            provider.detector().expect("detector path"),
            ModelArtifact::Path(path) if path.as_path() == Path::new("assets/craft.onnx")
        ));
        assert!(matches!(
            provider.recognizer(Language::English).expect("recognizer path"),
            ModelArtifact::Path(path) if path.as_path() == Path::new("assets/english.onnx")
        ));
    }

    #[test]
    fn should_reject_a_detector_path_without_a_recognizer_path() {
        let mut config = OcrConfig::default();
        config.model.detector_path = Some("assets/craft.onnx".into());

        let error = DefaultModelProvider::from_config(&config)
            .err()
            .expect("a recognizer path is required with a detector path");

        assert!(error.to_string().contains("model.recognizer_path is required"));
    }

    #[test]
    fn should_reject_a_recognizer_path_without_a_detector_path() {
        let mut config = OcrConfig::default();
        config.model.recognizer_path = Some("assets/english.onnx".into());

        let error = DefaultModelProvider::from_config(&config)
            .err()
            .expect("a detector path is required with a recognizer path");

        assert!(error.to_string().contains("model.detector_path is required"));
    }
}
