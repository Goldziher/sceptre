//! Model selection, inference backend, and cache location.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::error::{OcrError, Result};

/// A supported recognition language group (gen2 models only).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Language {
    /// English (`english_g2`).
    #[default]
    English,
    /// Latin-script languages (`latin_g2`).
    Latin,
    /// Simplified Chinese (`zh_sim_g2`).
    ChineseSimplified,
    /// Japanese (`japanese_g2`).
    Japanese,
    /// Korean (`korean_g2`).
    Korean,
    /// Cyrillic-script languages (`cyrillic_g2`).
    Cyrillic,
    /// Telugu (`telugu_g2`).
    Telugu,
    /// Kannada (`kannada_g2`).
    Kannada,
}

/// Which inference backend to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    /// Native ONNX Runtime (`ort`). Default on desktop/server.
    #[default]
    Ort,
    /// Pure-Rust ONNX (`tract`). For WASM/Android.
    Tract,
    /// Pure-Rust native-tensor backend (`candle`). Deferred.
    Candle,
}

/// Model selection and provisioning configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelConfig {
    /// Recognition languages to load.
    pub languages: Vec<Language>,
    /// Inference backend.
    pub backend: Backend,
    /// Explicit local CRAFT detector ONNX path for host-managed model assets.
    ///
    /// Must be configured together with [`Self::recognizer_path`]. When both are
    /// present, the default provider bypasses Hugging Face cache resolution.
    pub detector_path: Option<PathBuf>,
    /// Explicit local recognizer ONNX path for the configured language group.
    ///
    /// Must be configured together with [`Self::detector_path`].
    pub recognizer_path: Option<PathBuf>,
    /// Override for the Hugging Face hub cache ROOT that stores model artifacts.
    ///
    /// `None` (the default) resolves the root from the environment in Hugging
    /// Face's order: `HF_HUB_CACHE` → `HUGGINGFACE_HUB_CACHE` → `$HF_HOME/hub` →
    /// `~/.cache/huggingface/hub`. Setting it points the library, the CLI, and the
    /// tooling at one shared cache store (see ADR 0017); artifacts still live under
    /// `<root>/models--<owner>--<name>/snapshots/<rev>/<file>`.
    pub cache_dir: Option<PathBuf>,
    /// Override for the Hugging Face registry owner hosting the ONNX exports.
    ///
    /// `None` (the default) uses the first-party `sceptre-ocr` org. Setting it
    /// swaps only the owner segment of every model repo id, so identical exports
    /// can be served from a mirror or private account without any code change
    /// (the `<model>` repo name is preserved). See ADR 0011 and ADR 0025; the
    /// value is validated to a safe owner form before it reaches cache paths.
    pub registry_owner: Option<String>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            languages: vec![Language::English],
            backend: Backend::default(),
            detector_path: None,
            recognizer_path: None,
            cache_dir: None,
            registry_owner: None,
        }
    }
}

impl ModelConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        match (&self.detector_path, &self.recognizer_path) {
            (Some(_), None) => Err(OcrError::config(
                "model.recognizer_path is required when model.detector_path is configured",
            )),
            (None, Some(_)) => Err(OcrError::config(
                "model.detector_path is required when model.recognizer_path is configured",
            )),
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_accept_model_paths_only_as_a_pair() {
        let mut config = ModelConfig {
            detector_path: Some("craft.onnx".into()),
            ..ModelConfig::default()
        };
        assert!(config.validate().is_err());

        config.recognizer_path = Some("english.onnx".into());
        config.validate().expect("paired paths are valid");
    }
}
