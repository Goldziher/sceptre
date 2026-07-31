//! Model selection, inference backend, and cache location.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

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
    /// Override for the model cache directory (default `~/.cache/sceptre`).
    pub cache_dir: Option<PathBuf>,
    /// Override for the Hugging Face registry owner hosting the ONNX exports.
    ///
    /// `None` (the default) uses the upstream `itextresearch` org. Setting it
    /// swaps only the owner segment of every model repo id, so identical exports
    /// can be served from a mirror or private account without any code change
    /// (the `itext-EasyOCR-<model>` repo name is preserved). See ADR 0011; the
    /// value is validated to a safe owner form before it reaches cache paths.
    pub registry_owner: Option<String>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            languages: vec![Language::English],
            backend: Backend::default(),
            cache_dir: None,
            registry_owner: None,
        }
    }
}
