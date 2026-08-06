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

impl Backend {
    /// The serialized wire name, for diagnostics and error messages.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ort => "ort",
            Self::Tract => "tract",
            Self::Candle => "candle",
        }
    }
}

/// Which hardware accelerator the inference backend should run the graph on.
///
/// This is deliberately backend-neutral vocabulary: `ort` maps it onto an ONNX
/// Runtime execution provider, while `tract` and `candle` are CPU-only and answer
/// the same field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Accelerator {
    /// Run on the CPU.
    ///
    /// The default. The same ONNX graph produces different numeric output on
    /// different accelerators, and sceptre's published parity figures are CPU
    /// figures; defaulting to anything else would silently change a user's
    /// results on upgrade. Opting into an accelerator is therefore explicit.
    #[default]
    Cpu,
    /// Use the best accelerator available on this platform, falling back to CPU.
    Auto,
    /// Apple CoreML (macOS and iOS).
    #[serde(rename = "coreml")]
    CoreMl,
    /// Microsoft DirectML (Windows).
    #[serde(rename = "directml")]
    DirectMl,
    /// NVIDIA CUDA.
    Cuda,
}

impl Accelerator {
    /// The serialized wire name, for diagnostics and error messages.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Auto => "auto",
            Self::CoreMl => "coreml",
            Self::DirectMl => "directml",
            Self::Cuda => "cuda",
        }
    }

    /// Whether this selection can only ever resolve to the CPU.
    ///
    /// [`Accelerator::Auto`] counts: on a CPU-only backend the best available
    /// device *is* the CPU, so `Auto` resolves there without an error.
    pub(crate) const fn is_cpu_only(self) -> bool {
        matches!(self, Self::Cpu | Self::Auto)
    }
}

/// Model selection and provisioning configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelConfig {
    /// Recognition languages to load.
    pub languages: Vec<Language>,
    /// Inference backend.
    pub backend: Backend,
    /// Hardware accelerator the backend should run on.
    ///
    /// Only [`Backend::Ort`] can run on a non-CPU accelerator; configuring one on
    /// another backend is rejected by validation.
    pub accelerator: Accelerator,
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
            accelerator: Accelerator::default(),
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
            (Some(_), None) => {
                return Err(OcrError::config(
                    "model.recognizer_path is required when model.detector_path is configured",
                ));
            }
            (None, Some(_)) => {
                return Err(OcrError::config(
                    "model.detector_path is required when model.recognizer_path is configured",
                ));
            }
            _ => {}
        }
        self.validate_accelerator()
    }

    /// Reject a hardware accelerator on a backend that can only run on the CPU.
    fn validate_accelerator(&self) -> Result<()> {
        if self.accelerator.is_cpu_only() || self.backend == Backend::Ort {
            return Ok(());
        }
        Err(OcrError::config(format!(
            "model.accelerator = \"{}\" requires model.backend = \"ort\"; the \"{}\" backend is CPU-only",
            self.accelerator.as_str(),
            self.backend.as_str()
        )))
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

    #[test]
    fn should_default_the_accelerator_to_cpu() {
        assert_eq!(ModelConfig::default().accelerator, Accelerator::Cpu);
        assert_eq!(Accelerator::default(), Accelerator::Cpu);
    }

    #[test]
    fn should_round_trip_every_accelerator_wire_name() {
        let cases = [
            (Accelerator::Cpu, "cpu"),
            (Accelerator::Auto, "auto"),
            (Accelerator::CoreMl, "coreml"),
            (Accelerator::DirectMl, "directml"),
            (Accelerator::Cuda, "cuda"),
        ];
        for (accelerator, wire) in cases {
            let encoded = serde_json::to_string(&accelerator).expect("serialize the accelerator");
            assert_eq!(
                encoded,
                format!("\"{wire}\""),
                "unexpected wire name for {accelerator:?}"
            );
            assert_eq!(accelerator.as_str(), wire);
            let decoded: Accelerator = serde_json::from_str(&encoded).expect("deserialize the accelerator");
            assert_eq!(decoded, accelerator);
        }
    }

    #[test]
    fn should_reject_a_non_cpu_accelerator_when_the_backend_is_not_ort() {
        let config = ModelConfig {
            backend: Backend::Tract,
            accelerator: Accelerator::CoreMl,
            ..ModelConfig::default()
        };

        let error = config.validate().expect_err("coreml on tract must be rejected");

        assert!(matches!(error, OcrError::Config { .. }), "expected a config error");
        let message = error.to_string();
        assert!(
            message.contains("coreml"),
            "message must name the accelerator: {message}"
        );
        assert!(message.contains("tract"), "message must name the backend: {message}");
    }

    #[test]
    fn should_accept_cpu_and_auto_accelerators_on_a_cpu_only_backend() {
        for accelerator in [Accelerator::Cpu, Accelerator::Auto] {
            let config = ModelConfig {
                backend: Backend::Tract,
                accelerator,
                ..ModelConfig::default()
            };
            config.validate().expect("cpu-only selections are valid on tract");
        }
    }

    #[test]
    fn should_accept_every_accelerator_on_the_ort_backend() {
        for accelerator in [
            Accelerator::Cpu,
            Accelerator::Auto,
            Accelerator::CoreMl,
            Accelerator::DirectMl,
            Accelerator::Cuda,
        ] {
            let config = ModelConfig {
                backend: Backend::Ort,
                accelerator,
                ..ModelConfig::default()
            };
            config.validate().expect("ort accepts every accelerator at config time");
        }
    }
}
