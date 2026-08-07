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
    /// Pure-Rust native-tensor backend (`candle`), with Metal and CUDA support.
    Candle,
}

/// Every backend, for diagnostics that have to search across them.
const EVERY_BACKEND: [Backend; 3] = [Backend::Ort, Backend::Tract, Backend::Candle];

impl Backend {
    /// The serialized wire name, for diagnostics and error messages.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ort => "ort",
            Self::Tract => "tract",
            Self::Candle => "candle",
        }
    }

    /// The non-CPU accelerators this backend can run a graph on.
    ///
    /// Support is a property of the backend, not of the accelerator: `ort` reaches
    /// hardware through ONNX Runtime execution providers, while `candle` addresses
    /// devices directly. [`Accelerator::Cuda`] therefore appears on both — it names
    /// hardware — whereas [`Accelerator::CoreMl`] and [`Accelerator::Metal`] are two
    /// different frameworks over the same Apple GPU and each belongs to one backend.
    ///
    /// Compiling the support in is a separate question, answered at load time by the
    /// relevant cargo feature; this is the configuration vocabulary only.
    pub const fn hardware_accelerators(self) -> &'static [Accelerator] {
        match self {
            Self::Ort => &[Accelerator::CoreMl, Accelerator::DirectMl, Accelerator::Cuda],
            Self::Tract => &[],
            Self::Candle => &[Accelerator::Metal, Accelerator::Cuda],
        }
    }

    /// Whether this backend can honor `accelerator`.
    ///
    /// Always true for the CPU-only selections, which every backend answers.
    pub fn supports(self, accelerator: Accelerator) -> bool {
        accelerator.is_cpu_only() || self.hardware_accelerators().contains(&accelerator)
    }
}

/// Which hardware accelerator the inference backend should run the graph on.
///
/// This is deliberately backend-neutral vocabulary: `ort` maps it onto an ONNX
/// Runtime execution provider and `candle` onto a compute device, while `tract`
/// is CPU-only. Not every backend answers every value — see
/// [`Backend::hardware_accelerators`] for which pairings are valid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Accelerator {
    /// Run on the CPU.
    ///
    /// The default. The same ONNX graph produces different numeric output on
    /// different accelerators, and the only parity evidence this repository
    /// carries — the golden fixtures under `crates/sceptre/tests/data/golden/` —
    /// is CPU-generated. No accelerator has been validated against them, so
    /// defaulting to anything else would silently change a user's results on
    /// upgrade. Opting into an accelerator is therefore explicit.
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
    /// Apple Metal (macOS and iOS).
    ///
    /// Distinct from [`Self::CoreMl`]: both drive the same GPU, but through
    /// different frameworks with different numerics, and this value flows verbatim
    /// into published benchmark provenance.
    Metal,
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
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }

    /// The accelerator on `backend` that reaches the same hardware as this one.
    ///
    /// Only the Apple pair differs by name: CoreML and Metal are two frameworks over
    /// one GPU, so requesting the wrong one is a plausible mistake in either direction
    /// and deserves a better answer than "unsupported".
    fn equivalent_on(self, backend: Backend) -> Option<Self> {
        let equivalent = match self {
            Self::CoreMl => Self::Metal,
            Self::Metal => Self::CoreMl,
            _ => return None,
        };
        backend.supports(equivalent).then_some(equivalent)
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
    /// Which values a backend can honor is given by
    /// [`Backend::hardware_accelerators`]; any other pairing is rejected by
    /// validation rather than silently degraded to the CPU.
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

    /// Reject an accelerator the configured backend cannot run on.
    fn validate_accelerator(&self) -> Result<()> {
        if self.backend.supports(self.accelerator) {
            return Ok(());
        }
        Err(OcrError::config(unsupported_accelerator(
            self.backend,
            self.accelerator,
        )))
    }
}

/// Explain why `backend` cannot honor `accelerator`, and what to do instead.
///
/// The remedy matters more than the rejection: every one of these mistakes has a
/// concrete fix, either a different accelerator on this backend or a different
/// backend for this accelerator.
fn unsupported_accelerator(backend: Backend, accelerator: Accelerator) -> String {
    let supported = backend.hardware_accelerators();
    let mut message = if supported.is_empty() {
        format!(
            "model.accelerator = \"{}\" is not available: the \"{}\" backend is CPU-only",
            accelerator.as_str(),
            backend.as_str()
        )
    } else {
        let names: Vec<&str> = supported.iter().map(|supported| supported.as_str()).collect();
        format!(
            "model.accelerator = \"{}\" is not available on the \"{}\" backend, which runs on {}",
            accelerator.as_str(),
            backend.as_str(),
            names.join(" or ")
        )
    };
    if let Some(equivalent) = accelerator.equivalent_on(backend) {
        message.push_str(&format!(
            "; the same hardware is reached with model.accelerator = \"{}\"",
            equivalent.as_str()
        ));
    } else if let Some(other) = EVERY_BACKEND.iter().find(|other| other.supports(accelerator)) {
        message.push_str(&format!("; model.backend = \"{}\" runs on it", other.as_str()));
    }
    message
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
            (Accelerator::Metal, "metal"),
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
    fn should_reject_a_hardware_accelerator_on_a_cpu_only_backend() {
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
        assert!(
            message.contains("CPU-only"),
            "message must say why tract cannot honor it: {message}"
        );
    }

    #[test]
    fn should_reject_an_accelerator_that_belongs_to_another_backend() {
        let config = ModelConfig {
            backend: Backend::Candle,
            accelerator: Accelerator::DirectMl,
            ..ModelConfig::default()
        };

        let error = config.validate().expect_err("directml on candle must be rejected");

        let message = error.to_string();
        assert!(
            message.contains("directml") && message.contains("candle"),
            "message must name both sides: {message}"
        );
        assert!(
            message.contains("metal") && message.contains("cuda"),
            "message must list what candle does support: {message}"
        );
        assert!(
            message.contains("backend = \"ort\""),
            "message must point at the backend that does support directml: {message}"
        );
    }

    /// CoreML and Metal drive the same Apple GPU through different frameworks, so
    /// naming the wrong one is the mistake most likely to be made in either direction.
    #[test]
    fn should_name_the_apple_equivalent_when_the_wrong_framework_is_requested() {
        let cases = [
            (Backend::Candle, Accelerator::CoreMl, "metal"),
            (Backend::Ort, Accelerator::Metal, "coreml"),
        ];
        for (backend, accelerator, equivalent) in cases {
            let config = ModelConfig {
                backend,
                accelerator,
                ..ModelConfig::default()
            };

            let Err(error) = config.validate() else {
                panic!("{accelerator:?} on {backend:?} must be rejected");
            };

            let message = error.to_string();
            assert!(
                message.contains(equivalent),
                "on the {} backend the message must point at `{equivalent}`: {message}",
                backend.as_str()
            );
        }
    }

    #[test]
    fn should_accept_every_accelerator_the_support_table_lists() {
        for backend in [Backend::Ort, Backend::Tract, Backend::Candle] {
            for accelerator in backend.hardware_accelerators() {
                let config = ModelConfig {
                    backend,
                    accelerator: *accelerator,
                    ..ModelConfig::default()
                };
                config
                    .validate()
                    .unwrap_or_else(|error| panic!("{accelerator:?} is listed for {backend:?} but rejected: {error}"));
                assert!(backend.supports(*accelerator));
            }
        }
    }

    #[test]
    fn should_list_no_hardware_accelerator_that_is_really_the_cpu() {
        for backend in [Backend::Ort, Backend::Tract, Backend::Candle] {
            let listed = backend.hardware_accelerators();
            assert!(
                listed.iter().all(|accelerator| !accelerator.is_cpu_only()),
                "{backend:?} lists a CPU selection as hardware: {listed:?}"
            );
            assert!(
                backend.supports(Accelerator::Cpu) && backend.supports(Accelerator::Auto),
                "{backend:?} must accept the CPU-only selections"
            );
        }
    }

    #[test]
    fn should_run_candle_on_metal_and_cuda_but_not_on_the_onnx_runtime_providers() {
        assert_eq!(
            Backend::Candle.hardware_accelerators(),
            &[Accelerator::Metal, Accelerator::Cuda],
            "candle names hardware, not ONNX Runtime execution providers"
        );
        assert!(!Backend::Candle.supports(Accelerator::CoreMl));
        assert!(!Backend::Ort.supports(Accelerator::Metal));
        assert!(Backend::Tract.hardware_accelerators().is_empty());
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
    fn should_accept_every_onnx_runtime_provider_on_the_ort_backend() {
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
            config
                .validate()
                .expect("ort accepts every execution provider at config time");
        }
    }
}
