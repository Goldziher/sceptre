//! Runtime provenance: which build, which backend, which accelerator.
//!
//! Numeric OCR output depends on more than the ONNX graph — the same model can
//! produce different values on a different runtime build or execution provider.
//! [`runtime_info`] reports what is actually loaded so a parity claim can name the
//! runtime it was measured on.
//!
//! Probing lives here, inside `inference::`, because it is the only place allowed
//! to speak `ort` (see the `backend-seam` decision).

use serde::Serialize;

use crate::config::{Accelerator, Backend, ModelConfig};
use crate::error::Result;

/// The prefix ONNX Runtime uses for its release branch in the build-info string.
#[cfg(any(feature = "ort", test))]
const RELEASE_BRANCH_PREFIX: &str = "rel-";
/// Number of dot-separated segments in an ONNX Runtime release version.
#[cfg(any(feature = "ort", test))]
const VERSION_SEGMENTS: usize = 3;

/// A description of the runtime a [`Reader`](crate::Reader) would execute on.
#[derive(Debug, Clone, Serialize)]
pub struct RuntimeInfo {
    /// The sceptre crate version.
    pub sceptre_version: &'static str,
    /// Target operating system, from [`std::env::consts::OS`].
    pub os: &'static str,
    /// Target architecture, from [`std::env::consts::ARCH`].
    pub arch: &'static str,
    /// The configured inference backend.
    pub backend: Backend,
    /// The accelerator the configuration asked for.
    pub accelerator_requested: Accelerator,
    /// The accelerator that actually registered.
    ///
    /// `None` when it could not be determined — the backend is not compiled in,
    /// the native runtime could not be loaded, or the requested accelerator would
    /// fail to register.
    pub accelerator_registered: Option<Accelerator>,
    /// ONNX Runtime details, or `None` when the runtime is absent or unusable.
    pub ort: Option<OrtRuntimeInfo>,
}

/// Details of the ONNX Runtime native library backing the `ort` backend.
#[derive(Debug, Clone, Serialize)]
pub struct OrtRuntimeInfo {
    /// How the native library is provisioned: `bundled`, `dynamic`, or `system`.
    pub provisioning: &'static str,
    /// The value of `ORT_DYLIB_PATH`, when set.
    pub dylib_path: Option<String>,
    /// ONNX Runtime's raw build-info string.
    pub build_info: String,
    /// The release version parsed out of [`Self::build_info`], when it is present.
    ///
    /// `None` when the string carries no `rel-x.y.z` branch — sceptre does not
    /// substitute the compiled C API level, which is a different number.
    pub version: Option<String>,
}

/// Describe the runtime for the default model configuration.
///
/// See [`runtime_info_for`] to describe the runtime a specific configuration
/// would select.
pub fn runtime_info() -> Result<RuntimeInfo> {
    runtime_info_for(&ModelConfig::default())
}

/// Describe the runtime that `model` would execute on.
///
/// Loading the ONNX Runtime library and registering an execution provider are both
/// side effects, so this builds a throwaway session-options handle rather than
/// guessing; the handle is discarded before returning.
pub fn runtime_info_for(model: &ModelConfig) -> Result<RuntimeInfo> {
    model.validate()?;
    let (accelerator_registered, ort) = match model.backend {
        Backend::Ort => probe_ort(model.accelerator),
        // tract is CPU-only, which validation has already enforced. ~keep
        Backend::Tract => (Some(Accelerator::Cpu), None),
        Backend::Candle => (probe_candle(model.accelerator), None),
    };
    Ok(RuntimeInfo {
        sceptre_version: crate::VERSION,
        os: std::env::consts::OS,
        arch: std::env::consts::ARCH,
        backend: model.backend,
        accelerator_requested: model.accelerator,
        accelerator_registered,
        ort,
    })
}

/// How the ONNX Runtime native library is provisioned for this build.
#[cfg(any(feature = "ort", test))]
const fn ort_provisioning() -> &'static str {
    if cfg!(feature = "ort-bundled") {
        "bundled"
    } else if cfg!(feature = "ort-dynamic") {
        "dynamic"
    } else {
        "system"
    }
}

/// Probe the ONNX Runtime library and the requested accelerator.
///
/// Under `ort-dynamic` with a missing or unloadable `libonnxruntime`, `ort` panics
/// rather than returning an error, so the probe runs inside
/// [`std::panic::catch_unwind`] and reports an absent runtime instead of taking the
/// process down. A panic there still prints through the default hook; sceptre does
/// not install a global hook to suppress it.
#[cfg(feature = "ort")]
fn probe_ort(requested: Accelerator) -> (Option<Accelerator>, Option<OrtRuntimeInfo>) {
    let probed = std::panic::catch_unwind(|| {
        let builder = ort::session::Session::builder().ok()?;
        let build_info = ort::info().to_string();
        let registered = super::ort_ep::apply_accelerator(builder, requested)
            .map(|(_builder, registered)| registered)
            .ok();
        Some((registered, build_info))
    });
    match probed {
        Ok(Some((registered, build_info))) => {
            let version = parse_ort_version(&build_info);
            (
                registered,
                Some(OrtRuntimeInfo {
                    provisioning: ort_provisioning(),
                    dylib_path: std::env::var("ORT_DYLIB_PATH").ok(),
                    build_info,
                    version,
                }),
            )
        }
        Ok(None) => (None, None),
        Err(_) => {
            tracing::warn!("the ONNX Runtime native library could not be loaded; runtime details are unavailable");
            (None, None)
        }
    }
}

#[cfg(not(feature = "ort"))]
fn probe_ort(_requested: Accelerator) -> (Option<Accelerator>, Option<OrtRuntimeInfo>) {
    (None, None)
}

/// Resolve the device the candle backend would open for `requested`.
#[cfg(feature = "candle")]
fn probe_candle(requested: Accelerator) -> Option<Accelerator> {
    super::candle::probe_accelerator(requested)
}

/// Without the backend compiled in there is no device to resolve, and reporting the
/// CPU would claim a run that cannot happen.
#[cfg(not(feature = "candle"))]
fn probe_candle(_requested: Accelerator) -> Option<Accelerator> {
    None
}

/// Extract the `x.y.z` release version from an ONNX Runtime build-info string.
///
/// The string looks like `ORT Build Info: git-branch=rel-1.28.0, git-commit-id=…`.
/// Returns `None` when no `rel-` branch with three numeric segments is present —
/// a development build has no release version, and reporting the compiled C API
/// level in its place would be wrong.
#[cfg(any(feature = "ort", test))]
fn parse_ort_version(build_info: &str) -> Option<String> {
    let start = build_info.find(RELEASE_BRANCH_PREFIX)? + RELEASE_BRANCH_PREFIX.len();
    let rest = &build_info[start..];
    let end = rest
        .find(|character: char| !character.is_ascii_digit() && character != '.')
        .unwrap_or(rest.len());
    let candidate = &rest[..end];
    let segments: Vec<&str> = candidate.split('.').collect();
    let numeric = segments.len() == VERSION_SEGMENTS
        && segments
            .iter()
            .all(|segment| !segment.is_empty() && segment.bytes().all(|byte| byte.is_ascii_digit()));
    numeric.then(|| candidate.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_parse_the_release_version_from_a_build_info_string() {
        let build_info = "ORT Build Info: git-branch=rel-1.28.0, git-commit-id=26250ae, build type=Release";
        assert_eq!(parse_ort_version(build_info).as_deref(), Some("1.28.0"));
    }

    #[test]
    fn should_parse_a_release_version_at_the_end_of_the_string() {
        assert_eq!(parse_ort_version("git-branch=rel-1.19.2").as_deref(), Some("1.19.2"));
    }

    #[test]
    fn should_not_parse_a_version_from_a_non_release_branch() {
        assert_eq!(
            parse_ort_version("ORT Build Info: git-branch=main, build type=Release"),
            None
        );
    }

    #[test]
    fn should_not_parse_a_partial_or_malformed_version() {
        assert_eq!(parse_ort_version("git-branch=rel-1.28,"), None);
        assert_eq!(parse_ort_version("git-branch=rel-1.28."), None);
        assert_eq!(parse_ort_version("git-branch=rel-"), None);
        assert_eq!(parse_ort_version("git-branch=rel-1.2.3.4"), None);
    }

    #[test]
    fn should_not_report_the_c_api_level_as_the_runtime_version() {
        assert_eq!(parse_ort_version("ORT Build Info: api-18, git-branch=dev"), None);
    }

    #[test]
    fn should_report_a_cpu_only_backend_as_running_on_the_cpu() {
        let model = ModelConfig {
            backend: Backend::Tract,
            accelerator: Accelerator::Auto,
            ..ModelConfig::default()
        };

        let info = runtime_info_for(&model).expect("auto on tract is a valid configuration");

        assert_eq!(info.backend, Backend::Tract);
        assert_eq!(info.accelerator_requested, Accelerator::Auto);
        assert_eq!(info.accelerator_registered, Some(Accelerator::Cpu));
        assert!(info.ort.is_none(), "the tract backend has no ONNX Runtime details");
        assert_eq!(info.sceptre_version, crate::VERSION);
        assert_eq!(info.os, std::env::consts::OS);
        assert_eq!(info.arch, std::env::consts::ARCH);
    }

    /// The candle backend must report the device it would really open.
    ///
    /// Reporting a fixed CPU here would put "cpu" into published benchmark provenance
    /// for a run that happened on the GPU.
    #[cfg(feature = "candle")]
    #[test]
    fn should_report_the_device_the_candle_backend_would_open() {
        let model = ModelConfig {
            backend: Backend::Candle,
            accelerator: Accelerator::Auto,
            ..ModelConfig::default()
        };

        let info = runtime_info_for(&model).expect("auto on candle is a valid configuration");

        let registered = info.accelerator_registered.expect("auto always resolves");
        assert_ne!(registered, Accelerator::Auto, "auto must resolve to a real device");
        assert!(
            Backend::Candle.supports(registered),
            "candle reported {registered:?}, which it cannot run on"
        );
        assert!(info.ort.is_none(), "the candle backend has no ONNX Runtime details");
    }

    /// On a build that compiled Metal in, a Mac must actually resolve to it.
    #[cfg(all(feature = "candle-metal", target_os = "macos"))]
    #[test]
    fn should_resolve_auto_to_metal_on_a_mac_that_compiled_it_in() {
        let model = ModelConfig {
            backend: Backend::Candle,
            accelerator: Accelerator::Auto,
            ..ModelConfig::default()
        };

        let info = runtime_info_for(&model).expect("auto on candle is a valid configuration");

        assert_eq!(info.accelerator_registered, Some(Accelerator::Metal));
    }

    /// A backend that is not compiled in has no device, and claiming the CPU would
    /// promise a run that cannot happen.
    #[cfg(not(feature = "candle"))]
    #[test]
    fn should_not_claim_a_device_for_a_backend_that_is_not_compiled_in() {
        let model = ModelConfig {
            backend: Backend::Candle,
            ..ModelConfig::default()
        };

        let info = runtime_info_for(&model).expect("cpu on candle is a valid configuration");

        assert_eq!(
            info.accelerator_registered, None,
            "an absent backend must report an undetermined accelerator"
        );
    }

    #[test]
    fn should_reject_an_invalid_configuration() {
        let model = ModelConfig {
            backend: Backend::Tract,
            accelerator: Accelerator::Cuda,
            ..ModelConfig::default()
        };

        runtime_info_for(&model).expect_err("cuda on tract must not be described as runnable");
    }

    #[test]
    fn should_serialize_runtime_info_as_json() {
        let model = ModelConfig {
            backend: Backend::Tract,
            ..ModelConfig::default()
        };
        let info = runtime_info_for(&model).expect("the default tract configuration is valid");

        let json: serde_json::Value = serde_json::to_value(&info).expect("serialize the runtime info");

        assert_eq!(json["backend"], "tract");
        assert_eq!(json["accelerator_requested"], "cpu");
        assert_eq!(json["accelerator_registered"], "cpu");
        assert_eq!(json["sceptre_version"], crate::VERSION);
        assert!(json["ort"].is_null());
    }

    #[test]
    fn should_name_the_provisioning_mode() {
        assert!(
            matches!(ort_provisioning(), "bundled" | "dynamic" | "system"),
            "unexpected provisioning mode {}",
            ort_provisioning()
        );
    }

    /// The `ort-bundled` feature ships the native library, so the probe must
    /// succeed without any model file or environment setup.
    #[cfg(feature = "ort-bundled")]
    #[test]
    fn should_report_the_bundled_onnx_runtime_build() {
        let info = runtime_info().expect("the default configuration is valid");

        assert_eq!(info.backend, Backend::Ort);
        assert_eq!(info.accelerator_requested, Accelerator::Cpu);
        assert_eq!(info.accelerator_registered, Some(Accelerator::Cpu));
        let ort = info.ort.expect("the bundled ONNX Runtime must be loadable");
        assert_eq!(ort.provisioning, "bundled");
        assert!(
            ort.build_info.contains("ORT Build Info"),
            "unexpected build info: {}",
            ort.build_info
        );
        assert_eq!(
            ort.version,
            parse_ort_version(&ort.build_info),
            "the reported version must come from the build info string"
        );
    }

    /// `Auto` must resolve to a concrete accelerator, never stay unresolved.
    #[cfg(feature = "ort-bundled")]
    #[test]
    fn should_resolve_auto_to_a_concrete_accelerator_on_ort() {
        let model = ModelConfig {
            backend: Backend::Ort,
            accelerator: Accelerator::Auto,
            ..ModelConfig::default()
        };

        let info = runtime_info_for(&model).expect("auto on ort is a valid configuration");

        let registered = info.accelerator_registered.expect("auto always resolves");
        assert_ne!(registered, Accelerator::Auto, "auto must resolve to a real device");
    }
}
