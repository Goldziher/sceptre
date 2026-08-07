//! Execution-provider selection for the `ort` backend.
//!
//! Maps the backend-neutral [`Accelerator`] onto ONNX Runtime execution providers.
//! Per the `backend-seam` decision this vocabulary stays inside `inference::`; the
//! rest of the crate only ever names an [`Accelerator`].
//!
//! Registration is deliberately loud. `ort`'s [`ExecutionProviderDispatch`] defaults
//! to `fail_silently`, which would let a session asked for CoreML quietly run on the
//! CPU and be reported as CoreML — the exact defect this module exists to prevent —
//! so every registration here is `error_on_failure`. An explicit selection that
//! cannot register is an error; [`Accelerator::Auto`] instead walks its candidate
//! list, logs the failure, and reports the accelerator that actually registered.

use ort::ep::ExecutionProviderDispatch;
use ort::session::builder::SessionBuilder;

use crate::config::Accelerator;
use crate::error::{OcrError, Result};

/// A candidate execution provider together with its availability answer.
struct Provider {
    /// ONNX Runtime's own identifier, e.g. `CoreMLExecutionProvider`.
    name: &'static str,
    /// Whether the linked ONNX Runtime was *compiled with* this provider.
    ///
    /// This is not a promise that the provider will run the graph's nodes; it only
    /// yields a better error message than a bare registration failure.
    available: bool,
    dispatch: ExecutionProviderDispatch,
}

/// Register the execution provider for `accelerator` on `builder`.
///
/// Returns the builder together with the accelerator that actually registered,
/// which for [`Accelerator::Auto`] may be [`Accelerator::Cpu`] when no candidate
/// could be registered.
///
/// Call this before setting the graph optimization level: ONNX Runtime applies
/// EP-aware layout transforms during optimization, so the providers must already
/// be attached to the session options.
pub(crate) fn apply_accelerator(
    builder: SessionBuilder,
    accelerator: Accelerator,
) -> Result<(SessionBuilder, Accelerator)> {
    match accelerator {
        Accelerator::Cpu => Ok((builder, Accelerator::Cpu)),
        Accelerator::Auto => Ok(register_preferred(builder)),
        explicit => register_explicit(builder, explicit),
    }
}

/// Walk the platform's preferred accelerators, keeping the first that registers.
///
/// Falls back to ONNX Runtime's implicit CPU provider when none register.
fn register_preferred(builder: SessionBuilder) -> (SessionBuilder, Accelerator) {
    let mut builder = builder;
    for &candidate in preferred_accelerators() {
        let Some(provider) = provider(candidate) else {
            continue;
        };
        if !provider.available {
            tracing::debug!(
                accelerator = candidate.as_str(),
                provider = provider.name,
                "skipping an execution provider the linked ONNX Runtime was not built with"
            );
            continue;
        }
        match builder.with_execution_providers([provider.dispatch.error_on_failure()]) {
            Ok(configured) => {
                tracing::info!(
                    accelerator = candidate.as_str(),
                    provider = provider.name,
                    "registered execution provider"
                );
                return (configured, candidate);
            }
            Err(error) => {
                tracing::warn!(
                    accelerator = candidate.as_str(),
                    provider = provider.name,
                    %error,
                    "execution provider could not be registered; trying the next candidate"
                );
                builder = error.recover();
            }
        }
    }
    tracing::info!(
        accelerator = "cpu",
        "no accelerator registered; using the CPU execution provider"
    );
    (builder, Accelerator::Cpu)
}

/// Register a user-requested accelerator, failing loudly if it cannot be used.
fn register_explicit(builder: SessionBuilder, accelerator: Accelerator) -> Result<(SessionBuilder, Accelerator)> {
    let Some(feature) = cargo_feature(accelerator) else {
        return Err(OcrError::config(format!(
            "the `ort` backend has no execution provider for accelerator `{}`",
            accelerator.as_str()
        )));
    };
    let Some(provider) = provider(accelerator) else {
        return Err(OcrError::config(format!(
            "accelerator `{}` is not compiled into this build of sceptre; rebuild with the `{feature}` cargo feature",
            accelerator.as_str(),
        )));
    };
    if !provider.available {
        return Err(OcrError::config(format!(
            "accelerator `{}` is unavailable: the linked ONNX Runtime was built without `{}`",
            accelerator.as_str(),
            provider.name
        )));
    }
    match builder.with_execution_providers([provider.dispatch.error_on_failure()]) {
        Ok(configured) => {
            tracing::info!(
                accelerator = accelerator.as_str(),
                provider = provider.name,
                "registered execution provider"
            );
            Ok((configured, accelerator))
        }
        Err(error) => Err(OcrError::inference(format!(
            "ONNX Runtime backend failed to register the `{}` execution provider for accelerator `{}`: {error}",
            provider.name,
            accelerator.as_str()
        ))),
    }
}

/// The accelerators [`Accelerator::Auto`] tries, most preferred first.
fn preferred_accelerators() -> &'static [Accelerator] {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        &[Accelerator::CoreMl]
    }
    #[cfg(target_os = "windows")]
    {
        &[Accelerator::DirectMl, Accelerator::Cuda]
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios", target_os = "windows")))]
    {
        &[Accelerator::Cuda]
    }
}

/// The sceptre cargo feature that compiles in support for `accelerator`.
///
/// `None` for the selections that have no execution provider at all: the CPU ones,
/// which ONNX Runtime serves implicitly, and [`Accelerator::Metal`], which belongs to
/// the `candle` backend and is refused here rather than blamed on a missing feature.
fn cargo_feature(accelerator: Accelerator) -> Option<&'static str> {
    match accelerator {
        Accelerator::Cpu | Accelerator::Auto | Accelerator::Metal => None,
        Accelerator::CoreMl => Some("ort-coreml"),
        Accelerator::DirectMl => Some("ort-directml"),
        Accelerator::Cuda => Some("ort-cuda"),
    }
}

/// The execution provider for `accelerator`, or `None` when it is not compiled in.
fn provider(accelerator: Accelerator) -> Option<Provider> {
    match accelerator {
        Accelerator::Cpu | Accelerator::Auto | Accelerator::Metal => None,
        Accelerator::CoreMl => coreml_provider(),
        Accelerator::DirectMl => directml_provider(),
        Accelerator::Cuda => cuda_provider(),
    }
}

#[cfg(feature = "ort-coreml")]
fn coreml_provider() -> Option<Provider> {
    use ort::ep::ExecutionProvider;
    let provider = ort::ep::CoreML::default();
    Some(Provider {
        name: provider.name(),
        available: provider.is_available().unwrap_or(false),
        dispatch: provider.build(),
    })
}

#[cfg(not(feature = "ort-coreml"))]
fn coreml_provider() -> Option<Provider> {
    None
}

#[cfg(feature = "ort-directml")]
fn directml_provider() -> Option<Provider> {
    use ort::ep::ExecutionProvider;
    let provider = ort::ep::DirectML::default();
    Some(Provider {
        name: provider.name(),
        available: provider.is_available().unwrap_or(false),
        dispatch: provider.build(),
    })
}

#[cfg(not(feature = "ort-directml"))]
fn directml_provider() -> Option<Provider> {
    None
}

#[cfg(feature = "ort-cuda")]
fn cuda_provider() -> Option<Provider> {
    use ort::ep::ExecutionProvider;
    let provider = ort::ep::CUDA::default();
    Some(Provider {
        name: provider.name(),
        available: provider.is_available().unwrap_or(false),
        dispatch: provider.build(),
    })
}

#[cfg(not(feature = "ort-cuda"))]
fn cuda_provider() -> Option<Provider> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_and_auto_have_no_execution_provider_of_their_own() {
        assert!(provider(Accelerator::Cpu).is_none());
        assert!(provider(Accelerator::Auto).is_none());
    }

    #[test]
    fn every_execution_provider_names_the_cargo_feature_that_enables_it() {
        assert_eq!(cargo_feature(Accelerator::CoreMl), Some("ort-coreml"));
        assert_eq!(cargo_feature(Accelerator::DirectMl), Some("ort-directml"));
        assert_eq!(cargo_feature(Accelerator::Cuda), Some("ort-cuda"));
    }

    /// Metal is a candle device, so no `ort-*` feature can ever enable it here.
    #[test]
    fn metal_has_no_ort_execution_provider() {
        assert_eq!(cargo_feature(Accelerator::Metal), None);
        assert!(provider(Accelerator::Metal).is_none());
    }

    #[test]
    fn preferred_accelerators_never_list_cpu_or_auto() {
        let preferred = preferred_accelerators();
        assert!(!preferred.is_empty(), "every platform needs at least one candidate");
        assert!(
            preferred.iter().all(|candidate| !candidate.is_cpu_only()),
            "the CPU fallback is implicit, not a candidate: {preferred:?}"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_prefers_coreml() {
        assert_eq!(preferred_accelerators(), &[Accelerator::CoreMl]);
    }

    /// A requested accelerator whose cargo feature is off must fail loudly, and the
    /// message must name the feature to rebuild with.
    #[cfg(all(feature = "ort-bundled", not(feature = "ort-cuda")))]
    #[test]
    fn should_reject_an_accelerator_that_is_not_compiled_in() {
        let builder = ort::session::Session::builder().expect("build session options");

        // `SessionBuilder` is not `Debug`, so unwrap the error by hand. ~keep
        let Err(error) = apply_accelerator(builder, Accelerator::Cuda) else {
            panic!("cuda is not compiled in and must be rejected");
        };

        let message = error.to_string();
        assert!(message.contains("cuda"), "message must name the accelerator: {message}");
        assert!(message.contains("ort-cuda"), "message must name the feature: {message}");
    }

    /// The bundled `aarch64-apple-darwin` runtime is CoreML-enabled by construction.
    #[cfg(all(feature = "ort-bundled", feature = "ort-coreml", target_os = "macos"))]
    #[test]
    fn should_register_coreml_on_macos_when_compiled_in() {
        let builder = ort::session::Session::builder().expect("build session options");

        let (_builder, registered) = apply_accelerator(builder, Accelerator::CoreMl).expect("coreml must register");

        assert_eq!(registered, Accelerator::CoreMl);
    }

    /// The CPU selection must not touch execution providers at all.
    #[cfg(feature = "ort-bundled")]
    #[test]
    fn should_leave_the_cpu_selection_on_the_implicit_provider() {
        let builder = ort::session::Session::builder().expect("build session options");

        let (_builder, registered) = apply_accelerator(builder, Accelerator::Cpu).expect("cpu never fails");

        assert_eq!(registered, Accelerator::Cpu);
    }
}
