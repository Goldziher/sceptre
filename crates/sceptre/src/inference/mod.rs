//! Runtime-neutral inference backend seam.
//!
//! [`ModelBackend`] abstracts over the concrete ONNX/tensor runtimes so the
//! detection and recognition pipelines never depend on a specific engine:
//!
//! - `ort` — native ONNX Runtime (desktop/server default).
//! - `tract` — pure-Rust ONNX (WASM/Android).
//! - `candle` — pure-Rust native tensors (deferred).
//!
//! [`load_backend`] selects an implementation from [`Backend`]; backends not
//! compiled in return an [`OcrError::Inference`].

use ndarray::ArrayD;

use crate::config::{Accelerator, Backend};
use crate::error::Result;

// Every backend crosses the seam through these helpers, so the module exists exactly
// when one of them is compiled in. ~keep
#[cfg(any(feature = "ort", feature = "tract", feature = "candle"))]
mod buffer;
#[cfg(feature = "candle")]
mod candle;
#[cfg(feature = "ort")]
mod ort_backend;
#[cfg(feature = "ort")]
mod ort_ep;
mod runtime;
#[cfg(feature = "tract")]
mod tract_backend;

pub use runtime::{OrtRuntimeInfo, RuntimeInfo, runtime_info, runtime_info_for};

/// A dynamically-shaped `f32` tensor exchanged with a backend.
pub type Tensor = ArrayD<f32>;

/// A loaded model that can run inference on a single input tensor.
pub trait ModelBackend: Send + Sync {
    /// Short backend name, for diagnostics.
    fn name(&self) -> &str;

    /// Run inference, mapping one input tensor to one output tensor.
    fn run(&self, input: Tensor) -> Result<Tensor>;
}

/// Which network a set of model bytes is expected to contain.
///
/// The `ort` and `tract` backends execute the ONNX graph as given and ignore this.
/// A backend that runs a hand-written forward pass instead of interpreting the graph
/// cannot recover the architecture from the bytes alone, so the caller — which always
/// knows — states it. Both engine call sites have the answer for free.
///
/// Distinct from [`ModelRole`](crate::ModelRole), which names the same split for
/// provisioning and carries the recognizer's language; the backend has no use for that.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum NetworkKind {
    /// The CRAFT text detector.
    #[default]
    Detector,
    /// A gen2 CRNN text recognizer.
    Recognizer,
}

/// Backend-neutral options for [`load_backend`].
///
/// A struct rather than positional arguments so that adding a knob does not churn
/// every call site; backends ignore the options they cannot honor.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct BackendOptions<'a> {
    /// Cap on the backend's intra-op parallelism, drawn from the shared thread
    /// budget. `0` leaves the backend's own default in place.
    pub threads: usize,
    /// Pins the model's input to a concrete shape for backends that cannot
    /// shape-infer a graph with data-dependent dynamic dimensions: the `tract`
    /// CRAFT detector requires this (see ADR 0027), while `ort` handles dynamic
    /// shapes natively and ignores it.
    pub fixed_input: Option<&'a [usize]>,
    /// Hardware accelerator to run the graph on. Only the `ort` backend can honor
    /// a non-CPU selection; the others are CPU-only.
    pub accelerator: Accelerator,
    /// Which network the bytes hold. Backends that interpret the ONNX graph ignore it.
    pub network: NetworkKind,
}

/// Load a model from ONNX bytes using the requested backend.
pub(crate) fn load_backend(
    backend: Backend,
    model_bytes: &[u8],
    options: BackendOptions<'_>,
) -> Result<Box<dyn ModelBackend>> {
    // Every argument is consumed only by feature-gated arms below. ~keep
    let _ = (
        model_bytes,
        options.threads,
        options.fixed_input,
        options.accelerator,
        options.network,
    );
    match backend {
        #[cfg(feature = "ort")]
        Backend::Ort => Ok(Box::new(ort_backend::OrtBackend::load(model_bytes, options)?)),
        #[cfg(feature = "candle")]
        Backend::Candle => Ok(Box::new(candle::CandleBackend::load(model_bytes, options)?)),
        #[cfg(feature = "tract")]
        Backend::Tract => Ok(Box::new(tract_backend::TractBackend::load(
            model_bytes,
            options.fixed_input,
        )?)),
        // Covers whichever backends this build left out. With all three compiled in the
        // arms above are exhaustive, and an unreachable catch-all is a lint error. ~keep
        #[cfg(not(all(feature = "ort", feature = "tract", feature = "candle")))]
        other => Err(crate::error::OcrError::inference(format!(
            "backend {other:?} is not compiled in (enable the matching cargo feature)"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_default_the_model_role_to_the_detector() {
        assert_eq!(BackendOptions::default().network, NetworkKind::Detector);
    }

    /// Only meaningful in a build where the named backend really is absent. ~keep
    #[cfg(not(feature = "candle"))]
    #[test]
    fn should_report_an_uncompiled_backend_by_name() {
        use crate::error::OcrError;

        let Err(error) = load_backend(Backend::Candle, &[], BackendOptions::default()) else {
            panic!("the candle backend is not compiled in, so loading it must fail");
        };
        let OcrError::Inference { message, .. } = &error else {
            panic!("expected an inference error, got {error:?}");
        };
        assert!(
            message.contains("Candle") && message.contains("not compiled in"),
            "message must name the backend and the cause: {message}"
        );
    }
}
