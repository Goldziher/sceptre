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
use crate::error::{OcrError, Result};

mod buffer;
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
}

/// Load a model from ONNX bytes using the requested backend.
pub(crate) fn load_backend(
    backend: Backend,
    model_bytes: &[u8],
    options: BackendOptions<'_>,
) -> Result<Box<dyn ModelBackend>> {
    // Every argument is consumed only by feature-gated arms below. ~keep
    let _ = (model_bytes, options.threads, options.fixed_input, options.accelerator);
    match backend {
        #[cfg(feature = "ort")]
        Backend::Ort => Ok(Box::new(ort_backend::OrtBackend::load(model_bytes, options)?)),
        #[cfg(feature = "tract")]
        Backend::Tract => Ok(Box::new(tract_backend::TractBackend::load(
            model_bytes,
            options.fixed_input,
        )?)),
        other => Err(OcrError::inference(format!(
            "backend {other:?} is not compiled in (enable the matching cargo feature)"
        ))),
    }
}
