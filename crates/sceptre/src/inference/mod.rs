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

use crate::config::Backend;
use crate::error::{OcrError, Result};

#[cfg(feature = "ort")]
mod ort_backend;
#[cfg(feature = "tract")]
mod tract_backend;

/// A dynamically-shaped `f32` tensor exchanged with a backend.
pub type Tensor = ArrayD<f32>;

/// A loaded model that can run inference on a single input tensor.
pub trait ModelBackend: Send + Sync {
    /// Short backend name, for diagnostics.
    fn name(&self) -> &str;

    /// Run inference, mapping one input tensor to one output tensor.
    fn run(&self, input: Tensor) -> Result<Tensor>;
}

/// Load a model from ONNX bytes using the requested backend.
///
/// `threads` caps the backend's intra-op parallelism where supported. `fixed_input`
/// pins the model's input to a concrete shape for backends that cannot shape-infer a
/// graph with data-dependent dynamic dimensions: the `tract` CRAFT detector requires
/// this (see ADR 0027), while `ort` handles dynamic shapes natively and ignores it.
pub fn load_backend(
    backend: Backend,
    model_bytes: &[u8],
    threads: usize,
    fixed_input: Option<&[usize]>,
) -> Result<Box<dyn ModelBackend>> {
    let _ = (model_bytes, threads, fixed_input);
    match backend {
        #[cfg(feature = "ort")]
        Backend::Ort => Ok(Box::new(ort_backend::OrtBackend::load(model_bytes, threads)?)),
        #[cfg(feature = "tract")]
        Backend::Tract => Ok(Box::new(tract_backend::TractBackend::load(model_bytes, fixed_input)?)),
        other => Err(OcrError::inference(format!(
            "backend {other:?} is not compiled in (enable the matching cargo feature)"
        ))),
    }
}
