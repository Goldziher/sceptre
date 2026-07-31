//! Pure-Rust ONNX backend (`tract`) for WASM and Android targets.

use super::{ModelBackend, Tensor};
use crate::error::Result;

/// Tract typed-model wrapper.
pub(crate) struct TractBackend;

impl TractBackend {
    /// Build a runnable tract model from ONNX bytes.
    pub(crate) fn load(_model_bytes: &[u8]) -> Result<Self> {
        todo!("parse and optimize the ONNX model with tract-onnx")
    }
}

impl ModelBackend for TractBackend {
    fn name(&self) -> &str {
        "tract"
    }

    fn run(&self, _input: Tensor) -> Result<Tensor> {
        todo!("run the tract model")
    }
}
