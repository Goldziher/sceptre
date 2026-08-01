//! Pure-Rust ONNX backend (`tract`) for WASM and Android targets.

use super::{ModelBackend, Tensor};
use crate::error::{OcrError, Result};

/// Message returned by every unimplemented tract backend entry point.
const TRACT_UNIMPLEMENTED: &str = "the tract backend is not yet implemented";

/// Tract typed-model wrapper.
pub(crate) struct TractBackend;

impl TractBackend {
    /// Build a runnable tract model from ONNX bytes.
    pub(crate) fn load(_model_bytes: &[u8]) -> Result<Self> {
        Err(OcrError::inference(TRACT_UNIMPLEMENTED))
    }
}

impl ModelBackend for TractBackend {
    fn name(&self) -> &str {
        "tract"
    }

    fn run(&self, _input: Tensor) -> Result<Tensor> {
        Err(OcrError::inference(TRACT_UNIMPLEMENTED))
    }
}
