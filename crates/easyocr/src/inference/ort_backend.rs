//! Native ONNX Runtime backend (`ort`).

use super::{ModelBackend, Tensor};
use crate::error::Result;

/// ONNX Runtime session wrapper.
pub(crate) struct OrtBackend;

impl OrtBackend {
    /// Build a session from ONNX bytes, capping intra-op threads.
    pub(crate) fn load(_model_bytes: &[u8], _threads: usize) -> Result<Self> {
        todo!("construct an ONNX Runtime session and cap its intra-op threads")
    }
}

impl ModelBackend for OrtBackend {
    fn name(&self) -> &str {
        "ort"
    }

    fn run(&self, _input: Tensor) -> Result<Tensor> {
        todo!("run the ONNX Runtime session")
    }
}
