//! Pure-Rust native-tensor backend (`candle`).
//!
//! Unlike `ort` and `tract`, this backend does not interpret the ONNX graph. `candle-onnx`
//! cannot execute either of sceptre's models — it rejects the recognizer's bidirectional
//! LSTMs, CRAFT's padded `MaxPool`, and CRAFT's bilinear `Resize` — so the two networks are
//! written out by hand against `candle_nn` and the trained weights are read from the same
//! ONNX files the other backends load. No separate weight artifact exists.
//!
//! Per the `backend-seam` decision, `candle` APIs are referenced only from this module.

mod backend;
mod bilstm;
mod craft_net;
mod crnn_net;
mod onnx_proto;
mod ops;
mod weights;

pub(super) use backend::CandleBackend;

use crate::error::OcrError;

/// Build an [`OcrError::Inference`] wrapping a candle error with operation context.
pub(super) fn candle_error(operation: &str, error: candle_core::Error) -> OcrError {
    OcrError::Inference {
        message: format!("candle backend failed to {operation}"),
        source: Some(Box::new(error)),
    }
}
