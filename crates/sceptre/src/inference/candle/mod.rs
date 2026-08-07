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
mod device;
mod onnx_proto;
mod ops;
mod weights;

pub(super) use backend::CandleBackend;

use crate::config::Accelerator;
use crate::error::OcrError;

/// The accelerator this build would actually run on, or `None` when none can be opened.
///
/// Opening a device is a side effect, so this resolves one and drops it rather than
/// guessing — the same contract as the `ort` probe, and for the same reason: published
/// provenance must name what ran, not what was asked for.
pub(super) fn probe_accelerator(requested: Accelerator) -> Option<Accelerator> {
    device::select_device(requested)
        .map(|(_device, selected)| selected)
        .ok()
}

/// Build an [`OcrError::Inference`] wrapping a candle error with operation context.
pub(super) fn candle_error(operation: &str, error: candle_core::Error) -> OcrError {
    OcrError::Inference {
        message: format!("candle backend failed to {operation}"),
        source: Some(Box::new(error)),
    }
}
