//! Native ONNX Runtime backend (`ort`).
//!
//! Wraps an `ort` [`Session`] behind the runtime-neutral [`ModelBackend`] seam.
//! Per the `backend-seam` decision, `ort` APIs are referenced only from this
//! module; the rest of the crate speaks in [`Tensor`]s. See `adrs/` for the
//! backend selection rationale.

use std::sync::Mutex;

use ndarray::{ArrayD, IxDyn};
use ort::session::Session;
use ort::value::Tensor as OrtTensor;

use super::{ModelBackend, Tensor};
use crate::error::{OcrError, Result};

/// ONNX Runtime session wrapper.
///
/// The session is held behind a [`Mutex`] because `ort`'s `Session::run`
/// borrows the session mutably while [`ModelBackend::run`] takes `&self`; the
/// mutex serializes calls so the backend stays `Send + Sync`.
pub(crate) struct OrtBackend {
    session: Mutex<Session>,
}

impl OrtBackend {
    /// Build a session from ONNX bytes, capping intra-op threads.
    ///
    /// When `threads > 0` the session's intra-op thread pool is capped to that
    /// value; `0` leaves ONNX Runtime's own default in place. The ONNX bytes are
    /// parsed in memory, so no temporary file is written.
    pub(crate) fn load(model_bytes: &[u8], threads: usize) -> Result<Self> {
        let mut builder =
            Session::builder().map_err(|error| inference_error("create an ONNX Runtime session builder", error))?;
        if threads > 0 {
            builder = builder
                .with_intra_threads(threads)
                .map_err(|error| inference_error("configure ONNX Runtime intra-op threads", ort::Error::from(error)))?;
        }
        let session = builder
            .commit_from_memory(model_bytes)
            .map_err(|error| inference_error("load the ONNX model from memory", error))?;
        Ok(Self {
            session: Mutex::new(session),
        })
    }
}

impl ModelBackend for OrtBackend {
    fn name(&self) -> &str {
        "ort"
    }

    fn run(&self, input: Tensor) -> Result<Tensor> {
        let (shape, data) = input_buffer(input);
        let value = OrtTensor::from_array((shape, data))
            .map_err(|error| inference_error("build the ONNX Runtime input tensor", error))?;

        // A panic in a prior `run` poisons the mutex; recover the guard so one bad ~keep
        // call does not permanently brick every later inference on this backend. ~keep
        let mut session = self.session.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let outputs = session
            .run(ort::inputs![value])
            .map_err(|error| inference_error("run ONNX Runtime inference", error))?;

        // The EasyOCR CRAFT and gen2 CRNN graphs are single-output. ~keep
        let output = outputs
            .values()
            .next()
            .ok_or_else(|| OcrError::inference("ONNX Runtime returned no output tensor"))?;
        let (out_shape, out_data) = output
            .try_extract_tensor::<f32>()
            .map_err(|error| inference_error("extract the ONNX Runtime output tensor", error))?;
        array_from_output(out_shape, out_data)
    }
}

/// Move a tensor's backing buffer out in row-major order, copy-free when possible.
///
/// A standard-layout tensor whose buffer starts at offset 0 and is sized to the shape
/// has that buffer taken by move (the common case, copy-free). A standard-layout slice
/// with a nonzero offset or an over-long backing buffer, or a non-standard layout (e.g.
/// a transposed view), is copied into a fresh contiguous buffer. Either way the returned
/// [`Vec`] matches the tensor's logical `iter().copied()` order, so inference stays
/// bit-identical.
fn input_buffer(input: Tensor) -> (Vec<i64>, Vec<f32>) {
    let shape = shape_to_i64(input.shape());
    let element_count: usize = input.shape().iter().product();
    if input.is_standard_layout() {
        let (data, offset) = input.into_raw_vec_and_offset();
        let start = offset.unwrap_or(0);
        // Offset-0 buffers sized to the shape move out copy-free; a standard-layout ~keep
        // slice's logical elements are the contiguous run [start, start + count). ~keep
        if start == 0 && data.len() == element_count {
            return (shape, data);
        }
        return (shape, data.into_iter().skip(start).take(element_count).collect());
    }
    let (data, _) = input.as_standard_layout().into_owned().into_raw_vec_and_offset();
    (shape, data)
}

/// Convert an ndarray shape (`&[usize]`) to the `i64` dims `ort` expects.
fn shape_to_i64(shape: &[usize]) -> Vec<i64> {
    shape.iter().map(|&dim| dim as i64).collect()
}

/// Rebuild an owned [`Tensor`] from an `ort` output shape and its data slice.
///
/// The output shape is preserved exactly; a mismatch between the declared shape
/// and the data length is surfaced as an [`OcrError::Inference`].
fn array_from_output(dims: &[i64], data: &[f32]) -> Result<Tensor> {
    if let Some(&negative) = dims.iter().find(|&&dim| dim < 0) {
        return Err(OcrError::inference(format!(
            "ONNX Runtime returned a negative output dimension {negative} in shape {dims:?}"
        )));
    }
    let shape: Vec<usize> = dims.iter().map(|&dim| dim as usize).collect();
    ArrayD::from_shape_vec(IxDyn(&shape), data.to_vec()).map_err(|error| OcrError::Inference {
        message: format!(
            "ONNX Runtime output shape {dims:?} does not match {} elements",
            data.len()
        ),
        source: Some(Box::new(error)),
    })
}

/// Build an [`OcrError::Inference`] wrapping an `ort` error with operation context.
fn inference_error(operation: &str, source: ort::Error) -> OcrError {
    OcrError::Inference {
        message: format!("ONNX Runtime backend failed to {operation}"),
        source: Some(Box::new(source)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_to_i64_converts_dims() {
        assert_eq!(shape_to_i64(&[1, 3, 64, 128]), vec![1_i64, 3, 64, 128]);
    }

    #[test]
    fn shape_to_i64_handles_empty_shape() {
        assert_eq!(shape_to_i64(&[]), Vec::<i64>::new());
    }

    #[test]
    fn array_from_output_preserves_shape() {
        let data = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let array = array_from_output(&[1, 2, 3], &data).expect("shape matches element count");
        assert_eq!(array.shape(), &[1, 2, 3]);
        assert_eq!(array.iter().copied().collect::<Vec<_>>(), data);
    }

    #[test]
    fn array_from_output_errors_on_length_mismatch() {
        let error = array_from_output(&[2, 2], &[1.0_f32, 2.0, 3.0]).expect_err("length mismatch must fail");
        assert!(matches!(error, OcrError::Inference { .. }));
    }

    #[test]
    fn input_buffer_preserves_row_major_order_for_standard_layout() {
        let expected: Vec<f32> = (0..8).map(|value| value as f32).collect();
        let input = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2, 2]), expected.clone()).expect("build the tensor");
        let manual: Vec<f32> = input.iter().copied().collect();

        let (shape, data) = input_buffer(input);

        assert_eq!(shape, vec![1_i64, 2, 2, 2]);
        assert_eq!(data, manual);
        assert_eq!(data, expected);
    }

    #[test]
    fn input_buffer_matches_iter_order_for_non_standard_layout() {
        let base = ArrayD::from_shape_vec(IxDyn(&[2, 3]), (0..6).map(|value| value as f32).collect())
            .expect("build the tensor");
        let transposed = base.t().into_owned();
        assert!(
            !transposed.is_standard_layout(),
            "transpose must be non-standard layout"
        );
        let manual: Vec<f32> = transposed.iter().copied().collect();

        let (shape, data) = input_buffer(transposed);

        assert_eq!(shape, vec![3_i64, 2]);
        assert_eq!(data, manual);
    }

    #[test]
    fn input_buffer_slices_standard_layout_with_nonzero_offset() {
        use ndarray::{Axis, Slice};
        let mut base = ArrayD::from_shape_vec(IxDyn(&[3, 2]), (0..6).map(|value| value as f32).collect())
            .expect("build the tensor");
        base.slice_axis_inplace(Axis(0), Slice::from(1..));
        assert!(base.is_standard_layout(), "the sliced array must stay standard layout");
        let manual: Vec<f32> = base.iter().copied().collect();

        let (shape, data) = input_buffer(base);

        assert_eq!(shape, vec![2_i64, 2]);
        assert_eq!(
            data, manual,
            "must return the logical elements, not the full backing buffer"
        );
    }

    #[test]
    fn array_from_output_errors_on_negative_dimension() {
        let error = array_from_output(&[-1, 2], &[1.0_f32, 2.0]).expect_err("negative dimension must fail");
        assert!(matches!(error, OcrError::Inference { .. }));
    }

    /// End-to-end load-and-run over a real ONNX model.
    ///
    /// Ignored by default: it links and initializes the ONNX Runtime native
    /// library and needs a model file. Point `EASYOCR_TEST_ONNX` at a CRAFT
    /// detector (`craft_mlt_25k`) — the input below is a `[1, 3, 64, 64]` batch,
    /// which the detector maps to a rank-4 `[1, 2, 32, 32]` heat-map.
    #[test]
    #[ignore = "requires the ONNX Runtime native library and a model file"]
    fn load_and_run_over_real_model() {
        let model_path = std::env::var("EASYOCR_TEST_ONNX").expect("set EASYOCR_TEST_ONNX to an ONNX model path");
        let model_bytes = std::fs::read(&model_path).expect("read the model file");
        let backend = OrtBackend::load(&model_bytes, 1).expect("load the ONNX model");
        assert_eq!(backend.name(), "ort");

        let input = ArrayD::from_elem(IxDyn(&[1, 3, 64, 64]), 1.0_f32);
        let output = backend.run(input).expect("run inference");

        assert!(output.ndim() >= 2, "expected a multi-dimensional output");
        assert!(!output.is_empty(), "expected a non-empty output");
        assert!(
            output.iter().all(|value| value.is_finite()),
            "expected all outputs to be finite"
        );
    }
}
