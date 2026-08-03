//! Pure-Rust ONNX backend (`tract`) for WASM and Android targets.
//!
//! Wraps a tract [`TypedRunnableModel`] behind the runtime-neutral
//! [`ModelBackend`] seam. Per the `backend-seam` decision, `tract` APIs are
//! referenced only from this module; the rest of the crate speaks in
//! [`Tensor`]s. See `adrs/` for the backend selection rationale.
//!
//! tract pins its own (older) `ndarray`, so tensors cross this seam as a raw
//! shape plus a row-major `f32` buffer rather than as a shared `ArrayBase`.

use ndarray::{ArrayD, IxDyn};
use tract_onnx::onnx;
use tract_onnx::prelude::{
    Framework, InferenceModelExt, IntoTValue, IntoTensor, Tensor as TractTensor, TractError, TypedModel,
    TypedRunnableModel, tvec,
};

use super::{ModelBackend, Tensor};
use crate::error::{OcrError, Result};

/// Tract runnable-model wrapper.
///
/// The optimized, runnable plan executes inference through a shared `&self`
/// reference (tract builds a fresh execution state per call internally), so the
/// backend needs no interior mutability to stay `Send + Sync`.
///
/// This backend does not consume the [`ConcurrencyConfig`](crate::config::ConcurrencyConfig)
/// thread budget: tract's matmul kernels run single-threaded here (the `tract-linalg`
/// multithreading feature is not enabled), so `load` takes no thread count. The shared
/// Rayon pool remains the only source of parallelism on the tract path.
pub(crate) struct TractBackend {
    model: TypedRunnableModel<TypedModel>,
}

impl TractBackend {
    /// Build a runnable tract model from ONNX bytes.
    ///
    /// The bytes are parsed into an inference model, type-and-shape inferred and
    /// optimized, then lowered into a runnable plan. The ONNX bytes are read from
    /// memory, so no temporary file is written.
    pub(crate) fn load(model_bytes: &[u8]) -> Result<Self> {
        let inference_model = onnx()
            .model_for_read(&mut &model_bytes[..])
            .map_err(|error| tract_error("parse the ONNX model", error))?;
        let typed_model = inference_model
            .into_optimized()
            .map_err(|error| tract_error("optimize the ONNX model", error))?;
        let model = typed_model
            .into_runnable()
            .map_err(|error| tract_error("build the runnable tract plan", error))?;
        Ok(Self { model })
    }
}

impl ModelBackend for TractBackend {
    fn name(&self) -> &str {
        "tract"
    }

    fn run(&self, input: Tensor) -> Result<Tensor> {
        let (shape, data) = input_buffer(input);
        let tensor = TractTensor::from_shape(&shape, &data)
            .map_err(|error| tract_error("build the tract input tensor", error))?;
        let outputs = self
            .model
            .run(tvec!(tensor.into_tvalue()))
            .map_err(|error| tract_error("run tract inference", error))?;
        // The EasyOCR CRAFT and gen2 CRNN graphs are single-output. ~keep
        let output = outputs
            .into_iter()
            .next()
            .ok_or_else(|| OcrError::inference("tract returned no output tensor"))?;
        let output = output.into_tensor();
        let out_data = output
            .as_slice::<f32>()
            .map_err(|error| tract_error("read the tract output tensor as f32", error))?;
        array_from_parts(output.shape(), out_data.to_vec())
    }
}

/// Move a tensor's backing buffer out in row-major order, copy-free when possible.
///
/// A standard-layout tensor whose buffer starts at offset 0 and is sized to the shape
/// has that buffer taken by move (the common case, copy-free). Any other layout — a
/// nonzero offset, an over-long backing buffer, or a transposed view — is materialized
/// into a fresh contiguous buffer. Either way the returned [`Vec`] matches the tensor's
/// logical `iter().copied()` order, so inference stays bit-identical.
fn input_buffer(input: Tensor) -> (Vec<usize>, Vec<f32>) {
    let shape = input.shape().to_vec();
    let element_count: usize = shape.iter().product();
    if input.is_standard_layout() {
        let (data, offset) = input.into_raw_vec_and_offset();
        let start = offset.unwrap_or(0);
        if start == 0 && data.len() == element_count {
            return (shape, data);
        }
        return (shape, data.into_iter().skip(start).take(element_count).collect());
    }
    let (data, _) = input.as_standard_layout().into_owned().into_raw_vec_and_offset();
    (shape, data)
}

/// Rebuild an owned crate [`Tensor`] from a tract output shape and its data buffer.
///
/// The output shape is preserved exactly; a mismatch between the shape and the
/// data length is surfaced as an [`OcrError::Inference`].
fn array_from_parts(shape: &[usize], data: Vec<f32>) -> Result<Tensor> {
    let length = data.len();
    ArrayD::from_shape_vec(IxDyn(shape), data).map_err(|error| OcrError::Inference {
        message: format!("tract output shape {shape:?} does not match its element count {length}"),
        source: Some(Box::new(error)),
    })
}

/// Build an [`OcrError::Inference`] wrapping a tract error with operation context.
///
/// `TractError` is an `anyhow::Error`, which converts into a boxed [`std::error::Error`],
/// so the underlying tract failure is preserved as the error `source` rather than only
/// baked into the message — matching the sibling `ort` backend and the error-handling rule.
fn tract_error(operation: &str, error: TractError) -> OcrError {
    OcrError::Inference {
        message: format!("tract backend failed to {operation}"),
        source: Some(error.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_buffer_preserves_row_major_order_for_standard_layout() {
        let expected: Vec<f32> = (0..8).map(|value| value as f32).collect();
        let input = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2, 2]), expected.clone()).expect("build the tensor");
        let manual: Vec<f32> = input.iter().copied().collect();

        let (shape, data) = input_buffer(input);

        assert_eq!(shape, vec![1_usize, 2, 2, 2]);
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

        assert_eq!(shape, vec![3_usize, 2]);
        assert_eq!(data, manual);
    }

    #[test]
    fn array_from_parts_preserves_shape() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let array = array_from_parts(&[1, 2, 3], data.clone()).expect("shape matches element count");
        assert_eq!(array.shape(), &[1, 2, 3]);
        assert_eq!(array.iter().copied().collect::<Vec<_>>(), data);
    }

    #[test]
    fn array_from_parts_errors_on_length_mismatch() {
        let error = array_from_parts(&[2, 2], vec![1.0_f32, 2.0, 3.0]).expect_err("length mismatch must fail");
        assert!(matches!(error, OcrError::Inference { .. }));
    }

    #[test]
    fn input_buffer_round_trips_through_a_tract_tensor() {
        let expected: Vec<f32> = (0..12).map(|value| value as f32).collect();
        let input = ArrayD::from_shape_vec(IxDyn(&[2, 3, 2]), expected.clone()).expect("build the tensor");

        let (shape, data) = input_buffer(input);
        let tensor = TractTensor::from_shape(&shape, &data).expect("build the tract tensor");
        let restored =
            array_from_parts(tensor.shape(), tensor.as_slice::<f32>().expect("read as f32").to_vec()).expect("rebuild");

        assert_eq!(restored.shape(), &[2, 3, 2]);
        assert_eq!(restored.iter().copied().collect::<Vec<_>>(), expected);
    }

    /// End-to-end load-and-run over a real ONNX model.
    ///
    /// Ignored by default: it needs a model file on disk. Point
    /// `EASYOCR_TEST_ONNX` at an ONNX model and optionally set `EASYOCR_TEST_SHAPE`
    /// to a comma-separated input shape. The default `[1, 3, 64, 64]` suits a CRAFT
    /// detector; for a gen2 recognizer use `EASYOCR_TEST_SHAPE=1,1,64,200`. This is
    /// the quickest way to check a fresh first-party export loads under tract's
    /// `into_optimized()` (see ADR 0025).
    #[test]
    #[ignore = "requires a model file on disk (set EASYOCR_TEST_ONNX)"]
    fn load_and_run_over_real_model() {
        let model_path = std::env::var("EASYOCR_TEST_ONNX").expect("set EASYOCR_TEST_ONNX to an ONNX model path");
        let model_bytes = std::fs::read(&model_path).expect("read the model file");
        let backend = TractBackend::load(&model_bytes).expect("load the ONNX model");
        assert_eq!(backend.name(), "tract");

        let dims: Vec<usize> = std::env::var("EASYOCR_TEST_SHAPE")
            .ok()
            .map(|value| {
                value
                    .split(',')
                    .map(|part| part.trim().parse().expect("usize dim"))
                    .collect()
            })
            .unwrap_or_else(|| vec![1, 3, 64, 64]);
        let input = ArrayD::from_elem(IxDyn(&dims), 1.0_f32);
        let output = backend.run(input).expect("run inference");

        assert!(output.ndim() >= 2, "expected a multi-dimensional output");
        assert!(!output.is_empty(), "expected a non-empty output");
        assert!(
            output.iter().all(|value| value.is_finite()),
            "expected all outputs to be finite"
        );
    }
}
