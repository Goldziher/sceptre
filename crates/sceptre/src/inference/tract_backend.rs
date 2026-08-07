//! Pure-Rust ONNX backend (`tract`) for WASM and Android targets.
//!
//! Wraps a tract [`TypedRunnableModel`] behind the runtime-neutral
//! [`ModelBackend`] seam. Per the `backend-seam` decision, `tract` APIs are
//! referenced only from this module; the rest of the crate speaks in
//! [`Tensor`]s. See `adrs/` for the backend selection rationale.
//!
//! Tensors cross this backend boundary as a raw shape plus a row-major `f32`
//! buffer, keeping tract's tensor representation private to this module.

use std::sync::Arc;

use tract_onnx::onnx;
use tract_onnx::prelude::{
    DatumType, Framework, InferenceFact, InferenceModelExt, IntoRunnable, IntoTValue, IntoTensor,
    Tensor as TractTensor, TractError, TypedRunnableModel, tvec,
};

use super::buffer::{array_from_parts, input_buffer};
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
/// native Rayon pool remains the only source of parallelism on the tract path;
/// browser WASM executes sequentially.
pub(crate) struct TractBackend {
    model: Arc<TypedRunnableModel>,
}

impl TractBackend {
    /// Build a runnable tract model from ONNX bytes.
    ///
    /// The bytes are parsed into an inference model, type-and-shape inferred and
    /// optimized, then lowered into a runnable plan. The ONNX bytes are read from
    /// memory, so no temporary file is written.
    ///
    /// `fixed_input` pins the model's input tensor to a concrete shape before
    /// optimization. The recognizer graph shape-infers under a dynamic batch and
    /// width, so it passes `None`. The CRAFT detector's U-net cannot: tract cannot
    /// unify the `Resize`-upsampled and skip-connection extents when H/W are
    /// symbolic, so the engine pins CRAFT to a fixed square canvas (see ADR 0027).
    pub(crate) fn load(model_bytes: &[u8], fixed_input: Option<&[usize]>) -> Result<Self> {
        let mut inference_model = onnx()
            .model_for_read(&mut &model_bytes[..])
            .map_err(|error| tract_error("parse the ONNX model", error))?;
        if let Some(shape) = fixed_input {
            let fact = InferenceFact::dt_shape(DatumType::F32, shape);
            inference_model = inference_model
                .with_input_fact(0, fact)
                .map_err(|error| tract_error("pin the tract input shape", error))?;
        }
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
            .to_plain_array_view::<f32>()
            .map_err(|error| tract_error("read the tract output tensor as f32", error))?
            .iter()
            .copied()
            .collect();
        array_from_parts("tract", output.shape(), out_data)
    }
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
    use ndarray::{ArrayD, IxDyn};

    use super::*;

    #[test]
    fn input_buffer_round_trips_through_a_tract_tensor() {
        let expected: Vec<f32> = (0..12).map(|value| value as f32).collect();
        let input = ArrayD::from_shape_vec(IxDyn(&[2, 3, 2]), expected.clone()).expect("build the tensor");

        let (shape, data) = input_buffer(input);
        let tensor = TractTensor::from_shape(&shape, &data).expect("build the tract tensor");
        let restored_data = tensor
            .to_plain_array_view::<f32>()
            .expect("read as f32")
            .iter()
            .copied()
            .collect();
        let restored = array_from_parts("tract", tensor.shape(), restored_data).expect("rebuild");

        assert_eq!(restored.shape(), &[2, 3, 2]);
        assert_eq!(restored.iter().copied().collect::<Vec<_>>(), expected);
    }

    /// End-to-end load-and-run over a real ONNX model.
    ///
    /// Ignored by default: it needs a model file on disk. Point
    /// `EASYOCR_TEST_ONNX` at an ONNX model and optionally set `EASYOCR_TEST_SHAPE`
    /// to a comma-separated input shape. The default `[1, 3, 64, 64]` suits a CRAFT
    /// detector; for a gen2 recognizer use `EASYOCR_TEST_SHAPE=1,1,64,200`. Set
    /// `EASYOCR_TEST_FIX_SHAPE` to the same dims to pin the input before optimize
    /// (required for the CRAFT detector under tract; see ADR 0027). This is the
    /// quickest way to check a fresh first-party export loads under tract's
    /// `into_optimized()` (see ADR 0025).
    #[test]
    #[ignore = "requires a model file on disk (set EASYOCR_TEST_ONNX)"]
    fn load_and_run_over_real_model() {
        fn parse_shape(value: &str) -> Vec<usize> {
            value
                .split(',')
                .map(|part| part.trim().parse().expect("usize dim"))
                .collect()
        }
        let model_path = std::env::var("EASYOCR_TEST_ONNX").expect("set EASYOCR_TEST_ONNX to an ONNX model path");
        let model_bytes = std::fs::read(&model_path).expect("read the model file");
        let fixed = std::env::var("EASYOCR_TEST_FIX_SHAPE")
            .ok()
            .map(|value| parse_shape(&value));
        let backend = TractBackend::load(&model_bytes, fixed.as_deref()).expect("load the ONNX model");
        assert_eq!(backend.name(), "tract");

        let dims: Vec<usize> = std::env::var("EASYOCR_TEST_SHAPE")
            .ok()
            .map(|value| parse_shape(&value))
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
