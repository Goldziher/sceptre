//! CRNN forward pass through the inference backend.
//!
//! Runs the batched `[B, 1, 64, W]` tensor through the gen2 recognizer, yielding
//! CTC logits `[B, T, num_classes]`.

use ndarray::{Array3, Ix3};

use crate::error::{OcrError, Result};
use crate::inference::{ModelBackend, Tensor};

/// Run the gen2 recognizer over a `[B, 1, 64, W]` tensor and return CTC logits
/// `[B, T, num_classes]`. Errors via [`OcrError::inference`] if the backend output
/// is not rank 3.
pub(crate) fn run_crnn(backend: &dyn ModelBackend, tensor: Tensor) -> Result<Array3<f32>> {
    let output = backend.run(tensor)?;
    let shape = output.shape().to_vec();
    output.into_dimensionality::<Ix3>().map_err(|error| {
        OcrError::inference(format!(
            "recognizer output must be rank 3 [B, T, num_classes], got shape {shape:?}: {error}"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};

    /// A backend that ignores its input and returns a preset output tensor, used to
    /// exercise [`run_crnn`] without a real ONNX model.
    struct FixedBackend {
        output: ArrayD<f32>,
    }

    impl ModelBackend for FixedBackend {
        fn name(&self) -> &str {
            "fixed"
        }

        fn run(&self, _input: Tensor) -> Result<Tensor> {
            Ok(self.output.clone())
        }
    }

    #[test]
    fn should_return_rank_three_output_as_array3_with_same_dims_and_values() {
        // [1, 3, 2]: sequential values so element positions are checkable. ~keep
        let data: Vec<f32> = (0..6).map(|value| value as f32).collect();
        let output = ArrayD::from_shape_vec(IxDyn(&[1, 3, 2]), data).expect("valid shape");
        let backend = FixedBackend { output };

        let logits = run_crnn(&backend, ArrayD::zeros(IxDyn(&[1, 1, 64, 8]))).expect("rank-3 output");

        assert_eq!(logits.dim(), (1, 3, 2));
        assert_eq!(logits[[0, 0, 0]], 0.0);
        assert_eq!(logits[[0, 2, 1]], 5.0);
    }

    #[test]
    fn should_error_when_output_is_rank_two() {
        let data: Vec<f32> = vec![0.0; 6];
        let output = ArrayD::from_shape_vec(IxDyn(&[3, 2]), data).expect("valid shape");
        let backend = FixedBackend { output };

        let error = run_crnn(&backend, ArrayD::zeros(IxDyn(&[1, 1, 64, 8]))).expect_err("rank-2 must error");

        assert!(matches!(error, OcrError::Inference { .. }));
    }

    #[test]
    fn should_error_when_output_is_rank_four() {
        let data: Vec<f32> = vec![0.0; 24];
        let output = ArrayD::from_shape_vec(IxDyn(&[1, 2, 3, 4]), data).expect("valid shape");
        let backend = FixedBackend { output };

        let error = run_crnn(&backend, ArrayD::zeros(IxDyn(&[1, 1, 64, 8]))).expect_err("rank-4 must error");

        assert!(matches!(error, OcrError::Inference { .. }));
    }
}
