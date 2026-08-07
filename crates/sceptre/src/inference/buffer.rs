//! Tensor buffer helpers shared by every backend implementation.
//!
//! Each backend speaks its own tensor type but crosses the [`ModelBackend`](super::ModelBackend)
//! seam as a shape plus a row-major `f32` buffer. The conversion in both directions is
//! identical for every backend, so it lives here rather than being copied per backend.

use ndarray::{ArrayD, IxDyn};

use super::Tensor;
use crate::error::{OcrError, Result};

/// Move a tensor's backing buffer out in row-major order, copy-free when possible.
///
/// A standard-layout tensor whose buffer starts at offset 0 and is sized to the shape
/// has that buffer taken by move (the common case, copy-free). A standard-layout slice
/// with a nonzero offset or an over-long backing buffer, or a non-standard layout (e.g.
/// a transposed view), is copied into a fresh contiguous buffer. Either way the returned
/// [`Vec`] matches the tensor's logical `iter().copied()` order, so inference stays
/// bit-identical.
pub(super) fn input_buffer(input: Tensor) -> (Vec<usize>, Vec<f32>) {
    let shape = input.shape().to_vec();
    let element_count: usize = shape.iter().product();
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

/// Rebuild an owned crate [`Tensor`] from a backend output shape and its data buffer.
///
/// The output shape is preserved exactly; a mismatch between the shape and the data
/// length is surfaced as an [`OcrError::Inference`] naming `backend`, so the message
/// identifies which runtime produced the inconsistent output.
pub(super) fn array_from_parts(backend: &str, shape: &[usize], data: Vec<f32>) -> Result<Tensor> {
    let length = data.len();
    ArrayD::from_shape_vec(IxDyn(shape), data).map_err(|error| OcrError::Inference {
        message: format!("{backend} output shape {shape:?} does not match its element count {length}"),
        source: Some(Box::new(error)),
    })
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
    fn input_buffer_slices_standard_layout_with_nonzero_offset() {
        let base = ArrayD::from_shape_vec(IxDyn(&[3, 2]), (0..6).map(|value| value as f32).collect())
            .expect("build the tensor");
        let sliced = base.slice(ndarray::s![1.., ..]).into_owned().into_dyn();
        let manual: Vec<f32> = sliced.iter().copied().collect();

        let (shape, data) = input_buffer(sliced);

        assert_eq!(shape, vec![2_usize, 2]);
        assert_eq!(data, manual);
    }

    #[test]
    fn array_from_parts_preserves_shape() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let array = array_from_parts("test", &[1, 2, 3], data.clone()).expect("shape matches element count");
        assert_eq!(array.shape(), &[1, 2, 3]);
        assert_eq!(array.iter().copied().collect::<Vec<_>>(), data);
    }

    #[test]
    fn array_from_parts_errors_on_length_mismatch() {
        let error = array_from_parts("test", &[2, 2], vec![1.0_f32, 2.0, 3.0]).expect_err("length mismatch must fail");
        assert!(matches!(error, OcrError::Inference { .. }));
    }

    #[test]
    fn array_from_parts_names_the_backend_in_the_error() {
        let error = array_from_parts("candle", &[2, 2], vec![1.0_f32]).expect_err("length mismatch must fail");
        let OcrError::Inference { message, .. } = &error else {
            panic!("expected an inference error, got {error:?}");
        };
        assert!(message.contains("candle"), "message must name the backend: {message}");
    }
}
