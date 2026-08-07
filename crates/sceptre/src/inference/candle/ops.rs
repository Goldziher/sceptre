//! Tensor operations the hand-written networks need that `candle` does not provide.

use candle_core::{Result as CandleResult, Tensor};

/// 2-D max pooling with explicit padding, matching ONNX `MaxPool` semantics.
///
/// candle's `max_pool2d_with_stride` takes no padding. Padding with zeros would be wrong
/// here: CRAFT's padded pool consumes a raw convolution output with no rectifier between,
/// so its values can be negative and a fabricated `0.0` would win the maximum and brighten
/// the border. The border is filled with negative infinity instead, the identity for `max`.
pub(super) fn max_pool2d_padded(
    input: &Tensor,
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> CandleResult<Tensor> {
    let padded = pad_with_neg_infinity(input, padding)?;
    padded.max_pool2d_with_stride(kernel, stride)
}

/// Surround the spatial axes of an `[N, C, H, W]` tensor with negative infinity.
fn pad_with_neg_infinity(input: &Tensor, padding: (usize, usize)) -> CandleResult<Tensor> {
    let (pad_h, pad_w) = padding;
    if pad_h == 0 && pad_w == 0 {
        return Ok(input.clone());
    }
    let (batch, channels, height, width) = input.dims4()?;
    let mut padded = input.clone();
    if pad_h > 0 {
        let slab = Tensor::full(f32::NEG_INFINITY, (batch, channels, pad_h, width), input.device())?
            .to_dtype(input.dtype())?;
        padded = Tensor::cat(&[&slab, &padded, &slab], 2)?;
    }
    if pad_w > 0 {
        let height = height + 2 * pad_h;
        let slab = Tensor::full(f32::NEG_INFINITY, (batch, channels, height, pad_w), input.device())?
            .to_dtype(input.dtype())?;
        padded = Tensor::cat(&[&slab, &padded, &slab], 3)?;
    }
    Ok(padded)
}

/// Reverse an `[N, T, F]` tensor along its time axis.
///
/// `candle_nn`'s [`Direction::Backward`](candle_nn::rnn::Direction) selects the `_reverse`
/// weight names but does not run the sequence backwards — `RNN::seq` always iterates
/// forwards — so a bidirectional layer has to reverse the sequence itself.
pub(super) fn reverse_time(input: &Tensor) -> CandleResult<Tensor> {
    let steps = input.dim(1)?;
    let reversed: Vec<u32> = (0..steps as u32).rev().collect();
    let index = Tensor::from_vec(reversed, steps, input.device())?;
    input.index_select(&index, 1)
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device};

    use super::*;

    fn tensor(values: &[f32], shape: (usize, usize, usize, usize)) -> Tensor {
        Tensor::from_vec(values.to_vec(), shape, &Device::Cpu).expect("build the tensor")
    }

    #[test]
    fn should_pool_without_padding_like_a_plain_window() {
        let input = tensor(&[1.0, 2.0, 3.0, 4.0], (1, 1, 2, 2));

        let pooled = max_pool2d_padded(&input, (2, 2), (2, 2), (0, 0)).expect("pool");

        assert_eq!(pooled.dims(), &[1, 1, 1, 1]);
        assert_eq!(pooled.flatten_all().unwrap().to_vec1::<f32>().unwrap(), vec![4.0]);
    }

    /// The regression that zero-padding would pass and correct semantics must not.
    ///
    /// Every value is negative, so a zero-filled border would become the maximum of every
    /// window touching the edge and the result would be all zeros.
    #[test]
    fn should_pad_with_negative_infinity_rather_than_zero() {
        let input = tensor(&[-1.0, -2.0, -3.0, -4.0], (1, 1, 2, 2));

        let pooled = max_pool2d_padded(&input, (3, 3), (1, 1), (1, 1)).expect("pool");
        let values = pooled.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        assert_eq!(pooled.dims(), &[1, 1, 2, 2]);
        assert!(
            values.iter().all(|value| *value < 0.0),
            "zero padding would have produced non-negative maxima: {values:?}"
        );
        assert_eq!(values, vec![-1.0, -1.0, -1.0, -1.0]);
    }

    #[test]
    fn should_keep_the_shape_when_padding_a_three_by_three_window() {
        let input = Tensor::zeros((2, 3, 5, 7), DType::F32, &Device::Cpu).expect("zeros");

        let pooled = max_pool2d_padded(&input, (3, 3), (1, 1), (1, 1)).expect("pool");

        assert_eq!(pooled.dims(), &[2, 3, 5, 7]);
    }

    /// Ground truth for a 3x3 stride-1 pad-1 window, taken from ONNX Runtime itself.
    ///
    /// Pins candle's pooling semantics to the runtime the pipeline is validated against,
    /// so a future candle release that changes them fails here rather than in OCR output.
    #[test]
    fn should_match_onnx_runtime_for_a_padded_pool() {
        let values: Vec<f32> = (0..16).map(|value| value as f32 - 7.5).collect();
        let input = tensor(&values, (1, 1, 4, 4));

        let pooled = max_pool2d_padded(&input, (3, 3), (1, 1), (1, 1)).expect("pool");

        assert_eq!(
            pooled.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![
                -2.5, -1.5, -0.5, -0.5, // ~keep
                1.5, 2.5, 3.5, 3.5, // ~keep
                5.5, 6.5, 7.5, 7.5, // ~keep
                5.5, 6.5, 7.5, 7.5,
            ]
        );
    }

    /// Ground truth for a 2x bilinear upsample under ONNX `half_pixel`, from ONNX Runtime.
    ///
    /// `align_corners = false` is candle's spelling of that coordinate transform; this
    /// pins the equivalence, including the leading-edge clamp where the two could differ.
    #[test]
    fn should_match_onnx_runtime_for_a_half_pixel_bilinear_upsample() {
        let values: Vec<f32> = (0..16).map(|value| value as f32 - 7.5).collect();
        let input = tensor(&values, (1, 1, 4, 4));

        let upsampled = input.upsample_bilinear2d(8, 8, false).expect("upsample");

        let expected: Vec<f32> = vec![
            -7.5, -7.25, -6.75, -6.25, -5.75, -5.25, -4.75, -4.5, // ~keep
            -6.5, -6.25, -5.75, -5.25, -4.75, -4.25, -3.75, -3.5, // ~keep
            -4.5, -4.25, -3.75, -3.25, -2.75, -2.25, -1.75, -1.5, // ~keep
            -2.5, -2.25, -1.75, -1.25, -0.75, -0.25, 0.25, 0.5, // ~keep
            -0.5, -0.25, 0.25, 0.75, 1.25, 1.75, 2.25, 2.5, // ~keep
            1.5, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.5, // ~keep
            3.5, 3.75, 4.25, 4.75, 5.25, 5.75, 6.25, 6.5, // ~keep
            4.5, 4.75, 5.25, 5.75, 6.25, 6.75, 7.25, 7.5,
        ];
        let actual = upsampled.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (index, (left, right)) in expected.iter().zip(actual.iter()).enumerate() {
            assert!(
                (left - right).abs() < 1e-5,
                "element {index}: expected {left}, got {right}"
            );
        }
    }

    #[test]
    fn should_reverse_the_time_axis() {
        let input = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], (1, 3, 2), &Device::Cpu)
            .expect("build the tensor");

        let reversed = reverse_time(&input).expect("reverse");

        assert_eq!(
            reversed.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![5.0, 6.0, 3.0, 4.0, 1.0, 2.0]
        );
    }

    #[test]
    fn should_restore_the_original_when_reversed_twice() {
        let input = Tensor::from_vec((0..12).map(|v| v as f32).collect::<Vec<_>>(), (2, 3, 2), &Device::Cpu)
            .expect("build the tensor");

        let restored = reverse_time(&reverse_time(&input).unwrap()).unwrap();

        assert_eq!(
            restored.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            input.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }
}
