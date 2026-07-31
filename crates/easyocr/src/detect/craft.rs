//! CRAFT forward pass through the inference backend.
//!
//! Runs the detector and splits its two-channel output into the region and link
//! (affinity) score maps. Channel 0 is the region score, channel 1 the link
//! score; both are emitted at half the input resolution.
//!
//! The channel axis is detected from the output shape rather than assumed:
//! standard CRAFT ONNX exports are channel-last `[1, H, W, 2]`, but some are
//! channel-first `[1, 2, H, W]`. The axis whose extent equals [`CHANNEL_EXTENT`]
//! selects the channel dimension; the remaining two axes are the `[H, W]` map.
//!
//! This relies on the spatial dimensions exceeding [`CHANNEL_EXTENT`]: CRAFT emits
//! half-resolution heat-maps whose `H` and `W` are far larger than 2 for any real
//! input (detection preprocessing pads inputs to a multiple of 32). A shape with
//! two axes of extent 2 is therefore rejected as ambiguous rather than guessed.

use ndarray::{Array2, ArrayD, Axis, Ix2};

use crate::error::{OcrError, Result};
use crate::inference::{ModelBackend, Tensor};

/// Number of channels in the CRAFT heat-map output (region + link).
const CHANNEL_EXTENT: usize = 2;
/// Channel index of the region (text) score map.
const REGION_CHANNEL: usize = 0;
/// Channel index of the link (affinity) score map.
const LINK_CHANNEL: usize = 1;
/// Batch axis of a rank-4 output; its extent must be one.
const BATCH_AXIS: usize = 0;
/// Extent required of the batch axis (a single image per forward pass).
const BATCH_EXTENT: usize = 1;
/// Rank of the output once the batch axis has been squeezed away.
const SPATIAL_RANK: usize = 3;

/// CRAFT region and link (affinity) score maps, `[H, W]` each, in `[0, 1]`.
#[derive(Debug)]
pub(super) struct HeatMaps {
    pub region: Array2<f32>,
    pub link: Array2<f32>,
}

/// Run the CRAFT forward pass and split its two-channel output into region/link maps.
/// The output layout is detected from the shape: the axis of extent 2 selects the
/// channel dim (supports `[1, H, W, 2]` channel-last and `[1, 2, H, W]` channel-first).
/// Errors via [`OcrError::inference`] if no unambiguous extent-2 channel axis exists.
pub(super) fn run_craft(backend: &dyn ModelBackend, tensor: Tensor) -> Result<HeatMaps> {
    let output = backend.run(tensor)?;
    split_channels(output)
}

/// Split a raw CRAFT output tensor into its region and link maps by locating the
/// channel axis of extent [`CHANNEL_EXTENT`].
fn split_channels(output: ArrayD<f32>) -> Result<HeatMaps> {
    let spatial = squeeze_batch(output)?;
    let channel_axis = find_channel_axis(spatial.shape())?;
    let region = extract_channel(&spatial, channel_axis, REGION_CHANNEL)?;
    let link = extract_channel(&spatial, channel_axis, LINK_CHANNEL)?;
    Ok(HeatMaps { region, link })
}

/// Reduce a rank-4 `[1, ...]` output to its rank-3 spatial form, or pass a rank-3
/// output through unchanged. Any other rank (or a batch extent other than one) errors.
fn squeeze_batch(output: ArrayD<f32>) -> Result<ArrayD<f32>> {
    match output.ndim() {
        4 => {
            let batch = output.shape()[BATCH_AXIS];
            if batch != BATCH_EXTENT {
                return Err(OcrError::inference(format!(
                    "CRAFT output batch axis must have extent {BATCH_EXTENT}, got {batch} in shape {:?}",
                    output.shape()
                )));
            }
            Ok(output.index_axis_move(Axis(BATCH_AXIS), 0))
        }
        SPATIAL_RANK => Ok(output),
        other => Err(OcrError::inference(format!(
            "CRAFT output must be rank 3 or 4, got rank {other} with shape {:?}",
            output.shape()
        ))),
    }
}

/// Locate the single axis whose extent equals [`CHANNEL_EXTENT`]. Errors when no
/// axis (or more than one axis) matches, since the channel dimension is then ambiguous.
fn find_channel_axis(shape: &[usize]) -> Result<usize> {
    let candidates: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter(|&(_, &extent)| extent == CHANNEL_EXTENT)
        .map(|(axis, _)| axis)
        .collect();
    match candidates.as_slice() {
        [only] => Ok(*only),
        [] => Err(OcrError::inference(format!(
            "CRAFT output shape {shape:?} has no channel axis of extent {CHANNEL_EXTENT}"
        ))),
        _ => Err(OcrError::inference(format!(
            "CRAFT output shape {shape:?} is ambiguous: multiple axes have extent {CHANNEL_EXTENT}"
        ))),
    }
}

/// Extract one `[H, W]` channel from the spatial tensor by fixing `channel` along
/// `channel_axis`, yielding an owned two-dimensional map.
fn extract_channel(spatial: &ArrayD<f32>, channel_axis: usize, channel: usize) -> Result<Array2<f32>> {
    spatial
        .index_axis(Axis(channel_axis), channel)
        .to_owned()
        .into_dimensionality::<Ix2>()
        .map_err(|error| OcrError::inference(format!("CRAFT channel is not a 2-D map: {error}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::IxDyn;

    /// A backend that ignores its input and returns a preset output tensor, used to
    /// exercise [`run_craft`] without a real ONNX model.
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
    fn should_split_channel_first_output_into_region_and_link() {
        // [1, 2, 3, 4]: channel 0 holds 0..12, channel 1 holds 12..24. ~keep
        let data: Vec<f32> = (0..24).map(|value| value as f32).collect();
        let output = ArrayD::from_shape_vec(IxDyn(&[1, 2, 3, 4]), data).expect("valid shape");

        let heat = split_channels(output).expect("splits channel-first output");

        assert_eq!(heat.region.dim(), (3, 4));
        assert_eq!(heat.link.dim(), (3, 4));
        assert_eq!(heat.region[[0, 0]], 0.0);
        assert_eq!(heat.region[[2, 3]], 11.0);
        assert_eq!(heat.link[[0, 0]], 12.0);
        assert_eq!(heat.link[[2, 3]], 23.0);
    }

    #[test]
    fn should_split_channel_last_output_into_region_and_link() {
        // [1, 3, 4, 2]: region is the even indices, link the odd ones. ~keep
        let data: Vec<f32> = (0..24).map(|value| value as f32).collect();
        let output = ArrayD::from_shape_vec(IxDyn(&[1, 3, 4, 2]), data).expect("valid shape");

        let heat = split_channels(output).expect("splits channel-last output");

        assert_eq!(heat.region.dim(), (3, 4));
        assert_eq!(heat.link.dim(), (3, 4));
        assert_eq!(heat.region[[0, 0]], 0.0);
        assert_eq!(heat.region[[0, 1]], 2.0);
        assert_eq!(heat.link[[0, 0]], 1.0);
        assert_eq!(heat.link[[0, 1]], 3.0);
    }

    #[test]
    fn should_error_when_channel_axis_is_ambiguous() {
        // [1, 2, 2, 4] squeezes to [2, 2, 4]: two axes have extent 2. ~keep
        let data: Vec<f32> = vec![0.0; 16];
        let output = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2, 4]), data).expect("valid shape");

        let error = split_channels(output).expect_err("ambiguous channel axis must error");

        assert!(matches!(error, OcrError::Inference { .. }));
    }

    #[test]
    fn should_error_when_no_channel_axis_has_extent_two() {
        let data: Vec<f32> = vec![0.0; 12];
        let output = ArrayD::from_shape_vec(IxDyn(&[1, 3, 4]), data).expect("valid shape");

        let error = split_channels(output).expect_err("absent channel axis must error");

        assert!(matches!(error, OcrError::Inference { .. }));
    }

    #[test]
    fn should_error_when_batch_extent_is_not_one() {
        let data: Vec<f32> = vec![0.0; 48];
        let output = ArrayD::from_shape_vec(IxDyn(&[2, 3, 4, 2]), data).expect("valid shape");

        let error = split_channels(output).expect_err("non-unit batch must error");

        assert!(matches!(error, OcrError::Inference { .. }));
    }

    #[test]
    fn should_run_backend_and_split_its_output() {
        // Channel-first [1, 2, 1, 8]: region is 0..8, link is 8..16. ~keep
        let data: Vec<f32> = (0..16).map(|value| value as f32).collect();
        let output = ArrayD::from_shape_vec(IxDyn(&[1, 2, 1, 8]), data).expect("valid shape");
        let backend = FixedBackend { output };

        let heat = run_craft(&backend, ArrayD::zeros(IxDyn(&[1, 3, 8, 8]))).expect("runs and splits");

        assert_eq!(heat.region.dim(), (1, 8));
        assert_eq!(heat.link.dim(), (1, 8));
        assert_eq!(heat.region[[0, 0]], 0.0);
        assert_eq!(heat.link[[0, 0]], 8.0);
    }
}
