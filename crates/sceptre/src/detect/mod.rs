//! CRAFT text detection.
//!
//! Stages: [`preprocess`] (aspect-ratio resize + mean/variance normalization) →
//! [`craft`] (run the detector to region/link heat-maps) → [`postprocess`]
//! (threshold + connected components → boxes) → [`group`] (merge boxes into
//! lines, split horizontal vs. rotated).
//!
//! The detector seam and its stage DTOs live in [`detector`]; the individual
//! stages are module-private helpers wired together by [`detector::CraftDetector`].
//!
//! [`orientation`] is `pub(crate)`: the whole-page rotation decision has moved up
//! to the engine (see `engine::sceptre_engine`), so both detection and recognition
//! run in the same rotated frame, and only the final reported quads are mapped
//! back to the caller's original frame.

mod craft;
mod detector;
mod group;
pub(crate) mod orientation;
mod postprocess;
mod preprocess;

// DetectedRegion is only referenced from unit tests; the rest feed the engine pipeline. ~keep
#[allow(unused_imports)]
pub(crate) use detector::{CraftDetector, DetectedRegion, DetectedRegions, DetectorInput, TextDetector};

/// Run the detection preprocess stage and return only its normalized tensor.
///
/// A crate-internal shim over the `pub(super)` stage so the `bench` seam can
/// drive it without widening the public API or naming crate-private DTOs; the
/// stage's private `Prepared` wrapper never crosses this boundary (see ADR 0015).
#[cfg(feature = "bench")]
pub(crate) fn bench_preprocess(
    image: &crate::types::Image,
    canvas_size: u32,
    mag_ratio: f32,
) -> crate::error::Result<crate::inference::Tensor> {
    Ok(preprocess::prepare(image, canvas_size, mag_ratio)?.tensor)
}

/// Reference-implementation twin of [`bench_preprocess`] driving the pre-optimization
/// channel-strided `get_pixel` normalize, for the A/B benchmark baseline (ADR 0019).
#[cfg(feature = "bench")]
pub(crate) fn bench_preprocess_reference(
    image: &crate::types::Image,
    canvas_size: u32,
    mag_ratio: f32,
) -> crate::error::Result<crate::inference::Tensor> {
    Ok(preprocess::prepare_reference(image, canvas_size, mag_ratio)?.tensor)
}

/// Run the detection postprocess stage (box extraction + coordinate adjustment)
/// on heat-maps and return the adjusted boxes. Crate-internal `bench` shim.
#[cfg(feature = "bench")]
pub(crate) fn bench_postprocess(
    region: &ndarray::Array2<f32>,
    link: &ndarray::Array2<f32>,
    text_threshold: f32,
    link_threshold: f32,
    low_text: f32,
    inv_ratio: f32,
) -> crate::error::Result<Vec<[[f32; 2]; 4]>> {
    let mut boxes = postprocess::get_det_boxes(region, link, text_threshold, link_threshold, low_text)?;
    postprocess::adjust_coordinates(&mut boxes, inv_ratio);
    Ok(boxes)
}

/// Run the box-grouping stage and return its public-typed fields (horizontal
/// lines, free quads). Crate-internal `bench` shim over `group_boxes`.
#[cfg(feature = "bench")]
pub(crate) fn bench_group(
    boxes: &[[[f32; 2]; 4]],
    config: &crate::config::DetectionConfig,
) -> (Vec<[f32; 4]>, Vec<[[f32; 2]; 4]>) {
    let grouped = group::group_boxes(boxes, config);
    (grouped.horizontal, grouped.free)
}
