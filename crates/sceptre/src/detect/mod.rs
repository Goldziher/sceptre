//! CRAFT text detection.
//!
//! Stages: [`preprocess`] (aspect-ratio resize + mean/variance normalization) →
//! [`craft`] (run the detector to region/link heat-maps) → [`postprocess`]
//! (threshold + connected components → boxes) → [`group`] (merge boxes into
//! lines, split horizontal vs. rotated).
//!
//! The detector seam and its stage DTOs live in [`detector`]; the individual
//! stages are module-private helpers wired together by [`detector::CraftDetector`].

mod craft;
mod detector;
mod group;
mod postprocess;
mod preprocess;

// The detector seam surface for the engine; unreferenced while its pipeline body is a todo. ~keep
#[allow(unused_imports)]
pub(crate) use detector::{CraftDetector, DetectedRegion, DetectedRegions, DetectorInput, TextDetector};
