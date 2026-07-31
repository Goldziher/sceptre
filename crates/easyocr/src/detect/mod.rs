//! CRAFT text detection.
//!
//! Stages: [`preprocess`] (aspect-ratio resize + mean/variance normalization) →
//! [`craft`] (run the detector to region/link heat-maps) → [`postprocess`]
//! (threshold + connected components → boxes) → [`group`] (merge boxes into
//! lines, split horizontal vs. rotated).

pub mod craft;
pub mod group;
pub mod postprocess;
pub mod preprocess;

use crate::config::DetectionConfig;
use crate::error::Result;
use crate::imaging::LoadedImage;
use crate::inference::ModelBackend;
use crate::types::Quad;

/// The detector's output: axis-aligned lines and rotated (free) quads.
#[derive(Debug, Clone, Default)]
pub struct Detection {
    /// Horizontal (axis-aligned) text quads.
    pub horizontal: Vec<Quad>,
    /// Free (rotated) text quads.
    pub free: Vec<Quad>,
}

/// Run detection end-to-end, returning grouped text regions.
pub fn detect(_backend: &dyn ModelBackend, _image: &LoadedImage, _config: &DetectionConfig) -> Result<Detection> {
    todo!("preprocess, run CRAFT, postprocess the heat-maps, then group boxes")
}
