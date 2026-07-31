//! Internal text-detection seam and its stage DTOs.
//!
//! The detector consumes a decoded [`Image`] and yields raw corner regions. The
//! stage boundary deliberately uses plain `[x, y]` corner arrays rather than the
//! public [`Quad`](crate::types::Quad), keeping the detector decoupled from the
//! result surface until the engine maps regions back to public types.

use crate::error::Result;
use crate::types::Image;

/// Number of corners describing a detected quadrilateral region.
const REGION_CORNERS: usize = 4;

/// Internal seam: turns a decoded image into candidate text regions.
// Seam method the engine invokes to run CRAFT detection. ~keep
#[allow(dead_code)]
pub(crate) trait TextDetector: Send + Sync {
    /// Detect text regions in `input`.
    fn detect(&self, input: &DetectorInput) -> Result<DetectedRegions>;
}

/// Internal DTO: what the detector consumes. Borrows the decoded image.
// Stage-boundary DTO the engine constructs when invoking the detector. ~keep
#[allow(dead_code)]
pub(crate) struct DetectorInput<'a> {
    /// The decoded image to detect text in.
    pub image: &'a Image,
}

/// Internal DTO: the detected regions produced by a [`TextDetector`].
// Stage-boundary DTO the engine maps to public quads. ~keep
#[allow(dead_code)]
pub(crate) struct DetectedRegions {
    /// One entry per detected region.
    pub regions: Vec<DetectedRegion>,
}

/// Internal DTO: a single detected region as raw corners.
// Stage-boundary DTO the engine maps to a public quad. ~keep
#[allow(dead_code)]
pub(crate) struct DetectedRegion {
    /// Corner coordinates `[x, y]`, clockwise from top-left.
    pub corners: [[f32; 2]; REGION_CORNERS],
}

/// The CRAFT-based text detector.
pub(crate) struct CraftDetector {}

impl CraftDetector {
    /// Construct a CRAFT detector.
    pub(crate) fn new() -> Self {
        Self {}
    }
}

impl TextDetector for CraftDetector {
    fn detect(&self, _input: &DetectorInput) -> Result<DetectedRegions> {
        todo!("run CRAFT detection: preprocess, infer heat-maps, threshold to regions")
    }
}
