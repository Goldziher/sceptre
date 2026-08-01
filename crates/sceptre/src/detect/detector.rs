//! Internal text-detection seam, its stage DTOs, and the CRAFT runner.
//!
//! The detector consumes a decoded [`Image`] and yields raw corner regions. The
//! stage boundary deliberately uses plain `[x, y]` corner arrays rather than the
//! public [`Quad`](crate::types::Quad), keeping the detector decoupled from the
//! result surface until the engine maps regions back to public types.
//!
//! [`CraftDetector`] wires the four detection stages together: preprocess →
//! CRAFT forward pass → postprocess → group.

use std::sync::Arc;

use crate::config::DetectionConfig;
use crate::error::Result;
use crate::inference::ModelBackend;
use crate::types::{Image, QUAD_CORNERS as REGION_CORNERS};

use super::group::Grouped;

/// Internal seam: turns a decoded image into candidate text regions.
pub(crate) trait TextDetector: Send + Sync {
    /// Detect text regions in `input`.
    fn detect(&self, input: &DetectorInput) -> Result<DetectedRegions>;
}

/// Internal DTO: what the detector consumes. Borrows the decoded image.
pub(crate) struct DetectorInput<'a> {
    /// The decoded image to detect text in.
    pub image: &'a Image,
}

/// Internal DTO: the detected regions produced by a [`TextDetector`].
pub(crate) struct DetectedRegions {
    /// One entry per detected region.
    pub regions: Vec<DetectedRegion>,
}

/// Internal DTO: a single detected region as raw corners.
pub(crate) struct DetectedRegion {
    /// Corner coordinates `[x, y]`, clockwise from top-left.
    pub corners: [[f32; 2]; REGION_CORNERS],
    /// Whether this region is an axis-aligned box (`true`) or a free/rotated quad
    /// (`false`). The crop stage uses this to pick an axis-aligned crop versus a
    /// perspective warp.
    pub axis_aligned: bool,
}

/// The CRAFT-based text detector.
pub(crate) struct CraftDetector {
    backend: Arc<dyn ModelBackend>,
    config: DetectionConfig,
}

impl CraftDetector {
    /// Construct a CRAFT detector from a loaded model backend and detection config.
    pub(crate) fn new(backend: Arc<dyn ModelBackend>, detection_config: DetectionConfig) -> Self {
        Self {
            backend,
            config: detection_config,
        }
    }
}

impl TextDetector for CraftDetector {
    /// Run the full CRAFT pipeline: resize/normalize, forward pass to heat-maps,
    /// threshold into boxes, scale back to image space, then group into lines.
    fn detect(&self, input: &DetectorInput) -> Result<DetectedRegions> {
        let prepared = super::preprocess::prepare(input.image, self.config.canvas_size, self.config.mag_ratio)?;
        let heat = super::craft::run_craft(self.backend.as_ref(), prepared.tensor)?;
        let mut boxes = super::postprocess::get_det_boxes(
            &heat.region,
            &heat.link,
            self.config.text_threshold,
            self.config.link_threshold,
            self.config.low_text,
        )?;
        super::postprocess::adjust_coordinates(&mut boxes, prepared.inv_ratio);
        let grouped = super::group::group_boxes(&boxes, &self.config);
        let regions = map_grouped_to_regions(grouped, self.config.min_size);
        Ok(DetectedRegions { regions })
    }
}

/// Map grouped boxes to [`DetectedRegion`]s, converting horizontal boxes to
/// clockwise corners and dropping regions no larger than `min_size`.
///
/// A region is kept only when `max(width, height) > min_size` (strict, matching
/// EasyOCR's small-box filter); `width` and `height` are the corner-extent spans.
fn map_grouped_to_regions(grouped: Grouped, min_size: u32) -> Vec<DetectedRegion> {
    let min_size = min_size as f32;
    let mut regions = Vec::with_capacity(grouped.horizontal.len() + grouped.free.len());
    for [x_min, x_max, y_min, y_max] in grouped.horizontal {
        let corners = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]];
        push_if_large_enough(&mut regions, corners, true, min_size);
    }
    for corners in grouped.free {
        push_if_large_enough(&mut regions, corners, false, min_size);
    }
    regions
}

/// Push a region only if its larger extent strictly exceeds `min_size`, else drop it.
fn push_if_large_enough(
    regions: &mut Vec<DetectedRegion>,
    corners: [[f32; 2]; REGION_CORNERS],
    axis_aligned: bool,
    min_size: f32,
) {
    if region_max_extent(&corners) > min_size {
        regions.push(DetectedRegion { corners, axis_aligned });
    }
}

/// Larger of the corner-extent width and height of a region.
fn region_max_extent(corners: &[[f32; 2]; REGION_CORNERS]) -> f32 {
    let mut min_x = f32::MAX;
    let mut max_x = f32::MIN;
    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;
    for corner in corners {
        min_x = min_x.min(corner[0]);
        max_x = max_x.max(corner[0]);
        min_y = min_y.min(corner[1]);
        max_y = max_y.max(corner[1]);
    }
    (max_x - min_x).max(max_y - min_y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::Tensor;
    use ndarray::{ArrayD, IxDyn};

    /// A backend returning a fixed CRAFT heat-map, so [`CraftDetector::detect`] can
    /// be exercised end-to-end without a real ONNX model.
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

    fn solid_image(width: u32, height: u32) -> Image {
        let pixels = vec![255u8; (width * height * 3) as usize];
        Image::from_rgb8(width, height, pixels).expect("valid rgb buffer")
    }

    #[test]
    fn should_map_horizontal_box_to_axis_aligned_corners() {
        let grouped = Grouped {
            horizontal: vec![[10.0, 50.0, 20.0, 60.0]],
            free: Vec::new(),
        };

        let regions = map_grouped_to_regions(grouped, 0);

        assert_eq!(regions.len(), 1);
        assert!(regions[0].axis_aligned);
        assert_eq!(
            regions[0].corners,
            [[10.0, 20.0], [50.0, 20.0], [50.0, 60.0], [10.0, 60.0]]
        );
    }

    #[test]
    fn should_map_free_quad_to_non_axis_aligned_region() {
        let quad = [[10.0, 12.0], [50.0, 20.0], [48.0, 60.0], [8.0, 52.0]];
        let grouped = Grouped {
            horizontal: Vec::new(),
            free: vec![quad],
        };

        let regions = map_grouped_to_regions(grouped, 0);

        assert_eq!(regions.len(), 1);
        assert!(!regions[0].axis_aligned);
        assert_eq!(regions[0].corners, quad);
    }

    #[test]
    fn should_drop_region_smaller_than_min_size() {
        // width 8, height 6 -> max extent 8 < min_size 20 -> dropped. ~keep
        let grouped = Grouped {
            horizontal: vec![[10.0, 18.0, 20.0, 26.0]],
            free: Vec::new(),
        };

        let regions = map_grouped_to_regions(grouped, 20);

        assert!(regions.is_empty());
    }

    #[test]
    fn should_drop_region_whose_extent_equals_min_size() {
        // width 20, height 6 -> max extent 20 == min_size 20 -> dropped (strict >). ~keep
        let equal = Grouped {
            horizontal: vec![[10.0, 30.0, 20.0, 26.0]],
            free: Vec::new(),
        };
        assert!(map_grouped_to_regions(equal, 20).is_empty());

        // width 21 -> max extent 21 > 20 -> kept. ~keep
        let above = Grouped {
            horizontal: vec![[10.0, 31.0, 20.0, 26.0]],
            free: Vec::new(),
        };
        assert_eq!(map_grouped_to_regions(above, 20).len(), 1);
    }

    #[test]
    fn should_run_full_pipeline_and_produce_regions() {
        // Channel-last [1, 8, 8, 2]: a solid region block (rows 1..7, cols 1..7) ~keep
        // over zero link yields one connected component past the thresholds. ~keep
        let (height, width) = (8usize, 8usize);
        let mut output = ArrayD::<f32>::zeros(IxDyn(&[1, height, width, 2]));
        for row in 1..7 {
            for col in 1..7 {
                output[[0, row, col, 0]] = 1.0;
            }
        }
        let backend = Arc::new(FixedBackend { output });
        let config = DetectionConfig {
            min_size: 0,
            ..DetectionConfig::default()
        };
        let detector = CraftDetector::new(backend, config);

        let image = solid_image(16, 16);
        let input = DetectorInput { image: &image };
        let regions = detector.detect(&input).expect("detection succeeds");

        assert!(!regions.regions.is_empty(), "expected at least one region");
    }

    /// End-to-end detection over a real CRAFT ONNX model.
    ///
    /// Ignored by default: it links and initializes the ONNX Runtime native
    /// library and needs a model file. Point `EASYOCR_TEST_CRAFT_ONNX` at a CRAFT
    /// detector (`craft_mlt_25k`); the detector runs over a small synthetic image
    /// and must return without error.
    #[cfg(feature = "ort")]
    #[test]
    #[ignore = "requires the ONNX Runtime native library and a CRAFT model file"]
    fn detect_over_real_craft_model() {
        let model_path =
            std::env::var("EASYOCR_TEST_CRAFT_ONNX").expect("set EASYOCR_TEST_CRAFT_ONNX to a CRAFT ONNX model path");
        let model_bytes = std::fs::read(&model_path).expect("read the model file");
        let backend = crate::inference::load_backend(crate::config::Backend::Ort, &model_bytes, 1)
            .expect("load the CRAFT ONNX model");
        let detector = CraftDetector::new(Arc::from(backend), DetectionConfig::default());

        let image = solid_image(64, 64);
        let input = DetectorInput { image: &image };
        let result = detector.detect(&input);

        assert!(result.is_ok(), "detection over the real model must succeed");
    }
}
