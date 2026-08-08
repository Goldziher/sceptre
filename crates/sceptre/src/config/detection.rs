//! CRAFT text-detection configuration.
//!
//! Defaults mirror EasyOCR's `readtext` detection parameters.

use serde::{Deserialize, Serialize};

use crate::error::{OcrError, Result};

/// Parameters controlling CRAFT detection and box grouping.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DetectionConfig {
    /// Text confidence threshold (region score). EasyOCR default `0.7`.
    pub text_threshold: f32,
    /// Link confidence threshold (affinity score). EasyOCR default `0.4`.
    pub link_threshold: f32,
    /// Low-bound text score for region growth. EasyOCR default `0.4`.
    pub low_text: f32,
    /// Maximum image dimension before down-scaling. EasyOCR default `2560`.
    pub canvas_size: u32,
    /// Magnification ratio applied before detection. EasyOCR default `1.0`.
    pub mag_ratio: f32,
    /// Minimum box size (px) to keep. EasyOCR default `20`.
    pub min_size: u32,
    /// Slope threshold for splitting horizontal vs. free boxes. Default `0.1`.
    pub slope_ths: f32,
    /// Vertical-center threshold for line merging. Default `0.5`.
    pub ycenter_ths: f32,
    /// Height threshold for line merging. Default `0.5`.
    pub height_ths: f32,
    /// Width threshold for line merging. Default `0.5`.
    pub width_ths: f32,
    /// Fractional margin added around each box. Default `0.1`.
    pub add_margin: f32,
    /// Enable a whole-page orientation pre-pass: probe 0/90/180/270° rotations
    /// at [`orientation_probe_canvas_size`](Self::orientation_probe_canvas_size)
    /// and run the real detection pass on the best-scoring rotation instead of
    /// the page as given. Off by default: it is a net accuracy win on rotated
    /// pages but carries a small false-positive risk on small, dense-glyph
    /// script images (see ADR 0037) and a latency cost from the extra probe
    /// passes, so it is opt-in rather than silently changing default behavior.
    /// EasyOCR has no equivalent.
    pub detect_orientation: bool,
    /// Canvas size (px) used for each of the four orientation probe passes when
    /// [`detect_orientation`](Self::detect_orientation) is enabled. Smaller than
    /// `canvas_size` to keep the pre-pass cheap. Default `1280`.
    pub orientation_probe_canvas_size: u32,
    /// Minimum relative improvement a rotation's score must have over the
    /// unrotated (0°) score before the pre-pass switches away from it, guarding
    /// against flipping an already-upright page on a marginal score difference.
    /// Default `0.05` (5%).
    pub orientation_margin: f32,
    /// Opt-in cap on the padded detection input's area, in megapixels.
    ///
    /// Peak memory during detection tracks the padded CRAFT input's area, not its
    /// longest side (see ADR 0041), so this bounds memory directly where
    /// [`canvas_size`](Self::canvas_size) only bounds it indirectly. When set, it
    /// further constrains the resize target computed from `canvas_size` and
    /// `mag_ratio` so the padded (multiple-of-32) input area stays within the
    /// budget; the effective canvas is the minimum of what `canvas_size` and this
    /// budget each allow. `None` (the default) leaves detection input sizing
    /// exactly as `canvas_size`/`mag_ratio` compute it today.
    pub max_megapixels: Option<f32>,
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            text_threshold: 0.7,
            link_threshold: 0.4,
            low_text: 0.4,
            canvas_size: 2560,
            mag_ratio: 1.0,
            min_size: 20,
            slope_ths: 0.1,
            ycenter_ths: 0.5,
            height_ths: 0.5,
            width_ths: 0.5,
            add_margin: 0.1,
            detect_orientation: false,
            orientation_probe_canvas_size: 1280,
            orientation_margin: 0.05,
            max_megapixels: None,
        }
    }
}

impl DetectionConfig {
    /// Validate detection settings consumed by the engine.
    pub(crate) fn validate(&self) -> Result<()> {
        if let Some(max_megapixels) = self.max_megapixels {
            let valid = max_megapixels.is_finite() && max_megapixels > 0.0;
            if !valid {
                return Err(OcrError::config(format!(
                    "detection.max_megapixels must be finite and greater than 0, got {max_megapixels}"
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_default_orientation_pre_pass_to_disabled() {
        let config = DetectionConfig::default();
        assert!(!config.detect_orientation);
        assert_eq!(config.orientation_probe_canvas_size, 1280);
        assert_eq!(config.orientation_margin, 0.05);
    }

    #[test]
    fn should_default_max_megapixels_to_none() {
        let config = DetectionConfig::default();
        assert_eq!(config.max_megapixels, None);
    }

    #[test]
    fn should_accept_absent_max_megapixels() {
        let config = DetectionConfig {
            max_megapixels: None,
            ..DetectionConfig::default()
        };
        config.validate().expect("no budget is always valid");
    }

    #[test]
    fn should_accept_a_positive_finite_max_megapixels() {
        let config = DetectionConfig {
            max_megapixels: Some(4.0),
            ..DetectionConfig::default()
        };
        config.validate().expect("a positive finite budget is valid");
    }

    #[test]
    fn should_reject_zero_max_megapixels() {
        let config = DetectionConfig {
            max_megapixels: Some(0.0),
            ..DetectionConfig::default()
        };
        let error = config.validate().expect_err("zero budget must be rejected");
        assert!(error.to_string().contains("max_megapixels"));
    }

    #[test]
    fn should_reject_negative_max_megapixels() {
        let config = DetectionConfig {
            max_megapixels: Some(-1.0),
            ..DetectionConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn should_reject_nan_max_megapixels() {
        let config = DetectionConfig {
            max_megapixels: Some(f32::NAN),
            ..DetectionConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn should_reject_infinite_max_megapixels() {
        let config = DetectionConfig {
            max_megapixels: Some(f32::INFINITY),
            ..DetectionConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn should_round_trip_orientation_fields_through_json() {
        let config = DetectionConfig {
            detect_orientation: true,
            orientation_probe_canvas_size: 640,
            orientation_margin: 0.1,
            ..DetectionConfig::default()
        };

        let json = serde_json::to_string(&config).expect("serialize");
        let restored: DetectionConfig = serde_json::from_str(&json).expect("deserialize");

        assert!(restored.detect_orientation);
        assert_eq!(restored.orientation_probe_canvas_size, 640);
        assert_eq!(restored.orientation_margin, 0.1);
    }

    #[test]
    fn should_round_trip_max_megapixels_through_json() {
        let config = DetectionConfig {
            max_megapixels: Some(6.5),
            ..DetectionConfig::default()
        };

        let json = serde_json::to_string(&config).expect("serialize");
        let restored: DetectionConfig = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(restored.max_megapixels, Some(6.5));
    }

    #[test]
    fn should_reject_unknown_fields() {
        let json = r#"{"detect_orientation": true, "bogus_field": 1}"#;
        let error = serde_json::from_str::<DetectionConfig>(json).expect_err("unknown field must be rejected");
        assert!(error.to_string().contains("bogus_field"));
    }
}
