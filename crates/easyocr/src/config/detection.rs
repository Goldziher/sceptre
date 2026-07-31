//! CRAFT text-detection configuration.
//!
//! Defaults mirror EasyOCR's `readtext` detection parameters.

use serde::{Deserialize, Serialize};

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
        }
    }
}
