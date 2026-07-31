//! Core geometric and result types for the OCR pipeline.

use serde::{Deserialize, Serialize};

/// A 2D point in pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point {
    /// Horizontal coordinate.
    pub x: f32,
    /// Vertical coordinate.
    pub y: f32,
}

impl Point {
    /// Construct a new point.
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

/// An axis-aligned bounding box `[x_min, y_min, x_max, y_max]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BBox {
    /// Minimum x (left).
    pub x_min: f32,
    /// Minimum y (top).
    pub y_min: f32,
    /// Maximum x (right).
    pub x_max: f32,
    /// Maximum y (bottom).
    pub y_max: f32,
}

/// A four-point quadrilateral (clockwise from top-left), used for rotated text.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Quad {
    /// The four corners, clockwise starting top-left.
    pub points: [Point; 4],
}

/// A single recognized line of text with its location and confidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextLine {
    /// The quadrilateral bounding the text region.
    pub quad: Quad,
    /// The recognized text.
    pub text: String,
    /// Recognition confidence in `[0.0, 1.0]`.
    pub confidence: f32,
}

/// The full result of an OCR run over one image.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct OcrResult {
    /// Recognized text lines, in reading order where determinable.
    pub lines: Vec<TextLine>,
}
