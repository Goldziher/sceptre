//! Core geometric and result types for the OCR pipeline.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{OcrError, Result};

/// Number of channels in an RGB8 pixel (red, green, blue).
const RGB_CHANNELS: usize = 3;

/// Corners describing a quadrilateral text region (clockwise from top-left). Shared
/// by the public [`Quad`] and the internal detector/crop DTOs so they stay in sync.
pub(crate) const QUAD_CORNERS: usize = 4;

/// A decoded, owned RGB8 image — the public input DTO for the OCR engine.
///
/// Decoupled from the `image` crate: callers hand the engine raw pixels or an
/// encoded file/byte slice, and the engine works only in terms of this owned,
/// row-major RGB8 buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct Image {
    width: u32,
    height: u32,
    rgb8: Vec<u8>,
}

impl Image {
    /// Decode an image file (any format the `image` crate supports) into RGB8.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
    }

    /// Decode encoded image bytes into RGB8.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let decoded = image::load_from_memory(bytes).map_err(|error| OcrError::Image {
            message: "failed to decode image bytes".to_string(),
            source: Some(Box::new(error)),
        })?;
        let rgb = decoded.to_rgb8();
        let (width, height) = rgb.dimensions();
        Ok(Self {
            width,
            height,
            rgb8: rgb.into_raw(),
        })
    }

    /// Wrap raw RGB8 pixels; `rgb8.len()` must equal `width * height * 3`.
    pub fn from_rgb8(width: u32, height: u32, rgb8: Vec<u8>) -> Result<Self> {
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|pixels| pixels.checked_mul(RGB_CHANNELS));
        match expected {
            Some(expected) if expected == rgb8.len() => Ok(Self { width, height, rgb8 }),
            _ => Err(OcrError::image(format!(
                "RGB8 buffer length {} does not match width {} * height {} * {} channels",
                rgb8.len(),
                width,
                height,
                RGB_CHANNELS
            ))),
        }
    }

    /// Image width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Image height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Row-major RGB8 pixels, length `width * height * 3`.
    pub fn as_rgb8(&self) -> &[u8] {
        &self.rgb8
    }
}

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
