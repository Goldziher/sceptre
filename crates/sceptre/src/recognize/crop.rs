//! Crop each detected region from the full-image grayscale.
//!
//! Reference: EasyOCR `utils.py` (`get_image_list`, `four_point_transform`).
//! Axis-aligned regions are a bounds-clamped rectangular slice; free (rotated)
//! quads use a 4-point perspective warp. Resizing to the recognizer height is
//! handled later by the preprocessing stage.

use image::{GrayImage, Luma};
use imageproc::geometric_transformations::{Border, Interpolation, Projection, warp_into};

use crate::error::{OcrError, Result};
use crate::types::QUAD_CORNERS;

use super::recognizer::RegionCrop;

/// Minimum width or height, in pixels, of a warped free-quad crop. Mirrors the
/// implicit `>= 1` guard EasyOCR relies on after Python's `int()` truncation.
const MIN_WARP_DIMENSION: i32 = 1;

/// Crop one region from the full grayscale image. Axis-aligned regions are a
/// bounds-clamped rectangular slice; free (rotated) quads use a 4-point
/// perspective warp. `corners` are `[TL, TR, BR, BL]` as `[x, y]`, clockwise
/// from the top-left.
pub(crate) fn crop_region(
    gray: &GrayImage,
    corners: &[[f32; 2]; QUAD_CORNERS],
    axis_aligned: bool,
) -> Result<RegionCrop> {
    if axis_aligned {
        crop_axis_aligned(gray, corners)
    } else {
        crop_free_quad(gray, corners)
    }
}

/// Slice the bounding rectangle of `corners`, clamped to the image, row-major.
fn crop_axis_aligned(gray: &GrayImage, corners: &[[f32; 2]; QUAD_CORNERS]) -> Result<RegionCrop> {
    let width = gray.width() as f32;
    let height = gray.height() as f32;

    let xs = corners.map(|corner| corner[0]);
    let ys = corners.map(|corner| corner[1]);

    let x_min = clamp_extent(fold_min(&xs), width);
    let x_max = clamp_extent(fold_max(&xs), width);
    let y_min = clamp_extent(fold_min(&ys), height);
    let y_max = clamp_extent(fold_max(&ys), height);

    let (x0, x1) = (x_min as u32, x_max as u32);
    let (y0, y1) = (y_min as u32, y_max as u32);

    if x1 <= x0 || y1 <= y0 {
        return Err(OcrError::image(
            "axis-aligned crop is empty after clamping to the image",
        ));
    }

    let crop_width = x1 - x0;
    let crop_height = y1 - y0;
    let mut pixels = Vec::with_capacity((crop_width * crop_height) as usize);
    for row in y0..y1 {
        for col in x0..x1 {
            pixels.push(gray.get_pixel(col, row)[0]);
        }
    }

    Ok(RegionCrop {
        width: crop_width,
        height: crop_height,
        gray: pixels,
        corners: *corners,
    })
}

/// Perspective-warp the rotated quad `corners` into an upright natural-size crop.
fn crop_free_quad(gray: &GrayImage, corners: &[[f32; 2]; QUAD_CORNERS]) -> Result<RegionCrop> {
    let [top_left, top_right, bottom_right, bottom_left] = *corners;

    let width_bottom = distance(bottom_right, bottom_left).trunc() as i32;
    let width_top = distance(top_right, top_left).trunc() as i32;
    let max_width = width_bottom.max(width_top).max(MIN_WARP_DIMENSION);

    let height_right = distance(top_right, bottom_right).trunc() as i32;
    let height_left = distance(top_left, bottom_left).trunc() as i32;
    let max_height = height_right.max(height_left).max(MIN_WARP_DIMENSION);

    let last_x = (max_width - 1) as f32;
    let last_y = (max_height - 1) as f32;
    let source = [
        (top_left[0], top_left[1]),
        (top_right[0], top_right[1]),
        (bottom_right[0], bottom_right[1]),
        (bottom_left[0], bottom_left[1]),
    ];
    let destination = [(0.0, 0.0), (last_x, 0.0), (last_x, last_y), (0.0, last_y)];

    // `from_control_points(source, destination)` maps input->output; `warp_into` ~keep
    // inverts it internally so each output pixel samples its input pre-image. ~keep
    let projection = Projection::from_control_points(source, destination)
        .ok_or_else(|| OcrError::image("free-quad corners do not form an invertible perspective"))?;

    let mut warped = GrayImage::new(max_width as u32, max_height as u32);
    warp_into(
        gray,
        projection,
        Interpolation::Bilinear,
        Border::Constant(Luma([0u8])),
        &mut warped,
    );

    Ok(RegionCrop {
        width: warped.width(),
        height: warped.height(),
        gray: warped.into_raw(),
        corners: *corners,
    })
}

/// Euclidean distance between two `[x, y]` points.
fn distance(a: [f32; 2], b: [f32; 2]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// Clamp a coordinate extent to `[0.0, limit]`.
fn clamp_extent(value: f32, limit: f32) -> f32 {
    value.max(0.0).min(limit)
}

/// Smallest of four values (no `f32: Ord`, so fold by hand).
fn fold_min(values: &[f32; 4]) -> f32 {
    values.iter().copied().fold(f32::INFINITY, f32::min)
}

/// Largest of four values.
fn fold_max(values: &[f32; 4]) -> f32 {
    values.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 4x3 grayscale ramp: pixel value = row * 10 + col.
    fn sample_image() -> GrayImage {
        let (width, height) = (4u32, 3u32);
        let mut pixels = Vec::with_capacity((width * height) as usize);
        for row in 0..height {
            for col in 0..width {
                pixels.push((row * 10 + col) as u8);
            }
        }
        GrayImage::from_raw(width, height, pixels).expect("valid raw buffer")
    }

    #[test]
    fn should_slice_expected_sub_rectangle_when_axis_aligned() {
        let image = sample_image();
        // Columns 1..3, rows 0..2 -> a 2x2 window. ~keep
        let corners = [[1.0, 0.0], [3.0, 0.0], [3.0, 2.0], [1.0, 2.0]];

        let crop = crop_region(&image, &corners, true).expect("crop succeeds");

        assert_eq!(crop.width, 2);
        assert_eq!(crop.height, 2);
        assert_eq!(crop.gray, vec![1, 2, 11, 12]);
        assert_eq!(crop.corners, corners);
    }

    #[test]
    fn should_clamp_axis_aligned_corners_that_exceed_the_image() {
        let image = sample_image();
        // Requests x up to 10 and y up to 8; must clamp to width 4, height 3. ~keep
        let corners = [[2.0, 1.0], [10.0, 1.0], [10.0, 8.0], [2.0, 8.0]];

        let crop = crop_region(&image, &corners, true).expect("crop succeeds");

        assert_eq!(crop.width, 2);
        assert_eq!(crop.height, 2);
        // Rows 1..3, cols 2..4: (12,13) then (22,23). ~keep
        assert_eq!(crop.gray, vec![12, 13, 22, 23]);
    }

    #[test]
    fn should_error_when_axis_aligned_region_is_zero_area() {
        let image = sample_image();
        let corners = [[2.0, 1.0], [2.0, 1.0], [2.0, 1.0], [2.0, 1.0]];

        assert!(crop_region(&image, &corners, true).is_err());
    }

    /// 10x4 grayscale image whose value depends only on the column (a purely
    /// horizontal gradient), so any vertical rescale in the warp is irrelevant
    /// and the test isolates left/right orientation and direction.
    fn horizontal_gradient() -> GrayImage {
        let (width, height) = (10u32, 4u32);
        let mut pixels = Vec::with_capacity((width * height) as usize);
        for _row in 0..height {
            for col in 0..width {
                pixels.push(col as u8);
            }
        }
        GrayImage::from_raw(width, height, pixels).expect("valid raw buffer")
    }

    #[test]
    fn should_reproduce_source_when_rectangular_quad_warped() {
        let image = horizontal_gradient();
        // A perfectly rectangular quad passed as a free quad must warp to ~the ~keep
        // same pixels as the axis-aligned slice. A flipped or transposed warp ~keep
        // direction would sample outside and collapse to the border (0). ~keep
        let corners = [[0.0, 0.0], [9.0, 0.0], [9.0, 3.0], [0.0, 3.0]];

        let warped = crop_region(&image, &corners, false).expect("warp succeeds");
        let sliced = crop_region(&image, &corners, true).expect("slice succeeds");

        assert_eq!(warped.width, sliced.width, "warp width matches slice width");
        assert_eq!(warped.height, sliced.height, "warp height matches slice height");
        assert!(warped.gray.iter().any(|&p| p > 0), "warp is not a blank border fill");

        // Correct direction reproduces the horizontal gradient; EasyOCR's ~keep
        // `maxWidth - 1` destination introduces at most ~1 level of resample. ~keep
        for (index, (&actual, &expected)) in warped.gray.iter().zip(sliced.gray.iter()).enumerate() {
            let diff = (actual as i32 - expected as i32).abs();
            assert!(diff <= 1, "pixel {index}: warped {actual} vs sliced {expected}");
        }
    }
}
