//! Recognition preprocessing: normalize to [-1, 1] and pad a batch to equal width.
//!
//! Reference: EasyOCR `recognition.py` (`NormalizePAD`, `AlignCollate`).
//! Grayscale pixels are scaled to [0, 1] then `(x - 0.5) / 0.5`; crops are
//! right-padded (edge-replicated) to the batch's maximum width, forming a
//! `[B, 1, 64, W]` tensor.

use image::{GrayImage, imageops};
use ndarray::IxDyn;

use super::recognizer::RegionCrop;
use crate::error::{OcrError, Result};
use crate::inference::Tensor;

/// Recognizer input height in pixels (EasyOCR gen2 `imgH`).
const IMG_H: u32 = 64;
/// Single grayscale channel fed to the recognizer.
const CHANNELS: usize = 1;
/// Minimum resized width so a very short/wide crop never collapses to zero.
const MIN_WIDTH: u32 = 1;
/// Maximum 8-bit pixel value, used to scale into `[0, 1]`.
const PIXEL_MAX: f32 = 255.0;
/// Normalization mean subtracted after scaling to `[0, 1]`.
const NORM_MEAN: f32 = 0.5;
/// Normalization standard deviation dividing after mean subtraction.
const NORM_STD: f32 = 0.5;

/// Normalize one 8-bit grayscale value to `[-1, 1]` as `(v/255 - 0.5) / 0.5`.
fn normalize(value: u8) -> f32 {
    (value as f32 / PIXEL_MAX - NORM_MEAN) / NORM_STD
}

/// Resize one crop to height [`IMG_H`], width `ceil(IMG_H * w/h)` (min 1),
/// using a Catmull-Rom (bicubic-like) filter. Mirrors `AlignCollate.__call__`.
///
/// EasyOCR resizes each crop twice — once in `get_image_list`
/// (`compute_ratio_and_resize`) and again in `AlignCollate`. Both paths pin the
/// final height to `IMG_H` and the width to `ceil(IMG_H * w/h)`, so we fuse them
/// into this single resize; a portrait crop (`w < h`) becomes a narrow strip, not
/// a rotated tall image, exactly as EasyOCR's final `AlignCollate` output.
fn resize_crop(crop: &RegionCrop) -> Result<GrayImage> {
    if crop.width == 0 || crop.height == 0 {
        return Err(OcrError::image("crop has a zero width or height"));
    }
    let image = GrayImage::from_raw(crop.width, crop.height, crop.gray.clone())
        .ok_or_else(|| OcrError::image("crop grayscale buffer does not match its dimensions"))?;
    let ratio = crop.width as f32 / crop.height as f32;
    let resized_w = ((IMG_H as f32 * ratio).ceil() as u32).max(MIN_WIDTH);
    Ok(imageops::resize(
        &image,
        resized_w,
        IMG_H,
        imageops::FilterType::CatmullRom,
    ))
}

/// Resize each crop to height 64 (width = `ceil(64 * w/h)`, bicubic), normalize
/// to `[-1, 1]` via `(x/255 - 0.5)/0.5`, and right-pad every crop to the batch's
/// max width by edge-replicating the last real column. Returns `[B, 1, 64, maxW]`.
///
/// The padding width is the max over *this* batch, whereas EasyOCR pads to one max
/// width computed across every crop of the whole image. With the default
/// `batch_size` of 1 each crop is its own batch (no padding at all), so the two
/// agree; they can differ only for `batch_size > 1`, where edge-replicated padding
/// over dynamic width keeps the effect negligible.
///
/// An empty `crops` slice is an error: the runner never passes one.
pub(crate) fn prepare_batch(crops: &[RegionCrop]) -> Result<Tensor> {
    if crops.is_empty() {
        return Err(OcrError::inference("prepare_batch requires at least one crop"));
    }

    let resized: Vec<GrayImage> = crops.iter().map(resize_crop).collect::<Result<_>>()?;
    let max_w = resized.iter().map(GrayImage::width).max().unwrap_or(MIN_WIDTH);

    let plane_width = max_w as usize;
    let mut tensor = Tensor::zeros(IxDyn(&[resized.len(), CHANNELS, IMG_H as usize, plane_width]));
    let plane = IMG_H as usize * plane_width;
    let buffer = tensor
        .as_slice_mut()
        .ok_or_else(|| OcrError::inference("recognition tensor is not contiguous"))?;

    for (index, image) in resized.iter().enumerate() {
        fill_plane(buffer, image, index * plane, plane_width);
    }

    Ok(tensor)
}

/// Reference variant of [`prepare_batch`] that fills each plane with
/// [`fill_plane_reference`]; retained for the differential test and the A/B
/// benchmark baseline. The resize and layout are shared with [`prepare_batch`];
/// only the plane-fill path differs.
#[cfg(any(test, feature = "bench"))]
pub(crate) fn prepare_batch_reference(crops: &[RegionCrop]) -> Result<Tensor> {
    if crops.is_empty() {
        return Err(OcrError::inference("prepare_batch requires at least one crop"));
    }

    let resized: Vec<GrayImage> = crops.iter().map(resize_crop).collect::<Result<_>>()?;
    let max_w = resized.iter().map(GrayImage::width).max().unwrap_or(MIN_WIDTH);

    let plane_width = max_w as usize;
    let mut tensor = Tensor::zeros(IxDyn(&[resized.len(), CHANNELS, IMG_H as usize, plane_width]));
    let plane = IMG_H as usize * plane_width;
    let buffer = tensor
        .as_slice_mut()
        .ok_or_else(|| OcrError::inference("recognition tensor is not contiguous"))?;

    for (index, image) in resized.iter().enumerate() {
        fill_plane_reference(buffer, image, index * plane, plane_width);
    }

    Ok(tensor)
}

/// Write one `[1, 64, plane_width]` image plane into the contiguous NCHW `buffer`
/// starting at `base`, row-major, edge-replicating the last real column across the
/// padding. The written values are identical to the per-element formula: each real
/// pixel is [`normalize`]d and each padded cell repeats its row's last real column.
///
/// Iterates the image's contiguous grayscale backing slice ([`GrayImage::as_raw`],
/// row-major single channel) and zips each source row with its destination row, so
/// the per-pixel [`normalize`] loop autovectorizes; the arithmetic and write order
/// are identical to the bounds-checked `get_pixel` form (see ADR 0019).
fn fill_plane(buffer: &mut [f32], image: &GrayImage, base: usize, plane_width: usize) {
    let real_w = image.width() as usize;
    let raw = image.as_raw();
    for y in 0..IMG_H as usize {
        let row_base = base + y * plane_width;
        let source = &raw[y * real_w..y * real_w + real_w];
        let destination = &mut buffer[row_base..row_base + real_w];
        for (cell, &value) in destination.iter_mut().zip(source.iter()) {
            *cell = normalize(value);
        }
        let last_column = buffer[row_base + real_w - 1];
        buffer[row_base + real_w..row_base + plane_width].fill(last_column);
    }
}

/// Reference implementation of [`fill_plane`] retained for the differential test
/// and the A/B benchmark baseline: reads each pixel through the bounds-checked
/// [`GrayImage::get_pixel`], which blocks autovectorization (see ADR 0019).
#[cfg(any(test, feature = "bench"))]
fn fill_plane_reference(buffer: &mut [f32], image: &GrayImage, base: usize, plane_width: usize) {
    let real_w = image.width() as usize;
    for y in 0..IMG_H {
        let row_base = base + y as usize * plane_width;
        for x in 0..real_w {
            buffer[row_base + x] = normalize(image.get_pixel(x as u32, y)[0]);
        }
        let last_column = buffer[row_base + real_w - 1];
        buffer[row_base + real_w..row_base + plane_width].fill(last_column);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_crop(width: u32, height: u32, fill: u8) -> RegionCrop {
        RegionCrop {
            width,
            height,
            gray: vec![fill; (width * height) as usize],
            corners: [[0.0, 0.0]; 4],
        }
    }

    #[test]
    fn single_crop_yields_shape_with_width_ceil_of_aspect_ratio() {
        let crop = make_crop(20, 5, 128);
        let tensor = prepare_batch(&[crop]).expect("single crop should preprocess");
        let expected_w = (IMG_H as f32 * (20.0 / 5.0)).ceil() as usize;
        assert_eq!(tensor.shape(), &[1, 1, IMG_H as usize, expected_w]);
    }

    #[test]
    fn solid_gray_pixel_normalizes_to_exact_value() {
        let value: u8 = 200;
        let crop = make_crop(10, 5, value);
        let tensor = prepare_batch(&[crop]).expect("solid crop should preprocess");
        let expected = (value as f32 / 255.0 - 0.5) / 0.5;
        assert!(
            (tensor[[0, 0, 0, 0]] - expected).abs() < 1e-4,
            "pixel normalized to {expected}"
        );
    }

    #[test]
    fn portrait_crop_resizes_to_narrow_strip_matching_align_collate() {
        // w=10, h=40 (w < h): AlignCollate fixes height to 64 and width to ~keep
        // ceil(64 * 10/40) = 16 — a narrow strip, matching EasyOCR's final output ~keep
        // (get_image_list's intermediate resize does not change these dimensions). ~keep
        let crop = make_crop(10, 40, 128);
        let tensor = prepare_batch(&[crop]).expect("portrait crop should preprocess");
        assert_eq!(tensor.shape(), &[1, 1, IMG_H as usize, 16]);
    }

    #[test]
    fn zero_dimension_crop_is_rejected() {
        let crop = make_crop(0, 0, 0);
        assert!(prepare_batch(&[crop]).is_err(), "a zero-dimension crop must error");
    }

    #[test]
    fn narrow_crop_is_edge_replicated_to_max_width_not_zero() {
        let wide = make_crop(40, 5, 100);
        let narrow_value: u8 = 210;
        let narrow = make_crop(8, 5, narrow_value);
        let narrow_real_w = ((IMG_H as f32 * (8.0 / 5.0)).ceil() as u32).max(MIN_WIDTH);

        let tensor = prepare_batch(&[wide, narrow]).expect("batch should preprocess");
        let max_w = tensor.shape()[3];
        assert!((narrow_real_w as usize) < max_w, "narrow crop must be padded");

        let last_real = tensor[[1, 0, 0, (narrow_real_w - 1) as usize]];
        let padded = tensor[[1, 0, 0, max_w - 1]];
        let zero_normalized = (0.0f32 / 255.0 - 0.5) / 0.5;

        assert!(
            (padded - last_real).abs() < 1e-6,
            "padding replicates the last real column"
        );
        assert!((padded - zero_normalized).abs() > 1e-3, "padding is not a zeroed pixel");
    }

    #[test]
    fn optimized_fill_plane_matches_reference_bitwise() {
        // Vary aspect ratios (portrait, landscape, square, single-pixel) and pixel ~keep
        // values so the batch exercises real + padded columns across several plane ~keep
        // widths; the optimized contiguous-slice fill must match the get_pixel ~keep
        // reference bit-for-bit. ~keep
        let crops = vec![
            make_crop(40, 5, 90),
            make_crop(8, 5, 210),
            make_crop(3, 17, 45),
            make_crop(64, 64, 255),
            make_crop(1, 1, 0),
            make_crop(23, 9, 137),
        ];
        let optimized = prepare_batch(&crops).expect("optimized batch");
        let reference = prepare_batch_reference(&crops).expect("reference batch");

        assert_eq!(optimized.shape(), reference.shape(), "shapes must match");
        let optimized = optimized.as_slice().expect("optimized tensor is contiguous");
        let reference = reference.as_slice().expect("reference tensor is contiguous");
        assert_eq!(optimized.len(), reference.len());
        for (index, (a, b)) in optimized.iter().zip(reference.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "element {index} differs bitwise");
        }
    }

    #[test]
    fn tensor_matches_per_pixel_formula_exactly_including_padding() {
        let wide = make_crop(40, 5, 90);
        let narrow = make_crop(8, 5, 210);
        // Independently recompute the resized narrow crop to compare against. ~keep
        let expected = resize_crop(&make_crop(8, 5, 210)).expect("resize narrow crop");
        let real_w = expected.width();

        let tensor = prepare_batch(&[wide, narrow]).expect("batch should preprocess");
        let max_w = tensor.shape()[3] as u32;
        assert!(
            real_w < max_w,
            "narrow crop must be padded for this test to cover padding"
        );

        for (x, y) in [(0u32, 0u32), (3, 2), (real_w - 1, IMG_H - 1)] {
            let want = normalize(expected.get_pixel(x, y)[0]);
            assert_eq!(
                tensor[[1, 0, y as usize, x as usize]],
                want,
                "real pixel ({x}, {y}) must equal the per-element normalize formula"
            );
        }

        for y in [0u32, IMG_H / 2, IMG_H - 1] {
            let last_real = normalize(expected.get_pixel(real_w - 1, y)[0]);
            for x in real_w..max_w {
                assert_eq!(
                    tensor[[1, 0, y as usize, x as usize]],
                    last_real,
                    "padded cell ({x}, {y}) must equal the row's last real column"
                );
            }
        }
    }
}
