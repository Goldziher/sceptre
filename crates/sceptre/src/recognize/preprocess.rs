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

    let mut tensor = Tensor::zeros(IxDyn(&[resized.len(), CHANNELS, IMG_H as usize, max_w as usize]));

    for (index, image) in resized.iter().enumerate() {
        let real_w = image.width();
        for y in 0..IMG_H {
            let row = y as usize;
            for x in 0..real_w {
                tensor[[index, 0, row, x as usize]] = normalize(image.get_pixel(x, y)[0]);
            }
            let last_column = tensor[[index, 0, row, (real_w - 1) as usize]];
            for x in real_w..max_w {
                tensor[[index, 0, row, x as usize]] = last_column;
            }
        }
    }

    Ok(tensor)
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
        // w=10, h=40 (w < h): AlignCollate fixes height to 64 and width to
        // ceil(64 * 10/40) = 16 — a narrow strip, matching EasyOCR's final output
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
}
