//! Detection preprocessing: aspect-ratio resize and mean/variance normalization.
//!
//! Reference: EasyOCR `imgproc.py` (`resize_aspect_ratio`, `normalizeMeanVariance`).
//! RGB is normalized with ImageNet mean/std (each scaled by 255) and laid out as
//! an NCHW tensor before the CRAFT forward pass.

use image::imageops::{FilterType, resize};
use image::{ImageBuffer, Rgb, RgbImage};
use ndarray::IxDyn;

use crate::error::{OcrError, Result};
use crate::inference::Tensor;
use crate::types::Image;

/// Batch dimension of the CRAFT input tensor (single image per forward pass).
const BATCH: usize = 1;
/// RGB channel count.
const CHANNELS: usize = 3;
/// Full 8-bit range; ImageNet mean/std are defined in `[0, 1]` and scaled by this.
const U8_MAX: f32 = 255.0;
/// CRAFT requires spatial dimensions to be multiples of this alignment.
const ALIGN: u32 = 32;
/// Pixels per megapixel, for converting [`DetectionConfig::max_megapixels`]
/// (a fractional megapixel count) into a pixel-area budget.
///
/// [`DetectionConfig::max_megapixels`]: crate::config::DetectionConfig::max_megapixels
const PIXELS_PER_MEGAPIXEL: f64 = 1_000_000.0;
/// ImageNet per-channel mean (RGB order), in `[0, 1]`.
const IMAGENET_MEAN: [f32; CHANNELS] = [0.485, 0.456, 0.406];
/// ImageNet per-channel standard deviation (RGB order), in `[0, 1]`.
const IMAGENET_STD: [f32; CHANNELS] = [0.229, 0.224, 0.225];

/// Detection input prepared for the CRAFT forward pass.
pub(super) struct Prepared {
    /// NCHW `[1, 3, H, W]` f32 tensor, mean/variance normalized (RGB order).
    pub tensor: Tensor,
    /// Maps CRAFT heat-map coordinates back to original-image space.
    /// Equals `1.0 / resize_ratio`; the caller multiplies box coords by
    /// `inv_ratio * RATIO_NET` (RATIO_NET = 2) to undo resize + half-res heat-map.
    pub inv_ratio: f32,
}

/// Resize `image` preserving aspect ratio (long side ≤ `canvas_size`, scaled by
/// `mag_ratio`), pad to a multiple of 32, and normalize with ImageNet mean/std
/// (each × 255). Mirrors `resize_aspect_ratio` + `normalizeMeanVariance`.
///
/// EasyOCR pastes the raw resized image into a zero canvas and then normalizes the
/// whole padded canvas, so every padding pixel takes the normalized value of a raw
/// zero — `(0 - mean) / std`, not `0.0`. That normalized-zero border feeds CRAFT's
/// convolutions near the real-pixel edges, so we normalize the full canvas the same
/// way to preserve parity.
// Thin dynamic-canvas wrapper retained for the preprocess unit tests and the `bench`
// preprocess shim; production detection calls `prepare_with_canvas` directly. ~keep
#[cfg(any(test, feature = "bench"))]
pub(super) fn prepare(image: &Image, canvas_size: u32, mag_ratio: f32) -> Result<Prepared> {
    prepare_with_canvas(image, canvas_size, mag_ratio, None, None)
}

/// [`prepare`], but with an optional fixed square output canvas.
///
/// With `fixed_canvas` `None` the padded output is the aspect-preserving resized image
/// rounded up to a multiple of 32 (the default, dynamic-shape path). With `Some(canvas)`
/// the resized image is padded (bottom-right, like the dynamic path) into a fixed
/// `canvas × canvas` output, so every image yields the same CRAFT input shape — required
/// by the `tract` backend, which cannot shape-infer CRAFT under dynamic H/W and is instead
/// pinned to this one shape (see ADR 0027). `canvas` must be a multiple of 32 and at least
/// as large as the resized image; otherwise this errors.
///
/// `max_megapixels`, when set, further constrains the resize target so the padded
/// output area stays within the budget (see [`resize_dimensions_within_budget`]);
/// `None` reproduces the unconstrained `canvas_size`/`mag_ratio` sizing exactly. On
/// the `fixed_canvas` path the padded output area is pinned by `canvas` regardless,
/// so the budget only shrinks the real (unpadded) content pasted into that square.
pub(super) fn prepare_with_canvas(
    image: &Image,
    canvas_size: u32,
    mag_ratio: f32,
    fixed_canvas: Option<u32>,
    max_megapixels: Option<f32>,
) -> Result<Prepared> {
    let width = image.width();
    let height = image.height();
    if width == 0 || height == 0 {
        return Err(OcrError::image("cannot preprocess an image with zero width or height"));
    }

    let (ratio, target_h, target_w) =
        resize_dimensions_within_budget(height, width, canvas_size, mag_ratio, max_megapixels);
    let source = ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, image.as_rgb8())
        .ok_or_else(|| OcrError::image("failed to build RGB image view from raw RGB8 buffer"))?;
    let resized = resize(&source, target_w, target_h, FilterType::Triangle);

    let (padded_h, padded_w) = match fixed_canvas {
        Some(canvas) => {
            if target_h > canvas || target_w > canvas {
                return Err(OcrError::image(format!(
                    "resized detection input {target_w}x{target_h} exceeds the fixed canvas {canvas}"
                )));
            }
            (canvas, canvas)
        }
        None => (pad_to_multiple(target_h, ALIGN), pad_to_multiple(target_w, ALIGN)),
    };
    let tensor = normalize_into_tensor(&resized, padded_h, padded_w)?;

    Ok(Prepared {
        tensor,
        inv_ratio: 1.0 / ratio,
    })
}

/// Reference variant of [`prepare`] that normalizes via
/// [`normalize_into_tensor_reference`]; retained for the A/B benchmark baseline.
/// The resize and padding are shared with [`prepare`]; only the normalize path
/// differs.
#[cfg(feature = "bench")]
pub(super) fn prepare_reference(image: &Image, canvas_size: u32, mag_ratio: f32) -> Result<Prepared> {
    let width = image.width();
    let height = image.height();
    if width == 0 || height == 0 {
        return Err(OcrError::image("cannot preprocess an image with zero width or height"));
    }

    let (ratio, target_h, target_w) = resize_dimensions(height, width, canvas_size, mag_ratio);
    let source = ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, image.as_rgb8())
        .ok_or_else(|| OcrError::image("failed to build RGB image view from raw RGB8 buffer"))?;
    let resized = resize(&source, target_w, target_h, FilterType::Triangle);

    let padded_h = pad_to_multiple(target_h, ALIGN);
    let padded_w = pad_to_multiple(target_w, ALIGN);
    let tensor = normalize_into_tensor_reference(&resized, padded_h, padded_w)?;

    Ok(Prepared {
        tensor,
        inv_ratio: 1.0 / ratio,
    })
}

/// Compute the resize ratio and target height/width, following `resize_aspect_ratio`:
/// `target = min(mag_ratio * max(h, w), canvas_size)`, `ratio = target / max(h, w)`.
/// Target dimensions truncate (matching Python `int()`) and clamp to at least one pixel.
fn resize_dimensions(height: u32, width: u32, canvas_size: u32, mag_ratio: f32) -> (f32, u32, u32) {
    let max_side = height.max(width) as f32;
    let target_size = (mag_ratio * max_side).min(canvas_size as f32);
    let ratio = target_size / max_side;
    let target_h = ((height as f32) * ratio).trunc().max(1.0) as u32;
    let target_w = ((width as f32) * ratio).trunc().max(1.0) as u32;
    (ratio, target_h, target_w)
}

/// [`resize_dimensions`], further capped so the padded output area stays within
/// `max_megapixels` (`None` is a no-op, reproducing [`resize_dimensions`] exactly).
///
/// `canvas_size` already caps the resize target's longest side; as that cap shrinks,
/// [`resize_dimensions`]'s target dimensions (and therefore the padded area) are
/// monotonically non-decreasing, so binary-searching the largest effective cap in
/// `[1, canvas_size]` whose padded area fits the budget finds the least-restrictive
/// size satisfying it, without ever exceeding what `canvas_size` alone allows — the
/// two constraints compose as their minimum. A budget too small for even a
/// single-pixel image (padded to `ALIGN × ALIGN`) is a best-effort floor, not an
/// error: the search still returns the smallest achievable size.
fn resize_dimensions_within_budget(
    height: u32,
    width: u32,
    canvas_size: u32,
    mag_ratio: f32,
    max_megapixels: Option<f32>,
) -> (f32, u32, u32) {
    let unconstrained = resize_dimensions(height, width, canvas_size, mag_ratio);
    let Some(max_megapixels) = max_megapixels else {
        return unconstrained;
    };
    let budget_pixels = f64::from(max_megapixels) * PIXELS_PER_MEGAPIXEL;
    let (_, unconstrained_h, unconstrained_w) = unconstrained;
    if padded_area_pixels(unconstrained_h, unconstrained_w) <= budget_pixels {
        return unconstrained;
    }

    let mut lo: u32 = 1;
    let mut hi: u32 = canvas_size.max(1);
    while lo < hi {
        // Bias the midpoint up so the loop converges on the largest cap that fits,
        // rather than looping forever narrowing toward `lo`. ~keep
        let mid = lo + (hi - lo).div_ceil(2);
        let (_, mid_h, mid_w) = resize_dimensions(height, width, mid, mag_ratio);
        if padded_area_pixels(mid_h, mid_w) <= budget_pixels {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    resize_dimensions(height, width, lo, mag_ratio)
}

/// The padded (multiple-of-32) area, in pixels, that `resize_dimensions` output of
/// `target_h × target_w` would occupy once padded — the quantity the megapixel
/// budget bounds, since that is what the RSS-vs-area fit in ADR 0041 measured.
fn padded_area_pixels(target_h: u32, target_w: u32) -> f64 {
    f64::from(pad_to_multiple(target_h, ALIGN)) * f64::from(pad_to_multiple(target_w, ALIGN))
}

/// Round `value` up to the next multiple of `align` (leaving multiples unchanged).
fn pad_to_multiple(value: u32, align: u32) -> u32 {
    let remainder = value % align;
    if remainder == 0 {
        value
    } else {
        value + (align - remainder)
    }
}

/// Build the NCHW `[1, 3, padded_h, padded_w]` tensor by normalizing the full padded
/// canvas: the resized region carries the real pixels, and each padding pixel takes
/// the normalized value of a raw zero, `(0 - mean) / std` (matching EasyOCR, which
/// normalizes the whole zero-padded canvas — see [`prepare`]).
fn normalize_into_tensor(resized: &RgbImage, padded_h: u32, padded_w: u32) -> Result<Tensor> {
    let (target_w, target_h) = resized.dimensions();
    let plane = (padded_h * padded_w) as usize;
    let mut tensor = Tensor::zeros(IxDyn(&[BATCH, CHANNELS, padded_h as usize, padded_w as usize]));
    let data = tensor
        .as_slice_mut()
        .ok_or_else(|| OcrError::inference("detection tensor is not in contiguous standard layout"))?;
    let raw = resized.as_raw();
    let padded_w = padded_w as usize;
    let target_w = target_w as usize;
    let row_stride = target_w * CHANNELS;
    for channel in 0..CHANNELS {
        let mean = IMAGENET_MEAN[channel] * U8_MAX;
        let std = IMAGENET_STD[channel] * U8_MAX;
        let channel_plane = &mut data[channel * plane..(channel + 1) * plane];
        channel_plane.fill((0.0 - mean) / std);
        for y in 0..target_h as usize {
            let destination = &mut channel_plane[y * padded_w..y * padded_w + target_w];
            let source = &raw[y * row_stride..y * row_stride + row_stride];
            for (cell, pixel) in destination.iter_mut().zip(source.chunks_exact(CHANNELS)) {
                *cell = (f32::from(pixel[channel]) - mean) / std;
            }
        }
    }
    Ok(tensor)
}

/// Reference implementation of [`normalize_into_tensor`] retained for the
/// differential test and the A/B benchmark baseline: reads each channel byte
/// through the bounds-checked, channel-strided [`RgbImage::get_pixel`], which
/// blocks autovectorization (see ADR 0019).
#[cfg(any(test, feature = "bench"))]
fn normalize_into_tensor_reference(resized: &RgbImage, padded_h: u32, padded_w: u32) -> Result<Tensor> {
    let (target_w, target_h) = resized.dimensions();
    let plane = (padded_h * padded_w) as usize;
    let mut tensor = Tensor::zeros(IxDyn(&[BATCH, CHANNELS, padded_h as usize, padded_w as usize]));
    let data = tensor
        .as_slice_mut()
        .ok_or_else(|| OcrError::inference("detection tensor is not in contiguous standard layout"))?;
    for channel in 0..CHANNELS {
        let mean = IMAGENET_MEAN[channel] * U8_MAX;
        let std = IMAGENET_STD[channel] * U8_MAX;
        let channel_plane = &mut data[channel * plane..(channel + 1) * plane];
        channel_plane.fill((0.0 - mean) / std);
        for y in 0..target_h {
            let row = (y * padded_w) as usize;
            for x in 0..target_w {
                let raw = f32::from(resized.get_pixel(x, y).0[channel]);
                channel_plane[row + x as usize] = (raw - mean) / std;
            }
        }
    }
    Ok(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_image(width: u32, height: u32, rgb: [u8; 3]) -> Image {
        let mut pixels = Vec::with_capacity((width * height * 3) as usize);
        for _ in 0..(width * height) {
            pixels.extend_from_slice(&rgb);
        }
        Image::from_rgb8(width, height, pixels).expect("valid rgb buffer")
    }

    /// Sizes exercised by the megapixel-budget tests, chosen to cover square,
    /// landscape, portrait, sub-alignment, and already-aligned dimensions.
    const BUDGET_TEST_SIZES: [(u32, u32); 6] =
        [(100, 50), (50, 100), (2560, 2560), (4000, 3000), (33, 17), (1024, 1024)];

    /// Assert two `resize_dimensions`-shaped tuples are identical: bit-exact on the
    /// `f32` ratio (both sides come from the same deterministic formula, so bitwise
    /// equality — not an epsilon compare — is the right bar) and exact on target
    /// height/width.
    fn assert_dimensions_identical(actual: (f32, u32, u32), expected: (f32, u32, u32), context: &str) {
        assert_eq!(
            actual.0.to_bits(),
            expected.0.to_bits(),
            "{context}: ratio differs ({} vs {})",
            actual.0,
            expected.0
        );
        assert_eq!(
            (actual.1, actual.2),
            (expected.1, expected.2),
            "{context}: target dimensions differ"
        );
    }

    #[test]
    fn should_leave_resize_dimensions_unchanged_when_budget_is_none() {
        for (width, height) in BUDGET_TEST_SIZES {
            let unconstrained = resize_dimensions(height, width, 2560, 1.0);
            let with_budget = resize_dimensions_within_budget(height, width, 2560, 1.0, None);
            assert_dimensions_identical(
                with_budget,
                unconstrained,
                &format!("width={width} height={height}: absent budget must reproduce today's sizing exactly"),
            );
        }
    }

    #[test]
    fn should_leave_prepare_with_canvas_tensor_shape_unchanged_when_budget_is_none() {
        for (width, height) in BUDGET_TEST_SIZES {
            let image = solid_image(width, height, [7, 8, 9]);
            let without_field = prepare_with_canvas(&image, 2560, 1.0, None, None).expect("prepare succeeds");
            let baseline = prepare(&image, 2560, 1.0).expect("prepare succeeds");
            assert_eq!(
                without_field.tensor.shape(),
                baseline.tensor.shape(),
                "width={width} height={height}: an absent budget must not change the padded tensor shape"
            );
        }
    }

    #[test]
    fn should_be_a_no_op_when_budget_exceeds_the_image() {
        // 4000x3000 padded is well under 100 MP, so a 100 MP budget must never bind. ~keep
        for (width, height) in BUDGET_TEST_SIZES {
            let unconstrained = resize_dimensions(height, width, 2560, 1.0);
            let with_budget = resize_dimensions_within_budget(height, width, 2560, 1.0, Some(100.0));
            assert_dimensions_identical(
                with_budget,
                unconstrained,
                &format!("width={width} height={height}: a budget larger than the image must be a no-op"),
            );
        }
    }

    #[test]
    fn should_shrink_to_fit_a_restrictive_budget() {
        let (width, height) = (4000u32, 3000u32);
        let budget_mp = 0.5f32;
        let (ratio, target_h, target_w) = resize_dimensions_within_budget(height, width, 2560, 1.0, Some(budget_mp));

        let padded_h = pad_to_multiple(target_h, ALIGN);
        let padded_w = pad_to_multiple(target_w, ALIGN);
        assert!(
            f64::from(padded_h) * f64::from(padded_w) <= f64::from(budget_mp) * PIXELS_PER_MEGAPIXEL,
            "padded area {padded_h}x{padded_w} must fit the {budget_mp} MP budget"
        );
        assert_eq!(padded_h % ALIGN, 0);
        assert_eq!(padded_w % ALIGN, 0);
        assert!(
            ratio > 0.0,
            "a satisfiable budget must not collapse the image to nothing"
        );

        let expected_aspect = height as f64 / width as f64;
        let actual_aspect = target_h as f64 / target_w as f64;
        assert!(
            (actual_aspect - expected_aspect).abs() < 0.01,
            "budget-constrained resize must preserve aspect ratio: expected {expected_aspect}, got {actual_aspect}"
        );
    }

    #[test]
    fn should_compose_with_canvas_size_as_a_minimum() {
        let (width, height) = (4000u32, 3000u32);

        // A tight canvas_size is the binding constraint when the budget is generous. ~keep
        let canvas_only = resize_dimensions(height, width, 200, 1.0);
        let with_generous_budget = resize_dimensions_within_budget(height, width, 200, 1.0, Some(1000.0));
        assert_dimensions_identical(
            with_generous_budget,
            canvas_only,
            "a generous budget must not override a tighter canvas_size",
        );

        // A tight budget is the binding constraint when canvas_size is generous, and
        // raising canvas_size further changes nothing once the budget already binds. ~keep
        let (_, tight_budget_h, tight_budget_w) = resize_dimensions_within_budget(height, width, 2560, 1.0, Some(0.1));
        let (_, wider_canvas_h, wider_canvas_w) = resize_dimensions_within_budget(height, width, 4000, 1.0, Some(0.1));
        assert_eq!((tight_budget_h, tight_budget_w), (wider_canvas_h, wider_canvas_w));
        assert!(
            f64::from(pad_to_multiple(tight_budget_h, ALIGN)) * f64::from(pad_to_multiple(tight_budget_w, ALIGN))
                <= 0.1 * PIXELS_PER_MEGAPIXEL,
            "a tight budget must bind even under a generous canvas_size"
        );
    }

    #[test]
    fn should_keep_padded_area_within_budget_across_a_range_of_dimensions() {
        // Property-style sweep: for every (width, height) on a grid plus odd/prime
        // sizes, and every budget tried, the padded area must never exceed it. ~keep
        let dimensions: Vec<(u32, u32)> = (1..=20)
            .flat_map(|w| (1..=20).map(move |h| (w * 137, h * 97)))
            .collect();
        let budgets = [0.05f32, 0.25, 1.0, 4.0, 16.0];

        for (width, height) in dimensions {
            for budget in budgets {
                let (_, target_h, target_w) = resize_dimensions_within_budget(height, width, 2560, 1.0, Some(budget));
                let area = padded_area_pixels(target_h, target_w);
                let budget_pixels = f64::from(budget) * PIXELS_PER_MEGAPIXEL;
                assert!(
                    area <= budget_pixels,
                    "width={width} height={height} budget={budget}MP: padded area {area} exceeds budget {budget_pixels}"
                );
            }
        }
    }

    #[test]
    fn should_downscale_image_larger_than_canvas() {
        let image = solid_image(100, 50, [10, 20, 30]);
        let prepared = prepare(&image, 64, 1.0).expect("prepare succeeds");

        // ratio = 64 / 100 = 0.64 -> target 64x32, both already multiples of 32 ~keep
        assert_eq!(prepared.tensor.shape(), &[1, 3, 32, 64]);
        assert!((prepared.inv_ratio - 1.0 / 0.64).abs() < 1e-6);
    }

    #[test]
    fn should_scale_up_small_image_and_clamp_at_canvas() {
        let image = solid_image(10, 10, [0, 0, 0]);
        let prepared = prepare(&image, 64, 100.0).expect("prepare succeeds");

        // target = min(100 * 10, 64) = 64 -> ratio 6.4, upscaled and clamped ~keep
        assert_eq!(prepared.tensor.shape(), &[1, 3, 64, 64]);
        assert!(prepared.inv_ratio < 1.0, "upscaled image has inv_ratio < 1");
        assert!((prepared.inv_ratio - 1.0 / 6.4).abs() < 1e-6);
    }

    #[test]
    fn should_pad_output_to_multiples_of_32() {
        let image = solid_image(50, 50, [128, 128, 128]);
        let prepared = prepare(&image, 2560, 1.0).expect("prepare succeeds");

        let shape = prepared.tensor.shape();
        // ratio 1.0 -> target 50x50 -> padded to 64x64 ~keep
        assert_eq!(shape[0], 1);
        assert_eq!(shape[1], 3);
        assert_eq!(shape[2] % 32, 0);
        assert_eq!(shape[3] % 32, 0);
        assert_eq!(shape, &[1, 3, 64, 64]);
    }

    #[test]
    fn should_normalize_padding_region_to_normalized_zero() {
        let image = solid_image(50, 50, [255, 255, 255]);
        let prepared = prepare(&image, 2560, 1.0).expect("prepare succeeds");

        // Pixel (63, 63) is padding (image is 50x50); EasyOCR normalizes the zero ~keep
        // canvas, so it holds (0 - mean*255)/(std*255), not 0.0. ~keep
        let expected = (0.0 - 0.485 * 255.0) / (0.229 * 255.0);
        assert!((prepared.tensor[[0, 0, 63, 63]] - expected).abs() < 1e-4);
    }

    #[test]
    fn should_truncate_fractional_target_dimensions() {
        // 50w x 15h, mag_ratio 0.7 (target < canvas so ratio == 0.7): target_h = 15*0.7 ~keep
        // = 10.5. Python int() truncates to 10; round() would give 11. With target_h 10, ~keep
        // row 10 is padding (normalized zero); with 11 it would be a real white pixel. ~keep
        let image = solid_image(50, 15, [255, 255, 255]);
        let prepared = prepare(&image, 2560, 0.7).expect("prepare succeeds");

        let mean = 0.485 * 255.0;
        let std = 0.229 * 255.0;
        let real_white = (255.0 - mean) / std;
        let padding = (0.0 - mean) / std;
        assert!(
            (prepared.tensor[[0, 0, 9, 0]] - real_white).abs() < 1e-3,
            "row 9 must be a real white pixel"
        );
        assert!(
            (prepared.tensor[[0, 0, 10, 0]] - padding).abs() < 1e-3,
            "row 10 must be padding, proving target_h truncated to 10 not 11"
        );
    }

    #[test]
    fn should_normalize_solid_color_to_expected_value() {
        let image = solid_image(32, 32, [100, 150, 200]);
        let prepared = prepare(&image, 2560, 1.0).expect("prepare succeeds");

        let expected = [
            (100.0 - 0.485 * 255.0) / (0.229 * 255.0),
            (150.0 - 0.456 * 255.0) / (0.224 * 255.0),
            (200.0 - 0.406 * 255.0) / (0.225 * 255.0),
        ];
        for (channel, expected_value) in expected.iter().enumerate() {
            let actual = prepared.tensor[[0, channel, 5, 5]];
            assert!(
                (actual - expected_value).abs() < 1e-4,
                "channel {channel}: expected {expected_value}, got {actual}"
            );
        }
    }

    #[test]
    fn should_compute_inv_ratio_as_reciprocal_of_ratio() {
        let image = solid_image(100, 50, [1, 2, 3]);
        let prepared = prepare(&image, 64, 1.0).expect("prepare succeeds");

        // ratio = 0.64 -> inv_ratio = 1.5625 ~keep
        assert!((prepared.inv_ratio - 1.5625).abs() < 1e-6);
    }

    #[test]
    fn should_reject_zero_dimension_image() {
        let image = Image::from_rgb8(0, 0, Vec::new()).expect("empty image is valid");
        assert!(prepare(&image, 64, 1.0).is_err());
    }

    #[test]
    fn optimized_normalize_matches_reference_bitwise() {
        // Fill a resized image with deterministic pseudo-random RGB (an LCG), pad to ~keep
        // multiples of 32, and require the optimized contiguous-slice normalize to ~keep
        // match the channel-strided get_pixel reference bit-for-bit across every ~keep
        // channel plane, covering real and padding regions. ~keep
        let (target_w, target_h) = (37u32, 19u32);
        let mut resized = RgbImage::new(target_w, target_h);
        let mut state: u32 = 0x1234_5678;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        };
        for y in 0..target_h {
            for x in 0..target_w {
                resized.put_pixel(x, y, image::Rgb([next(), next(), next()]));
            }
        }
        let padded_h = pad_to_multiple(target_h, ALIGN);
        let padded_w = pad_to_multiple(target_w, ALIGN);

        let optimized = normalize_into_tensor(&resized, padded_h, padded_w).expect("optimized tensor");
        let reference = normalize_into_tensor_reference(&resized, padded_h, padded_w).expect("reference tensor");

        assert_eq!(optimized.shape(), reference.shape(), "shapes must match");
        let optimized = optimized.as_slice().expect("optimized tensor is contiguous");
        let reference = reference.as_slice().expect("reference tensor is contiguous");
        assert_eq!(optimized.len(), reference.len());
        for (index, (a, b)) in optimized.iter().zip(reference.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "element {index} differs bitwise");
        }
    }

    #[test]
    fn should_match_per_pixel_formula_in_real_and_padding_regions() {
        // 3x2 resized image padded to 4x4: assert exact f32 equality against the old ~keep
        // per-pixel formula across the whole tensor, covering real and padding pixels. ~keep
        let mut resized = RgbImage::new(3, 2);
        resized.put_pixel(0, 0, image::Rgb([10, 20, 30]));
        resized.put_pixel(1, 0, image::Rgb([40, 50, 60]));
        resized.put_pixel(2, 0, image::Rgb([70, 80, 90]));
        resized.put_pixel(0, 1, image::Rgb([15, 25, 35]));
        resized.put_pixel(1, 1, image::Rgb([45, 55, 65]));
        resized.put_pixel(2, 1, image::Rgb([75, 85, 95]));
        let (padded_h, padded_w) = (4u32, 4u32);
        let tensor = normalize_into_tensor(&resized, padded_h, padded_w).expect("tensor");

        let expected = |channel: usize, y: u32, x: u32| -> f32 {
            let mean = IMAGENET_MEAN[channel] * U8_MAX;
            let std = IMAGENET_STD[channel] * U8_MAX;
            let raw = if y < 2 && x < 3 {
                f32::from(resized.get_pixel(x, y).0[channel])
            } else {
                0.0
            };
            (raw - mean) / std
        };

        for channel in 0..CHANNELS {
            for y in 0..padded_h {
                for x in 0..padded_w {
                    let actual = tensor[[0, channel, y as usize, x as usize]];
                    assert_eq!(actual, expected(channel, y, x), "channel {channel} at ({x},{y})");
                }
            }
        }
    }
}
