//! Colour-space preparation for recognition.
//!
//! Recognition runs on grayscale. EasyOCR (`reformat_input`) derives its
//! recognition grayscale from the source image; for file-path input — our primary
//! parity target, since the CLI and golden fixtures read from paths — it uses
//! `cv2.imread(path, IMREAD_GRAYSCALE)`, i.e. true BT.601 luminance
//! `Y = 0.299R + 0.587G + 0.114B`. [`to_grayscale`] reproduces that luminance with
//! OpenCV's 14-bit fixed-point coefficients, applied to the true RGB channels.
//!
//! EasyOCR's `bytes` code path instead re-labels an RGB array as BGR before
//! graying (`cvtColor(rgb, COLOR_BGR2GRAY)`), swapping the R and B weights — a known
//! upstream quirk we deliberately do not reproduce: sceptre decodes every input to
//! true RGB8 and always applies true luminance, regardless of how the image arrived.

use image::GrayImage;

use crate::error::{OcrError, Result};
use crate::types::Image;

/// OpenCV 14-bit fixed-point red weight for `COLOR_*2GRAY` (BT.601 luminance).
const RED_WEIGHT: u32 = 4899;
/// OpenCV 14-bit fixed-point green weight.
const GREEN_WEIGHT: u32 = 9617;
/// OpenCV 14-bit fixed-point blue weight (`RED + GREEN + BLUE == 1 << 14`).
const BLUE_WEIGHT: u32 = 1868;
/// Fixed-point fractional bits; the weighted sum is descaled by this shift.
const FIXED_SHIFT: u32 = 14;
/// Rounding bias added before the descale shift (`CV_DESCALE`: `(x + 1<<13) >> 14`).
const ROUND_BIAS: u32 = 1 << (FIXED_SHIFT - 1);
/// Channel count of a packed RGB8 pixel.
const RGB_CHANNELS: usize = 3;

/// Convert a decoded RGB image to grayscale using OpenCV's BT.601 luminance,
/// matching EasyOCR's file-path recognition grayscale (`cv2.imread` grayscale).
pub(crate) fn to_grayscale(image: &Image) -> Result<GrayImage> {
    let rgb = image.as_rgb8();
    let mut luma = Vec::with_capacity(rgb.len() / RGB_CHANNELS);
    for pixel in rgb.chunks_exact(RGB_CHANNELS) {
        let (red, green, blue) = (pixel[0] as u32, pixel[1] as u32, pixel[2] as u32);
        let gray = (red * RED_WEIGHT + green * GREEN_WEIGHT + blue * BLUE_WEIGHT + ROUND_BIAS) >> FIXED_SHIFT;
        luma.push(gray as u8);
    }
    GrayImage::from_raw(image.width(), image.height(), luma)
        .ok_or_else(|| OcrError::image("grayscale buffer does not match the image dimensions"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgb_image(width: u32, height: u32, pixels: Vec<u8>) -> Image {
        Image::from_rgb8(width, height, pixels).expect("valid rgb8 buffer")
    }

    #[test]
    fn should_produce_one_luma_byte_per_pixel() {
        let image = rgb_image(2, 2, vec![255u8; 2 * 2 * 3]);
        let grey = to_grayscale(&image).expect("grayscale succeeds");
        assert_eq!(grey.dimensions(), (2, 2));
        assert_eq!(grey.as_raw().len(), 4);
    }

    #[test]
    fn should_map_white_and_black_to_extremes() {
        let image = rgb_image(2, 1, vec![255, 255, 255, 0, 0, 0]);
        let grey = to_grayscale(&image).expect("grayscale succeeds");
        assert_eq!(grey.get_pixel(0, 0)[0], 255);
        assert_eq!(grey.get_pixel(1, 0)[0], 0);
    }

    #[test]
    fn should_weight_green_above_red_above_blue() {
        // Pure green must be brighter than pure red, which must beat pure blue, ~keep
        // matching the BT.601 ordering GREEN_WEIGHT > RED_WEIGHT > BLUE_WEIGHT. ~keep
        let image = rgb_image(3, 1, vec![255, 0, 0, 0, 255, 0, 0, 0, 255]);
        let grey = to_grayscale(&image).expect("grayscale succeeds");
        let (red, green, blue) = (
            grey.get_pixel(0, 0)[0],
            grey.get_pixel(1, 0)[0],
            grey.get_pixel(2, 0)[0],
        );
        assert!(green > red && red > blue, "green {green} > red {red} > blue {blue}");
    }

    #[test]
    fn should_match_opencv_fixed_point_luma_for_known_pixel() {
        // OpenCV: (100*4899 + 150*9617 + 200*1868 + 8192) >> 14 ~keep
        //       = (489900 + 1442550 + 373600 + 8192) >> 14 = 2314242 >> 14 = 141. ~keep
        let image = rgb_image(1, 1, vec![100, 150, 200]);
        let grey = to_grayscale(&image).expect("grayscale succeeds");
        assert_eq!(grey.get_pixel(0, 0)[0], 141);
    }
}
