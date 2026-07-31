//! Low-confidence second-pass contrast adjustment.
//!
//! Reference: EasyOCR `recognition.py` (`contrast_grey` / `adjust_contrast_grey`).
//! When a crop's contrast is below `contrast_ths`, it is re-normalized using the
//! 10th/90th percentile range and recognized again.

/// Percentile for the dark reference point (`np.percentile(img, 10)`).
const PCT_LOW: f32 = 10.0;
/// Percentile for the bright reference point (`np.percentile(img, 90)`).
const PCT_HIGH: f32 = 90.0;
/// Divisor turning a percentile into a `[0, 1]` fraction of the sorted span.
const PERCENT_SCALE: f32 = 100.0;
/// Floor on `high + low` in the contrast ratio's denominator (`np.maximum(10, ...)`).
const SUM_FLOOR: f32 = 10.0;
/// Floor on `high - low` when deriving the stretch ratio (`np.maximum(10, ...)`).
const RANGE_FLOOR: f32 = 10.0;
/// Numerator of the stretch ratio (`200. / max(10, high - low)`).
const CONTRAST_NUM: f32 = 200.0;
/// Additive shift applied before scaling (`img - low + 25`).
const SHIFT: f32 = 25.0;
/// Lower clamp for the adjusted pixel value.
const U8_MIN: f32 = 0.0;
/// Upper clamp for the adjusted pixel value.
const U8_MAX: f32 = 255.0;

/// Linear-interpolated percentile over an ascending-sorted slice, matching
/// numpy's default (`np.percentile`): `rank = p/100 * (n - 1)`, interpolating
/// between the floor and ceil ranks.
fn percentile(sorted: &[f32], percent: f32) -> f32 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted[0];
    }
    let rank = percent / PERCENT_SCALE * (n - 1) as f32;
    let lower = rank.floor();
    let lower_index = lower as usize;
    let upper_index = rank.ceil() as usize;
    let lower_value = sorted[lower_index];
    let upper_value = sorted[upper_index];
    lower_value + (upper_value - lower_value) * (rank - lower)
}

/// `(contrast, high, low)` where `high`/`low` are the 90th/10th percentiles of `gray`
/// and `contrast = (high - low) / max(10, high + low)`. Empty input → `(0.0, 0.0, 0.0)`.
pub(crate) fn contrast_grey(gray: &[u8]) -> (f32, f32, f32) {
    if gray.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut sorted: Vec<f32> = gray.iter().map(|&value| value as f32).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let high = percentile(&sorted, PCT_HIGH);
    let low = percentile(&sorted, PCT_LOW);
    let contrast = (high - low) / (high + low).max(SUM_FLOOR);
    (contrast, high, low)
}

/// Return contrast-adjusted grayscale (same length). If `contrast_grey < target`,
/// remap `v -> (v - low + 25) * (200 / max(10, high - low))` clamped to `[0, 255]`;
/// otherwise return the pixels unchanged.
pub(crate) fn adjust_contrast_grey(gray: &[u8], target: f32) -> Vec<u8> {
    let (contrast, high, low) = contrast_grey(gray);
    if contrast >= target {
        return gray.to_vec();
    }
    let ratio = CONTRAST_NUM / (high - low).max(RANGE_FLOOR);
    gray.iter()
        .map(|&value| {
            let adjusted = ((value as f32) - low + SHIFT) * ratio;
            adjusted.clamp(U8_MIN, U8_MAX).trunc() as u8
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const FLOAT_TOLERANCE: f32 = 1e-4;

    fn assert_close(actual: f32, expected: f32, label: &str) {
        assert!(
            (actual - expected).abs() < FLOAT_TOLERANCE,
            "{label}: expected {expected}, got {actual}"
        );
    }

    #[test]
    fn should_compute_linear_interpolated_percentiles_and_contrast() {
        // sorted: [0, 50, 100, 150, 200], n = 5.
        // high (90th): rank = 0.9 * 4 = 3.6 -> 150 + (200-150)*0.6 = 180.
        // low  (10th): rank = 0.1 * 4 = 0.4 ->   0 + (50-0)*0.4   = 20.
        // contrast = (180 - 20) / max(10, 200) = 160 / 200 = 0.8.
        let gray = [0u8, 50, 100, 150, 200];
        let (contrast, high, low) = contrast_grey(&gray);
        assert_close(high, 180.0, "high");
        assert_close(low, 20.0, "low");
        assert_close(contrast, 0.8, "contrast");
    }

    #[test]
    fn should_stretch_low_contrast_input() {
        // sorted: [70,80,90,100,110,120,130,140,150,180,200], n = 11.
        // high (90th): rank = 0.9 * 10 = 9 -> 180. low (10th): rank = 1 -> 80.
        // contrast = (180 - 80) / max(10, 260) = 100 / 260 ~= 0.3846 < 0.4 -> stretch.
        // ratio = 200 / max(10, 100) = 2.0; v -> (v - 80 + 25) * 2, clamped.
        let gray = [70u8, 80, 90, 100, 110, 120, 130, 140, 150, 180, 200];
        let (contrast, _, _) = contrast_grey(&gray);
        assert!(contrast < 0.4, "input must be low-contrast, got {contrast}");
        let adjusted = adjust_contrast_grey(&gray, 0.4);
        let expected = vec![30u8, 50, 70, 90, 110, 130, 150, 170, 190, 250, 255];
        assert_eq!(adjusted, expected);
        // Spread widens from 130 (200-70) to 225 (255-30).
        let input_spread = *gray.iter().max().unwrap() - *gray.iter().min().unwrap();
        let output_spread = *adjusted.iter().max().unwrap() - *adjusted.iter().min().unwrap();
        assert!(
            output_spread > input_spread,
            "output spread {output_spread} should exceed input spread {input_spread}"
        );
    }

    #[test]
    fn should_return_high_contrast_input_unchanged() {
        // contrast = 0.8 >= target 0.4 -> pixels returned unchanged.
        let gray = [0u8, 50, 100, 150, 200];
        let adjusted = adjust_contrast_grey(&gray, 0.4);
        assert_eq!(adjusted, gray.to_vec());
    }

    #[test]
    fn should_preserve_output_length_when_stretching() {
        let gray = [100u8, 101, 102, 103, 104, 105, 106, 107, 108, 109];
        let adjusted = adjust_contrast_grey(&gray, 0.4);
        assert_eq!(adjusted.len(), gray.len());
    }

    #[test]
    fn should_return_zeros_for_empty_input() {
        assert_eq!(contrast_grey(&[]), (0.0, 0.0, 0.0));
    }

    #[test]
    fn should_return_empty_vec_for_empty_input() {
        assert_eq!(adjust_contrast_grey(&[], 0.4), Vec::<u8>::new());
    }
}
