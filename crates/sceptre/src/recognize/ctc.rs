//! CTC decoding.
//!
//! Reference: EasyOCR `utils.py` (`CTCLabelConverter.decode_greedy`). Blank is
//! index 0; greedy decoding collapses repeats and drops blanks. Confidence uses
//! the `custom_mean` formula from `recognition.py`.

use ndarray::{Array2, ArrayView1, ArrayViewMut1};

use super::charset::Charset;
use super::recognizer::RecognizedText;

/// CTC blank class index. EasyOCR builds its class list as `['[blank]'] + chars`,
/// so class 0 is the blank symbol.
const BLANK_CLASS: usize = 0;

/// Numerator of the `custom_mean` exponent `2.0 / sqrt(n)`, from EasyOCR
/// `recognition.py::custom_mean`.
const CUSTOM_MEAN_EXPONENT_NUMERATOR: f32 = 2.0;

/// Greedy-decode one crop's logits `[T, num_classes]` (raw, pre-softmax) into text
/// with a confidence. `ignore` lists CTC class indices to suppress (allowlist/blocklist).
pub(crate) fn decode_greedy(logits: &Array2<f32>, charset: &Charset, ignore: &[usize]) -> RecognizedText {
    let probs = probability_rows(logits, ignore);
    let per_timestep: Vec<(usize, f32)> = probs.rows().into_iter().map(argmax_row).collect();
    let confidence = custom_mean(&collect_max_probs(&per_timestep));
    let text = collapse(&per_timestep, charset);
    RecognizedText { text, confidence }
}

/// Softmax each timestep row, zero the `ignore` columns, then renormalize the row.
///
/// Mirrors `recognition.py::recognizer_predict`: `softmax` → zero `ignore_idx` →
/// divide by the row sum (guarding a zero sum).
fn probability_rows(logits: &Array2<f32>, ignore: &[usize]) -> Array2<f32> {
    let mut probs = Array2::<f32>::zeros(logits.raw_dim());
    for (out_row, in_row) in probs.rows_mut().into_iter().zip(logits.rows()) {
        fill_probability_row(out_row, in_row, ignore);
    }
    probs
}

/// Fill one output row with the softmax of `input`, suppress `ignore` classes, and
/// renormalize. Both normalization steps guard against a zero sum.
fn fill_probability_row(mut out: ArrayViewMut1<f32>, input: ArrayView1<f32>, ignore: &[usize]) {
    let max = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for (cell, &value) in out.iter_mut().zip(input.iter()) {
        let weight = (value - max).exp();
        *cell = weight;
        sum += weight;
    }
    if sum > 0.0 {
        out.iter_mut().for_each(|cell| *cell /= sum);
    }
    for &class in ignore {
        if let Some(cell) = out.get_mut(class) {
            *cell = 0.0;
        }
    }
    let renorm: f32 = out.iter().sum();
    if renorm > 0.0 {
        out.iter_mut().for_each(|cell| *cell /= renorm);
    }
}

/// The `(class, probability)` of the highest-scoring class in a timestep row.
///
/// Ties resolve to the first class, matching `numpy.argmax`.
fn argmax_row(row: ArrayView1<f32>) -> (usize, f32) {
    let mut best_class = BLANK_CLASS;
    let mut best_prob = f32::NEG_INFINITY;
    for (class, &prob) in row.iter().enumerate() {
        if prob > best_prob {
            best_prob = prob;
            best_class = class;
        }
    }
    (best_class, best_prob)
}

/// The argmax probability at every non-blank timestep, collected before collapsing
/// repeats. An all-blank sequence yields `[0.0]`, matching `recognizer_predict`.
fn collect_max_probs(per_timestep: &[(usize, f32)]) -> Vec<f32> {
    let max_probs: Vec<f32> = per_timestep
        .iter()
        .filter(|(class, _)| *class != BLANK_CLASS)
        .map(|(_, prob)| *prob)
        .collect();
    if max_probs.is_empty() { vec![0.0] } else { max_probs }
}

/// Greedy CTC collapse: keep a timestep iff its class differs from the previous one
/// and is not the blank, then map each kept class through `charset`.
fn collapse(per_timestep: &[(usize, f32)], charset: &Charset) -> String {
    let mut text = String::new();
    let mut previous: Option<usize> = None;
    for &(class, _) in per_timestep {
        let is_new = previous != Some(class);
        previous = Some(class);
        if is_new && class != BLANK_CLASS {
            if let Some(character) = charset.char_at_class(class) {
                text.push(character);
            }
        }
    }
    text
}

/// EasyOCR's confidence: `product(values).powf(2.0 / sqrt(values.len()))`.
fn custom_mean(values: &[f32]) -> f32 {
    let product: f32 = values.iter().product();
    let exponent = CUSTOM_MEAN_EXPONENT_NUMERATOR / (values.len() as f32).sqrt();
    product.powf(exponent)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Language;
    use ndarray::arr2;

    fn english() -> Charset {
        Charset::for_language(Language::English)
    }

    #[test]
    fn should_decode_two_distinct_timesteps_into_two_chars() {
        // Columns are [blank, class 1 = '0', class 2 = '1']; each timestep favors a
        // different non-blank class, so no collapse occurs. ~keep
        let logits = arr2(&[[0.0f32, 5.0, 0.0], [0.0, 0.0, 5.0]]);
        let decoded = decode_greedy(&logits, &english(), &[]);
        assert_eq!(decoded.text, "01");
        assert!(decoded.confidence > 0.0, "two confident timesteps score above zero");
    }

    #[test]
    fn should_collapse_repeated_argmax_into_single_char() {
        let logits = arr2(&[[0.0f32, 5.0, 0.0], [0.0, 5.0, 0.0]]);
        let decoded = decode_greedy(&logits, &english(), &[]);
        assert_eq!(decoded.text, "0");
    }

    #[test]
    fn should_return_empty_text_and_zero_confidence_when_all_blank() {
        let logits = arr2(&[[5.0f32, 0.0, 0.0], [5.0, 0.0, 0.0]]);
        let decoded = decode_greedy(&logits, &english(), &[]);
        assert_eq!(decoded.text, "");
        assert_eq!(decoded.confidence, 0.0);
    }

    #[test]
    fn should_compute_exact_custom_mean_confidence() {
        // Softmax([ln 0.25, ln 0.75]) = [0.25, 0.75]; argmax is class 1 ('0') at
        // prob 0.75, so custom_mean([0.75]) = 0.75.powf(2.0 / sqrt(1)) = 0.5625. ~keep
        let logits = arr2(&[[(0.25f32).ln(), (0.75f32).ln()]]);
        let decoded = decode_greedy(&logits, &english(), &[]);
        assert_eq!(decoded.text, "0");
        assert!(
            (decoded.confidence - 0.5625).abs() < 1e-5,
            "confidence {} should equal 0.5625",
            decoded.confidence
        );
    }

    #[test]
    fn should_let_ignore_change_the_decoded_character() {
        // Class 1 ('0') outscores class 2 ('1'); ignoring class 1 lets class 2 win. ~keep
        let logits = arr2(&[[(0.1f32).ln(), (0.6f32).ln(), (0.3f32).ln()]]);
        let without_ignore = decode_greedy(&logits, &english(), &[]);
        assert_eq!(without_ignore.text, "0");
        let with_ignore = decode_greedy(&logits, &english(), &[1]);
        assert_eq!(with_ignore.text, "1");
    }
}
