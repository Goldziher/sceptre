//! CJK bigram tokenizer, greedy matching, and reading-order scoring.
//!
//! Ported from xberg's `tools/benchmark-harness`:
//!
//! * [`tokenize`] -- NFKC normalization, invisible-character stripping, and CJK bigram
//!   expansion (`src/quality.rs::tokenize` / `tokenize_cjk_bigrams`), minus the
//!   Markdown-link stripping and thousands-separator normalization that harness pulls in
//!   for extracted document text -- neither applies to a single OCR line.
//! * [`greedy_match`] / [`f1_parts_from`] -- the matching machinery from
//!   `src/structural_sidecar.rs`, generalized here over any similarity function rather
//!   than SF1's Markdown-structure scoring. Applied to lines (box-IoU) instead of
//!   headings/tables/lists.
//! * [`reading_order_score`] -- anchor-LIS ordering score (`src/quality.rs`).
//!
//! The Python mirror lives in `python/sceptre_rs_tools/text_metrics.py`; the two are kept
//! in lockstep by `tests/data/metrics_vectors.json`, asserted at 1e-6 by both
//! `tests/helpers/mod.rs` and `python/tests/test_metrics_vectors.py`.

use unicode_normalization::UnicodeNormalization;

/// A token occurring more than once on either side is ambiguous as a reading-order anchor
/// (which occurrence maps to which?), so anchors require an exact one-to-one match. Below
/// this many anchors, an ordering score is noise rather than a signal. xberg's own
/// `MIN_READING_ORDER_ANCHORS` value was not available to this port at the time it was
/// written; 3 is this port's own choice (not a verified copy of xberg's constant) and
/// should be reconciled against xberg's source before being treated as load-bearing. ~keep
pub const MIN_READING_ORDER_ANCHORS: usize = 3;

const INVISIBLE_CHARACTERS: [char; 5] = [
    '\u{200b}', // zero-width space ~keep
    '\u{200c}', // zero-width non-joiner ~keep
    '\u{200d}', // zero-width joiner ~keep
    '\u{feff}', // BOM ~keep
    '\u{00ad}', // soft hyphen ~keep
];

/// True for CJK ideographs, kana, and Hangul syllables.
fn is_cjk_character(character: char) -> bool {
    let code = character as u32;
    (0x4E00..=0x9FFF).contains(&code) // CJK Unified Ideographs ~keep
        || (0x3400..=0x4DBF).contains(&code) // CJK Extension A ~keep
        || (0x3040..=0x30FF).contains(&code) // Hiragana + Katakana ~keep
        || (0xAC00..=0xD7A3).contains(&code) // Hangul syllables ~keep
        || (0xF900..=0xFAFF).contains(&code) // CJK compatibility ideographs ~keep
}

/// Overlapping bigrams of a CJK run; a single character stays a unigram.
fn cjk_bigrams(run: &[char]) -> Vec<String> {
    if run.len() <= 1 {
        return run.iter().collect::<String>().chars().map(String::from).collect();
    }
    run.windows(2).map(|pair| pair.iter().collect()).collect()
}

/// Split a token into maximal runs of (is_cjk) characters, in order.
fn split_runs(token: &str) -> Vec<(Vec<char>, bool)> {
    let mut runs: Vec<(Vec<char>, bool)> = Vec::new();
    for character in token.chars() {
        let is_cjk = is_cjk_character(character);
        match runs.last_mut() {
            Some((buffer, buffer_is_cjk)) if *buffer_is_cjk == is_cjk => buffer.push(character),
            _ => runs.push((vec![character], is_cjk)),
        }
    }
    runs
}

/// Bigram-expand the CJK runs of one whitespace-delimited token; keep other runs whole.
fn expand_token(token: &str) -> Vec<String> {
    let mut expanded = Vec::new();
    for (run, is_cjk) in split_runs(token) {
        if is_cjk {
            expanded.extend(cjk_bigrams(&run));
        } else {
            expanded.push(run.into_iter().collect());
        }
    }
    expanded
}

/// NFKC-normalize, strip invisible characters, lowercase, and CJK-bigram-expand.
///
/// Whitespace-delimited scripts (English, Cyrillic, ...) tokenize as before -- one token
/// per whitespace-separated run. A CJK run within a token expands into overlapping
/// bigrams (a lone CJK character stays a unigram), so word-level F1 scores CJK text on
/// adjacent-character pairs instead of treating an entire line as one degenerate token.
pub fn tokenize(text: &str) -> Vec<String> {
    let normalized: String = text.nfkc().collect();
    let cleaned: String = normalized
        .chars()
        .filter(|c| !INVISIBLE_CHARACTERS.contains(c))
        .collect();
    let lowered = cleaned.to_lowercase().replace('|', " ");
    lowered.split_whitespace().flat_map(expand_token).collect()
}

/// One greedy-matched (prediction, reference) index pair and its similarity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatchedPair {
    /// Index into the prediction slice.
    pub pred_index: usize,
    /// Index into the reference slice.
    pub ref_index: usize,
    /// The similarity score that earned this pair its match.
    pub score: f32,
}

/// Greedily pair predictions with references by descending similarity.
///
/// Every candidate pair scoring at least `threshold` is considered, highest similarity
/// first; each prediction and each reference index is used in at most one pair. This is
/// the matching machinery from xberg's `structural_sidecar.rs::greedy_match`, generalized
/// over any similarity function (here: box-IoU between detected and reference lines,
/// rather than SF1's Markdown-structure content similarity).
pub fn greedy_match<P, R>(
    predicted: &[P],
    reference: &[R],
    similarity: impl Fn(&P, &R) -> f32,
    threshold: f32,
) -> Vec<MatchedPair> {
    let mut candidates: Vec<(f32, usize, usize)> = Vec::new();
    for (pred_index, pred) in predicted.iter().enumerate() {
        for (ref_index, ref_item) in reference.iter().enumerate() {
            let score = similarity(pred, ref_item);
            if score >= threshold {
                candidates.push((score, pred_index, ref_index));
            }
        }
    }
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut used_preds = vec![false; predicted.len()];
    let mut used_refs = vec![false; reference.len()];
    let mut matches = Vec::new();
    for (score, pred_index, ref_index) in candidates {
        if used_preds[pred_index] || used_refs[ref_index] {
            continue;
        }
        used_preds[pred_index] = true;
        used_refs[ref_index] = true;
        matches.push(MatchedPair {
            pred_index,
            ref_index,
            score,
        });
    }
    matches
}

/// `(f1, precision, recall)` from a matched credit against prediction/reference counts.
///
/// Both-empty scores `(1.0, 1.0, 1.0)`; one-empty scores `(0.0, 0.0, 0.0)` -- ported from
/// xberg's `structural_sidecar.rs::f1_parts_from`.
pub fn f1_parts_from(matched_credit: f32, n_pred: usize, n_ref: usize) -> (f32, f32, f32) {
    if n_pred == 0 && n_ref == 0 {
        return (1.0, 1.0, 1.0);
    }
    if n_pred == 0 || n_ref == 0 {
        return (0.0, 0.0, 0.0);
    }
    let precision = matched_credit / n_pred as f32;
    let recall = matched_credit / n_ref as f32;
    if precision + recall == 0.0 {
        return (0.0, precision, recall);
    }
    let f1 = 2.0 * precision * recall / (precision + recall);
    (f1, precision, recall)
}

/// Length of the longest strictly increasing subsequence, via patience sorting (O(n log n)).
fn longest_increasing_subsequence_len(values: &[usize]) -> usize {
    let mut tails: Vec<usize> = Vec::new();
    for &value in values {
        match tails.binary_search(&value) {
            Ok(position) | Err(position) => {
                if position == tails.len() {
                    tails.push(value);
                } else {
                    tails[position] = value;
                }
            }
        }
    }
    tails.len()
}

/// Anchor-LIS reading-order agreement in `[0, 1]`, or `None` below the anchor floor.
///
/// An "anchor" is a token that occurs exactly once in both `hypothesis_text` and
/// `reference_text` -- unambiguous, so its position in one sequence can be compared to
/// its position in the other. Anchors are ordered by their reference-text position, and
/// the score is the length of the longest increasing subsequence of their
/// hypothesis-text positions, divided by the anchor count: `1.0` means every anchor
/// appears in the same relative order in both texts. Ported from xberg's
/// `quality.rs::reading_order_score`; catches column-order and vertical-script ordering
/// regressions that bag-of-words F1 and box-IoU are both blind to.
pub fn reading_order_score(hypothesis_text: &str, reference_text: &str) -> Option<f32> {
    let hypothesis_tokens = tokenize(hypothesis_text);
    let reference_tokens = tokenize(reference_text);
    if hypothesis_tokens.is_empty() || reference_tokens.is_empty() {
        return None;
    }

    let hypothesis_positions = token_positions(&hypothesis_tokens);
    let reference_positions = token_positions(&reference_tokens);

    let mut anchors: Vec<(usize, usize)> = reference_positions
        .iter()
        .filter_map(|(token, ref_positions)| {
            if ref_positions.len() != 1 {
                return None;
            }
            let hyp_positions = hypothesis_positions.get(token)?;
            if hyp_positions.len() != 1 {
                return None;
            }
            Some((ref_positions[0], hyp_positions[0]))
        })
        .collect();
    if anchors.len() < MIN_READING_ORDER_ANCHORS {
        return None;
    }

    anchors.sort_by_key(|&(ref_position, _)| ref_position);
    let hypothesis_order: Vec<usize> = anchors.iter().map(|&(_, hyp_position)| hyp_position).collect();
    Some(longest_increasing_subsequence_len(&hypothesis_order) as f32 / anchors.len() as f32)
}

/// Map each distinct token to every index it occurs at, in order.
fn token_positions(tokens: &[String]) -> std::collections::HashMap<&str, Vec<usize>> {
    let mut positions: std::collections::HashMap<&str, Vec<usize>> = std::collections::HashMap::new();
    for (index, token) in tokens.iter().enumerate() {
        positions.entry(token.as_str()).or_default().push(index);
    }
    positions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_keep_latin_words_whole() {
        assert_eq!(tokenize("Easy OCR rocks"), vec!["easy", "ocr", "rocks"]);
    }

    #[test]
    fn should_expand_cjk_runs_into_overlapping_bigrams() {
        assert_eq!(
            tokenize("清潔できれいな"),
            vec!["清潔", "潔で", "でき", "きれ", "れい", "いな"]
        );
    }

    #[test]
    fn should_keep_a_lone_cjk_character_as_a_unigram() {
        assert_eq!(tokenize("清"), vec!["清"]);
    }

    #[test]
    fn should_treat_pipe_as_whitespace() {
        assert_eq!(tokenize("a|b"), vec!["a", "b"]);
    }

    #[test]
    fn should_strip_invisible_characters() {
        assert_eq!(tokenize("a\u{200b}b"), tokenize("ab"));
    }

    #[test]
    fn should_apply_nfkc_normalization() {
        // U+FF21 (fullwidth "A") NFKC-normalizes to ASCII "a" after lowercasing. ~keep
        assert_eq!(tokenize("\u{FF21}"), vec!["a"]);
    }

    #[test]
    fn should_split_mixed_cjk_and_latin_within_one_token() {
        assert_eq!(tokenize("画像1234test"), vec!["画像", "1234test"]);
    }

    #[test]
    fn should_tokenize_empty_string_as_empty() {
        assert!(tokenize("").is_empty());
    }

    #[test]
    fn should_pair_greedy_matches_by_descending_similarity() {
        let similarity = |a: &i32, b: &i32| match (a, b) {
            (0, 0) => 0.9,
            (0, 1) => 0.1,
            (1, 0) => 0.2,
            (1, 1) => 0.8,
            _ => 0.0,
        };
        let matches = greedy_match(&[0, 1], &[0, 1], similarity, 0.0);
        let mut pairs: Vec<(usize, usize)> = matches.iter().map(|m| (m.pred_index, m.ref_index)).collect();
        pairs.sort();
        assert_eq!(pairs, vec![(0, 0), (1, 1)]);
    }

    #[test]
    fn should_exclude_greedy_match_pairs_below_threshold() {
        let matches = greedy_match(&[0], &[0], |_: &i32, _: &i32| 0.3, 0.5);
        assert!(matches.is_empty());
    }

    #[test]
    fn should_not_reuse_an_index_in_greedy_match() {
        let similarity = |a: &i32, _b: &i32| if *a == 0 { 0.9 } else { 0.8 };
        let matches = greedy_match(&[0, 1], &[0], similarity, 0.0);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].pred_index, 0);
        assert_eq!(matches[0].ref_index, 0);
    }

    #[test]
    fn should_score_f1_parts_of_both_empty_as_perfect() {
        assert_eq!(f1_parts_from(0.0, 0, 0), (1.0, 1.0, 1.0));
    }

    #[test]
    fn should_score_f1_parts_of_one_empty_as_zero() {
        assert_eq!(f1_parts_from(0.0, 3, 0), (0.0, 0.0, 0.0));
        assert_eq!(f1_parts_from(0.0, 0, 3), (0.0, 0.0, 0.0));
    }

    #[test]
    fn should_compute_f1_parts_precision_recall_and_harmonic_mean() {
        let (f1, precision, recall) = f1_parts_from(2.0, 4, 2);
        assert_eq!(precision, 0.5);
        assert_eq!(recall, 1.0);
        assert!((f1 - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn should_score_reading_order_of_identical_order_as_one() {
        let text = (0..MIN_READING_ORDER_ANCHORS)
            .map(|i| format!("token{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(reading_order_score(&text, &text), Some(1.0));
    }

    #[test]
    fn should_score_reading_order_of_fully_reversed_order() {
        let tokens: Vec<String> = (0..5).map(|i| format!("token{i}")).collect();
        let hypothesis = tokens.join(" ");
        let reference = tokens.iter().rev().cloned().collect::<Vec<_>>().join(" ");
        let score = reading_order_score(&hypothesis, &reference).expect("enough anchors");
        assert!((score - 1.0 / 5.0).abs() < 1e-6);
    }

    #[test]
    fn should_score_reading_order_as_none_below_the_anchor_floor() {
        let text = (0..MIN_READING_ORDER_ANCHORS - 1)
            .map(|i| format!("token{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(reading_order_score(&text, &text), None);
    }

    #[test]
    fn should_score_reading_order_as_none_for_empty_text() {
        assert_eq!(reading_order_score("", "something"), None);
        assert_eq!(reading_order_score("something", ""), None);
    }
}
