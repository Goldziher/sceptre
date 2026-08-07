//! CTC prefix beam-search decoding.
//!
//! Reference: EasyOCR `utils.py` (`ctcBeamSearch` / `CTCLabelConverter.decode_beamsearch`,
//! itself adapted from <https://github.com/githubharald/CTCDecoder>). Unlike greedy
//! decoding, beam search sums the probability of every path that collapses to the
//! same label, not just the single most likely path, so it can recover a label whose
//! individually-most-likely path never wins a per-timestep argmax. See ADR 0036.
//!
//! EasyOCR always computes confidence from the greedy argmax over the same
//! probability matrix regardless of decoder (`recognition.py::recognizer_predict`
//! computes `preds_max_prob` once, before branching on `decoder`), so this module
//! reuses [`super::ctc::decode_row`] for confidence and only changes how the text is
//! produced.

use std::collections::HashMap;

use ndarray::ArrayView2;

use super::charset::Charset;
use super::ctc::{self, BLANK_CLASS, IgnoreMask};
use super::recognizer::RecognizedText;

/// Only classes at or above `0.5 / num_classes` extend a beam at a timestep, matching
/// `char_highscore = np.where(mat[t, :] >= 0.5 / maxC)[0]`: a performance cutoff, not
/// an accuracy one, since a slimmer margin than uniform-over-classes rarely wins the
/// label sum.
const HIGH_SCORE_FRACTION: f32 = 0.5;

/// One CTC beam's accumulated path mass: total, non-blank-ending, and blank-ending
/// probability, mirroring EasyOCR's `BeamEntry` (`prText`/`lmApplied` are omitted —
/// this port never applies a language model, matching `decode_beamsearch`'s `lm=None`).
#[derive(Debug, Default, Clone, Copy)]
struct BeamEntry {
    total: f32,
    non_blank: f32,
    blank: f32,
}

/// Beam-search-decode one crop's logits `[T, num_classes]` (raw, pre-softmax) into
/// text with a confidence, using `ignore_mask` to suppress allow/blocklisted classes
/// the same way greedy decoding does.
pub(super) fn decode_beam_search(
    logits: ArrayView2<f32>,
    charset: &Charset,
    ignore_mask: &IgnoreMask,
    beam_width: usize,
) -> RecognizedText {
    let probabilities = ctc::probability_matrix(logits, ignore_mask);
    let confidence = confidence_from(logits, ignore_mask);
    let labeling = ctc_beam_search(probabilities.view(), beam_width.max(1));
    let text = ctc::collapse_classes(labeling, charset);
    RecognizedText { text, confidence }
}

/// The same greedy-argmax confidence EasyOCR computes for every decoder.
fn confidence_from(logits: ArrayView2<f32>, ignore_mask: &IgnoreMask) -> f32 {
    let mut weights: Vec<f32> = Vec::with_capacity(logits.ncols());
    let per_timestep: Vec<(usize, f32)> = logits
        .rows()
        .into_iter()
        .map(|row| ctc::decode_row(row, ignore_mask, &mut weights))
        .collect();
    ctc::custom_mean(&ctc::collect_max_probs(&per_timestep))
}

/// Prefix beam search over an ignore-masked probability matrix `[T, num_classes]`,
/// returning the highest-total-mass labeling's raw class sequence (not yet collapsed).
///
/// Faithfully ports `ctcBeamSearch`'s label-merging behavior, including its blank
/// handling quirk: `char_highscore` (the per-timestep extension candidates) is not
/// blank-excluded, so an explicit blank extension and the implicit "beam unchanged"
/// continuation both contribute mass to the same blank-ending labeling. That is
/// upstream's actual algorithm, not a normalized probability update, and is kept
/// as-is rather than "fixed" so this decoder matches EasyOCR's `beamsearch` output.
fn ctc_beam_search(matrix: ArrayView2<f32>, beam_width: usize) -> Vec<usize> {
    let (timesteps, num_classes) = matrix.dim();
    let high_score_floor = HIGH_SCORE_FRACTION / num_classes as f32;

    let mut beams: HashMap<Vec<usize>, BeamEntry> = HashMap::new();
    beams.insert(
        Vec::new(),
        BeamEntry {
            total: 1.0,
            non_blank: 0.0,
            blank: 1.0,
        },
    );

    for t in 0..timesteps {
        let mut next: HashMap<Vec<usize>, BeamEntry> = HashMap::new();
        let mut ranked: Vec<(Vec<usize>, BeamEntry)> = beams.into_iter().collect();
        ranked.sort_by(|(a_labeling, a), (b_labeling, b)| rank_beams(a.total, a_labeling, b.total, b_labeling));
        ranked.truncate(beam_width);

        for (labeling, entry) in &ranked {
            extend_beam(&mut next, labeling, entry, matrix.row(t), high_score_floor);
        }
        beams = next;
    }

    beams
        .into_iter()
        .min_by(|(a_labeling, a), (b_labeling, b)| rank_beams(a.total, a_labeling, b.total, b_labeling))
        .map(|(labeling, _)| labeling)
        .unwrap_or_default()
}

/// Total order over `(total mass, labeling)`, highest mass first, ties broken by the
/// labeling's own `Ord` so beam pruning and the final pick are reproducible.
///
/// Because the order is descending, the best beam is the `min_by` under it, not the
/// `max_by`.
///
/// `HashMap`'s per-instance random seed makes iteration order — and so any float tie
/// under a bare mass comparison — vary between calls with identical input, which for a
/// decoder means the same crop could recognize to different text on different runs.
/// Breaking every tie on the labeling itself (unique per `HashMap` key) removes that
/// nondeterminism without pretending float equality is deterministic on its own.
fn rank_beams(a_total: f32, a_labeling: &[usize], b_total: f32, b_labeling: &[usize]) -> std::cmp::Ordering {
    b_total
        .partial_cmp(&a_total)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| a_labeling.cmp(b_labeling))
}

/// Advance one beam by one timestep: the "no new character" continuation, then an
/// extension attempt for every class whose probability clears `high_score_floor`.
fn extend_beam(
    next: &mut HashMap<Vec<usize>, BeamEntry>,
    labeling: &[usize],
    entry: &BeamEntry,
    timestep: ndarray::ArrayView1<f32>,
    high_score_floor: f32,
) {
    let last_class = labeling.last().copied();

    let continuation_non_blank = last_class.map_or(0.0, |last| entry.non_blank * timestep[last]);
    let continuation_blank = entry.total * timestep[BLANK_CLASS];
    let continuation = next.entry(labeling.to_vec()).or_default();
    continuation.non_blank += continuation_non_blank;
    continuation.blank += continuation_blank;
    continuation.total += continuation_blank + continuation_non_blank;

    for (class, &probability) in timestep.iter().enumerate() {
        if probability < high_score_floor {
            continue;
        }
        let extended = fast_simplify_label(labeling, class, BLANK_CLASS);
        let mass = if last_class == Some(class) {
            probability * entry.blank
        } else {
            probability * entry.total
        };
        let extension = next.entry(extended).or_default();
        extension.non_blank += mass;
        extension.total += mass;
    }
}

/// Extend `labeling` with class `c`, keeping the CTC invariant that a blank only
/// ever separates two equal non-blank classes (so "aa" and "a-a" stay distinguishable
/// while a blank between different classes is dropped immediately, not deferred to
/// the final collapse).
///
/// A direct port of `utils.py::fast_simplify_label`, whose last two branches
/// (`labeling and c != blankIdx`, and an `else` that re-derives the same result via
/// the slower `simplify_label`) collapse into one arm here because both always append.
fn fast_simplify_label(labeling: &[usize], c: usize, blank: usize) -> Vec<usize> {
    match labeling.last() {
        Some(&last) if c == blank && last != blank => append(labeling, c),
        Some(&last) if c != blank && last == blank => {
            // `labeling` ends in a blank preceded by a real class, per the CTC ~keep
            // invariant above; a labeling can only ever be `[blank]` on its own via ~keep
            // this branch, which never fires from an empty start (see the match arms ~keep
            // below), so `labeling.len() >= 2` here. ~keep
            if labeling[labeling.len() - 2] == c {
                append(labeling, c)
            } else {
                replace_last(labeling, c)
            }
        }
        Some(&last) if c == blank && last == blank => labeling.to_vec(),
        Some(_) => append(labeling, c),
        None if c == blank => Vec::new(),
        None => vec![c],
    }
}

fn append(labeling: &[usize], c: usize) -> Vec<usize> {
    let mut extended = Vec::with_capacity(labeling.len() + 1);
    extended.extend_from_slice(labeling);
    extended.push(c);
    extended
}

fn replace_last(labeling: &[usize], c: usize) -> Vec<usize> {
    let mut replaced = labeling[..labeling.len() - 1].to_vec();
    replaced.push(c);
    replaced
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Language;
    use ndarray::arr2;

    fn english() -> Charset {
        Charset::for_language(Language::English)
    }

    fn no_ignore(charset: &Charset) -> IgnoreMask {
        IgnoreMask::new(charset.num_classes(), &[])
    }

    #[test]
    fn should_leave_empty_labeling_unchanged_on_blank() {
        assert_eq!(fast_simplify_label(&[], 0, 0), Vec::<usize>::new());
    }

    #[test]
    fn should_append_first_non_blank_to_empty_labeling() {
        assert_eq!(fast_simplify_label(&[], 3, 0), vec![3]);
    }

    #[test]
    fn should_append_blank_after_a_non_blank_class() {
        assert_eq!(fast_simplify_label(&[5], 0, 0), vec![5, 0]);
    }

    #[test]
    fn should_leave_consecutive_blanks_unchanged() {
        assert_eq!(fast_simplify_label(&[5, 0], 0, 0), vec![5, 0]);
    }

    #[test]
    fn should_append_same_class_across_a_separating_blank() {
        // labeling[-2] (5) equals c (5), so the blank stays: "5-5" is a real repeat. ~keep
        assert_eq!(fast_simplify_label(&[5, 0], 5, 0), vec![5, 0, 5]);
    }

    #[test]
    fn should_drop_a_separating_blank_between_different_classes() {
        // labeling[-2] (5) differs from c (7): the blank was never load-bearing, so ~keep
        // it is replaced rather than kept. ~keep
        assert_eq!(fast_simplify_label(&[5, 0], 7, 0), vec![5, 7]);
    }

    #[test]
    fn should_append_a_repeated_non_blank_class_without_a_separator() {
        assert_eq!(fast_simplify_label(&[5], 5, 0), vec![5, 5]);
    }

    #[test]
    fn should_append_a_different_non_blank_class() {
        assert_eq!(fast_simplify_label(&[5], 7, 0), vec![5, 7]);
    }

    #[test]
    fn should_decode_a_single_dominant_path_like_greedy() {
        // Two timesteps, each overwhelmingly favoring one distinct non-blank class: ~keep
        // beam search and greedy must agree when there is no ambiguity to resolve. ~keep
        let charset = english();
        let mask = no_ignore(&charset);
        let logits = arr2(&[[0.0f32, 8.0, 0.0], [0.0, 0.0, 8.0]]);
        let beam = decode_beam_search(logits.view(), &charset, &mask, 5);
        let greedy = super::super::ctc::decode_greedy_with_mask(logits.view(), &charset, &mask);
        assert_eq!(beam.text, greedy.text);
        assert_eq!(beam.text, "01");
    }

    #[test]
    fn should_recover_a_label_that_best_path_greedy_misses() {
        // Best-path-vs-beam-search divergence in the spirit of Hannun's "Sequence ~keep
        // Modeling With CTC": with per-timestep probabilities [blank=0.4, a=0.35, ~keep
        // b=0.25] at every one of 3 steps, blank wins every per-timestep argmax (no ~keep
        // ties), so greedy's single most-likely PATH collapses to empty. Beam search ~keep
        // sums every path's mass per label instead of following the one most likely ~keep
        // path, and (numerically verified) that sum favors class 1 ('0') here. ~keep
        let charset = english();
        let mask = no_ignore(&charset);
        let row = [(0.4f32).ln(), (0.35f32).ln(), (0.25f32).ln()];
        let logits = arr2(&[row, row, row]);

        let greedy = super::super::ctc::decode_greedy_with_mask(logits.view(), &charset, &mask);
        assert_eq!(greedy.text, "", "the single most likely path is all-blank");

        let beam = decode_beam_search(logits.view(), &charset, &mask, 5);
        assert_eq!(beam.text, "0", "class 1 ('0') sums the most path mass across timesteps");
    }

    #[test]
    fn should_use_the_same_confidence_as_greedy_decoding() {
        // EasyOCR computes confidence identically for every decoder, from the greedy ~keep
        // argmax over the same probability matrix (recognition.py::recognizer_predict). ~keep
        let charset = english();
        let mask = no_ignore(&charset);
        let logits = arr2(&[[0.0f32, 8.0, 0.0], [0.0, 0.0, 8.0]]);
        let beam = decode_beam_search(logits.view(), &charset, &mask, 5);
        let greedy = super::super::ctc::decode_greedy_with_mask(logits.view(), &charset, &mask);
        assert_eq!(beam.confidence.to_bits(), greedy.confidence.to_bits());
    }

    #[test]
    fn should_respect_the_ignore_mask() {
        let charset = english();
        let ignored = [1usize];
        let mask = IgnoreMask::new(charset.num_classes(), &ignored);
        // Class 1 ('0') outscores class 2 ('1'), but is suppressed by the mask. ~keep
        let logits = arr2(&[[(0.1f32).ln(), (0.6f32).ln(), (0.3f32).ln()]]);
        let beam = decode_beam_search(logits.view(), &charset, &mask, 5);
        assert_eq!(beam.text, "1");
    }

    #[test]
    fn should_return_empty_text_for_an_all_blank_input() {
        let charset = english();
        let mask = no_ignore(&charset);
        let logits = arr2(&[[5.0f32, 0.0, 0.0], [5.0, 0.0, 0.0]]);
        let beam = decode_beam_search(logits.view(), &charset, &mask, 5);
        assert_eq!(beam.text, "");
        assert_eq!(beam.confidence, 0.0);
    }

    #[test]
    fn should_decode_deterministically_across_repeated_calls() {
        // `HashMap`'s per-instance random seed means two searches over identical input ~keep
        // can iterate beams in a different order; without the labeling tie-break in ~keep
        // `rank_beams`, the stable `sort_by` then preserves that arbitrary order among ~keep
        // equal-mass beams, so pruning silently keeps a different labeling and the same ~keep
        // crop recognizes to different text on different runs. ~keep
        //
        // Classes 1 and 2 carry *exactly* equal probability, so the labelings "0" and ~keep
        // "1" tie on mass at every timestep, and `beam_width` 1 forces the tie to decide ~keep
        // the survivor. Distinct probabilities would never exercise the tie-break at ~keep
        // all — verified by reverting `rank_beams` to a bare mass comparison, under ~keep
        // which this fixture fails and an unequal one still passes. ~keep
        let charset = english();
        let mask = no_ignore(&charset);
        let row = [(0.2f32).ln(), (0.4f32).ln(), (0.4f32).ln()];
        let logits = arr2(&[row, row, row, row]);
        let first = decode_beam_search(logits.view(), &charset, &mask, 1);
        for _ in 0..50 {
            let repeat = decode_beam_search(logits.view(), &charset, &mask, 1);
            assert_eq!(
                repeat.text, first.text,
                "beam search must decode the same crop identically every call"
            );
        }
    }

    #[test]
    fn should_tolerate_a_beam_width_of_one() {
        let charset = english();
        let mask = no_ignore(&charset);
        let logits = arr2(&[[0.0f32, 8.0, 0.0], [0.0, 0.0, 8.0]]);
        let beam = decode_beam_search(logits.view(), &charset, &mask, 1);
        assert_eq!(beam.text, "01");
    }
}
