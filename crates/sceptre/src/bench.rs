//! Benchmark seam: thin `#[doc(hidden)]` wrappers over the crate-internal
//! detection and recognition hot paths, plus deterministic input builders.
//!
//! The detection and recognition stages are `pub(crate)`, so they cannot be
//! measured from an external `benches/` crate without widening the public API.
//! This module — compiled only under the `bench` cargo feature and hidden from
//! the docs — exposes just enough surface to drive those stages from Criterion
//! benchmarks. Every wrapper binds its result and hands it to
//! [`std::hint::black_box`] so the optimizer cannot elide the work; wrappers
//! return `()` to avoid naming crate-private result types across the seam. See
//! ADR 0015.

use std::hint::black_box;
use std::path::{Path, PathBuf};

use image::GrayImage;
use ndarray::Array2;

use crate::config::{DetectionConfig, Language};
use crate::recognize::Charset;
use crate::types::Image;

/// Subdirectory of the `test_documents` corpus holding benchmark images.
const CORPUS_IMAGE_SUBDIR: &str = "images";

/// The repository root, two levels up from this crate's manifest directory
/// (`<root>/crates/sceptre`).
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Directory holding the `test_documents` corpus: `TEST_DOCUMENTS_DIR` when set, otherwise
/// the `test_documents` submodule checked out at the repository root.
fn test_documents_dir() -> PathBuf {
    std::env::var_os("TEST_DOCUMENTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root().join("test_documents"))
}

/// Region-score value written into synthetic heat-map blobs; above the default
/// `text_threshold` (0.7) and `low_text` (0.4).
const BLOB_SCORE: f32 = 0.9;
/// Faint link-score baseline; below the default `link_threshold` (0.4).
const LINK_BASELINE: f32 = 0.1;
/// Side length, in pixels, of one synthetic heat-map blob.
const BLOB_BLOCK: usize = 6;

/// Horizontal stride between successive synthetic boxes.
const BOX_STRIDE: f32 = 45.0;
/// Width of each synthetic box.
const BOX_WIDTH: f32 = 40.0;
/// Top edge (y) of the synthetic box row.
const BOX_TOP: f32 = 10.0;
/// Height of each synthetic box.
const BOX_HEIGHT: f32 = 20.0;

/// Slope of the per-timestep logit ramp; larger values sharpen the argmax peak.
const LOGIT_SLOPE: f32 = 0.5;

/// `inv_ratio` fed to [`detect_postprocess_for_benchmark`]'s coordinate-adjust
/// step; any positive scale exercises the same work.
const BENCH_INV_RATIO: f32 = 2.0;

/// Load a corpus image by filename from the `test_documents` corpus (see
/// [`test_documents_dir`]). Fails loudly rather than falling back to a substitute image when
/// the corpus is absent or the file is missing.
pub fn load_corpus_image(filename: &str) -> Image {
    let path = test_documents_dir().join(CORPUS_IMAGE_SUBDIR).join(filename);
    Image::from_path(&path).unwrap_or_else(|error| panic!("failed to decode corpus image {}: {error}", path.display()))
}

/// Build synthetic CRAFT `(region, link)` heat-maps of shape `[height, width]`.
///
/// The region map carries a grid of high-score rectangular blobs (so
/// connected-component labelling has realistic work), and the link map holds a
/// uniform sub-threshold baseline. Postprocessing consumes model *output*, not
/// an image, so synthetic heat-maps are the intrinsic input for that stage.
pub fn synthetic_heatmaps(height: usize, width: usize) -> (Array2<f32>, Array2<f32>) {
    let mut region = Array2::<f32>::zeros((height, width));
    let mut link = Array2::<f32>::zeros((height, width));

    let step = BLOB_BLOCK * 2;
    let mut y = BLOB_BLOCK;
    while y + BLOB_BLOCK < height {
        let mut x = BLOB_BLOCK;
        while x + BLOB_BLOCK < width {
            for row in y..(y + BLOB_BLOCK) {
                for col in x..(x + BLOB_BLOCK) {
                    region[[row, col]] = BLOB_SCORE;
                }
            }
            x += step;
        }
        y += step;
    }
    link.fill(LINK_BASELINE);
    (region, link)
}

/// Build `count` axis-aligned boxes (corners `[TL, TR, BR, BL]`) laid out in a
/// single row with overlapping strides, so line grouping has adjacent boxes to
/// merge. Grouping consumes detector output, so synthetic boxes are intrinsic.
pub fn synthetic_boxes(count: usize) -> Vec<[[f32; 2]; 4]> {
    (0..count)
        .map(|index| {
            let x0 = index as f32 * BOX_STRIDE;
            let x1 = x0 + BOX_WIDTH;
            let (y0, y1) = (BOX_TOP, BOX_TOP + BOX_HEIGHT);
            [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        })
        .collect()
}

/// Build deterministic recognizer logits of shape `[timesteps, classes]`.
///
/// Each timestep has a single peak class (`t % classes`) formed by a symmetric
/// ramp, so greedy CTC decoding selects varied non-blank classes and exercises
/// the repeat-collapse path. CTC decoding consumes model output, so synthetic
/// logits are the intrinsic input for that stage.
pub fn synthetic_logits(timesteps: usize, classes: usize) -> Array2<f32> {
    let mut logits = Array2::<f32>::zeros((timesteps, classes));
    let span = classes.max(1);
    for t in 0..timesteps {
        let peak = (t % span) as f32;
        for c in 0..classes {
            logits[[t, c]] = -(c as f32 - peak).abs() * LOGIT_SLOPE;
        }
    }
    logits
}

/// Number of CTC classes for the English gen2 charset, including the blank.
/// Lets an external bench size [`synthetic_logits`] to a realistic class count.
pub fn english_class_count() -> usize {
    english_charset().num_classes()
}

/// The English gen2 charset (crate-internal; used by [`ctc_decode_for_benchmark`]).
pub(crate) fn english_charset() -> Charset {
    Charset::for_language(Language::English)
}

/// Drive detection preprocessing (aspect-ratio resize + ImageNet normalization)
/// on a full image.
pub fn detect_preprocess_for_benchmark(image: &Image, canvas_size: u32, mag_ratio: f32) {
    let tensor = crate::detect::bench_preprocess(image, canvas_size, mag_ratio).expect("corpus image preprocesses");
    black_box(&tensor);
}

/// A/B baseline twin of [`detect_preprocess_for_benchmark`] driving the reference
/// (pre-optimization) detection normalize.
pub fn detect_preprocess_reference_for_benchmark(image: &Image, canvas_size: u32, mag_ratio: f32) {
    let tensor =
        crate::detect::bench_preprocess_reference(image, canvas_size, mag_ratio).expect("corpus image preprocesses");
    black_box(&tensor);
}

/// Drive detection postprocessing (threshold + connected components → boxes,
/// then coordinate adjustment) on synthetic heat-maps.
pub fn detect_postprocess_for_benchmark(
    region: &Array2<f32>,
    link: &Array2<f32>,
    text_threshold: f32,
    link_threshold: f32,
    low_text: f32,
) {
    let boxes =
        crate::detect::bench_postprocess(region, link, text_threshold, link_threshold, low_text, BENCH_INV_RATIO)
            .expect("synthetic heat-maps decode to boxes");
    black_box(&boxes);
}

/// Drive line grouping (slope split + line merge + margin) on synthetic boxes.
pub fn detect_group_for_benchmark(boxes: &[[[f32; 2]; 4]], config: &DetectionConfig) {
    let grouped = crate::detect::bench_group(boxes, config);
    black_box(&grouped);
}

/// Drive recognition crop + batch preprocessing: crop one axis-aligned region
/// from the grayscale image, then normalize and pad it into a batch tensor.
pub fn recognize_crop_preprocess_for_benchmark(gray: &GrayImage, corners: &[[f32; 2]; 4]) {
    let tensor = crate::recognize::bench_crop_preprocess(gray, corners).expect("crop preprocesses to a batch");
    black_box(&tensor);
}

/// A/B baseline twin of [`recognize_crop_preprocess_for_benchmark`] driving the
/// reference (pre-optimization) recognition normalize.
pub fn recognize_crop_preprocess_reference_for_benchmark(gray: &GrayImage, corners: &[[f32; 2]; 4]) {
    let tensor =
        crate::recognize::bench_crop_preprocess_reference(gray, corners).expect("crop preprocesses to a batch");
    black_box(&tensor);
}

/// Drive greedy CTC decoding (softmax + argmax + collapse + confidence) on
/// synthetic logits, using the English charset and no ignored classes.
pub fn ctc_decode_for_benchmark(logits: &Array2<f32>) {
    let decoded = crate::recognize::bench_ctc_decode(logits.view());
    black_box(&decoded);
}

/// A/B baseline twin of [`ctc_decode_for_benchmark`] driving the reference
/// (materialized `Array2`) greedy CTC decoder.
pub fn ctc_decode_reference_for_benchmark(logits: &Array2<f32>) {
    let decoded = crate::recognize::bench_ctc_decode_reference(logits.view());
    black_box(&decoded);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_load_a_committed_corpus_image() {
        // Skip (rather than fail) when the corpus is absent or this particular fixture has
        // not been fetched/published yet; `should_panic_loading_a_missing_corpus_image`
        // below still proves a genuinely wrong filename fails loudly. ~keep
        let path = test_documents_dir().join(CORPUS_IMAGE_SUBDIR).join("english.png");
        if !path.exists() {
            return;
        }
        let image = load_corpus_image("english.png");
        assert!(image.width() > 0, "corpus image has non-zero width");
        assert!(image.height() > 0, "corpus image has non-zero height");
    }

    #[test]
    #[should_panic(expected = "failed to decode corpus image")]
    fn should_panic_loading_a_missing_corpus_image() {
        load_corpus_image("does_not_exist_in_any_corpus.png");
    }

    #[test]
    fn should_build_heatmaps_with_requested_shape_and_a_high_score_blob() {
        let (region, link) = synthetic_heatmaps(40, 60);
        assert_eq!(region.dim(), (40, 60));
        assert_eq!(link.dim(), (40, 60));
        let region_max = region.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            (region_max - BLOB_SCORE).abs() < 1e-6,
            "region contains a high-score blob"
        );
    }

    #[test]
    fn should_build_the_requested_number_of_boxes() {
        let boxes = synthetic_boxes(7);
        assert_eq!(boxes.len(), 7);
    }

    #[test]
    fn should_build_logits_with_requested_shape() {
        let logits = synthetic_logits(12, 97);
        assert_eq!(logits.dim(), (12, 97));
    }

    #[test]
    fn english_class_count_includes_the_blank() {
        assert!(english_class_count() > 1, "charset has classes beyond the blank");
    }
}
