//! Recognition hot-path microbenchmarks.
//!
//! Crop + batch preprocessing takes a full grayscale image, so it runs on a real
//! corpus image (with the offline fallback baked into `load_corpus_image`). CTC
//! decoding consumes model *output* — recognizer logits `[T, classes]` — not an
//! image, so synthetic logits are intrinsic to that stage, not a shortcut.

use criterion::{Criterion, criterion_group, criterion_main};
use image::{DynamicImage, GrayImage, RgbImage};
use sceptre::bench::{
    ctc_decode_for_benchmark, ctc_decode_reference_for_benchmark, english_class_count, load_corpus_image,
    recognize_crop_preprocess_for_benchmark, recognize_crop_preprocess_reference_for_benchmark, synthetic_logits,
};

/// Timesteps in the synthetic logits fed to the CTC decode bench.
const TIMESTEPS: usize = 256;
/// Fraction of the image width spanned by the benchmark crop region.
const REGION_WIDTH_FRACTION: f32 = 0.5;
/// Fraction of the image height spanned by the benchmark crop region.
const REGION_HEIGHT_FRACTION: f32 = 0.25;
/// Minimum extent (pixels) of the crop region, guarding tiny fallback images.
const MIN_REGION_EXTENT: f32 = 2.0;
/// Inset (pixels) of the crop region's top-left corner from the image origin.
const REGION_INSET: f32 = 1.0;

/// Decode a corpus image and convert it to grayscale for the crop stage.
fn corpus_gray(relative: &str) -> GrayImage {
    let image = load_corpus_image(relative);
    let rgb = RgbImage::from_raw(image.width(), image.height(), image.as_rgb8().to_vec())
        .expect("corpus RGB buffer is well-formed");
    DynamicImage::ImageRgb8(rgb).into_luma8()
}

/// An axis-aligned region within `gray`, sized as a fraction of its dimensions.
fn region_corners(gray: &GrayImage) -> [[f32; 2]; 4] {
    let x1 = (gray.width() as f32 * REGION_WIDTH_FRACTION).max(MIN_REGION_EXTENT);
    let y1 = (gray.height() as f32 * REGION_HEIGHT_FRACTION).max(MIN_REGION_EXTENT);
    [
        [REGION_INSET, REGION_INSET],
        [x1, REGION_INSET],
        [x1, y1],
        [REGION_INSET, y1],
    ]
}

fn bench_crop_preprocess(criterion: &mut Criterion) {
    let gray = corpus_gray("images/english_and_korean.png");
    let corners = region_corners(&gray);
    criterion.bench_function("recognize/crop_preprocess/optimized", |b| {
        b.iter(|| recognize_crop_preprocess_for_benchmark(&gray, &corners));
    });
    criterion.bench_function("recognize/crop_preprocess/reference", |b| {
        b.iter(|| recognize_crop_preprocess_reference_for_benchmark(&gray, &corners));
    });
}

fn bench_ctc_decode(criterion: &mut Criterion) {
    let logits = synthetic_logits(TIMESTEPS, english_class_count());
    criterion.bench_function("recognize/ctc_decode/optimized", |b| {
        b.iter(|| ctc_decode_for_benchmark(&logits));
    });
    criterion.bench_function("recognize/ctc_decode/reference", |b| {
        b.iter(|| ctc_decode_reference_for_benchmark(&logits));
    });
}

criterion_group!(recognition, bench_crop_preprocess, bench_ctc_decode);
criterion_main!(recognition);
