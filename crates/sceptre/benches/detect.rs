//! Detection hot-path microbenchmarks.
//!
//! Preprocessing takes a full image, so it runs on a real corpus image (a committed
//! fixture loaded by `load_corpus_image`). Postprocessing and grouping
//! consume model *output* — CRAFT region/link heat-maps and detector boxes — not
//! images, so synthetic inputs are intrinsic to those stages, not a shortcut.

use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use sceptre::DetectionConfig;
use sceptre::bench::{
    detect_group_for_benchmark, detect_postprocess_for_benchmark, detect_preprocess_for_benchmark,
    detect_preprocess_reference_for_benchmark, load_corpus_image, synthetic_boxes, synthetic_heatmaps,
};

/// Detection canvas cap (EasyOCR default).
const CANVAS_SIZE: u32 = 2560;
/// Detection magnification ratio (EasyOCR default).
const MAG_RATIO: f32 = 1.0;
/// Region text threshold (EasyOCR default).
const TEXT_THRESHOLD: f32 = 0.7;
/// Link threshold (EasyOCR default).
const LINK_THRESHOLD: f32 = 0.4;
/// Low-text threshold (EasyOCR default).
const LOW_TEXT: f32 = 0.4;
/// Synthetic heat-map dimensions for the postprocess bench.
const HEATMAP_HEIGHT: usize = 320;
const HEATMAP_WIDTH: usize = 320;
/// Number of synthetic boxes fed to the grouping bench.
const GROUP_BOX_COUNT: usize = 200;

/// Sample count for the expensive full-image preprocess bench.
const PREPROCESS_SAMPLES: usize = 20;
/// Warm-up duration for the detection benches.
const WARM_UP_SECS: u64 = 2;
/// Measurement duration for the detection benches.
const MEASUREMENT_SECS: u64 = 8;

fn bench_preprocess(criterion: &mut Criterion) {
    let image = load_corpus_image("invoice_image.png");
    criterion.bench_function("detect/preprocess/optimized", |b| {
        b.iter(|| detect_preprocess_for_benchmark(&image, CANVAS_SIZE, MAG_RATIO));
    });
    criterion.bench_function("detect/preprocess/reference", |b| {
        b.iter(|| detect_preprocess_reference_for_benchmark(&image, CANVAS_SIZE, MAG_RATIO));
    });
}

fn bench_postprocess(criterion: &mut Criterion) {
    let (region, link) = synthetic_heatmaps(HEATMAP_HEIGHT, HEATMAP_WIDTH);
    criterion.bench_function("detect/postprocess", |b| {
        b.iter(|| detect_postprocess_for_benchmark(&region, &link, TEXT_THRESHOLD, LINK_THRESHOLD, LOW_TEXT));
    });
}

fn bench_group(criterion: &mut Criterion) {
    let boxes = synthetic_boxes(GROUP_BOX_COUNT);
    let config = DetectionConfig::default();
    criterion.bench_function("detect/group", |b| {
        b.iter(|| detect_group_for_benchmark(&boxes, &config));
    });
}

criterion_group! {
    name = detection;
    config = Criterion::default()
        .sample_size(PREPROCESS_SAMPLES)
        .warm_up_time(Duration::from_secs(WARM_UP_SECS))
        .measurement_time(Duration::from_secs(MEASUREMENT_SECS));
    targets = bench_preprocess, bench_postprocess, bench_group
}
criterion_main!(detection);
