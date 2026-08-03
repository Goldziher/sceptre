//! Cross-backend agreement: the pure-Rust `tract` pipeline must produce the same
//! recognized text as native `ort` on the same images (ADR 0027).
//!
//! `tract` cannot shape-infer the CRAFT detector under dynamic H/W, so on that backend
//! detection runs on a fixed square canvas while the recognizers stay dynamic. This test
//! is the end-to-end proof that the fixed-canvas detection path plus the tract recognizers
//! reproduce the `ort` output. It is opt-in (heavy: it loads real models and tract's CRAFT
//! optimization is slow) and only compiled when both backends are available.
#![cfg(all(feature = "ort", feature = "tract"))]

use std::path::PathBuf;

use sceptre::{Backend, Language, OcrConfig, ReadOptions, Reader};

/// Truthy `SCEPTRE_REQUIRE_MODELS` opts this heavy, model-backed test in.
fn require_models() -> bool {
    match std::env::var("SCEPTRE_REQUIRE_MODELS") {
        Ok(value) => !matches!(value.trim().to_ascii_lowercase().as_str(), "" | "0" | "false" | "no"),
        Err(_) => false,
    }
}

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

/// A single-language config on `backend`, with a modest detection canvas so the fixed
/// tract canvas stays small enough to optimize and run quickly.
fn config_for(language: Language, backend: Backend) -> OcrConfig {
    let mut config = OcrConfig::default();
    config.model.languages = vec![language];
    config.model.backend = backend;
    // Both backends share this cap so they see the same resized image; tract additionally
    // pads to a fixed `canvas x canvas` square. 1024 keeps the small example images at full
    // resolution while bounding tract's CRAFT optimization cost. ~keep
    config.detection.canvas_size = 1024;
    config
}

fn models_available(config: &OcrConfig) -> bool {
    sceptre::model_manifest(config)
        .map(|manifest| manifest.iter().all(|info| info.cached))
        .unwrap_or(false)
}

/// Recognized text lines for `image_file` under `backend`, top-to-bottom.
fn recognize_text(image_file: &str, language: Language, backend: Backend) -> Vec<String> {
    let config = config_for(language, backend);
    assert!(
        models_available(&config),
        "SCEPTRE_REQUIRE_MODELS is set but the models for {image_file} are not cached"
    );
    let reader = Reader::builder().config(config).build().expect("build the reader");
    let image_path = data_dir().join("images").join(image_file);
    let result = reader
        .readtext(&image_path, &ReadOptions::default())
        .unwrap_or_else(|err| panic!("{backend:?} runs end to end over {image_file}: {err}"));
    result.lines.iter().map(|line| line.text.clone()).collect()
}

/// The sorted multiset of whitespace-separated words across all recognized lines.
///
/// Compared instead of the raw line vectors because the fixed tract detection canvas
/// pads differently than ort's dynamic canvas, which can shift CRAFT's heat-map enough
/// to split or merge a line differently (e.g. a trailing word landing on its own line).
/// That is a detection-grouping difference, not a recognition difference: the words —
/// the actual OCR output — must be identical.
fn word_multiset(lines: &[String]) -> Vec<String> {
    let mut words: Vec<String> = lines
        .iter()
        .flat_map(|line| line.split_whitespace().map(str::to_string))
        .collect();
    words.sort();
    words
}

/// The tract pipeline must recognize the same words as ort on each image.
fn assert_backends_agree(image_file: &str, language: Language) {
    if !require_models() {
        return;
    }
    let ort = recognize_text(image_file, language, Backend::Ort);
    let tract = recognize_text(image_file, language, Backend::Tract);
    assert_eq!(
        word_multiset(&ort),
        word_multiset(&tract),
        "ort and tract disagree on {image_file}: ort={ort:?} tract={tract:?}"
    );
}

#[test]
fn ort_and_tract_agree_on_english() {
    assert_backends_agree("english.png", Language::English);
}

#[test]
fn ort_and_tract_agree_on_cyrillic() {
    assert_backends_agree("cyrillic.png", Language::Cyrillic);
}
