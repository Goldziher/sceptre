//! Cross-backend agreement: every non-default backend must produce the same recognized
//! text as native `ort` on the same images.
//!
//! `tract` cannot shape-infer the CRAFT detector under dynamic H/W, so on that backend
//! detection runs on a fixed square canvas while the recognizers stay dynamic (ADR 0027).
//! `candle` does not execute the graph at all — it runs hand-written networks over the
//! same weights — so for it this is the end-to-end proof that the transcription is right
//! where the tensor-level comparison in `inference::candle` proves the arithmetic is.
//!
//! Opt-in and heavy: it loads real models, and both alternative backends are slower than
//! `ort`. Each pairing compiles only when both of its backends are available.
#![cfg(feature = "ort")]

use std::path::{Path, PathBuf};

use sceptre::{Accelerator, Backend, Language, OcrConfig, ReadOptions, Reader};

/// Truthy `SCEPTRE_REQUIRE_MODELS` opts this heavy, model-backed test in.
fn require_models() -> bool {
    match std::env::var("SCEPTRE_REQUIRE_MODELS") {
        Ok(value) => !matches!(value.trim().to_ascii_lowercase().as_str(), "" | "0" | "false" | "no"),
        Err(_) => false,
    }
}

/// The repository root, two levels up from this crate's manifest directory
/// (`<root>/crates/sceptre`).
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Directory holding the corpus images: `TEST_DOCUMENTS_DIR` when set, otherwise the
/// `test_documents` submodule checked out at the repository root.
fn images_dir() -> PathBuf {
    std::env::var_os("TEST_DOCUMENTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root().join("test_documents"))
        .join("images")
}

/// A single-language config on `backend`, with a modest detection canvas so the fixed
/// tract canvas stays small enough to optimize and run quickly.
fn config_for(language: Language, backend: Backend, accelerator: Accelerator) -> OcrConfig {
    let mut config = OcrConfig::default();
    config.model.languages = vec![language];
    config.model.backend = backend;
    config.model.accelerator = accelerator;
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
fn recognize_text(image_file: &str, language: Language, backend: Backend, accelerator: Accelerator) -> Vec<String> {
    let config = config_for(language, backend, accelerator);
    assert!(
        models_available(&config),
        "SCEPTRE_REQUIRE_MODELS is set but the models for {image_file} are not cached"
    );
    let reader = Reader::builder().config(config).build().expect("build the reader");
    let image_path = images_dir().join(image_file);
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

/// `other` must recognize the same words as ort on each image.
#[cfg_attr(
    not(any(feature = "tract", feature = "candle")),
    expect(dead_code, reason = "no alternative backend is compiled in")
)]
fn assert_backends_agree(image_file: &str, language: Language, other: Backend) {
    assert_backends_agree_on(image_file, language, other, Accelerator::Cpu);
}

/// `other`, running on `accelerator`, must recognize the same words as ort on the CPU.
#[cfg_attr(
    not(any(feature = "tract", feature = "candle")),
    expect(dead_code, reason = "no alternative backend is compiled in")
)]
fn assert_backends_agree_on(image_file: &str, language: Language, other: Backend, accelerator: Accelerator) {
    if !require_models() {
        return;
    }
    let ort = recognize_text(image_file, language, Backend::Ort, Accelerator::Cpu);
    let actual = recognize_text(image_file, language, other, accelerator);
    assert_eq!(
        word_multiset(&ort),
        word_multiset(&actual),
        "ort and {other:?} on {} disagree on {image_file}: ort={ort:?} {other:?}={actual:?}",
        accelerator.as_str()
    );
}

#[cfg(feature = "tract")]
#[test]
fn ort_and_tract_agree_on_english() {
    assert_backends_agree("english.png", Language::English, Backend::Tract);
}

#[cfg(feature = "tract")]
#[test]
fn ort_and_tract_agree_on_cyrillic() {
    assert_backends_agree("cyrillic.png", Language::Cyrillic, Backend::Tract);
}

#[cfg(feature = "candle")]
#[test]
fn ort_and_candle_agree_on_english() {
    assert_backends_agree("english.png", Language::English, Backend::Candle);
}

#[cfg(feature = "candle")]
#[test]
fn ort_and_candle_agree_on_cyrillic() {
    assert_backends_agree("cyrillic.png", Language::Cyrillic, Backend::Candle);
}

/// The GPU path must clear the same bar as the CPU one, on a machine that has a GPU.
///
/// CI has none — no hosted runner offers Metal — so this compiles everywhere and runs
/// only on real hardware. A green CI says nothing about Metal.
#[cfg(feature = "candle-metal")]
#[test]
fn ort_and_candle_on_metal_agree_on_english() {
    assert_backends_agree_on("english.png", Language::English, Backend::Candle, Accelerator::Metal);
}
