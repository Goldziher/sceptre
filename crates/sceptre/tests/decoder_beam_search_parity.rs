// This harness's per-image metrics line is its result output, not a diagnostic: it is
// what the accuracy-vs-cost comparison against greedy decoding reads. Integration tests
// are separate crates and do not inherit the library's crate-root allow. ~keep
#![allow(clippy::print_stdout)]

//! Ad hoc greedy-vs-beam-search accuracy/cost comparison over the tier-2 corpus.
//!
//! This is a throwaway measurement harness, not a gate: it exists to answer "does
//! `Decoder::BeamSearch` move word-F1 on the real corpus, and at what decode-time
//! cost", using the same corpus and golden fixtures as `tier2_golden.rs` but without
//! touching that file (its thresholds gate the greedy default and must not move to
//! accommodate an opt-in decoder). Gated on `SCEPTRE_REQUIRE_MODELS` exactly like the
//! golden harness; skips (passing) when models are not cached.

mod helpers;

#[cfg(feature = "ort")]
use std::path::{Path, PathBuf};
#[cfg(feature = "ort")]
use std::time::Instant;

#[cfg(feature = "ort")]
use sceptre::{Decoder, Language, OcrConfig, ReadOptions, Reader, RecognitionConfig};

#[cfg(feature = "ort")]
fn require_models() -> bool {
    match std::env::var("SCEPTRE_REQUIRE_MODELS") {
        Ok(value) => {
            let value = value.trim().to_ascii_lowercase();
            !matches!(value.as_str(), "" | "0" | "false" | "no")
        }
        Err(_) => false,
    }
}

#[cfg(feature = "ort")]
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

#[cfg(feature = "ort")]
fn images_dir() -> PathBuf {
    std::env::var_os("TEST_DOCUMENTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root().join("test_documents"))
        .join("images")
}

#[cfg(feature = "ort")]
fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/golden")
}

#[cfg(feature = "ort")]
fn config_for(language: Language, decoder: Decoder) -> OcrConfig {
    config_for_width(language, decoder, RecognitionConfig::default().beam_width)
}

#[cfg(feature = "ort")]
fn config_for_width(language: Language, decoder: Decoder, beam_width: usize) -> OcrConfig {
    let mut config = OcrConfig::default();
    config.model.languages = vec![language];
    config.recognition = RecognitionConfig {
        decoder,
        beam_width,
        ..Default::default()
    };
    config
}

#[cfg(feature = "ort")]
fn models_available(config: &OcrConfig) -> bool {
    sceptre::model_manifest(config)
        .map(|manifest| manifest.iter().all(|info| info.cached))
        .unwrap_or(false)
}

/// Run one image under one decoder, returning (word_f1, elapsed).
#[cfg(feature = "ort")]
fn run_one(image_file: &str, golden_stem: &str, language: Language, decoder: Decoder) -> (f32, std::time::Duration) {
    run_one_width(
        image_file,
        golden_stem,
        language,
        decoder,
        RecognitionConfig::default().beam_width,
    )
}

#[cfg(feature = "ort")]
fn run_one_width(
    image_file: &str,
    golden_stem: &str,
    language: Language,
    decoder: Decoder,
    beam_width: usize,
) -> (f32, std::time::Duration) {
    let config = config_for_width(language, decoder, beam_width);
    let reader = Reader::builder().config(config).build().expect("build reader");
    let image_path = images_dir().join(image_file);

    let start = Instant::now();
    let result = reader
        .readtext(&image_path, &ReadOptions::default())
        .unwrap_or_else(|err| panic!("real engine run over {image_file} ({decoder:?}): {err}"));
    let elapsed = start.elapsed();

    let golden_path = golden_dir().join(format!("{golden_stem}.json"));
    let golden_json =
        std::fs::read_to_string(&golden_path).unwrap_or_else(|err| panic!("reading {golden_path:?}: {err}"));
    let golden = helpers::DualGolden::parse(&golden_json);

    let actual_text = result
        .lines
        .iter()
        .map(|line| line.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let reference_text = golden
        .easyocr
        .lines
        .iter()
        .map(|line| line.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let f1 = helpers::word_f1(&actual_text, &reference_text);
    if std::env::var_os("SCEPTRE_DEBUG_DECODER_DIFF").is_some() {
        println!("  [{decoder:?}] {image_file} actual:    {actual_text}");
        println!("  [{decoder:?}] {image_file} reference: {reference_text}");
    }
    (f1, elapsed)
}

#[cfg(feature = "ort")]
fn compare(image_file: &str, golden_stem: &str, language: Language) {
    if !require_models() {
        return;
    }
    if !models_available(&config_for(language, Decoder::Greedy)) {
        return;
    }

    let (greedy_f1, greedy_time) = run_one(image_file, golden_stem, language, Decoder::Greedy);
    let (beam_f1, beam_time) = run_one(image_file, golden_stem, language, Decoder::BeamSearch);

    println!(
        "decoder-comparison\t{image_file}\tgreedy_f1={greedy_f1:.3}\tbeam_f1={beam_f1:.3}\tdelta={:+.3}\t\
         greedy_ms={}\tbeam_ms={}\tslowdown={:.2}x",
        beam_f1 - greedy_f1,
        greedy_time.as_millis(),
        beam_time.as_millis(),
        beam_time.as_secs_f64() / greedy_time.as_secs_f64().max(1e-9)
    );
}

#[test]
#[cfg(feature = "ort")]
fn compare_english() {
    compare("english.png", "english", Language::English);
}

#[test]
#[cfg(feature = "ort")]
fn compare_chinese() {
    compare("chinese.jpg", "chinese", Language::ChineseSimplified);
}

#[test]
#[cfg(feature = "ort")]
fn compare_japanese() {
    compare("japanese.jpg", "japanese", Language::Japanese);
}

#[test]
#[cfg(feature = "ort")]
fn compare_korean() {
    compare("korean.png", "korean", Language::Korean);
}

#[test]
#[cfg(feature = "ort")]
fn compare_cyrillic() {
    compare("cyrillic.png", "cyrillic", Language::Cyrillic);
}

#[test]
#[cfg(feature = "ort")]
fn compare_telugu() {
    compare("telugu.png", "telugu", Language::Telugu);
}

#[test]
#[cfg(feature = "ort")]
fn compare_kannada() {
    compare("kannada.png", "kannada", Language::Kannada);
}

#[test]
#[cfg(feature = "ort")]
fn compare_french() {
    compare("french.jpg", "french", Language::Latin);
}

#[test]
#[cfg(feature = "ort")]
fn compare_english_at_wide_beam_widths() {
    if !require_models() {
        return;
    }
    if !models_available(&config_for(Language::English, Decoder::Greedy)) {
        return;
    }
    for beam_width in [5usize, 15, 50] {
        let (f1, elapsed) = run_one_width(
            "english.png",
            "english",
            Language::English,
            Decoder::BeamSearch,
            beam_width,
        );
        println!(
            "decoder-comparison\tenglish.png\tbeam_width={beam_width}\tbeam_f1={f1:.3}\tbeam_ms={}",
            elapsed.as_millis()
        );
    }
}
