//! Tier-2 golden parity harness.
//!
//! Each committed example image (`tests/data/images/*`) is replayed through the
//! real, default-engine `Reader` and compared against a dual golden fixture in
//! `tests/data/golden/*.json` — an authoritative Python EasyOCR reference and a
//! sceptre snapshot (see `tests/data/golden/README.md`). Models are resolved from
//! the local Hugging Face hub cache by [`helpers::HfCacheModelProvider`]; no
//! download happens here.
//!
//! Real-model tests are gated on `SCEPTRE_REQUIRE_MODELS`: when the models are not
//! resolvable, the tests skip (pass) by default, but panic when the env var is
//! truthy so CI surfaces a misconfigured model cache. Their bodies are additionally
//! `#[cfg(feature = "ort")]`-gated so the default backend-less `cargo test` compiles
//! and passes. The pure helpers carry their own always-on unit tests in
//! `tests/helpers/mod.rs`.

mod helpers;

#[cfg(feature = "ort")]
use std::path::PathBuf;
#[cfg(feature = "ort")]
use std::sync::Arc;

#[cfg(feature = "ort")]
use sceptre::{ReadOptions, Reader};

/// Fuzzy word-F1 floor for reference parity, mirroring the corpus tolerance.
#[cfg(feature = "ort")]
const WORD_F1_THRESHOLD: f32 = 0.8;

/// Per-line box-IoU floor, matching the EasyOCR parity threshold.
#[cfg(feature = "ort")]
const BOX_IOU_THRESHOLD: f32 = 0.5;

/// Whether `SCEPTRE_REQUIRE_MODELS` is set to a truthy value, forcing real-model
/// tests to run (and fail loudly if models are missing) instead of skipping.
fn require_models() -> bool {
    match std::env::var("SCEPTRE_REQUIRE_MODELS") {
        Ok(value) => {
            let value = value.trim().to_ascii_lowercase();
            !matches!(value.as_str(), "" | "0" | "false" | "no")
        }
        Err(_) => false,
    }
}

/// Absolute path to `tests/data/` in this crate.
#[cfg(feature = "ort")]
fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

#[test]
fn should_treat_unset_require_models_as_optional() {
    // With the env var unset in the default test environment, real-model tests must ~keep
    // be allowed to skip rather than being forced to run. ~keep
    if std::env::var("SCEPTRE_REQUIRE_MODELS").is_err() {
        assert!(!require_models());
    }
}

/// Exercises the full detect -> crop -> recognize pipeline through the public
/// `Reader` with models resolved from the Hugging Face cache, and checks the dual
/// golden: exact equality against the sceptre snapshot plus fuzzy word-F1 and
/// per-line box-IoU against the EasyOCR reference. Parity assertions are skipped
/// while a fixture is still the committed placeholder.
#[test]
#[cfg(feature = "ort")]
fn should_match_dual_golden_for_english_png() {
    if !helpers::HfCacheModelProvider::available() {
        assert!(
            !require_models(),
            "SCEPTRE_REQUIRE_MODELS is set but CRAFT + english_g2 could not be resolved from the \
             Hugging Face cache (~/.cache/huggingface/hub); run the model-provisioning step first"
        );
        return;
    }

    let reader = Reader::builder()
        .model_provider(Arc::new(helpers::HfCacheModelProvider::new()))
        .build()
        .expect("building the reader with the Hugging Face cache model provider");

    let image_path = data_dir().join("images/english.png");
    let result = reader
        .readtext(&image_path, &ReadOptions::default())
        .expect("the real engine runs end to end over english.png");

    assert!(
        !result.lines.is_empty(),
        "the real engine should detect and recognize at least one line in english.png"
    );

    let golden_path = data_dir().join("golden/english.json");
    let golden_json =
        std::fs::read_to_string(&golden_path).unwrap_or_else(|err| panic!("reading {}: {err}", golden_path.display()));
    let golden = helpers::DualGolden::parse(&golden_json);

    if golden.placeholder {
        // Fixtures are not yet regenerated; the pipeline ran, which is all we can ~keep
        // assert until real goldens land (see tests/data/golden/README.md). ~keep
        return;
    }

    assert_sceptre_snapshot(&result, &golden);
    assert_easyocr_reference(&result, &golden);
}

/// Exact-equality check of recognized text against the sceptre snapshot golden.
#[cfg(feature = "ort")]
fn assert_sceptre_snapshot(result: &sceptre::OcrResult, golden: &helpers::DualGolden) {
    let actual: Vec<String> = result.lines.iter().map(|line| line.text.clone()).collect();
    let expected: Vec<String> = golden.sceptre.lines.iter().map(|line| line.text.clone()).collect();
    assert_eq!(
        actual, expected,
        "recognized text must match the sceptre snapshot golden exactly"
    );
}

/// Fuzzy word-F1 + per-line box-IoU check against the EasyOCR reference golden.
#[cfg(feature = "ort")]
fn assert_easyocr_reference(result: &sceptre::OcrResult, golden: &helpers::DualGolden) {
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
    assert!(
        f1 >= WORD_F1_THRESHOLD,
        "word-F1 {f1:.3} against the EasyOCR reference is below the {WORD_F1_THRESHOLD} threshold"
    );

    for (index, reference_line) in golden.easyocr.lines.iter().enumerate() {
        let reference_bbox = reference_line.bbox();
        let best = result
            .lines
            .iter()
            .map(|line| helpers::box_iou(quad_bbox(&line.quad), reference_bbox))
            .fold(0.0f32, f32::max);
        assert!(
            best >= BOX_IOU_THRESHOLD,
            "reference line {index} (`{}`) has no detection with box-IoU >= {BOX_IOU_THRESHOLD} (best {best:.3})",
            reference_line.text
        );
    }
}

/// Axis-aligned bounds of a recognized quad, for IoU against a reference box.
#[cfg(feature = "ort")]
fn quad_bbox(quad: &sceptre::Quad) -> sceptre::BBox {
    let xs = quad.points.map(|point| point.x);
    let ys = quad.points.map(|point| point.y);
    sceptre::BBox {
        x_min: xs.iter().copied().fold(f32::INFINITY, f32::min),
        y_min: ys.iter().copied().fold(f32::INFINITY, f32::min),
        x_max: xs.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        y_max: ys.iter().copied().fold(f32::NEG_INFINITY, f32::max),
    }
}
