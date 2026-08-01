//! Tier-2 golden parity skeletons.
//!
//! These replay each committed example image (`tests/data/images/*`) through the
//! real, default-engine `Reader` and compare recognized text against a
//! hand-authored fixture in `tests/data/golden/*.json` (see
//! `tests/data/golden/README.md`). They require downloaded ONNX models and a
//! working detect/recognize pipeline, so they are `#[ignore]`d and excluded from
//! the default `cargo test` run; opt in with `cargo test -- --ignored` once
//! models are available locally.
//!
//! The `iou` and `golden_lines` helpers below are pure logic with no model
//! dependency, so they carry their own always-on unit tests.

use std::path::PathBuf;

use sceptre::{BBox, ReadOptions, Reader};

#[cfg(feature = "ort")]
use sceptre::{Language, ModelProvider};
#[cfg(feature = "ort")]
use std::sync::Arc;

/// A [`ModelProvider`] that serves local ONNX files named by environment variables,
/// so the real engine can be exercised without any network access. `EASYOCR_TEST_
/// CRAFT_ONNX` points at a CRAFT detector and `EASYOCR_TEST_RECOG_ONNX` at a gen2
/// recognizer; the same recognizer serves every language group.
#[cfg(feature = "ort")]
struct LocalModelProvider {
    detector: PathBuf,
    recognizer: PathBuf,
}

#[cfg(feature = "ort")]
impl LocalModelProvider {
    fn from_env() -> Self {
        let detector = std::env::var("EASYOCR_TEST_CRAFT_ONNX").expect("set EASYOCR_TEST_CRAFT_ONNX to a CRAFT model");
        let recognizer =
            std::env::var("EASYOCR_TEST_RECOG_ONNX").expect("set EASYOCR_TEST_RECOG_ONNX to a recognizer model");
        Self {
            detector: PathBuf::from(detector),
            recognizer: PathBuf::from(recognizer),
        }
    }
}

#[cfg(feature = "ort")]
impl ModelProvider for LocalModelProvider {
    fn detector(&self) -> sceptre::Result<PathBuf> {
        Ok(self.detector.clone())
    }

    fn recognizer(&self, _language: Language) -> sceptre::Result<PathBuf> {
        Ok(self.recognizer.clone())
    }
}

/// Absolute path to `tests/data/` in this crate.
fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

/// Intersection-over-union of two axis-aligned boxes, used to score box parity
/// against the golden fixture once real detection output exists.
fn iou(a: BBox, b: BBox) -> f32 {
    let x_min = a.x_min.max(b.x_min);
    let y_min = a.y_min.max(b.y_min);
    let x_max = a.x_max.min(b.x_max);
    let y_max = a.y_max.min(b.y_max);
    let intersection = (x_max - x_min).max(0.0) * (y_max - y_min).max(0.0);

    let area_a = (a.x_max - a.x_min) * (a.y_max - a.y_min);
    let area_b = (b.x_max - b.x_min) * (b.y_max - b.y_min);
    let union = area_a + area_b - intersection;

    if union <= 0.0 { 0.0 } else { intersection / union }
}

/// Extracts the `{"lines": ["...", ...]}` string array from a golden fixture (see
/// `tests/data/golden/README.md`). Parsed with `serde_json` so fixture text may
/// contain commas, quotes, and escapes without splitting incorrectly.
fn golden_lines(json: &str) -> Vec<String> {
    let fixture: serde_json::Value = serde_json::from_str(json).expect("golden fixture must be valid JSON");
    fixture["lines"]
        .as_array()
        .expect("golden fixture must have a `lines` array")
        .iter()
        .map(|line| {
            line.as_str()
                .expect("each golden line must be a JSON string")
                .to_string()
        })
        .collect()
}

#[test]
fn should_score_identical_boxes_as_iou_one() {
    let a = BBox {
        x_min: 0.0,
        y_min: 0.0,
        x_max: 10.0,
        y_max: 10.0,
    };

    assert_eq!(iou(a, a), 1.0);
}

#[test]
fn should_score_disjoint_boxes_as_iou_zero() {
    let a = BBox {
        x_min: 0.0,
        y_min: 0.0,
        x_max: 1.0,
        y_max: 1.0,
    };
    let b = BBox {
        x_min: 5.0,
        y_min: 5.0,
        x_max: 6.0,
        y_max: 6.0,
    };

    assert_eq!(iou(a, b), 0.0);
}

#[test]
fn should_score_half_overlapping_boxes_as_iou_one_third() {
    let a = BBox {
        x_min: 0.0,
        y_min: 0.0,
        x_max: 2.0,
        y_max: 1.0,
    };
    let b = BBox {
        x_min: 1.0,
        y_min: 0.0,
        x_max: 3.0,
        y_max: 1.0,
    };

    // intersection = 1x1 = 1, union = 2 + 2 - 1 = 3. ~keep
    assert_eq!(iou(a, b), 1.0 / 3.0);
}

#[test]
fn should_parse_golden_fixture_lines() {
    let lines = golden_lines(r#"{"lines": ["EASY OCR"]}"#);

    assert_eq!(lines, vec!["EASY OCR".to_string()]);
}

#[test]
fn should_parse_golden_fixture_with_multiple_lines() {
    let lines = golden_lines(r#"{"lines": ["first line", "second line"]}"#);

    assert_eq!(lines, vec!["first line".to_string(), "second line".to_string()]);
}

#[test]
#[cfg(feature = "ort")]
#[ignore = "requires local CRAFT + recognizer ONNX models (EASYOCR_TEST_*_ONNX)"]
fn should_run_the_real_engine_end_to_end_over_english_png() {
    // Exercises the full detect → crop → recognize wiring through the public ~keep
    // `Reader` with a local-file model provider; asserts the pipeline runs without ~keep
    // error and produces at least one line. Text parity is the golden test's job. ~keep
    let provider = Arc::new(LocalModelProvider::from_env());
    let reader = Reader::builder()
        .model_provider(provider)
        .build()
        .expect("building the reader with a local model provider");

    let image_path = data_dir().join("images/english.png");
    let result = reader
        .readtext(&image_path, &ReadOptions::default())
        .expect("the real engine runs end to end over english.png");

    assert!(
        !result.lines.is_empty(),
        "the real engine should detect and recognize at least one line in english.png"
    );
}

#[test]
#[ignore = "requires downloaded models"]
fn should_match_golden_text_for_english_png() {
    let golden_path = data_dir().join("golden/english.json");
    let golden_json =
        std::fs::read_to_string(&golden_path).unwrap_or_else(|err| panic!("reading {}: {err}", golden_path.display()));
    let expected_lines = golden_lines(&golden_json);

    let reader = Reader::builder().build().expect("building the default reader");
    let image_path = data_dir().join("images/english.png");
    let result = reader
        .readtext(&image_path, &ReadOptions::default())
        .expect("running the default engine over english.png");

    let actual_lines: Vec<String> = result.lines.iter().map(|line| line.text.clone()).collect();
    assert_eq!(
        actual_lines, expected_lines,
        "recognized text must match the golden fixture exactly"
    );

    // Once detection produces real quads, extend this with a per-line box IoU ~keep
    // check against expected boxes recorded alongside `expected_lines`, using ~keep
    // the `iou` helper above and the EasyOCR parity threshold (>= 0.5). ~keep
}
