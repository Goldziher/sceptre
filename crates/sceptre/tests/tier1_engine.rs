//! Black-box tests for the [`sceptre::OcrEngine`] seam, [`sceptre::FallbackEngine`],
//! and `Reader`/`ReaderBuilder` wiring.
//!
//! Every test injects a test-local fake `OcrEngine` via `ReaderBuilder::engine`, so
//! none of these depend on models, ONNX, or network access — they run as part of
//! the default `cargo test` invocation.

use std::sync::Arc;

use sceptre::{
    FallbackEngine, Image, OcrConfig, OcrEngine, OcrError, OcrResult, Point, Quad, ReadOptions, Reader, TextLine,
};

/// Absolute path to a committed fixture under `tests/data/images/`.
fn image_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/images")
        .join(name)
}

/// A minimal, arbitrary-but-valid `Image` for tests that call `Reader::recognize`
/// directly (no decoding involved, so any dimensions work).
fn tiny_image() -> Image {
    Image::from_rgb8(1, 1, vec![0, 0, 0]).expect("1x1 rgb8 buffer has the exact required length")
}

/// Builds a `TextLine` fixture with an arbitrary quad, carrying only `text` as the
/// value under test.
fn text_line(text: &str) -> TextLine {
    TextLine {
        quad: Quad {
            points: [
                Point::new(0.0, 0.0),
                Point::new(10.0, 0.0),
                Point::new(10.0, 5.0),
                Point::new(0.0, 5.0),
            ],
        },
        text: text.to_string(),
        confidence: 0.99,
    }
}

/// A test-local `OcrEngine` whose output is fully controlled by the test.
enum FakeEngine {
    Empty,
    Lines(Vec<TextLine>),
    Error(String),
}

impl FakeEngine {
    fn empty() -> Arc<Self> {
        Arc::new(Self::Empty)
    }

    fn lines(lines: Vec<TextLine>) -> Arc<Self> {
        Arc::new(Self::Lines(lines))
    }

    fn error(message: impl Into<String>) -> Arc<Self> {
        Arc::new(Self::Error(message.into()))
    }
}

impl OcrEngine for FakeEngine {
    fn recognize(&self, _image: &Image, _options: &ReadOptions) -> sceptre::Result<OcrResult> {
        match self {
            Self::Empty => Ok(OcrResult::default()),
            Self::Lines(lines) => Ok(OcrResult { lines: lines.clone() }),
            Self::Error(message) => Err(OcrError::Other(message.clone())),
        }
    }
}

#[test]
fn should_return_injected_engine_result_verbatim_from_recognize() {
    let expected_lines = vec![text_line("hello"), text_line("world")];
    let reader = Reader::builder()
        .engine(FakeEngine::lines(expected_lines.clone()))
        .build()
        .expect("building a reader with an injected engine");

    let result = reader
        .recognize(&tiny_image(), &ReadOptions::default())
        .expect("fake engine never errors");

    assert_eq!(result.lines.len(), 2);
    assert_eq!(result, OcrResult { lines: expected_lines });
}

#[test]
fn should_return_injected_engine_result_verbatim_from_readtext() {
    let expected_lines = vec![text_line("readtext delegates to the engine")];
    let reader = Reader::builder()
        .engine(FakeEngine::lines(expected_lines.clone()))
        .build()
        .expect("building a reader with an injected engine");

    let result = reader
        .readtext(&image_path("english.png"), &ReadOptions::default())
        .expect("fake engine never errors");

    assert_eq!(result, OcrResult { lines: expected_lines });
}

#[test]
fn should_return_error_when_readtext_path_does_not_exist() {
    let reader = Reader::builder()
        .engine(FakeEngine::empty())
        .build()
        .expect("building a reader");

    let result = reader.readtext(&image_path("does-not-exist.png"), &ReadOptions::default());

    assert!(
        matches!(result, Err(OcrError::Io(_))),
        "readtext must surface the decode I/O error for a missing file, got {result:?}"
    );
}

#[test]
fn should_reflect_config_set_via_builder_in_reader_config() {
    let mut config = OcrConfig::default();
    config.detection.text_threshold = 0.55;
    config.recognition.batch_size = 4;

    let reader = Reader::builder()
        .config(config)
        .engine(FakeEngine::empty())
        .build()
        .expect("building a reader with a custom config");

    assert_eq!(reader.config().detection.text_threshold, 0.55);
    assert_eq!(reader.config().recognition.batch_size, 4);
    assert_eq!(
        reader.config().detection.link_threshold,
        0.4,
        "fields left unset must keep their defaults"
    );
}

#[test]
fn should_return_second_engine_result_when_first_is_empty() {
    let expected_lines = vec![text_line("second engine result")];
    let fallback = FallbackEngine::new(vec![FakeEngine::empty(), FakeEngine::lines(expected_lines.clone())])
        .expect("non-empty engine list");

    let result = fallback
        .recognize(&tiny_image(), &ReadOptions::default())
        .expect("second engine succeeds");

    assert_eq!(result.lines.len(), 1);
    assert_eq!(result.lines[0].text, "second engine result");
}

#[test]
fn should_return_empty_result_when_all_engines_are_empty() {
    let fallback = FallbackEngine::new(vec![FakeEngine::empty(), FakeEngine::empty()]).expect("non-empty engine list");

    let result = fallback
        .recognize(&tiny_image(), &ReadOptions::default())
        .expect("an empty result is not an error");

    assert_eq!(result, OcrResult::default());
    assert_eq!(result.lines.len(), 0);
}

#[test]
fn should_return_success_when_first_engine_errors_then_second_succeeds() {
    let expected_lines = vec![text_line("recovered")];
    let fallback = FallbackEngine::new(vec![
        FakeEngine::error("first engine unavailable"),
        FakeEngine::lines(expected_lines),
    ])
    .expect("non-empty engine list");

    let result = fallback
        .recognize(&tiny_image(), &ReadOptions::default())
        .expect("second engine recovers");

    assert_eq!(result.lines.len(), 1);
    assert_eq!(result.lines[0].text, "recovered");
}

#[test]
fn should_return_last_error_when_all_engines_fail() {
    let fallback = FallbackEngine::new(vec![
        FakeEngine::error("first failure"),
        FakeEngine::error("second failure"),
    ])
    .expect("non-empty engine list");

    let error = fallback
        .recognize(&tiny_image(), &ReadOptions::default())
        .expect_err("all engines failing must return the last error");

    assert_eq!(error.to_string(), "second failure");
}

#[test]
fn should_return_error_when_earlier_engine_errors_and_later_engine_is_empty() {
    // An error anywhere in the chain is preferred over a trailing empty success,
    // regardless of position — the caller learns something went wrong.
    let fallback =
        FallbackEngine::new(vec![FakeEngine::error("boom"), FakeEngine::empty()]).expect("non-empty engine list");

    let error = fallback
        .recognize(&tiny_image(), &ReadOptions::default())
        .expect_err("an earlier error must win over a later empty result");

    assert_eq!(error.to_string(), "boom");
}

#[test]
fn should_return_config_error_when_engine_list_is_empty() {
    let error = FallbackEngine::new(Vec::new()).expect_err("an empty engine list must be rejected");

    assert!(
        matches!(error, OcrError::Config { .. }),
        "expected OcrError::Config, got {error:?}"
    );
}

#[test]
fn should_include_pushed_engine_in_fallback_chain() {
    let expected_lines = vec![text_line("pushed engine result")];
    let fallback = FallbackEngine::new(vec![FakeEngine::empty()])
        .expect("non-empty engine list")
        .push(FakeEngine::lines(expected_lines));

    let result = fallback
        .recognize(&tiny_image(), &ReadOptions::default())
        .expect("pushed engine succeeds");

    assert_eq!(result.lines.len(), 1);
    assert_eq!(result.lines[0].text, "pushed engine result");
}
