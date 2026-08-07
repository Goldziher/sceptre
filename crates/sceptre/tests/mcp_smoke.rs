#![cfg(feature = "mcp")]
//! In-process smoke tests for the MCP `readtext` tool.
//!
//! These exercise the frozen public surface of [`sceptre::mcp::tools::SceptreServer`]
//! against a test-local fake [`sceptre::OcrEngine`], so they need no models, ONNX, or
//! network access. The happy path still decodes a real image from disk (the `test_documents`
//! corpus, see [`image_path`]), since the tool runs `Image::from_path` before invoking the
//! engine; those tests skip when the corpus has not been fetched.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use rmcp::handler::server::wrapper::Parameters;
use sceptre::mcp::tools::SceptreServer;
use sceptre::mcp::types::ReadTextRequest;
use sceptre::{Image, OcrResult, Point, Quad, ReadOptions, Reader, TextLine};

/// The repository root, two levels up from this crate's manifest directory
/// (`<root>/crates/sceptre`).
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Absolute path to `english.png` in the `test_documents` corpus: `TEST_DOCUMENTS_DIR`
/// when set, otherwise the submodule checked out at the repository root.
fn image_path() -> PathBuf {
    std::env::var_os("TEST_DOCUMENTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root().join("test_documents"))
        .join("images")
        .join("english.png")
}

/// A test-local `OcrEngine` that always returns a single canned line, so the tool's
/// output is fully controlled by the test rather than by any real recognizer.
struct FakeEngine;

impl sceptre::OcrEngine for FakeEngine {
    fn recognize(&self, _image: &Image, _options: &ReadOptions) -> sceptre::Result<OcrResult> {
        let line = TextLine {
            quad: Quad {
                points: [
                    Point::new(0.0, 0.0),
                    Point::new(10.0, 0.0),
                    Point::new(10.0, 5.0),
                    Point::new(0.0, 5.0),
                ],
            },
            text: "hello".to_string(),
            confidence: 0.9,
        };
        Ok(OcrResult { lines: vec![line] })
    }
}

/// Builds a `Reader` backed by [`FakeEngine`] — no models or network required.
fn fake_reader() -> Reader {
    Reader::builder()
        .engine(Arc::new(FakeEngine))
        .build()
        .expect("building a reader with an injected engine")
}

#[tokio::test]
async fn readtext_returns_recognized_text_for_a_decodable_image() {
    let path = image_path();
    if !path.exists() {
        // The test_documents submodule is present but the corpus binaries have not been
        // fetched (see test_documents/scripts/fetch_corpus.py); skip rather than fail. ~keep
        return;
    }
    let server = SceptreServer::new(fake_reader());

    let result = server
        .readtext(Parameters(ReadTextRequest {
            image_path: path.to_string_lossy().into_owned(),
            detail: None,
        }))
        .await
        .expect("readtext must not fail at the protocol level for a decodable image");

    assert!(
        !result.is_error.unwrap_or(false),
        "a decodable image must not be flagged as a tool error, got {result:?}"
    );
    let serialized = serde_json::to_string(&result).expect("CallToolResult serializes to JSON");
    assert!(
        serialized.contains("hello"),
        "the structured result must carry the recognized text, got {serialized}"
    );
    assert!(
        serialized.contains("confidence"),
        "the default (detail) result must include per-line confidence, got {serialized}"
    );
    assert!(
        serialized.contains("quad"),
        "the default (detail) result must include the bounding box, got {serialized}"
    );
}

#[tokio::test]
async fn readtext_reports_a_tool_error_for_a_missing_image() {
    let server = SceptreServer::new(fake_reader());

    let result = server
        .readtext(Parameters(ReadTextRequest {
            image_path: "/no/such/file.png".to_string(),
            detail: None,
        }))
        .await
        .expect("a missing file must surface as a tool error, not a protocol Err");

    assert_eq!(
        result.is_error,
        Some(true),
        "a missing image must be flagged as a tool error, got {result:?}"
    );
}

#[tokio::test]
async fn readtext_without_detail_returns_only_line_text() {
    let path = image_path();
    if !path.exists() {
        // See the skip note in `readtext_returns_recognized_text_for_a_decodable_image`. ~keep
        return;
    }
    let server = SceptreServer::new(fake_reader());

    let result = server
        .readtext(Parameters(ReadTextRequest {
            image_path: path.to_string_lossy().into_owned(),
            detail: Some(false),
        }))
        .await
        .expect("readtext must succeed for a decodable image");

    assert!(
        !result.is_error.unwrap_or(false),
        "a decodable image must not be flagged as a tool error, got {result:?}"
    );
    let serialized = serde_json::to_string(&result).expect("CallToolResult serializes to JSON");
    assert!(
        serialized.contains("hello"),
        "detail=false must still carry the text, got {serialized}"
    );
    assert!(
        !serialized.contains("confidence"),
        "detail=false must omit per-line confidence, got {serialized}"
    );
    assert!(
        !serialized.contains("quad"),
        "detail=false must omit the bounding box, got {serialized}"
    );
}

/// The orientation pre-pass (`DetectionConfig::detect_orientation`) is a
/// `Reader`-construction-time setting, exactly like `--lang`/`--backend`: it has no
/// per-call `readtext` parameter, so `sceptre mcp` picks it up only through the
/// `OcrConfig` a `Reader` is built with. This proves that building a `Reader` with
/// the flag enabled neither breaks the tool call nor gets silently dropped before
/// reaching the server.
#[tokio::test]
async fn readtext_serves_a_reader_built_with_orientation_detection_enabled() {
    let path = image_path();
    if !path.exists() {
        // See the skip note in `readtext_returns_recognized_text_for_a_decodable_image`. ~keep
        return;
    }
    let mut config = sceptre::OcrConfig::default();
    config.detection.detect_orientation = true;
    config.detection.orientation_probe_canvas_size = 640;
    config.detection.orientation_margin = 0.1;
    let reader = Reader::builder()
        .config(config)
        .engine(Arc::new(FakeEngine))
        .build()
        .expect("building a reader with orientation detection enabled");
    assert!(reader.config().detection.detect_orientation);

    let server = SceptreServer::new(reader);
    let result = server
        .readtext(Parameters(ReadTextRequest {
            image_path: path.to_string_lossy().into_owned(),
            detail: None,
        }))
        .await
        .expect("readtext must not fail at the protocol level for a decodable image");

    assert!(
        !result.is_error.unwrap_or(false),
        "an orientation-enabled reader must serve the tool call normally, got {result:?}"
    );
}
