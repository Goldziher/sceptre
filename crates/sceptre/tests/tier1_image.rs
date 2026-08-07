//! Black-box tests for the public [`sceptre::Image`] DTO.
//!
//! These exercise only the frozen public surface (`Image::from_rgb8`,
//! `Image::from_path`, `Image::from_bytes`) with no model dependency, so they
//! run as part of the default `cargo test` invocation. The two tests that decode a real
//! fixture resolve it from the `test_documents` corpus and skip (rather than fail) when
//! that corpus has not been fetched; see [`image_path`].

use std::path::{Path, PathBuf};

use sceptre::{Image, OcrError};

/// The repository root, two levels up from this crate's manifest directory
/// (`<root>/crates/sceptre`).
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Absolute path to a corpus fixture under the `test_documents` submodule's `images/`
/// directory: `TEST_DOCUMENTS_DIR` when set, otherwise the submodule checked out at the
/// repository root.
fn image_path(name: &str) -> PathBuf {
    std::env::var_os("TEST_DOCUMENTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root().join("test_documents"))
        .join("images")
        .join(name)
}

/// A tiny, hand-computed 2x2 RGB8 buffer: red, green, blue, white pixels, row-major.
fn checkerboard_2x2_rgb8() -> Vec<u8> {
    vec![
        255, 0, 0, // (0,0) red ~keep
        0, 255, 0, // (1,0) green ~keep
        0, 0, 255, // (0,1) blue ~keep
        255, 255, 255, // (1,1) white ~keep
    ]
}

#[test]
fn should_round_trip_rgb8_pixels_when_length_matches_dimensions() {
    let pixels = checkerboard_2x2_rgb8();
    let image = Image::from_rgb8(2, 2, pixels.clone()).expect("2x2 buffer has the exact required length");

    assert_eq!(image.width(), 2);
    assert_eq!(image.height(), 2);
    assert_eq!(image.as_rgb8(), pixels.as_slice());
}

#[test]
fn should_return_error_when_rgb8_length_does_not_match_dimensions() {
    // A 2x2 RGB8 buffer needs 12 bytes; 10 is short by one pixel. ~keep
    let too_short = vec![0u8; 10];

    let result = Image::from_rgb8(2, 2, too_short);

    assert!(
        matches!(result, Err(OcrError::Image { .. })),
        "expected OcrError::Image for a pixel buffer whose length != width * height * 3, got {result:?}"
    );
}

#[test]
fn should_decode_real_png_from_path_with_positive_dimensions_and_matching_buffer_length() {
    let path = image_path("english.png");
    if !path.exists() {
        // The test_documents submodule is present but the corpus binaries have not been
        // fetched (see test_documents/scripts/fetch_corpus.py); skip rather than fail. ~keep
        return;
    }
    let image = Image::from_path(&path).expect("decoding the english.png corpus fixture");

    assert!(
        image.width() > 0,
        "decoded width must be positive, got {}",
        image.width()
    );
    assert!(
        image.height() > 0,
        "decoded height must be positive, got {}",
        image.height()
    );
    assert_eq!(
        image.as_rgb8().len(),
        (image.width() as usize) * (image.height() as usize) * 3,
        "rgb8 buffer length must equal width * height * 3"
    );
}

#[test]
fn should_decode_identical_image_from_bytes_and_from_path() {
    let path = image_path("english.png");
    if !path.exists() {
        // See the skip note in `should_decode_real_png_from_path_with_positive_dimensions_and_matching_buffer_length`. ~keep
        return;
    }
    let bytes = std::fs::read(&path).expect("reading english.png bytes from disk");

    let from_path = Image::from_path(&path).expect("decoding english.png via from_path");
    let from_bytes = Image::from_bytes(&bytes).expect("decoding english.png via from_bytes");

    assert_eq!(
        from_path, from_bytes,
        "from_path and from_bytes must decode the same file identically"
    );
}

#[test]
fn should_return_error_when_path_does_not_exist() {
    let result = Image::from_path(image_path("does-not-exist.png"));

    assert!(
        matches!(result, Err(OcrError::Io(_))),
        "expected OcrError::Io for a nonexistent path, got {result:?}"
    );
}

#[test]
fn should_return_error_when_bytes_are_not_a_valid_image() {
    let result = Image::from_bytes(b"not an image");

    assert!(
        matches!(result, Err(OcrError::Image { .. })),
        "expected OcrError::Image for bytes that do not decode as any supported image format, got {result:?}"
    );
}
