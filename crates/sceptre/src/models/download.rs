//! Download and cache model artifacts.
//!
//! Resolves a [`ModelEntry`] to a local path under the cache directory (default
//! `~/.cache/sceptre`), downloading from Hugging Face when the `download`
//! feature is enabled and verifying its SHA-256 against the registry pin when one
//! is present (the gen2 artifacts are currently unpinned; see the registry).
//!
//! The registry host is re-pointable: the optional `registry_owner` override
//! (see [`crate::config::ModelConfig`] and ADR 0003) swaps the owner segment of
//! the repo id so the same exports can be served from a mirror.

use std::path::{Path, PathBuf};

use crate::error::{OcrError, Result};
use crate::models::registry::ModelEntry;

#[cfg(feature = "download")]
use crate::models::registry::effective_repo;

/// Ensure a model artifact is present locally, returning its path.
///
/// The returned path is stable and lives under `cache_dir`, namespaced by the
/// effective repo id and file name so distinct models never collide. A cached
/// file is reused when its SHA-256 matches [`ModelEntry::sha256`] (or when that
/// pin is empty); otherwise the artifact is fetched from Hugging Face, verified,
/// and only then written into place.
#[cfg(feature = "download")]
pub fn ensure(entry: &ModelEntry, cache_dir: &Path, registry_owner: Option<&str>) -> Result<PathBuf> {
    let repo = effective_repo(entry, registry_owner)?;
    let path = cache_path(cache_dir, &repo, entry.file);

    if path.is_file() && cached_file_is_valid(&path, entry.sha256)? {
        return Ok(path);
    }

    let bytes = fetch_from_hub(&repo, entry.file)?;
    verify_sha256(&bytes, entry.sha256, entry.name)?;
    write_atomically(&path, &bytes)?;
    Ok(path)
}

/// Ensure a model artifact is present locally, returning its path.
#[cfg(not(feature = "download"))]
pub fn ensure(_entry: &ModelEntry, _cache_dir: &Path, _registry_owner: Option<&str>) -> Result<PathBuf> {
    Err(OcrError::model(
        "model download requires the `download` feature; provide a local model path instead",
    ))
}

/// Default cache directory: `~/.cache/sceptre` (or the platform cache dir).
pub fn default_cache_dir() -> Result<PathBuf> {
    dirs::cache_dir()
        .map(|d| d.join("sceptre"))
        .ok_or_else(|| OcrError::model("could not determine a cache directory"))
}

/// Number of hex characters emitted per byte when formatting a digest.
#[cfg(feature = "download")]
const HEX_CHARS_PER_BYTE: usize = 2;

/// Chunk size used when streaming a cached file through the SHA-256 hasher.
#[cfg(feature = "download")]
const HASH_CHUNK_BYTES: usize = 64 * 1024;

/// Build the stable on-disk path for `file` from `repo`, under `cache_dir`.
///
/// Each `owner/name` segment of the repo id becomes a directory component, so
/// the layout is `<cache_dir>/<owner>/<name>/<file>`.
pub(crate) fn cache_path(cache_dir: &Path, repo: &str, file: &str) -> PathBuf {
    let mut path = cache_dir.to_path_buf();
    for segment in repo.split('/') {
        path.push(segment);
    }
    path.push(file);
    path
}

/// Whether an already-cached file satisfies the (possibly empty) SHA-256 pin.
///
/// An empty pin accepts any cached bytes; a non-empty pin requires a
/// case-insensitive hex match of the file's computed digest.
#[cfg(feature = "download")]
fn cached_file_is_valid(path: &Path, expected_sha256: &str) -> Result<bool> {
    if expected_sha256.is_empty() {
        return Ok(true);
    }
    let actual = sha256_file(path)?;
    Ok(actual.eq_ignore_ascii_case(expected_sha256))
}

/// Fetch `file` from `repo` on Hugging Face into memory (blocking, rustls).
#[cfg(feature = "download")]
fn fetch_from_hub(repo: &str, file: &str) -> Result<Vec<u8>> {
    let (owner, name) = match repo.split_once('/') {
        Some((owner, name)) => (owner, name),
        None => ("", repo),
    };
    let client = hf_hub::HFClientSync::new()
        .map_err(|source| model_error(format!("could not create the Hugging Face client for `{repo}`"), source))?;
    let bytes = client
        .model(owner, name)
        .download_file_to_bytes()
        .filename(file.to_string())
        .send()
        .map_err(|source| model_error(format!("could not download `{file}` from `{repo}`"), source))?;
    Ok(bytes.to_vec())
}

/// Verify `bytes` against a non-empty SHA-256 pin, warning when unpinned.
#[cfg(feature = "download")]
fn verify_sha256(bytes: &[u8], expected_sha256: &str, model_name: &str) -> Result<()> {
    if expected_sha256.is_empty() {
        tracing::warn!(
            model = model_name,
            "model artifact has no pinned sha256; skipping integrity verification"
        );
        return Ok(());
    }
    let actual = sha256_hex(bytes);
    if actual.eq_ignore_ascii_case(expected_sha256) {
        Ok(())
    } else {
        Err(OcrError::model(format!(
            "sha256 mismatch for `{model_name}`: expected {expected_sha256}, got {actual}"
        )))
    }
}

/// Write `bytes` to `path` atomically: write a sibling temp file, then rename.
///
/// The rename keeps a partially written artifact from ever being observed at the
/// final path.
#[cfg(feature = "download")]
fn write_atomically(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::sync::atomic::{AtomicU64, Ordering};

    // Per-process-and-call-unique temp name so concurrent downloads of the same
    // model never share (and clobber) one in-flight temp file. ~keep
    static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let unique = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let temp = path.with_extension(format!("{}.{unique}.part", std::process::id()));
    std::fs::write(&temp, bytes)?;
    std::fs::rename(&temp, path)?;
    Ok(())
}

/// Compute the lowercase hex SHA-256 of an in-memory buffer.
#[cfg(feature = "download")]
fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    to_hex(&hasher.finalize())
}

/// Compute the lowercase hex SHA-256 of a file, streamed in fixed-size chunks.
#[cfg(feature = "download")]
fn sha256_file(path: &Path) -> Result<String> {
    use std::io::Read as _;

    use sha2::{Digest, Sha256};

    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; HASH_CHUNK_BYTES];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(to_hex(&hasher.finalize()))
}

/// Render bytes as a lowercase hex string.
#[cfg(feature = "download")]
fn to_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;

    let mut hex = String::with_capacity(bytes.len() * HEX_CHARS_PER_BYTE);
    for byte in bytes {
        // Infallible: writing to a String never errors. ~keep
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

/// Wrap a foreign error in [`OcrError::Model`] with a descriptive message.
#[cfg(feature = "download")]
fn model_error(message: String, source: impl std::error::Error + Send + Sync + 'static) -> OcrError {
    OcrError::Model {
        message,
        source: Some(Box::new(source)),
    }
}

#[cfg(all(test, feature = "download"))]
mod tests {
    use super::*;

    /// Lowercase hex SHA-256 of the empty input, per the FIPS 180-4 test vector.
    const EMPTY_SHA256: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    /// Lowercase hex SHA-256 of the ASCII bytes `abc`, per FIPS 180-4.
    const ABC_SHA256: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    #[test]
    fn sha256_hex_matches_known_test_vectors() {
        assert_eq!(sha256_hex(b""), EMPTY_SHA256);
        assert_eq!(sha256_hex(b"abc"), ABC_SHA256);
    }

    #[test]
    fn verify_sha256_accepts_a_matching_pin_case_insensitively() {
        assert!(verify_sha256(b"abc", ABC_SHA256, "abc_model").is_ok());
        assert!(verify_sha256(b"abc", &ABC_SHA256.to_uppercase(), "abc_model").is_ok());
    }

    #[test]
    fn verify_sha256_rejects_a_mismatched_pin() {
        let error = verify_sha256(b"abc", EMPTY_SHA256, "abc_model").unwrap_err();
        assert!(matches!(error, OcrError::Model { .. }));
    }

    #[test]
    fn verify_sha256_skips_verification_when_the_pin_is_empty() {
        assert!(verify_sha256(b"any bytes", "", "unpinned_model").is_ok());
    }

    #[test]
    fn cache_path_namespaces_by_owner_repo_and_file() {
        let cache_dir = Path::new("/cache/sceptre");
        let path = cache_path(cache_dir, "itextresearch/itext-EasyOCR-english_g2", "english_g2.onnx");
        assert_eq!(
            path,
            Path::new("/cache/sceptre/itextresearch/itext-EasyOCR-english_g2/english_g2.onnx")
        );
    }

    #[test]
    fn sha256_file_hashes_written_bytes() {
        let dir = std::env::temp_dir().join(format!("sceptre-hash-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("abc.bin");
        std::fs::write(&file, b"abc").unwrap();

        assert_eq!(sha256_file(&file).unwrap(), ABC_SHA256);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn cached_file_is_valid_matches_pin_and_rejects_wrong_pin() {
        let dir = std::env::temp_dir().join(format!("sceptre-cache-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("abc.bin");
        std::fs::write(&file, b"abc").unwrap();

        assert!(cached_file_is_valid(&file, ABC_SHA256).unwrap());
        assert!(!cached_file_is_valid(&file, EMPTY_SHA256).unwrap());
        assert!(cached_file_is_valid(&file, "").unwrap());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    #[ignore = "requires network access to Hugging Face"]
    fn ensure_downloads_and_caches_the_craft_model() {
        use crate::models::registry::craft_entry;

        let cache_dir = std::env::temp_dir().join(format!("sceptre-net-{}", std::process::id()));
        let path = ensure(&craft_entry(), &cache_dir, None).expect("craft download should succeed");
        assert!(path.is_file(), "downloaded artifact must exist at {}", path.display());
        assert!(path.starts_with(&cache_dir), "artifact must live under the cache dir");

        // A second call must hit the cache and return the same stable path. ~keep
        let again = ensure(&craft_entry(), &cache_dir, None).expect("cached lookup should succeed");
        assert_eq!(path, again);

        std::fs::remove_dir_all(&cache_dir).ok();
    }
}
