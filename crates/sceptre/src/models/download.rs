//! Download and cache model artifacts through the Hugging Face hub cache.
//!
//! Resolves a [`ModelEntry`] to a local path inside Hugging Face's native on-disk
//! cache — `<root>/models--<owner>--<name>/snapshots/<rev>/<file>` — so the
//! library, the CLI `models download`, the `sceptre-tools snapshot` tool, and the
//! parity harness all share one cache store (see ADR 0017, which extends ADR
//! 0003). The cache root defaults to `HF_HUB_CACHE` → `HUGGINGFACE_HUB_CACHE` →
//! `$HF_HOME/hub` → `~/.cache/huggingface/hub`, overridable via
//! [`crate::config::ModelConfig::cache_dir`]. Downloads run through `hf-hub` when
//! the `download` feature is enabled, verifying the SHA-256 against the registry
//! pin (every CRAFT and gen2 artifact is pinned; see the registry). A cached
//! artifact is trusted (verified when first downloaded) and returned without a
//! network round-trip.
//!
//! The registry host is re-pointable: the optional `registry_owner` override
//! (see [`crate::config::ModelConfig`] and ADR 0003) swaps the owner segment of
//! the repo id so the same exports can be served from a mirror.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::error::{OcrError, Result};
use crate::models::registry::ModelEntry;

#[cfg(feature = "download")]
use crate::models::registry::effective_repo;

/// Ensure a model artifact is present locally, returning its path.
///
/// Offline-first: when the artifact already resolves in the Hugging Face hub cache
/// it is returned directly — no network — skipping hf-hub's per-call revision
/// revalidation (a `304` round-trip to the Hub that otherwise costs a network RTT
/// on every run). Only a cache miss fetches from the Hub, verifying the download
/// against [`ModelEntry::sha256`] (empty pin → skipped). `cache_dir_override`
/// overrides the hub cache root (see [`hf_cache_root`]).
#[cfg(feature = "download")]
pub fn ensure(entry: &ModelEntry, cache_dir_override: Option<&Path>, registry_owner: Option<&str>) -> Result<PathBuf> {
    let repo = effective_repo(entry, registry_owner)?;
    let root = hf_cache_root(cache_dir_override)?;

    if let Some(path) = resolve_cached(&root, &repo, entry.file) {
        return Ok(path);
    }

    let (owner, name) = match repo.split_once('/') {
        Some((owner, name)) => (owner, name),
        None => ("", repo.as_str()),
    };
    let client = hf_hub::HFClient::builder()
        .cache_dir(root)
        .build_sync()
        .map_err(|source| model_error(format!("could not create the Hugging Face client for `{repo}`"), source))?;

    let path = client
        .model(owner, name)
        .download_file()
        .filename(entry.file.to_string())
        .send()
        .map_err(|source| model_error(format!("could not download `{}` from `{repo}`", entry.file), source))?;

    verify_sha256_file(&path, entry.sha256, entry.name)?;
    Ok(path)
}

/// Ensure a model artifact is present locally, returning its path.
#[cfg(not(feature = "download"))]
pub fn ensure(
    _entry: &ModelEntry,
    _cache_dir_override: Option<&Path>,
    _registry_owner: Option<&str>,
) -> Result<PathBuf> {
    Err(OcrError::model(
        "model download requires the `download` feature; provide a local model path instead",
    ))
}

/// Resolve the Hugging Face hub cache root.
///
/// When `override_dir` is `Some`, it is returned verbatim (the config override).
/// Otherwise the root is resolved from the environment in HF's documented order:
/// `HF_HUB_CACHE` → `HUGGINGFACE_HUB_CACHE` → `$HF_HOME/hub` →
/// `~/.cache/huggingface/hub`, where the home directory is `$HOME` on Unix and
/// `%USERPROFILE%` on Windows (mirroring how `huggingface_hub` expands `~`). Empty
/// environment values are ignored. Errors with [`OcrError::Model`] when the home
/// directory cannot be resolved and no earlier source applied. Dependency-free so
/// it works without the `download` feature.
pub(crate) fn hf_cache_root(override_dir: Option<&Path>) -> Result<PathBuf> {
    if let Some(dir) = override_dir {
        return Ok(dir.to_path_buf());
    }
    hf_cache_root_from_env(non_empty_env)
}

/// Resolve the hub cache root from an environment lookup, in HF's documented order.
///
/// Split out from [`hf_cache_root`] so the resolution order — including the
/// `$HOME` → `%USERPROFILE%` home fallback — is unit-testable without mutating the
/// shared process environment.
fn hf_cache_root_from_env(env: impl Fn(&str) -> Option<String>) -> Result<PathBuf> {
    if let Some(dir) = env("HF_HUB_CACHE") {
        return Ok(PathBuf::from(dir));
    }
    if let Some(dir) = env("HUGGINGFACE_HUB_CACHE") {
        return Ok(PathBuf::from(dir));
    }
    if let Some(home) = env("HF_HOME") {
        return Ok(PathBuf::from(home).join("hub"));
    }
    let home = env("HOME").or_else(|| env("USERPROFILE")).ok_or_else(|| {
        OcrError::model("could not determine the Hugging Face cache root: neither $HOME nor %USERPROFILE% is set")
    })?;
    Ok(PathBuf::from(home).join(".cache").join("huggingface").join("hub"))
}

/// Read an environment variable, treating an unset or empty value as absent.
fn non_empty_env(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|value| !value.is_empty())
}

/// The on-disk cache directory name for a `owner/name` repo id.
///
/// Mirrors Hugging Face's layout: `sceptre-ocr/english_g2`
/// becomes `models--sceptre-ocr--english_g2`.
pub(crate) fn repo_cache_dir_name(repo_id: &str) -> String {
    format!("models--{}", repo_id.replace('/', "--"))
}

/// Resolve a cached model file within the hub cache without touching the network.
///
/// Under `<root>/<repo_cache_dir_name>/snapshots/`, returns the file inside the
/// newest snapshot subdirectory (by modification time) that actually contains
/// `file`, or `None` when the repo, the snapshots directory, or the file is
/// absent. This is the offline "is it cached, and where" check.
pub(crate) fn resolve_cached(root: &Path, repo_id: &str, file: &str) -> Option<PathBuf> {
    let snapshots = root.join(repo_cache_dir_name(repo_id)).join("snapshots");
    let mut candidates: Vec<(SystemTime, PathBuf)> = std::fs::read_dir(&snapshots)
        .ok()?
        .filter_map(std::result::Result::ok)
        .filter(|entry| entry.path().is_dir())
        .filter_map(|entry| {
            let candidate = entry.path().join(file);
            if candidate.exists() {
                let modified = entry
                    .metadata()
                    .and_then(|meta| meta.modified())
                    .unwrap_or(SystemTime::UNIX_EPOCH);
                Some((modified, candidate))
            } else {
                None
            }
        })
        .collect();
    candidates.sort_by_key(|(modified, _)| *modified);
    candidates.pop().map(|(_, path)| path)
}

/// Number of hex characters emitted per byte when formatting a digest.
#[cfg(feature = "download")]
const HEX_CHARS_PER_BYTE: usize = 2;

/// Chunk size used when streaming a cached file through the SHA-256 hasher.
#[cfg(feature = "download")]
const HASH_CHUNK_BYTES: usize = 64 * 1024;

/// Verify a file at `path` against a non-empty SHA-256 pin, warning when unpinned.
///
/// Streams the file through the hasher (`sha256_file`) and compares
/// case-insensitively against `expected_sha256`. An empty pin skips verification
/// with a warning, matching the registry's currently-unpinned artifacts.
#[cfg(feature = "download")]
fn verify_sha256_file(path: &Path, expected_sha256: &str, model_name: &str) -> Result<()> {
    if expected_sha256.is_empty() {
        tracing::warn!(
            model = model_name,
            "model artifact has no pinned sha256; skipping integrity verification"
        );
        return Ok(());
    }
    let actual = sha256_file(path)?;
    if actual.eq_ignore_ascii_case(expected_sha256) {
        Ok(())
    } else {
        Err(OcrError::model(format!(
            "sha256 mismatch for `{model_name}`: expected {expected_sha256}, got {actual}"
        )))
    }
}

/// Verify `bytes` against a non-empty SHA-256 pin, warning when unpinned.
#[cfg(all(test, feature = "download"))]
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

/// Compute the lowercase hex SHA-256 of an in-memory buffer.
#[cfg(all(test, feature = "download"))]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_cache_dir_name_mirrors_the_hugging_face_layout() {
        assert_eq!(
            repo_cache_dir_name("sceptre-ocr/english_g2"),
            "models--sceptre-ocr--english_g2"
        );
    }

    #[test]
    fn hf_cache_root_returns_the_override_verbatim() {
        let override_dir = Path::new("/custom/hub/cache");
        assert_eq!(hf_cache_root(Some(override_dir)).unwrap(), override_dir);
    }

    /// Build an env lookup from a fixed `(key, value)` table for the resolver tests,
    /// so resolution order is exercised without touching the shared process env.
    fn env_from(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: std::collections::HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect();
        move |key: &str| map.get(key).cloned()
    }

    #[test]
    fn hf_cache_root_prefers_explicit_hub_cache_over_every_home() {
        let root = hf_cache_root_from_env(env_from(&[
            ("HF_HUB_CACHE", "/explicit/hub"),
            ("HF_HOME", "/hf/home"),
            ("HOME", "/home/user"),
        ]))
        .unwrap();
        assert_eq!(root, PathBuf::from("/explicit/hub"));
    }

    #[test]
    fn hf_cache_root_derives_the_hub_subdir_from_hf_home() {
        let root = hf_cache_root_from_env(env_from(&[("HF_HOME", "/hf/home"), ("HOME", "/home/user")])).unwrap();
        assert_eq!(root, PathBuf::from("/hf/home").join("hub"));
    }

    #[test]
    fn hf_cache_root_falls_back_to_the_home_cache_layout() {
        let root = hf_cache_root_from_env(env_from(&[("HOME", "/home/user")])).unwrap();
        assert_eq!(
            root,
            PathBuf::from("/home/user")
                .join(".cache")
                .join("huggingface")
                .join("hub")
        );
    }

    #[test]
    fn hf_cache_root_uses_userprofile_when_home_is_unset_on_windows() {
        let root = hf_cache_root_from_env(env_from(&[("USERPROFILE", "C:/Users/dev")])).unwrap();
        assert_eq!(
            root,
            PathBuf::from("C:/Users/dev")
                .join(".cache")
                .join("huggingface")
                .join("hub")
        );
    }

    #[test]
    fn hf_cache_root_errors_when_no_home_source_is_available() {
        assert!(hf_cache_root_from_env(env_from(&[])).is_err());
    }

    #[test]
    fn resolve_cached_finds_a_file_planted_in_a_snapshot() {
        let root = std::env::temp_dir().join(format!("sceptre-resolve-{}", std::process::id()));
        let repo = "sceptre-ocr/english_g2";
        let file = "english_g2.onnx";
        let snapshot = root.join(repo_cache_dir_name(repo)).join("snapshots").join("deadbeef");
        std::fs::create_dir_all(&snapshot).unwrap();
        let planted = snapshot.join(file);
        std::fs::write(&planted, b"onnx").unwrap();

        assert_eq!(resolve_cached(&root, repo, file), Some(planted));

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn resolve_cached_returns_none_when_the_file_is_absent() {
        let root = std::env::temp_dir().join(format!("sceptre-resolve-empty-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();

        assert_eq!(resolve_cached(&root, "sceptre-ocr/english_g2", "english_g2.onnx"), None);

        std::fs::remove_dir_all(&root).ok();
    }
}

#[cfg(all(test, feature = "download"))]
mod download_tests {
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
    fn sha256_file_hashes_written_bytes() {
        let dir = std::env::temp_dir().join(format!("sceptre-hash-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("abc.bin");
        std::fs::write(&file, b"abc").unwrap();

        assert_eq!(sha256_file(&file).unwrap(), ABC_SHA256);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn verify_sha256_file_matches_pin_and_rejects_wrong_pin() {
        let dir = std::env::temp_dir().join(format!("sceptre-verify-file-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("abc.bin");
        std::fs::write(&file, b"abc").unwrap();

        assert!(verify_sha256_file(&file, ABC_SHA256, "abc_model").is_ok());
        assert!(verify_sha256_file(&file, EMPTY_SHA256, "abc_model").is_err());
        // An empty pin skips verification and accepts any bytes. ~keep
        assert!(verify_sha256_file(&file, "", "unpinned_model").is_ok());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn ensure_returns_a_cached_artifact_without_network() {
        use crate::models::registry::craft_entry;

        let entry = craft_entry();
        let root = std::env::temp_dir().join(format!("sceptre-ensure-cache-{}", std::process::id()));
        let snapshot = root
            .join(repo_cache_dir_name(entry.hf_repo))
            .join("snapshots")
            .join("rev0");
        std::fs::create_dir_all(&snapshot).unwrap();
        let planted = snapshot.join(entry.file);
        std::fs::write(&planted, b"onnx-bytes").unwrap();

        // Resolves from the planted cache with no Hub client built and no network. ~keep
        let path = ensure(&entry, Some(root.as_path()), None).expect("cached artifact resolves offline");
        assert_eq!(path, planted);

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    #[ignore = "requires network access to Hugging Face"]
    fn ensure_downloads_and_caches_the_craft_model() {
        use crate::models::registry::craft_entry;

        let cache_dir = std::env::temp_dir().join(format!("sceptre-net-{}", std::process::id()));
        let path = ensure(&craft_entry(), Some(cache_dir.as_path()), None).expect("craft download should succeed");
        assert!(path.is_file(), "downloaded artifact must exist at {}", path.display());
        assert!(path.starts_with(&cache_dir), "artifact must live under the cache dir");

        // A second call must hit the cache and return the same stable path. ~keep
        let again = ensure(&craft_entry(), Some(cache_dir.as_path()), None).expect("cached lookup should succeed");
        assert_eq!(path, again);

        std::fs::remove_dir_all(&cache_dir).ok();
    }
}
