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
//! network round-trip, but only once it passes a cheap usability check that keeps
//! a concurrently-downloading or interrupted cache entry from being handed back;
//! an artifact that fails its pin is evicted so the next run re-downloads it.
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

    verify_or_evict(&path, entry.sha256, entry.name)?;
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
/// Mirrors Hugging Face's layout: `xberg-io/sceptre-english_g2`
/// becomes `models--xberg-io--sceptre-english_g2`.
pub(crate) fn repo_cache_dir_name(repo_id: &str) -> String {
    format!("models--{}", repo_id.replace('/', "--"))
}

/// Resolve a cached model file within the hub cache without touching the network.
///
/// Under `<root>/<repo_cache_dir_name>/snapshots/`, returns the newest snapshot
/// subdirectory (by modification time) holding a *usable* `file`, or `None` when
/// the repo, the snapshots directory, or every candidate is absent or unusable.
/// This is the offline "is it cached, and where" check.
///
/// A `None` here is not an error: the caller falls through to the hf-hub download
/// path, which takes the hub's advisory per-file lock. Rejecting an unusable
/// candidate is therefore how a concurrent or interrupted download is made safe —
/// see [`is_usable_artifact`] for what "usable" means.
pub(crate) fn resolve_cached(root: &Path, repo_id: &str, file: &str) -> Option<PathBuf> {
    let snapshots = root.join(repo_cache_dir_name(repo_id)).join("snapshots");
    std::fs::read_dir(&snapshots)
        .ok()?
        .filter_map(std::result::Result::ok)
        .filter_map(|entry| {
            let candidate = entry.path().join(file);
            if !is_usable_artifact(&candidate) {
                return None;
            }
            let modified = entry
                .metadata()
                .and_then(|meta| meta.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH);
            Some((modified, candidate))
        })
        .max_by_key(|(modified, _)| *modified)
        .map(|(_, path)| path)
}

/// Whether a snapshot path is a model artifact safe to hand back without a download.
///
/// Uses [`std::fs::metadata`], which follows symlinks — hf-hub exposes snapshot
/// entries as symlinks into the content-addressed `blobs/` store, so a download
/// still in flight can be observed as a dangling link or a zero-length placeholder.
/// A path is usable only when it stats successfully (excluding dangling links and
/// unreadable entries), is a regular file, and is non-empty. Anything else is
/// treated as a cache miss so the caller re-enters hf-hub's locked download path.
fn is_usable_artifact(path: &Path) -> bool {
    std::fs::metadata(path).is_ok_and(|meta| meta.is_file() && meta.len() > 0)
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

/// Verify an artifact against its pin, deleting it from the cache on mismatch.
///
/// Without eviction a corrupt artifact is permanently sticky: [`resolve_cached`]
/// would keep returning it on every subsequent run and the error would never clear
/// without manual cache surgery. Removing it turns the mismatch into a cache miss,
/// so the next run re-downloads through hf-hub's locked path and self-heals.
///
/// Both the snapshot entry and — when the entry is a symlink — its `blobs/` target
/// are removed. Removing only the symlink would leave hf-hub free to re-link the
/// same corrupt content-addressed blob; removing only the blob would leave a
/// dangling link behind. Deletion is scoped to exactly those two paths: the link
/// target is followed only when it resolves to a regular file directly inside a
/// directory named `blobs`, so nothing outside the hub cache layout is touched.
/// A failed deletion is logged and the original mismatch error is still returned.
#[cfg(feature = "download")]
fn verify_or_evict(path: &Path, expected_sha256: &str, model_name: &str) -> Result<()> {
    let error = match verify_sha256_file(path, expected_sha256, model_name) {
        Ok(()) => return Ok(()),
        Err(error) => error,
    };
    evict_artifact(path, model_name);
    Err(error)
}

/// The hub cache directory holding content-addressed blobs.
#[cfg(feature = "download")]
const BLOBS_DIR_NAME: &str = "blobs";

/// Remove a corrupt artifact and its backing blob from the hub cache.
///
/// See [`verify_or_evict`] for why both are removed and how the blob is scoped.
#[cfg(feature = "download")]
fn evict_artifact(path: &Path, model_name: &str) {
    let blob = backing_blob(path);
    if let Err(source) = std::fs::remove_file(path) {
        tracing::warn!(
            model = model_name,
            path = %path.display(),
            error = %source,
            "could not evict the corrupt model artifact; clear the cache entry manually"
        );
    } else {
        tracing::warn!(
            model = model_name,
            path = %path.display(),
            "evicted a corrupt model artifact from the cache; it will be re-downloaded"
        );
    }
    let Some(blob) = blob else { return };
    if let Err(source) = std::fs::remove_file(&blob) {
        tracing::warn!(
            model = model_name,
            path = %blob.display(),
            error = %source,
            "could not evict the corrupt model blob; clear the cache entry manually"
        );
    } else {
        tracing::warn!(
            model = model_name,
            path = %blob.display(),
            "evicted the corrupt model blob backing the cached artifact"
        );
    }
}

/// The `blobs/` file a snapshot symlink points at, when it is one.
///
/// Returns `None` for a regular file, an unresolvable link, or any target that is
/// not a regular file living directly inside a `blobs` directory — the guard that
/// keeps eviction inside the hub cache layout.
#[cfg(feature = "download")]
fn backing_blob(path: &Path) -> Option<PathBuf> {
    let metadata = std::fs::symlink_metadata(path).ok()?;
    if !metadata.file_type().is_symlink() {
        return None;
    }
    let target = std::fs::canonicalize(path).ok()?;
    let parent_is_blobs = target
        .parent()
        .and_then(Path::file_name)
        .is_some_and(|name| name == BLOBS_DIR_NAME);
    if parent_is_blobs && target.is_file() {
        Some(target)
    } else {
        None
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
            repo_cache_dir_name("xberg-io/sceptre-english_g2"),
            "models--xberg-io--sceptre-english_g2"
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
        let repo = "xberg-io/sceptre-english_g2";
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

        assert_eq!(
            resolve_cached(&root, "xberg-io/sceptre-english_g2", "english_g2.onnx"),
            None
        );

        std::fs::remove_dir_all(&root).ok();
    }

    /// A unique, per-test temporary directory path so tests stay order-independent
    /// and safely parallel.
    pub(super) fn unique_temp_dir(label: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);

        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("sceptre-{label}-{}-{unique}", std::process::id()))
    }

    /// Create `<root>/<repo cache dir>/snapshots/<revision>` and return it.
    pub(super) fn snapshot_dir(root: &Path, repo: &str, revision: &str) -> PathBuf {
        let snapshot = root.join(repo_cache_dir_name(repo)).join("snapshots").join(revision);
        std::fs::create_dir_all(&snapshot).expect("snapshot directory should be creatable");
        snapshot
    }

    const TEST_REPO: &str = "xberg-io/sceptre-english_g2";
    const TEST_FILE: &str = "english_g2.onnx";

    #[test]
    fn resolve_cached_returns_none_for_a_dangling_symlink() {
        #[cfg(unix)]
        {
            let root = unique_temp_dir("resolve-dangling");
            let snapshot = snapshot_dir(&root, TEST_REPO, "rev0");
            let blobs = root.join(repo_cache_dir_name(TEST_REPO)).join("blobs");
            std::fs::create_dir_all(&blobs).unwrap();
            std::os::unix::fs::symlink(blobs.join("missing-etag"), snapshot.join(TEST_FILE)).unwrap();

            assert_eq!(
                resolve_cached(&root, TEST_REPO, TEST_FILE),
                None,
                "a dangling snapshot symlink must not be reported as cached"
            );

            std::fs::remove_dir_all(&root).ok();
        }
    }

    #[test]
    fn resolve_cached_returns_none_for_a_zero_length_file() {
        let root = unique_temp_dir("resolve-empty-file");
        let snapshot = snapshot_dir(&root, TEST_REPO, "rev0");
        std::fs::write(snapshot.join(TEST_FILE), b"").unwrap();

        assert_eq!(
            resolve_cached(&root, TEST_REPO, TEST_FILE),
            None,
            "a zero-length artifact must not be reported as cached"
        );

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn resolve_cached_returns_the_path_for_a_healthy_file() {
        let root = unique_temp_dir("resolve-healthy");
        let snapshot = snapshot_dir(&root, TEST_REPO, "rev0");
        let planted = snapshot.join(TEST_FILE);
        std::fs::write(&planted, b"onnx-bytes").unwrap();

        assert_eq!(resolve_cached(&root, TEST_REPO, TEST_FILE), Some(planted));

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn resolve_cached_prefers_a_healthy_snapshot_over_a_broken_newer_one() {
        let root = unique_temp_dir("resolve-mixed");
        let healthy = snapshot_dir(&root, TEST_REPO, "rev-old").join(TEST_FILE);
        std::fs::write(&healthy, b"onnx-bytes").unwrap();
        let broken = snapshot_dir(&root, TEST_REPO, "rev-new").join(TEST_FILE);
        std::fs::write(&broken, b"").unwrap();

        assert_eq!(resolve_cached(&root, TEST_REPO, TEST_FILE), Some(healthy));

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
    fn verify_or_evict_removes_a_corrupt_artifact_and_leaves_the_cache_a_miss() {
        use super::tests::{snapshot_dir, unique_temp_dir};

        let root = unique_temp_dir("evict-plain");
        let repo = "xberg-io/sceptre-english_g2";
        let file = "english_g2.onnx";
        let planted = snapshot_dir(&root, repo, "rev0").join(file);
        std::fs::write(&planted, b"corrupt").unwrap();

        let error = verify_or_evict(&planted, ABC_SHA256, "english_g2").expect_err("a mismatched pin must error");
        assert!(matches!(error, OcrError::Model { .. }), "expected OcrError::Model");
        assert!(!planted.exists(), "the corrupt artifact must be evicted");
        assert_eq!(
            resolve_cached(&root, repo, file),
            None,
            "the next run must see a cache miss and re-download"
        );

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn verify_or_evict_removes_both_the_snapshot_symlink_and_its_blob() {
        #[cfg(unix)]
        {
            use super::tests::{snapshot_dir, unique_temp_dir};

            let root = unique_temp_dir("evict-symlink");
            let repo = "xberg-io/sceptre-english_g2";
            let file = "english_g2.onnx";
            let snapshot = snapshot_dir(&root, repo, "rev0");
            let blobs = root.join(repo_cache_dir_name(repo)).join("blobs");
            std::fs::create_dir_all(&blobs).unwrap();
            let blob = blobs.join("etag0");
            std::fs::write(&blob, b"corrupt").unwrap();
            let link = snapshot.join(file);
            std::os::unix::fs::symlink(&blob, &link).unwrap();

            let error = verify_or_evict(&link, ABC_SHA256, "english_g2").expect_err("a mismatched pin must error");
            assert!(matches!(error, OcrError::Model { .. }), "expected OcrError::Model");
            assert!(!blob.exists(), "the corrupt blob must be evicted");
            assert!(
                std::fs::symlink_metadata(&link).is_err(),
                "the snapshot symlink must be evicted"
            );
            assert_eq!(resolve_cached(&root, repo, file), None, "the cache must read as a miss");

            std::fs::remove_dir_all(&root).ok();
        }
    }

    #[test]
    fn verify_or_evict_keeps_a_matching_artifact() {
        use super::tests::{snapshot_dir, unique_temp_dir};

        let root = unique_temp_dir("evict-ok");
        let planted = snapshot_dir(&root, "xberg-io/sceptre-english_g2", "rev0").join("english_g2.onnx");
        std::fs::write(&planted, b"abc").unwrap();

        verify_or_evict(&planted, ABC_SHA256, "english_g2").expect("a matching pin must pass");
        assert!(planted.exists(), "a healthy artifact must be left in place");

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
