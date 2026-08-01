//! Public model provisioning: enumerate and prefetch the configured models.

use std::path::{Path, PathBuf};

use crate::config::{Language, OcrConfig};
use crate::error::Result;

use super::download::{self, hf_cache_root, resolve_cached};
use super::registry::{ModelEntry, craft_entry, effective_repo, recognizer_entry};

/// The role a model plays in the pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelRole {
    /// The CRAFT text detector shared across languages.
    Detector,
    /// A per-language gen2 recognizer.
    Recognizer(Language),
}

/// A model required by a configuration, with its resolved repo id and cache status.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Logical model name (EasyOCR naming), e.g. `craft_mlt_25k`.
    pub name: String,
    /// Effective Hugging Face repo id (honors `registry_owner`).
    pub repo: String,
    /// Whether this is the detector or a per-language recognizer.
    pub role: ModelRole,
    /// Whether the artifact is already present in the cache.
    pub cached: bool,
    /// The cache path when present (`Some` iff `cached`).
    pub path: Option<PathBuf>,
}

/// The models required by `config`: the CRAFT detector plus one recognizer per
/// configured language (in order, duplicates preserved), each annotated with cache
/// status. Pure filesystem inspection — no network.
pub fn model_manifest(config: &OcrConfig) -> Result<Vec<ModelInfo>> {
    let cache_dir = resolve_cache_dir(config)?;
    let owner = config.model.registry_owner.as_deref();
    let mut manifest = Vec::with_capacity(config.model.languages.len() + 1);
    manifest.push(info_for(&craft_entry(), ModelRole::Detector, &cache_dir, owner)?);
    for &language in &config.model.languages {
        let entry = recognizer_entry(language);
        manifest.push(info_for(&entry, ModelRole::Recognizer(language), &cache_dir, owner)?);
    }
    Ok(manifest)
}

/// Ensure every model in `config`'s manifest is present locally, downloading any
/// that are missing, and return the resulting (now-cached) manifest. Requires the
/// `download` feature; without it the underlying fetch returns an [`crate::OcrError::Model`].
pub fn download_models(config: &OcrConfig) -> Result<Vec<ModelInfo>> {
    let cache_override = config.model.cache_dir.as_deref();
    let owner = config.model.registry_owner.as_deref();
    let mut manifest = Vec::with_capacity(config.model.languages.len() + 1);
    manifest.push(fetch(&craft_entry(), ModelRole::Detector, cache_override, owner)?);
    for &language in &config.model.languages {
        let entry = recognizer_entry(language);
        manifest.push(fetch(&entry, ModelRole::Recognizer(language), cache_override, owner)?);
    }
    Ok(manifest)
}

/// Resolve the Hugging Face hub cache root: the config override or the environment default.
fn resolve_cache_dir(config: &OcrConfig) -> Result<PathBuf> {
    hf_cache_root(config.model.cache_dir.as_deref())
}

/// Inspect the cache for `entry` without touching the network.
fn info_for(entry: &ModelEntry, role: ModelRole, cache_dir: &Path, owner: Option<&str>) -> Result<ModelInfo> {
    let repo = effective_repo(entry, owner)?;
    let path = resolve_cached(cache_dir, &repo, entry.file);
    let cached = path.is_some();
    Ok(ModelInfo {
        name: entry.name.to_string(),
        repo,
        role,
        cached,
        path,
    })
}

/// Download `entry` if missing and describe it as a now-cached [`ModelInfo`].
///
/// `cache_override` is the raw config override for the hub cache root (`None` uses
/// the environment default); `ensure` builds the hub client from it.
fn fetch(entry: &ModelEntry, role: ModelRole, cache_override: Option<&Path>, owner: Option<&str>) -> Result<ModelInfo> {
    let repo = effective_repo(entry, owner)?;
    let path = download::ensure(entry, cache_override, owner)?;
    Ok(ModelInfo {
        name: entry.name.to_string(),
        repo,
        role,
        cached: true,
        path: Some(path),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::OcrConfig;

    const CRAFT_REPO: &str = "itextresearch/itext-EasyOCR-craft_mlt_25k";
    const ENGLISH_REPO: &str = "itextresearch/itext-EasyOCR-english_g2";

    /// A config pointed at a fresh, empty temp cache dir so nothing reads cached.
    fn config_with_empty_cache(languages: Vec<Language>) -> OcrConfig {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);

        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        let cache_dir = std::env::temp_dir().join(format!("sceptre-manifest-{}-{unique}", std::process::id()));
        let mut config = OcrConfig::default();
        config.model.languages = languages;
        config.model.cache_dir = Some(cache_dir);
        config
    }

    #[test]
    fn model_manifest_lists_detector_then_english_recognizer_for_default_config() {
        let config = config_with_empty_cache(vec![Language::English]);
        let manifest = model_manifest(&config).unwrap();

        assert_eq!(manifest.len(), 2);
        assert_eq!(manifest[0].name, "craft_mlt_25k");
        assert_eq!(manifest[0].role, ModelRole::Detector);
        assert_eq!(manifest[0].repo, CRAFT_REPO);
        assert_eq!(manifest[1].name, "english_g2");
        assert_eq!(manifest[1].role, ModelRole::Recognizer(Language::English));
        assert_eq!(manifest[1].repo, ENGLISH_REPO);
    }

    #[test]
    fn model_manifest_reports_not_cached_against_a_fresh_temp_cache_dir() {
        let config = config_with_empty_cache(vec![Language::English]);
        let manifest = model_manifest(&config).unwrap();

        for info in &manifest {
            assert!(!info.cached, "expected `{}` to be uncached", info.name);
            assert_eq!(info.path, None);
        }
    }

    #[test]
    #[cfg(not(feature = "download"))]
    fn download_models_errors_without_the_download_feature() {
        // Without the `download` feature the underlying fetch cannot run, so the ~keep
        // whole prefetch fails with an `OcrError::Model`. ~keep
        let config = config_with_empty_cache(vec![Language::English]);
        let error = download_models(&config).expect_err("download requires the `download` feature");
        assert!(
            matches!(error, crate::error::OcrError::Model { .. }),
            "expected OcrError::Model, got {error:?}"
        );
    }

    #[test]
    fn model_manifest_yields_one_recognizer_per_language_in_order() {
        let languages = vec![Language::Cyrillic, Language::Japanese, Language::Korean];
        let config = config_with_empty_cache(languages.clone());
        let manifest = model_manifest(&config).unwrap();

        assert_eq!(manifest.len(), languages.len() + 1);
        assert_eq!(manifest[0].role, ModelRole::Detector);
        let recognizer_roles: Vec<ModelRole> = manifest[1..].iter().map(|info| info.role.clone()).collect();
        let expected: Vec<ModelRole> = languages.into_iter().map(ModelRole::Recognizer).collect();
        assert_eq!(recognizer_roles, expected);
    }
}
