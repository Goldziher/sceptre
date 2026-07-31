//! The gen2 model registry.
//!
//! Source: the `itextresearch/itext-EasyOCR-*` ONNX repos on Hugging Face
//! (Apache-2.0, dynamic-width). `sha256` values are filled in once each artifact
//! is pinned.

use crate::config::Language;
use crate::error::{OcrError, Result};

/// A single downloadable ONNX model artifact.
// Fields describe the artifact consumed by `download::ensure`. ~keep
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct ModelEntry {
    /// Logical model name (EasyOCR naming).
    pub name: &'static str,
    /// Hugging Face repo id hosting the ONNX file.
    pub hf_repo: &'static str,
    /// File name within the repo.
    pub file: &'static str,
    /// Expected SHA-256 (hex) of the ONNX file; empty until pinned.
    pub sha256: &'static str,
}

/// The CRAFT text-detection model.
pub const fn craft_entry() -> ModelEntry {
    ModelEntry {
        name: "craft_mlt_25k",
        hf_repo: "itextresearch/itext-EasyOCR-craft_mlt_25k",
        file: "itext-EasyOCR-craft_mlt_25k.onnx",
        sha256: "",
    }
}

/// Resolve the effective Hugging Face repo id for `entry`.
///
/// With `registry_owner` `None`, the entry's own [`ModelEntry::hf_repo`] is
/// returned verbatim — the default `itextresearch/itext-EasyOCR-*` ids from the
/// mirror decision in ADR 0011. With `Some(owner)`, only the owner segment is
/// swapped, so identical exports can be hosted under a different account or
/// mirror without any code change; the repo-name segment (`itext-EasyOCR-<model>`)
/// is kept. A `hf_repo` without a `/` separator (short-form id) is treated as a
/// bare repo name and prefixed with `owner`.
///
/// The override is validated ([`owner_is_safe`]) before it reaches the cache-path
/// construction in [`crate::models::download`], so it cannot inject a path
/// separator or `..` traversal; an invalid owner is an [`OcrError::Config`].
pub fn effective_repo(entry: &ModelEntry, registry_owner: Option<&str>) -> Result<String> {
    let Some(owner) = registry_owner else {
        return Ok(entry.hf_repo.to_string());
    };
    if !owner_is_safe(owner) {
        return Err(OcrError::config(format!(
            "registry_owner `{owner}` is not a valid Hugging Face owner (allowed: letters, digits, `-`, `_`)"
        )));
    }
    Ok(match entry.hf_repo.split_once('/') {
        Some((_, repo_name)) => format!("{owner}/{repo_name}"),
        None => format!("{owner}/{}", entry.hf_repo),
    })
}

/// Whether `owner` is a safe Hugging Face owner segment.
///
/// Restricting to ASCII alphanumerics plus `-`/`_` keeps a `registry_owner`
/// override from smuggling path separators (`/`, `\`) or `..` traversal into the
/// on-disk cache layout derived from the repo id.
fn owner_is_safe(owner: &str) -> bool {
    !owner.is_empty() && owner.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

/// The gen2 recognizer for a language group.
pub const fn recognizer_entry(language: Language) -> ModelEntry {
    match language {
        Language::English => ModelEntry {
            name: "english_g2",
            hf_repo: "itextresearch/itext-EasyOCR-english_g2",
            file: "itext-EasyOCR-english_g2.onnx",
            sha256: "",
        },
        Language::Latin => ModelEntry {
            name: "latin_g2",
            hf_repo: "itextresearch/itext-EasyOCR-latin_g2",
            file: "itext-EasyOCR-latin_g2.onnx",
            sha256: "",
        },
        Language::ChineseSimplified => ModelEntry {
            name: "zh_sim_g2",
            hf_repo: "itextresearch/itext-EasyOCR-zh_sim_g2",
            file: "itext-EasyOCR-zh_sim_g2.onnx",
            sha256: "",
        },
        Language::Japanese => ModelEntry {
            name: "japanese_g2",
            hf_repo: "itextresearch/itext-EasyOCR-japanese_g2",
            file: "itext-EasyOCR-japanese_g2.onnx",
            sha256: "",
        },
        Language::Korean => ModelEntry {
            name: "korean_g2",
            hf_repo: "itextresearch/itext-EasyOCR-korean_g2",
            file: "itext-EasyOCR-korean_g2.onnx",
            sha256: "",
        },
        Language::Cyrillic => ModelEntry {
            name: "cyrillic_g2",
            hf_repo: "itextresearch/itext-EasyOCR-cyrillic_g2",
            file: "itext-EasyOCR-cyrillic_g2.onnx",
            sha256: "",
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_repo_without_override_yields_default_itextresearch_id() {
        assert_eq!(
            effective_repo(&craft_entry(), None).unwrap(),
            "itextresearch/itext-EasyOCR-craft_mlt_25k"
        );
        assert_eq!(
            effective_repo(&recognizer_entry(Language::English), None).unwrap(),
            "itextresearch/itext-EasyOCR-english_g2"
        );
    }

    #[test]
    fn effective_repo_with_override_swaps_only_the_owner_segment() {
        assert_eq!(
            effective_repo(&craft_entry(), Some("my-mirror")).unwrap(),
            "my-mirror/itext-EasyOCR-craft_mlt_25k"
        );
        assert_eq!(
            effective_repo(&recognizer_entry(Language::Cyrillic), Some("acme-org")).unwrap(),
            "acme-org/itext-EasyOCR-cyrillic_g2"
        );
    }

    #[test]
    fn effective_repo_prefixes_a_short_form_repo_id_with_the_owner() {
        let entry = ModelEntry {
            name: "bare",
            hf_repo: "bare-model",
            file: "bare-model.onnx",
            sha256: "",
        };
        assert_eq!(effective_repo(&entry, Some("owner")).unwrap(), "owner/bare-model");
    }

    #[test]
    fn effective_repo_rejects_an_owner_that_would_escape_the_cache_dir() {
        for malicious in ["../../etc", "a/b", "..", "own\\er", "", "owner with space"] {
            let error = effective_repo(&craft_entry(), Some(malicious))
                .expect_err("path-unsafe registry_owner must be rejected");
            assert!(
                matches!(error, OcrError::Config { .. }),
                "expected OcrError::Config for owner `{malicious}`, got {error:?}"
            );
        }
    }
}
