//! The gen2 model registry.
//!
//! Source: the first-party `sceptre-ocr/*` ONNX repos on Hugging Face (Apache-2.0,
//! dynamic-width), exported by the `sceptre_rs_tools` pipeline from EasyOCR's gen2
//! weights (see ADR 0025). Each `sha256` is pinned to our exported artifact and
//! verified on download (see [`crate::models::download`]).

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
        hf_repo: "sceptre-ocr/craft_mlt_25k",
        file: "craft_mlt_25k.onnx",
        sha256: "159f5ffecde90d567f54da3e449fb0eba54a1da791c429c3509984dcfd7f684e",
    }
}

/// Resolve the effective Hugging Face repo id for `entry`.
///
/// With `registry_owner` `None`, the entry's own [`ModelEntry::hf_repo`] is
/// returned verbatim — the default first-party `sceptre-ocr/*` ids (ADR 0025).
/// With `Some(owner)`, only the owner segment is swapped, so identical exports can
/// be hosted under a different account or mirror without any code change; the
/// repo-name segment (`<model>`) is kept (ADR 0011). A `hf_repo` without a `/`
/// separator (short-form id) is treated as a bare repo name and prefixed with
/// `owner`.
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
            hf_repo: "sceptre-ocr/english_g2",
            file: "english_g2.onnx",
            sha256: "29ef336a7ad835d3e16eb70f23973d0fefd7a81fe6070845850954fe1cece0db",
        },
        Language::Latin => ModelEntry {
            name: "latin_g2",
            hf_repo: "sceptre-ocr/latin_g2",
            file: "latin_g2.onnx",
            sha256: "69ff1f543bb2b733708d4ab9e25d7dd3a9a25033da163698236211e36df4c787",
        },
        Language::ChineseSimplified => ModelEntry {
            name: "zh_sim_g2",
            hf_repo: "sceptre-ocr/zh_sim_g2",
            file: "zh_sim_g2.onnx",
            sha256: "4ce0cde647eb9305ec2ed9dfd79bd27ab573750424be2fe46daf91093dd0464f",
        },
        Language::Japanese => ModelEntry {
            name: "japanese_g2",
            hf_repo: "sceptre-ocr/japanese_g2",
            file: "japanese_g2.onnx",
            sha256: "daa18f93ff2cb5d3666fa3e0f14909d520e17c90bb49d2aa1ad633d8c4f8bd45",
        },
        Language::Korean => ModelEntry {
            name: "korean_g2",
            hf_repo: "sceptre-ocr/korean_g2",
            file: "korean_g2.onnx",
            sha256: "ccfd19e313c112999f6faa72d9e5bc4d67e11fedc96169a8de9e72c8110b3de8",
        },
        Language::Cyrillic => ModelEntry {
            name: "cyrillic_g2",
            hf_repo: "sceptre-ocr/cyrillic_g2",
            file: "cyrillic_g2.onnx",
            sha256: "71fb248ecb7fd5e47333ea5269fd1abdb82934b01ae877bbdad95e9ce2e2ea9b",
        },
        Language::Telugu => ModelEntry {
            name: "telugu_g2",
            hf_repo: "sceptre-ocr/telugu_g2",
            file: "telugu_g2.onnx",
            sha256: "d9bd97bc07c48504722c8a0e69ac5e1394e069d79fd51a17fd40d51192ac2ef4",
        },
        Language::Kannada => ModelEntry {
            name: "kannada_g2",
            hf_repo: "sceptre-ocr/kannada_g2",
            file: "kannada_g2.onnx",
            sha256: "86b925fd3eef9557792b017206c01b3fcf6824099472a3be54748042ffe21334",
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_repo_without_override_yields_default_sceptre_ocr_id() {
        assert_eq!(
            effective_repo(&craft_entry(), None).unwrap(),
            "sceptre-ocr/craft_mlt_25k"
        );
        assert_eq!(
            effective_repo(&recognizer_entry(Language::English), None).unwrap(),
            "sceptre-ocr/english_g2"
        );
        assert_eq!(
            effective_repo(&recognizer_entry(Language::Telugu), None).unwrap(),
            "sceptre-ocr/telugu_g2"
        );
    }

    #[test]
    fn effective_repo_with_override_swaps_only_the_owner_segment() {
        assert_eq!(
            effective_repo(&craft_entry(), Some("my-mirror")).unwrap(),
            "my-mirror/craft_mlt_25k"
        );
        assert_eq!(
            effective_repo(&recognizer_entry(Language::Cyrillic), Some("acme-org")).unwrap(),
            "acme-org/cyrillic_g2"
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
