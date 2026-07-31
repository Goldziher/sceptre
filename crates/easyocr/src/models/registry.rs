//! The gen2 model registry.
//!
//! Source: the `itextresearch/itext-EasyOCR-*` ONNX repos on Hugging Face
//! (Apache-2.0, dynamic-width). `sha256` values are filled in once each artifact
//! is pinned.

use crate::config::Language;

/// A single downloadable ONNX model artifact.
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
