//! Shared helpers for the parity / golden test harness.
//!
//! This module carries pure, always-on logic (corpus path resolution, fuzzy
//! word-level F1, axis-aligned box IoU, golden-fixture parsing) plus the
//! [`HfCacheModelProvider`], which resolves the CRAFT + gen2 ONNX models from the
//! Hugging Face hub cache on disk without any download. See ADR 0016.

// Only a subset of these helpers is exercised by any single test binary or
// feature configuration, so unused-symbol warnings here are expected. ~keep
#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use sceptre::{BBox, Language, ModelProvider, OcrError, Result};

/// Logical name of the CRAFT detector export, matching `models::registry`.
const CRAFT_MODEL_NAME: &str = "craft_mlt_25k";

/// Owner segment of the upstream Hugging Face model repos, matching `models::registry`.
const REGISTRY_OWNER: &str = "itextresearch";

/// Walk up from `CARGO_MANIFEST_DIR` to the first ancestor that both looks like a
/// Cargo root (`Cargo.toml`) and contains a `test_documents/` directory, and return
/// that `test_documents` path. Falls back to the workspace-root guess (two levels
/// up from the crate manifest) when no ancestor matches, so the returned path is
/// always well-formed even when the submodule is absent.
pub fn test_documents_dir() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for ancestor in manifest.ancestors() {
        if ancestor.join("Cargo.toml").exists() && ancestor.join("test_documents").is_dir() {
            return ancestor.join("test_documents");
        }
    }
    manifest
        .parent()
        .and_then(Path::parent)
        .unwrap_or(&manifest)
        .join("test_documents")
}

/// Resolve a path relative to the `test_documents` corpus, returning `Some` only
/// when the file actually exists (the submodule may be absent or LFS-unpopulated).
pub fn skip_if_missing(relative_path: &str) -> Option<PathBuf> {
    let path = test_documents_dir().join(relative_path);
    path.exists().then_some(path)
}

/// Bag-of-words F1 between a hypothesis and a reference string.
///
/// Whitespace-tokenized and case-folded; scores multiset precision/recall so
/// repeated words count. Two empty strings score `1.0`; a single empty side
/// scores `0.0`.
pub fn word_f1(hypothesis: &str, reference: &str) -> f32 {
    let hypothesis_words = tokenize(hypothesis);
    let reference_words = tokenize(reference);

    if hypothesis_words.is_empty() && reference_words.is_empty() {
        return 1.0;
    }
    if hypothesis_words.is_empty() || reference_words.is_empty() {
        return 0.0;
    }

    let mut remaining = reference_words.clone();
    let mut matched = 0usize;
    for word in &hypothesis_words {
        if let Some(position) = remaining.iter().position(|candidate| candidate == word) {
            remaining.remove(position);
            matched += 1;
        }
    }

    if matched == 0 {
        return 0.0;
    }

    let precision = matched as f32 / hypothesis_words.len() as f32;
    let recall = matched as f32 / reference_words.len() as f32;
    2.0 * precision * recall / (precision + recall)
}

fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace().map(str::to_lowercase).collect()
}

/// Intersection-over-union of two axis-aligned boxes, used to score box parity
/// against the golden fixture. Returns `0.0` for degenerate or disjoint boxes.
pub fn box_iou(a: BBox, b: BBox) -> f32 {
    let x_min = a.x_min.max(b.x_min);
    let y_min = a.y_min.max(b.y_min);
    let x_max = a.x_max.min(b.x_max);
    let y_max = a.y_max.min(b.y_max);
    let intersection = (x_max - x_min).max(0.0) * (y_max - y_min).max(0.0);

    let area_a = (a.x_max - a.x_min) * (a.y_max - a.y_min);
    let area_b = (b.x_max - b.x_min) * (b.y_max - b.y_min);
    let union = area_a + area_b - intersection;

    if union <= 0.0 { 0.0 } else { intersection / union }
}

/// A single golden line: recognized text plus its four-corner quad.
#[derive(Debug, Clone)]
pub struct GoldenLine {
    /// Recognized text for the line.
    pub text: String,
    /// The four corners `[[x, y]; 4]`, clockwise from top-left.
    pub quad: [[f32; 2]; 4],
}

impl GoldenLine {
    /// Axis-aligned bounds of this line's quad, for IoU comparison.
    pub fn bbox(&self) -> BBox {
        let xs = self.quad.map(|point| point[0]);
        let ys = self.quad.map(|point| point[1]);
        BBox {
            x_min: xs.iter().copied().fold(f32::INFINITY, f32::min),
            y_min: ys.iter().copied().fold(f32::INFINITY, f32::min),
            x_max: xs.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            y_max: ys.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        }
    }

    fn from_value(value: &serde_json::Value) -> Self {
        let text = value["text"]
            .as_str()
            .expect("golden line `text` must be a string")
            .to_string();
        let corners = value["quad"]
            .as_array()
            .expect("golden line `quad` must be an array of 4 points");
        assert_eq!(corners.len(), 4, "golden line `quad` must have exactly 4 points");
        let mut quad = [[0.0f32; 2]; 4];
        for (slot, corner) in quad.iter_mut().zip(corners) {
            let point = corner.as_array().expect("each quad corner must be an [x, y] array");
            assert_eq!(point.len(), 2, "each quad corner must be an [x, y] pair");
            slot[0] = point[0].as_f64().expect("quad x must be a number") as f32;
            slot[1] = point[1].as_f64().expect("quad y must be a number") as f32;
        }
        Self { text, quad }
    }
}

/// One side of a dual golden (either the EasyOCR reference or the sceptre snapshot).
#[derive(Debug, Clone, Default)]
pub struct GoldenVariant {
    /// Recognized lines in reading order.
    pub lines: Vec<GoldenLine>,
}

impl GoldenVariant {
    fn from_value(value: &serde_json::Value) -> Self {
        let lines = value["lines"]
            .as_array()
            .map(|entries| entries.iter().map(GoldenLine::from_value).collect())
            .unwrap_or_default();
        Self { lines }
    }
}

/// A dual golden fixture: an authoritative EasyOCR reference and a sceptre snapshot
/// for one image. See `tests/data/golden/README.md` for the schema.
#[derive(Debug, Clone, Default)]
pub struct DualGolden {
    /// `true` while the fixture is an un-regenerated placeholder; parity assertions
    /// are skipped against a placeholder.
    pub placeholder: bool,
    /// The upstream Python EasyOCR reference output.
    pub easyocr: GoldenVariant,
    /// The sceptre snapshot output.
    pub sceptre: GoldenVariant,
}

impl DualGolden {
    /// Parse a dual golden fixture from JSON.
    pub fn parse(json: &str) -> Self {
        let value: serde_json::Value = serde_json::from_str(json).expect("dual golden fixture must be valid JSON");
        Self {
            placeholder: value["placeholder"].as_bool().unwrap_or(false),
            easyocr: GoldenVariant::from_value(&value["easyocr"]),
            sceptre: GoldenVariant::from_value(&value["sceptre"]),
        }
    }
}

/// Extract the `{"lines": ["...", ...]}` string array from a simple golden fixture.
/// Retained for the flat placeholder format and its unit tests.
pub fn golden_lines(json: &str) -> Vec<String> {
    let fixture: serde_json::Value = serde_json::from_str(json).expect("golden fixture must be valid JSON");
    fixture["lines"]
        .as_array()
        .expect("golden fixture must have a `lines` array")
        .iter()
        .map(|line| {
            line.as_str()
                .expect("each golden line must be a JSON string")
                .to_string()
        })
        .collect()
}

/// A [`ModelProvider`] that resolves the CRAFT + gen2 ONNX models from the local
/// Hugging Face hub cache — pure path resolution, no network and no download.
///
/// The cache root is `HF_HUB_CACHE` if set, else `$HF_HOME/hub`, else
/// `~/.cache/huggingface/hub`. Each model lives at
/// `<root>/models--<owner>--<name>/snapshots/<rev>/<file>`; the newest snapshot
/// directory containing the file wins.
pub struct HfCacheModelProvider;

impl HfCacheModelProvider {
    /// Construct a provider bound to the local Hugging Face hub cache.
    pub fn new() -> Self {
        Self
    }

    /// Whether both the CRAFT detector and the English recognizer resolve on disk,
    /// used to gate the real-model parity tests.
    pub fn available() -> bool {
        let provider = Self::new();
        provider.detector().is_ok() && provider.recognizer(Language::English).is_ok()
    }
}

impl Default for HfCacheModelProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelProvider for HfCacheModelProvider {
    fn detector(&self) -> Result<PathBuf> {
        let (repo, file) = model_repo_and_file(CRAFT_MODEL_NAME);
        resolve_in_hf_cache(&repo, &file)
            .ok_or_else(|| OcrError::model(format!("CRAFT model `{file}` not found in the Hugging Face cache")))
    }

    fn recognizer(&self, language: Language) -> Result<PathBuf> {
        let (repo, file) = model_repo_and_file(recognizer_model_name(language));
        resolve_in_hf_cache(&repo, &file)
            .ok_or_else(|| OcrError::model(format!("recognizer model `{file}` not found in the Hugging Face cache")))
    }
}

/// Logical gen2 recognizer model name for a language, matching `models::registry`.
fn recognizer_model_name(language: Language) -> &'static str {
    match language {
        Language::English => "english_g2",
        Language::Latin => "latin_g2",
        Language::ChineseSimplified => "zh_sim_g2",
        Language::Japanese => "japanese_g2",
        Language::Korean => "korean_g2",
        Language::Cyrillic => "cyrillic_g2",
    }
}

/// Map a logical model name to its `(repo_id, file_name)`, matching `models::registry`.
fn model_repo_and_file(model_name: &str) -> (String, String) {
    (
        format!("{REGISTRY_OWNER}/itext-EasyOCR-{model_name}"),
        format!("itext-EasyOCR-{model_name}.onnx"),
    )
}

/// Root of the Hugging Face hub cache, honoring `HF_HUB_CACHE` then `HF_HOME`.
fn hf_hub_cache_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("HF_HUB_CACHE") {
        if !dir.is_empty() {
            return Some(PathBuf::from(dir));
        }
    }
    if let Ok(home) = std::env::var("HF_HOME") {
        if !home.is_empty() {
            return Some(PathBuf::from(home).join("hub"));
        }
    }
    let home = std::env::var("HOME").ok().filter(|home| !home.is_empty())?;
    Some(PathBuf::from(home).join(".cache").join("huggingface").join("hub"))
}

/// The on-disk cache directory name for a `owner/name` repo id.
fn repo_cache_dir_name(repo_id: &str) -> String {
    format!("models--{}", repo_id.replace('/', "--"))
}

/// Resolve `<hub>/models--owner--name/snapshots/<rev>/<file>` against the discovered
/// Hugging Face cache root. `None` if the root or the file is unresolved.
fn resolve_in_hf_cache(repo_id: &str, file: &str) -> Option<PathBuf> {
    resolve_in_hub(&hf_hub_cache_dir()?, repo_id, file)
}

/// Resolve a model file within an explicit hub root, preferring the newest snapshot
/// directory that actually contains the file. Kept env-free so it is unit-testable.
fn resolve_in_hub(hub: &Path, repo_id: &str, file: &str) -> Option<PathBuf> {
    let snapshots = hub.join(repo_cache_dir_name(repo_id)).join("snapshots");
    let mut candidates: Vec<(SystemTime, PathBuf)> = fs::read_dir(&snapshots)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn square(x_min: f32, y_min: f32, x_max: f32, y_max: f32) -> BBox {
        BBox {
            x_min,
            y_min,
            x_max,
            y_max,
        }
    }

    #[test]
    fn should_score_identical_strings_as_word_f1_one() {
        assert_eq!(word_f1("easy ocr rocks", "easy ocr rocks"), 1.0);
    }

    #[test]
    fn should_score_disjoint_strings_as_word_f1_zero() {
        assert_eq!(word_f1("alpha beta", "gamma delta"), 0.0);
    }

    #[test]
    fn should_be_case_insensitive_in_word_f1() {
        assert_eq!(word_f1("Easy OCR", "easy ocr"), 1.0);
    }

    #[test]
    fn should_score_two_empty_strings_as_word_f1_one() {
        assert_eq!(word_f1("", ""), 1.0);
    }

    #[test]
    fn should_score_one_empty_string_as_word_f1_zero() {
        assert_eq!(word_f1("something", ""), 0.0);
    }

    #[test]
    fn should_score_partial_overlap_word_f1() {
        // hypothesis {a, b}, reference {a, b, c}: precision 1.0, recall 2/3, F1 0.8. ~keep
        assert!((word_f1("a b", "a b c") - 0.8).abs() < 1e-6);
    }

    #[test]
    fn should_score_identical_boxes_as_iou_one() {
        let a = square(0.0, 0.0, 10.0, 10.0);
        assert_eq!(box_iou(a, a), 1.0);
    }

    #[test]
    fn should_score_disjoint_boxes_as_iou_zero() {
        assert_eq!(box_iou(square(0.0, 0.0, 1.0, 1.0), square(5.0, 5.0, 6.0, 6.0)), 0.0);
    }

    #[test]
    fn should_score_half_overlapping_boxes_as_iou_one_third() {
        // intersection = 1x1 = 1, union = 2 + 2 - 1 = 3. ~keep
        assert_eq!(
            box_iou(square(0.0, 0.0, 2.0, 1.0), square(1.0, 0.0, 3.0, 1.0)),
            1.0 / 3.0
        );
    }

    #[test]
    fn should_parse_flat_golden_lines() {
        assert_eq!(golden_lines(r#"{"lines": ["EASY OCR"]}"#), vec!["EASY OCR".to_string()]);
    }

    #[test]
    fn should_parse_dual_golden_with_quads() {
        let json = r#"{
            "easyocr": {"lines": [{"text": "hi", "quad": [[0,0],[10,0],[10,4],[0,4]]}]},
            "sceptre": {"lines": [{"text": "hi", "quad": [[0,0],[10,0],[10,4],[0,4]]}]}
        }"#;
        let golden = DualGolden::parse(json);

        assert!(!golden.placeholder);
        assert_eq!(golden.easyocr.lines.len(), 1);
        assert_eq!(golden.easyocr.lines[0].text, "hi");
        assert_eq!(golden.easyocr.lines[0].bbox(), square(0.0, 0.0, 10.0, 4.0));
    }

    #[test]
    fn should_recognize_a_placeholder_dual_golden() {
        let golden = DualGolden::parse(r#"{"placeholder": true, "easyocr": {"lines": []}, "sceptre": {"lines": []}}"#);
        assert!(golden.placeholder);
        assert!(golden.easyocr.lines.is_empty());
    }

    #[test]
    fn should_map_registry_repo_and_file_for_english() {
        let (repo, file) = model_repo_and_file(recognizer_model_name(Language::English));
        assert_eq!(repo, "itextresearch/itext-EasyOCR-english_g2");
        assert_eq!(file, "itext-EasyOCR-english_g2.onnx");
    }

    #[test]
    fn should_map_registry_repo_and_file_for_craft() {
        let (repo, file) = model_repo_and_file(CRAFT_MODEL_NAME);
        assert_eq!(repo, "itextresearch/itext-EasyOCR-craft_mlt_25k");
        assert_eq!(file, "itext-EasyOCR-craft_mlt_25k.onnx");
    }

    #[test]
    fn should_build_hf_cache_dir_name_from_repo_id() {
        assert_eq!(
            repo_cache_dir_name("itextresearch/itext-EasyOCR-english_g2"),
            "models--itextresearch--itext-EasyOCR-english_g2"
        );
    }

    #[test]
    fn should_resolve_a_model_planted_in_a_fake_hub_root() {
        let root = std::env::temp_dir().join(format!("sceptre-hf-resolve-{}", std::process::id()));
        let snapshot = root
            .join("models--itextresearch--itext-EasyOCR-english_g2")
            .join("snapshots")
            .join("deadbeef");
        fs::create_dir_all(&snapshot).expect("create fake snapshot dir");
        let file = snapshot.join("itext-EasyOCR-english_g2.onnx");
        fs::write(&file, b"onnx").expect("write fake model file");

        let resolved = resolve_in_hub(
            &root,
            "itextresearch/itext-EasyOCR-english_g2",
            "itext-EasyOCR-english_g2.onnx",
        );

        assert_eq!(resolved, Some(file));
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn should_not_resolve_a_missing_model_in_an_empty_hub_root() {
        let root = std::env::temp_dir().join(format!("sceptre-hf-empty-{}", std::process::id()));
        fs::create_dir_all(&root).expect("create empty hub root");

        let resolved = resolve_in_hub(
            &root,
            "itextresearch/itext-EasyOCR-english_g2",
            "itext-EasyOCR-english_g2.onnx",
        );

        assert_eq!(resolved, None);
        fs::remove_dir_all(&root).ok();
    }
}
