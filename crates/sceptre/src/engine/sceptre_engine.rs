//! The default [`OcrEngine`] implementation: [`SceptreEngine`].

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use image::GrayImage;
use rayon::prelude::*;

use crate::config::{Language, OcrConfig, resolve_thread_budget};
use crate::detect::{CraftDetector, DetectedRegions, DetectorInput, TextDetector};
use crate::error::{OcrError, Result};
use crate::imaging::to_grayscale;
use crate::inference::{ModelBackend, load_backend};
use crate::recognize::{Charset, CrnnRecognizer, RecognizedText, RegionCrop, TextRecognizer, crop_region};
use crate::types::{Image, OcrResult, Point, QUAD_CORNERS, Quad, TextLine};

use super::seams::{ModelProvider, ProgressSink};
use super::{OcrEngine, ReadOptions};

/// Progress label emitted before the detection stage.
const STAGE_DETECT: &str = "detect";
/// Progress label emitted before the recognition stage.
const STAGE_RECOGNIZE: &str = "recognize";

/// The default engine: CRAFT detection followed by CRNN + CTC recognition.
///
/// Holds the injectable seams and configuration consumed by the detect → crop →
/// recognize pipeline. The detector and recognizer backends are built lazily on the
/// first [`recognize`](SceptreEngine::recognize) call — loading a model is fallible
/// I/O that cannot happen in an infallible constructor — then cached, so a reused
/// [`Reader`](crate::Reader) initializes each ONNX session once, not per call.
pub(crate) struct SceptreEngine {
    config: OcrConfig,
    models: Arc<dyn ModelProvider>,
    progress: Arc<dyn ProgressSink>,
    detector_cache: OnceLock<Arc<dyn ModelBackend>>,
    recognizer_cache: OnceLock<Arc<dyn ModelBackend>>,
}

impl SceptreEngine {
    /// Build the default engine from resolved seams and configuration.
    pub(crate) fn new(config: OcrConfig, models: Arc<dyn ModelProvider>, progress: Arc<dyn ProgressSink>) -> Self {
        Self {
            config,
            models,
            progress,
            detector_cache: OnceLock::new(),
            recognizer_cache: OnceLock::new(),
        }
    }

    /// The recognizer to serve the run, resolved from the configured language set.
    ///
    /// Every gen2 recognizer already packs Latin/English into its charset, so English
    /// is the common base: a set is served by the single non-English recognizer that
    /// covers it (e.g. `[English, Korean]` → `korean_g2`, which recognizes both), and
    /// an English-only or empty set falls back to `english_g2`. Two or more distinct
    /// non-English languages need different models and are rejected, mirroring
    /// EasyOCR's language-group model selection.
    fn language(&self) -> Result<Language> {
        resolve_recognition_language(&self.config.model.languages)
    }

    /// Read a model file resolved by the provider and load it into the configured
    /// backend, capped to the shared thread budget.
    fn load(&self, model_path: PathBuf) -> Result<Arc<dyn ModelBackend>> {
        let bytes = std::fs::read(&model_path)?;
        let budget = resolve_thread_budget(Some(&self.config.concurrency));
        let backend: Arc<dyn ModelBackend> = Arc::from(load_backend(self.config.model.backend, &bytes, budget)?);
        tracing::debug!(backend = backend.name(), path = %model_path.display(), "loaded inference backend");
        Ok(backend)
    }

    /// The CRAFT detector backend, loaded once and cached for later calls.
    fn detector_backend(&self) -> Result<Arc<dyn ModelBackend>> {
        if let Some(backend) = self.detector_cache.get() {
            return Ok(backend.clone());
        }
        let backend = self.load(self.models.detector()?)?;
        // A concurrent first call may win the race to `set`; both loaded sessions are ~keep
        // valid, so the loser just uses its own copy and drops the cache write. ~keep
        let _ = self.detector_cache.set(backend.clone());
        Ok(backend)
    }

    /// The recognizer backend for `language`, loaded once and cached for later calls.
    fn recognizer_backend(&self, language: Language) -> Result<Arc<dyn ModelBackend>> {
        if let Some(backend) = self.recognizer_cache.get() {
            return Ok(backend.clone());
        }
        let backend = self.load(self.models.recognizer(language)?)?;
        let _ = self.recognizer_cache.set(backend.clone());
        Ok(backend)
    }

    /// Detect candidate text regions with the CRAFT detector.
    fn detect_regions(&self, image: &Image) -> Result<DetectedRegions> {
        let detector = CraftDetector::new(self.detector_backend()?, self.config.detection.clone());
        detector.detect(&DetectorInput { image })
    }

    /// Recognize the cropped regions with the CRNN + CTC recognizer.
    fn recognize_crops(&self, crops: &[RegionCrop]) -> Result<Vec<RecognizedText>> {
        let language = self.language()?;
        let recognizer = CrnnRecognizer::new(
            self.recognizer_backend(language)?,
            Charset::for_language(language),
            self.config.recognition.clone(),
        );
        recognizer.recognize(crops)
    }
}

impl OcrEngine for SceptreEngine {
    fn recognize(&self, image: &Image, _options: &ReadOptions) -> Result<OcrResult> {
        self.config.recognition.validate()?;
        self.progress.on_stage(STAGE_DETECT);
        let regions = self.detect_regions(image)?;
        if regions.regions.is_empty() {
            return Ok(OcrResult::default());
        }

        let grey = to_grayscale(image)?;
        let crops = crop_regions(&grey, &regions);
        if crops.is_empty() {
            return Ok(OcrResult::default());
        }

        self.progress.on_stage(STAGE_RECOGNIZE);
        let texts = self.recognize_crops(&crops)?;
        Ok(build_result(&crops, texts, self.config.recognition.filter_ths))
    }

    fn detect(&self, image: &Image, _options: &ReadOptions) -> Result<Vec<Quad>> {
        self.progress.on_stage(STAGE_DETECT);
        let regions = self.detect_regions(image)?;
        Ok(regions
            .regions
            .iter()
            .map(|region| corners_to_quad(&region.corners))
            .collect())
    }

    fn recognize_line(&self, image: &Image, _options: &ReadOptions) -> Result<TextLine> {
        self.config.recognition.validate()?;
        self.progress.on_stage(STAGE_RECOGNIZE);
        let grey = to_grayscale(image)?;
        let corners = full_image_corners(grey.width(), grey.height());
        let crop = RegionCrop {
            width: grey.width(),
            height: grey.height(),
            gray: grey.into_raw(),
            corners,
        };
        let recognized = self
            .recognize_crops(&[crop])?
            .into_iter()
            .next()
            .ok_or_else(|| OcrError::inference("recognizer returned no result for the line crop"))?;
        Ok(TextLine {
            quad: corners_to_quad(&corners),
            text: recognized.text,
            confidence: recognized.confidence,
        })
    }
}

/// Resolve the single gen2 recognizer that serves a configured language set.
///
/// English is the common base every gen2 charset already covers, so the set is served
/// by its one non-English language (`[English, Korean]` → `Korean`), or by `English`
/// when the set is English-only or empty. Two or more distinct non-English languages
/// need different models and are rejected (mirrors EasyOCR's group selection).
fn resolve_recognition_language(languages: &[Language]) -> Result<Language> {
    // Distinct non-English groups only: repeated entries (`--lang korean --lang korean`) ~keep
    // name one recognizer, so dedupe before checking arity rather than matching length. ~keep
    let mut non_english: Vec<Language> = Vec::new();
    for language in languages
        .iter()
        .copied()
        .filter(|language| *language != Language::English)
    {
        if !non_english.contains(&language) {
            non_english.push(language);
        }
    }
    match non_english.as_slice() {
        [] => Ok(Language::English),
        [single] => Ok(*single),
        _ => Err(OcrError::config(format!(
            "languages {non_english:?} require different recognizer models; configure at most one \
             non-English language (each gen2 model already covers Latin/English)"
        ))),
    }
}

/// Crop every detected region from `grey`, dropping any region that produces no crop
/// (a zero-area box after clamping, or a degenerate quad) so one bad region never
/// aborts the whole run. Dropped regions are logged rather than silently discarded.
///
/// Regions crop in parallel over the shared Rayon pool; `crop_region` is pure CPU and
/// `GrayImage` is `Sync`, so this is safe. `filter_map(...).collect()` preserves the
/// input order, keeping the output bit-identical to a sequential crop.
fn crop_regions(grey: &GrayImage, regions: &DetectedRegions) -> Vec<RegionCrop> {
    regions
        .regions
        .par_iter()
        .filter_map(|region| match crop_region(grey, &region.corners, region.axis_aligned) {
            Ok(crop) => Some(crop),
            Err(error) => {
                tracing::debug!(%error, "skipping a detected region that produced no crop");
                None
            }
        })
        .collect()
}

/// Map recognizer outputs above the configured confidence threshold back to public
/// text lines, carrying each surviving crop's source quad.
fn build_result(crops: &[RegionCrop], texts: Vec<RecognizedText>, filter_ths: f32) -> OcrResult {
    debug_assert_eq!(
        crops.len(),
        texts.len(),
        "the recognizer must return exactly one result per crop"
    );
    let lines = crops
        .iter()
        .zip(texts)
        // The threshold is inclusive so a caller can retain an exact boundary score. ~keep
        .filter(|(_, text)| text.confidence >= filter_ths)
        .map(|(crop, text)| TextLine {
            quad: corners_to_quad(&crop.corners),
            text: text.text,
            confidence: text.confidence,
        })
        .collect();
    OcrResult { lines }
}

/// Convert `[TL, TR, BR, BL]` `[x, y]` corners to a public [`Quad`].
fn corners_to_quad(corners: &[[f32; 2]; QUAD_CORNERS]) -> Quad {
    Quad {
        points: corners.map(|corner| Point::new(corner[0], corner[1])),
    }
}

/// The clockwise `[TL, TR, BR, BL]` corners spanning a whole `width`×`height` image.
fn full_image_corners(width: u32, height: u32) -> [[f32; 2]; QUAD_CORNERS] {
    let width = width as f32;
    let height = height as f32;
    [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RecognitionConfig;
    use crate::detect::DetectedRegion;

    #[test]
    fn resolve_language_uses_the_non_english_group_for_a_mixed_set() {
        // [English, Korean] is served by korean_g2, whose charset covers both. ~keep
        assert_eq!(
            resolve_recognition_language(&[Language::English, Language::Korean]).unwrap(),
            Language::Korean
        );
        assert_eq!(
            resolve_recognition_language(&[Language::Korean, Language::English]).unwrap(),
            Language::Korean
        );
    }

    #[test]
    fn resolve_language_falls_back_to_english_for_english_only_or_empty() {
        assert_eq!(resolve_recognition_language(&[]).unwrap(), Language::English);
        assert_eq!(
            resolve_recognition_language(&[Language::English]).unwrap(),
            Language::English
        );
    }

    #[test]
    fn resolve_language_keeps_a_single_non_english_group() {
        assert_eq!(
            resolve_recognition_language(&[Language::Latin]).unwrap(),
            Language::Latin
        );
        assert_eq!(
            resolve_recognition_language(&[Language::Japanese]).unwrap(),
            Language::Japanese
        );
    }

    #[test]
    fn resolve_language_dedupes_repeated_non_english_groups() {
        // Repeated flags/config entries name one recognizer, not two. ~keep
        assert_eq!(
            resolve_recognition_language(&[Language::Korean, Language::Korean]).unwrap(),
            Language::Korean
        );
        assert_eq!(
            resolve_recognition_language(&[Language::English, Language::Korean, Language::Korean]).unwrap(),
            Language::Korean
        );
    }

    #[test]
    fn resolve_language_rejects_two_distinct_non_english_groups() {
        let error = resolve_recognition_language(&[Language::Korean, Language::Japanese])
            .expect_err("korean + japanese need different models");
        assert!(matches!(error, OcrError::Config { .. }));
    }

    fn crop_with_corners(corners: [[f32; 2]; QUAD_CORNERS]) -> RegionCrop {
        RegionCrop {
            width: 2,
            height: 2,
            gray: vec![0u8; 4],
            corners,
        }
    }

    fn text(value: &str, confidence: f32) -> RecognizedText {
        RecognizedText {
            text: value.to_string(),
            confidence,
        }
    }

    #[test]
    fn should_map_corners_to_quad_in_clockwise_order() {
        let quad = corners_to_quad(&[[1.0, 2.0], [9.0, 2.0], [9.0, 6.0], [1.0, 6.0]]);
        assert_eq!(quad.points[0], Point::new(1.0, 2.0));
        assert_eq!(quad.points[1], Point::new(9.0, 2.0));
        assert_eq!(quad.points[2], Point::new(9.0, 6.0));
        assert_eq!(quad.points[3], Point::new(1.0, 6.0));
    }

    #[test]
    fn should_pair_each_crop_with_its_recognized_text_and_quad() {
        let crops = vec![
            crop_with_corners([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]]),
            crop_with_corners([[5.0, 0.0], [9.0, 0.0], [9.0, 3.0], [5.0, 3.0]]),
        ];
        let texts = vec![text("hello", 0.9), text("world", 0.8)];

        let result = build_result(&crops, texts, RecognitionConfig::default().filter_ths);

        assert_eq!(result.lines.len(), 2);
        assert_eq!(result.lines[0].text, "hello");
        assert_eq!(result.lines[0].quad.points[0], Point::new(0.0, 0.0));
        assert_eq!(result.lines[1].text, "world");
        assert_eq!(result.lines[1].confidence, 0.8);
    }

    #[test]
    fn should_drop_line_below_filter_threshold() {
        let crops = vec![crop_with_corners([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]])];
        let texts = vec![text("below", 0.49)];

        let result = build_result(&crops, texts, 0.5);

        assert_eq!(result.lines, Vec::<TextLine>::new());
    }

    #[test]
    fn should_keep_line_equal_to_filter_threshold() {
        let crops = vec![crop_with_corners([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]])];
        let texts = vec![text("equal", 0.5)];

        let result = build_result(&crops, texts, 0.5);

        assert_eq!(result.lines.len(), 1);
        assert_eq!(result.lines[0].text, "equal");
    }

    #[test]
    fn should_keep_line_above_filter_threshold() {
        let crops = vec![crop_with_corners([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]])];
        let texts = vec![text("above", 0.51)];

        let result = build_result(&crops, texts, 0.5);

        assert_eq!(result.lines.len(), 1);
        assert_eq!(result.lines[0].text, "above");
    }

    #[test]
    fn should_crop_only_regions_that_survive_clamping() {
        let grey = GrayImage::from_raw(4, 3, (0..12u8).collect()).expect("valid grayscale buffer");
        let regions = DetectedRegions {
            regions: vec![
                // A valid 2x2 window. ~keep
                DetectedRegion {
                    corners: [[1.0, 0.0], [3.0, 0.0], [3.0, 2.0], [1.0, 2.0]],
                    axis_aligned: true,
                },
                // A zero-area box that clamps to nothing and must be dropped. ~keep
                DetectedRegion {
                    corners: [[2.0, 1.0], [2.0, 1.0], [2.0, 1.0], [2.0, 1.0]],
                    axis_aligned: true,
                },
            ],
        };

        let crops = crop_regions(&grey, &regions);

        assert_eq!(crops.len(), 1, "the degenerate region is skipped, the valid one kept");
        assert_eq!(crops[0].width, 2);
        assert_eq!(crops[0].height, 2);
    }

    #[test]
    fn should_preserve_input_order_when_cropping_regions_in_parallel() {
        let grey = GrayImage::from_raw(4, 3, (0..12u8).collect()).expect("valid grayscale buffer");
        let regions = DetectedRegions {
            regions: vec![
                // A 1-wide window, identifiable by its width. ~keep
                DetectedRegion {
                    corners: [[0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [0.0, 2.0]],
                    axis_aligned: true,
                },
                // A zero-area box between the survivors that must be dropped. ~keep
                DetectedRegion {
                    corners: [[2.0, 1.0], [2.0, 1.0], [2.0, 1.0], [2.0, 1.0]],
                    axis_aligned: true,
                },
                // A 2-wide window. ~keep
                DetectedRegion {
                    corners: [[1.0, 0.0], [3.0, 0.0], [3.0, 2.0], [1.0, 2.0]],
                    axis_aligned: true,
                },
                // A 3-wide window. ~keep
                DetectedRegion {
                    corners: [[0.0, 0.0], [3.0, 0.0], [3.0, 2.0], [0.0, 2.0]],
                    axis_aligned: true,
                },
            ],
        };

        let crops = crop_regions(&grey, &regions);

        // The degenerate region is dropped; the survivors keep their input order, ~keep
        // proving parallel cropping does not reorder. ~keep
        let widths: Vec<u32> = crops.iter().map(|crop| crop.width).collect();
        assert_eq!(widths, vec![1, 2, 3], "surviving crops keep their input order");
    }
}
