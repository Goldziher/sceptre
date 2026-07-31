//! The default [`OcrEngine`] implementation: [`SceptreEngine`].

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use image::GrayImage;

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

    /// The recognition language group. gen2 packs each group into one recognizer, so
    /// a single model serves the run; a request for more than one language is an error
    /// because combined multi-language model selection is not yet implemented.
    fn language(&self) -> Result<Language> {
        match self.config.model.languages.as_slice() {
            [] => Ok(Language::default()),
            [single] => Ok(*single),
            _ => Err(OcrError::config(
                "multiple recognition languages are configured, but only single-language recognition is \
                 implemented; configure exactly one language",
            )),
        }
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
        // A concurrent first call may win the race to `set`; both loaded sessions are
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
    fn detect(&self, image: &Image) -> Result<DetectedRegions> {
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
        self.progress.on_stage(STAGE_DETECT);
        let regions = self.detect(image)?;
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
        Ok(build_result(&crops, texts))
    }
}

/// Crop every detected region from `grey`, dropping any region that produces no crop
/// (a zero-area box after clamping, or a degenerate quad) so one bad region never
/// aborts the whole run. Dropped regions are logged rather than silently discarded.
fn crop_regions(grey: &GrayImage, regions: &DetectedRegions) -> Vec<RegionCrop> {
    let mut crops = Vec::with_capacity(regions.regions.len());
    for region in &regions.regions {
        match crop_region(grey, &region.corners, region.axis_aligned) {
            Ok(crop) => crops.push(crop),
            Err(error) => tracing::debug!(%error, "skipping a detected region that produced no crop"),
        }
    }
    crops
}

/// Map recognizer outputs back to public text lines, carrying each crop's source
/// quad. Every crop yields a line, matching EasyOCR, which emits one result per
/// detected region without any confidence filtering (`filter_ths` is inert upstream).
fn build_result(crops: &[RegionCrop], texts: Vec<RecognizedText>) -> OcrResult {
    debug_assert_eq!(
        crops.len(),
        texts.len(),
        "the recognizer must return exactly one result per crop"
    );
    let lines = crops
        .iter()
        .zip(texts)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect::DetectedRegion;

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

        let result = build_result(&crops, texts);

        assert_eq!(result.lines.len(), 2);
        assert_eq!(result.lines[0].text, "hello");
        assert_eq!(result.lines[0].quad.points[0], Point::new(0.0, 0.0));
        assert_eq!(result.lines[1].text, "world");
        assert_eq!(result.lines[1].confidence, 0.8);
    }

    #[test]
    fn should_emit_every_line_including_low_confidence_matching_upstream() {
        // EasyOCR applies no confidence filter, so even a very low-confidence line is
        // emitted; sceptre must not silently drop it. ~keep
        let crops = vec![
            crop_with_corners([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]]),
            crop_with_corners([[5.0, 0.0], [9.0, 0.0], [9.0, 3.0], [5.0, 3.0]]),
        ];
        let texts = vec![text("confident", 0.9), text("faint", 0.0001)];

        let result = build_result(&crops, texts);

        assert_eq!(result.lines.len(), 2, "no line is dropped for low confidence");
        assert_eq!(result.lines[1].text, "faint");
    }

    #[test]
    fn should_emit_empty_all_blank_line_matching_upstream() {
        // An all-blank CTC decode yields empty text at confidence 0.0. EasyOCR still
        // appends it, so sceptre must too (parity over cleanliness). ~keep
        let crops = vec![crop_with_corners([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]])];
        let texts = vec![text("", 0.0)];

        let result = build_result(&crops, texts);

        assert_eq!(result.lines.len(), 1, "the empty region is emitted, not filtered");
        assert_eq!(result.lines[0].text, "");
    }

    #[test]
    fn should_crop_only_regions_that_survive_clamping() {
        let grey = GrayImage::from_raw(4, 3, (0..12u8).collect()).expect("valid grayscale buffer");
        let regions = DetectedRegions {
            regions: vec![
                // A valid 2x2 window.
                DetectedRegion {
                    corners: [[1.0, 0.0], [3.0, 0.0], [3.0, 2.0], [1.0, 2.0]],
                    axis_aligned: true,
                },
                // A zero-area box that clamps to nothing and must be dropped.
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
}
