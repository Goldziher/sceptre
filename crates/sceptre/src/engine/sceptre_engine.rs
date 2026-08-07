//! The default [`OcrEngine`] implementation: [`SceptreEngine`].

use std::borrow::Cow;
use std::sync::{Arc, Mutex};

use image::GrayImage;
use once_cell::sync::OnceCell;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::config::{Backend, Language, OcrConfig, resolve_thread_budget};
use crate::detect::orientation::{self, Rotation};
use crate::detect::{CraftDetector, DetectedRegions, DetectorInput, TextDetector};
use crate::error::{OcrError, Result};
use crate::imaging::to_grayscale;
use crate::inference::{BackendOptions, ModelBackend, NetworkKind, load_backend};
use crate::recognize::{Charset, CrnnRecognizer, RecognizedText, RegionCrop, TextRecognizer, crop_region};
use crate::types::{Image, OcrResult, Point, QUAD_CORNERS, Quad, TextLine};

use super::seams::{ModelArtifact, ModelProvider, ProgressSink};
use super::{OcrEngine, ReadOptions};

/// Progress label emitted before the detection stage.
const STAGE_DETECT: &str = "detect";
/// Progress label emitted before the recognition stage.
const STAGE_RECOGNIZE: &str = "recognize";
/// CRAFT spatial-dimension alignment; the fixed tract canvas is rounded up to this.
const DETECT_ALIGN: u32 = 32;

/// The default engine: CRAFT detection followed by CRNN + CTC recognition.
///
/// Holds the injectable seams and configuration consumed by the detect → crop →
/// recognize pipeline. The detector backend and recognizer are built lazily on the
/// first [`recognize`](SceptreEngine::recognize) call — loading a model is fallible
/// I/O that cannot happen in an infallible constructor — then cached, so a reused
/// [`Reader`](crate::Reader) initializes each ONNX session and charset once.
pub(crate) struct SceptreEngine {
    config: OcrConfig,
    models: Mutex<Option<Arc<dyn ModelProvider>>>,
    progress: Arc<dyn ProgressSink>,
    detector_cache: OnceCell<Arc<dyn ModelBackend>>,
    recognizer_cache: OnceCell<Arc<CrnnRecognizer>>,
}

impl SceptreEngine {
    /// Build the default engine from resolved seams and configuration.
    pub(crate) fn new(config: OcrConfig, models: Arc<dyn ModelProvider>, progress: Arc<dyn ProgressSink>) -> Self {
        Self {
            config,
            models: Mutex::new(Some(models)),
            progress,
            detector_cache: OnceCell::new(),
            recognizer_cache: OnceCell::new(),
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
    ///
    /// `network` states which network the bytes hold, for backends that run a hand-written
    /// forward pass rather than interpreting the graph. `fixed_input` pins the model
    /// input to a concrete shape (see [`load_backend`]); the CRAFT detector passes its
    /// fixed square canvas on the tract backend, and the recognizer always passes `None`.
    fn load(
        &self,
        artifact: ModelArtifact,
        network: NetworkKind,
        fixed_input: Option<&[usize]>,
    ) -> Result<Arc<dyn ModelBackend>> {
        let (bytes, source) = match artifact {
            ModelArtifact::Path(path) => {
                let bytes = std::fs::read(&path).map_err(|source| OcrError::Model {
                    message: format!("could not read model artifact `{}`", path.display()),
                    source: Some(Box::new(source)),
                })?;
                (bytes.into(), path.display().to_string())
            }
            ModelArtifact::Bytes(bytes) => (bytes, "memory".to_string()),
        };
        let options = BackendOptions {
            threads: resolve_thread_budget(Some(&self.config.concurrency)),
            fixed_input,
            accelerator: self.config.model.accelerator,
            network,
        };
        let backend: Arc<dyn ModelBackend> = Arc::from(load_backend(self.config.model.backend, &bytes, options)?);
        tracing::debug!(backend = backend.name(), %source, "loaded inference backend");
        Ok(backend)
    }

    fn model_provider(&self) -> Result<Arc<dyn ModelProvider>> {
        self.models
            .lock()
            .map_err(|_| OcrError::model("model provider lock was poisoned"))?
            .clone()
            .ok_or_else(|| OcrError::model("model provider was released before initialization completed"))
    }

    fn release_provider_if_warmed(&self) -> Result<()> {
        if self.detector_cache.get().is_none() || self.recognizer_cache.get().is_none() {
            return Ok(());
        }
        let mut models = self
            .models
            .lock()
            .map_err(|_| OcrError::model("model provider lock was poisoned while releasing initialized models"))?;
        *models = None;
        Ok(())
    }

    /// The fixed square CRAFT canvas for the tract backend, or `None` for dynamic shapes.
    ///
    /// tract cannot shape-infer CRAFT under dynamic H/W, so on that backend detection
    /// pads every image to a fixed `canvas × canvas` square (rounded up to the CRAFT
    /// alignment) and the model is pinned to the matching shape (see ADR 0027). ort
    /// handles dynamic shapes and uses `None`.
    fn detector_fixed_canvas(&self) -> Option<u32> {
        (self.config.model.backend == Backend::Tract)
            .then(|| self.config.detection.canvas_size.next_multiple_of(DETECT_ALIGN))
    }

    /// The CRAFT detector backend, loaded once and cached for later calls.
    fn detector_backend(&self) -> Result<Arc<dyn ModelBackend>> {
        let backend = self.detector_cache.get_or_try_init(|| {
            let fixed_shape = self
                .detector_fixed_canvas()
                .map(|canvas| [1, 3, canvas as usize, canvas as usize]);
            self.load(
                self.model_provider()?.detector()?,
                NetworkKind::Detector,
                fixed_shape.as_ref().map(|shape| shape.as_slice()),
            )
        })?;
        self.release_provider_if_warmed()?;
        Ok(backend.clone())
    }

    /// The recognizer for the configured language set, loaded once and cached for later calls.
    fn recognizer(&self) -> Result<Arc<CrnnRecognizer>> {
        let recognizer = self.recognizer_cache.get_or_try_init(|| {
            let language = self.language()?;
            Ok::<Arc<CrnnRecognizer>, OcrError>(Arc::new(CrnnRecognizer::new(
                self.load(
                    self.model_provider()?.recognizer(language)?,
                    NetworkKind::Recognizer,
                    None,
                )?,
                Charset::for_language(language),
                self.config.recognition.clone(),
            )))
        })?;
        self.release_provider_if_warmed()?;
        Ok(recognizer.clone())
    }

    /// Detect candidate text regions with the CRAFT detector.
    fn detect_regions(&self, image: &Image) -> Result<DetectedRegions> {
        let detector = CraftDetector::new(
            self.detector_backend()?,
            self.config.detection.clone(),
            self.detector_fixed_canvas(),
        );
        detector.detect(&DetectorInput { image })
    }

    /// Recognize the cropped regions with the CRNN + CTC recognizer.
    fn recognize_crops(&self, crops: &[RegionCrop]) -> Result<Vec<RecognizedText>> {
        self.recognizer()?.recognize(crops)
    }

    /// Resolve `image`'s whole-page rotation: `Deg0` when the orientation
    /// pre-pass is disabled (the default, and the common case — no probe passes
    /// run, so the disabled path costs nothing extra), otherwise the
    /// best-scoring rotation from [`orientation::select_rotation`], probed with
    /// the same CRAFT backend `detect_regions` uses.
    fn resolve_rotation(&self, image: &Image) -> Result<Rotation> {
        if !self.config.detection.detect_orientation {
            return Ok(Rotation::Deg0);
        }
        let backend = self.detector_backend()?;
        orientation::select_rotation(
            backend.as_ref(),
            image,
            self.config.detection.orientation_probe_canvas_size,
            self.config.detection.low_text,
            self.config.detection.link_threshold,
            self.config.detection.orientation_margin,
        )
    }

    /// Resolve the rotation and the single frame both detection and recognition
    /// run in for the rest of the call: `image` itself, unchanged, at `Deg0` (no
    /// copy); a rotated copy otherwise. Detection, grayscale conversion, and
    /// cropping must all run against this same returned frame — mixing frames
    /// between them is exactly the bug this rotation-at-the-engine design fixes
    /// (a crop taken from the original page while detection ran on a rotated
    /// copy still contains sideways glyphs). The caller maps its final output
    /// quads back to `image`'s original frame with [`orientation::unrotate_corners`].
    fn oriented_image<'a>(&self, image: &'a Image) -> Result<(Rotation, Cow<'a, Image>)> {
        let rotation = self.resolve_rotation(image)?;
        let working = match rotation {
            Rotation::Deg0 => Cow::Borrowed(image),
            other => Cow::Owned(orientation::rotate_image(image, other)?),
        };
        Ok((rotation, working))
    }
}

impl OcrEngine for SceptreEngine {
    fn warm_up(&self) -> Result<()> {
        self.detector_backend()?;
        self.recognizer()?;
        Ok(())
    }

    fn recognize(&self, image: &Image, _options: &ReadOptions) -> Result<OcrResult> {
        self.config.recognition.validate()?;
        let (rotation, working) = self.oriented_image(image)?;
        self.progress.on_stage(STAGE_DETECT);
        let regions = self.detect_regions(&working)?;
        if regions.regions.is_empty() {
            return Ok(OcrResult::default());
        }

        let grey = to_grayscale(&working)?;
        let crops = crop_regions(&grey, &regions);
        if crops.is_empty() {
            return Ok(OcrResult::default());
        }

        self.progress.on_stage(STAGE_RECOGNIZE);
        let texts = self.recognize_crops(&crops)?;
        Ok(build_result(
            &crops,
            texts,
            self.config.recognition.filter_ths,
            rotation,
            image.width(),
            image.height(),
        ))
    }

    fn detect(&self, image: &Image, _options: &ReadOptions) -> Result<Vec<Quad>> {
        let (rotation, working) = self.oriented_image(image)?;
        self.progress.on_stage(STAGE_DETECT);
        let regions = self.detect_regions(&working)?;
        Ok(regions
            .regions
            .iter()
            .map(|region| {
                let corners = orientation::unrotate_corners(region.corners, rotation, image.width(), image.height());
                corners_to_quad(&corners)
            })
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
/// Native targets crop regions on the reader's Rayon pool; browser WASM uses the
/// sequential iterator. Both preserve input order and produce identical crops.
fn crop_regions(grey: &GrayImage, regions: &DetectedRegions) -> Vec<RegionCrop> {
    #[cfg(not(target_arch = "wasm32"))]
    let regions = regions.regions.par_iter();
    #[cfg(target_arch = "wasm32")]
    let regions = regions.regions.iter();

    regions
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
/// text lines, carrying each surviving crop's source quad mapped from the working
/// (possibly rotated) frame back to the caller's original `original_width` ×
/// `original_height` frame via [`orientation::unrotate_corners`].
fn build_result(
    crops: &[RegionCrop],
    texts: Vec<RecognizedText>,
    filter_ths: f32,
    rotation: Rotation,
    original_width: u32,
    original_height: u32,
) -> OcrResult {
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
        .map(|(crop, text)| {
            let corners = orientation::unrotate_corners(crop.corners, rotation, original_width, original_height);
            TextLine {
                quad: corners_to_quad(&corners),
                text: text.text,
                confidence: text.confidence,
            }
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
    use std::sync::atomic::AtomicUsize;

    use ndarray::{ArrayD, IxDyn};

    use super::*;
    use crate::config::RecognitionConfig;
    use crate::detect::DetectedRegion;
    use crate::inference::Tensor;

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

        let result = build_result(
            &crops,
            texts,
            RecognitionConfig::default().filter_ths,
            Rotation::Deg0,
            0,
            0,
        );

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

        let result = build_result(&crops, texts, 0.5, Rotation::Deg0, 0, 0);

        assert_eq!(result.lines, Vec::<TextLine>::new());
    }

    #[test]
    fn should_keep_line_equal_to_filter_threshold() {
        let crops = vec![crop_with_corners([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]])];
        let texts = vec![text("equal", 0.5)];

        let result = build_result(&crops, texts, 0.5, Rotation::Deg0, 0, 0);

        assert_eq!(result.lines.len(), 1);
        assert_eq!(result.lines[0].text, "equal");
    }

    #[test]
    fn should_keep_line_above_filter_threshold() {
        let crops = vec![crop_with_corners([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]])];
        let texts = vec![text("above", 0.51)];

        let result = build_result(&crops, texts, 0.5, Rotation::Deg0, 0, 0);

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

    fn solid_image(width: u32, height: u32) -> Image {
        Image::from_rgb8(width, height, vec![128u8; (width * height * 3) as usize]).expect("valid rgb buffer")
    }

    /// A no-op progress sink for tests that need a concrete [`ProgressSink`]
    /// without caring about its notifications.
    struct SilentProgress;
    impl ProgressSink for SilentProgress {}

    /// A detector backend that, in order, answers the four orientation probes
    /// (`Rotation::ALL`) with `probe_outputs`, then every later call (the real
    /// detection pass, run once per `SceptreEngine::detect_regions` call) with
    /// `final_output`. Mirrors the orientation-probe fixture the detector-level
    /// tests used before the orientation decision moved to the engine.
    struct RotationAwareDetectorBackend {
        call: AtomicUsize,
        probe_outputs: [ArrayD<f32>; 4],
        final_output: ArrayD<f32>,
    }

    impl ModelBackend for RotationAwareDetectorBackend {
        fn name(&self) -> &str {
            "rotation-aware-detector"
        }

        fn run(&self, _input: Tensor) -> Result<Tensor> {
            let index = self.call.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(if index < self.probe_outputs.len() {
                self.probe_outputs[index].clone()
            } else {
                self.final_output.clone()
            })
        }
    }

    /// A channel-last `[1, H, W, 2]` heat-map with no activation anywhere.
    fn empty_probe_output() -> ArrayD<f32> {
        ArrayD::<f32>::zeros(IxDyn(&[1, 4, 4, 2]))
    }

    /// A channel-last `[1, H, W, 2]` heat-map saturated above every threshold.
    fn saturated_probe_output() -> ArrayD<f32> {
        ArrayD::<f32>::from_elem(IxDyn(&[1, 4, 4, 2]), 0.9)
    }

    /// A recognizer backend that records the shape of every CRNN input tensor it
    /// receives (`[B, 1, 64, W]`) and returns a fixed, low-confidence rank-3
    /// output so recognition always completes without a real model or charset.
    struct CapturingRecognizerBackend {
        shapes: Mutex<Vec<Vec<usize>>>,
        num_classes: usize,
    }

    impl ModelBackend for CapturingRecognizerBackend {
        fn name(&self) -> &str {
            "capturing-recognizer"
        }

        fn run(&self, input: Tensor) -> Result<Tensor> {
            self.shapes.lock().expect("shapes lock").push(input.shape().to_vec());
            let batch = input.shape()[0];
            Ok(ArrayD::<f32>::zeros(IxDyn(&[batch, 2, self.num_classes])))
        }
    }

    /// Regression coverage for the bug the rotate-once-in-the-engine design
    /// fixes: `SceptreEngine::recognize` must run detection *and* cropping in
    /// the same rotated working frame, not detect on a rotated copy while
    /// cropping from the caller's original (still-sideways) page.
    ///
    /// The detector backend picks `Rotation::Deg270` (only its probe is
    /// saturated) and reports one region from a real CRAFT heat-map blob run
    /// through the actual postprocess/group pipeline — this exercises the real
    /// `CraftDetector`, not a hand-rolled region. The recognizer backend just
    /// records the shape of every tensor it is asked to recognize.
    ///
    /// Before this design, `CraftDetector::detect` rotated the page internally
    /// and detected on it, but then re-derived the region's corners back into
    /// the *original* frame before returning, and the engine cropped from
    /// `to_grayscale(image)` — the original, still-sideways page. Because an
    /// axis-aligned box's width and height swap under a 90°/270° inverse
    /// rotation, that crop was a transposed (sideways) slice: tall and narrow
    /// instead of the wide, short shape a real horizontal text line has once
    /// the page is upright. This test's expected width asserts on the
    /// *upright* shape; run against the pre-fix design this recorded a
    /// transposed, narrower tensor instead (verified by hand against commit
    /// `bdf8533`'s `CraftDetector::detect` + `SceptreEngine::recognize`).
    #[test]
    fn should_recognize_a_rotated_pages_crop_from_the_rotated_frame_not_the_original() {
        let probe_outputs = [
            empty_probe_output(),
            empty_probe_output(),
            empty_probe_output(),
            saturated_probe_output(),
        ];
        // Channel-last [1, 40, 20, 2]: a region block at rows 17..23 (6 rows), cols ~keep
        // 2..18 (16 cols) -- a wide, short blob once scaled to image space, well ~keep
        // clear of every edge. ~keep
        let (height, width) = (40usize, 20usize);
        let mut final_output = ArrayD::<f32>::zeros(IxDyn(&[1, height, width, 2]));
        for row in 17..23 {
            for col in 2..18 {
                final_output[[0, row, col, 0]] = 1.0;
            }
        }
        let detector_backend = Arc::new(RotationAwareDetectorBackend {
            call: AtomicUsize::new(0),
            probe_outputs,
            final_output,
        });
        let recognizer_backend = Arc::new(CapturingRecognizerBackend {
            shapes: Mutex::new(Vec::new()),
            num_classes: Charset::for_language(Language::English).num_classes(),
        });

        let mut config = OcrConfig::default();
        config.detection.detect_orientation = true;
        config.detection.orientation_probe_canvas_size = 8;
        config.detection.orientation_margin = 0.05;
        config.detection.min_size = 0;

        let engine = SceptreEngine {
            config,
            models: Mutex::new(None),
            progress: Arc::new(SilentProgress),
            detector_cache: OnceCell::new(),
            recognizer_cache: OnceCell::new(),
        };
        engine
            .detector_cache
            .set(detector_backend.clone() as Arc<dyn ModelBackend>)
            .ok()
            .expect("detector cache starts empty");
        engine
            .recognizer_cache
            .set(Arc::new(CrnnRecognizer::new(
                recognizer_backend.clone() as Arc<dyn ModelBackend>,
                Charset::for_language(Language::English),
                RecognitionConfig::default(),
            )))
            .ok()
            .expect("recognizer cache starts empty");

        // Non-square, comfortably larger than the heat-map's own coordinate range. ~keep
        let image = solid_image(200, 100);
        let _ = engine
            .recognize(&image, &ReadOptions::default())
            .expect("recognize succeeds against the synthetic backends");

        assert_eq!(
            detector_backend.call.load(std::sync::atomic::Ordering::SeqCst),
            5,
            "four orientation probes plus one real detection pass"
        );
        let shapes = recognizer_backend.shapes.lock().expect("shapes lock");
        // A zero-confidence first pass triggers EasyOCR's contrast-adjusted retry
        // (`contrast_ths`), so the same crop may be recognized twice; both passes
        // preprocess the identical crop dimensions, so any recorded shape does. ~keep
        assert!(!shapes.is_empty(), "at least one crop reached the recognizer");
        // `[B, 1, 64, W]`: recognizer input is always resized to height 64, so `W` is
        // the only dimension that still carries the crop's aspect ratio -- and thus
        // whether the crop was upright (wide) or sideways (narrow). ~keep
        let recognized_width = shapes[0][3];
        assert!(
            recognized_width > 64,
            "a wide, short text-line crop resizes to a width well over its height-64 \
             frame; got {recognized_width}, which is what a transposed (sideways) crop \
             from the wrong frame would produce"
        );
    }
}
