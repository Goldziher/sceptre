//! Internal text-recognition seam and its stage DTOs.
//!
//! The recognizer consumes already-cropped, grayscale regions and yields decoded
//! text with a confidence. Crops carry their source corners so the engine can map
//! each result back to a public [`Quad`](crate::types::Quad).

use std::sync::Arc;

use ndarray::Axis;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::config::{Decoder, RecognitionConfig};
use crate::error::{OcrError, Result};
use crate::inference::ModelBackend;
use crate::types::QUAD_CORNERS as REGION_CORNERS;

use super::charset::Charset;

/// Internal seam: turns cropped regions into recognized text.
pub(crate) trait TextRecognizer: Send + Sync {
    /// Recognize text for each crop, preserving order.
    fn recognize(&self, crops: &[RegionCrop]) -> Result<Vec<RecognizedText>>;
}

/// Internal DTO: one cropped region ready for the recognizer (grayscale, owned).
pub(crate) struct RegionCrop {
    /// Crop width in pixels.
    pub width: u32,
    /// Crop height in pixels.
    pub height: u32,
    /// Row-major grayscale pixels, length `width * height`.
    pub gray: Vec<u8>,
    /// Source location `[x, y]` corners, to map back to a public quad.
    pub corners: [[f32; 2]; REGION_CORNERS],
}

/// Internal DTO: recognizer output for one crop.
pub(crate) struct RecognizedText {
    /// The decoded text.
    pub text: String,
    /// Recognition confidence in `[0.0, 1.0]`.
    pub confidence: f32,
}

/// The CRNN + CTC text recognizer.
///
/// Runs a first greedy pass over every crop, then a contrast-adjusted second pass
/// over the low-confidence crops (EasyOCR's `contrast_ths` retry), keeping whichever
/// pass scored higher per crop.
pub(crate) struct CrnnRecognizer {
    backend: Arc<dyn ModelBackend>,
    charset: Charset,
    config: RecognitionConfig,
    ignore_mask: super::ctc::IgnoreMask,
}

impl CrnnRecognizer {
    /// Construct a CRNN recognizer from a loaded backend, charset, and config.
    pub(crate) fn new(backend: Arc<dyn ModelBackend>, charset: Charset, config: RecognitionConfig) -> Self {
        let ignore = build_ignore(&charset, &config);
        let ignore_mask = super::ctc::IgnoreMask::new(charset.num_classes(), &ignore);
        Self {
            backend,
            charset,
            config,
            ignore_mask,
        }
    }

    /// Recognize a batch of crops in a single greedy pass, preserving order.
    ///
    /// Chunks `crops` by `batch_size` (min 1), preprocesses each chunk into a
    /// recognizer tensor, runs the CRNN, then CTC-decodes each row.
    fn run_pass(&self, crops: &[RegionCrop]) -> Result<Vec<RecognizedText>> {
        let batch_size = self.config.batch_size.max(1);
        let mut results = Vec::with_capacity(crops.len());
        for chunk in crops.chunks(batch_size) {
            let tensor = super::preprocess::prepare_batch(chunk)?;
            let logits = super::crnn::run_crnn(self.backend.as_ref(), tensor)?;
            if chunk.len() == 1 {
                results.push(self.decode_row(logits.index_axis(Axis(0), 0)));
                continue;
            }
            // Native targets use the reader's Rayon pool; browser WASM uses the sequential range. ~keep
            // Both preserve input order and borrow each row view. ~keep
            #[cfg(not(target_arch = "wasm32"))]
            let rows = (0..chunk.len()).into_par_iter();
            #[cfg(target_arch = "wasm32")]
            let rows = 0..chunk.len();
            let decoded: Vec<RecognizedText> = rows
                .map(|row| self.decode_row(logits.index_axis(Axis(0), row)))
                .collect();
            results.extend(decoded);
        }
        Ok(results)
    }

    /// Decode one crop's logits with the configured decoder. Confidence is identical
    /// across decoders (EasyOCR always computes it from the greedy argmax); only the
    /// text-production strategy changes.
    fn decode_row(&self, logits: ndarray::ArrayView2<f32>) -> RecognizedText {
        match self.config.decoder {
            Decoder::Greedy => super::ctc::decode_greedy_with_mask(logits, &self.charset, &self.ignore_mask),
            Decoder::BeamSearch => {
                super::beam::decode_beam_search(logits, &self.charset, &self.ignore_mask, self.config.beam_width)
            }
            Decoder::WordBeamSearch => {
                unreachable!("word-beam-search is rejected in `recognize` before any pass runs")
            }
        }
    }

    /// Re-run the low-confidence crops with contrast adjustment, replacing a result
    /// only when the adjusted pass scores strictly higher.
    fn apply_second_pass(&self, crops: &[RegionCrop], results: &mut [RecognizedText]) -> Result<()> {
        let indices: Vec<usize> = results
            .iter()
            .enumerate()
            .filter(|(_, result)| result.confidence < self.config.contrast_ths)
            .map(|(index, _)| index)
            .collect();
        if indices.is_empty() {
            return Ok(());
        }
        let adjusted: Vec<RegionCrop> = indices.iter().map(|&index| self.adjust_crop(&crops[index])).collect();
        let second = self.run_pass(&adjusted)?;
        for (candidate, &index) in second.into_iter().zip(indices.iter()) {
            // Ties go to the second (contrast-adjusted) pass, matching EasyOCR's ~keep
            // `if pred1 > pred2 { pred1 } else { pred2 }`. ~keep
            if candidate.confidence >= results[index].confidence {
                results[index] = candidate;
            }
        }
        Ok(())
    }

    /// Build a contrast-adjusted copy of one crop for the second pass.
    fn adjust_crop(&self, crop: &RegionCrop) -> RegionCrop {
        RegionCrop {
            width: crop.width,
            height: crop.height,
            gray: super::contrast::adjust_contrast_grey(&crop.gray, self.config.adjust_contrast),
            corners: crop.corners,
        }
    }
}

/// CTC class indices to suppress, derived from the allow/block lists.
///
/// The lists are mutually exclusive, matching EasyOCR (`if allowlist ... elif
/// blocklist ...`): a non-empty allowlist suppresses every class whose character is
/// not listed and the blocklist is then ignored; otherwise the blocklist suppresses
/// every class whose character is listed. EasyOCR's third branch — restricting to
/// the requested language's own `lang_char` when neither list is set — needs the
/// per-language character subset and is deferred; a single-language gen2 model's
/// charset already is its language, so the default suppresses nothing.
fn build_ignore(charset: &Charset, config: &RecognitionConfig) -> Vec<usize> {
    if !config.allowlist.is_empty() {
        let allowed: Vec<usize> = config.allowlist.chars().filter_map(|ch| charset.class_of(ch)).collect();
        (1..charset.num_classes())
            .filter(|class| !allowed.contains(class))
            .collect()
    } else {
        config.blocklist.chars().filter_map(|ch| charset.class_of(ch)).collect()
    }
}

impl TextRecognizer for CrnnRecognizer {
    fn recognize(&self, crops: &[RegionCrop]) -> Result<Vec<RecognizedText>> {
        if self.config.decoder == Decoder::WordBeamSearch {
            return Err(OcrError::config(
                "word-beam-search CTC decoding is not implemented; set decoder = \"greedy\" or \"beamsearch\"",
            ));
        }
        let mut results = self.run_pass(crops)?;
        self.apply_second_pass(crops, &mut results)?;
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Language;
    use crate::inference::Tensor;
    use ndarray::{ArrayD, IxDyn};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A backend that returns preset logits, advancing one script entry per call so
    /// the first and second recognition passes can return different outputs. The last
    /// entry repeats once the script is exhausted.
    struct ScriptedBackend {
        outputs: Vec<ArrayD<f32>>,
        calls: AtomicUsize,
    }

    impl ScriptedBackend {
        fn new(outputs: Vec<ArrayD<f32>>) -> Self {
            Self {
                outputs,
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl ModelBackend for ScriptedBackend {
        fn name(&self) -> &str {
            "scripted"
        }

        fn run(&self, _input: Tensor) -> Result<Tensor> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            let index = call.min(self.outputs.len() - 1);
            Ok(self.outputs[index].clone())
        }
    }

    /// One-timestep logits `[1, 1, 3]` over classes `[blank, '0', '1']`.
    fn logits(values: [f32; 3]) -> ArrayD<f32> {
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 3]), values.to_vec()).expect("valid logits shape")
    }

    fn sample_crop() -> RegionCrop {
        RegionCrop {
            width: 4,
            height: 2,
            gray: vec![120u8; 8],
            corners: [[0.0, 0.0]; 4],
        }
    }

    fn english_recognizer(backend: Arc<dyn ModelBackend>, config: RecognitionConfig) -> CrnnRecognizer {
        CrnnRecognizer::new(backend, Charset::for_language(Language::English), config)
    }

    #[test]
    fn should_decode_single_crop_to_expected_text() {
        // class 1 ('0') dominates, so the crop decodes to "0". ~keep
        let backend = Arc::new(ScriptedBackend::new(vec![logits([0.0, 5.0, 0.0])]));
        let recognizer = english_recognizer(backend, RecognitionConfig::default());

        let results = recognizer.recognize(&[sample_crop()]).expect("recognition succeeds");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].text, "0");
        assert!(results[0].confidence > 0.0, "a confident crop scores above zero");
    }

    #[test]
    fn should_decode_multi_region_batch_in_input_order() {
        // One chunk of three regions, each `[T=2, C=3]` over [blank, '0', '1'] with a ~keep
        // distinct argmax path: region 0 -> "0", region 1 -> "1", region 2 -> "01". A ~keep
        // parallel, order-preserving decode must return them in that exact order. ~keep
        let rows = [
            [0.0f32, 5.0, 0.0, 0.0, 5.0, 0.0], // region 0: "0" ~keep
            [0.0, 0.0, 5.0, 0.0, 0.0, 5.0],    // region 1: "1" ~keep
            [0.0, 5.0, 0.0, 0.0, 0.0, 5.0],    // region 2: "01" ~keep
        ];
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        let batch_logits = ArrayD::from_shape_vec(IxDyn(&[3, 2, 3]), flat).expect("valid batch logits shape");
        let backend = Arc::new(ScriptedBackend::new(vec![batch_logits]));
        // batch_size 3 keeps all regions in one chunk; contrast_ths 0.0 disables the retry. ~keep
        let config = RecognitionConfig {
            batch_size: 3,
            contrast_ths: 0.0,
            ..RecognitionConfig::default()
        };
        let recognizer = english_recognizer(backend, config);

        let crops = [sample_crop(), sample_crop(), sample_crop()];
        let results = recognizer.recognize(&crops).expect("recognition succeeds");

        let decoded: Vec<&str> = results.iter().map(|result| result.text.as_str()).collect();
        assert_eq!(decoded, vec!["0", "1", "01"], "results stay in input order");
    }

    #[test]
    fn should_replace_low_confidence_result_when_second_pass_scores_higher() {
        // First pass: softmax([0.2, 0.5, 0.3]) -> class 1 ('0') at 0.5, custom_mean 0.25. ~keep
        // Second pass: softmax([0.05, 0.05, 0.9]) -> class 2 ('1') at 0.9, custom_mean 0.81. ~keep
        // contrast_ths 0.9 forces the retry; 0.81 > 0.25 replaces the result with "1". ~keep
        let first = logits([(0.2f32).ln(), (0.5f32).ln(), (0.3f32).ln()]);
        let second = logits([(0.05f32).ln(), (0.05f32).ln(), (0.9f32).ln()]);
        let backend = Arc::new(ScriptedBackend::new(vec![first, second]));
        let config = RecognitionConfig {
            contrast_ths: 0.9,
            ..RecognitionConfig::default()
        };
        let recognizer = english_recognizer(backend, config);

        let results = recognizer.recognize(&[sample_crop()]).expect("recognition succeeds");

        assert_eq!(
            results[0].text, "1",
            "the higher-confidence second pass replaces the result"
        );
        assert!(results[0].confidence > 0.5, "confidence reflects the second pass");
    }

    #[test]
    fn should_keep_first_result_when_second_pass_scores_lower() {
        // First pass strong on class 2 ('1') at 0.9 (custom_mean 0.81); the retry is ~keep
        // still triggered by contrast_ths 0.95 but its weaker class 1 result loses. ~keep
        let first = logits([(0.05f32).ln(), (0.05f32).ln(), (0.9f32).ln()]);
        let second = logits([(0.2f32).ln(), (0.5f32).ln(), (0.3f32).ln()]);
        let backend = Arc::new(ScriptedBackend::new(vec![first, second]));
        let config = RecognitionConfig {
            contrast_ths: 0.95,
            ..RecognitionConfig::default()
        };
        let recognizer = english_recognizer(backend, config);

        let results = recognizer.recognize(&[sample_crop()]).expect("recognition succeeds");

        assert_eq!(results[0].text, "1", "the stronger first pass is retained");
    }

    #[test]
    fn should_ignore_classes_outside_the_allowlist() {
        let charset = Charset::for_language(Language::English);
        let config = RecognitionConfig {
            allowlist: "01".to_string(),
            ..RecognitionConfig::default()
        };
        let ignore = build_ignore(&charset, &config);
        // Classes 1 ('0') and 2 ('1') are allowed; class 3 ('2') is suppressed. ~keep
        assert!(!ignore.contains(&1));
        assert!(!ignore.contains(&2));
        assert!(ignore.contains(&3));
    }

    #[test]
    fn should_ignore_blocklisted_classes() {
        let charset = Charset::for_language(Language::English);
        let config = RecognitionConfig {
            blocklist: "0".to_string(),
            ..RecognitionConfig::default()
        };
        let ignore = build_ignore(&charset, &config);
        // class_of('0') == 1, and nothing else is suppressed. ~keep
        assert_eq!(ignore, vec![1]);
    }

    #[test]
    fn should_ignore_blocklist_only_when_allowlist_is_empty() {
        let charset = Charset::for_language(Language::English);
        // Both set: EasyOCR uses the allowlist and ignores the blocklist entirely. ~keep
        let config = RecognitionConfig {
            allowlist: "01".to_string(),
            blocklist: "01".to_string(),
            ..RecognitionConfig::default()
        };
        let ignore = build_ignore(&charset, &config);
        // Allowlist wins: classes 1 ('0') and 2 ('1') stay allowed despite the blocklist. ~keep
        assert!(!ignore.contains(&1));
        assert!(!ignore.contains(&2));
    }

    #[test]
    fn should_apply_precomputed_allowlist_during_recognition() {
        // Class 1 ('0') wins the raw logits, but the cached allowlist mask suppresses ~keep
        // it and leaves class 2 ('1') as the decoded output. ~keep
        let backend = Arc::new(ScriptedBackend::new(vec![logits([0.0, 5.0, 4.0])]));
        let config = RecognitionConfig {
            allowlist: "1".to_string(),
            contrast_ths: 0.0,
            ..RecognitionConfig::default()
        };
        let recognizer = english_recognizer(backend, config);

        let results = recognizer.recognize(&[sample_crop()]).expect("recognition succeeds");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].text, "1");
    }

    #[test]
    fn should_reject_word_beam_search_decoder() {
        let backend = Arc::new(ScriptedBackend::new(vec![logits([0.0, 5.0, 0.0])]));
        let config = RecognitionConfig {
            decoder: crate::config::Decoder::WordBeamSearch,
            ..RecognitionConfig::default()
        };
        let recognizer = english_recognizer(backend, config);
        let result = recognizer.recognize(&[sample_crop()]);
        assert!(
            matches!(result, Err(OcrError::Config { .. })),
            "word-beam-search decoding must be rejected with a config error"
        );
    }

    #[test]
    fn should_decode_via_beam_search_when_configured() {
        // Same fixture as should_decode_single_crop_to_expected_text, but routed ~keep
        // through the beam-search decoder: an unambiguous single-path input decodes ~keep
        // the same way regardless of decoder. ~keep
        let backend = Arc::new(ScriptedBackend::new(vec![logits([0.0, 5.0, 0.0])]));
        let config = RecognitionConfig {
            decoder: crate::config::Decoder::BeamSearch,
            contrast_ths: 0.0,
            ..RecognitionConfig::default()
        };
        let recognizer = english_recognizer(backend, config);

        let results = recognizer
            .recognize(&[sample_crop()])
            .expect("beam search decoding succeeds");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].text, "0");
    }

    #[test]
    fn should_default_to_greedy_decoding() {
        assert_eq!(RecognitionConfig::default().decoder, crate::config::Decoder::Greedy);
    }

    /// End-to-end recognition over a real gen2 recognizer ONNX model.
    ///
    /// Ignored by default: it links and initializes the ONNX Runtime native library
    /// and needs a model file. Point `EASYOCR_TEST_RECOG_ONNX` at a gen2 recognizer
    /// (`english_g2`); the recognizer runs over a small synthetic crop and must
    /// return without error.
    #[cfg(feature = "ort")]
    #[test]
    #[ignore = "requires the ONNX Runtime native library and a recognizer model file"]
    fn recognize_over_real_recognizer_model() {
        let model_path = std::env::var("EASYOCR_TEST_RECOG_ONNX")
            .expect("set EASYOCR_TEST_RECOG_ONNX to a recognizer ONNX model path");
        let model_bytes = std::fs::read(&model_path).expect("read the model file");
        let options = crate::inference::BackendOptions {
            threads: 1,
            ..Default::default()
        };
        let backend = crate::inference::load_backend(crate::config::Backend::Ort, &model_bytes, options)
            .expect("load the recognizer ONNX model");
        let recognizer = CrnnRecognizer::new(
            Arc::from(backend),
            Charset::for_language(Language::English),
            RecognitionConfig::default(),
        );

        let crop = RegionCrop {
            width: 32,
            height: 16,
            gray: vec![200u8; 32 * 16],
            corners: [[0.0, 0.0]; 4],
        };
        let result = recognizer.recognize(&[crop]);

        assert!(result.is_ok(), "recognition over the real model must succeed");
    }
}
