//! The default [`OcrEngine`] implementation: [`EasyOcrEngine`].

use std::sync::Arc;

use crate::config::OcrConfig;
use crate::detect::{CraftDetector, TextDetector};
use crate::error::Result;
use crate::recognize::{CrnnRecognizer, TextRecognizer};
use crate::types::{Image, OcrResult};

use super::seams::{ModelProvider, ProgressSink};
use super::{OcrEngine, ReadOptions};

/// The default engine: CRAFT detection followed by CRNN + CTC recognition.
///
/// Holds the injectable seams and configuration consumed by the detect → crop →
/// recognize pipeline.
// Fields hold the resolved seams and configuration for the detect → crop → recognize pipeline. ~keep
#[allow(dead_code)]
pub(crate) struct EasyOcrEngine {
    config: OcrConfig,
    models: Arc<dyn ModelProvider>,
    progress: Arc<dyn ProgressSink>,
    detector: Box<dyn TextDetector>,
    recognizer: Box<dyn TextRecognizer>,
}

impl EasyOcrEngine {
    /// Build the default engine (CRAFT detector + CRNN recognizer) from resolved seams.
    pub(crate) fn new(config: OcrConfig, models: Arc<dyn ModelProvider>, progress: Arc<dyn ProgressSink>) -> Self {
        Self {
            config,
            models,
            progress,
            detector: Box::new(CraftDetector::new()),
            recognizer: Box::new(CrnnRecognizer::new()),
        }
    }
}

impl OcrEngine for EasyOcrEngine {
    fn recognize(&self, _image: &Image, _options: &ReadOptions) -> Result<OcrResult> {
        todo!("detect regions, crop, recognize, then map internal DTOs to OcrResult")
    }
}
