//! The default [`OcrEngine`] implementation: [`SceptreEngine`].

use std::sync::Arc;

use crate::config::OcrConfig;
use crate::error::Result;
use crate::types::{Image, OcrResult};

use super::seams::{ModelProvider, ProgressSink};
use super::{OcrEngine, ReadOptions};

/// The default engine: CRAFT detection followed by CRNN + CTC recognition.
///
/// Holds the injectable seams and configuration consumed by the detect → crop →
/// recognize pipeline. The detector and recognizer are built lazily during
/// [`recognize`](SceptreEngine::recognize) because loading their model backends is
/// fallible I/O and cannot happen in an infallible constructor.
// Seams and config feed the detect → crop → recognize pipeline, whose body is still a todo. ~keep
#[allow(dead_code)]
pub(crate) struct SceptreEngine {
    config: OcrConfig,
    models: Arc<dyn ModelProvider>,
    progress: Arc<dyn ProgressSink>,
}

impl SceptreEngine {
    /// Build the default engine from resolved seams and configuration.
    pub(crate) fn new(config: OcrConfig, models: Arc<dyn ModelProvider>, progress: Arc<dyn ProgressSink>) -> Self {
        Self {
            config,
            models,
            progress,
        }
    }
}

impl OcrEngine for SceptreEngine {
    fn recognize(&self, _image: &Image, _options: &ReadOptions) -> Result<OcrResult> {
        todo!(
            "build the inference backends from models, construct CraftDetector and CrnnRecognizer, \
             detect regions, crop each region, recognize with CRNN + CTC, then map internal DTOs \
             to OcrResult"
        )
    }
}
