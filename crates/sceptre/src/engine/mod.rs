//! The [`Reader`] handle, its builder, and the [`OcrEngine`] extension seam.
//!
//! `Reader` is a cheap, `Arc`-backed, cloneable handle over an injected
//! [`OcrEngine`] and the loaded config. It is built through [`ReaderBuilder`],
//! mirroring xberg's engine/seams pattern: every extension point is a trait with
//! an in-crate default, so callers can inject alternatives without touching the
//! default path. The default engine is the internal [`SceptreEngine`].

pub(crate) mod seams;

mod fallback;
mod ocr_engine;
mod sceptre_engine;

use std::path::Path;
use std::sync::Arc;

use crate::config::{OcrConfig, init_thread_pools, resolve_thread_budget};
use crate::error::Result;
use crate::types::{Image, OcrResult};

use sceptre_engine::SceptreEngine;
use seams::{DefaultModelProvider, ModelProvider, NoopProgress, ProgressSink};

pub use fallback::FallbackEngine;
pub use ocr_engine::OcrEngine;

/// Per-call options for a [`Reader::readtext`] or [`Reader::recognize`] invocation.
#[derive(Debug, Clone)]
pub struct ReadOptions {
    /// When `false`, only text is returned (locations/confidence omitted downstream).
    pub detail: bool,
}

impl Default for ReadOptions {
    fn default() -> Self {
        Self { detail: true }
    }
}

/// A ready-to-use OCR reader.
#[derive(Clone)]
pub struct Reader {
    inner: Arc<Inner>,
}

struct Inner {
    config: OcrConfig,
    engine: Arc<dyn OcrEngine>,
}

impl Reader {
    /// Start building a reader.
    pub fn builder() -> ReaderBuilder {
        ReaderBuilder::default()
    }

    /// The effective configuration.
    pub fn config(&self) -> &OcrConfig {
        &self.inner.config
    }

    /// Decode an image at `image` and run the engine over it.
    pub fn readtext(&self, image: &Path, options: &ReadOptions) -> Result<OcrResult> {
        let decoded = Image::from_path(image)?;
        self.recognize(&decoded, options)
    }

    /// Run the engine directly on an already-decoded image.
    pub fn recognize(&self, image: &Image, options: &ReadOptions) -> Result<OcrResult> {
        self.inner.engine.recognize(image, options)
    }
}

/// Builder for [`Reader`], filling injectable seams with in-crate defaults.
#[derive(Default)]
pub struct ReaderBuilder {
    config: OcrConfig,
    engine: Option<Arc<dyn OcrEngine>>,
    models: Option<Arc<dyn ModelProvider>>,
    progress: Option<Arc<dyn ProgressSink>>,
}

impl ReaderBuilder {
    /// Set the OCR configuration.
    pub fn config(mut self, config: OcrConfig) -> Self {
        self.config = config;
        self
    }

    /// Inject a custom engine (default: the internal `SceptreEngine`).
    pub fn engine(mut self, engine: Arc<dyn OcrEngine>) -> Self {
        self.engine = Some(engine);
        self
    }

    /// Inject a custom model provider (default: [`DefaultModelProvider`]).
    pub fn model_provider(mut self, provider: Arc<dyn ModelProvider>) -> Self {
        self.models = Some(provider);
        self
    }

    /// Inject a progress sink (default: [`NoopProgress`]).
    pub fn progress(mut self, progress: Arc<dyn ProgressSink>) -> Self {
        self.progress = Some(progress);
        self
    }

    /// Finalize the reader, initializing the shared thread budget.
    ///
    /// If an engine was injected it is used as-is; otherwise the default
    /// [`SceptreEngine`] is constructed from the config and the resolved model
    /// provider and progress sink.
    pub fn build(self) -> Result<Reader> {
        let budget = resolve_thread_budget(Some(&self.config.concurrency));
        init_thread_pools(budget);

        let engine: Arc<dyn OcrEngine> = match self.engine {
            Some(engine) => engine,
            None => {
                let models = match self.models {
                    Some(models) => models,
                    None => Arc::new(DefaultModelProvider::from_config(&self.config)?),
                };
                let progress = self.progress.unwrap_or_else(|| Arc::new(NoopProgress));
                Arc::new(SceptreEngine::new(self.config.clone(), models, progress))
            }
        };

        Ok(Reader {
            inner: Arc::new(Inner {
                config: self.config,
                engine,
            }),
        })
    }
}
