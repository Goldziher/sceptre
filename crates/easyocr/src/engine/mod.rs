//! The [`Reader`] handle and its builder.
//!
//! `Reader` is a cheap, `Arc`-backed, cloneable handle over the loaded config and
//! injectable [`seams`] (model provider, progress sink). It is built through
//! [`ReaderBuilder`], mirroring xberg's engine/seams pattern: every extension
//! point is a trait with an in-crate default, so callers can inject alternatives
//! without touching the default path.

pub mod seams;

use std::path::Path;
use std::sync::Arc;

use crate::config::{OcrConfig, init_thread_pools, resolve_thread_budget};
use crate::error::Result;
use crate::types::OcrResult;

use seams::{DefaultModelProvider, ModelProvider, NoopProgress, ProgressSink};

/// Per-call options for a [`Reader::readtext`] invocation.
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

// Injected seams are held for the pipeline stages that consume them. ~keep
#[allow(dead_code)]
struct Inner {
    config: OcrConfig,
    models: Arc<dyn ModelProvider>,
    progress: Arc<dyn ProgressSink>,
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

    /// Run the full detect → recognize pipeline over an image path.
    pub fn readtext(&self, _image: &Path, _options: &ReadOptions) -> Result<OcrResult> {
        todo!("run detection then recognition through the injected seams")
    }
}

/// Builder for [`Reader`], filling injectable seams with in-crate defaults.
#[derive(Default)]
pub struct ReaderBuilder {
    config: OcrConfig,
    models: Option<Arc<dyn ModelProvider>>,
    progress: Option<Arc<dyn ProgressSink>>,
}

impl ReaderBuilder {
    /// Set the OCR configuration.
    pub fn config(mut self, config: OcrConfig) -> Self {
        self.config = config;
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
    pub fn build(self) -> Result<Reader> {
        let budget = resolve_thread_budget(Some(&self.config.concurrency));
        init_thread_pools(budget);

        let models = match self.models {
            Some(m) => m,
            None => Arc::new(DefaultModelProvider::from_config(&self.config)?),
        };
        let progress = self.progress.unwrap_or_else(|| Arc::new(NoopProgress));

        Ok(Reader {
            inner: Arc::new(Inner {
                config: self.config,
                models,
                progress,
            }),
        })
    }
}
