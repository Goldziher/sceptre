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

use rayon::ThreadPool;

use crate::config::{OcrConfig, build_thread_pool, resolve_thread_budget};
use crate::error::Result;
use crate::types::{Image, OcrResult, Quad, TextLine};

use sceptre_engine::SceptreEngine;
use seams::{DefaultModelProvider, ModelProvider, NoopProgress, ProgressSink};

pub use fallback::FallbackEngine;
pub use ocr_engine::OcrEngine;

/// Per-call options for a [`Reader::readtext`] or [`Reader::recognize`] invocation.
#[derive(Debug, Clone)]
pub struct ReadOptions {
    /// Presentation hint consumed by callers (e.g. the CLI) when formatting output:
    /// when `false`, only the text of each line is meant to be shown, dropping its
    /// quad and confidence. The engine always computes full detail; this flag does
    /// not change the [`OcrResult`] it returns.
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
    thread_pool: ThreadPool,
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
        self.inner
            .thread_pool
            .install(|| self.inner.engine.recognize(image, options))
    }

    /// Detect text regions in an already-decoded image, returning their quads.
    pub fn detect(&self, image: &Image, options: &ReadOptions) -> Result<Vec<Quad>> {
        self.inner
            .thread_pool
            .install(|| self.inner.engine.detect(image, options))
    }

    /// Recognize an already-decoded, pre-cropped single line image.
    pub fn recognize_line(&self, image: &Image, options: &ReadOptions) -> Result<TextLine> {
        self.inner
            .thread_pool
            .install(|| self.inner.engine.recognize_line(image, options))
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

    /// Finalize the reader, initializing its private worker pool.
    ///
    /// If an engine was injected it is used as-is; otherwise the default
    /// [`SceptreEngine`] is constructed from the config and the resolved model
    /// provider and progress sink.
    pub fn build(self) -> Result<Reader> {
        let budget = resolve_thread_budget(Some(&self.config.concurrency));
        let thread_pool = build_thread_pool(budget)?;

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
                thread_pool,
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    struct ThreadCountEngine {
        observed_threads: Arc<AtomicUsize>,
    }

    impl OcrEngine for ThreadCountEngine {
        fn recognize(&self, _image: &Image, _options: &ReadOptions) -> Result<OcrResult> {
            self.observed_threads
                .store(rayon::current_num_threads(), Ordering::SeqCst);
            Ok(OcrResult::default())
        }
    }

    fn reader_with_threads(max_threads: usize, observed_threads: Arc<AtomicUsize>) -> Reader {
        let mut config = OcrConfig::default();
        config.concurrency.max_threads = Some(max_threads);
        Reader::builder()
            .config(config)
            .engine(Arc::new(ThreadCountEngine { observed_threads }))
            .build()
            .expect("the reader should build")
    }

    fn assert_entry_points_use_budget(reader: &Reader, observed_threads: &AtomicUsize, expected: usize) {
        let image = Image::from_rgb8(1, 1, vec![0, 0, 0]).expect("the image should be valid");
        let options = ReadOptions::default();

        reader.recognize(&image, &options).expect("recognize should succeed");
        assert_eq!(observed_threads.load(Ordering::SeqCst), expected);

        observed_threads.store(0, Ordering::SeqCst);
        reader.detect(&image, &options).expect("detect should succeed");
        assert_eq!(observed_threads.load(Ordering::SeqCst), expected);

        observed_threads.store(0, Ordering::SeqCst);
        reader
            .recognize_line(&image, &options)
            .expect("recognize_line should succeed");
        assert_eq!(observed_threads.load(Ordering::SeqCst), expected);
    }

    #[test]
    fn should_isolate_rayon_thread_budgets_between_readers() {
        let first_count = Arc::new(AtomicUsize::new(0));
        let second_count = Arc::new(AtomicUsize::new(0));
        let first = reader_with_threads(1, first_count.clone());
        let second = reader_with_threads(3, second_count.clone());

        assert_entry_points_use_budget(&first, &first_count, 1);
        assert_entry_points_use_budget(&second, &second_count, 3);
    }
}
