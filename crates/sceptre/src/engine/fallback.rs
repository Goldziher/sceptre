//! The [`FallbackEngine`] combinator.

use std::sync::Arc;

use crate::error::{OcrError, Result};
use crate::types::{Image, OcrResult};

use super::{OcrEngine, ReadOptions};

/// Composes engines, trying each in order until one returns a non-empty result.
///
/// On an `Err` or an empty [`OcrResult`] (no lines) the combinator advances to the
/// next engine. It returns the first non-empty result; if none succeeds it returns
/// the last error, or — when every engine returned an empty result without error —
/// that last empty result. When some engines error and none produce text, the last
/// error is surfaced in preference to a trailing empty result, regardless of order.
pub struct FallbackEngine {
    engines: Vec<Arc<dyn OcrEngine>>,
}

impl FallbackEngine {
    /// Create from an ordered list of engines (must be non-empty).
    pub fn new(engines: Vec<Arc<dyn OcrEngine>>) -> Result<Self> {
        if engines.is_empty() {
            return Err(OcrError::config("FallbackEngine requires at least one engine"));
        }
        Ok(Self { engines })
    }

    /// Append an engine to the end of the chain.
    pub fn push(mut self, engine: Arc<dyn OcrEngine>) -> Self {
        self.engines.push(engine);
        self
    }
}

impl std::fmt::Debug for FallbackEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `dyn OcrEngine` is not `Debug`, so summarize by arity rather than contents.
        f.debug_struct("FallbackEngine")
            .field("engines", &self.engines.len())
            .finish()
    }
}

impl OcrEngine for FallbackEngine {
    fn recognize(&self, image: &Image, options: &ReadOptions) -> Result<OcrResult> {
        let mut last_error: Option<OcrError> = None;
        let mut last_empty: Option<OcrResult> = None;
        for engine in &self.engines {
            match engine.recognize(image, options) {
                Ok(result) if !result.lines.is_empty() => return Ok(result),
                Ok(empty) => last_empty = Some(empty),
                Err(error) => last_error = Some(error),
            }
        }
        // No engine produced a non-empty result. Prefer surfacing an error when one
        // occurred; otherwise return the last (empty) success. The non-empty engine
        // list guaranteed by `new` means at least one branch is populated.
        if let Some(error) = last_error {
            return Err(error);
        }
        last_empty.ok_or_else(|| OcrError::config("FallbackEngine had no engines to run"))
    }
}
