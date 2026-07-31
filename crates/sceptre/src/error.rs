//! Error types for sceptre.
//!
//! Philosophy (mirrors the xberg convention):
//!
//! - System errors ([`OcrError::Io`]) bubble up unchanged via `#[from]`.
//! - Application errors are structured variants carrying a `message` plus an
//!   optional `#[source]` so the underlying error chain is preserved.
//!
//! The library uses `thiserror`; the CLI uses `anyhow` on top of it.

use thiserror::Error;

/// Standard result type for all fallible operations in sceptre.
pub type Result<T> = std::result::Result<T, OcrError>;

/// The single error type for all sceptre operations.
#[derive(Debug, Error)]
pub enum OcrError {
    /// A file-system or I/O operation failed. Always bubbles up unchanged.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Loading, resolving, downloading, or verifying a model failed.
    #[error("Model error: {message}")]
    Model {
        /// Human-readable description of the model failure.
        message: String,
        /// Underlying error, if any.
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// An inference backend returned an error or unusable output.
    #[error("Inference error: {message}")]
    Inference {
        /// Human-readable description of the inference failure.
        message: String,
        /// Underlying error from the backend, if any.
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Image decoding or manipulation failed.
    #[error("Image error: {message}")]
    Image {
        /// Human-readable description of the image failure.
        message: String,
        /// Underlying error from the image library, if any.
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Invalid configuration or input parameters were supplied.
    #[error("Config error: {message}")]
    Config {
        /// Human-readable description of the configuration failure.
        message: String,
        /// Underlying error, if any.
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// A catch-all for uncommon errors that do not fit another variant.
    #[error("{0}")]
    Other(String),
}

impl OcrError {
    /// Create a [`OcrError::Model`] error without a source.
    pub fn model(message: impl Into<String>) -> Self {
        Self::Model {
            message: message.into(),
            source: None,
        }
    }

    /// Create an [`OcrError::Inference`] error without a source.
    pub fn inference(message: impl Into<String>) -> Self {
        Self::Inference {
            message: message.into(),
            source: None,
        }
    }

    /// Create an [`OcrError::Image`] error without a source.
    pub fn image(message: impl Into<String>) -> Self {
        Self::Image {
            message: message.into(),
            source: None,
        }
    }

    /// Create an [`OcrError::Config`] error without a source.
    pub fn config(message: impl Into<String>) -> Self {
        Self::Config {
            message: message.into(),
            source: None,
        }
    }
}
