//! Model artifact providers for filesystem and in-memory hosts.

use std::path::PathBuf;
use std::sync::Arc;

use crate::config::Language;
use crate::error::{OcrError, Result};
use crate::models::integrity::verify_sha256_bytes;
use crate::models::provision::{ModelDescriptor, ModelRole};

/// A model artifact resolved either as a local path or in-memory bytes.
#[derive(Debug, Clone)]
pub enum ModelArtifact {
    /// A local ONNX file read when the inference backend initializes.
    Path(PathBuf),
    /// ONNX bytes supplied by an embedding host without filesystem access.
    Bytes(Arc<[u8]>),
}

/// Resolves detector and recognizer model artifacts.
pub trait ModelProvider: Send + Sync {
    /// Resolve the CRAFT detector ONNX artifact.
    fn detector(&self) -> Result<ModelArtifact>;

    /// Resolve the recognizer ONNX artifact for a language group.
    fn recognizer(&self, language: Language) -> Result<ModelArtifact>;
}

/// An in-memory provider that verifies every artifact before exposing it.
#[derive(Debug)]
pub struct VerifiedModelProvider {
    detector: Arc<[u8]>,
    recognizers: Vec<(Language, Arc<[u8]>)>,
}

impl VerifiedModelProvider {
    /// Build a provider from descriptor/byte pairs after SHA-256 verification.
    ///
    /// Exactly one detector is required. Duplicate detector or recognizer roles are
    /// rejected so model selection remains deterministic.
    pub fn new(artifacts: impl IntoIterator<Item = (ModelDescriptor, Arc<[u8]>)>) -> Result<Self> {
        let mut detector = None;
        let mut recognizers = Vec::new();
        for (descriptor, bytes) in artifacts {
            verify_sha256_bytes(&bytes, &descriptor.sha256, &descriptor.name)?;
            match descriptor.role {
                ModelRole::Detector => {
                    if detector.replace(bytes).is_some() {
                        return Err(OcrError::model(
                            "verified model provider received more than one detector",
                        ));
                    }
                }
                ModelRole::Recognizer(language) => {
                    if recognizers.iter().any(|(existing, _)| *existing == language) {
                        return Err(OcrError::model(format!(
                            "verified model provider received more than one {language:?} recognizer"
                        )));
                    }
                    recognizers.push((language, bytes));
                }
            }
        }
        let detector = detector.ok_or_else(|| OcrError::model("verified model provider requires one detector"))?;
        Ok(Self { detector, recognizers })
    }
}

impl ModelProvider for VerifiedModelProvider {
    fn detector(&self) -> Result<ModelArtifact> {
        Ok(ModelArtifact::Bytes(self.detector.clone()))
    }

    fn recognizer(&self, language: Language) -> Result<ModelArtifact> {
        self.recognizers
            .iter()
            .find(|(candidate, _)| *candidate == language)
            .map(|(_, bytes)| ModelArtifact::Bytes(bytes.clone()))
            .ok_or_else(|| OcrError::model(format!("no verified recognizer bytes were supplied for {language:?}")))
    }
}
