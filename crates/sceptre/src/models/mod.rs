//! Model registry and provisioning.
//!
//! [`registry`] enumerates the gen2 detector/recognizer ONNX artifacts (hosted by
//! the `itextresearch` org on Hugging Face). [`download`] fetches and caches them.
//! [`provision`] is the public surface that enumerates (`model_manifest`) and
//! prefetches (`download_models`) the models a configuration requires.

pub(crate) mod download;
pub(crate) mod provision;
pub(crate) mod registry;
