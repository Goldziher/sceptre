//! Model registry and provisioning.
//!
//! [`registry`] enumerates the gen2 detector/recognizer ONNX artifacts (hosted by
//! the `itextresearch` org on Hugging Face). [`download`] fetches and caches them.

pub mod download;
pub mod registry;

pub use registry::{ModelEntry, craft_entry, recognizer_entry};
