//! Request schemas for the MCP tool surface.
//!
//! The `readtext` tool's structured response is shaped inline by `tool_payload` in
//! [`super::tools`]: the full [`crate::OcrResult`] serialized to JSON when `detail`
//! is true, or a `{ "lines": [...] }` array of just the recognized text when false.
//! There is no separate response DTO here.

use schemars::JsonSchema;
use serde::Deserialize;

/// Parameters for the `readtext` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReadTextRequest {
    /// Filesystem path to the image to run OCR over. Read locally with the same
    /// privileges as the server process; the stdio transport is a local, trusted client.
    pub image_path: String,
    /// When false, only line text is returned (no box/confidence). Default true.
    #[serde(default)]
    pub detail: Option<bool>,
}
