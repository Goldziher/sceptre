//! The MCP tool router: the `readtext` tool that runs OCR over an image.

use std::path::Path;

use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock, ErrorData};
use rmcp::{tool, tool_router};

use crate::engine::{ReadOptions, Reader};

use super::types::ReadTextRequest;

/// The MCP server: a thin wrapper holding a ready-to-use [`Reader`].
///
/// Tool bodies build [`ReadOptions`] and call the library; no OCR logic lives here.
#[derive(Clone)]
pub struct SceptreServer {
    reader: Reader,
}

impl SceptreServer {
    /// Build a server backed by `reader`.
    pub fn new(reader: Reader) -> Self {
        Self { reader }
    }
}

#[tool_router(vis = "pub(crate)")]
impl SceptreServer {
    /// Run CRAFT+CRNN OCR over an image file and return the recognized text lines
    /// with their bounding boxes and confidences.
    #[tool(description = "Run CRAFT+CRNN OCR over an image file and return the recognized \
                       text lines with their bounding boxes and confidences.")]
    pub async fn readtext(&self, Parameters(req): Parameters<ReadTextRequest>) -> Result<CallToolResult, ErrorData> {
        let detail = req.detail.unwrap_or(true);
        match self
            .reader
            .readtext(Path::new(&req.image_path), &ReadOptions { detail })
        {
            Ok(result) => Ok(CallToolResult::structured(tool_payload(&result, detail)?)),
            Err(ocr_error) => Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                "OCR failed: {ocr_error}"
            ))])),
        }
    }
}

/// Shape the recognition result for the tool response. With `detail` the full
/// [`OcrResult`](crate::OcrResult) (text, confidence, and bounding box per line) is
/// returned; without it, only each line's text under a `lines` array, so callers that
/// want just the words are not forced to parse box geometry.
fn tool_payload(result: &crate::OcrResult, detail: bool) -> Result<rmcp::serde_json::Value, ErrorData> {
    if detail {
        rmcp::serde_json::to_value(result).map_err(|error| ErrorData::internal_error(error.to_string(), None))
    } else {
        let lines: Vec<&str> = result.lines.iter().map(|line| line.text.as_str()).collect();
        Ok(rmcp::serde_json::json!({ "lines": lines }))
    }
}
