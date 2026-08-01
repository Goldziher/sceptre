//! The rmcp `ServerHandler` implementation and its transport wiring.

use rmcp::model::{ServerCapabilities, ServerInfo};
use rmcp::transport::stdio;
use rmcp::{ServerHandler, ServiceExt, tool_handler};

use crate::engine::Reader;
use crate::error::OcrError;

use super::tools::SceptreServer;

#[tool_handler]
impl ServerHandler for SceptreServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build()).with_instructions(
            "sceptre OCR server: call `readtext` with an image path to detect and \
             recognize text (CRAFT detection + gen2 CRNN recognition).",
        )
    }
}

/// Serve the MCP protocol over stdio until the client disconnects.
///
/// Builds a [`SceptreServer`] from `reader`, serves it, and awaits completion,
/// mapping rmcp transport and initialization failures to [`OcrError::Inference`].
pub(crate) async fn serve_stdio(reader: Reader) -> crate::Result<()> {
    let server = SceptreServer::new(reader);
    let running = server
        .serve(stdio())
        .await
        .map_err(|error| OcrError::inference(format!("MCP server failed to start: {error}")))?;
    running
        .waiting()
        .await
        .map_err(|error| OcrError::inference(format!("MCP server terminated abnormally: {error}")))?;
    Ok(())
}
