//! MCP (Model Context Protocol) server surface.
//!
//! Exposes the OCR pipeline over [`rmcp`], so an MCP client can call a `readtext`
//! tool. [`server`] hosts the handler, [`tools`] defines the tool router, and
//! [`types`] holds the request schemas.

pub mod server;
pub mod tools;
pub mod types;

use crate::engine::Reader;
use crate::error::{OcrError, Result};

/// Run the MCP stdio server backed by `reader`.
///
/// This is the synchronous entry point the CLI calls: it builds a multi-thread
/// tokio runtime, then blocks on the async stdio server until the client
/// disconnects. Runtime construction failures map to [`OcrError::Inference`].
pub fn serve(reader: Reader) -> Result<()> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|error| OcrError::inference(format!("failed to build the MCP runtime: {error}")))?;
    runtime.block_on(server::serve_stdio(reader))
}
