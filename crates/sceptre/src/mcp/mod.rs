//! MCP (Model Context Protocol) server surface.
//!
//! Exposes the OCR pipeline over [`rmcp`], so an MCP client can call a `readtext`
//! tool. [`server`] hosts the handler, [`tools`] defines the tool router, and
//! [`types`] holds the request/response schemas.

pub mod server;
pub mod tools;
pub mod types;

use crate::engine::Reader;
use crate::error::{OcrError, Result};

/// Run the MCP stdio server backed by `reader`.
pub fn serve(_reader: Reader) -> Result<()> {
    Err(OcrError::config("the MCP server is not yet implemented"))
}
