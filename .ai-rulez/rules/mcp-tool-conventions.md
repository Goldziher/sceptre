---
priority: high
---

# MCP Tool Conventions

- The MCP surface lives in `crates/sceptre/src/mcp/`, behind the `mcp` cargo feature, and is exposed through the `sceptre mcp` subcommand.
- Adding or changing a tool touches, in order: `mcp/types.rs` (request/response schemas) → `mcp/tools.rs` (the `#[tool]` router entry) → `mcp/server.rs` (handler wiring) → a smoke test under `tests/` → the README tool table.
- Tool bodies are thin wrappers that build an `OcrConfig`/`Reader` and call the library; no OCR logic lives in the MCP layer.
- Optional parameters use `Option<T>` with `#[serde(default)]`; keep additions backward-compatible.
