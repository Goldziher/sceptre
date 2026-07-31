---
status: accepted
date: 2026-07-31
deciders: Na'aman Hirschfeld
---

# MCP as a feature-gated module + subcommand

## Context and Problem Statement

The project must expose OCR over MCP (via `rmcp`) in addition to the CLI. We need
to decide whether the MCP server is its own crate/binary or part of the existing
surface.

## Considered Options

- A dedicated `crates/easyocr-mcp` binary crate.
- An `mcp` feature-gated module in the core lib, exposed via an `easyocr-rs mcp`
  subcommand.

## Decision Outcome

Chosen option: **feature-gated `mcp` module + `easyocr-rs mcp` subcommand**,
mirroring the xberg pattern. `rmcp`/`tokio`/`schemars` are optional dependencies
enabled only by the feature, so default builds stay lean. One binary ships both
the CLI and the server.

### Consequences

- Good: one binary to build and distribute; no duplicated wiring.
- Good: MCP dependencies are absent from default builds.
- Bad: the CLI crate gains an optional feature and a cfg-gated subcommand.
