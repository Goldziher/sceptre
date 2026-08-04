# Architecture Decision Records

This directory holds the project's Architecture Decision Records (ADRs) in
[MADR](https://adr.github.io/madr/) format.

## Conventions

- One decision per file, named `NNNN-kebab-title.md`, numbered sequentially.
- Each ADR carries a `status` (`proposed` | `accepted` | `rejected` | `superseded`),
  a `date`, and `deciders`.
- ADRs are immutable once accepted. To reverse a decision, add a new ADR that
  supersedes the old one; mark the old one `superseded by NNNN`.
- Add an ADR whenever a choice between real alternatives has lasting consequences
  (a dependency, backend, algorithm, public API shape, model source, or target
  strategy). See the `adr-discipline` rule in `.ai-rulez/rules/`.

## Index

| ADR | Title | Status |
| --- | --- | --- |
| [0000](0000-use-madr-for-adrs.md) | Record decisions as MADR ADRs | Accepted |
| [0001](0001-cargo-workspace-rlib-core-thin-cli.md) | Cargo workspace: rlib core + thin CLI | Accepted |
| [0002](0002-scope-gen2-recognizers-and-craft.md) | Scope to gen2 recognizers + CRAFT | Superseded by 0026 |
| [0003](0003-model-source-itext-onnx-runtime-download.md) | Model source: iText ONNX, runtime download | Superseded by 0025 |
| [0004](0004-inference-backend-seam.md) | Inference backend seam: ort / tract / candle | Accepted |
| [0005](0005-mcp-as-feature-module-and-subcommand.md) | MCP as a feature-gated module + subcommand | Accepted |
| [0006](0006-ai-rulez-for-agent-config.md) | Generate agent config with ai-rulez | Accepted |
| [0007](0007-tooling-scaffolding-from-basemind.md) | Tooling scaffolding adapted from basemind | Accepted |
| [0028](0028-host-supplied-model-artifacts-and-wasm-execution.md) | Host-supplied model artifacts and sequential browser-WASM execution | Accepted |
