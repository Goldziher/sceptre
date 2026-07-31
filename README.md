# easyocr-rs

A Rust reimplementation of [EasyOCR](https://github.com/JaidedAI/EasyOCR)'s OCR
pipeline — **CRAFT** text detection followed by **gen2 CRNN** recognition with
CTC decoding — running the models over ONNX. Ships as a **library**, a **CLI**
(`easyocr-rs`), and an **MCP server**.

> **Status:** early scaffolding. The workspace, configuration, and module
> structure are in place; the pipeline stages are being implemented.

## Why

- **Native performance, low RSS.** Rust + ONNX Runtime, CPU-optimized, with
  configurable Rayon parallelism bounded by a single thread budget.
- **Pure-Rust option.** A runtime-neutral backend seam lets the same pipeline run
  on `ort` (native) or `tract` (pure-Rust, for WASM/Android).
- **Library-first.** The OCR logic is a reusable rlib; the CLI and MCP server are
  thin layers over it.

## Scope

Targets EasyOCR's current models — the **CRAFT** detector and the **gen2**
(`*_g2`) recognizers: `english`, `latin`, `zh_sim`, `japanese`, `korean`,
`cyrillic`. Legacy gen1 models, DBNet, and beam-search decoding are out of scope
for now (see [`adrs/0002`](adrs/0002-scope-gen2-recognizers-and-craft.md)).

## Models

ONNX artifacts come from the
[`itextresearch/itext-EasyOCR-*`](https://huggingface.co/itextresearch) repos on
Hugging Face (Apache-2.0, dynamic-width). They are downloaded on first use,
cached under `~/.cache/easyocr-rs`, and sha256-verified — nothing is committed to
the repo (see [`adrs/0003`](adrs/0003-model-source-itext-onnx-runtime-download.md)).

## Install

```sh
cargo install --path crates/easyocr-cli
```

## CLI usage

```sh
# Full pipeline
easyocr-rs run image.png --lang english --format json

# Detection only
easyocr-rs detect image.png

# Manage models
easyocr-rs models list
easyocr-rs models download

# Shell completions
easyocr-rs completions zsh > _easyocr-rs
```

Diagnostics go to stderr (`--log-level`, `EASYOCR_LOG`); structured results go to
stdout. Colour honors `NO_COLOR`.

## Library usage

```rust
use easyocr::{OcrConfig, ReadOptions, Reader};

let reader = Reader::builder().config(OcrConfig::default()).build()?;
let result = reader.readtext("image.png".as_ref(), &ReadOptions::default())?;
for line in result.lines {
    println!("{} ({:.2})", line.text, line.confidence);
}
# Ok::<(), easyocr::OcrError>(())
```

## MCP server

Build with the `mcp` feature and run the stdio server, which exposes a `readtext`
tool:

```sh
cargo run -p easyocr-cli --features mcp -- mcp
```

## Feature flags

| Feature | Enables |
| --- | --- |
| `ort` | Native ONNX Runtime backend (desktop/server) |
| `tract` | Pure-Rust ONNX backend (WASM/Android) |
| `candle` | Pure-Rust native-tensor backend |
| `download` | Runtime model download + cache from Hugging Face |
| `mcp` | MCP (`rmcp`) server surface |

Configuration (`OcrConfig`) layers as: defaults < config file < environment < CLI
flags, and is backend-agnostic.

## Development

```sh
task setup    # fetch deps, install git hooks
task build
task check    # cargo fmt + clippy -D warnings + test + poly lint
```

- Coding conventions live in `.ai-rulez/` (the source of truth for the generated
  `CLAUDE.md` / `AGENTS.md`); run `ai-rulez generate` after editing them.
- Architecture decisions are recorded under [`adrs/`](adrs/).

## License

MIT. Model weights are distributed by third parties under their own licenses
(the gen2 EasyOCR models and the iText ONNX exports are Apache-2.0).
