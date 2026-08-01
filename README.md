# sceptre

A Rust reimplementation of [EasyOCR](https://github.com/JaidedAI/EasyOCR)'s OCR
pipeline — **CRAFT** text detection followed by **gen2 CRNN** recognition with
CTC decoding — running the models over ONNX. Ships as a **library**, a **CLI**
(`sceptre`), and an **MCP server**.

> **Status:** working. The full CRAFT detection → CRNN + CTC recognition pipeline
> runs end-to-end through the library, CLI, and MCP server on both the `ort`
> (native) and `tract` (pure-Rust) backends; models download on first use.
> Cross-backend parity fixtures against EasyOCR are opt-in and not yet generated.

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
Hugging Face (Apache-2.0, dynamic-width). They are downloaded on first use into
the shared Hugging Face hub cache (`HF_HUB_CACHE` / `HF_HOME` /
`~/.cache/huggingface/hub`), and sha256-verified — nothing is committed to the
repo (see [`adrs/0003`](adrs/0003-model-source-itext-onnx-runtime-download.md) and
[`adrs/0017`](adrs/0017-hf-hub-native-cache.md)).

## Install

```sh
cargo install --path crates/sceptre-cli
```

## CLI usage

```sh
# Full pipeline
sceptre run image.png --lang english --format json

# Detection only
sceptre detect image.png

# Manage models
sceptre models list
sceptre models download

# Shell completions
sceptre completions zsh > _sceptre
```

Diagnostics go to stderr (`--log-level`, `EASYOCR_LOG`); structured results go to
stdout. Colour honors `NO_COLOR`.

## Library usage

```rust
use sceptre::{OcrConfig, ReadOptions, Reader};

let reader = Reader::builder().config(OcrConfig::default()).build()?;
let result = reader.readtext("image.png".as_ref(), &ReadOptions::default())?;
for line in result.lines {
    println!("{} ({:.2})", line.text, line.confidence);
}
# Ok::<(), sceptre::OcrError>(())
```

## MCP server

Build with the `mcp` feature and run the stdio server, which exposes a single
`readtext` tool:

```sh
cargo run -p sceptre-cli --features mcp -- mcp --lang english
```

| Tool | Parameters | Returns |
| --- | --- | --- |
| `readtext` | `image_path` (string, required); `detail` (bool, optional, default `true`) | Structured JSON. With `detail`, the full result — each line's `text`, `confidence`, and bounding-box `quad`. Without it, a `lines` array of just the recognized text. |

Recognition language, backend, and thread budget follow the flags passed to
`sceptre mcp` (e.g. `--lang`, `--backend`, `--threads`).

## Feature flags

| Feature | Enables |
| --- | --- |
| `ort` | Native ONNX Runtime backend (desktop/server) |
| `tract` | Pure-Rust ONNX backend (WASM/Android) |
| `candle` | Reserved for a future pure-Rust native-tensor backend — not yet implemented (see ADR 0009) |
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

### Benchmarks

Microbenchmarks use criterion behind a `bench` cargo feature:

```sh
task bench
cargo bench -p sceptre --features bench            # full run
cargo bench -p sceptre --features bench -- --test  # fast smoke, skips timing
```

Results are local only — there is no CI perf gate; CI just compile-checks the benches.

### Parity testing

The crate ships golden/parity tests against real EasyOCR output. They skip by
default and opt in when you supply downloaded models plus the `test_documents`
submodule via an environment flag. See
[`crates/sceptre/tests/data/golden/README.md`](crates/sceptre/tests/data/golden/README.md)
for regeneration.

## License

MIT. Model weights are distributed by third parties under their own licenses
(the gen2 EasyOCR models and the iText ONNX exports are Apache-2.0).
