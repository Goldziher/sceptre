<!-- markdownlint-disable MD033 MD041 -->
<div align="center">

<img src="https://raw.githubusercontent.com/Goldziher/sceptre/main/docs/assets/banner.svg" alt="sceptre" width="820">

**EasyOCR's accuracy. Rust's speed and footprint.**

sceptre is a from-scratch Rust reimplementation of [EasyOCR](https://github.com/JaidedAI/EasyOCR)'s
OCR pipeline — **CRAFT** text detection then **gen2 CRNN** recognition with CTC decoding, over ONNX.
It matches EasyOCR's output on every script it supports, runs **~2.8× faster on a fraction of the
memory** (and a cold one-shot run is ~4.4× faster than EasyOCR warm), and ships as one self-contained
binary with **no Python runtime**. Use it as a **library**, a **CLI**, or an **MCP server**.

6 scripts · CRAFT + gen2 CRNN · ONNX Runtime **or** pure-Rust · library · CLI · MCP · offline-first

[![crates.io](https://img.shields.io/crates/v/sceptre?color=f5b301&style=flat-square)](https://crates.io/crates/sceptre)
[![CI](https://img.shields.io/github/actions/workflow/status/Goldziher/sceptre/ci.yaml?style=flat-square&color=f5b301)](https://github.com/Goldziher/sceptre/actions/workflows/ci.yaml)
[![docs.rs](https://img.shields.io/docsrs/sceptre?style=flat-square&color=f5b301)](https://docs.rs/sceptre)
[![Documentation](https://img.shields.io/badge/docs-goldziher.github.io%2Fsceptre-f5b301?style=flat-square)](https://goldziher.github.io/sceptre)
[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)

[Documentation](https://goldziher.github.io/sceptre) · [Install](#install) · [Quickstart](#quickstart) · [Why sceptre](#why-sceptre) · [Benchmarks](#benchmarks) · [How it works](#how-it-works) · [Contributing](CONTRIBUTING.md)

</div>

---

## Why sceptre

EasyOCR is excellent and accurate — but it's a PyTorch stack: a Python interpreter, a multi-gigabyte
runtime, and a heavy process to keep warm. sceptre keeps the accuracy and drops all of that.

| | What you get | Why it matters |
|---|---|---|
| **Parity accuracy** | Validated against real EasyOCR output across all six gen2 scripts — English, Latin, Chinese (simplified), Japanese, Korean, Cyrillic — matching text (word/char-F1) and boxes (IoU). | A faithful reimplementation, not an approximation. What EasyOCR reads, sceptre reads. |
| **Substantially faster** | ~2.8× higher throughput than EasyOCR warm on the same corpus — and even a cold, one-shot CLI run (~4.4×) beats EasyOCR's warm, already-loaded reader. | More pages per second, less waiting, cheaper batch jobs. |
| **A fraction of the memory** | Peak RSS around **3× lower** than the Python + torch process, measured like-for-like (both whole-process peaks). | Runs where EasyOCR won't — small containers, edge boxes, many workers. |
| **One binary, no Python** | A single static executable. Models download once, cache locally, and run **offline** thereafter. | `cargo install` and go — nothing to `pip install`, no interpreter to ship. |
| **Three surfaces** | The same engine as a Rust **library**, a **CLI** (`sceptre`), and an **MCP server** for agents. | Drop it into a service, a shell pipeline, or an AI tool without re-plumbing. |
| **Native or pure-Rust** | ONNX Runtime (`ort`) for native speed, or a pure-Rust backend (`tract`) for WASM / Android — behind one seam. | Portability when you need it, native performance when you don't. |

---

## Install

```sh
cargo install sceptre-cli
```

The models — CRAFT plus the gen2 recognizers — are fetched from Hugging Face on first use, cached
under the standard HF cache, and sha256-verified on download. Every run after that reads the cached
models with no network.

> The default build loads ONNX Runtime at runtime (`ORT_DYLIB_PATH`). For a zero-config native
> binary that bundles the runtime, install with `--features ort-bundled`.

## Quickstart

**CLI** — one image or many in a single warm process:

```sh
sceptre run receipt.png --lang english --format json
sceptre run page1.png page2.png page3.png            # batch: models load once
sceptre run sign.jpg --lang english --lang korean    # multi-language
sceptre run page.png --lang english --timings        # per-stage load/detect/recognize breakdown
sceptre run huge-scan.png --canvas-size 1600         # trade some accuracy for lower peak memory
```

Decodes PNG, JPEG, BMP, GIF, TIFF, WebP, and NetPBM (all pure-Rust; HEIF/AVIF/JPEG-2000
are out of scope — see [`adrs/0022`](adrs/0022-input-image-format-coverage.md)).

**Library:**

```rust
use sceptre::{Reader, ReadOptions};

let reader = Reader::builder().build()?;
for line in reader.readtext("receipt.png".as_ref(), &ReadOptions::default())?.lines {
    println!("{} ({:.2})", line.text, line.confidence);
}
# Ok::<(), sceptre::OcrError>(())
```

The library ships `default = []`, so enable a backend and model download:
`sceptre = { version = "0.1", features = ["ort-bundled", "download"] }` (see [Feature flags](#feature-flags)).

**MCP server** — expose a `readtext` tool to an agent:

```sh
sceptre mcp --lang english
```

## Benchmarks

Measured against upstream EasyOCR over a 43-image mixed corpus (documents, tables, rotated scans,
scene text, receipts, five scripts), on CPU. Both engines are measured **identically** — each a fresh
subprocess per language group under `/usr/bin/time`, at its native multi-threaded default, loading its
model/reader once and processing every image — so peak RSS is a like-for-like whole-process figure
(EasyOCR's legitimately includes the torch runtime). Numbers vary with hardware and load; the
**ratios** are the point.

| Engine | Throughput (img/s) | Peak RSS | Mean CER | Mean token-F1 |
|---|---|---|---|---|
| EasyOCR (warm/batch) | 0.14 | 22.6 GB | 0.554 | 0.348 |
| **sceptre** (warm/batch) | **0.39** (~2.8×) | **6.6 GB** (~3× lower) | 0.568 | **0.356** |
| sceptre (cold CLI run) | 0.60 (~4.4×) | 6.6 GB | 0.568 | 0.356 |

CER and token-F1 are at parity (sceptre is marginally ahead on token-F1); the win is speed and memory.
Even a cold, one-shot CLI run — which pays model load every invocation — is ~4.4× faster than
EasyOCR's already-warm reader.

`cargo bench` covers the internal hot paths; the head-to-head harness (`task python:benchmark`)
reproduces the table above and writes `benchmark-results/comparison.{json,md}`. Use
`--group labeled --limit 3 --repeats 1` for a fast inner-loop run, `--baseline <prior.json>` to see
per-image deltas, and `--assert` for the regression gate (see
[ADR 0021](adrs/0021-benchmark-methodology-and-gate.md)). Parity fixtures live under
[`crates/sceptre/tests/data/golden/`](crates/sceptre/tests/data/golden/).

## How it works

Three stages behind one `Reader`, mirroring EasyOCR's latest pipeline:

1. **Detect** — CRAFT produces region/link heat-maps; thresholding, connected components, and
   min-area boxes become text lines (horizontal and rotated).
2. **Recognize** — each line is cropped, normalized, and run through a gen2 CRNN; CTC greedy decoding
   turns the logits into text and a confidence.
3. **Inference** — every model call goes through one backend seam: `ort` (native ONNX Runtime) or
   `tract` (pure Rust). `config`, `types`, and geometry stay backend-agnostic.

Design decisions live as [MADR records under `adrs/`](adrs/); conventions live in `.ai-rulez/`.

## Scope

Targets EasyOCR's current models — the **CRAFT** detector and the six **gen2** (`*_g2`) recognizers.
Legacy gen1 models, DBNet, and beam-search decoding are out of scope
(see [`adrs/0002`](adrs/0002-scope-gen2-recognizers-and-craft.md)). ONNX artifacts come from the
[`itextresearch/itext-EasyOCR-*`](https://huggingface.co/itextresearch) repos (Apache-2.0).

## Feature flags

| Feature | Enables |
| --- | --- |
| `ort` | Native ONNX Runtime backend (desktop / server) |
| `ort-bundled` | `ort` with a prebuilt runtime fetched at build time (zero-config) |
| `tract` | Pure-Rust ONNX backend (WASM / Android) |
| `download` | Runtime model download + cache from Hugging Face |
| `mcp` | MCP (`rmcp`) server surface |
| `candle` | Reserved for a future pure-Rust native-tensor backend (see ADR 0009) |

Configuration (`OcrConfig`) layers as defaults < config file < environment < CLI flags, and is
backend-agnostic.

## Development

```sh
task setup    # fetch deps, install git hooks
task check    # cargo fmt + clippy -D warnings + test + poly lint
task bench    # criterion microbenchmarks (--features bench)
```

Coding conventions are generated from `.ai-rulez/` into `CLAUDE.md` / `AGENTS.md` — edit the source,
then run `ai-rulez generate`.

## License

MIT. Model weights are distributed by third parties under their own licenses (the gen2 EasyOCR models
and the iText ONNX exports are Apache-2.0).
