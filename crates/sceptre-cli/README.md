# sceptre-cli

**EasyOCR's accuracy. Rust's speed and footprint.**

The command-line interface for [sceptre](https://crates.io/crates/sceptre) — a from-scratch Rust
reimplementation of [EasyOCR](https://github.com/JaidedAI/EasyOCR)'s pipeline (**CRAFT** detection
then **gen2 CRNN** recognition over ONNX). One self-contained binary, no Python runtime, matching
EasyOCR across six scripts (English, Latin, Chinese-simplified, Japanese, Korean, Cyrillic).

Models download from Hugging Face on first use, cache locally, and are sha256-verified — every run
after that is fully offline.

## Install

```sh
cargo install sceptre-cli
```

## Usage

```sh
sceptre run receipt.png --lang english --format json
sceptre run page1.png page2.png page3.png            # batch: models load once
sceptre run sign.jpg --lang english --lang korean    # multi-language
sceptre mcp --lang english                           # expose a readtext MCP tool
```

Subcommands: `run`, `detect`, `models`, `mcp`, `completions`.

Full documentation, benchmarks, and design notes live at the
[project repository](https://github.com/Goldziher/sceptre).

## License

MIT. Model weights are distributed by third parties under their own licenses (Apache-2.0).
