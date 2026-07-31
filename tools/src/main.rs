//! `easyocr-tools` — dev-only model export/conversion tooling for easyocr-rs.
//!
//! This binary is the preferred, Rust-first path for exporting and converting the
//! CRAFT and gen2 CRNN models (candle-based) into the ONNX/safetensors artifacts the
//! library loads at runtime. The Python `easyocr_rs_tools` package is the fallback
//! export path and the golden-fixture generator; see ADR 0008.
//!
//! The conversion logic is not yet implemented — this binary currently only exposes
//! its CLI shape.

use anyhow::bail;
use clap::Parser;

/// Dev tooling for exporting and converting easyocr-rs models.
#[derive(Debug, Parser)]
#[command(name = "easyocr-tools", about, long_about = None)]
struct Cli {}

fn main() -> anyhow::Result<()> {
    let _cli = Cli::parse();
    bail!(
        "easyocr-tools: model export/conversion dev tool — not yet implemented. \
         This is the preferred Rust-first (candle-based) export path; the Python \
         `easyocr_rs_tools` package is the fallback. See ADR 0008."
    )
}
