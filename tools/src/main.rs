//! `sceptre-tools` — dev-only model export/conversion and golden tooling for sceptre.
//!
//! This binary is the preferred, Rust-first path for exporting and converting the
//! CRAFT and gen2 CRNN models (candle-based) into the ONNX/safetensors artifacts the
//! library loads at runtime, and it generates the sceptre-snapshot side of the golden
//! parity fixtures. The Python `sceptre_rs_tools` package is the fallback export path
//! and the EasyOCR-reference golden generator; see ADR 0008 and ADR 0016.
//!
//! Model export/conversion is not yet implemented. The `snapshot` subcommand is only
//! compiled with the `ort` feature (`cargo run -p sceptre-tools --features ort`).

use anyhow::bail;
use clap::{Parser, Subcommand};

/// Dev tooling for exporting sceptre models and regenerating golden fixtures.
#[derive(Debug, Parser)]
#[command(name = "sceptre-tools", about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

/// Subcommands exposed by `sceptre-tools`.
#[derive(Debug, Subcommand)]
enum Command {
    /// Export/convert upstream checkpoints into ONNX/safetensors (not yet implemented).
    Export,
    /// Regenerate the sceptre-snapshot side of the golden fixtures (requires `--features ort`).
    #[cfg(feature = "ort")]
    Snapshot,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Some(Command::Export) | None => bail!(
            "sceptre-tools: model export/conversion dev tool — not yet implemented. \
             This is the preferred Rust-first (candle-based) export path; the Python \
             `sceptre_rs_tools` package is the fallback. See ADR 0008."
        ),
        #[cfg(feature = "ort")]
        Some(Command::Snapshot) => snapshot::run(),
    }
}

#[cfg(feature = "ort")]
mod snapshot {
    //! The `snapshot` subcommand: run the sceptre pipeline over the committed example
    //! images and write the `sceptre` side of each dual golden fixture, preserving any
    //! existing `easyocr` reference side. See `crates/sceptre/tests/data/golden/README.md`.

    use std::path::{Path, PathBuf};

    use anyhow::{Context, Result};
    use sceptre::{Language, OcrConfig, Quad, ReadOptions, Reader};
    use serde_json::{Value, json};

    /// Each example image paired with the gen2 recognizer language group to load for it.
    const IMAGES: &[(&str, Language)] = &[
        ("english.png", Language::English),
        ("example.png", Language::English),
        ("french.jpg", Language::Latin),
        ("chinese.jpg", Language::ChineseSimplified),
        ("japanese.jpg", Language::Japanese),
        ("korean.png", Language::Korean),
        ("cyrillic.png", Language::Cyrillic),
    ];

    /// Run the snapshot generator over every example image.
    pub fn run() -> Result<()> {
        let images = images_dir();
        let goldens = golden_dir();
        std::fs::create_dir_all(&goldens).with_context(|| format!("creating {}", goldens.display()))?;

        for (image_name, language) in IMAGES {
            let image_path = images.join(image_name);
            if !image_path.exists() {
                continue;
            }
            let lines = recognize_lines(&image_path, *language)
                .with_context(|| format!("running the sceptre pipeline over {}", image_path.display()))?;

            let stem = Path::new(image_name)
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or(image_name);
            let fixture_path = goldens.join(format!("{stem}.json"));
            let merged = merge_sceptre_side(load_existing(&fixture_path), lines);
            let serialized = serde_json::to_string_pretty(&merged).context("serializing the golden fixture")?;
            std::fs::write(&fixture_path, format!("{serialized}\n"))
                .with_context(|| format!("writing {}", fixture_path.display()))?;
        }

        Ok(())
    }

    /// Recognize one image and return golden line JSON objects (`text` + `quad`).
    fn recognize_lines(image_path: &Path, language: Language) -> Result<Vec<Value>> {
        let mut config = OcrConfig::default();
        config.model.languages = vec![language];
        let reader = Reader::builder()
            .config(config)
            .build()
            .context("building the sceptre reader")?;
        let result = reader.readtext(image_path, &ReadOptions::default())?;
        Ok(result
            .lines
            .iter()
            .map(|line| line_to_json(&line.quad, &line.text))
            .collect())
    }

    /// Serialize a recognized line into the golden `{ "text", "quad" }` shape.
    fn line_to_json(quad: &Quad, text: &str) -> Value {
        let corners: Vec<Value> = quad.points.iter().map(|point| json!([point.x, point.y])).collect();
        json!({ "text": text, "quad": corners })
    }

    /// Overwrite only the `sceptre` side, preserving any existing `easyocr` side.
    fn merge_sceptre_side(existing: Value, lines: Vec<Value>) -> Value {
        let easyocr = existing
            .get("easyocr")
            .cloned()
            .unwrap_or_else(|| json!({ "lines": [] }));
        json!({
            "placeholder": false,
            "easyocr": easyocr,
            "sceptre": { "lines": lines },
        })
    }

    /// Load an existing fixture, or an empty dual golden if none/invalid.
    fn load_existing(path: &Path) -> Value {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_else(|| json!({ "easyocr": { "lines": [] } }))
    }

    /// Repository root, derived from this crate's manifest directory (`<root>/tools`).
    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."))
    }

    fn images_dir() -> PathBuf {
        repo_root().join("crates/sceptre/tests/data/images")
    }

    fn golden_dir() -> PathBuf {
        repo_root().join("crates/sceptre/tests/data/golden")
    }
}
