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
    //! The `snapshot` subcommand: run the sceptre pipeline over the example images from
    //! the `test_documents` corpus and write the `sceptre` side of each dual golden
    //! fixture, preserving any existing `easyocr` reference side. See
    //! `crates/sceptre/tests/data/golden/README.md`.

    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::time::{SystemTime, UNIX_EPOCH};

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
        ("telugu.png", Language::Telugu),
        ("kannada.png", Language::Kannada),
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

    /// Overwrite only the `sceptre` side and its provenance, preserving any existing
    /// `easyocr` side and the `easyocr` half of the `metadata` block.
    fn merge_sceptre_side(existing: Value, lines: Vec<Value>) -> Value {
        let easyocr = existing
            .get("easyocr")
            .cloned()
            .unwrap_or_else(|| json!({ "lines": [] }));
        let mut metadata = match existing.get("metadata") {
            Some(Value::Object(object)) => Value::Object(object.clone()),
            _ => json!({}),
        };
        metadata["sceptre"] = sceptre_provenance();
        json!({
            "placeholder": false,
            "metadata": metadata,
            "easyocr": easyocr,
            "sceptre": { "lines": lines },
        })
    }

    /// The `metadata.sceptre` provenance block: which build of this crate, on which
    /// runtime, produced the snapshot side. See the golden fixtures' README.
    ///
    /// `accelerator` is recorded as `cpu` because the generator runs the default
    /// configuration, and `onnxruntime_version` is `null` because the runtime version
    /// is not exposed through the `ModelBackend` seam.
    fn sceptre_provenance() -> Value {
        let backend = serde_json::to_value(OcrConfig::default().model.backend)
            .ok()
            .and_then(|value| value.as_str().map(str::to_string))
            .unwrap_or_else(|| "unknown".to_string());
        json!({
            "sceptre_version": sceptre::VERSION,
            "git_commit": git_commit(),
            "backend": backend,
            "accelerator": "cpu",
            "onnxruntime_version": Value::Null,
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "generated_at": generated_at(),
        })
    }

    /// The full commit SHA of the checkout that generated the fixture, or `null` when
    /// git is unavailable (for example in a source tarball).
    fn git_commit() -> Value {
        let output = Command::new("git")
            .args(["-C", &repo_root().to_string_lossy(), "rev-parse", "HEAD"])
            .output()
            .ok()
            .filter(|output| output.status.success());
        match output {
            Some(output) => match String::from_utf8(output.stdout) {
                Ok(text) => Value::String(text.trim().to_string()),
                Err(_) => Value::Null,
            },
            None => Value::Null,
        }
    }

    /// The current UTC time as an RFC 3339 timestamp with second resolution.
    fn generated_at() -> String {
        let seconds = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|elapsed| elapsed.as_secs())
            .unwrap_or_default();
        format_utc(seconds)
    }

    /// Format `seconds` since the Unix epoch as `YYYY-MM-DDTHH:MM:SSZ`.
    fn format_utc(seconds: u64) -> String {
        let (year, month, day) = civil_from_days((seconds / 86_400) as i64);
        let time_of_day = seconds % 86_400;
        format!(
            "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
            time_of_day / 3_600,
            (time_of_day / 60) % 60,
            time_of_day % 60
        )
    }

    /// Convert days since the Unix epoch into a proleptic Gregorian `(year, month, day)`.
    ///
    /// This is Howard Hinnant's `civil_from_days`, inlined so the dev tool needs no date
    /// dependency.
    fn civil_from_days(days: i64) -> (i64, u32, u32) {
        let shifted = days + 719_468;
        let era = shifted.div_euclid(146_097);
        let day_of_era = shifted.rem_euclid(146_097);
        let year_of_era = (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
        let year = year_of_era + era * 400;
        let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
        let shifted_month = (5 * day_of_year + 2) / 153;
        let day = (day_of_year - (153 * shifted_month + 2) / 5 + 1) as u32;
        let month = if shifted_month < 10 {
            shifted_month + 3
        } else {
            shifted_month - 9
        } as u32;
        (if month <= 2 { year + 1 } else { year }, month, day)
    }

    /// Load an existing fixture, or an empty dual golden if none/invalid.
    fn load_existing(path: &Path) -> Value {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_else(|| json!({ "metadata": {}, "easyocr": { "lines": [] } }))
    }

    /// Repository root, derived from this crate's manifest directory (`<root>/tools`).
    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."))
    }

    /// Directory holding the `test_documents` corpus: `TEST_DOCUMENTS_DIR` when set,
    /// otherwise the `test_documents` submodule checked out at the repository root.
    fn test_documents_dir() -> PathBuf {
        std::env::var_os("TEST_DOCUMENTS_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| repo_root().join("test_documents"))
    }

    fn images_dir() -> PathBuf {
        test_documents_dir().join("images")
    }

    fn golden_dir() -> PathBuf {
        repo_root().join("crates/sceptre/tests/data/golden")
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn should_preserve_the_easyocr_side_and_its_provenance() {
            let existing = json!({
                "placeholder": false,
                "metadata": { "easyocr": { "easyocr_version": "1.7.2" } },
                "easyocr": { "lines": [{ "text": "hi", "quad": [[0, 0], [1, 0], [1, 1], [0, 1]] }] },
                "sceptre": { "lines": [] },
            });

            let merged = merge_sceptre_side(existing, vec![json!({ "text": "hi", "quad": [] })]);

            assert_eq!(merged["metadata"]["easyocr"]["easyocr_version"], json!("1.7.2"));
            assert_eq!(merged["easyocr"]["lines"][0]["text"], json!("hi"));
            assert_eq!(
                merged["metadata"]["sceptre"]["sceptre_version"],
                json!(sceptre::VERSION)
            );
            assert_eq!(merged["placeholder"], json!(false));
        }

        #[test]
        fn should_write_sceptre_provenance_when_the_fixture_has_no_metadata() {
            let merged = merge_sceptre_side(json!({ "easyocr": { "lines": [] } }), Vec::new());

            assert_eq!(merged["metadata"]["sceptre"]["os"], json!(std::env::consts::OS));
            assert!(merged["metadata"]["easyocr"].is_null());
        }

        #[test]
        fn should_format_the_unix_epoch_as_rfc_3339() {
            assert_eq!(format_utc(0), "1970-01-01T00:00:00Z");
        }

        #[test]
        fn should_format_a_leap_day_timestamp() {
            assert_eq!(format_utc(1_709_209_845), "2024-02-29T12:30:45Z");
        }
    }
}
