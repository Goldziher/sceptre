//! Rendering of OCR results and model listings to text or JSON.

use std::io::{self, Write};
use std::path::{Path, PathBuf};

use anstyle::Style;
use clap::ValueEnum;
use sceptre::{ModelInfo, ModelRole, OcrResult, Quad, TextLine};

use crate::style;

/// Number of decimal places used when formatting confidence scores.
const CONFIDENCE_PRECISION: usize = 3;

/// Output serialization for command results.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum OutputFormat {
    /// Human-readable tab-separated text.
    Text,
    /// Pretty-printed JSON.
    Json,
}

/// Serializable projection of [`ModelInfo`] (which is not itself `Serialize`).
#[derive(serde::Serialize)]
struct ModelReport {
    name: String,
    repo: String,
    role: String,
    cached: bool,
    path: Option<String>,
}

/// Render a full OCR result as text or JSON.
pub fn render_result(result: &OcrResult, format: OutputFormat, detail: bool, writer: &mut dyn Write) -> io::Result<()> {
    match format {
        OutputFormat::Json => write_json(result, writer),
        OutputFormat::Text => {
            for line in &result.lines {
                write_text_line(line, detail, writer)?;
            }
            Ok(())
        }
    }
}

/// One image's outcome in a batch run: recognized text, or a failure message.
pub type ImageOutcome = Result<OcrResult, String>;

/// Serializable batch entry: `{ "image", "lines" }` on success, `{ "image", "error" }` on failure.
#[derive(serde::Serialize)]
struct BatchEntry<'a> {
    image: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    lines: Option<&'a [TextLine]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<&'a str>,
}

/// Render a batch of per-image OCR outcomes as text or JSON.
///
/// JSON emits an array of `{ "image", "lines" }` (or `{ "image", "error" }`) objects in input
/// order. Text prints a per-image path header followed by that image's lines or its error, with a
/// blank line separating images.
pub fn render_batch(
    items: &[(PathBuf, ImageOutcome)],
    format: OutputFormat,
    detail: bool,
    writer: &mut dyn Write,
) -> io::Result<()> {
    match format {
        OutputFormat::Json => {
            let entries: Vec<BatchEntry<'_>> = items.iter().map(|(path, outcome)| batch_entry(path, outcome)).collect();
            write_json(&entries, writer)
        }
        OutputFormat::Text => {
            for (path, outcome) in items {
                writeln!(writer, "{}", styled(&path.display().to_string(), style::heading()))?;
                match outcome {
                    Ok(result) => {
                        for line in &result.lines {
                            write_text_line(line, detail, writer)?;
                        }
                    }
                    Err(message) => {
                        writeln!(writer, "{}", styled(&format!("error: {message}"), style::error()))?;
                    }
                }
                writeln!(writer)?;
            }
            Ok(())
        }
    }
}

/// Project one image outcome into its serializable [`BatchEntry`].
fn batch_entry<'a>(path: &'a Path, outcome: &'a ImageOutcome) -> BatchEntry<'a> {
    let image = path.display().to_string();
    match outcome {
        Ok(result) => BatchEntry {
            image,
            lines: Some(&result.lines),
            error: None,
        },
        Err(message) => BatchEntry {
            image,
            lines: None,
            error: Some(message),
        },
    }
}

/// Render detected quadrilaterals as text or JSON.
pub fn render_quads(quads: &[Quad], format: OutputFormat, writer: &mut dyn Write) -> io::Result<()> {
    match format {
        OutputFormat::Json => write_json(&quads, writer),
        OutputFormat::Text => {
            for quad in quads {
                writeln!(writer, "{}", format_quad(quad))?;
            }
            Ok(())
        }
    }
}

/// Render a single recognized line as text or JSON.
pub fn render_line(line: &TextLine, format: OutputFormat, detail: bool, writer: &mut dyn Write) -> io::Result<()> {
    match format {
        OutputFormat::Json => write_json(line, writer),
        OutputFormat::Text => {
            if detail {
                let conf = styled(&format!("{:.*}", CONFIDENCE_PRECISION, line.confidence), style::dim());
                writeln!(writer, "{}\t{}", line.text, conf)
            } else {
                writeln!(writer, "{}", line.text)
            }
        }
    }
}

/// Render a model listing as text or JSON.
pub fn render_models(models: &[ModelInfo], format: OutputFormat, writer: &mut dyn Write) -> io::Result<()> {
    match format {
        OutputFormat::Json => {
            let reports: Vec<ModelReport> = models.iter().map(model_report).collect();
            write_json(&reports, writer)
        }
        OutputFormat::Text => {
            writeln!(writer, "{}", styled("MODELS", style::heading()))?;
            for info in models {
                write_model_line(info, writer)?;
            }
            Ok(())
        }
    }
}

/// Write a single OCR text line, honoring the detail flag.
fn write_text_line(line: &TextLine, detail: bool, writer: &mut dyn Write) -> io::Result<()> {
    if detail {
        let conf = styled(&format!("{:.*}", CONFIDENCE_PRECISION, line.confidence), style::dim());
        writeln!(writer, "{}\t{}\t{}", line.text, conf, format_quad(&line.quad))
    } else {
        writeln!(writer, "{}", line.text)
    }
}

/// Write a single model listing row.
fn write_model_line(info: &ModelInfo, writer: &mut dyn Write) -> io::Result<()> {
    let role = role_label(&info.role);
    let status = if info.cached {
        styled("cached", style::success())
    } else {
        styled("missing", style::warning())
    };
    writeln!(writer, "{}\t{}\t{}\t{}", info.name, role, status, info.repo)
}

/// Project a [`ModelInfo`] into its serializable [`ModelReport`].
fn model_report(info: &ModelInfo) -> ModelReport {
    ModelReport {
        name: info.name.clone(),
        repo: info.repo.clone(),
        role: role_label(&info.role),
        cached: info.cached,
        path: info.path.as_ref().map(|path| path.display().to_string()),
    }
}

/// Pretty-print `value` as JSON followed by a trailing newline.
fn write_json<T: serde::Serialize>(value: &T, writer: &mut dyn Write) -> io::Result<()> {
    serde_json::to_writer_pretty(&mut *writer, value).map_err(io::Error::other)?;
    writeln!(writer)
}

/// Format a quad's four corners as `"x0,y0 x1,y1 x2,y2 x3,y3"` with integer coordinates.
fn format_quad(quad: &Quad) -> String {
    quad.points
        .iter()
        .map(|point| format!("{},{}", point.x.round() as i64, point.y.round() as i64))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Human-readable label for a model role.
///
/// The recognizer language uses its serde snake_case token (e.g. `english`,
/// `chinese_simplified`) so the text label matches the JSON/config spelling.
fn role_label(role: &ModelRole) -> String {
    match role {
        ModelRole::Detector => "detector".to_string(),
        ModelRole::Recognizer(language) => format!("recognizer:{}", language_token(language)),
    }
}

/// The serde snake_case token for a [`Language`], or its Debug form if serialization fails.
fn language_token(language: &sceptre::Language) -> String {
    match serde_json::to_value(language) {
        Ok(serde_json::Value::String(token)) => token,
        _ => format!("{language:?}"),
    }
}

/// Wrap `text` in the given style's ANSI escapes; escapes auto-strip on non-TTY output.
fn styled(text: &str, style: Style) -> String {
    format!("{}{}{}", style.render(), text, style.render_reset())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sceptre::Point;

    fn sample_quad() -> Quad {
        Quad {
            points: [
                Point::new(1.4, 2.6),
                Point::new(10.5, 2.6),
                Point::new(10.5, 8.9),
                Point::new(1.4, 8.9),
            ],
        }
    }

    fn sample_result() -> OcrResult {
        OcrResult {
            lines: vec![TextLine {
                quad: sample_quad(),
                text: "hello".to_string(),
                confidence: 0.876,
            }],
        }
    }

    #[test]
    fn should_render_result_as_parseable_json() {
        let mut buffer: Vec<u8> = Vec::new();
        render_result(&sample_result(), OutputFormat::Json, true, &mut buffer).unwrap();
        let text = String::from_utf8(buffer).unwrap();
        let value: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(value["lines"][0]["text"], "hello");
    }

    #[test]
    fn should_render_result_text_without_detail_prints_only_text() {
        let mut buffer: Vec<u8> = Vec::new();
        render_result(&sample_result(), OutputFormat::Text, false, &mut buffer).unwrap();
        let text = String::from_utf8(buffer).unwrap();
        assert_eq!(text, "hello\n");
    }

    #[test]
    fn should_render_result_text_with_detail_includes_confidence_and_quad() {
        let mut buffer: Vec<u8> = Vec::new();
        render_result(&sample_result(), OutputFormat::Text, true, &mut buffer).unwrap();
        let text = String::from_utf8(buffer).unwrap();
        assert!(text.contains("hello"), "missing text: {text}");
        assert!(text.contains("0.876"), "missing confidence: {text}");
        assert!(text.contains("1,3"), "missing rounded quad coords: {text}");
        assert!(text.contains("11,3"), "missing rounded quad coords: {text}");
    }

    #[test]
    fn should_render_models_text_with_name_status_and_repo() {
        let models = vec![
            ModelInfo {
                name: "craft_mlt_25k".to_string(),
                repo: "itextresearch/itext-EasyOCR-craft".to_string(),
                role: ModelRole::Detector,
                cached: true,
                path: None,
            },
            ModelInfo {
                name: "english_g2".to_string(),
                repo: "itextresearch/itext-EasyOCR-english".to_string(),
                role: ModelRole::Detector,
                cached: false,
                path: None,
            },
        ];
        let mut buffer: Vec<u8> = Vec::new();
        render_models(&models, OutputFormat::Text, &mut buffer).unwrap();
        let text = String::from_utf8(buffer).unwrap();
        assert!(text.contains("craft_mlt_25k"), "missing name: {text}");
        assert!(text.contains("english_g2"), "missing name: {text}");
        assert!(text.contains("cached"), "missing cached word: {text}");
        assert!(text.contains("missing"), "missing missing word: {text}");
        assert!(
            text.contains("itextresearch/itext-EasyOCR-craft"),
            "missing repo: {text}"
        );
    }

    #[test]
    fn should_format_quad_rounding_coordinates() {
        assert_eq!(format_quad(&sample_quad()), "1,3 11,3 11,9 1,9");
    }

    #[test]
    fn should_label_recognizer_role_with_snake_case_language() {
        let role = ModelRole::Recognizer(sceptre::Language::ChineseSimplified);
        assert_eq!(role_label(&role), "recognizer:chinese_simplified");
    }

    #[test]
    fn should_label_recognizer_role_with_lowercase_english() {
        let role = ModelRole::Recognizer(sceptre::Language::English);
        assert_eq!(role_label(&role), "recognizer:english");
    }
}
