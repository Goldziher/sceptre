//! Rendering of OCR results and model listings to text or JSON.

use std::io::{self, Write};
use std::path::{Path, PathBuf};

use anstyle::Style;
use clap::ValueEnum;
use sceptre::{ModelDescriptor, ModelInfo, ModelRole, OcrResult, OrtRuntimeInfo, Quad, RuntimeInfo, TextLine};

use crate::style;
use crate::timing::TimingsReport;

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

/// Text-mode placeholder for a runtime fact that could not be determined.
const UNKNOWN: &str = "unknown";

/// Serializable environment report: which build, backend, runtime and model pins.
///
/// The key names are a compatibility contract with `python/sceptre_rs_tools/benchmark.py`,
/// which stores this payload as a report's `metadata.environment` block. It reads
/// `version`, `backend`, `accelerator`, `onnxruntime` (taking its `version` field when the
/// value is an object), and `models[].{name, repo, sha256}`. Renaming any of those silently
/// nulls the corresponding provenance field in every published benchmark report.
#[derive(serde::Serialize)]
struct EnvironmentReport<'a> {
    version: &'a str,
    os: &'a str,
    arch: &'a str,
    backend: &'a str,
    /// The accelerator that actually registered, or `None` when it could not be
    /// determined — never the requested one, which would overstate what ran.
    accelerator: Option<&'a str>,
    accelerator_requested: &'a str,
    onnxruntime: Option<&'a OrtRuntimeInfo>,
    models: Vec<ModelPin>,
}

/// Serializable projection of [`ModelDescriptor`] (which is not itself `Serialize`).
#[derive(serde::Serialize)]
struct ModelPin {
    name: String,
    repo: String,
    revision: String,
    file: String,
    sha256: String,
    role: String,
}

/// Render the runtime environment and model pins as text or JSON.
pub fn render_environment(
    runtime: &RuntimeInfo,
    models: &[ModelDescriptor],
    format: OutputFormat,
    writer: &mut dyn Write,
) -> io::Result<()> {
    let report = environment_report(runtime, models);
    match format {
        OutputFormat::Json => write_json(&report, writer),
        OutputFormat::Text => write_environment_text(&report, writer),
    }
}

/// Project a [`RuntimeInfo`] and its model pins into the serializable report.
fn environment_report<'a>(runtime: &'a RuntimeInfo, models: &[ModelDescriptor]) -> EnvironmentReport<'a> {
    EnvironmentReport {
        version: runtime.sceptre_version,
        os: runtime.os,
        arch: runtime.arch,
        backend: runtime.backend.as_str(),
        accelerator: runtime.accelerator_registered.map(|accelerator| accelerator.as_str()),
        accelerator_requested: runtime.accelerator_requested.as_str(),
        onnxruntime: runtime.ort.as_ref(),
        models: models.iter().map(model_pin).collect(),
    }
}

/// Write the environment report as human-readable tab-separated rows.
fn write_environment_text(report: &EnvironmentReport<'_>, writer: &mut dyn Write) -> io::Result<()> {
    writeln!(writer, "{}", styled("ENVIRONMENT", style::heading()))?;
    writeln!(writer, "version\t{}", report.version)?;
    writeln!(writer, "platform\t{}/{}", report.os, report.arch)?;
    writeln!(writer, "backend\t{}", report.backend)?;
    writeln!(
        writer,
        "accelerator\t{}\t{}",
        report.accelerator.unwrap_or(UNKNOWN),
        styled(&format!("(requested {})", report.accelerator_requested), style::dim())
    )?;
    match report.onnxruntime {
        Some(ort) => {
            writeln!(
                writer,
                "onnxruntime\t{}\t{}",
                ort.version.as_deref().unwrap_or(UNKNOWN),
                styled(&format!("({})", ort.provisioning), style::dim())
            )?;
            if let Some(path) = &ort.dylib_path {
                writeln!(writer, "ort_dylib_path\t{path}")?;
            }
            writeln!(writer, "ort_build_info\t{}", ort.build_info)?;
        }
        None => writeln!(writer, "onnxruntime\t{}", styled(UNKNOWN, style::warning()))?,
    }
    writeln!(writer, "{}", styled("MODELS", style::heading()))?;
    for pin in &report.models {
        writeln!(writer, "{}\t{}\t{}\t{}", pin.name, pin.role, pin.sha256, pin.repo)?;
    }
    Ok(())
}

/// Project a [`ModelDescriptor`] into its serializable [`ModelPin`].
fn model_pin(descriptor: &ModelDescriptor) -> ModelPin {
    ModelPin {
        name: descriptor.name.clone(),
        repo: descriptor.repo.clone(),
        revision: descriptor.revision.clone(),
        file: descriptor.file.clone(),
        sha256: descriptor.sha256.clone(),
        role: role_label(&descriptor.role),
    }
}

/// A single result carrying the run's stage timings alongside its lines.
///
/// The result is flattened, so `--timings` only *adds* a `timings` key to the
/// object `--format json` already emitted.
#[derive(serde::Serialize)]
struct TimedResult<'a> {
    #[serde(flatten)]
    result: &'a OcrResult,
    timings: TimingsReport,
}

/// Render a full OCR result as text or JSON, optionally carrying stage timings.
pub fn render_result(
    result: &OcrResult,
    format: OutputFormat,
    detail: bool,
    timings: Option<TimingsReport>,
    writer: &mut dyn Write,
) -> io::Result<()> {
    match format {
        OutputFormat::Json => match timings {
            Some(timings) => write_json(&TimedResult { result, timings }, writer),
            None => write_json(result, writer),
        },
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

/// A batch carrying the run's stage timings alongside its per-image entries.
///
/// Timings are run-wide, so unlike [`TimedResult`] the array cannot absorb them and
/// `--timings` wraps it in an envelope instead.
#[derive(serde::Serialize)]
struct TimedBatch<'a> {
    images: Vec<BatchEntry<'a>>,
    timings: TimingsReport,
}

/// Render a batch of per-image OCR outcomes as text or JSON, optionally carrying stage timings.
///
/// JSON emits an array of `{ "image", "lines" }` (or `{ "image", "error" }`) objects in input
/// order — or, with `--timings`, a `{ "images", "timings" }` envelope around that array. Text
/// prints a per-image path header followed by that image's lines or its error, with a blank line
/// separating images.
pub fn render_batch(
    items: &[(PathBuf, ImageOutcome)],
    format: OutputFormat,
    detail: bool,
    timings: Option<TimingsReport>,
    writer: &mut dyn Write,
) -> io::Result<()> {
    match format {
        OutputFormat::Json => {
            let images: Vec<BatchEntry<'_>> = items.iter().map(|(path, outcome)| batch_entry(path, outcome)).collect();
            match timings {
                Some(timings) => write_json(&TimedBatch { images, timings }, writer),
                None => write_json(&images, writer),
            }
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
        render_result(&sample_result(), OutputFormat::Json, true, None, &mut buffer).unwrap();
        let text = String::from_utf8(buffer).unwrap();
        let value: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(value["lines"][0]["text"], "hello");
    }

    fn sample_timings() -> TimingsReport {
        TimingsReport {
            setup_ms: 100.0,
            detect_ms: 150.0,
            recognize_ms: 150.0,
            total_ms: 400.0,
        }
    }

    #[test]
    fn should_add_timings_alongside_lines_when_timed() {
        let mut buffer: Vec<u8> = Vec::new();
        render_result(
            &sample_result(),
            OutputFormat::Json,
            true,
            Some(sample_timings()),
            &mut buffer,
        )
        .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&buffer).unwrap();
        assert_eq!(value["lines"][0]["text"], "hello");
        assert_eq!(value["timings"]["detect_ms"], 150.0);
        assert_eq!(value["timings"]["total_ms"], 400.0);
    }

    #[test]
    fn should_wrap_batch_in_an_envelope_when_timed() {
        let items = vec![(PathBuf::from("a.png"), Ok(sample_result()))];
        let mut buffer: Vec<u8> = Vec::new();
        render_batch(&items, OutputFormat::Json, true, Some(sample_timings()), &mut buffer).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&buffer).unwrap();
        assert_eq!(value["images"][0]["image"], "a.png");
        assert_eq!(value["timings"]["setup_ms"], 100.0);
    }

    #[test]
    fn should_keep_batch_a_bare_array_when_untimed() {
        let items = vec![(PathBuf::from("a.png"), Ok(sample_result()))];
        let mut buffer: Vec<u8> = Vec::new();
        render_batch(&items, OutputFormat::Json, true, None, &mut buffer).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&buffer).unwrap();
        assert_eq!(value[0]["image"], "a.png");
    }

    #[test]
    fn should_render_result_text_without_detail_prints_only_text() {
        let mut buffer: Vec<u8> = Vec::new();
        render_result(&sample_result(), OutputFormat::Text, false, None, &mut buffer).unwrap();
        let text = String::from_utf8(buffer).unwrap();
        assert_eq!(text, "hello\n");
    }

    #[test]
    fn should_render_result_text_with_detail_includes_confidence_and_quad() {
        let mut buffer: Vec<u8> = Vec::new();
        render_result(&sample_result(), OutputFormat::Text, true, None, &mut buffer).unwrap();
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
                repo: "sceptre-ocr/craft_mlt_25k".to_string(),
                role: ModelRole::Detector,
                cached: true,
                path: None,
            },
            ModelInfo {
                name: "english_g2".to_string(),
                repo: "sceptre-ocr/english_g2".to_string(),
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
        assert!(text.contains("sceptre-ocr/craft_mlt_25k"), "missing repo: {text}");
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
