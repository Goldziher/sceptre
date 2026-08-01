//! Command definitions and dispatch.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{CommandFactory, Parser, Subcommand};
use sceptre::{OcrConfig, ReadOptions, Reader};

use crate::output::{self, OutputFormat};
use crate::overrides::OcrOverrides;

/// CRAFT + gen2 CRNN optical character recognition over ONNX.
#[derive(Parser)]
#[command(name = "sceptre", version, about)]
pub struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Log level: error, warn, info, debug, or trace.
    #[arg(long, global = true, default_value = "warn", env = "EASYOCR_LOG")]
    log_level: String,
}

/// Model-management actions.
#[derive(Subcommand)]
pub enum ModelsAction {
    /// List the known models and their cache status.
    List {
        #[command(flatten)]
        overrides: OcrOverrides,
        /// Output format.
        #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
        format: OutputFormat,
    },
    /// Download the models for the configured languages.
    Download {
        #[command(flatten)]
        overrides: OcrOverrides,
        /// Output format.
        #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
        format: OutputFormat,
    },
}

#[derive(Subcommand)]
enum Commands {
    /// Run the full OCR pipeline over an image.
    Run {
        /// Path to the input image.
        image: PathBuf,
        #[command(flatten)]
        overrides: OcrOverrides,
        /// Output format.
        #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
        format: OutputFormat,
        /// Emit only the recognized text, omitting confidence and box detail.
        #[arg(long)]
        no_detail: bool,
    },
    /// Detect text regions only, without recognition.
    Detect {
        /// Path to the input image.
        image: PathBuf,
        #[command(flatten)]
        overrides: OcrOverrides,
        /// Output format.
        #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
        format: OutputFormat,
    },
    /// Recognize text in a pre-cropped line image.
    Recognize {
        /// Path to the cropped line image.
        image: PathBuf,
        #[command(flatten)]
        overrides: OcrOverrides,
        /// Output format.
        #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
        format: OutputFormat,
    },
    /// List or download models.
    Models {
        #[command(subcommand)]
        action: ModelsAction,
    },
    /// Print shell completions to stdout.
    Completions {
        /// Target shell.
        shell: clap_complete::Shell,
    },
    /// Run the MCP stdio server.
    #[cfg(feature = "mcp")]
    Mcp,
}

/// Build an `OcrConfig` from defaults with the CLI overrides applied.
fn config_from(overrides: &OcrOverrides) -> OcrConfig {
    let mut config = OcrConfig::default();
    overrides.apply(&mut config);
    config
}

/// Build a `Reader` from the CLI overrides.
fn build_reader(overrides: &OcrOverrides) -> Result<Reader> {
    Reader::builder()
        .config(config_from(overrides))
        .build()
        .context("building the OCR reader")
}

/// A color-aware stdout writer that strips escapes on non-TTY / `NO_COLOR`.
fn stdout() -> anstream::Stdout {
    anstream::stdout()
}

/// Run the full OCR pipeline and render the recognized lines.
fn run_ocr(image: PathBuf, overrides: OcrOverrides, format: OutputFormat, no_detail: bool) -> Result<()> {
    let reader = build_reader(&overrides)?;
    let options = ReadOptions { detail: !no_detail };
    let result = reader
        .readtext(&image, &options)
        .with_context(|| format!("running OCR over {image:?}"))?;
    output::render_result(&result, format, !no_detail, &mut stdout()).context("writing OCR results")?;
    Ok(())
}

/// Detect text regions and render their quads.
fn run_detect(image: PathBuf, overrides: OcrOverrides, format: OutputFormat) -> Result<()> {
    let reader = build_reader(&overrides)?;
    let image_data = sceptre::Image::from_path(&image).with_context(|| format!("loading {image:?}"))?;
    let quads = reader
        .detect(&image_data, &ReadOptions::default())
        .with_context(|| format!("detecting text regions in {image:?}"))?;
    output::render_quads(&quads, format, &mut stdout()).context("writing detected regions")?;
    Ok(())
}

/// Recognize a single cropped line and render it.
fn run_recognize(image: PathBuf, overrides: OcrOverrides, format: OutputFormat) -> Result<()> {
    let reader = build_reader(&overrides)?;
    let image_data = sceptre::Image::from_path(&image).with_context(|| format!("loading {image:?}"))?;
    let line = reader
        .recognize_line(&image_data, &ReadOptions::default())
        .with_context(|| format!("recognizing text in {image:?}"))?;
    output::render_line(&line, format, true, &mut stdout()).context("writing recognized line")?;
    Ok(())
}

/// Dispatch a `models` subcommand.
fn run_models(action: ModelsAction) -> Result<()> {
    match action {
        ModelsAction::List { overrides, format } => {
            let config = config_from(&overrides);
            let models = sceptre::model_manifest(&config).context("building the model manifest")?;
            output::render_models(&models, format, &mut stdout()).context("writing the model list")?;
        }
        ModelsAction::Download { overrides, format } => {
            let config = config_from(&overrides);
            let models = sceptre::download_models(&config).context("downloading models")?;
            output::render_models(&models, format, &mut stdout()).context("writing the model list")?;
        }
    }
    Ok(())
}

impl Cli {
    /// Initialize tracing to stderr from the configured log level.
    pub fn init_tracing(&self) {
        use tracing_subscriber::EnvFilter;
        let filter = EnvFilter::try_new(&self.log_level).unwrap_or_else(|_| EnvFilter::new("warn"));
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_writer(std::io::stderr)
            .init();
    }

    /// Dispatch the selected command.
    pub fn run(self) -> Result<()> {
        match self.command {
            Commands::Run {
                image,
                overrides,
                format,
                no_detail,
            } => run_ocr(image, overrides, format, no_detail),
            Commands::Detect {
                image,
                overrides,
                format,
            } => run_detect(image, overrides, format),
            Commands::Recognize {
                image,
                overrides,
                format,
            } => run_recognize(image, overrides, format),
            Commands::Models { action } => run_models(action),
            Commands::Completions { shell } => {
                let mut command = Cli::command();
                clap_complete::generate(shell, &mut command, "sceptre", &mut std::io::stdout());
                Ok(())
            }
            #[cfg(feature = "mcp")]
            Commands::Mcp => {
                let reader = build_reader(&OcrOverrides::default())?;
                sceptre::mcp::serve(reader).context("running the MCP server")
            }
        }
    }
}
