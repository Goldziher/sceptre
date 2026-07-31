//! Command definitions and dispatch.

use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::{CommandFactory, Parser, Subcommand, ValueEnum};

use crate::overrides::OcrOverrides;

/// CRAFT + gen2 CRNN optical character recognition over ONNX.
#[derive(Parser)]
#[command(name = "easyocr-rs", version, about)]
pub struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Log level: error, warn, info, debug, or trace.
    #[arg(long, global = true, default_value = "warn", env = "EASYOCR_LOG")]
    log_level: String,
}

/// Output serialization for recognition results.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum OutputFormat {
    /// Human-readable text.
    Text,
    /// JSON.
    Json,
}

/// Model-management actions.
#[derive(Subcommand)]
pub enum ModelsAction {
    /// List the known models and their cache status.
    List,
    /// Download the models for the configured languages.
    Download,
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
    },
    /// Detect text regions only, without recognition.
    Detect {
        /// Path to the input image.
        image: PathBuf,
    },
    /// Recognize text in a pre-cropped line image.
    Recognize {
        /// Path to the cropped line image.
        image: PathBuf,
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
            } => {
                let mut config = easyocr::OcrConfig::default();
                overrides.apply(&mut config);
                let _ = (image, format, config);
                bail!("`run` is not yet implemented")
            }
            Commands::Detect { image } => {
                let _ = image;
                bail!("`detect` is not yet implemented")
            }
            Commands::Recognize { image } => {
                let _ = image;
                bail!("`recognize` is not yet implemented")
            }
            Commands::Models { action } => match action {
                ModelsAction::List => bail!("`models list` is not yet implemented"),
                ModelsAction::Download => bail!("`models download` is not yet implemented"),
            },
            Commands::Completions { shell } => {
                let mut command = Cli::command();
                clap_complete::generate(shell, &mut command, "easyocr-rs", &mut std::io::stdout());
                Ok(())
            }
            #[cfg(feature = "mcp")]
            Commands::Mcp => bail!("`mcp` is not yet implemented"),
        }
    }
}
