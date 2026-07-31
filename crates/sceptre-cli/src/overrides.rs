//! CLI flags that override configuration, flattened into subcommands.

use clap::Args;

use sceptre::OcrConfig;

/// Optional overrides applied on top of the loaded configuration.
#[derive(Debug, Args)]
pub struct OcrOverrides {
    /// Recognition languages (repeatable), e.g. `--lang english --lang latin`.
    #[arg(long = "lang", value_enum)]
    languages: Vec<LanguageArg>,

    /// Maximum number of worker threads.
    #[arg(long)]
    threads: Option<usize>,

    /// Inference backend.
    #[arg(long, value_enum)]
    backend: Option<BackendArg>,

    /// Text confidence threshold for detection.
    #[arg(long)]
    text_threshold: Option<f32>,

    /// Link confidence threshold for detection.
    #[arg(long)]
    link_threshold: Option<f32>,
}

/// Language choices exposed on the command line.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum LanguageArg {
    /// English (`english_g2`).
    English,
    /// Latin-script (`latin_g2`).
    Latin,
    /// Simplified Chinese (`zh_sim_g2`).
    ChineseSimplified,
    /// Japanese (`japanese_g2`).
    Japanese,
    /// Korean (`korean_g2`).
    Korean,
    /// Cyrillic-script (`cyrillic_g2`).
    Cyrillic,
}

/// Backend choices exposed on the command line.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum BackendArg {
    /// Native ONNX Runtime.
    Ort,
    /// Pure-Rust ONNX.
    Tract,
    /// Pure-Rust native tensors.
    Candle,
}

impl From<LanguageArg> for sceptre::Language {
    fn from(value: LanguageArg) -> Self {
        use sceptre::Language;
        match value {
            LanguageArg::English => Language::English,
            LanguageArg::Latin => Language::Latin,
            LanguageArg::ChineseSimplified => Language::ChineseSimplified,
            LanguageArg::Japanese => Language::Japanese,
            LanguageArg::Korean => Language::Korean,
            LanguageArg::Cyrillic => Language::Cyrillic,
        }
    }
}

impl From<BackendArg> for sceptre::Backend {
    fn from(value: BackendArg) -> Self {
        use sceptre::Backend;
        match value {
            BackendArg::Ort => Backend::Ort,
            BackendArg::Tract => Backend::Tract,
            BackendArg::Candle => Backend::Candle,
        }
    }
}

impl OcrOverrides {
    /// Apply the set overrides onto `config`, leaving unset fields untouched.
    pub fn apply(&self, config: &mut OcrConfig) {
        if !self.languages.is_empty() {
            config.model.languages = self.languages.iter().copied().map(Into::into).collect();
        }
        if let Some(threads) = self.threads {
            config.concurrency.max_threads = Some(threads);
        }
        if let Some(backend) = self.backend {
            config.model.backend = backend.into();
        }
        if let Some(text_threshold) = self.text_threshold {
            config.detection.text_threshold = text_threshold;
        }
        if let Some(link_threshold) = self.link_threshold {
            config.detection.link_threshold = link_threshold;
        }
    }
}
