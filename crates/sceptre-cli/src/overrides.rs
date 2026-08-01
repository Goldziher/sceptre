//! CLI flags that override configuration, flattened into subcommands.

use clap::Args;

use sceptre::OcrConfig;

/// Lower bound (inclusive) for detection probability thresholds.
const MIN_PROBABILITY: f32 = 0.0;
/// Upper bound (inclusive) for detection probability thresholds.
const MAX_PROBABILITY: f32 = 1.0;

/// Parse and range-check a probability threshold at the clap layer.
///
/// Rejects `NaN` and any value outside `[MIN_PROBABILITY, MAX_PROBABILITY]`,
/// so out-of-range thresholds fail at parse time with a clear message.
fn parse_probability(raw: &str) -> core::result::Result<f32, String> {
    let value: f32 = raw.parse().map_err(|_| format!("`{raw}` is not a number"))?;
    if value.is_nan() || !(MIN_PROBABILITY..=MAX_PROBABILITY).contains(&value) {
        return Err(format!("must be between {MIN_PROBABILITY} and {MAX_PROBABILITY}"));
    }
    Ok(value)
}

/// Optional overrides applied on top of the loaded configuration.
#[derive(Debug, Default, Args)]
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
    #[arg(long, value_parser = parse_probability)]
    text_threshold: Option<f32>,

    /// Link confidence threshold for detection.
    #[arg(long, value_parser = parse_probability)]
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

#[cfg(test)]
mod tests {
    use super::parse_probability;

    #[test]
    fn should_accept_probabilities_within_the_unit_interval() {
        assert_eq!(parse_probability("0.0").expect("zero is valid"), 0.0);
        assert_eq!(parse_probability("1.0").expect("one is valid"), 1.0);
        assert_eq!(parse_probability("0.5").expect("a mid value is valid"), 0.5);
    }

    #[test]
    fn should_reject_out_of_range_nan_and_unparseable_probabilities() {
        assert!(parse_probability("-0.1").is_err(), "a negative value is rejected");
        assert!(parse_probability("1.1").is_err(), "a value above one is rejected");
        assert!(parse_probability("nan").is_err(), "NaN is rejected");
        assert!(parse_probability("abc").is_err(), "a non-numeric value is rejected");
    }
}
