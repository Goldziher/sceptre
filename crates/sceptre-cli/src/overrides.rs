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
    ///
    /// The explicit `id` lets sibling flags name this argument in a
    /// `conflicts_with` relation by its user-facing spelling.
    #[arg(id = "lang", long = "lang", value_enum)]
    languages: Vec<LanguageArg>,

    /// Maximum number of worker threads.
    #[arg(long)]
    threads: Option<usize>,

    /// Inference backend.
    #[arg(long, value_enum)]
    backend: Option<BackendArg>,

    /// Hardware accelerator for the inference backend (`ort` only).
    #[arg(long, value_enum)]
    accelerator: Option<AcceleratorArg>,

    /// Text confidence threshold for detection.
    #[arg(long, value_parser = parse_probability)]
    text_threshold: Option<f32>,

    /// Link confidence threshold for detection.
    #[arg(long, value_parser = parse_probability)]
    link_threshold: Option<f32>,

    /// Detection canvas size (longest side, px). Lower cuts peak memory and detection
    /// time on large pages at some accuracy cost; the default (2560) matches EasyOCR.
    #[arg(long)]
    canvas_size: Option<u32>,
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
    /// Telugu (`telugu_g2`).
    Telugu,
    /// Kannada (`kannada_g2`).
    Kannada,
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

/// Accelerator choices exposed on the command line.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum AcceleratorArg {
    /// Run on the CPU (the default, and what the published parity figures describe).
    Cpu,
    /// Best accelerator available on this platform, falling back to the CPU.
    Auto,
    /// Apple CoreML.
    Coreml,
    /// Microsoft DirectML.
    Directml,
    /// NVIDIA CUDA.
    Cuda,
}

/// Every supported recognition language, in the order declared by [`LanguageArg`].
///
/// Derived from the `ValueEnum` variant list, so a new language is picked up
/// without a second place to update.
pub fn every_language() -> Vec<sceptre::Language> {
    use clap::ValueEnum as _;

    LanguageArg::value_variants().iter().copied().map(Into::into).collect()
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
            LanguageArg::Telugu => Language::Telugu,
            LanguageArg::Kannada => Language::Kannada,
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

impl From<AcceleratorArg> for sceptre::Accelerator {
    fn from(value: AcceleratorArg) -> Self {
        use sceptre::Accelerator;
        match value {
            AcceleratorArg::Cpu => Accelerator::Cpu,
            AcceleratorArg::Auto => Accelerator::Auto,
            AcceleratorArg::Coreml => Accelerator::CoreMl,
            AcceleratorArg::Directml => Accelerator::DirectMl,
            AcceleratorArg::Cuda => Accelerator::Cuda,
        }
    }
}

impl OcrOverrides {
    /// Whether the caller named at least one language on the command line.
    ///
    /// Lets a subcommand distinguish "the default language set" from "these
    /// languages", which the flattened flags otherwise hide behind an empty vector.
    pub fn has_languages(&self) -> bool {
        !self.languages.is_empty()
    }

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
        if let Some(accelerator) = self.accelerator {
            config.model.accelerator = accelerator.into();
        }
        if let Some(text_threshold) = self.text_threshold {
            config.detection.text_threshold = text_threshold;
        }
        if let Some(link_threshold) = self.link_threshold {
            config.detection.link_threshold = link_threshold;
        }
        if let Some(canvas_size) = self.canvas_size {
            config.detection.canvas_size = canvas_size;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{AcceleratorArg, LanguageArg, every_language, parse_probability};

    #[test]
    fn should_map_every_accelerator_arg_to_its_library_wire_name() {
        use clap::ValueEnum as _;

        let expected = ["cpu", "auto", "coreml", "directml", "cuda"];
        let variants = AcceleratorArg::value_variants();
        assert_eq!(variants.len(), expected.len(), "every variant must be covered");
        for (variant, wire) in variants.iter().copied().zip(expected) {
            assert_eq!(sceptre::Accelerator::from(variant).as_str(), wire);
        }
    }

    #[test]
    fn should_expand_every_language_to_all_value_enum_variants() {
        use clap::ValueEnum as _;

        let languages = every_language();
        assert_eq!(languages.len(), LanguageArg::value_variants().len());
        assert!(languages.contains(&sceptre::Language::English));
        assert!(languages.contains(&sceptre::Language::Kannada));

        let mut deduplicated = languages.clone();
        deduplicated.dedup();
        assert_eq!(deduplicated.len(), languages.len(), "no language may repeat");
    }

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
