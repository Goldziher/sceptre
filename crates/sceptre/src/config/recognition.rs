//! CRNN recognition + CTC decoding configuration.

use serde::{Deserialize, Serialize};

use crate::error::{OcrError, Result};

/// CTC decoding strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Decoder {
    /// Greedy (best-path) CTC decoding.
    #[default]
    Greedy,
    /// Beam-search CTC decoding (EasyOCR `decoder='beamsearch'`, `beamWidth` from
    /// [`RecognitionConfig::beam_width`]). Not a general accuracy win: measured on the
    /// tier-2 corpus it gained word-F1 on Japanese but lost it (net negative overall) on
    /// English and Korean by dropping characters mid-word — a length bias inherent to CTC
    /// beam search without a language model, confirmed (not just assumed) by testing and
    /// rejecting a targeted fix for one contributing cause. Opt in only where it is
    /// measured to help (see ADR 0036).
    BeamSearch,
    /// Dictionary-constrained word-beam-search. Not implemented: EasyOCR's variant needs
    /// per-language dictionaries and, for non-Latin separators, word segmentation that
    /// sceptre has no equivalent of; selecting it is a config error (ADR 0036).
    WordBeamSearch,
}

/// Parameters controlling recognition and decoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RecognitionConfig {
    /// CTC decoding strategy.
    pub decoder: Decoder,
    /// Beam width for beam-search decoders. EasyOCR default `5`.
    pub beam_width: usize,
    /// Recognition batch size. EasyOCR default `1`.
    pub batch_size: usize,
    /// Only these characters may be produced (empty = model charset).
    pub allowlist: String,
    /// These characters are never produced.
    pub blocklist: String,
    /// Contrast below which a low-confidence second pass runs. Default `0.1`.
    pub contrast_ths: f32,
    /// Target contrast for the adjustment pass. Default `0.5`.
    pub adjust_contrast: f32,
    /// Minimum confidence required for a recognized region to be emitted.
    /// Default `0.1`, chosen to suppress low-confidence noise without discarding
    /// the useful recognition results in the quality corpus.
    pub filter_ths: f32,
}

impl RecognitionConfig {
    /// Validate recognition settings consumed by the engine.
    pub(crate) fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.filter_ths) {
            return Err(OcrError::config(format!(
                "recognition.filter_ths must be finite and within [0, 1], got {}",
                self.filter_ths
            )));
        }
        if self.beam_width == 0 {
            return Err(OcrError::config("recognition.beam_width must be at least 1, got 0"));
        }
        Ok(())
    }
}

impl Default for RecognitionConfig {
    fn default() -> Self {
        Self {
            decoder: Decoder::Greedy,
            beam_width: 5,
            batch_size: 1,
            allowlist: String::new(),
            blocklist: String::new(),
            contrast_ths: 0.1,
            adjust_contrast: 0.5,
            filter_ths: 0.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_accept_filter_threshold_bounds() {
        for filter_ths in [0.0, 1.0] {
            let config = RecognitionConfig {
                filter_ths,
                ..RecognitionConfig::default()
            };

            config
                .validate()
                .expect("inclusive filter threshold bound should be valid");
        }
    }

    #[test]
    fn should_default_to_quality_corpus_filter_threshold() {
        assert_eq!(RecognitionConfig::default().filter_ths, 0.1);
    }

    #[test]
    fn should_reject_invalid_filter_thresholds() {
        for filter_ths in [-0.1, 1.1, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let config = RecognitionConfig {
                filter_ths,
                ..RecognitionConfig::default()
            };

            let error = config.validate().expect_err("invalid filter threshold should fail");
            assert!(matches!(error, OcrError::Config { .. }));
        }
    }

    #[test]
    fn should_default_to_greedy_decoding_with_easyocr_beam_width() {
        let config = RecognitionConfig::default();
        assert_eq!(config.decoder, Decoder::Greedy);
        assert_eq!(config.beam_width, 5);
    }

    #[test]
    fn should_reject_a_zero_beam_width() {
        let config = RecognitionConfig {
            beam_width: 0,
            ..RecognitionConfig::default()
        };
        let error = config.validate().expect_err("a zero beam width should fail");
        assert!(matches!(error, OcrError::Config { .. }));
    }
}
