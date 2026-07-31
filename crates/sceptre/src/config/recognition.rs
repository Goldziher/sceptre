//! CRNN recognition + CTC decoding configuration.

use serde::{Deserialize, Serialize};

/// CTC decoding strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Decoder {
    /// Greedy (best-path) CTC decoding.
    #[default]
    Greedy,
    /// Beam-search CTC decoding.
    BeamSearch,
    /// Dictionary-constrained word-beam-search.
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
    /// Accepted for EasyOCR parity. Default `0.003`. Upstream threads this through
    /// `get_text` but never applies it, so like upstream it currently has no effect
    /// on the output (every recognized region is emitted regardless of confidence).
    pub filter_ths: f32,
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
            filter_ths: 0.003,
        }
    }
}
