//! Per-language character sets for CTC class mapping.
//!
//! The CTC class list is `['[blank]'] + characters`, so index 0 is the blank and
//! character `characters[i]` maps to class `i + 1`. Each gen2 recognizer has its
//! own exact character string (digits, symbols, space, then language glyphs), in a
//! model-specific order — embedded verbatim from `assets/character/*_g2.txt`, which
//! are the `characters` fields of EasyOCR's `config.py` recognition models. The
//! per-language `*_char.txt` letter lists are NOT the CTC charset (they omit the
//! digit/symbol prefix and use a different order), so we embed the full strings.

use crate::config::Language;

/// Raw gen2 charset strings, embedded at compile time (one per recognizer).
const ENGLISH_G2: &str = include_str!("../../assets/character/english_g2.txt");
const LATIN_G2: &str = include_str!("../../assets/character/latin_g2.txt");
const ZH_SIM_G2: &str = include_str!("../../assets/character/zh_sim_g2.txt");
const JAPANESE_G2: &str = include_str!("../../assets/character/japanese_g2.txt");
const KOREAN_G2: &str = include_str!("../../assets/character/korean_g2.txt");
const CYRILLIC_G2: &str = include_str!("../../assets/character/cyrillic_g2.txt");
const TELUGU_G2: &str = include_str!("../../assets/character/telugu_g2.txt");
const KANNADA_G2: &str = include_str!("../../assets/character/kannada_g2.txt");

/// An ordered gen2 alphabet plus the derived blank-prefixed CTC class mapping.
///
/// Class 0 is the CTC blank; class `i` (1-based) is `characters[i - 1]`.
#[derive(Debug, Clone)]
pub(crate) struct Charset {
    characters: Vec<char>,
}

impl Charset {
    /// Load the embedded gen2 alphabet for a language group.
    pub(crate) fn for_language(language: Language) -> Self {
        let raw = match language {
            Language::English => ENGLISH_G2,
            Language::Latin => LATIN_G2,
            Language::ChineseSimplified => ZH_SIM_G2,
            Language::Japanese => JAPANESE_G2,
            Language::Korean => KOREAN_G2,
            Language::Cyrillic => CYRILLIC_G2,
            Language::Telugu => TELUGU_G2,
            Language::Kannada => KANNADA_G2,
        };
        // Guard against a stray trailing newline (LF or CRLF) from asset tooling; ~keep
        // the gen2 charset itself never contains a newline or carriage return. ~keep
        let raw = raw.trim_end_matches(['\n', '\r']);
        Self {
            characters: raw.chars().collect(),
        }
    }

    /// Number of CTC classes, including the blank at index 0.
    pub(crate) fn num_classes(&self) -> usize {
        self.characters.len() + 1
    }

    /// The character for CTC class `class`, or `None` for the blank (class 0) or an
    /// out-of-range class.
    pub(crate) fn char_at_class(&self, class: usize) -> Option<char> {
        if class == 0 {
            return None;
        }
        self.characters.get(class - 1).copied()
    }

    /// The CTC class (1-based) of `ch`, or `None` if `ch` is not in this charset.
    pub(crate) fn class_of(&self, ch: char) -> Option<usize> {
        self.characters.iter().position(|&c| c == ch).map(|index| index + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_load_english_with_ninety_seven_classes() {
        let charset = Charset::for_language(Language::English);
        // 96 characters + 1 blank. ~keep
        assert_eq!(charset.num_classes(), 97);
    }

    #[test]
    fn should_map_class_zero_to_blank() {
        let charset = Charset::for_language(Language::English);
        assert_eq!(charset.char_at_class(0), None);
    }

    #[test]
    fn should_map_first_class_to_first_character() {
        let charset = Charset::for_language(Language::English);
        // english_g2 begins "0123456789...", so class 1 is '0'. ~keep
        assert_eq!(charset.char_at_class(1), Some('0'));
        assert_eq!(charset.class_of('0'), Some(1));
    }

    #[test]
    fn should_round_trip_class_and_character() {
        let charset = Charset::for_language(Language::English);
        for class in 1..charset.num_classes() {
            let ch = charset.char_at_class(class).expect("class in range has a char");
            assert_eq!(charset.class_of(ch), Some(class), "round-trip class {class}");
        }
    }

    #[test]
    fn should_return_none_for_out_of_range_class() {
        let charset = Charset::for_language(Language::English);
        assert_eq!(charset.char_at_class(charset.num_classes()), None);
    }

    #[test]
    fn should_load_every_language_with_expected_class_counts() {
        // characters + blank, per the gen2 `characters` strings in EasyOCR config.py. ~keep
        let expected = [
            (Language::English, 97),
            (Language::Latin, 352),
            (Language::ChineseSimplified, 6719),
            (Language::Japanese, 2215),
            (Language::Korean, 1009),
            (Language::Cyrillic, 208),
            (Language::Telugu, 166),
            (Language::Kannada, 168),
        ];
        for (language, classes) in expected {
            assert_eq!(Charset::for_language(language).num_classes(), classes, "{language:?}");
        }
    }
}
