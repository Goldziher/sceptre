//! Per-language character sets for CTC class mapping.
//!
//! The CTC class list is `['[blank]'] + characters`, so index 0 is the blank and
//! character `c` maps to index `i + 1`. Alphabets are embedded from
//! `assets/character/*.txt`, copied from EasyOCR.

use crate::config::Language;

/// An ordered alphabet plus the derived blank-prefixed CTC class list.
#[derive(Debug, Clone, Default)]
pub struct Charset {
    /// Ordered characters (index `i` → CTC class `i + 1`).
    pub characters: Vec<char>,
}

impl Charset {
    /// Number of CTC classes, including the blank at index 0.
    pub fn num_classes(&self) -> usize {
        self.characters.len() + 1
    }

    /// Load the embedded alphabet for a language group.
    pub fn for_language(_language: Language) -> Self {
        todo!("load the embedded alphabet for the language group")
    }
}
