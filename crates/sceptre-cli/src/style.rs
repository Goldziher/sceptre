//! Terminal styling palette.
//!
//! Uses `anstyle` styles rendered through `anstream`, which honors `NO_COLOR`
//! and non-TTY output automatically.

// Palette consumed by the command handlers as their output is implemented. ~keep
#![allow(dead_code)]

use anstyle::{AnsiColor, Style};

/// Style for headings and section labels.
pub fn heading() -> Style {
    Style::new().bold().fg_color(Some(AnsiColor::Cyan.into()))
}

/// Style for success messages.
pub fn success() -> Style {
    Style::new().fg_color(Some(AnsiColor::Green.into()))
}

/// Style for warnings.
pub fn warning() -> Style {
    Style::new().fg_color(Some(AnsiColor::Yellow.into()))
}

/// Style for errors.
pub fn error() -> Style {
    Style::new().bold().fg_color(Some(AnsiColor::Red.into()))
}

/// Dimmed style for secondary detail.
pub fn dim() -> Style {
    Style::new().dimmed()
}
