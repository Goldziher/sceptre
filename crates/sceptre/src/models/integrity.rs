//! Model artifact integrity verification shared by byte-backed providers.

use sha2::{Digest, Sha256};

use crate::error::{OcrError, Result};

/// Verify in-memory model bytes against a pinned SHA-256 digest.
pub(crate) fn verify_sha256_bytes(bytes: &[u8], expected_sha256: &str, model_name: &str) -> Result<()> {
    if expected_sha256.is_empty() {
        return Err(OcrError::model(format!(
            "model descriptor `{model_name}` has no sha256 pin"
        )));
    }
    let actual = sha256_hex(bytes);
    if actual.eq_ignore_ascii_case(expected_sha256) {
        return Ok(());
    }
    Err(OcrError::model(format!(
        "sha256 mismatch for `{model_name}`: expected {expected_sha256}, got {actual}"
    )))
}

fn sha256_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;

    const HEX_CHARS_PER_BYTE: usize = 2;
    let digest = Sha256::digest(bytes);
    let mut hex = String::with_capacity(digest.len() * HEX_CHARS_PER_BYTE);
    for byte in digest {
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_verify_known_sha256_digest() {
        verify_sha256_bytes(
            b"detector",
            "f2b3cbe41413047352141e5b863d87e696ec4f52b503040dba3a5700acd529a0",
            "detector",
        )
        .expect("known digest must verify");
    }

    #[test]
    fn should_reject_an_unpinned_byte_artifact() {
        let error = verify_sha256_bytes(b"model", "", "custom").expect_err("missing pin must fail closed");
        assert!(error.to_string().contains("has no sha256 pin"));
    }
}
