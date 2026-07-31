//! CTC decoding.
//!
//! Reference: EasyOCR `utils.py` (`CTCLabelConverter.decode_greedy`). Blank is
//! index 0; greedy decoding collapses repeats and drops blanks. Confidence uses
//! the `custom_mean` formula from `recognition.py`.
