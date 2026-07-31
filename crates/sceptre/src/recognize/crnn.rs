//! CRNN forward pass through the inference backend.
//!
//! Runs the batched `[B, 1, 64, W]` tensor through the gen2 recognizer, yielding
//! CTC logits `[B, T, num_classes]`.
