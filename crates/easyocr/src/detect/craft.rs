//! CRAFT forward pass through the inference backend.
//!
//! Produces a `[H, W, 2]` heat-map — channel 0 is the region score, channel 1 the
//! link (affinity) score — at half the input resolution.
