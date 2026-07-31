//! Crop each detected region and resize to the recognizer's input height.
//!
//! Reference: EasyOCR `utils.py` (`get_image_list`, `four_point_transform`,
//! `compute_ratio_and_resize`). Free quads use a perspective transform; every
//! crop is resized to height 64 preserving aspect ratio.
