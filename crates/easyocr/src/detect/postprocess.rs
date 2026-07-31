//! CRAFT postprocessing: heat-maps to text boxes.
//!
//! Reference: EasyOCR `craft_utils.py` (`getDetBoxes_core`, `adjustResultCoordinates`).
//! Threshold the region/link maps, run connected components, fit min-area
//! rectangles, then scale coordinates back to input space (`ratio_net = 2`).
