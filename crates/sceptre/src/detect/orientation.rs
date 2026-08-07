//! Whole-page orientation pre-pass.
//!
//! Probes 0/90/180/270° rotations of the page at a smaller canvas, scores each
//! by CRAFT's own region + link heat-map mass, and — when a rotation clears a
//! margin over the unrotated score — runs the real detection pass on that
//! rotation instead. Detected regions are then mapped back into the caller's
//! original coordinate frame: coordinates are inverse-rotated *and* each quad's
//! corners are cyclically re-indexed, so "clockwise from top-left" still means
//! top-left of the original image (not of the rotated probe frame). Getting the
//! re-indexing wrong would still produce a valid quadrilateral but crop it
//! sideways downstream. See ADR 0037 for the scoring function, its cost, and a
//! known false-positive risk on small, dense-glyph scripts.

use image::imageops::{rotate90, rotate180, rotate270};
use image::{ImageBuffer, Rgb};

use crate::error::{OcrError, Result};
use crate::inference::ModelBackend;
use crate::types::{Image, QUAD_CORNERS as REGION_CORNERS};

use super::craft::{self, HeatMaps};
use super::detector::DetectedRegion;
use super::preprocess;

/// One of the four axis-preserving whole-image rotations the pre-pass chooses
/// between. All rotations are clockwise, matching
/// `image::imageops::rotate90`/`rotate180`/`rotate270`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Rotation {
    Deg0,
    Deg90,
    Deg180,
    Deg270,
}

impl Rotation {
    /// All four candidate rotations, in the fixed order the probe scans them.
    /// [`pick_rotation`] tie-breaks toward the earliest entry, so this order is
    /// also the tie-break priority (favoring `Deg0`, then `Deg90`, ...).
    pub(super) const ALL: [Rotation; 4] = [Rotation::Deg0, Rotation::Deg90, Rotation::Deg180, Rotation::Deg270];
}

/// Rotate `image` clockwise by `rotation`, returning a new owned image (`Deg0`
/// clones the input unchanged). `Deg90`/`Deg270` swap width and height.
pub(super) fn rotate_image(image: &Image, rotation: Rotation) -> Result<Image> {
    if rotation == Rotation::Deg0 {
        return Ok(image.clone());
    }
    let width = image.width();
    let height = image.height();
    let buffer = ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, image.as_rgb8())
        .ok_or_else(|| OcrError::image("failed to build RGB image view for orientation rotation"))?;
    let rotated = match rotation {
        Rotation::Deg90 => rotate90(&buffer),
        Rotation::Deg180 => rotate180(&buffer),
        Rotation::Deg270 => rotate270(&buffer),
        Rotation::Deg0 => unreachable!("Deg0 returns above"),
    };
    let (rotated_width, rotated_height) = rotated.dimensions();
    Image::from_rgb8(rotated_width, rotated_height, rotated.into_raw())
}

/// CRAFT region+link heat-map "mass" for one candidate rotation: the sum of
/// region-score values above `low_text` plus the sum of link-score values above
/// `link_threshold`, normalized by pixel count so probes of different
/// (rotation-dependent) shapes are comparable.
///
/// The link (affinity) term is what makes this discriminate 180° rotations: an
/// upside-down line of text still looks like "text" to CRAFT's region head (it
/// responds to stroke density more than glyph orientation), but the affinity
/// head is trained on horizontally-flowing character pairs and its response
/// drops for any of the three wrong rotations, including 180° (measured on the
/// tier-2 corpus, see ADR 0037).
fn heat_mass_score(heat: &HeatMaps, low_text: f32, link_threshold: f32) -> f32 {
    let total = heat.region.len().max(1) as f32;
    let region_mass: f32 = heat.region.iter().filter(|&&value| value > low_text).sum();
    let link_mass: f32 = heat.link.iter().filter(|&&value| value > link_threshold).sum();
    (region_mass + link_mass) / total
}

/// Floor added to the `Deg0` baseline before applying `margin`, so a
/// near-blank page (baseline mass ~0) doesn't flip on noise alone.
const BASELINE_FLOOR: f32 = 1e-6;

/// Choose the best-scoring rotation, requiring a non-zero rotation to beat the
/// `Deg0` score by at least `margin` (a relative improvement) before it
/// overrides — an outright tie, or a marginal win, keeps the page as-is. Among
/// the non-zero rotations, an earlier entry in [`Rotation::ALL`] wins a tie.
fn pick_rotation(scores: [f32; 4], margin: f32) -> Rotation {
    let baseline = scores[0];
    let mut best = Rotation::Deg0;
    let mut best_score = baseline;
    for (rotation, &score) in Rotation::ALL.iter().zip(scores.iter()).skip(1) {
        if score > best_score {
            best_score = score;
            best = *rotation;
        }
    }
    let threshold = baseline.max(BASELINE_FLOOR) * (1.0 + margin);
    if best != Rotation::Deg0 && best_score > threshold {
        best
    } else {
        Rotation::Deg0
    }
}

/// Probe all four rotations of `image` at `probe_canvas` and pick the
/// best-scoring one per [`pick_rotation`].
pub(super) fn select_rotation(
    backend: &dyn ModelBackend,
    image: &Image,
    probe_canvas: u32,
    low_text: f32,
    link_threshold: f32,
    margin: f32,
) -> Result<Rotation> {
    let mut scores = [0.0f32; 4];
    for (index, rotation) in Rotation::ALL.into_iter().enumerate() {
        let rotated = rotate_image(image, rotation)?;
        let prepared = preprocess::prepare_with_canvas(&rotated, probe_canvas, 1.0, None)?;
        let heat = craft::run_craft(backend, prepared.tensor)?;
        scores[index] = heat_mass_score(&heat, low_text, link_threshold);
    }
    Ok(pick_rotation(scores, margin))
}

/// Map one region's corners, detected in the rotated probe frame, back into the
/// original coordinate frame. Two things happen together: the coordinates are
/// inverse-rotated, and the corners are re-indexed so index 0 is top-left *of
/// the original image* again — rotating the whole page clockwise cycles which
/// physical corner counts as "top-left", so undoing the rotation must cycle it
/// back. Getting only the coordinates right (and not the index order) would
/// still yield a valid quadrilateral, but recognition would crop it sideways or
/// upside-down. Re-indexing is done by re-deriving "clockwise from top-left"
/// fresh in the original frame (see [`clockwise_from_top_left`]) rather than a
/// fixed per-rotation offset, so it holds for any quad shape, not just the
/// elongated axis-aligned/free boxes this pipeline actually produces.
fn unrotate_corners(
    corners: [[f32; 2]; REGION_CORNERS],
    rotation: Rotation,
    original_width: u32,
    original_height: u32,
) -> [[f32; 2]; REGION_CORNERS] {
    // A true no-op: Deg0 must return `corners` completely unchanged, not just with
    // identity coordinates. Re-deriving "clockwise from top-left" is only meaningful
    // after an actual rotation — running it unconditionally would silently reorder a
    // free (non-axis-aligned) quad whenever its existing order didn't already happen
    // to start at the minimum-y corner, corrupting the perspective-crop even when the
    // page was never rotated at all. ~keep
    let inverse: fn([f32; 2], f32, f32) -> [f32; 2] = match rotation {
        Rotation::Deg0 => return corners,
        Rotation::Deg90 => |[x, y], _width, height| [y, height - x],
        Rotation::Deg180 => |[x, y], width, height| [width - x, height - y],
        Rotation::Deg270 => |[x, y], width, _height| [width - y, x],
    };
    let width = original_width as f32;
    let height = original_height as f32;
    let transformed = corners.map(|point| inverse(point, width, height));
    clockwise_from_top_left(transformed)
}

/// Cyclically rotate an already clockwise-wound quad so index 0 becomes its
/// top-left-most corner (minimum `y`, ties broken by minimum `x`) — the
/// convention [`DetectedRegion::corners`] documents. A rigid rotation, forward
/// or inverse, never flips winding direction, so a cyclic rotation (not a full
/// re-sort) is enough to restore the convention after coordinates move frames.
fn clockwise_from_top_left(corners: [[f32; 2]; REGION_CORNERS]) -> [[f32; 2]; REGION_CORNERS] {
    let start = corners
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| (a[1], a[0]).partial_cmp(&(b[1], b[0])).expect("finite coordinates"))
        .map(|(index, _)| index)
        .expect("exactly four corners");
    std::array::from_fn(|i| corners[(start + i) % REGION_CORNERS])
}

/// Map every region's corners back into the original coordinate frame (see
/// [`unrotate_corners`]); a no-op for `Deg0`.
pub(super) fn unrotate_regions(
    regions: Vec<DetectedRegion>,
    rotation: Rotation,
    original_width: u32,
    original_height: u32,
) -> Vec<DetectedRegion> {
    regions
        .into_iter()
        .map(|region| DetectedRegion {
            corners: unrotate_corners(region.corners, rotation, original_width, original_height),
            axis_aligned: region.axis_aligned,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn solid_image(width: u32, height: u32, rgb: [u8; 3]) -> Image {
        let mut pixels = Vec::with_capacity((width * height * 3) as usize);
        for _ in 0..(width * height) {
            pixels.extend_from_slice(&rgb);
        }
        Image::from_rgb8(width, height, pixels).expect("valid rgb buffer")
    }

    #[test]
    fn should_leave_dimensions_and_pixels_unchanged_at_deg0() {
        let image = solid_image(5, 3, [10, 20, 30]);
        let rotated = rotate_image(&image, Rotation::Deg0).expect("rotate");
        assert_eq!(rotated, image);
    }

    #[test]
    fn should_swap_dimensions_for_deg90_and_deg270() {
        let image = solid_image(7, 4, [1, 2, 3]);
        for rotation in [Rotation::Deg90, Rotation::Deg270] {
            let rotated = rotate_image(&image, rotation).expect("rotate");
            assert_eq!(rotated.width(), 4, "{rotation:?} must swap to original height");
            assert_eq!(rotated.height(), 7, "{rotation:?} must swap to original width");
        }
    }

    #[test]
    fn should_keep_dimensions_for_deg180() {
        let image = solid_image(7, 4, [1, 2, 3]);
        let rotated = rotate_image(&image, Rotation::Deg180).expect("rotate");
        assert_eq!(rotated.width(), 7);
        assert_eq!(rotated.height(), 4);
    }

    #[test]
    fn should_move_a_corner_pixel_clockwise_under_deg90() {
        // A 2x1 image: pixel (0,0) red, pixel (1,0) blue. Rotated 90 clockwise ~keep
        // becomes 1x2 with the original top-left pixel now at the top-right, i.e. ~keep
        // row 0 (red moves to (0,0) since the 1-wide output only has column 0). ~keep
        let mut pixels = Vec::with_capacity(6);
        pixels.extend_from_slice(&[255, 0, 0]); // (0,0) red
        pixels.extend_from_slice(&[0, 0, 255]); // (1,0) blue
        let image = Image::from_rgb8(2, 1, pixels).expect("valid rgb buffer");

        let rotated = rotate_image(&image, Rotation::Deg90).expect("rotate");

        assert_eq!((rotated.width(), rotated.height()), (1, 2));
        // Clockwise: the left column of the source becomes the top row of the result. ~keep
        assert_eq!(&rotated.as_rgb8()[0..3], &[255, 0, 0]);
        assert_eq!(&rotated.as_rgb8()[3..6], &[0, 0, 255]);
    }

    /// A genuinely asymmetric (non-rectangular) quad, clockwise from top-left,
    /// with distinct `y` on every corner so "top-left" is unambiguous — used to
    /// prove [`unrotate_corners`] both relocates *and* re-indexes correctly.
    /// Sits inside a `100 x 50` original image.
    const ORIGINAL: [[f32; 2]; 4] = [[10.0, 5.0], [40.0, 8.0], [35.0, 22.0], [12.0, 25.0]];
    const ORIGINAL_WIDTH: u32 = 100;
    const ORIGINAL_HEIGHT: u32 = 50;

    /// Forward-rotate one point the way `image::imageops::rotate90` et al. would
    /// move the pixel at that position (see the module docs' derivation).
    fn forward_point(rotation: Rotation, [x, y]: [f32; 2], width: f32, height: f32) -> [f32; 2] {
        match rotation {
            Rotation::Deg0 => [x, y],
            Rotation::Deg90 => [height - y, x],
            Rotation::Deg180 => [width - x, height - y],
            Rotation::Deg270 => [y, width - x],
        }
    }

    #[test]
    fn should_round_trip_asymmetric_quad_through_every_rotation() {
        for rotation in Rotation::ALL {
            let mapped =
                ORIGINAL.map(|point| forward_point(rotation, point, ORIGINAL_WIDTH as f32, ORIGINAL_HEIGHT as f32));
            // A rigid rotation preserves winding, so mimicking what CRAFT's own corner ~keep
            // assignment would see in the rotated frame only needs re-deriving the new ~keep
            // top-left start, via the same primitive `unrotate_corners` itself uses. ~keep
            let rotated_frame_corners = clockwise_from_top_left(mapped);

            let recovered = unrotate_corners(rotated_frame_corners, rotation, ORIGINAL_WIDTH, ORIGINAL_HEIGHT);

            for (index, (expected, actual)) in ORIGINAL.iter().zip(recovered.iter()).enumerate() {
                assert!(
                    (expected[0] - actual[0]).abs() < 1e-3 && (expected[1] - actual[1]).abs() < 1e-3,
                    "{rotation:?} corner {index}: expected {expected:?}, got {actual:?}"
                );
            }
        }
    }

    #[test]
    fn should_preserve_axis_aligned_flag_when_unrotating_regions() {
        let regions = vec![
            DetectedRegion {
                corners: ORIGINAL,
                axis_aligned: true,
            },
            DetectedRegion {
                corners: ORIGINAL,
                axis_aligned: false,
            },
        ];

        let unrotated = unrotate_regions(regions, Rotation::Deg90, ORIGINAL_WIDTH, ORIGINAL_HEIGHT);

        assert!(unrotated[0].axis_aligned);
        assert!(!unrotated[1].axis_aligned);
    }

    #[test]
    fn should_leave_a_free_quads_corner_order_untouched_at_deg0() {
        // ORIGINAL's corners already start at its own minimum-y point, which would
        // hide a bug that reorders unconditionally; rotate the array so index 0 is
        // deliberately *not* the minimum-y corner, the way a free quad from
        // `min_area_rect` need not happen to start there either. At Deg0 this must
        // come back byte-for-byte identical — reordering it would corrupt an
        // already-correct perspective crop on a page that was never rotated. ~keep
        let shuffled = [ORIGINAL[2], ORIGINAL[3], ORIGINAL[0], ORIGINAL[1]];

        let recovered = unrotate_corners(shuffled, Rotation::Deg0, ORIGINAL_WIDTH, ORIGINAL_HEIGHT);

        assert_eq!(recovered, shuffled);
    }

    fn heat_maps(region_value: f32, link_value: f32, shape: (usize, usize)) -> HeatMaps {
        HeatMaps {
            region: Array2::from_elem(shape, region_value),
            link: Array2::from_elem(shape, link_value),
        }
    }

    #[test]
    fn should_score_higher_activation_as_higher_mass() {
        let low = heat_maps(0.1, 0.1, (4, 4));
        let high = heat_maps(0.9, 0.9, (4, 4));

        assert!(heat_mass_score(&high, 0.4, 0.4) > heat_mass_score(&low, 0.4, 0.4));
    }

    #[test]
    fn should_ignore_values_at_or_below_threshold() {
        let heat = heat_maps(0.4, 0.4, (4, 4));
        assert_eq!(heat_mass_score(&heat, 0.4, 0.4), 0.0);
    }

    #[test]
    fn should_keep_deg0_when_no_rotation_clears_the_margin() {
        // Deg90 edges Deg0 out by only 2%, under a 5% margin. ~keep
        let scores = [1.00, 1.02, 0.90, 0.80];
        assert_eq!(pick_rotation(scores, 0.05), Rotation::Deg0);
    }

    #[test]
    fn should_switch_when_a_rotation_clears_the_margin() {
        // Deg270 beats Deg0 by 20%, clearing a 5% margin. ~keep
        let scores = [1.00, 0.90, 0.95, 1.20];
        assert_eq!(pick_rotation(scores, 0.05), Rotation::Deg270);
    }

    #[test]
    fn should_favor_deg0_on_an_exact_tie() {
        let scores = [1.00, 1.00, 1.00, 1.00];
        assert_eq!(pick_rotation(scores, 0.0), Rotation::Deg0);
    }

    #[test]
    fn should_not_flip_a_near_blank_page_on_noise() {
        // Deg0 is ~0; Deg90's tiny absolute score would clear a naive relative ~keep
        // margin against a raw-zero baseline, but the floor keeps it at Deg0. ~keep
        let scores = [0.0000001, 0.000001, 0.0, 0.0];
        assert_eq!(pick_rotation(scores, 0.05), Rotation::Deg0);
    }
}

/// Real-model regression coverage: does [`select_rotation`] pick the correcting
/// rotation on the `test_documents` corpus's known-rotated images, and does it
/// leave the tier-2 parity corpus alone? Requires the real CRAFT ONNX model
/// (`ort` + `download`, cached per ADR 0017) and the `test_documents` submodule,
/// so every test here is `#[ignore]`d by default — run with
/// `cargo test -p sceptre --features ort-bundled,download -- --ignored`. See
/// ADR 0037 for the measured numbers this encodes.
#[cfg(all(test, feature = "ort", feature = "download"))]
mod real_model_selection {
    use super::*;
    use crate::config::DetectionConfig;
    use std::path::{Path, PathBuf};

    fn craft_backend() -> Box<dyn ModelBackend> {
        let entry = crate::models::registry::craft_entry();
        let path = crate::models::download::ensure(&entry, None, None).expect("craft model cached");
        let bytes = std::fs::read(path).expect("read craft model bytes");
        let options = crate::inference::BackendOptions {
            threads: 1,
            ..Default::default()
        };
        crate::inference::load_backend(crate::config::Backend::Ort, &bytes, options).expect("load craft backend")
    }

    fn images_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("repo root two levels up from the crate manifest dir")
            .join("test_documents/images")
    }

    fn pick(backend: &dyn ModelBackend, name: &str, config: &DetectionConfig) -> Rotation {
        let image = Image::from_path(images_dir().join(name)).expect("decode corpus image");
        select_rotation(
            backend,
            &image,
            config.orientation_probe_canvas_size,
            config.low_text,
            config.link_threshold,
            config.orientation_margin,
        )
        .expect("probe succeeds")
    }

    #[test]
    #[ignore = "requires the real cached CRAFT model; run with --ignored"]
    fn should_pick_the_correcting_rotation_on_known_rotated_pages() {
        let backend = craft_backend();
        let config = DetectionConfig::default();
        // Expected rotation is the one that makes each corrupted page upright again
        // (verified independently against ground truth with ImageMagick `-rotate`,
        // not derived from this code): a page named `_rotated_N` needs `360 - N`. ~keep
        let cases = [
            ("ocr_test_rotated_90.png", Rotation::Deg270),
            ("ocr_test_rotated_180.png", Rotation::Deg180),
            ("ocr_test_rotated_270.png", Rotation::Deg90),
            ("complex_document_rotated_90.png", Rotation::Deg270),
            ("complex_document_rotated_180.png", Rotation::Deg180),
            ("complex_document_rotated_270.png", Rotation::Deg90),
        ];
        for (name, expected) in cases {
            assert_eq!(pick(backend.as_ref(), name, &config), expected, "{name}");
        }
    }

    #[test]
    #[ignore = "requires the real cached CRAFT model; run with --ignored"]
    fn should_leave_the_upright_tier2_parity_corpus_unrotated() {
        let backend = craft_backend();
        let config = DetectionConfig::default();
        // Every tier-2 golden parity image except `kannada.png` (see the next test): ~keep
        // these must never flip, or `tier2_golden` stops being byte-identical. ~keep
        let names = [
            "english.png",
            "french.jpg",
            "chinese.jpg",
            "japanese.jpg",
            "korean.png",
            "cyrillic.png",
            "telugu.png",
        ];
        for name in names {
            assert_eq!(pick(backend.as_ref(), name, &config), Rotation::Deg0, "{name}");
        }
    }

    #[test]
    #[ignore = "requires the real cached CRAFT model; run with --ignored"]
    fn should_document_the_known_kannada_false_positive_when_enabled() {
        // Honest limitation, not a regression: kannada.png's short lines and dense,
        // loopy glyph strokes give it an unusually high baseline CRAFT activation in
        // every orientation, so the margin gate that protects the other seven parity
        // images doesn't protect this one — it flips to Deg270 by a ~12% margin, over
        // the 5% default. This is why `detect_orientation` defaults to off: with the
        // pre-pass disabled by default, this case never affects a real pipeline run.
        // A future scorer change that fixes this may turn this into a `should_leave_*`
        // case instead; if it starts failing because kannada stopped flipping, that's
        // good news — update the test rather than treating it as a break. See ADR 0037.
        let backend = craft_backend();
        let config = DetectionConfig::default();
        assert_eq!(pick(backend.as_ref(), "kannada.png", &config), Rotation::Deg270);
    }
}
