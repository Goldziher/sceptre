//! CRAFT postprocessing: heat-maps to text boxes.
//!
//! Reference: EasyOCR `craft_utils.py` (`getDetBoxes_core`, `adjustResultCoordinates`).
//! Threshold the region/link maps, run connected components, fit min-area
//! rectangles, then scale coordinates back to input space (`ratio_net = 2`).
//!
//! Connected-component labelling is provided by the `imageproc` crate (see ADR
//! 0013); the CRAFT dilation is a custom cv2-exact box kernel (see ADR 0018,
//! superseding the imageproc dilation); minimum-area rectangle fitting is our own
//! OpenCV-faithful rotating-calipers implementation ([`super::min_area_rect`], see
//! ADR 0039, superseding ADR 0013's min-area-rect half).

use image::{GrayImage, ImageBuffer, Luma};
use imageproc::point::Point;
use imageproc::region_labelling::{Connectivity, connected_components};
use ndarray::Array2;

use super::min_area_rect::min_area_rect;
use crate::error::{OcrError, Result};

/// Ratio between the CRAFT heat-map resolution and the network input: the
/// heat-maps are emitted at half resolution, so box coordinates are scaled by
/// this factor (mirrors EasyOCR `ratio_net = 2`).
const RATIO_NET: f32 = 2.0;

/// Minimum connected-component pixel area; smaller components are discarded
/// (mirrors EasyOCR's `if size < 10: continue`).
const MIN_COMPONENT_AREA: u32 = 10;

/// A box is treated as a diamond and re-aligned to an axis-aligned rectangle
/// when its edge-length ratio is within this tolerance of 1.
const DIAMOND_RATIO_TOLERANCE: f32 = 0.1;

/// Small epsilon guarding the box-ratio division against a zero shorter edge.
const BOX_RATIO_EPSILON: f32 = 1e-5;

/// Foreground value written into the single-component segmentation map.
const SEGMAP_FOREGROUND: u8 = 255;

/// Multiplier applied inside the dilation-iteration count formula.
const NITER_SCALE: f32 = 2.0;

/// Kernel-size offset in EasyOCR's dilation: the structuring element is
/// `(1 + niter)` square (`cv2.getStructuringElement(MORPH_RECT, (1 + niter, ...))`).
const DILATION_KERNEL_OFFSET: u32 = 1;

/// A detected box: 4 corners `[x, y]`, clockwise from top-left, in heat-map space
/// until [`adjust_coordinates`] scales them to original-image space.
pub(super) type BoxPoints = [[f32; 2]; 4];

/// Connected-component label image produced by [`connected_components`].
type LabelImage = ImageBuffer<Luma<u32>, Vec<u32>>;

/// Per-label statistics gathered in a single pass over the label image, mirroring
/// the fields EasyOCR reads from `cv2.connectedComponentsWithStats`.
#[derive(Clone, Copy)]
struct LabelStat {
    /// Number of pixels carrying this label (`CC_STAT_AREA`).
    area: u32,
    /// Left/top/right/bottom pixel extent of the component's bounding box.
    min_x: u32,
    min_y: u32,
    max_x: u32,
    max_y: u32,
    /// Maximum region score over the component's pixels (for the text threshold).
    max_region: f32,
}

impl LabelStat {
    fn new(x: u32, y: u32, region_value: f32) -> Self {
        Self {
            area: 1,
            min_x: x,
            min_y: y,
            max_x: x,
            max_y: y,
            max_region: region_value,
        }
    }

    fn update(&mut self, x: u32, y: u32, region_value: f32) {
        self.area += 1;
        self.min_x = self.min_x.min(x);
        self.min_y = self.min_y.min(y);
        self.max_x = self.max_x.max(x);
        self.max_y = self.max_y.max(y);
        self.max_region = self.max_region.max(region_value);
    }

    /// Component bounding-box width in pixels.
    fn width(&self) -> u32 {
        self.max_x - self.min_x + 1
    }

    /// Component bounding-box height in pixels.
    fn height(&self) -> u32 {
        self.max_y - self.min_y + 1
    }
}

/// Extract rotated text boxes from the region/link heat-maps.
/// `region`/`link` are `[H, W]` score maps in `[0, 1]`.
pub(super) fn get_det_boxes(
    region: &Array2<f32>,
    link: &Array2<f32>,
    text_threshold: f32,
    link_threshold: f32,
    low_text: f32,
) -> Result<Vec<BoxPoints>> {
    if region.dim() != link.dim() {
        return Err(OcrError::inference(
            "region and link heat-maps must share the same shape",
        ));
    }

    let text_score = region.mapv(|value| value > low_text);
    let link_score = link.mapv(|value| value > link_threshold);
    let comb = build_comb_image(&text_score, &link_score)?;
    let labels = connected_components(&comb, Connectivity::Four, Luma([0u8]));
    let stats = compute_label_stats(&labels, region);

    let mut boxes = Vec::new();
    for (index, stat) in stats.iter().enumerate() {
        if stat.area < MIN_COMPONENT_AREA || stat.max_region < text_threshold {
            continue;
        }
        let label = index as u32 + 1;
        if let Some(detected) = extract_box(label, &labels, &text_score, &link_score, stat)? {
            boxes.push(detected);
        }
    }
    Ok(boxes)
}

/// Scale heat-map-space boxes to original-image space, in place.
/// Multiplies every coordinate by `inv_ratio * RATIO_NET` (RATIO_NET = 2).
pub(super) fn adjust_coordinates(boxes: &mut [BoxPoints], inv_ratio: f32) {
    let scale = inv_ratio * RATIO_NET;
    for detected in boxes.iter_mut() {
        for corner in detected.iter_mut() {
            corner[0] *= scale;
            corner[1] *= scale;
        }
    }
}

/// Build the binary `text_score + link_score` combination image (clipped to 0/1),
/// stored as a `GrayImage` with foreground pixels at [`SEGMAP_FOREGROUND`].
fn build_comb_image(text_score: &Array2<bool>, link_score: &Array2<bool>) -> Result<GrayImage> {
    let (height, width) = text_score.dim();
    let mut buffer = Vec::with_capacity(width * height);
    for (&text, &link) in text_score.iter().zip(link_score.iter()) {
        buffer.push(if text || link { SEGMAP_FOREGROUND } else { 0 });
    }
    GrayImage::from_raw(width as u32, height as u32, buffer)
        .ok_or_else(|| OcrError::inference("failed to build connected-component input image"))
}

/// Gather per-label statistics in a single pass. Background (label 0) is skipped.
/// `connected_components` emits dense labels `1..=count`, so the result is indexed
/// by `label - 1` in ascending label order (avoiding per-pixel map lookups).
fn compute_label_stats(labels: &LabelImage, region: &Array2<f32>) -> Vec<LabelStat> {
    let mut stats: Vec<Option<LabelStat>> = Vec::new();
    for (x, y, pixel) in labels.enumerate_pixels() {
        let label = pixel[0];
        if label == 0 {
            continue;
        }
        let index = (label - 1) as usize;
        if index >= stats.len() {
            stats.resize(index + 1, None);
        }
        let region_value = region[[y as usize, x as usize]];
        match &mut stats[index] {
            Some(stat) => stat.update(x, y, region_value),
            slot => *slot = Some(LabelStat::new(x, y, region_value)),
        }
    }
    stats.into_iter().flatten().collect()
}

/// Global-coordinate window covering a component's bounding box expanded by the
/// `niter` dilation margin and clamped to the label image, so per-component
/// segmentation, dilation, and point collection are bounded to O(bbox area) rather
/// than O(H·W). The margin (matching EasyOCR's dilated slice) keeps the dilated
/// halo identical to a whole-map build.
struct Window {
    x0: u32,
    y0: u32,
    width: u32,
    height: u32,
}

impl Window {
    fn new(stat: &LabelStat, dimensions: (u32, u32), margin: u32) -> Self {
        let (image_width, image_height) = dimensions;
        let x0 = stat.min_x.saturating_sub(margin);
        let y0 = stat.min_y.saturating_sub(margin);
        let x1 = (stat.max_x + margin).min(image_width - 1);
        let y1 = (stat.max_y + margin).min(image_height - 1);
        Self {
            x0,
            y0,
            width: x1 - x0 + 1,
            height: y1 - y0 + 1,
        }
    }
}

/// Build the single-component segmentation map, dilate it, and fit a box.
fn extract_box(
    label: u32,
    labels: &LabelImage,
    text_score: &Array2<bool>,
    link_score: &Array2<bool>,
    stat: &LabelStat,
) -> Result<Option<BoxPoints>> {
    let niter = compute_niter(stat);
    let window = Window::new(stat, labels.dimensions(), niter);
    let segmap = build_segmap(label, labels, text_score, link_score, &window)?;
    let segmap = dilate_segmap(segmap, niter);
    let points = collect_points(&segmap, &window);
    Ok(fit_box(&points))
}

/// Segmentation map over `window`: 255 where `labels == k`, then cleared where the
/// pixel is link-only (`link_score && !text_score`) to drop pure affinity area.
/// Pixels outside the component bbox are background, so bounding the scan to the
/// window reproduces the whole-map result exactly.
fn build_segmap(
    label: u32,
    labels: &LabelImage,
    text_score: &Array2<bool>,
    link_score: &Array2<bool>,
    window: &Window,
) -> Result<GrayImage> {
    let mut buffer = vec![0u8; (window.width * window.height) as usize];
    for local_y in 0..window.height {
        let y = (window.y0 + local_y) as usize;
        for local_x in 0..window.width {
            let x = (window.x0 + local_x) as usize;
            if labels.get_pixel(x as u32, y as u32)[0] != label || (link_score[[y, x]] && !text_score[[y, x]]) {
                continue;
            }
            buffer[(local_y * window.width + local_x) as usize] = SEGMAP_FOREGROUND;
        }
    }
    GrayImage::from_raw(window.width, window.height, buffer)
        .ok_or_else(|| OcrError::inference("failed to build component segmentation image"))
}

/// Dilation-iteration count: `int(sqrt(area * min(w, h) / (w * h)) * 2)`, truncated
/// toward zero to match EasyOCR's `int(...)`.
fn compute_niter(stat: &LabelStat) -> u32 {
    let width = stat.width();
    let height = stat.height();
    let min_side = width.min(height) as f32;
    let value = (stat.area as f32 * min_side / (width as f32 * height as f32)).sqrt() * NITER_SCALE;
    value.trunc() as u32
}

/// Dilate the 0/255 segmentation map with a `(1 + niter)` square kernel, matching
/// EasyOCR's `cv2.dilate` over a `cv2.MORPH_RECT` structuring element.
///
/// Reproduces OpenCV exactly: `dst(x, y) = max` over kernel offsets `(i, j)` in
/// `[0, 1 + niter)` of `src(x + i - anchor, y + j - anchor)`, where `anchor` is
/// OpenCV's default `(1 + niter) / 2` (integer division). The kernel is symmetric
/// for even `niter` and grows one pixel more on the anchor side for odd `niter`.
/// Out-of-buffer neighbours are background (`BORDER_CONSTANT`). A zero `niter`
/// (unit kernel) is a no-op. See ADR 0018.
///
/// A box structuring element is the Minkowski composition of its horizontal and
/// vertical line elements, so the 2D window max is computed separably as a
/// horizontal line-max pass followed by a vertical one — both using the same
/// per-axis `anchor`. This is `O(area * kernel)` instead of `O(area * kernel^2)`
/// while remaining bit-identical for symmetric (even `niter`) and asymmetric
/// (odd `niter`, even kernel) cases alike.
fn dilate_segmap(segmap: GrayImage, niter: u32) -> GrayImage {
    if niter == 0 {
        return segmap;
    }
    let (width, height) = segmap.dimensions();
    let kernel = (DILATION_KERNEL_OFFSET + niter) as i32;
    let anchor = kernel / 2;
    let source = segmap.as_raw();
    let mut horizontal = vec![0u8; source.len()];
    dilate_along_axis(source, &mut horizontal, width, height, kernel, anchor, Axis::Horizontal);
    let mut dilated = vec![0u8; source.len()];
    dilate_along_axis(&horizontal, &mut dilated, width, height, kernel, anchor, Axis::Vertical);
    GrayImage::from_raw(width, height, dilated).expect("dilated segmap keeps the source dimensions")
}

/// Axis a single separable dilation pass runs along.
#[derive(Clone, Copy)]
enum Axis {
    Horizontal,
    Vertical,
}

/// One separable line-max pass: for every output pixel, write foreground when any
/// in-bounds sample in the `kernel`-length line window (anchored at `anchor`) is
/// foreground. Since the buffer is 0/255, the max reduces to a first-hit test.
fn dilate_along_axis(
    source: &[u8],
    destination: &mut [u8],
    width: u32,
    height: u32,
    kernel: i32,
    anchor: i32,
    axis: Axis,
) {
    // Stride/extent along the pass axis vs. the orthogonal iteration axis. ~keep
    let (extent, stride) = match axis {
        Axis::Horizontal => (width as i32, 1usize),
        Axis::Vertical => (height as i32, width as usize),
    };
    let (outer, inner) = match axis {
        Axis::Horizontal => (height, width),
        Axis::Vertical => (width, height),
    };
    for outer_index in 0..outer {
        let base = match axis {
            Axis::Horizontal => (outer_index * width) as usize,
            Axis::Vertical => outer_index as usize,
        };
        for center in 0..inner as i32 {
            if line_hits_foreground(source, base, stride, extent, center, kernel, anchor) {
                destination[base + center as usize * stride] = SEGMAP_FOREGROUND;
            }
        }
    }
}

/// Whether any in-bounds sample along a single line window is foreground — the max
/// over one axis of the dilation window. `base` is the buffer offset of line index
/// `0`, `stride` the buffer step between consecutive samples, and `extent` the line
/// length; out-of-line samples are background (`BORDER_CONSTANT`).
fn line_hits_foreground(
    source: &[u8],
    base: usize,
    stride: usize,
    extent: i32,
    center: i32,
    kernel: i32,
    anchor: i32,
) -> bool {
    for offset in 0..kernel {
        let sample = center + offset - anchor;
        if sample < 0 || sample >= extent {
            continue;
        }
        if source[base + sample as usize * stride] != 0 {
            return true;
        }
    }
    false
}

/// Collect the non-zero segmentation pixels as `(x, y)` in the label image's global
/// coordinate space, mapping each window-local pixel back through `window`.
fn collect_points(segmap: &GrayImage, window: &Window) -> Vec<Point<i32>> {
    let mut points = Vec::new();
    for (local_x, local_y, pixel) in segmap.enumerate_pixels() {
        if pixel[0] != 0 {
            let x = (window.x0 + local_x) as i32;
            let y = (window.y0 + local_y) as i32;
            points.push(Point::new(x, y));
        }
    }
    points
}

/// Fit a minimum-area rectangle, apply diamond re-alignment, and order the
/// corners clockwise starting at the corner with the smallest `x + y`.
fn fit_box(points: &[Point<i32>]) -> Option<BoxPoints> {
    let rect = min_area_rect(points)?;
    // The single final cast to f32 (mirroring cv2.boxPoints' float32 output) — no
    // per-corner floor/ceil, unlike the imageproc rectangle this replaced. ~keep
    let mut corners: BoxPoints = [
        [rect[0][0] as f32, rect[0][1] as f32],
        [rect[1][0] as f32, rect[1][1] as f32],
        [rect[2][0] as f32, rect[2][1] as f32],
        [rect[3][0] as f32, rect[3][1] as f32],
    ];

    let edge_w = distance(corners[0], corners[1]);
    let edge_h = distance(corners[1], corners[2]);
    let box_ratio = edge_w.max(edge_h) / (edge_w.min(edge_h) + BOX_RATIO_EPSILON);
    if (1.0 - box_ratio).abs() <= DIAMOND_RATIO_TOLERANCE {
        corners = axis_aligned_box(points);
    }

    Some(roll_to_top_left(corners))
}

/// Axis-aligned box `[[l, t], [r, t], [r, b], [l, b]]` over the point extents.
fn axis_aligned_box(points: &[Point<i32>]) -> BoxPoints {
    let mut left = i32::MAX;
    let mut right = i32::MIN;
    let mut top = i32::MAX;
    let mut bottom = i32::MIN;
    for point in points {
        left = left.min(point.x);
        right = right.max(point.x);
        top = top.min(point.y);
        bottom = bottom.max(point.y);
    }
    let (l, r, t, b) = (left as f32, right as f32, top as f32, bottom as f32);
    [[l, t], [r, t], [r, b], [l, b]]
}

/// Euclidean distance between two corners.
fn distance(a: [f32; 2], b: [f32; 2]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// Rotate the corners so the one with the smallest `x + y` becomes index 0,
/// preserving clockwise order (mirrors EasyOCR's `np.roll(box, 4 - startidx)`).
fn roll_to_top_left(corners: BoxPoints) -> BoxPoints {
    let mut start = 0usize;
    let mut best = f32::MAX;
    for (index, corner) in corners.iter().enumerate() {
        let sum = corner[0] + corner[1];
        if sum < best {
            best = sum;
            start = index;
        }
    }
    let mut rolled: BoxPoints = [[0.0; 2]; 4];
    for (index, slot) in rolled.iter_mut().enumerate() {
        *slot = corners[(index + start) % 4];
    }
    rolled
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Build an `[H, W]` map that is `high` inside `[y0, y1) x [x0, x1)`, else 0.
    fn rect_map(height: usize, width: usize, y: (usize, usize), x: (usize, usize), high: f32) -> Array2<f32> {
        let mut map = Array2::<f32>::zeros((height, width));
        for row in y.0..y.1 {
            for col in x.0..x.1 {
                map[[row, col]] = high;
            }
        }
        map
    }

    fn corner_extents(detected: &BoxPoints) -> (f32, f32, f32, f32) {
        let mut min_x = f32::MAX;
        let mut max_x = f32::MIN;
        let mut min_y = f32::MAX;
        let mut max_y = f32::MIN;
        for corner in detected {
            min_x = min_x.min(corner[0]);
            max_x = max_x.max(corner[0]);
            min_y = min_y.min(corner[1]);
            max_y = max_y.max(corner[1]);
        }
        (min_x, max_x, min_y, max_y)
    }

    #[test]
    fn should_return_single_box_covering_solid_blob() {
        // Region 1.0 over rows 3..9, cols 4..14; link all zero. ~keep
        let region = rect_map(16, 20, (3, 9), (4, 14), 1.0);
        let link = Array2::<f32>::zeros((16, 20));

        let boxes = get_det_boxes(&region, &link, 0.7, 0.4, 0.4).expect("boxes");

        assert_eq!(boxes.len(), 1, "one blob must yield exactly one box");
        let (min_x, max_x, min_y, max_y) = corner_extents(&boxes[0]);
        // The box must contain the source rectangle (dilation only grows it). ~keep
        assert!((0.0..=4.0).contains(&min_x), "min_x was {min_x}");
        assert!((13.0..=19.0).contains(&max_x), "max_x was {max_x}");
        assert!((0.0..=3.0).contains(&min_y), "min_y was {min_y}");
        assert!((8.0..=15.0).contains(&max_y), "max_y was {max_y}");
    }

    #[test]
    fn should_drop_blob_smaller_than_min_component_area() {
        // 2x2 blob has area 4 < MIN_COMPONENT_AREA (10). ~keep
        let region = rect_map(10, 10, (2, 4), (2, 4), 1.0);
        let link = Array2::<f32>::zeros((10, 10));

        let boxes = get_det_boxes(&region, &link, 0.7, 0.4, 0.4).expect("boxes");

        assert_eq!(boxes.len(), 0, "sub-threshold-area blob must yield no boxes");
    }

    #[test]
    fn should_drop_blob_below_text_threshold() {
        // 5x5 blob (area 25) but region max 0.5 < text_threshold 0.7. ~keep
        let region = rect_map(12, 12, (2, 7), (2, 7), 0.5);
        let link = Array2::<f32>::zeros((12, 12));

        let boxes = get_det_boxes(&region, &link, 0.7, 0.4, 0.4).expect("boxes");

        assert_eq!(boxes.len(), 0, "blob below text threshold must yield no boxes");
    }

    #[test]
    fn should_error_on_mismatched_heatmap_shapes() {
        let region = Array2::<f32>::zeros((8, 8));
        let link = Array2::<f32>::zeros((8, 10));

        let result = get_det_boxes(&region, &link, 0.7, 0.4, 0.4);

        assert!(result.is_err(), "mismatched shapes must error");
    }

    #[test]
    fn should_scale_coordinates_by_inv_ratio_times_two() {
        let mut boxes = vec![[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]];

        adjust_coordinates(&mut boxes, 1.0);

        assert_eq!(boxes[0], [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0], [14.0, 16.0]]);
    }

    #[test]
    fn should_leave_coordinates_when_inv_ratio_is_half() {
        let mut boxes = vec![[[2.0, 4.0], [6.0, 8.0], [10.0, 12.0], [14.0, 16.0]]];

        adjust_coordinates(&mut boxes, 0.5);

        assert_eq!(boxes[0], [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0], [14.0, 16.0]]);
    }

    /// Whole-map reference for one component: the original unbounded build (allocate
    /// H·W, scan every pixel, dilate the full map, collect global points).
    fn whole_map_points(
        label: u32,
        labels: &LabelImage,
        text_score: &Array2<bool>,
        link_score: &Array2<bool>,
        niter: u32,
    ) -> Vec<Point<i32>> {
        let (width, height) = labels.dimensions();
        let mut buffer = vec![0u8; (width * height) as usize];
        for (x, y, pixel) in labels.enumerate_pixels() {
            let text = text_score[[y as usize, x as usize]];
            let link = link_score[[y as usize, x as usize]];
            let mut value = if pixel[0] == label { SEGMAP_FOREGROUND } else { 0 };
            if link && !text {
                value = 0;
            }
            buffer[(y * width + x) as usize] = value;
        }
        let segmap = dilate_segmap(GrayImage::from_raw(width, height, buffer).expect("segmap"), niter);
        let mut points = Vec::new();
        for (x, y, pixel) in segmap.enumerate_pixels() {
            if pixel[0] != 0 {
                points.push(Point::new(x as i32, y as i32));
            }
        }
        points
    }

    #[test]
    fn should_bound_segmap_to_bbox_matching_whole_map_points() {
        // Two separated blobs; the bounded per-component build (segmap + dilation + ~keep
        // point collection) must yield identical global points to the whole-map path. ~keep
        let mut region = Array2::<f32>::zeros((30, 24));
        for row in 2..8 {
            for col in 2..9 {
                region[[row, col]] = 1.0;
            }
        }
        for row in 15..22 {
            for col in 14..23 {
                region[[row, col]] = 1.0;
            }
        }
        let link = Array2::<f32>::zeros((30, 24));

        let text_score = region.mapv(|value| value > 0.4);
        let link_score = link.mapv(|value| value > 0.4);
        let comb = build_comb_image(&text_score, &link_score).expect("comb");
        let labels = connected_components(&comb, Connectivity::Four, Luma([0u8]));
        let stats = compute_label_stats(&labels, &region);

        assert_eq!(stats.len(), 2, "two blobs must yield two components");
        for (index, stat) in stats.iter().enumerate() {
            let label = index as u32 + 1;
            let niter = compute_niter(stat);
            let window = Window::new(stat, labels.dimensions(), niter);
            let segmap = build_segmap(label, &labels, &text_score, &link_score, &window).expect("segmap");
            let bounded = collect_points(&dilate_segmap(segmap, niter), &window);
            let reference = whole_map_points(label, &labels, &text_score, &link_score, niter);
            assert_eq!(bounded, reference, "label {label} points must match whole-map build");
        }
    }

    /// Straightforward O(kernel^2) box-dilation reference: `dst(x, y) = max` over
    /// the full `(i, j)` kernel window of `src(x + i - anchor, y + j - anchor)`,
    /// with out-of-buffer samples treated as background. Written inline so the
    /// separable pass is checked against an independent two-loop implementation.
    fn reference_box_dilate(segmap: &GrayImage, niter: u32) -> GrayImage {
        if niter == 0 {
            return segmap.clone();
        }
        let (width, height) = segmap.dimensions();
        let kernel = (DILATION_KERNEL_OFFSET + niter) as i32;
        let anchor = kernel / 2;
        let source = segmap.as_raw();
        let mut dilated = vec![0u8; source.len()];
        for y in 0..height as i32 {
            for x in 0..width as i32 {
                let mut hit = false;
                for j in 0..kernel {
                    let sample_y = y + j - anchor;
                    if sample_y < 0 || sample_y >= height as i32 {
                        continue;
                    }
                    for i in 0..kernel {
                        let sample_x = x + i - anchor;
                        if sample_x < 0 || sample_x >= width as i32 {
                            continue;
                        }
                        if source[(sample_y as u32 * width + sample_x as u32) as usize] != 0 {
                            hit = true;
                        }
                    }
                }
                if hit {
                    dilated[(y as u32 * width + x as u32) as usize] = SEGMAP_FOREGROUND;
                }
            }
        }
        GrayImage::from_raw(width, height, dilated).expect("reference dilation keeps dimensions")
    }

    /// A hand-built dilation test case: map dimensions plus foreground pixels.
    struct DilationPattern {
        width: u32,
        height: u32,
        foreground: Vec<(u32, u32)>,
    }

    impl DilationPattern {
        fn new(width: u32, height: u32, foreground: &[(u32, u32)]) -> Self {
            Self {
                width,
                height,
                foreground: foreground.to_vec(),
            }
        }
    }

    fn segmap_from(width: u32, height: u32, foreground: &[(u32, u32)]) -> GrayImage {
        let mut buffer = vec![0u8; (width * height) as usize];
        for &(x, y) in foreground {
            buffer[(y * width + x) as usize] = SEGMAP_FOREGROUND;
        }
        GrayImage::from_raw(width, height, buffer).expect("segmap pattern")
    }

    #[test]
    fn should_match_reference_box_dilation_byte_for_byte() {
        // Hand-built patterns: isolated pixel, small block, L-shape, and two ~keep
        // near-border blocks that exercise BORDER_CONSTANT clipping. ~keep
        let patterns: [DilationPattern; 5] = [
            DilationPattern::new(7, 7, &[(3, 3)]),
            DilationPattern::new(10, 10, &[(4, 4), (5, 4), (4, 5), (5, 5)]),
            DilationPattern::new(9, 9, &[(2, 2), (2, 3), (2, 4), (3, 4), (4, 4)]),
            DilationPattern::new(8, 8, &[(0, 0), (1, 0), (0, 1), (1, 1)]),
            DilationPattern::new(8, 6, &[(6, 4), (7, 4), (6, 5), (7, 5)]),
        ];
        for pattern in patterns {
            let DilationPattern {
                width,
                height,
                foreground,
            } = pattern;
            let segmap = segmap_from(width, height, &foreground);
            // Covers even niter (symmetric kernel) and odd niter (even kernel, ~keep
            // asymmetric anchor: e.g. niter 3 -> kernel 4, anchor 2). ~keep
            for niter in 0..=5u32 {
                let separable = dilate_segmap(segmap.clone(), niter);
                let reference = reference_box_dilate(&segmap, niter);
                assert_eq!(
                    separable.as_raw(),
                    reference.as_raw(),
                    "niter {niter} on {width}x{height} pattern must match the O(k^2) reference byte-for-byte",
                );
            }
        }
    }

    #[test]
    fn should_detect_two_separate_blobs_as_two_boxes() {
        let mut region = Array2::<f32>::zeros((30, 24));
        for row in 2..8 {
            for col in 2..9 {
                region[[row, col]] = 1.0;
            }
        }
        for row in 15..22 {
            for col in 14..23 {
                region[[row, col]] = 1.0;
            }
        }
        let link = Array2::<f32>::zeros((30, 24));

        let boxes = get_det_boxes(&region, &link, 0.7, 0.4, 0.4).expect("boxes");

        assert_eq!(boxes.len(), 2, "two separated blobs must yield two boxes");
    }
}
