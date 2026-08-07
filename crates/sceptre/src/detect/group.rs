//! Box grouping: merge character/word boxes into text lines.
//!
//! Reference: EasyOCR `utils.py` (`group_text_box`). Splits horizontal vs. free
//! (rotated) boxes via the slope threshold and merges using the y-center,
//! height, and width thresholds, applying `add_margin`.

use crate::config::DetectionConfig;

/// Floor applied to the horizontal extent when computing box slopes, mirroring
/// EasyOCR's `np.maximum(10, ...)` guard against division by a tiny run.
const SLOPE_DENOM_FLOOR: f32 = 10.0;

/// Diagonal-length scaling used when expanding free (rotated) quads by a margin
/// (`1.44 * add_margin * min(width, height)` in `group_text_box`).
const FREE_MARGIN_FACTOR: f32 = 1.44;

/// Grouped detection output: axis-aligned lines and free (rotated) quads.
pub(super) struct Grouped {
    /// Axis-aligned boxes as `[x_min, x_max, y_min, y_max]` (with margins applied).
    pub horizontal: Vec<[f32; 4]>,
    /// Free (rotated) quads as 4 corners `[x, y]` (with margins applied).
    pub free: Vec<[[f32; 2]; 4]>,
}

/// Internal working form of a horizontal box: extents plus the derived y-center
/// and height that the line-merge algorithm sorts and groups on. Mirrors the
/// `[x_min, x_max, y_min, y_max, 0.5*(y_min+y_max), y_max-y_min]` list entry in
/// `group_text_box`.
#[derive(Clone, Copy)]
struct HBox {
    x_min: f32,
    x_max: f32,
    y_min: f32,
    y_max: f32,
    ycenter: f32,
    height: f32,
}

/// Outcome of classifying a single box by slope.
enum Classified {
    /// An axis-aligned box, in working form.
    Horizontal(HBox),
    /// A rotated quad, corners already margin-expanded.
    Free([[f32; 2]; 4]),
}

/// Split each box into horizontal vs. free by slope, then merge horizontal boxes
/// into lines and apply `add_margin`. Mirrors `group_text_box`.
pub(super) fn group_boxes(boxes: &[[[f32; 2]; 4]], config: &DetectionConfig) -> Grouped {
    let mut horizontals: Vec<HBox> = Vec::new();
    let mut free: Vec<[[f32; 2]; 4]> = Vec::new();

    for corners in boxes {
        match classify(corners, config) {
            Classified::Horizontal(hbox) => horizontals.push(hbox),
            Classified::Free(quad) => free.push(quad),
        }
    }

    horizontals.sort_by(|a, b| a.ycenter.partial_cmp(&b.ycenter).unwrap_or(std::cmp::Ordering::Equal));

    let mut horizontal: Vec<[f32; 4]> = Vec::new();
    for line in combine_into_lines(&horizontals, config.ycenter_ths) {
        horizontal.extend(merge_line(&line, config));
    }

    Grouped { horizontal, free }
}

/// Classify one 4-corner box as horizontal or free using the slope of its top
/// and bottom edges. Corners are `[TL, TR, BR, BL]` as `[x, y]`.
fn classify(corners: &[[f32; 2]; 4], config: &DetectionConfig) -> Classified {
    let (x1, y1) = (corners[0][0], corners[0][1]);
    let (x2, y2) = (corners[1][0], corners[1][1]);
    let (x3, y3) = (corners[2][0], corners[2][1]);
    let (x4, y4) = (corners[3][0], corners[3][1]);

    let slope_up = (y2 - y1) / (x2 - x1).max(SLOPE_DENOM_FLOOR);
    let slope_down = (y3 - y4) / (x3 - x4).max(SLOPE_DENOM_FLOOR);

    if slope_up.abs().max(slope_down.abs()) < config.slope_ths {
        let x_min = x1.min(x2).min(x3).min(x4);
        let x_max = x1.max(x2).max(x3).max(x4);
        let y_min = y1.min(y2).min(y3).min(y4);
        let y_max = y1.max(y2).max(y3).max(y4);
        Classified::Horizontal(HBox {
            x_min,
            x_max,
            y_min,
            y_max,
            ycenter: 0.5 * (y_min + y_max),
            height: y_max - y_min,
        })
    } else {
        Classified::Free(expand_free_quad(corners, config.add_margin))
    }
}

/// Expand a rotated quad outward by a slope-aware margin, mirroring the
/// `theta13`/`theta24` cos/sin expansion in `group_text_box`.
fn expand_free_quad(corners: &[[f32; 2]; 4], add_margin: f32) -> [[f32; 2]; 4] {
    let (x1, y1) = (corners[0][0], corners[0][1]);
    let (x2, y2) = (corners[1][0], corners[1][1]);
    let (x3, y3) = (corners[2][0], corners[2][1]);
    let (x4, y4) = (corners[3][0], corners[3][1]);

    let width = ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt();
    let height = ((x4 - x1).powi(2) + (y4 - y1).powi(2)).sqrt();
    let margin = (FREE_MARGIN_FACTOR * add_margin * width.min(height)).trunc();

    let theta13 = ((y1 - y3) / (x1 - x3).max(SLOPE_DENOM_FLOOR)).atan().abs();
    let theta24 = ((y2 - y4) / (x2 - x4).max(SLOPE_DENOM_FLOOR)).atan().abs();

    [
        [x1 - theta13.cos() * margin, y1 - theta13.sin() * margin],
        [x2 + theta24.cos() * margin, y2 - theta24.sin() * margin],
        [x3 + theta13.cos() * margin, y3 + theta13.sin() * margin],
        [x4 - theta24.cos() * margin, y4 + theta24.sin() * margin],
    ]
}

/// Group y-center-sorted boxes into lines by y-center proximity relative to the
/// running mean height (`ycenter_ths * mean(height)`). The running sums accumulate
/// in the same order the per-line vecs were built, so the means are bit-identical
/// to re-summing the vecs each step.
fn combine_into_lines(sorted: &[HBox], ycenter_ths: f32) -> Vec<Vec<HBox>> {
    let mut combined: Vec<Vec<HBox>> = Vec::new();
    let mut current: Vec<HBox> = Vec::new();
    let mut height_sum = 0.0_f32;
    let mut ycenter_sum = 0.0_f32;
    let mut count = 0.0_f32;

    for &hbox in sorted {
        let same_line =
            !current.is_empty() && (ycenter_sum / count - hbox.ycenter).abs() < ycenter_ths * (height_sum / count);
        if current.is_empty() || same_line {
            height_sum += hbox.height;
            ycenter_sum += hbox.ycenter;
            count += 1.0;
        } else {
            combined.push(std::mem::take(&mut current));
            height_sum = hbox.height;
            ycenter_sum = hbox.ycenter;
            count = 1.0;
        }
        current.push(hbox);
    }
    combined.push(current);
    combined
}

/// Merge the boxes of a single line into one or more `[x_min, x_max, y_min, y_max]`
/// entries, splitting on height/width discontinuities and applying `add_margin`.
fn merge_line(line: &[HBox], config: &DetectionConfig) -> Vec<[f32; 4]> {
    if line.len() == 1 {
        return vec![margin_entry(&line[0], config.add_margin)];
    }

    let mut sorted: Vec<HBox> = line.to_vec();
    sorted.sort_by(|a, b| a.x_min.partial_cmp(&b.x_min).unwrap_or(std::cmp::Ordering::Equal));

    group_adjacent(&sorted, config)
        .into_iter()
        .map(|group| merge_group(&group, config.add_margin))
        .collect()
}

/// Split an x-sorted line into runs of adjacent boxes with comparable height and
/// small horizontal gaps, per the `height_ths`/`width_ths` test in `group_text_box`.
fn group_adjacent(sorted: &[HBox], config: &DetectionConfig) -> Vec<Vec<HBox>> {
    let mut groups: Vec<Vec<HBox>> = Vec::new();
    let mut current: Vec<HBox> = Vec::new();
    let mut height_sum = 0.0_f32;
    let mut count = 0.0_f32;
    let mut running_x_max = 0.0_f32;

    for &hbox in sorted {
        let mergeable = !current.is_empty()
            && (height_sum / count - hbox.height).abs() < config.height_ths * (height_sum / count)
            && (hbox.x_min - running_x_max) < config.width_ths * (hbox.y_max - hbox.y_min);
        if current.is_empty() || mergeable {
            height_sum += hbox.height;
            count += 1.0;
        } else {
            groups.push(std::mem::take(&mut current));
            height_sum = hbox.height;
            count = 1.0;
        }
        running_x_max = hbox.x_max;
        current.push(hbox);
    }
    if !current.is_empty() {
        groups.push(current);
    }
    groups
}

/// Apply `add_margin` to a single box, truncating the pixel margin like Python's
/// `int(...)`. Returns `[x_min, x_max, y_min, y_max]`.
fn margin_entry(b: &HBox, add_margin: f32) -> [f32; 4] {
    let margin = (add_margin * (b.x_max - b.x_min).min(b.y_max - b.y_min)).trunc();
    [b.x_min - margin, b.x_max + margin, b.y_min - margin, b.y_max + margin]
}

/// Collapse a run of merged boxes into one margin-expanded entry over their extent.
fn merge_group(group: &[HBox], add_margin: f32) -> [f32; 4] {
    if group.len() == 1 {
        return margin_entry(&group[0], add_margin);
    }
    let x_min = group.iter().map(|b| b.x_min).fold(f32::INFINITY, f32::min);
    let x_max = group.iter().map(|b| b.x_max).fold(f32::NEG_INFINITY, f32::max);
    let y_min = group.iter().map(|b| b.y_min).fold(f32::INFINITY, f32::min);
    let y_max = group.iter().map(|b| b.y_max).fold(f32::NEG_INFINITY, f32::max);
    let margin = (add_margin * (x_max - x_min).min(y_max - y_min)).trunc();
    [x_min - margin, x_max + margin, y_min - margin, y_max + margin]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Axis-aligned corners `[TL, TR, BR, BL]` spanning `[x0, x1] × [y0, y1]`.
    fn axis_box(x0: f32, x1: f32, y0: f32, y1: f32) -> [[f32; 2]; 4] {
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    }

    fn config_with(add_margin: f32) -> DetectionConfig {
        DetectionConfig {
            add_margin,
            ..DetectionConfig::default()
        }
    }

    #[test]
    fn should_classify_level_box_as_horizontal_with_correct_extents() {
        let config = config_with(0.0);
        let grouped = group_boxes(&[axis_box(10.0, 50.0, 10.0, 30.0)], &config);

        assert!(grouped.free.is_empty());
        assert_eq!(grouped.horizontal, vec![[10.0, 50.0, 10.0, 30.0]]);
    }

    #[test]
    fn should_classify_steeply_slanted_box_as_free() {
        let config = config_with(0.0);
        let slanted = [[10.0, 10.0], [50.0, 40.0], [50.0, 60.0], [10.0, 30.0]];
        let grouped = group_boxes(&[slanted], &config);

        assert!(grouped.horizontal.is_empty());
        assert_eq!(grouped.free.len(), 1);
    }

    #[test]
    fn should_merge_two_adjacent_same_height_boxes_on_one_line() {
        let config = config_with(0.0);
        let left = axis_box(10.0, 50.0, 10.0, 30.0);
        let right = axis_box(55.0, 95.0, 10.0, 30.0);
        let grouped = group_boxes(&[left, right], &config);

        assert_eq!(grouped.horizontal, vec![[10.0, 95.0, 10.0, 30.0]]);
    }

    #[test]
    fn should_keep_vertically_distant_boxes_as_separate_entries() {
        let config = config_with(0.0);
        let top = axis_box(10.0, 50.0, 10.0, 30.0);
        let bottom = axis_box(10.0, 50.0, 200.0, 220.0);
        let grouped = group_boxes(&[top, bottom], &config);

        assert_eq!(grouped.horizontal.len(), 2);
        assert!(grouped.horizontal.contains(&[10.0, 50.0, 10.0, 30.0]));
        assert!(grouped.horizontal.contains(&[10.0, 50.0, 200.0, 220.0]));
    }

    #[test]
    fn should_apply_margin_when_add_margin_is_positive() {
        let no_margin = group_boxes(&[axis_box(10.0, 50.0, 10.0, 30.0)], &config_with(0.0));
        let with_margin = group_boxes(&[axis_box(10.0, 50.0, 10.0, 30.0)], &config_with(0.1));

        // width 40, height 20 → margin = int(0.1 * min(40, 20)) = 2. ~keep
        assert_eq!(no_margin.horizontal, vec![[10.0, 50.0, 10.0, 30.0]]);
        assert_eq!(with_margin.horizontal, vec![[8.0, 52.0, 8.0, 32.0]]);
    }

    #[test]
    fn should_return_empty_grouped_for_no_boxes() {
        let grouped = group_boxes(&[], &DetectionConfig::default());
        assert!(grouped.horizontal.is_empty());
        assert!(grouped.free.is_empty());
    }

    /// Reproduces the `LES ARTS` / `DÉCORATIFS` boxes CRAFT emits for
    /// `test_documents/images/french.jpg`: two 30px-tall boxes plus a 46px-tall
    /// box that sceptre's postprocessing measures for `DÉCORATIFS`. The running-mean
    /// height test (`|mean(30,30) - 46| = 16` against `0.5*30 = 15`) rejects the
    /// merge by 1px — a boundary miss in the upstream box extent, not in this
    /// merge logic (see the `LES ARTS DÉCORATIFS` line-grouping investigation).
    #[test]
    fn should_split_line_when_third_box_height_exceeds_the_height_ths_boundary() {
        let config = config_with(0.0);
        let les = axis_box(246.0, 288.0, 440.0, 470.0);
        let arts = axis_box(296.0, 358.0, 438.0, 468.0);
        let decoratifs = axis_box(360.0, 512.0, 422.0, 468.0);

        let grouped = group_boxes(&[les, arts, decoratifs], &config);

        assert_eq!(
            grouped.horizontal,
            vec![[246.0, 358.0, 438.0, 470.0], [360.0, 512.0, 422.0, 468.0]]
        );
    }

    /// Same three boxes as above, except `DÉCORATIFS` is 44px tall — the height EasyOCR's
    /// own CRAFT run measures for this exact region, obtained by dumping the box list
    /// `utils.py:group_text_box` receives. `|mean(30,30) - 44| = 14 < 15`
    /// merges. This proves the running-mean merge test in `group_adjacent` is not the
    /// defect: it merges or splits correctly depending on the input box height, and
    /// sceptre's own CRAFT postprocessing produces a 46px box where EasyOCR's produces 44px. ~keep
    #[test]
    fn should_merge_line_when_third_box_height_is_within_the_height_ths_boundary() {
        let config = config_with(0.0);
        let les = axis_box(246.0, 288.0, 440.0, 470.0);
        let arts = axis_box(296.0, 358.0, 438.0, 468.0);
        let decoratifs = axis_box(360.0, 511.0, 422.0, 466.0);

        let grouped = group_boxes(&[les, arts, decoratifs], &config);

        assert_eq!(grouped.horizontal, vec![[246.0, 511.0, 422.0, 470.0]]);
    }

    /// Corners measured by sceptre's CRAFT postprocessing for the `LOUVRE` region on
    /// `french.jpg`: `slope_down = 10/96 ≈ 0.1042`, just over `slope_ths = 0.1`, so the
    /// quad is routed to the free (rotated) path and excluded from horizontal grouping
    /// with `[Palais du`. EasyOCR's own CRAFT run measures nearly the same quad shifted by
    /// 1-2px (`slope_down = 9/94 ≈ 0.0957`, just under the threshold), which it keeps
    /// horizontal and merges. Both quads are classified correctly relative to
    /// `slope_ths`: the divergence is a sub-2px difference in the upstream box corners,
    /// not in this threshold comparison. ~keep
    #[test]
    fn should_route_near_identical_quads_to_opposite_paths_at_the_slope_boundary() {
        let config = DetectionConfig::default();
        let sceptre_quad = [[378.0, 338.0], [474.0, 346.0], [470.0, 380.0], [374.0, 370.0]];
        let easyocr_quad = [[378.0, 339.0], [472.0, 347.0], [469.0, 378.0], [375.0, 369.0]];

        let sceptre_grouped = group_boxes(&[sceptre_quad], &config);
        assert!(sceptre_grouped.horizontal.is_empty());
        assert_eq!(sceptre_grouped.free.len(), 1);

        let easyocr_grouped = group_boxes(&[easyocr_quad], &config);
        assert!(easyocr_grouped.free.is_empty());
        assert_eq!(easyocr_grouped.horizontal.len(), 1);
    }

    /// A merged multi-box group's margin must derive from the merged extent
    /// (`x_max-x_min`, `y_max-y_min` over the whole group), not from either
    /// contributing box's own height. Box A is 20px tall at y[10,30]; box B is
    /// also 20px tall but offset to y[19,39], so the union spans 29px even though
    /// each box's own height is 20. `merge_group` must use 29 (`int(0.3*29) = 8`),
    /// not 20 (`int(0.3*20) = 6`), matching `group_text_box`'s "adjacent box in
    /// same line" case. ~keep
    #[test]
    fn should_derive_multi_box_margin_from_the_merged_extent_not_a_member_box_height() {
        let config = config_with(0.3);
        let a = axis_box(10.0, 50.0, 10.0, 30.0);
        let b = axis_box(55.0, 95.0, 19.0, 39.0);

        let grouped = group_boxes(&[a, b], &config);

        assert_eq!(grouped.horizontal, vec![[2.0, 103.0, 2.0, 47.0]]);
    }

    #[test]
    fn should_merge_line_of_three_and_keep_distant_line_separate() {
        // Exercises the running-mean accumulation: three adjacent same-height boxes ~keep
        // accumulate into one line/group, a fourth far below forms its own line. ~keep
        let config = config_with(0.0);
        let a = axis_box(10.0, 50.0, 10.0, 30.0);
        let b = axis_box(55.0, 95.0, 10.0, 30.0);
        let c = axis_box(100.0, 140.0, 10.0, 30.0);
        let d = axis_box(10.0, 50.0, 200.0, 220.0);

        let grouped = group_boxes(&[a, b, c, d], &config);

        assert_eq!(
            grouped.horizontal,
            vec![[10.0, 140.0, 10.0, 30.0], [10.0, 50.0, 200.0, 220.0]]
        );
    }
}
