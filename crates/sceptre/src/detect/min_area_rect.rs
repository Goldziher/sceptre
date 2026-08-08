//! Minimum-area rotated rectangle fitting.
//!
//! Matches OpenCV's `minAreaRect` + `boxPoints` contract that EasyOCR relies on
//! (`craft_utils.py:getDetBoxes_core`): the convex hull is walked edge by edge with
//! the rotating-calipers method, the axis-aligned extent of the hull is tested in
//! each edge's rotated frame, the globally smallest-area candidate is kept, and its
//! corners are returned as raw `f64` values with **no** rounding.
//!
//! `imageproc::geometry::min_area_rect` computes the same rectangle in `f64` but
//! then snaps each corner outward with a per-corner `.floor()`/`.ceil()` before
//! returning — a step OpenCV's `minAreaRect`/`boxPoints` never performs (EasyOCR
//! keeps boxes in floating point all the way through `group_text_box`). That
//! outward snap is the root cause of sceptre's rotated-quad boxes drifting from
//! EasyOCR's by up to a couple of pixels after the detector's `x2` scale-up, which
//! in turn flips borderline slope/height merge decisions in `group.rs`. See ADR
//! 0039 (supersedes ADR 0013's min-area-rect half for rotated quads).

use std::cmp::Ordering;
use std::f64::consts::PI;

use imageproc::geometry::convex_hull;
use imageproc::point::Point;

/// A right angle in radians. A rectangle's axis-aligned test repeats every quarter
/// turn, so calipers angles are folded into `[0, RIGHT_ANGLE)`.
const RIGHT_ANGLE: f64 = PI / 2.0;

/// Rectangle corners in cyclic (rectangle-traversal) order: top-left, top-right,
/// bottom-right, bottom-left — mirroring EasyOCR's "poly top-left, top-right,
/// low-right, low-left" convention. Left un-rounded; the caller owns the single
/// final cast to its own coordinate type.
pub(super) type RectCorners = [[f64; 2]; 4];

/// A 2D point in `f64`, used internally for the rotation arithmetic.
#[derive(Clone, Copy)]
struct Vec2 {
    x: f64,
    y: f64,
}

impl Vec2 {
    fn from_point(point: Point<i32>) -> Self {
        Self {
            x: f64::from(point.x),
            y: f64::from(point.y),
        }
    }

    /// Rotate this point by the angle whose sine/cosine are given.
    fn rotate(self, sin_theta: f64, cos_theta: f64) -> Self {
        Self {
            x: self.x * cos_theta + self.y * sin_theta,
            y: self.y * cos_theta - self.x * sin_theta,
        }
    }

    /// Invert [`Vec2::rotate`] by the same angle.
    fn invert_rotate(self, sin_theta: f64, cos_theta: f64) -> Self {
        Self {
            x: self.x * cos_theta - self.y * sin_theta,
            y: self.y * cos_theta + self.x * sin_theta,
        }
    }
}

/// Fit the minimum-area rectangle enclosing `points`. Returns `None` only when
/// `points` (and therefore its convex hull) is empty; callers already guard this
/// case, but the geometry itself never panics on it.
pub(super) fn min_area_rect(points: &[Point<i32>]) -> Option<RectCorners> {
    if points.is_empty() {
        return None;
    }
    let hull = convex_hull(points);
    match hull.len() {
        0 => None,
        1 => {
            let p = [f64::from(hull[0].x), f64::from(hull[0].y)];
            Some([p, p, p, p])
        }
        2 => {
            let a = [f64::from(hull[0].x), f64::from(hull[0].y)];
            let b = [f64::from(hull[1].x), f64::from(hull[1].y)];
            Some([a, b, b, a])
        }
        _ => Some(rotating_calipers(&hull)),
    }
}

/// Test the axis-aligned extent of `hull` in every edge's rotated frame and keep
/// the smallest-area candidate, mirroring OpenCV's `rotatingCalipers`. Unlike
/// `imageproc::geometry::rotating_calipers`, this also tests the hull's closing
/// edge (last vertex back to the first) — omitting it can only ever pick an
/// equal-or-larger rectangle, never a smaller one.
fn rotating_calipers(hull: &[Point<i32>]) -> RectCorners {
    let vertices: Vec<Vec2> = hull.iter().copied().map(Vec2::from_point).collect();
    let edge_count = vertices.len();

    let mut min_area = f64::MAX;
    let mut best: RectCorners = [[0.0; 2]; 4];

    for edge_index in 0..edge_count {
        let start = vertices[edge_index];
        let end = vertices[(edge_index + 1) % edge_count];
        let edge_angle = ((end.y - start.y).atan2(end.x - start.x) + PI) % RIGHT_ANGLE;
        let (sin_theta, cos_theta) = edge_angle.abs().sin_cos();

        let mut min_x = f64::MAX;
        let mut max_x = f64::MIN;
        let mut min_y = f64::MAX;
        let mut max_y = f64::MIN;
        for vertex in &vertices {
            let rotated = vertex.rotate(sin_theta, cos_theta);
            min_x = min_x.min(rotated.x);
            max_x = max_x.max(rotated.x);
            min_y = min_y.min(rotated.y);
            max_y = max_y.max(rotated.y);
        }

        let area = (max_x - min_x) * (max_y - min_y);
        if area < min_area {
            min_area = area;
            let corners = [
                Vec2 { x: max_x, y: min_y },
                Vec2 { x: min_x, y: min_y },
                Vec2 { x: min_x, y: max_y },
                Vec2 { x: max_x, y: max_y },
            ]
            .map(|corner| corner.invert_rotate(sin_theta, cos_theta));
            best = order_corners(corners);
        }
    }

    best
}

/// Reorder four rectangle corners (in any consistent cyclic order) into
/// top-left/top-right/bottom-right/bottom-left by sorting on `x` and splitting
/// each x-pair on `y`. Mirrors `imageproc::geometry::rotating_calipers`'s
/// classification step, minus its outward corner snap.
fn order_corners(mut corners: [Vec2; 4]) -> RectCorners {
    corners.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(Ordering::Equal));

    let (top_left, bottom_left) = if corners[1].y > corners[0].y {
        (corners[0], corners[1])
    } else {
        (corners[1], corners[0])
    };
    let (top_right, bottom_right) = if corners[3].y > corners[2].y {
        (corners[2], corners[3])
    } else {
        (corners[3], corners[2])
    };

    [
        [top_left.x, top_left.y],
        [top_right.x, top_right.y],
        [bottom_right.x, bottom_right.y],
        [bottom_left.x, bottom_left.y],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn area_of(corners: RectCorners) -> f64 {
        let w = ((corners[1][0] - corners[0][0]).powi(2) + (corners[1][1] - corners[0][1]).powi(2)).sqrt();
        let h = ((corners[2][0] - corners[1][0]).powi(2) + (corners[2][1] - corners[1][1]).powi(2)).sqrt();
        w * h
    }

    #[test]
    fn should_return_none_for_empty_points() {
        assert_eq!(min_area_rect(&[]), None);
    }

    #[test]
    fn should_fit_exact_box_for_axis_aligned_rectangle() {
        let points = [Point::new(2, 3), Point::new(9, 3), Point::new(9, 8), Point::new(2, 8)];

        let rect = min_area_rect(&points).expect("rect");

        assert_eq!(rect, [[2.0, 3.0], [9.0, 3.0], [9.0, 8.0], [2.0, 8.0]]);
    }

    /// A tilted rectangle whose corners are rounded to integer pixels before
    /// fitting (simulating a discrete blob outline), so the true minimum-area
    /// rectangle has fractional corner coordinates. `imageproc::geometry::
    /// min_area_rect` snaps each corner outward to the nearest enclosing integer,
    /// inflating the box; this implementation must keep the fractional value and
    /// therefore return a strictly smaller area. If this test is pointed at the
    /// old `imageproc` call instead, the strict-inequality assertion below fails,
    /// proving the two disagree on tilted input rather than merely on style.
    #[test]
    fn should_not_inflate_area_via_outward_corner_snap_on_tilted_rectangle() {
        // A 40x20 rectangle centered at the origin, rotated ~20 degrees, with
        // corners rounded to the nearest pixel. ~keep
        let theta = 20.0_f64.to_radians();
        let (sin_t, cos_t) = theta.sin_cos();
        let half_w = 20.0;
        let half_h = 10.0;
        let local = [
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h),
        ];
        let rotated: Vec<Point<i32>> = local
            .iter()
            .map(|&(x, y)| {
                let rx = x * cos_t - y * sin_t + 100.0;
                let ry = x * sin_t + y * cos_t + 100.0;
                Point::new(rx.round() as i32, ry.round() as i32)
            })
            .collect();

        let new_rect = min_area_rect(&rotated).expect("new rect");
        let old_rect = imageproc::geometry::min_area_rect(&rotated);
        let old_area = area_of([
            [old_rect[0].x as f64, old_rect[0].y as f64],
            [old_rect[1].x as f64, old_rect[1].y as f64],
            [old_rect[2].x as f64, old_rect[2].y as f64],
            [old_rect[3].x as f64, old_rect[3].y as f64],
        ]);
        let new_area = area_of(new_rect);
        let expected_area = 40.0 * 20.0;

        assert!(
            new_area < old_area - 1.0,
            "new area {new_area} must be strictly smaller than the outward-snapped old area {old_area}"
        );
        assert!(
            (new_area - expected_area).abs() < 2.0,
            "new area {new_area} must stay close to the analytic 40x20={expected_area} rectangle"
        );
        assert!(
            new_rect
                .iter()
                .any(|corner| corner[0].fract() != 0.0 || corner[1].fract() != 0.0),
            "a tilted rectangle's true corners are fractional; an all-integer result means \
             the outward snap is still happening"
        );
    }

    #[test]
    fn should_fit_degenerate_rect_for_single_point() {
        let points = [Point::new(5, 7)];

        let rect = min_area_rect(&points).expect("rect");

        assert_eq!(rect, [[5.0, 7.0], [5.0, 7.0], [5.0, 7.0], [5.0, 7.0]]);
    }

    #[test]
    fn should_fit_degenerate_rect_for_two_points() {
        let points = [Point::new(0, 0), Point::new(4, 3)];

        let rect = min_area_rect(&points).expect("rect");

        assert_eq!(rect, [[0.0, 0.0], [4.0, 3.0], [4.0, 3.0], [0.0, 0.0]]);
    }

    /// `rotating_calipers` restricted to a subset of hull edges, mirroring
    /// `imageproc::geometry::rotating_calipers`'s `windows(2)` scan (which never
    /// tests the edge from the last hull vertex back to the first). Used only to
    /// prove that omission costs area on a concrete case; production code always
    /// tests every edge including the closing one.
    fn rotating_calipers_windows_only(hull: &[Point<i32>]) -> RectCorners {
        let vertices: Vec<Vec2> = hull.iter().copied().map(Vec2::from_point).collect();
        let mut min_area = f64::MAX;
        let mut best: RectCorners = [[0.0; 2]; 4];
        for window in vertices.windows(2) {
            let (start, end) = (window[0], window[1]);
            let edge_angle = ((end.y - start.y).atan2(end.x - start.x) + PI) % RIGHT_ANGLE;
            let (sin_theta, cos_theta) = edge_angle.abs().sin_cos();

            let mut min_x = f64::MAX;
            let mut max_x = f64::MIN;
            let mut min_y = f64::MAX;
            let mut max_y = f64::MIN;
            for vertex in &vertices {
                let rotated = vertex.rotate(sin_theta, cos_theta);
                min_x = min_x.min(rotated.x);
                max_x = max_x.max(rotated.x);
                min_y = min_y.min(rotated.y);
                max_y = max_y.max(rotated.y);
            }
            let area = (max_x - min_x) * (max_y - min_y);
            if area < min_area {
                min_area = area;
                let corners = [
                    Vec2 { x: max_x, y: min_y },
                    Vec2 { x: min_x, y: min_y },
                    Vec2 { x: min_x, y: max_y },
                    Vec2 { x: max_x, y: max_y },
                ]
                .map(|corner| corner.invert_rotate(sin_theta, cos_theta));
                best = order_corners(corners);
            }
        }
        best
    }

    #[test]
    fn should_test_closing_hull_edge_that_imageproc_omits() {
        // Convex hull (verified against the real `imageproc::geometry::convex_hull`
        // ordering by brute-force search, not assumed) whose minimum-area
        // rectangle is flush with the closing edge — from the last hull vertex,
        // (45, 48), back to the first, (1, 0) — the one edge
        // `imageproc::geometry::rotating_calipers`'s `windows(2)` scan never
        // tests. ~keep
        let hull = [
            Point::new(1, 0),
            Point::new(17, 7),
            Point::new(31, 14),
            Point::new(45, 48),
        ];
        assert_eq!(
            convex_hull(hull.to_vec()),
            hull,
            "the fixture must already be its own hull"
        );

        let full_area = area_of(rotating_calipers(&hull));
        let windows_only_area = area_of(rotating_calipers_windows_only(&hull));

        assert!(
            full_area < windows_only_area - 1.0,
            "testing every hull edge (full_area {full_area}) must beat the windows(2)-only scan \
             ({windows_only_area}) that omits the closing edge"
        );
    }
}
