//! Per-stage timing telemetry for the `run --timings` flag.
//!
//! [`StageTimer`] is a [`ProgressSink`] that timestamps each pipeline stage boundary
//! the engine reports (`detect`, `recognize`). Pairing consecutive timestamps with the
//! overall wall clock yields a load/detect/recognize breakdown without any library
//! change — a lightweight profiling aid for the benchmark loop and for users asking
//! where OCR time goes. Stage labels mirror the engine's `on_stage` strings.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use sceptre::ProgressSink;

/// Engine stage label for detection (matches `sceptre_engine::STAGE_DETECT`).
const STAGE_DETECT: &str = "detect";
/// Engine stage label for recognition (matches `sceptre_engine::STAGE_RECOGNIZE`).
const STAGE_RECOGNIZE: &str = "recognize";

/// A progress sink that records the instant each stage begins.
///
/// Shared as `Arc<StageTimer>`: the engine calls [`on_stage`](ProgressSink::on_stage)
/// (through `&self`), and the CLI reads the recorded events afterward via [`breakdown`].
#[derive(Default)]
pub struct StageTimer {
    events: Mutex<Vec<(String, Instant)>>,
}

/// Wall-clock split of a run into load/decode, detection, and recognition.
///
/// For a batch, `detect` and `recognize` are the summed per-image phases; `setup` is
/// everything before the first detection (model load on the first call + image decode).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Breakdown {
    /// Time before the first detection stage: one-time model load + image decode.
    pub setup: Duration,
    /// Total time spent in detection (CRAFT + grayscale + cropping).
    pub detect: Duration,
    /// Total time spent in recognition (CRNN + CTC).
    pub recognize: Duration,
    /// The full measured wall time of the run.
    pub total: Duration,
}

impl StageTimer {
    /// A fresh timer with no recorded events.
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute the stage breakdown against the run's `start` and `end` instants.
    pub fn breakdown(&self, start: Instant, end: Instant) -> Breakdown {
        let events = self.events.lock().expect("stage-timer mutex is not poisoned");
        let relative: Vec<(String, Duration)> = events
            .iter()
            .map(|(stage, at)| (stage.clone(), at.saturating_duration_since(start)))
            .collect();
        compute_breakdown(&relative, end.saturating_duration_since(start))
    }
}

impl ProgressSink for StageTimer {
    fn on_stage(&self, stage: &str) {
        self.events
            .lock()
            .expect("stage-timer mutex is not poisoned")
            .push((stage.to_string(), Instant::now()));
    }
}

/// Split `total` across stages from ordered `(stage, offset-from-start)` events.
///
/// Each event owns the interval from its own offset to the next event's (or to `total`
/// for the last); the interval before the first event is `setup`. Unknown stage labels
/// contribute to neither bucket but still delimit intervals.
fn compute_breakdown(events: &[(String, Duration)], total: Duration) -> Breakdown {
    let setup = events.first().map(|(_, at)| *at).unwrap_or(total);
    let mut detect = Duration::ZERO;
    let mut recognize = Duration::ZERO;
    for (index, (stage, at)) in events.iter().enumerate() {
        let next = events.get(index + 1).map(|(_, at)| *at).unwrap_or(total);
        let span = next.saturating_sub(*at);
        match stage.as_str() {
            STAGE_DETECT => detect += span,
            STAGE_RECOGNIZE => recognize += span,
            _ => {}
        }
    }
    Breakdown {
        setup,
        detect,
        recognize,
        total,
    }
}

/// Millisecond projection of a [`Breakdown`] for the `--format json` payload.
///
/// [`Duration`] has no stable JSON encoding worth committing to as a public contract,
/// so the report fixes the unit in the field names instead.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct TimingsReport {
    /// Load and decode time before the first detection stage.
    pub setup_ms: f64,
    /// Total detection time.
    pub detect_ms: f64,
    /// Total recognition time.
    pub recognize_ms: f64,
    /// Full measured wall time of the run.
    pub total_ms: f64,
}

impl From<&Breakdown> for TimingsReport {
    fn from(breakdown: &Breakdown) -> Self {
        let ms = |duration: Duration| duration.as_secs_f64() * 1000.0;
        Self {
            setup_ms: ms(breakdown.setup),
            detect_ms: ms(breakdown.detect),
            recognize_ms: ms(breakdown.recognize),
            total_ms: ms(breakdown.total),
        }
    }
}

/// Render the breakdown as a one-block stderr summary with millisecond precision.
pub fn render(breakdown: &Breakdown) -> String {
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    let pct = |d: Duration| {
        if breakdown.total.is_zero() {
            0.0
        } else {
            d.as_secs_f64() / breakdown.total.as_secs_f64() * 100.0
        }
    };
    format!(
        "timings (ms): total {:.1} | load+decode {:.1} ({:.0}%) | detect {:.1} ({:.0}%) | recognize {:.1} ({:.0}%)",
        ms(breakdown.total),
        ms(breakdown.setup),
        pct(breakdown.setup),
        ms(breakdown.detect),
        pct(breakdown.detect),
        ms(breakdown.recognize),
        pct(breakdown.recognize),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(value: u64) -> Duration {
        Duration::from_millis(value)
    }

    #[test]
    fn single_image_splits_setup_detect_recognize() {
        // detect at 100ms, recognize at 250ms, total 400ms: ~keep
        // setup=100, detect=250-100=150, recognize=400-250=150. ~keep
        let events = vec![
            (STAGE_DETECT.to_string(), ms(100)),
            (STAGE_RECOGNIZE.to_string(), ms(250)),
        ];
        let breakdown = compute_breakdown(&events, ms(400));
        assert_eq!(breakdown.setup, ms(100));
        assert_eq!(breakdown.detect, ms(150));
        assert_eq!(breakdown.recognize, ms(150));
        assert_eq!(breakdown.total, ms(400));
    }

    #[test]
    fn batch_sums_each_stage_across_images() {
        // Two images: detect@50, recog@120, detect@200, recog@260; total 300. ~keep
        // detect = (120-50)+(260-200)=70+60=130; recognize=(200-120)+(300-260)=80+40=120. ~keep
        let events = vec![
            (STAGE_DETECT.to_string(), ms(50)),
            (STAGE_RECOGNIZE.to_string(), ms(120)),
            (STAGE_DETECT.to_string(), ms(200)),
            (STAGE_RECOGNIZE.to_string(), ms(260)),
        ];
        let breakdown = compute_breakdown(&events, ms(300));
        assert_eq!(breakdown.setup, ms(50));
        assert_eq!(breakdown.detect, ms(130));
        assert_eq!(breakdown.recognize, ms(120));
    }

    #[test]
    fn no_events_attributes_everything_to_setup() {
        let breakdown = compute_breakdown(&[], ms(200));
        assert_eq!(breakdown.setup, ms(200));
        assert_eq!(breakdown.detect, Duration::ZERO);
        assert_eq!(breakdown.recognize, Duration::ZERO);
    }

    #[test]
    fn render_reports_total_and_percentages() {
        let breakdown = Breakdown {
            setup: ms(100),
            detect: ms(150),
            recognize: ms(150),
            total: ms(400),
        };
        let line = render(&breakdown);
        assert!(line.contains("total 400.0"), "{line}");
        assert!(line.contains("load+decode 100.0"), "{line}");
        assert!(line.contains("detect 150.0"), "{line}");
        assert!(line.contains("recognize 150.0"), "{line}");
    }
}
