//! Backend x accelerator benchmark matrix: an intra-sceptre liveness and throughput
//! comparison, not a parity gate.
//!
//! Each `#[ignore]`d test builds one [`Reader`] for a single backend/accelerator pairing,
//! runs a warm-up call, then times a handful of steady-state repeats, and writes a small
//! JSON report naming the accelerator [`runtime_info_for`] says actually *registered* —
//! not merely the one requested. `backend_agreement.rs` remains the only correctness bar;
//! this file only asks "does it run, and how fast" (see ADR 0035, which amends the
//! "CUDA is compiled and never executed" clause of ADR 0032 with the same distinction).
//!
//! `canvas_size` is fixed at 1024 for every leg. This is methodology, not a detail: tract
//! cannot shape-infer dynamic CRAFT (ADR 0027) and pads to a fixed square, so at the 2560
//! default its optimization cost would dominate and swamp any cross-backend comparison.
//!
//! Opt-in and heavy, mirroring `backend_agreement.rs`: it loads real models and is gated
//! behind `SCEPTRE_REQUIRE_MODELS`, plus the `test_documents` corpus. An absent corpus or
//! model set is a `#[ignore]`d skip printed to stdout, never a substituted image.
#![cfg(feature = "ort")]
#![allow(
    clippy::print_stdout,
    reason = "the report and skip reasons are this test's own output"
)]

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use sceptre::{Accelerator, Backend, Language, OcrConfig, ProgressSink, ReadOptions, Reader, runtime_info_for};
use serde::Serialize;

/// Detection canvas shared by every leg (see module docs: fixed by methodology, not tuned
/// per backend).
const CANVAS_SIZE: u32 = 1024;
/// One untimed call to page in models/backends/caches before steady-state timing starts.
const WARMUP_REPEATS: usize = 1;
/// Steady-state repeats a leg's median is drawn from.
const STEADY_STATE_REPEATS: usize = 5;
/// The fixed image subset every leg measures, so legs are comparable to each other.
const IMAGES: &[(&str, Language)] = &[("english.png", Language::English), ("cyrillic.png", Language::Cyrillic)];

/// Truthy `SCEPTRE_REQUIRE_MODELS` opts this heavy, model-backed suite in.
fn require_models() -> bool {
    match std::env::var("SCEPTRE_REQUIRE_MODELS") {
        Ok(value) => !matches!(value.trim().to_ascii_lowercase().as_str(), "" | "0" | "false" | "no"),
        Err(_) => false,
    }
}

/// The repository root, two levels up from this crate's manifest directory
/// (`<root>/crates/sceptre`).
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Directory holding the corpus images: `TEST_DOCUMENTS_DIR` when set, otherwise the
/// `test_documents` submodule checked out at the repository root. Mirrors
/// `backend_agreement.rs`'s resolver.
fn images_dir() -> PathBuf {
    std::env::var_os("TEST_DOCUMENTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root().join("test_documents"))
        .join("images")
}

/// Where per-leg JSON reports land: `SCEPTRE_BACKEND_MATRIX_OUTPUT_DIR` when set,
/// otherwise a gitignored `benchmark-results/backends/` at the repository root.
fn output_dir() -> PathBuf {
    std::env::var_os("SCEPTRE_BACKEND_MATRIX_OUTPUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root().join("benchmark-results").join("backends"))
}

fn config_for(backend: Backend, accelerator: Accelerator) -> OcrConfig {
    let mut config = OcrConfig::default();
    config.model.backend = backend;
    config.model.accelerator = accelerator;
    config.detection.canvas_size = CANVAS_SIZE;
    config
}

fn models_available(config: &OcrConfig) -> bool {
    sceptre::model_manifest(config)
        .map(|manifest| manifest.iter().all(|info| info.cached))
        .unwrap_or(false)
}

/// A progress sink that timestamps each pipeline stage boundary, mirroring the CLI's
/// `--timings` sink (`crates/sceptre-cli/src/timing.rs`) so this harness reports the same
/// setup/detect/recognize split without depending on the binary crate.
#[derive(Default)]
struct StageTimer {
    events: Mutex<Vec<(String, Instant)>>,
}

impl StageTimer {
    fn reset(&self) {
        self.events.lock().expect("stage-timer mutex is not poisoned").clear();
    }

    fn breakdown(&self, start: Instant, end: Instant) -> LegTiming {
        let events = self.events.lock().expect("stage-timer mutex is not poisoned");
        let relative: Vec<(&str, Duration)> = events
            .iter()
            .map(|(stage, at)| (stage.as_str(), at.saturating_duration_since(start)))
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

fn compute_breakdown(events: &[(&str, Duration)], total: Duration) -> LegTiming {
    let setup = events.first().map(|(_, at)| *at).unwrap_or(total);
    let mut detect = Duration::ZERO;
    let mut recognize = Duration::ZERO;
    for (index, (stage, at)) in events.iter().enumerate() {
        let next = events.get(index + 1).map(|(_, at)| *at).unwrap_or(total);
        let span = next.saturating_sub(*at);
        match *stage {
            "detect" => detect += span,
            "recognize" => recognize += span,
            _ => {}
        }
    }
    LegTiming {
        setup_ms: to_ms(setup),
        detect_ms: to_ms(detect),
        recognize_ms: to_ms(recognize),
        total_ms: to_ms(total),
    }
}

fn to_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
struct LegTiming {
    setup_ms: f64,
    detect_ms: f64,
    recognize_ms: f64,
    total_ms: f64,
}

#[derive(Serialize)]
struct ImageReport {
    image: String,
    language: String,
    repeats: usize,
    median: LegTiming,
}

#[derive(Serialize)]
struct LegReport {
    leg: String,
    canvas_size: u32,
    sceptre_version: &'static str,
    os: &'static str,
    arch: &'static str,
    backend: Backend,
    accelerator_requested: Accelerator,
    accelerator_registered: Option<Accelerator>,
    ort_version: Option<String>,
    /// True when this leg was built with an unoptimized profile. `ort` is prebuilt C++ and
    /// indifferent to the Rust profile, while `tract` and `candle` are pure Rust, so such a
    /// leg's cross-backend numbers are not comparable and must not be quoted.
    unoptimized_build: bool,
    /// One-time cost of `Reader::builder().build_warmed()`, kept separate from
    /// steady-state inference per the matrix methodology.
    model_load_ms: f64,
    images: Vec<ImageReport>,
}

/// Skip (never substitute) when the corpus or models are unavailable; run and write a
/// report otherwise.
fn run_leg(leg: &str, backend: Backend, accelerator: Accelerator) {
    if !require_models() {
        println!("skipping backend-matrix leg '{leg}': SCEPTRE_REQUIRE_MODELS is not set");
        return;
    }
    let corpus_dir = images_dir();
    if !corpus_dir.is_dir() {
        println!("skipping backend-matrix leg '{leg}': test_documents corpus not found at {corpus_dir:?}");
        return;
    }

    let config = config_for(backend, accelerator);
    assert!(
        models_available(&config),
        "SCEPTRE_REQUIRE_MODELS is set but the models for leg '{leg}' are not cached"
    );

    let runtime = runtime_info_for(&config.model).expect("the leg configuration is valid");
    assert_eq!(
        runtime.backend, backend,
        "leg '{leg}' must describe the backend it configured"
    );

    let timer = std::sync::Arc::new(StageTimer::default());
    let load_start = Instant::now();
    let reader = Reader::builder()
        .config(config)
        .progress(timer.clone())
        .build_warmed()
        .unwrap_or_else(|err| panic!("leg '{leg}' builds and warms up: {err}"));
    let model_load_ms = to_ms(load_start.elapsed());

    let mut images = Vec::with_capacity(IMAGES.len());
    for (image_file, language) in IMAGES {
        let image_path = corpus_dir.join(image_file);
        if !image_path.is_file() {
            println!("skipping image '{image_file}' for leg '{leg}': not present in the corpus");
            continue;
        }

        for _ in 0..WARMUP_REPEATS {
            timer.reset();
            reader
                .readtext(&image_path, &ReadOptions::default())
                .unwrap_or_else(|err| panic!("leg '{leg}' warms up on {image_file}: {err}"));
        }

        let mut totals = Vec::with_capacity(STEADY_STATE_REPEATS);
        let mut setups = Vec::with_capacity(STEADY_STATE_REPEATS);
        let mut detects = Vec::with_capacity(STEADY_STATE_REPEATS);
        let mut recognizes = Vec::with_capacity(STEADY_STATE_REPEATS);
        for _ in 0..STEADY_STATE_REPEATS {
            timer.reset();
            let start = Instant::now();
            reader
                .readtext(&image_path, &ReadOptions::default())
                .unwrap_or_else(|err| panic!("leg '{leg}' runs end to end over {image_file}: {err}"));
            let end = Instant::now();
            let breakdown = timer.breakdown(start, end);
            setups.push(breakdown.setup_ms);
            detects.push(breakdown.detect_ms);
            recognizes.push(breakdown.recognize_ms);
            totals.push(breakdown.total_ms);
        }

        images.push(ImageReport {
            image: (*image_file).to_string(),
            language: format!("{language:?}"),
            repeats: STEADY_STATE_REPEATS,
            median: LegTiming {
                setup_ms: median(&mut setups),
                detect_ms: median(&mut detects),
                recognize_ms: median(&mut recognizes),
                total_ms: median(&mut totals),
            },
        });
    }

    let report = LegReport {
        leg: leg.to_string(),
        canvas_size: CANVAS_SIZE,
        sceptre_version: runtime.sceptre_version,
        os: runtime.os,
        arch: runtime.arch,
        backend: runtime.backend,
        accelerator_requested: runtime.accelerator_requested,
        accelerator_registered: runtime.accelerator_registered,
        ort_version: runtime.ort.and_then(|ort| ort.version),
        unoptimized_build: cfg!(debug_assertions),
        model_load_ms,
        images,
    };

    let dir = output_dir();
    std::fs::create_dir_all(&dir).unwrap_or_else(|err| panic!("create {dir:?}: {err}"));
    let path = dir.join(format!("{leg}.json"));
    let json = serde_json::to_string_pretty(&report).expect("serialize the leg report");
    std::fs::write(&path, json).unwrap_or_else(|err| panic!("write {path:?}: {err}"));
    println!("backend-matrix leg '{leg}' wrote {path:?}");
}

#[test]
#[ignore = "heavy: real models, real inference, opt in via SCEPTRE_REQUIRE_MODELS"]
fn bench_ort_cpu() {
    run_leg("ort-cpu", Backend::Ort, Accelerator::Cpu);
}

#[cfg(feature = "tract")]
#[test]
#[ignore = "heavy: real models, real inference, opt in via SCEPTRE_REQUIRE_MODELS"]
fn bench_tract_cpu() {
    run_leg("tract-cpu", Backend::Tract, Accelerator::Cpu);
}

#[cfg(feature = "candle")]
#[test]
#[ignore = "heavy: real models, real inference, opt in via SCEPTRE_REQUIRE_MODELS"]
fn bench_candle_cpu() {
    run_leg("candle-cpu", Backend::Candle, Accelerator::Cpu);
}

#[cfg(feature = "ort-coreml")]
#[test]
#[ignore = "heavy: real models, real inference, opt in via SCEPTRE_REQUIRE_MODELS"]
fn bench_ort_coreml() {
    run_leg("ort-coreml", Backend::Ort, Accelerator::CoreMl);
}

/// No hosted CI runner offers Metal (macOS runners are VMs with no Neural Engine either),
/// so this compiles everywhere `candle-metal` is enabled and only runs, meaningfully, on
/// real hardware or a Metal-capable macOS runner. See ADR 0035's open questions.
#[cfg(feature = "candle-metal")]
#[test]
#[ignore = "heavy: real models, real inference, opt in via SCEPTRE_REQUIRE_MODELS"]
fn bench_candle_metal() {
    run_leg("candle-metal", Backend::Candle, Accelerator::Metal);
}

#[cfg(feature = "ort-cuda")]
#[test]
#[ignore = "heavy: real models, real inference, opt in via SCEPTRE_REQUIRE_MODELS"]
fn bench_ort_cuda() {
    run_leg("ort-cuda", Backend::Ort, Accelerator::Cuda);
}

#[cfg(feature = "candle-cuda")]
#[test]
#[ignore = "heavy: real models, real inference, opt in via SCEPTRE_REQUIRE_MODELS"]
fn bench_candle_cuda() {
    run_leg("candle-cuda", Backend::Candle, Accelerator::Cuda);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_compute_median_of_odd_length() {
        let mut values = vec![3.0, 1.0, 2.0];
        assert_eq!(median(&mut values), 2.0);
    }

    #[test]
    fn should_compute_median_of_even_length() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(median(&mut values), 2.5);
    }

    #[test]
    fn should_split_setup_detect_recognize() {
        let events = vec![
            ("detect", Duration::from_millis(100)),
            ("recognize", Duration::from_millis(250)),
        ];
        let breakdown = compute_breakdown(&events, Duration::from_millis(400));
        assert_eq!(breakdown.setup_ms, 100.0);
        assert_eq!(breakdown.detect_ms, 150.0);
        assert_eq!(breakdown.recognize_ms, 150.0);
        assert_eq!(breakdown.total_ms, 400.0);
    }
}
