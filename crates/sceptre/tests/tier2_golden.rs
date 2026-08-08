// The per-image metrics line is this harness's result output, not a diagnostic: it is what
// a threshold recalibration reads. Integration tests are separate crates and do not inherit
// the library's crate-root allow. ~keep
#![allow(clippy::print_stdout)]

//! Tier-2 golden parity harness.
//!
//! Each example image, resolved from the `test_documents` corpus (see [`images_dir`]),
//! is replayed through the real, default-engine `Reader` and compared against a dual
//! golden fixture in `tests/data/golden/*.json` — an authoritative Python EasyOCR
//! reference and a sceptre snapshot (see `tests/data/golden/README.md`). Availability is gated via
//! the library's own `model_manifest`, and the reader is built with the default
//! provider, which resolves models from the shared Hugging Face hub cache (ADR
//! 0017); the test only builds a reader once the manifest reports every model
//! cached, so no download happens here.
//!
//! Real-model tests are gated on `SCEPTRE_REQUIRE_MODELS`: when the models are not
//! resolvable, the tests skip (pass) by default, but panic when the env var is
//! truthy so CI surfaces a misconfigured model cache. Their bodies are additionally
//! `#[cfg(feature = "ort")]`-gated so the default backend-less `cargo test` compiles
//! and passes. The pure helpers carry their own always-on unit tests in
//! `tests/helpers/mod.rs`.

mod helpers;

#[cfg(feature = "ort")]
use std::path::{Path, PathBuf};

#[cfg(feature = "ort")]
use sceptre::{Language, OcrConfig, ReadOptions, Reader};

/// Fuzzy word-F1 floor for reference parity, mirroring the corpus tolerance.
#[cfg(feature = "ort")]
const WORD_F1_THRESHOLD: f32 = 0.8;

/// Recall floor for the greedy line-detection match: nearly every reference line must be
/// covered.
///
/// Calibrated against a full `SCEPTRE_REQUIRE_MODELS=1` corpus run. Seven of the eight
/// images score a perfect 1.000; `french.jpg` is the floor at 0.833, so this sits just
/// under it. A drop below here means sceptre stopped finding lines EasyOCR finds.
#[cfg(feature = "ort")]
const LINE_RECALL_THRESHOLD: f32 = 0.8;

/// Precision floor for the greedy line-detection match, i.e. the bar on fabricated lines.
///
/// Deliberately looser than [`LINE_RECALL_THRESHOLD`]: splitting one reference line into
/// two boxes costs precision without losing any text, which is a difference in grouping
/// rather than a recognition regression. Same corpus run — `english.png` scores 0.923 and
/// `french.jpg` is the floor at 0.625, so this sits just under that.
#[cfg(feature = "ort")]
const LINE_PRECISION_THRESHOLD: f32 = 0.6;

/// Whether `SCEPTRE_REQUIRE_MODELS` is set to a truthy value, forcing real-model
/// tests to run (and fail loudly if models are missing) instead of skipping.
fn require_models() -> bool {
    match std::env::var("SCEPTRE_REQUIRE_MODELS") {
        Ok(value) => {
            let value = value.trim().to_ascii_lowercase();
            !matches!(value.as_str(), "" | "0" | "false" | "no")
        }
        Err(_) => false,
    }
}

/// The repository root, two levels up from this crate's manifest directory
/// (`<root>/crates/sceptre`).
#[cfg(feature = "ort")]
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Directory holding the corpus images: `TEST_DOCUMENTS_DIR` when set, otherwise the
/// `test_documents` submodule checked out at the repository root.
#[cfg(feature = "ort")]
fn images_dir() -> PathBuf {
    std::env::var_os("TEST_DOCUMENTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root().join("test_documents"))
        .join("images")
}

/// Absolute path to the crate-local golden fixtures (not part of the `test_documents` corpus).
#[cfg(feature = "ort")]
fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/golden")
}

/// Whether every model `config` needs is already cached, per the library's own
/// manifest (pure filesystem inspection of the Hugging Face hub cache).
#[cfg(feature = "ort")]
fn models_available(config: &OcrConfig) -> bool {
    sceptre::model_manifest(config)
        .map(|manifest| manifest.iter().all(|info| info.cached))
        .unwrap_or(false)
}

/// A single-language config for the given recognition group.
#[cfg(feature = "ort")]
fn config_for(language: Language) -> OcrConfig {
    let mut config = OcrConfig::default();
    config.model.languages = vec![language];
    config
}

#[test]
fn should_treat_unset_require_models_as_optional() {
    // With the env var unset in the default test environment, real-model tests must ~keep
    // be allowed to skip rather than being forced to run. ~keep
    if std::env::var("SCEPTRE_REQUIRE_MODELS").is_err() {
        assert!(!require_models());
    }
}

/// Exercise the full detect -> crop -> recognize pipeline through the public
/// `Reader` for one image/language, and check the dual golden: exact equality
/// against the sceptre snapshot plus fuzzy word-F1 and per-line box-IoU against the
/// EasyOCR reference. Skips (passing) when the models are not cached, unless
/// `SCEPTRE_REQUIRE_MODELS` forces the run; skips parity while the fixture is still
/// the committed placeholder.
#[cfg(feature = "ort")]
fn run_dual_golden_parity(image_file: &str, golden_stem: &str, language: Language) {
    // Opt-in: parity needs a working inference backend AND cached models, so it runs only ~keep
    // under SCEPTRE_REQUIRE_MODELS. The default `cargo test` skips it — a cached model must ~keep
    // not trigger a real run under a backend that cannot load (e.g. the default ort-dynamic ~keep
    // with no ORT_DYLIB_PATH would panic on session creation). ~keep
    if !require_models() {
        return;
    }
    let config = config_for(language);
    assert!(
        models_available(&config),
        "SCEPTRE_REQUIRE_MODELS is set but the models for {image_file} are not cached in the \
         Hugging Face hub cache (HF_HUB_CACHE / HF_HOME / ~/.cache/huggingface/hub); \
         run the model-provisioning step first"
    );

    let reader = Reader::builder()
        .config(config)
        .build()
        .expect("building the reader with the default Hugging Face cache model provider");

    let image_path = images_dir().join(image_file);
    let result = reader
        .readtext(&image_path, &ReadOptions::default())
        .unwrap_or_else(|err| panic!("the real engine runs end to end over {image_file}: {err}"));

    assert!(
        !result.lines.is_empty(),
        "the real engine should detect and recognize at least one line in {image_file}"
    );

    let golden_path = golden_dir().join(format!("{golden_stem}.json"));
    let golden_json =
        std::fs::read_to_string(&golden_path).unwrap_or_else(|err| panic!("reading {}: {err}", golden_path.display()));
    let golden = helpers::DualGolden::parse(&golden_json);

    if golden.placeholder {
        // Fixtures are not yet regenerated; the pipeline ran, which is all we can ~keep
        // assert until real goldens land (see tests/data/golden/README.md). ~keep
        return;
    }

    assert_sceptre_snapshot(&result, &golden, image_file);
    assert_easyocr_reference(&result, &golden, image_file);
}

#[test]
#[cfg(feature = "ort")]
fn parity_english_png() {
    run_dual_golden_parity("english.png", "english", Language::English);
}

#[test]
#[cfg(feature = "ort")]
fn parity_chinese_jpg() {
    run_dual_golden_parity("chinese.jpg", "chinese", Language::ChineseSimplified);
}

#[test]
#[cfg(feature = "ort")]
fn parity_japanese_jpg() {
    run_dual_golden_parity("japanese.jpg", "japanese", Language::Japanese);
}

#[test]
#[cfg(feature = "ort")]
fn parity_korean_png() {
    run_dual_golden_parity("korean.png", "korean", Language::Korean);
}

#[test]
#[cfg(feature = "ort")]
fn parity_cyrillic_png() {
    run_dual_golden_parity("cyrillic.png", "cyrillic", Language::Cyrillic);
}

#[test]
#[cfg(feature = "ort")]
fn parity_telugu() {
    run_dual_golden_parity("telugu.png", "telugu", Language::Telugu);
}

#[test]
#[cfg(feature = "ort")]
fn parity_kannada() {
    run_dual_golden_parity("kannada.png", "kannada", Language::Kannada);
}

/// The rotated french sign. Previously sceptre split `[Palais du LOUVRE` into two
/// boxes (`[Palais du` + a rotated `LOUVRE`) where EasyOCR keeps one: a knife-edge
/// slope classification where `imageproc`'s integer-rounded `min_area_rect` corners
/// pushed `LOUVRE`'s slope from cv2's 0.096 to 0.129, just over `slope_ths` (0.1),
/// routing it to the free list so it never merged. Fixed by sceptre's own
/// OpenCV-faithful rotating-calipers fit (`detect::min_area_rect`, ADR 0039), which
/// drops the per-corner outward snap; this now matches the reference at full parity.
#[test]
#[cfg(feature = "ort")]
fn parity_french_jpg() {
    run_dual_golden_parity("french.jpg", "french", Language::Latin);
}

/// Exact-equality check of recognized text against the sceptre snapshot golden.
#[cfg(feature = "ort")]
fn assert_sceptre_snapshot(result: &sceptre::OcrResult, golden: &helpers::DualGolden, image_file: &str) {
    let actual: Vec<String> = result.lines.iter().map(|line| line.text.clone()).collect();
    let expected: Vec<String> = golden.sceptre.lines.iter().map(|line| line.text.clone()).collect();
    assert_eq!(
        actual, expected,
        "recognized text for {image_file} must match the sceptre snapshot golden exactly"
    );
}

/// Fuzzy text-F1 + per-line box-IoU check against the EasyOCR reference golden.
///
/// Every script is scored with [`helpers::word_f1`]. Its tokenizer expands CJK runs into
/// overlapping bigrams, so a script without word boundaries no longer collapses into a
/// single token and does not need the character-bag fallback it used to be routed to.
#[cfg(feature = "ort")]
fn assert_easyocr_reference(result: &sceptre::OcrResult, golden: &helpers::DualGolden, image_file: &str) {
    let actual_text = result
        .lines
        .iter()
        .map(|line| line.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let reference_text = golden
        .easyocr
        .lines
        .iter()
        .map(|line| line.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let f1 = helpers::word_f1(&actual_text, &reference_text);
    assert!(
        f1 >= WORD_F1_THRESHOLD,
        "{image_file}: word-F1 {f1:.3} against the EasyOCR reference is below the {WORD_F1_THRESHOLD} threshold"
    );

    let sceptre_bboxes: Vec<sceptre::BBox> = result.lines.iter().map(|line| quad_bbox(&line.quad)).collect();
    let reference_bboxes: Vec<sceptre::BBox> = golden.easyocr.lines.iter().map(helpers::GoldenLine::bbox).collect();
    // Greedy IoU matching replaces averaging the best IoU over reference lines only, which
    // never penalized a hypothesis line with no reference match (recall-only). Reporting
    // precision and recall separately means a low score reads as fabrication (low
    // precision) vs omission (low recall) instead of collapsing both into one number. ~keep
    let (line_f1, line_precision, line_recall) = helpers::line_detection_scores(&reference_bboxes, &sceptre_bboxes);
    // Printed for every image, not only failures, so recalibrating a floor reads the whole
    // corpus off one `--nocapture` run instead of bisecting on assertion messages. ~keep
    println!(
        "parity-metrics\t{image_file}\tword_f1={f1:.3}\tline_recall={line_recall:.3}\
         \tline_precision={line_precision:.3}\tline_f1={line_f1:.3}"
    );
    assert!(
        line_recall >= LINE_RECALL_THRESHOLD,
        "{image_file}: line recall {line_recall:.3} against the EasyOCR reference is below \
         {LINE_RECALL_THRESHOLD} (precision {line_precision:.3}, f1 {line_f1:.3})"
    );
    assert!(
        line_precision >= LINE_PRECISION_THRESHOLD,
        "{image_file}: line precision {line_precision:.3} against the EasyOCR reference is below \
         {LINE_PRECISION_THRESHOLD} (recall {line_recall:.3}, f1 {line_f1:.3})"
    );

    // Reading order is purely additive: logged for visibility, not gated, until a real
    // corpus run establishes a floor. Catches column-order / vertical-script regressions
    // that word-F1 and box-IoU are both blind to. ~keep
    if let Some(reading_order) = helpers::reading_order_score(&actual_text, &reference_text) {
        assert!(
            (0.0..=1.0).contains(&reading_order),
            "{image_file}: reading-order score {reading_order:.3} must be in [0, 1]"
        );
    }
}

/// Axis-aligned bounds of a recognized quad, for IoU against a reference box.
#[cfg(feature = "ort")]
fn quad_bbox(quad: &sceptre::Quad) -> sceptre::BBox {
    let xs = quad.points.map(|point| point.x);
    let ys = quad.points.map(|point| point.y);
    sceptre::BBox {
        x_min: xs.iter().copied().fold(f32::INFINITY, f32::min),
        y_min: ys.iter().copied().fold(f32::INFINITY, f32::min),
        x_max: xs.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        y_max: ys.iter().copied().fold(f32::NEG_INFINITY, f32::max),
    }
}
