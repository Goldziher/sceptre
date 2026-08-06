//! Offline smoke tests for the `sceptre` CLI binary.
//!
//! Every test is offline: none execute real OCR, load model weights, or hit the
//! network. They exercise argument parsing, help/version output, shell completion
//! generation, the filesystem-only model manifest, and colour stripping.

use assert_cmd::Command;
use predicates::prelude::*;

/// Build a fresh command handle for the `sceptre` binary.
fn sceptre() -> Command {
    Command::cargo_bin("sceptre").expect("sceptre binary should be built for tests")
}

#[test]
fn should_print_version() {
    sceptre()
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains(env!("CARGO_PKG_VERSION")));
}

#[test]
fn should_list_subcommands_in_help() {
    sceptre()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("run"))
        .stdout(predicate::str::contains("detect"))
        .stdout(predicate::str::contains("recognize"))
        .stdout(predicate::str::contains("models"))
        .stdout(predicate::str::contains("completions"));
}

#[test]
fn should_emit_shell_completions() {
    sceptre()
        .args(["completions", "bash"])
        .assert()
        .success()
        .stdout(predicate::str::is_empty().not())
        .stdout(predicate::str::contains("sceptre"));
}

#[test]
fn should_fail_run_on_missing_image() {
    sceptre().args(["run", "/no/such/file.png"]).assert().failure();
}

#[test]
fn should_show_variadic_images_arg_in_run_help() {
    sceptre()
        .args(["run", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("IMAGES"));
}

#[test]
fn should_report_each_missing_image_and_fail_without_aborting_batch() {
    sceptre()
        .args(["run", "/no/such/a.png", "/no/such/b.png"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("/no/such/a.png"))
        .stderr(predicate::str::contains("/no/such/b.png"));
}

#[test]
fn should_emit_json_array_for_batch_of_missing_images() {
    sceptre()
        .args(["run", "/no/such/a.png", "/no/such/b.png", "--format", "json"])
        .assert()
        .failure()
        .stdout(predicate::str::starts_with("[").trim())
        .stdout(predicate::str::contains("/no/such/a.png"))
        .stdout(predicate::str::contains("/no/such/b.png"))
        .stdout(predicate::str::contains("error"));
}

#[test]
fn should_fail_detect_on_missing_image() {
    sceptre().args(["detect", "/no/such/file.png"]).assert().failure();
}

#[test]
fn should_fail_recognize_on_missing_image() {
    sceptre().args(["recognize", "/no/such/file.png"]).assert().failure();
}

#[test]
fn should_reject_out_of_range_text_threshold_at_parse() {
    sceptre()
        .args(["run", "some.png", "--text-threshold", "2.0"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("between 0 and 1"));
}

#[test]
fn should_list_models_offline() {
    sceptre()
        .args(["models", "list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("craft_mlt_25k"))
        .stdout(predicate::str::contains("english_g2"));
}

#[test]
fn should_list_models_as_json() {
    sceptre()
        .args(["models", "list", "--format", "json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("craft_mlt_25k"))
        .stdout(predicate::str::starts_with("[").trim());
}

#[test]
fn should_strip_color_when_no_color_set() {
    sceptre()
        .args(["models", "list"])
        .env("NO_COLOR", "1")
        .assert()
        .success()
        .stdout(predicate::str::contains("\u{1b}").not());
}

/// Every gen2 recognizer artifact, i.e. what `--all` must expand to.
const ALL_RECOGNIZERS: [&str; 8] = [
    "english_g2",
    "latin_g2",
    "zh_sim_g2",
    "japanese_g2",
    "korean_g2",
    "cyrillic_g2",
    "telugu_g2",
    "kannada_g2",
];

#[test]
fn should_list_every_language_when_all_is_set() {
    let mut assertion = sceptre()
        .args(["models", "list", "--all"])
        .assert()
        .success()
        .stdout(predicate::str::contains("craft_mlt_25k"));
    for model in ALL_RECOGNIZERS {
        assertion = assertion.stdout(predicate::str::contains(model));
    }
}

#[test]
fn should_list_only_the_selected_language_without_all() {
    sceptre()
        .args(["models", "list", "--lang", "korean"])
        .assert()
        .success()
        .stdout(predicate::str::contains("korean_g2"))
        .stdout(predicate::str::contains("kannada_g2").not());
}

#[test]
fn should_reject_all_combined_with_an_explicit_language() {
    sceptre()
        .args(["models", "list", "--all", "--lang", "korean"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("cannot be used with"));
}

#[test]
fn should_document_all_in_models_download_help() {
    sceptre()
        .args(["models", "download", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--all"));
}

#[test]
fn should_report_the_environment_as_text() {
    sceptre()
        .args(["env"])
        .assert()
        .success()
        .stdout(predicate::str::contains("version"))
        .stdout(predicate::str::contains("backend"))
        .stdout(predicate::str::contains("accelerator"))
        .stdout(predicate::str::contains("craft_mlt_25k"));
}

#[test]
fn should_report_the_environment_as_json_with_the_benchmark_contract_keys() {
    let output = sceptre().args(["env", "--format", "json"]).assert().success();
    let stdout = String::from_utf8(output.get_output().stdout.clone()).expect("stdout should be UTF-8");
    let payload: serde_json::Value = serde_json::from_str(&stdout).expect("env must emit valid JSON");

    assert_eq!(
        payload["version"],
        env!("CARGO_PKG_VERSION"),
        "the version key must carry the binary's version"
    );
    for key in ["backend", "accelerator", "onnxruntime", "models"] {
        assert!(payload.get(key).is_some(), "missing contract key `{key}`: {stdout}");
    }
    assert_eq!(payload["backend"], "ort", "the default build reports the ort backend");

    let models = payload["models"].as_array().expect("models must be an array");
    assert!(!models.is_empty(), "models must not be empty: {stdout}");
    for model in models {
        for key in ["name", "repo", "sha256"] {
            let value = model[key].as_str().unwrap_or_default();
            assert!(!value.is_empty(), "model entry missing `{key}`: {model}");
        }
    }
}

#[test]
fn should_report_the_requested_accelerator_in_the_environment() {
    let output = sceptre()
        .args(["env", "--format", "json", "--accelerator", "auto"])
        .assert()
        .success();
    let stdout = String::from_utf8(output.get_output().stdout.clone()).expect("stdout should be UTF-8");
    let payload: serde_json::Value = serde_json::from_str(&stdout).expect("env must emit valid JSON");

    assert_eq!(payload["accelerator_requested"], "auto");
}

#[test]
fn should_document_the_accelerator_flag_in_run_help() {
    sceptre()
        .args(["run", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--accelerator"));
}

#[test]
fn should_reject_an_unknown_accelerator_at_parse() {
    sceptre()
        .args(["run", "some.png", "--accelerator", "quantum"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("quantum"));
}

#[test]
fn should_reject_out_of_range_link_threshold_at_parse() {
    sceptre()
        .args(["run", "some.png", "--link-threshold", "2.0"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("between 0 and 1"));
}
