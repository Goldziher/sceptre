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

#[test]
fn should_reject_out_of_range_link_threshold_at_parse() {
    sceptre()
        .args(["run", "some.png", "--link-threshold", "2.0"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("between 0 and 1"));
}
