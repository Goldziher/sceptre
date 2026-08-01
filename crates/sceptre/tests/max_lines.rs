//! Enforces the workspace module-size cap: every `crates/*/src/**/*.rs` file stays at
//! or under [`MAX_LINES`] lines. Small modules stay reviewable; when a file approaches
//! the cap, split it into submodules (see the `module-size-cap` rule) rather than
//! raising the cap.
//!
//! This is the real enforcement behind that rule. poly cannot count lines — its custom
//! rules are ast-grep pattern matches — so the cap lives here, in the standard
//! `cargo test` gate CI runs on every matrix leg (feature-agnostic: it only reads files).

use std::path::{Path, PathBuf};

/// The per-file line cap for every `crates/*/src/**/*.rs` module.
const MAX_LINES: usize = 1000;

/// The workspace root: two levels up from this crate's manifest directory
/// (`<root>/crates/<crate>` → `<root>`).
fn workspace_root() -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("crate manifest dir must sit at <root>/crates/<crate>")
        .to_path_buf()
}

/// Every member crate's `src` directory under `<root>/crates/*`.
fn crate_src_dirs(root: &Path) -> Vec<PathBuf> {
    let crates_dir = root.join("crates");
    let entries =
        std::fs::read_dir(&crates_dir).unwrap_or_else(|error| panic!("read_dir {}: {error}", crates_dir.display()));
    let mut dirs = Vec::new();
    for entry in entries {
        let path = entry.expect("read directory entry").path();
        let src = path.join("src");
        if src.is_dir() {
            dirs.push(src);
        }
    }
    dirs
}

/// Collect every `.rs` file under `dir`, recursing into subdirectories.
fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir).unwrap_or_else(|error| panic!("read_dir {}: {error}", dir.display()));
    for entry in entries {
        let path = entry.expect("read directory entry").path();
        if path.is_dir() {
            collect_rs_files(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn no_source_file_exceeds_the_module_line_cap() {
    let root = workspace_root();
    let src_dirs = crate_src_dirs(&root);
    assert!(
        !src_dirs.is_empty(),
        "found no crates/*/src directories under {}",
        root.display()
    );

    let mut files = Vec::new();
    for dir in &src_dirs {
        collect_rs_files(dir, &mut files);
    }
    assert!(!files.is_empty(), "found no .rs files under any crates/*/src");

    let mut offenders: Vec<(PathBuf, usize)> = Vec::new();
    for file in &files {
        let contents = std::fs::read_to_string(file).unwrap_or_else(|error| panic!("read {}: {error}", file.display()));
        let lines = contents.lines().count();
        if lines > MAX_LINES {
            offenders.push((file.clone(), lines));
        }
    }
    offenders.sort_by_key(|offender| std::cmp::Reverse(offender.1));

    let report = offenders
        .iter()
        .map(|(path, lines)| {
            let shown = path.strip_prefix(&root).unwrap_or(path);
            format!("  {} ({lines} lines)", shown.display())
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        offenders.is_empty(),
        "these crates/*/src/**/*.rs files exceed the {MAX_LINES}-line cap — split them into submodules:\n{report}",
    );
}
