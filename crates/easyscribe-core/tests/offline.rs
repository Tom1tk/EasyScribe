//! The shipped app must not contain a runtime HTTP stack.
//!
//! Rune's Linux accessibility backend pulls a small async runtime, so this
//! checks for the network crates themselves rather than tokio. The Windows
//! build has no HTTP client; ffmpeg is separately restricted to `file:`.

use easyscribe_core::offline::FORBIDDEN_RUNTIME_CRATES;

#[test]
fn no_network_crate_is_a_runtime_dependency() {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let output = std::process::Command::new(cargo)
        .args([
            "tree",
            "-p",
            "easyscribe",
            "-e",
            "normal",
            "--prefix",
            "none",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("cargo tree");
    assert!(output.status.success(), "cargo tree failed");
    let tree = String::from_utf8_lossy(&output.stdout);
    for name in FORBIDDEN_RUNTIME_CRATES {
        assert!(
            !tree
                .lines()
                .any(|line| line.split_whitespace().next() == Some(*name)),
            "runtime dependency `{name}` is forbidden"
        );
    }
}

#[test]
fn ffmpeg_only_reads_local_files() {
    let source = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/ffmpeg.rs"),
    )
    .expect("ffmpeg.rs");
    assert!(source.contains(r#""-protocol_whitelist", "file""#));
}
