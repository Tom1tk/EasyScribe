//! Offline guarantee.
//!
//! The Python app monkey-patched `socket` at startup. A Rust app has no
//! interpreter-level socket layer to replace, so the guarantee is structural:
//! the shipped app links no HTTP client, performs no DNS lookup and has no
//! update check. This module sets the same opt-out environment variables and
//! exposes a source/dependency audit used by the test suite.
//!
//! The only subprocess that could reach the network is ffmpeg; it always runs
//! with `-protocol_whitelist file` (see `ffmpeg.rs`).

/// Opt-out signals honoured by common libraries. None ship with EasyScribe;
/// this is defence in depth and keeps any future dependency from phoning home.
pub const OFFLINE_ENV: &[(&str, &str)] = &[
    ("DO_NOT_TRACK", "1"),
    ("HF_HUB_OFFLINE", "1"),
    ("HF_HUB_DISABLE_TELEMETRY", "1"),
    ("TRANSFORMERS_OFFLINE", "1"),
    ("HF_DATASETS_OFFLINE", "1"),
    ("NO_PROXY", "*"),
    ("no_proxy", "*"),
];

/// Install the offline environment before any other work.
pub fn install() {
    for (key, value) in OFFLINE_ENV {
        // Safe in edition 2024: the process is single-threaded at this point.
        unsafe { std::env::set_var(key, value) };
    }
}

/// Crates that would open a network connection at runtime. The shipped app
/// must never contain one; the audit test in `tests/offline.rs` checks this.
pub const FORBIDDEN_RUNTIME_CRATES: &[&str] = &[
    "reqwest",
    "ureq",
    "hyper",
    "curl",
    "isahc",
    "surf",
    "attohttpc",
    "awc",
    "tungstenite",
    "tokio-tungstenite",
    "rustls",
    "native-tls",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_opt_out_is_set_after_install() {
        install();
        for (key, value) in OFFLINE_ENV {
            assert_eq!(std::env::var(key).ok().as_deref(), Some(*value));
        }
    }
}
