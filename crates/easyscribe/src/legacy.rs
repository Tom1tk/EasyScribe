//! Import settings and find crash-recovery files from a v3 install.
//!
//! The Python app kept `recordings/settings.json`, its crash-safe PCM work
//! folder and `logs/` beside the executable. The Rust app keeps user data in
//! `%LOCALAPPDATA%\\EasyScribe`, so a first run must look for those old files
//! without moving or deleting them.

use std::path::{Path, PathBuf};

use easyscribe_core::{Error, Paths, Result};

/// Places a v3 install may live, nearest first.
pub fn candidates(paths: &Paths) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        out.push(dir.to_path_buf());
        out.push(dir.join("EasyScribe"));
    }
    if let Some(local) = dirs::data_local_dir() {
        out.push(local.join("Programs").join("EasyScribe"));
    }
    if let Some(home) = dirs::home_dir() {
        out.push(home.join("EasyScribe"));
    }
    out.retain(|dir| looks_like_v3(dir));
    out.sort();
    out.dedup();
    out.retain(|dir| dir != &paths.base_dir);
    out
}

fn looks_like_v3(dir: &Path) -> bool {
    dir.join("recordings").is_dir()
        || dir.join("logs").is_dir()
        || dir.join("EasyScribe.exe").is_file()
}

/// Copy a legacy `recordings/settings.json` into the new data root when the
/// user has not chosen anything yet. Returns true when imported.
pub fn import(paths: &Paths) -> Result<bool> {
    if paths.settings_file.exists() {
        return Ok(false);
    }
    for dir in candidates(paths) {
        let legacy = dir.join("recordings").join("settings.json");
        if legacy.is_file() {
            if let Some(parent) = paths.settings_file.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::copy(&legacy, &paths.settings_file).map_err(Error::Io)?;
            log::info!("Imported v3 settings from {}", legacy.display());
            return Ok(true);
        }
    }
    Ok(false)
}

/// Folders that may still contain crash-safe `recording_*.pcm` + `.json`.
pub fn recordings_dirs(paths: &Paths) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = candidates(paths)
        .into_iter()
        .map(|dir| dir.join("recordings"))
        .filter(|dir| dir.is_dir())
        .collect();
    out.push(paths.recordings_work_dir.clone());
    out.sort();
    out.dedup();
    out
}
