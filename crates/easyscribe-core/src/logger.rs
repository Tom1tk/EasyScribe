//! Rotating file logging, ported from `src/logger.py`.
//!
//! One `EasyScribe_YYYYMMDD_HHMMSS.log` per run in the install's `logs/`
//! folder; the ten newest are kept. Logs contain file names and progress,
//! never transcript text.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use chrono::Local;

use crate::Result;

/// The logger is process-global. Startup calls `setup` once before the payload
/// is prepared so first-launch timings are not lost, then `app::startup` calls
/// it again with the resolved runtime paths. The second call must not rotate
/// the log or create a second file.
static SETUP_DONE: AtomicBool = AtomicBool::new(false);

struct FileLogger {
    file: Mutex<std::fs::File>,
    mirror_stdout: bool,
}

impl log::Log for FileLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::Level::Debug
    }

    fn log(&self, record: &log::Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let line = format!(
            "{} [{:<8}] {}: {}\n",
            Local::now().format("%Y-%m-%d %H:%M:%S"),
            record.level().as_str(),
            record.target(),
            record.args()
        );
        if let Ok(mut file) = self.file.lock() {
            let _ = file.write_all(line.as_bytes());
            let _ = file.flush();
        }
        if self.mirror_stdout {
            print!("{line}");
        }
    }

    fn flush(&self) {
        if let Ok(mut file) = self.file.lock() {
            let _ = file.flush();
        }
    }
}

/// Set up the process logger. Safe to call more than once; later calls are
/// ignored so an early setup does not create a second log file. A failed setup
/// leaves the flag clear so the caller can retry with the resolved paths.
pub fn setup(logs_dir: &Path) -> Result<()> {
    if SETUP_DONE.load(Ordering::SeqCst) {
        return Ok(());
    }
    setup_inner(logs_dir)?;
    SETUP_DONE.store(true, Ordering::SeqCst);
    Ok(())
}

fn setup_inner(logs_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(logs_dir)?;
    rotate(logs_dir);

    let timestamp = Local::now().format("%Y%m%d_%H%M%S");
    let log_file = logs_dir.join(format!("{}_{}.log", crate::APP_NAME, timestamp));
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_file)?;

    let logger = FileLogger {
        file: Mutex::new(file),
        mirror_stdout: !cfg!(windows),
    };
    // If another logger is already installed (tests, for example), keep it
    // rather than replacing it; the log file still exists for diagnostics.
    let _ = log::set_boxed_logger(Box::new(logger));
    log::set_max_level(log::LevelFilter::Debug);

    log::info!("{}", "=".repeat(60));
    log::info!("{} v{} starting", crate::APP_NAME, crate::APP_VERSION);
    log::info!("{}", "=".repeat(60));
    log::info!("Log file    : {}", log_file.display());
    Ok(())
}

/// Keep at most `MAX_LOG_FILES - 1` old logs before creating this run's file.
fn rotate(logs_dir: &Path) {
    let prefix = format!("{}_", crate::APP_NAME);
    let Ok(entries) = std::fs::read_dir(logs_dir) else {
        return;
    };
    let mut logs: Vec<(std::time::SystemTime, std::path::PathBuf)> = entries
        .filter_map(|e| e.ok())
        .filter(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            name.starts_with(&prefix) && name.ends_with(".log")
        })
        .filter_map(|e| {
            let modified = e.metadata().ok()?.modified().ok()?;
            Some((modified, e.path()))
        })
        .collect();
    logs.sort_by_key(|(time, _)| *time);
    while logs.len() >= crate::config::MAX_LOG_FILES {
        if let Some((_, oldest)) = logs.first()
            && std::fs::remove_file(oldest).is_err()
        {
            break;
        }
        logs.remove(0);
    }
}

/// The most recent log file, for the Privacy panel's "Open logs folder".
pub fn newest_log(logs_dir: &Path) -> Option<std::path::PathBuf> {
    std::fs::read_dir(logs_dir)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("log"))
        .max_by_key(|p| p.metadata().and_then(|m| m.modified()).ok())
}

/// Remove temporary WAV and JSON files, at startup and at exit.
pub fn clean_temp_files(temp_dir: &Path) {
    let Ok(entries) = std::fs::read_dir(temp_dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase);
        if matches!(ext.as_deref(), Some("wav" | "json")) {
            let _ = std::fs::remove_file(path);
        }
    }
}
