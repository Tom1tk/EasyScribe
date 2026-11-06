//! Application paths and constants. Mirrors `src/config.py` of v3.2.3.
//!
//! Release builds run from an extracted payload directory; development builds
//! run from the repository root. `EASYSCRIBE_BASE_DIR` overrides both, which
//! tests and the headless renderer use.

use std::path::{Path, PathBuf};

use crate::Result;

pub const APP_NAME: &str = "EasyScribe";
pub const APP_VERSION: &str = env!("CARGO_PKG_VERSION");

pub const VAD_SAMPLE_RATE: u32 = 16_000;
pub const VAD_CHUNK_SAMPLES: usize = 512;

pub const NUM_THREADS_FALLBACK: usize = 4;
pub const MIN_FREE_DISK_BYTES: u64 = 512 * 1024 * 1024;
pub const MAX_LOG_FILES: usize = 10;

pub const SUPPORTED_EXTENSIONS: &[&str] = &[
    "mp4", "mkv", "mov", "avi", "webm", "mp3", "wav", "m4a", "flac", "ogg", "opus", "aac",
];

/// One thread per physical core, clamped to 4..=8, as the Python app does.
pub fn num_threads() -> usize {
    // A hidden diarization worker receives the parent's thread count. Honor it
    // so the child uses the same budget as the process that spawned it.
    if let Some(value) = std::env::var_os("EASYSCRIBE_NUM_THREADS")
        && let Some(threads) = value.to_str().and_then(|value| value.parse::<usize>().ok())
        && threads > 0
    {
        return threads.min(64);
    }
    let logical = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    (logical / 2).clamp(4, 8)
}

/// Resolve the application's data directories.
#[derive(Debug, Clone)]
pub struct Paths {
    pub base_dir: PathBuf,
    pub models_dir: PathBuf,
    pub whisper_model_dir: PathBuf,
    pub diarization_dir: PathBuf,
    pub vad_model: PathBuf,
    pub separator_model: PathBuf,
    pub embedding_model: PathBuf,
    pub whisper_encoder: PathBuf,
    pub whisper_decoder: PathBuf,
    pub whisper_tokens: PathBuf,
    pub ffmpeg_dir: PathBuf,
    pub ffmpeg_bin: PathBuf,
    pub ffprobe_bin: PathBuf,
    pub whispercpp_dir: PathBuf,
    pub whispercpp_bin: PathBuf,
    pub whispercpp_model: PathBuf,
    pub data_dir: PathBuf,
    pub logs_dir: PathBuf,
    pub temp_dir: PathBuf,
    pub recordings_work_dir: PathBuf,
    pub settings_file: PathBuf,
    pub default_recordings_dir: PathBuf,
    pub assets_dir: PathBuf,
    pub icon_ico: PathBuf,
    pub icon_png: PathBuf,
}

impl Paths {
    /// Resolve paths for the running app and create the runtime directories.
    pub fn discover() -> Result<Self> {
        if let Some(base) = std::env::var_os("EASYSCRIBE_BASE_DIR") {
            let base = resolve_env_path(base);
            // Worker processes inherit both so they read the same extracted
            // runtime while writing settings/temp files to the parent's data
            // directory instead of a second copy beside the models.
            if let Some(data) = std::env::var_os("EASYSCRIBE_DATA_DIR") {
                return Self::for_runtime(base, resolve_env_path(data));
            }
            return Self::at(base);
        }
        let prepared = crate::payload::prepare()?;
        Self::for_runtime(prepared.runtime_dir, prepared.data_dir)
    }

    /// Resolve paths rooted at an explicit directory, with user data in the
    /// same directory (development and tests).
    pub fn at(base: PathBuf) -> Result<Self> {
        Self::for_runtime(base.clone(), base)
    }

    /// Resolve an extracted runtime directory, with user data elsewhere.
    pub fn for_runtime(base: PathBuf, data_dir: PathBuf) -> Result<Self> {
        let models_dir = base.join("models");
        let whisper_model_dir = models_dir.join("whisper");
        let diarization_dir = models_dir.join("diarization");
        let ffmpeg_dir = base.join("ffmpeg");
        let exe = if cfg!(windows) {
            "ffmpeg.exe"
        } else {
            "ffmpeg"
        };
        let probe = if cfg!(windows) {
            "ffprobe.exe"
        } else {
            "ffprobe"
        };
        let bundled_ffmpeg = ffmpeg_dir.join(exe);
        let bundled_probe = ffmpeg_dir.join(probe);
        // Development builds may use a system ffmpeg. An installed runtime
        // must use the bundled copy: a missing bundled tool is a startup error,
        // never a silent PATH fallback.
        let development = base == data_dir;
        let (ffmpeg_bin, ffprobe_bin) = if bundled_ffmpeg.is_file() {
            (bundled_ffmpeg, bundled_probe)
        } else if development {
            (PathBuf::from("ffmpeg"), PathBuf::from("ffprobe"))
        } else {
            (bundled_ffmpeg, bundled_probe)
        };
        let whispercpp_dir = base.join("whispercpp");
        let whispercpp_bin = whispercpp_dir.join(if cfg!(windows) {
            "whisper-cli.exe"
        } else {
            "whisper-cli"
        });
        let logs_dir = data_dir.join("logs");
        let temp_dir = data_dir.join("temp");
        let recordings_work_dir = data_dir.join("recordings");
        std::fs::create_dir_all(&logs_dir)?;
        std::fs::create_dir_all(&temp_dir)?;
        std::fs::create_dir_all(&recordings_work_dir)?;

        Ok(Self {
            models_dir: models_dir.clone(),
            whisper_model_dir: whisper_model_dir.clone(),
            diarization_dir: diarization_dir.clone(),
            vad_model: models_dir.join("silero_vad.onnx"),
            separator_model: diarization_dir.join("segmentation.onnx"),
            embedding_model: diarization_dir.join("embedding.onnx"),
            whisper_encoder: whisper_model_dir.join("turbo-encoder.int8.onnx"),
            whisper_decoder: whisper_model_dir.join("turbo-decoder.int8.onnx"),
            whisper_tokens: whisper_model_dir.join("turbo-tokens.txt"),
            ffmpeg_bin,
            ffprobe_bin,
            ffmpeg_dir,
            whispercpp_bin,
            whispercpp_model: whispercpp_dir.join("ggml-large-v3-turbo-q5_0.bin"),
            whispercpp_dir,
            logs_dir,
            temp_dir: temp_dir.clone(),
            settings_file: recordings_work_dir.join("settings.json"),
            recordings_work_dir,
            data_dir,
            default_recordings_dir: default_recordings_dir(),
            assets_dir: base.join("assets"),
            icon_ico: base.join("assets").join("EasyScribe.ico"),
            icon_png: base.join("assets").join("EasyScribe.png"),
            base_dir: base,
        })
    }

    /// The three Whisper ONNX files are present.
    pub fn whisper_model_available(&self) -> bool {
        self.whisper_encoder.is_file()
            && self.whisper_decoder.is_file()
            && self.whisper_tokens.is_file()
    }

    /// Speaker models are present, so the Name the speakers option is enabled.
    pub fn diarization_available(&self) -> bool {
        self.separator_model.is_file() && self.embedding_model.is_file()
    }

    /// A bundled whisper.cpp binary + GGUF model are present.
    pub fn whispercpp_available(&self) -> bool {
        self.whispercpp_bin.is_file() && self.whispercpp_model.is_file()
    }

    /// The same checks `transcriber.validate_model_directory()` makes; returns
    /// human-readable missing-file lines.
    pub fn validate_models(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if !self.vad_model.is_file() {
            errors.push(format!("Missing: {}", self.vad_model.display()));
        }
        for path in [
            &self.whisper_encoder,
            &self.whisper_decoder,
            &self.whisper_tokens,
        ] {
            if !path.is_file() {
                errors.push(format!("Missing: {}", path.display()));
            }
        }
        errors
    }

    /// A unique temporary WAV path.
    pub fn temp_wav(&self) -> PathBuf {
        self.temp_dir
            .join(format!("{}.wav", uuid::Uuid::new_v4().simple()))
    }
}

/// Where the app reads models and tools from.
pub fn dev_base_dir() -> PathBuf {
    discover_base_dir()
}

/// Resolve an environment-provided directory. Relative paths are anchored to
/// the executable, never to a shortcut's working directory (often System32).
pub(crate) fn resolve_env_path(value: std::ffi::OsString) -> PathBuf {
    let path = PathBuf::from(value);
    if path.is_absolute() {
        return path;
    }
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join(&path)))
        .unwrap_or(path)
}

fn discover_base_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("EASYSCRIBE_BASE_DIR") {
        return resolve_env_path(dir);
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
        && (dir.join("models").is_dir() || dir.join("assets").is_dir())
    {
        return dir.to_path_buf();
    }
    // `crates/easyscribe-core` → repository root.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf()
}

/// The user's Documents folder, with the same fallback the Python app uses.
pub fn default_recordings_dir() -> PathBuf {
    let documents = dirs::document_dir()
        .filter(|p| p.is_dir())
        .or_else(|| dirs::home_dir().filter(|p| p.is_dir()))
        .unwrap_or_else(|| PathBuf::from("."));
    documents.join(APP_NAME)
}

/// True when `path` has one of the supported media extensions (case-insensitive).
pub fn is_supported_media(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| {
            SUPPORTED_EXTENSIONS
                .iter()
                .any(|x| e.eq_ignore_ascii_case(x))
        })
        .unwrap_or(false)
}
