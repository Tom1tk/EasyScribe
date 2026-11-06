//! File transcription through whisper.cpp, in-process via `whisper-rs`.
//!
//! The Python app drove the `whisper-cli` subprocess. Keeping the engine in
//! process makes the single-exe payload smaller and gives the same beam-search
//! results without writing JSON to a temporary file; the transcript shape and
//! beam size (5) are unchanged.

use std::path::Path;

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::audio::load_wav_float32;
use crate::model::Segment;
use crate::{Error, Paths, Reporter, Result};
use crate::report::CancelToken;

pub const BEAM_SIZE: i32 = 5;

pub fn is_available(paths: &Paths) -> bool {
    paths.whispercpp_model.is_file()
}

pub fn validate(paths: &Paths) -> Result<()> {
    if !paths.whispercpp_model.is_file() {
        return Err(Error::WhisperCpp(format!(
            "Whisper GGUF model not found at: {}",
            paths.whispercpp_model.display()
        )));
    }
    Ok(())
}

/// Transcribe a mono 16 kHz WAV with whisper.cpp beam search.
pub fn transcribe_file(
    paths: &Paths,
    audio_path: &Path,
    cancel: &CancelToken,
    reporter: &dyn Reporter,
) -> Result<Vec<Segment>> {
    cancel.check()?;
    validate(paths)?;

    reporter.log(&format!(
        "[whisper.cpp] Transcribing with beam_size={BEAM_SIZE}: {}",
        audio_path.file_name().and_then(|n| n.to_str()).unwrap_or("?")
    ));
    let (samples, rate) = load_wav_float32(audio_path)
        .map_err(|e| Error::Transcription(format!("Could not load audio: {e}")))?;
    if rate != 16_000 {
        return Err(Error::Transcription(format!(
            "whisper.cpp needs 16 kHz audio, got {rate} Hz"
        )));
    }
    cancel.check()?;

    let context = WhisperContext::new_with_params(
        &paths.whispercpp_model,
        WhisperContextParameters::default(),
    )
    .map_err(|e| Error::WhisperCpp(format!("Could not load Whisper model: {e}")))?;
    let mut state = context
        .create_state()
        .map_err(|e| Error::WhisperCpp(format!("Could not create Whisper state: {e}")))?;

    let mut params = FullParams::new(SamplingStrategy::BeamSearch {
        beam_size: BEAM_SIZE,
        patience: -1.0,
    });
    params.set_language(Some("en"));
    params.set_translate(false);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_n_threads(crate::config::num_threads() as i32);
    reporter.log("[whisper.cpp] Engine: CPU (whisper.cpp)");
    reporter.log(&format!(
        "[whisper.cpp] CPU: {} | {} threads",
        cpu_name(),
        crate::config::num_threads()
    ));

    state
        .full(params, &samples)
        .map_err(|e| Error::WhisperCpp(format!("whisper.cpp failed: {e}")))?;

    let mut segments = Vec::new();
    for segment in state.as_iter() {
        let text = segment
            .to_str_lossy()
            .map_err(|e| Error::WhisperCpp(format!("invalid Whisper text: {e}")))?
            .trim()
            .to_owned();
        if text.is_empty() {
            continue;
        }
        segments.push(Segment {
            start: segment.start_timestamp() as f64 / 100.0,
            end: segment.end_timestamp() as f64 / 100.0,
            text,
        });
    }
    reporter.progress(1.0);
    reporter.log(&format!("[whisper.cpp] Decoded {} segment(s)", segments.len()));
    Ok(segments)
}

/// Best-effort processor name for the log, matching the Python device lines.
pub fn cpu_name() -> String {
    #[cfg(target_os = "linux")]
    {
        if let Ok(text) = std::fs::read_to_string("/proc/cpuinfo") {
            if let Some(line) = text.lines().find(|l| l.starts_with("model name")) {
                return line.split_once(':').map(|(_, v)| v.trim().to_owned()).unwrap_or_default();
            }
        }
    }
    #[cfg(windows)]
    {
        // The registry value the Python app reads. Use PowerShell-free winreg
        // would add a crate; `platform.processor` equivalent is enough here.
        if let Ok(name) = std::env::var("PROCESSOR_IDENTIFIER") {
            return name;
        }
    }
    std::env::consts::ARCH.to_string()
}
