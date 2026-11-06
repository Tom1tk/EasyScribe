//! File transcription through whisper.cpp, in-process via `whisper-rs`.
//!
//! The Python app drove the `whisper-cli` subprocess. Keeping the engine in
//! process makes the single-exe payload smaller and gives the same beam-search
//! results without writing JSON to a temporary file; the transcript shape and
//! beam size (5) are unchanged.
//!
//! Stop is immediate: whisper.cpp's abort callback polls the shared cancel
//! token during decoding, so a long file can be stopped without waiting for
//! the decode to finish. A GPU decode failure is retried once on the CPU (see
//! `REFERENCE_BACKEND.md` §10 and the plan's audit disposition).

use std::path::Path;

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::audio::load_wav_float32;
use crate::model::Segment;
use crate::report::CancelToken;
use crate::{Error, Paths, Reporter, Result};

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

/// The GPU ggml will use, when this build has Vulkan and the machine has a
/// device. ggml lists discrete GPUs before integrated ones, matching the
/// Python app's `_pick_gpu`.
pub fn selected_gpu() -> Option<(i32, String)> {
    #[cfg(feature = "vulkan")]
    {
        let devices = whisper_rs::vulkan::list_devices();
        for device in &devices {
            log::info!(
                "whisper.cpp GPU {}: {} ({:.1} GB VRAM)",
                device.id,
                device.name,
                device.vram.total as f64 / 1_073_741_824.0
            );
        }
        devices.first().map(|d| (d.id, d.name.clone()))
    }
    #[cfg(not(feature = "vulkan"))]
    {
        None
    }
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

    reporter.log(&format!(
        "[whisper.cpp] CPU: {} | {} threads",
        cpu_name(),
        crate::config::num_threads()
    ));

    let gpu = selected_gpu();
    if let Some((id, name)) = &gpu {
        reporter.log(&format!("[whisper.cpp] Engine: Vulkan (device {id}: {name})"));
    } else {
        reporter.log("[whisper.cpp] Engine: CPU (whisper.cpp)");
    }

    let use_gpu = gpu.is_some();
    let first = decode(paths, &samples, cancel, reporter, use_gpu, gpu.as_ref().map(|g| g.0));
    let segments = match first {
        Ok(segments) => segments,
        Err(Error::Cancelled) => return Err(Error::Cancelled),
        Err(error) if use_gpu => {
            reporter.log(&format!(
                "[whisper.cpp] GPU decode failed ({error}). Retrying on the CPU…"
            ));
            log::warn!("whisper.cpp GPU decode failed, retrying on CPU: {error}");
            decode(paths, &samples, cancel, reporter, false, None)
                .map_err(|e| Error::WhisperCpp(format!("whisper.cpp failed: {e}")))?
        }
        Err(error) => return Err(Error::WhisperCpp(format!("whisper.cpp failed: {error}"))),
    };

    reporter.progress(1.0);
    reporter.log(&format!("[whisper.cpp] Decoded {} segment(s)", segments.len()));
    Ok(segments)
}

fn decode(
    paths: &Paths,
    samples: &[f32],
    cancel: &CancelToken,
    reporter: &dyn Reporter,
    use_gpu: bool,
    device: Option<i32>,
) -> Result<Vec<Segment>> {
    let mut context_params = WhisperContextParameters::default();
    context_params.use_gpu(use_gpu);
    if let Some(device) = device {
        context_params.gpu_device(device);
    }
    let context = WhisperContext::new_with_params(&paths.whispercpp_model, context_params)
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
    let cancelled = cancel.clone();
    params.set_abort_callback_safe(move || cancelled.is_cancelled());

    let started = std::time::Instant::now();
    let result = state.full(params, samples);
    if cancel.is_cancelled() {
        return Err(Error::Cancelled);
    }
    result.map_err(|e| Error::WhisperCpp(format!("whisper.cpp failed: {e}")))?;
    log::info!("whisper.cpp decode took {:.1}s", started.elapsed().as_secs_f32());
    let _ = reporter;

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
        // The registry value the Python app reads. Reading the registry would
        // add a crate for one log line; PROCESSOR_IDENTIFIER is equivalent.
        if let Ok(name) = std::env::var("PROCESSOR_IDENTIFIER") {
            return name;
        }
    }
    std::env::consts::ARCH.to_string()
}
