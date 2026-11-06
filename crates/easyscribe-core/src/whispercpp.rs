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
use std::sync::Arc;

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

/// A Vulkan device ggml can decode on.
#[derive(Clone, Debug)]
pub struct GpuInfo {
    pub id: i32,
    pub name: String,
    /// Stable identity stored after a failed decode. It pairs the name with
    /// the VRAM size, which changes when the driver/GPU changes.
    pub identity: String,
}

/// The GPU ggml will use, when this build has Vulkan and the machine has a
/// device. ggml lists discrete GPUs before integrated ones, matching the
/// Python app's `_pick_gpu`. A `preferred` index wins when it exists.
pub fn selected_gpu(preferred: Option<i32>) -> Option<GpuInfo> {
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
        let device = preferred
            .and_then(|id| devices.iter().find(|d| d.id == id))
            .or_else(|| devices.first())?;
        Some(GpuInfo {
            id: device.id,
            name: device.name.clone(),
            identity: format!("{}|{}", device.name, device.vram.total),
        })
    }
    #[cfg(not(feature = "vulkan"))]
    {
        let _ = preferred;
        None
    }
}

/// Transcribe a mono 16 kHz WAV with whisper.cpp beam search.
pub fn transcribe_file(
    paths: &Paths,
    audio_path: &Path,
    cancel: &CancelToken,
    reporter: &Arc<dyn Reporter>,
) -> Result<Vec<Segment>> {
    cancel.check()?;
    validate(paths)?;

    reporter.log(&format!(
        "[whisper.cpp] Transcribing with beam_size={BEAM_SIZE}: {}",
        audio_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("?")
    ));
    let (samples, rate) = load_wav_float32(audio_path)
        .map_err(|e| Error::Transcription(format!("Could not load audio: {e}")))?;
    let samples = std::sync::Arc::new(samples);
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

    let mut settings = crate::settings::Settings::load(paths.settings_file.clone());
    let regions = speech_regions(paths, &samples, cancel, reporter.as_ref())?;
    if regions.len() > 1
        || regions
            .first()
            .is_some_and(|(start, end)| *start > 0 || *end < samples.len())
    {
        let covered: usize = regions.iter().map(|(start, end)| end - start).sum();
        reporter.log(&format!(
            "[whisper.cpp] Speech gate: decoding {:.1} min of speech in {} region(s), skipping {:.1} min of silence",
            covered as f64 / 16_000.0 / 60.0,
            regions.len(),
            (samples.len().saturating_sub(covered)) as f64 / 16_000.0 / 60.0
        ));
    }

    let gpu = selected_gpu(settings.load_gpu_device());
    let disabled = settings.load_gpu_disabled_for();
    let gpu_usable = gpu
        .as_ref()
        .filter(|gpu| disabled.as_deref() != Some(gpu.identity.as_str()));
    if let Some(gpu) = gpu_usable {
        reporter.log(&format!(
            "[whisper.cpp] Engine: Vulkan (device {}: {})",
            gpu.id, gpu.name
        ));
    } else if gpu.is_some() {
        reporter.log("[whisper.cpp] Engine: CPU (this GPU failed before)");
    } else {
        reporter.log("[whisper.cpp] Engine: CPU (whisper.cpp)");
    }

    let use_gpu = gpu_usable.is_some();
    let device = gpu_usable.map(|g| g.id);
    let first =
        decode_on_fresh_thread(paths, &samples, &regions, cancel, reporter, use_gpu, device);
    let segments = match first {
        Ok(segments) => segments,
        Err(Error::Cancelled) => return Err(Error::Cancelled),
        Err(error) if use_gpu => {
            reporter.log(&format!(
                "[whisper.cpp] GPU decode failed ({error}). Retrying on the CPU…"
            ));
            log::warn!("whisper.cpp GPU decode failed, retrying on CPU: {error}");
            if let Some(gpu) = gpu_usable {
                if let Err(save_error) = settings.save_gpu_disabled_for(Some(&gpu.identity)) {
                    log::warn!("Could not remember the failed GPU: {save_error}");
                } else {
                    reporter.log(&format!(
                        "[whisper.cpp] Using the CPU for {} until the graphics driver changes.",
                        gpu.name
                    ));
                }
            }
            decode_on_fresh_thread(paths, &samples, &regions, cancel, reporter, false, None)
                .map_err(|e| Error::WhisperCpp(format!("whisper.cpp failed: {e}")))?
        }
        Err(error) => return Err(Error::WhisperCpp(format!("whisper.cpp failed: {error}"))),
    };

    reporter.progress(1.0);
    reporter.log(&format!(
        "[whisper.cpp] Decoded {} segment(s)",
        segments.len()
    ));
    Ok(segments)
}

/// Split a long file at very long silent gaps (30 s+) and trim silence at the
/// edges, using the same Silero model as the sherpa path. Short files, files
/// without a VAD model and files with little silence keep one whole-file decode,
/// which preserves the byte-identical golden transcript.
fn speech_regions(
    paths: &Paths,
    samples: &[f32],
    cancel: &CancelToken,
    _reporter: &dyn Reporter,
) -> Result<Vec<(usize, usize)>> {
    let whole = vec![(0, samples.len())];
    if samples.len() < 30 * 16_000 || !paths.vad_model.is_file() {
        return Ok(whole);
    }
    // The pre-whisper gate is deliberately a little more sensitive than live
    // mode: 0.4 / 160 ms keeps quiet first words while still trimming silence.
    let mut detector = match crate::sherpa_engine::build_vad_with(paths, 60.0, 0.4, 0.16) {
        Ok(detector) => detector,
        Err(error) => {
            log::warn!("Speech gate unavailable, decoding the whole file: {error}");
            return Ok(whole);
        }
    };
    let bounds = match crate::sherpa_engine::file_segments(&mut detector, samples, cancel) {
        Ok(bounds) => bounds,
        Err(Error::Cancelled) => return Err(Error::Cancelled),
        Err(error) => {
            log::warn!("Speech gate failed, decoding the whole file: {error}");
            return Ok(whole);
        }
    };
    Ok(merge_gate_regions(&bounds, samples.len()))
}

/// Turn VAD bounds into whisper decode regions. Ordinary pauses stay in one
/// region so Whisper keeps its context; only a 30 s+ silence starts a new one.
/// A file with little leading/trailing silence and no long gap stays whole,
/// preserving the byte-identical golden transcript for ordinary recordings.
fn merge_gate_regions(bounds: &[(usize, usize)], total: usize) -> Vec<(usize, usize)> {
    let whole = vec![(0, total)];
    if bounds.is_empty() {
        return whole;
    }
    let leading = bounds[0].0;
    let trailing = total.saturating_sub(bounds.last().map(|b| b.1).unwrap_or(0));
    let has_long_gap = bounds
        .windows(2)
        .any(|pair| pair[1].0.saturating_sub(pair[0].1) >= 30 * 16_000);
    if leading < 2 * 16_000 && trailing < 2 * 16_000 && !has_long_gap {
        return whole;
    }

    let pad = (crate::vad::VAD_PAD_SEC * 16_000.0) as usize;
    let mut regions: Vec<(usize, usize)> = Vec::new();
    for &(start, end) in bounds {
        let start = start.saturating_sub(pad);
        let end = (end + pad).min(total);
        match regions.last_mut() {
            Some(last) if start.saturating_sub(last.1) < 30 * 16_000 => {
                last.1 = last.1.max(end);
            }
            _ => regions.push((start, end)),
        }
    }
    if regions.is_empty() { whole } else { regions }
}

/// Run whisper.cpp on a fresh thread. ONNX Runtime leaves thread-local state
/// behind after a sherpa model is loaded, and a later whisper.cpp encode on
/// that same thread fails. Each decode gets a clean thread; the sample buffer
/// is shared through an `Arc`, so long files are not duplicated.
fn decode_on_fresh_thread(
    paths: &Paths,
    samples: &std::sync::Arc<Vec<f32>>,
    regions: &[(usize, usize)],
    cancel: &CancelToken,
    reporter: &Arc<dyn Reporter>,
    use_gpu: bool,
    device: Option<i32>,
) -> Result<Vec<Segment>> {
    cancel.check()?;
    let paths = paths.clone();
    let samples = samples.clone();
    let regions = regions.to_vec();
    let cancel = cancel.clone();
    let reporter = reporter.clone();
    std::thread::Builder::new()
        .name("WhisperDecode".into())
        .spawn(move || {
            decode_regions(
                &paths,
                samples.as_slice(),
                &regions,
                &cancel,
                &reporter,
                use_gpu,
                device,
            )
        })
        .map_err(|e| Error::WhisperCpp(format!("Could not start whisper.cpp: {e}")))?
        .join()
        .map_err(|_| Error::WhisperCpp("whisper.cpp worker stopped unexpectedly".into()))?
}

fn decode_regions(
    paths: &Paths,
    samples: &[f32],
    regions: &[(usize, usize)],
    cancel: &CancelToken,
    reporter: &Arc<dyn Reporter>,
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

    let mut segments = Vec::new();
    let started = std::time::Instant::now();
    for (start, end) in regions.iter().copied() {
        cancel.check()?;
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
        let progress_reporter = reporter.clone();
        let total_samples = samples.len().max(1) as f64;
        let region_start = start as f64 / total_samples;
        let region_weight = end.saturating_sub(start) as f64 / total_samples;
        let mut last_percent = -1;
        params.set_progress_callback_safe(move |percent| {
            if percent != last_percent {
                last_percent = percent;
                progress_reporter.progress(
                    region_start + region_weight * (percent.clamp(0, 100) as f64 / 100.0),
                );
            }
        });

        let result = state.full(params, &samples[start..end]);
        if cancel.is_cancelled() {
            return Err(Error::Cancelled);
        }
        result.map_err(|e| Error::WhisperCpp(format!("whisper.cpp failed: {e}")))?;

        let offset = start as f64 / 16_000.0;
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
                start: offset + segment.start_timestamp() as f64 / 100.0,
                end: offset + segment.end_timestamp() as f64 / 100.0,
                text,
            });
        }
    }
    log::info!(
        "whisper.cpp decode took {:.1}s across {} region(s)",
        started.elapsed().as_secs_f32(),
        regions.len()
    );
    Ok(segments)
}

/// Best-effort processor name for the log, matching the Python device lines.
pub fn cpu_name() -> String {
    #[cfg(target_os = "linux")]
    {
        if let Ok(text) = std::fs::read_to_string("/proc/cpuinfo")
            && let Some(line) = text.lines().find(|l| l.starts_with("model name"))
        {
            return line
                .split_once(':')
                .map(|(_, v)| v.trim().to_owned())
                .unwrap_or_default();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_keeps_short_files_and_ordinary_pauses_whole() {
        let total = 32 * 16_000;
        let bounds = [(16_000, 20 * 16_000), (25 * 16_000, 31 * 16_000)];
        assert_eq!(merge_gate_regions(&bounds, total), vec![(0, total)]);
    }

    #[test]
    #[ignore = "needs the development models and EasyScribe/tests/data/test_speech.wav"]
    fn long_silence_gate_preserves_the_transcript() {
        let base = Paths::at(crate::config::dev_base_dir()).expect("paths");
        let source = std::env::var_os("EASYSCRIBE_TEST_SPEECH")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../../EasyScribe/tests/data/test_speech.wav")
            });
        if !source.is_file() || !base.vad_model.is_file() || !base.whispercpp_model.is_file() {
            return;
        }
        let (samples, rate) = load_wav_float32(&source).unwrap();
        assert_eq!(rate, 16_000);
        let silence = vec![0.0_f32; 40 * 16_000];
        let mut padded = Vec::with_capacity(silence.len() * 2 + samples.len());
        padded.extend_from_slice(&silence);
        padded.extend_from_slice(&samples);
        padded.extend_from_slice(&silence);

        let regions = speech_regions(
            &base,
            &padded,
            &CancelToken::new(),
            &crate::report::SilentReporter,
        )
        .unwrap();
        assert_eq!(regions.len(), 1);
        assert!(regions[0].0 <= silence.len());
        assert!(regions[0].1 >= silence.len() + samples.len() * 3 / 4);

        let dir = tempfile::tempdir().unwrap();
        let wav = dir.path().join("padded.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&wav, spec).unwrap();
        for sample in &padded {
            writer
                .write_sample((sample.clamp(-1.0, 1.0) * 32768.0) as i16)
                .unwrap();
        }
        writer.finalize().unwrap();
        let reporter: Arc<dyn Reporter> = Arc::new(crate::report::SilentReporter);
        let segments = transcribe_file(&base, &wav, &CancelToken::new(), &reporter).unwrap();
        let text = segments
            .iter()
            .map(|segment| segment.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(text.contains("quarterly results"), "{text}");
    }

    #[test]
    fn gate_splits_only_at_very_long_silence() {
        let total = 120 * 16_000;
        let bounds = [(5 * 16_000, 15 * 16_000), (70 * 16_000, 80 * 16_000)];
        let regions = merge_gate_regions(&bounds, total);
        assert_eq!(regions.len(), 2);
        assert!(regions[0].0 < 5 * 16_000);
        assert!(regions[0].1 >= 15 * 16_000);
        assert!(regions[1].0 <= 70 * 16_000);
        assert!(regions[1].1 >= 80 * 16_000);
    }
}
