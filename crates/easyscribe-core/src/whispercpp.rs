//! File transcription through whisper.cpp. The bundled `whisper-cli` child
//! process is the primary engine (see `whispercli`); this module owns the
//! in-process `whisper-rs` fallback, which is used when the payload has only
//! the GGUF weights and in Linux development/tests.
//!
//! The in-process path keeps the same beam size (5) and transcript shape as
//! the CLI. Stop is cooperative: whisper.cpp polls the shared cancel token at
//! graph boundaries and while decoding, so a long file can be stopped without
//! waiting for the whole decode to finish. A GPU decode failure is retried
//! once on the CPU (see `REFERENCE_BACKEND.md` §10 and the plan's audit
//! disposition).

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

/// A short label for the log, matching the actual engine that will run.
pub fn engine_label(paths: &Paths) -> &'static str {
    if crate::whispercli::is_available(paths) {
        "whisper-cli"
    } else {
        "whisper.cpp in-process"
    }
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

/// Transcribe a mono 16 kHz WAV. The bundled whisper-cli is the primary path
/// (the same Vulkan binary the Python release uses); the in-process engine is
/// the fallback for development builds and tests that have no CLI.
pub fn transcribe_file(
    paths: &Paths,
    audio_path: &Path,
    cancel: &CancelToken,
    reporter: &Arc<dyn Reporter>,
) -> Result<Vec<Segment>> {
    if crate::whispercli::is_available(paths) {
        crate::whispercli::transcribe_file(paths, audio_path, cancel, reporter)
    } else {
        reporter.detail("Built-in engine: the bundled whisper-cli is not available");
        log::warn!("whisper-cli is not available, using the in-process engine");
        transcribe_in_process(paths, audio_path, cancel, reporter)
    }
}

/// Transcribe with the whisper.cpp library linked into this executable.
/// This path has no Vulkan and no child process; it exists so the app still
/// works when the payload has only the GGUF weights, and so CI can test the
/// engine independently of whisper-cli.
pub fn transcribe_in_process(
    paths: &Paths,
    audio_path: &Path,
    cancel: &CancelToken,
    reporter: &Arc<dyn Reporter>,
) -> Result<Vec<Segment>> {
    cancel.check()?;
    validate(paths)?;
    reporter.status(crate::Status::SlowFallback);
    reporter.log(
        "[Warning] Using the built-in whisper.cpp engine. It has no GPU and is much slower than the bundled whisper-cli; this can take many times the audio length.",
    );

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
    let regions = speech_regions(paths, &samples, cancel, reporter.as_ref(), true)?;
    if std::env::var_os("EASYSCRIBE_WHISPER_TRACE").is_some() {
        eprintln!(
            "[whisper.cpp] {} samples at {rate} Hz, {} region(s) {regions:?}, threads={}",
            samples.len(),
            regions.len(),
            crate::config::num_threads()
        );
    }
    if regions.is_empty() {
        reporter.progress(1.0);
        reporter.log("[whisper.cpp] No speech found. Nothing to transcribe.");
        log::info!(
            "whisper.cpp: no speech detected in {}",
            audio_path.display()
        );
        return Ok(Vec::new());
    }
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
/// edges, using the same Silero model as the sherpa path. Short files and files
/// without a VAD model keep one whole-file decode, which preserves the
/// byte-identical golden transcript. An empty VAD result means the file has no
/// speech at all: no regions, no hallucinated "Thank you." transcript.
pub(crate) fn speech_regions(
    paths: &Paths,
    samples: &[f32],
    cancel: &CancelToken,
    reporter: &dyn Reporter,
    split_long: bool,
) -> Result<Vec<(usize, usize)>> {
    if samples.is_empty() {
        return Ok(Vec::new());
    }
    let whole = vec![(0, samples.len())];
    if !paths.vad_model.is_file() {
        return Ok(whole);
    }
    // The pre-whisper gate is deliberately a little more sensitive than live
    // mode: 0.4 / 160 ms keeps quiet first words while still trimming silence.
    let mut detector = match crate::sherpa_engine::build_vad_with(paths, 60.0, 0.4, 0.16) {
        Ok(detector) => detector,
        Err(error) => {
            log::warn!("Speech gate unavailable, decoding the whole file: {error}");
            reporter.log(
                "[whisper.cpp] Warning: speech detection is unavailable; decoding the whole file.",
            );
            return Ok(whole);
        }
    };
    let bounds = match crate::sherpa_engine::file_segments(&mut detector, samples, cancel) {
        Ok(bounds) => bounds,
        Err(Error::Cancelled) => return Err(Error::Cancelled),
        Err(error) => {
            log::warn!("Speech gate failed, decoding the whole file: {error}");
            reporter
                .log("[whisper.cpp] Warning: speech detection failed; decoding the whole file.");
            return Ok(whole);
        }
    };
    if bounds.is_empty() {
        return Ok(Vec::new());
    }
    // whisper-cli gets real speech as one whole file, exactly like the Python
    // release. The gate is used only to answer "is there any speech at all?".
    if !split_long {
        return Ok(whole);
    }
    // Short files still run the gate so silence is recognised, but speech
    // stays undivided: the golden transcript depends on one whole decode.
    if samples.len() < 30 * 16_000 {
        return Ok(whole);
    }
    Ok(merge_gate_regions(&bounds, samples.len()))
}

/// Turn VAD bounds into whisper decode regions. Ordinary pauses stay in one
/// region so Whisper keeps its context; only a 30 s+ silence starts a new one.
/// A file with little leading/trailing silence and no long gap stays whole,
/// preserving the byte-identical golden transcript for ordinary recordings.
fn merge_gate_regions(bounds: &[(usize, usize)], total: usize) -> Vec<(usize, usize)> {
    if bounds.is_empty() {
        return Vec::new();
    }
    let whole = vec![(0, total)];
    let leading = bounds[0].0;
    let trailing = total.saturating_sub(bounds.last().map(|b| b.1).unwrap_or(0));
    let has_long_gap = bounds
        .windows(2)
        .any(|pair| pair[1].0.saturating_sub(pair[0].1) >= 30 * 16_000);
    if leading < 2 * 16_000 && trailing < 2 * 16_000 && !has_long_gap {
        return whole;
    }

    let pad = (crate::vad::VAD_PAD_SEC * 16_000.0) as usize;
    // Merge unpadded bounds first, then pad. Padding before the gap test would
    // make a 30.0–30.4 s pause lose 2 * pad and be merged again.
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for &(start, end) in bounds {
        match merged.last_mut() {
            Some(last) if start.saturating_sub(last.1) < 30 * 16_000 => {
                last.1 = last.1.max(end);
            }
            _ => merged.push((start, end)),
        }
    }
    if merged.is_empty() {
        whole
    } else {
        merged
            .into_iter()
            .map(|(start, end)| (start.saturating_sub(pad), (end + pad).min(total)))
            .collect()
    }
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

/// State shared with whisper.cpp's C abort callback. The boxed value must
/// outlive the `state.full` call; whisper.cpp only dereferences the pointer
/// while that call is running.
struct AbortProbe {
    cancel: CancelToken,
    calls: std::sync::atomic::AtomicU64,
    cancelled_calls: std::sync::atomic::AtomicU64,
}

unsafe extern "C" fn abort_trampoline(user_data: *mut std::ffi::c_void) -> bool {
    if user_data.is_null() {
        return false;
    }
    // SAFETY: `decode_regions` keeps the `AbortProbe` alive for the whole
    // `state.full` call and passes a pointer to it here. The callback only
    // reads the token and updates atomics, so it is safe on any thread.
    let probe = unsafe { &*(user_data as *const AbortProbe) };
    let cancelled = probe.cancel.is_cancelled();
    probe
        .calls
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if cancelled {
        probe
            .cancelled_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    cancelled
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
    let trace = std::env::var_os("EASYSCRIBE_WHISPER_TRACE").is_some();
    let mut context_params = WhisperContextParameters::default();
    context_params.use_gpu(use_gpu);
    // whisper-cli (the Python app) enables flash attention by default. Match
    // it: about 18% faster and a 4x smaller encode buffer.
    context_params.flash_attn(true);
    if let Some(device) = device {
        context_params.gpu_device(device);
    }
    let load_started = std::time::Instant::now();
    let context = WhisperContext::new_with_params(&paths.whispercpp_model, context_params)
        .map_err(|e| Error::WhisperCpp(format!("Could not load Whisper model: {e}")))?;
    let mut state = context
        .create_state()
        .map_err(|e| Error::WhisperCpp(format!("Could not create Whisper state: {e}")))?;
    if trace {
        eprintln!(
            "[whisper.cpp] model and state loaded in {:.1}s",
            load_started.elapsed().as_secs_f64()
        );
    }

    let mut segments = Vec::new();
    let started = std::time::Instant::now();
    for (region_index, (start, end)) in regions.iter().copied().enumerate() {
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
        // whisper-rs' `set_abort_callback_safe` boxes the closure twice and
        // then calls it through the wrong type, so it reads arbitrary memory
        // instead of the cancel token. Pass a plain C callback with a pointer
        // to a live probe.
        let probe = Box::new(AbortProbe {
            cancel: cancel.clone(),
            calls: 0.into(),
            cancelled_calls: 0.into(),
        });
        let probe_data = (&*probe as *const AbortProbe)
            .cast_mut()
            .cast::<std::ffi::c_void>();
        // SAFETY: `probe` stays alive until after `state.full` returns, and
        // `abort_trampoline` only touches the token and atomics inside it.
        unsafe {
            params.set_abort_callback(Some(abort_trampoline));
            params.set_abort_callback_user_data(probe_data);
        }
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

        let region_started = std::time::Instant::now();
        let result = state.full(params, &samples[start..end]);
        let calls = probe.calls.load(std::sync::atomic::Ordering::Relaxed);
        let cancelled_calls = probe
            .cancelled_calls
            .load(std::sync::atomic::Ordering::Relaxed);
        drop(probe);
        if trace {
            let seconds = region_started.elapsed().as_secs_f64();
            let audio_seconds = (end - start) as f64 / 16_000.0;
            match &result {
                Ok(()) => eprintln!(
                    "[whisper.cpp] region {}/{}: {audio_seconds:.1}s audio, full=ok in {seconds:.1}s, abort calls={calls} cancelled={cancelled_calls}",
                    region_index + 1,
                    regions.len()
                ),
                Err(error) => eprintln!(
                    "[whisper.cpp] region {}/{}: {audio_seconds:.1}s audio, full=error {error:?} in {seconds:.1}s, abort calls={calls} cancelled={cancelled_calls}",
                    region_index + 1,
                    regions.len()
                ),
            }
            context.print_timings();
        }
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

/// whisper.cpp's own CPU feature line, e.g.
/// `AVX = 1 | AVX2 = 1 | AVX512 = 0 | FMA = 1 | F16C = 1 ...`.
pub fn system_info() -> &'static str {
    whisper_rs::print_system_info()
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
    fn empty_gate_bounds_mean_no_speech() {
        assert!(merge_gate_regions(&[], 60 * 16_000).is_empty());
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
            true,
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
        let segments = transcribe_in_process(&base, &wav, &CancelToken::new(), &reporter).unwrap();
        let text = segments
            .iter()
            .map(|segment| segment.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(text.contains("quarterly results"), "{text}");
    }

    #[test]
    #[ignore = "needs the development models and assets/selftest_speech.wav"]
    fn golden_speech_text_matches_whisper_cli() {
        let base = Paths::at(crate::config::dev_base_dir()).expect("paths");
        let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets");
        let wav = data.join("selftest_speech.wav");
        let expected = data.join("selftest_speech.txt");
        assert!(
            wav.is_file(),
            "missing test fixture: {} (run with the development models and assets/)",
            wav.display()
        );
        assert!(
            expected.is_file(),
            "missing golden transcript: {}",
            expected.display()
        );
        assert!(
            base.whispercpp_model.is_file(),
            "missing development model: {}",
            base.whispercpp_model.display()
        );
        let expected = std::fs::read_to_string(expected).unwrap();
        let reporter: Arc<dyn Reporter> = Arc::new(crate::report::SilentReporter);
        let segments = transcribe_in_process(&base, &wav, &CancelToken::new(), &reporter).unwrap();
        let text = crate::transcript::build_plain_transcript(&segments, false);
        assert_eq!(text.trim_end(), expected.trim_end());
    }

    #[test]
    #[ignore]
    fn bench_gate_against_full_vad() {
        let wav = std::env::var("EASYSCRIBE_BENCH_WAV").unwrap();
        let base = Paths::discover().unwrap();
        let (samples, _) = crate::audio::load_wav_float32(Path::new(&wav)).unwrap();
        let cancel = CancelToken::new();
        let reporter: Arc<dyn Reporter> = Arc::new(crate::report::SilentReporter);
        let started = std::time::Instant::now();
        let regions = speech_regions(&base, &samples, &cancel, reporter.as_ref(), false).unwrap();
        println!(
            "full VAD: {} regions in {:.3}s for {:.1}s audio",
            regions.len(),
            started.elapsed().as_secs_f64(),
            samples.len() as f64 / 16_000.0
        );
        let mut vad = crate::sherpa_engine::build_vad_with(&base, 60.0, 0.4, 0.16).unwrap();
        let started = std::time::Instant::now();
        let has_speech = crate::sherpa_engine::has_speech(&mut vad, &samples, &cancel).unwrap();
        println!(
            "gate: speech={} in {:.3}s",
            has_speech,
            started.elapsed().as_secs_f64()
        );
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
