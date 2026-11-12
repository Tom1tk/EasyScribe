//! File transcription through the bundled `whisper-cli` child process.
//!
//! The Python v3.2.3 app ran the Vulkan build of whisper.cpp this way. Keeping
//! that proven path means a GPU driver crash only kills the child, Stop is a
//! process kill, and the output comes from exactly the same binary and flags
//! as the stable release. The in-process `whispercpp` engine remains the
//! fallback when no bundled binary is present (development and Linux tests).

use std::collections::VecDeque;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use regex::Regex;
use serde::Deserialize;

use crate::model::Segment;
use crate::report::CancelToken;
use crate::{Error, Paths, Reporter, Result};

pub const BEAM_SIZE: i32 = 5;

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// A Vulkan device reported by whisper-cli's own stderr.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CliGpu {
    pub id: i32,
    pub name: String,
    /// Everything after the name (`uma: 0 | fp16: 1 | ...`). Used both for
    /// logging and as the stable piece of the remembered-failure identity.
    pub details: String,
}

impl CliGpu {
    pub fn identity(&self) -> String {
        format!("{}|{}", self.name, self.details)
    }
}

/// True when the payload contains both the CLI and the GGUF model.
pub fn is_available(paths: &Paths) -> bool {
    paths.whispercpp_bin.is_file() && paths.whispercpp_model.is_file()
}

pub fn validate(paths: &Paths) -> Result<()> {
    if !paths.whispercpp_bin.is_file() {
        return Err(Error::WhisperCpp(format!(
            "whisper-cli not found at: {}",
            paths.whispercpp_bin.display()
        )));
    }
    if !paths.whispercpp_model.is_file() {
        return Err(Error::WhisperCpp(format!(
            "Whisper GGUF model not found at: {}",
            paths.whispercpp_model.display()
        )));
    }
    Ok(())
}

/// The result of asking whisper-cli which Vulkan devices it sees.
#[derive(Clone, Debug, Default)]
pub struct CliProbe {
    pub devices: Vec<CliGpu>,
    /// False when the probe could not run (missing DLL, timeout, temp-file
    /// failure). An empty device list with `ok` is a real "no GPU", which may
    /// safely use `-ng`; a failed probe must let whisper-cli choose instead.
    pub ok: bool,
}

/// Ask whisper-cli which Vulkan devices it sees, without loading the model.
/// An empty file is enough: the CLI lists devices, then rejects the model.
pub fn probe_devices(paths: &Paths, cancel: &CancelToken) -> CliProbe {
    let empty = paths
        .temp_dir
        .join(format!("gpu_probe_{}.bin", uuid::Uuid::new_v4().simple()));
    if std::fs::write(&empty, []).is_err() {
        return CliProbe::default();
    }
    let mut command = Command::new(&paths.whispercpp_bin);
    command
        .arg("-m")
        .arg(&empty)
        .arg("-f")
        .arg(&empty)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    let outcome = run_probe(command, Duration::from_secs(30), cancel);
    let _ = std::fs::remove_file(&empty);
    match outcome {
        Some(output) => CliProbe {
            devices: parse_devices(&String::from_utf8_lossy(&output)),
            ok: true,
        },
        None => CliProbe::default(),
    }
}

/// Probe once per process. The same binary cannot gain or lose a Vulkan
/// device during one transcription session, and a probe starts a child
/// process, so repeat files should not pay for it again. A cancelled probe is
/// not cached.
fn cached_probe(paths: &Paths, cancel: &CancelToken) -> CliProbe {
    static CACHE: OnceLock<Mutex<Option<(PathBuf, CliProbe)>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    if let Ok(guard) = cache.lock()
        && let Some((bin, probe)) = guard.as_ref()
        && bin == &paths.whispercpp_bin
    {
        return probe.clone();
    }
    let probe = probe_devices(paths, cancel);
    if !cancel.is_cancelled()
        && let Ok(mut guard) = cache.lock()
    {
        *guard = Some((paths.whispercpp_bin.clone(), probe.clone()));
    }
    probe
}

fn run_probe(mut command: Command, timeout: Duration, cancel: &CancelToken) -> Option<Vec<u8>> {
    let mut child = command.spawn().ok()?;
    #[cfg(windows)]
    crate::win_job::assign(&child);
    let Some(stderr) = child.stderr.take() else {
        kill_and_reap(&mut child);
        return None;
    };
    let (line_tx, line_rx) = mpsc::channel::<Vec<u8>>();
    // Drain stderr while the probe runs. Waiting for exit before reading can
    // deadlock once validation or device output fills the pipe buffer.
    let reader = std::thread::Builder::new()
        .name("whisper-cli-probe".into())
        .spawn(move || {
            let mut reader = BufReader::new(stderr);
            let mut line = Vec::new();
            loop {
                line.clear();
                match reader.read_until(b'\n', &mut line) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        if line_tx.send(line.clone()).is_err() {
                            break;
                        }
                    }
                }
            }
        })
        .ok();
    let started = Instant::now();
    let mut reaped = true;
    let outcome = loop {
        if cancel.is_cancelled() {
            reaped = kill_and_reap(&mut child);
            break None;
        }
        match child.try_wait() {
            Ok(Some(_)) => {
                // The child has exited; let the reader see EOF before
                // collecting the final lines.
                break Some(());
            }
            Ok(None) if started.elapsed() >= timeout => {
                reaped = kill_and_reap(&mut child);
                break None;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(20)),
            Err(_) => {
                reaped = kill_and_reap(&mut child);
                break None;
            }
        }
    };
    if reaped && let Some(reader) = reader {
        let _ = reader.join();
    }
    let mut output = Vec::new();
    while let Ok(line) = line_rx.try_recv() {
        output.extend_from_slice(&line);
    }
    outcome.map(|()| output)
}

/// Kill a child and give it a bounded time to exit. A GPU driver stuck in
/// kernel mode can make `wait()` hang forever; the job object reaps the child
/// when the app exits, so returning without a complete reap is safer.
fn kill_and_reap(child: &mut std::process::Child) -> bool {
    let _ = child.kill();
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        match child.try_wait() {
            Ok(Some(_)) | Err(_) => return true,
            Ok(None) => std::thread::sleep(Duration::from_millis(50)),
        }
    }
    false
}

fn parse_devices(stderr: &str) -> Vec<CliGpu> {
    static DEVICE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    let regex = DEVICE.get_or_init(|| {
        Regex::new(r"ggml_vulkan: (\d+) = (.+?) \| (.*)").expect("valid Vulkan device regex")
    });
    let mut devices = Vec::new();
    for line in stderr.lines() {
        if let Some(caps) = regex.captures(line)
            && let Ok(id) = caps[1].parse::<i32>()
        {
            devices.push(CliGpu {
                id,
                name: caps[2].to_owned(),
                details: caps[3].to_owned(),
            });
        }
    }
    devices
}

/// Dedicated GPU before integrated (`uma: 1`), then one with matrix cores,
/// then Vulkan's own order. A remembered preferred index wins when present,
/// unless that device is the one whose driver failed before.
pub fn pick_gpu(
    devices: &[CliGpu],
    preferred: Option<i32>,
    disabled: Option<&str>,
) -> Option<CliGpu> {
    let usable = |device: &&CliGpu| disabled.is_none_or(|identity| identity != device.identity());
    if let Some(preferred) = preferred
        && let Some(device) = devices
            .iter()
            .find(|device| device.id == preferred)
            .filter(usable)
    {
        return Some(device.clone());
    }
    devices
        .iter()
        .filter(usable)
        .max_by_key(|device| {
            (
                !device.details.contains("uma: 1"),
                !device.details.contains("matrix cores: none"),
                -device.id,
            )
        })
        .cloned()
}

/// Transcribe a mono 16 kHz WAV with the bundled whisper-cli.
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

    // Silence is the worst input for Whisper: it hallucinates "Thank you."
    // and wastes a full encode. Ask the same Silero gate the in-process path
    // uses, but always hand real speech to whisper-cli as one whole file so
    // the output matches the Python release.
    let (samples, rate) = crate::audio::load_wav_float32(audio_path)
        .map_err(|e| Error::Transcription(format!("Could not load audio: {e}")))?;
    if rate != 16_000 {
        return Err(Error::Transcription(format!(
            "whisper.cpp needs 16 kHz audio, got {rate} Hz"
        )));
    }
    let peak = samples
        .iter()
        .fold(0.0_f32, |peak, sample| peak.max(sample.abs()));
    // The gate only answers "is there any speech?". Stop at the first
    // confirmed segment instead of scanning every minute of a long file.
    reporter.log("[whisper.cpp] Checking for speech…");
    let gate_started = Instant::now();
    let has_speech = match crate::sherpa_engine::build_vad_with(paths, 60.0, 0.4, 0.16) {
        Ok(mut detector) => crate::sherpa_engine::has_speech(&mut detector, &samples, cancel)?,
        Err(error) => {
            log::warn!("Speech gate unavailable, decoding the whole file: {error}");
            reporter.log(
                "[whisper.cpp] Warning: speech detection is unavailable; decoding the whole file.",
            );
            true
        }
    };
    log::info!(
        "Speech gate took {:.3}s for {:.1}s of audio (speech={has_speech})",
        gate_started.elapsed().as_secs_f64(),
        samples.len() as f64 / 16_000.0
    );
    // The child process reads the WAV itself. Drop our copy before starting it
    // so a long file does not hold hundreds of megabytes beside the model.
    drop(samples);
    if !has_speech {
        if peak < 0.01 {
            reporter.progress(1.0);
            reporter.log("[whisper.cpp] No speech found. Nothing to transcribe.");
            return Ok(Vec::new());
        }
        // Silero can miss quiet speech, music or unusual languages. A nonzero
        // signal must still reach whisper-cli instead of becoming an empty
        // transcript.
        reporter.log(
            "[whisper.cpp] Warning: speech detection found no speech, but the audio is not silent; decoding the whole file.",
        );
    }

    let mut settings = crate::settings::Settings::load(paths.settings_file.clone());
    let disabled = settings.load_gpu_disabled_for();
    let force_cpu = std::env::var_os("EASYSCRIBE_WHISPER_CPU").is_some();
    let probe = if force_cpu {
        CliProbe::default()
    } else {
        cached_probe(paths, cancel)
    };
    cancel.check()?;
    for device in &probe.devices {
        reporter.log(&format!(
            "[whisper.cpp] GPU {}: {} | {}",
            device.id, device.name, device.details
        ));
    }
    let preferred = settings.load_gpu_device();
    let gpu = if force_cpu {
        None
    } else {
        pick_gpu(&probe.devices, preferred, disabled.as_deref())
    };
    let gpu_usable = gpu.as_ref().filter(|_| !force_cpu);
    let choice = if force_cpu {
        reporter.log("[whisper.cpp] Engine: CPU (forced by EASYSCRIBE_WHISPER_CPU)");
        DeviceChoice::Cpu
    } else if !probe.ok {
        reporter.log(
            "[whisper.cpp] Warning: the Vulkan device probe failed; letting whisper-cli choose a device.",
        );
        DeviceChoice::Auto
    } else if let Some(gpu) = &gpu_usable {
        reporter.log(&format!(
            "[whisper.cpp] Engine: Vulkan (device {}: {})",
            gpu.id, gpu.name
        ));
        DeviceChoice::Gpu(gpu.id)
    } else if gpu.is_some() {
        reporter.log("[whisper.cpp] Engine: CPU (this GPU failed before)");
        DeviceChoice::Cpu
    } else {
        reporter.log("[whisper.cpp] Engine: CPU (no Vulkan device found)");
        DeviceChoice::Cpu
    };

    let output = paths
        .temp_dir
        .join(format!("whispercpp_{}.json", uuid::Uuid::new_v4().simple()));
    let first = run_cli(
        paths,
        audio_path,
        &output,
        choice,
        cancel,
        reporter,
        CliOptions::default(),
    );
    let mut segments = match first {
        Ok(segments) => segments,
        Err(Error::Cancelled) => {
            cleanup_output(&output);
            return Err(Error::Cancelled);
        }
        Err(error) if is_start_failure(&error) => {
            cleanup_output(&output);
            return fallback_in_process(paths, audio_path, cancel, reporter, error);
        }
        Err(error) if matches!(choice, DeviceChoice::Gpu(_) | DeviceChoice::Auto) => {
            let gpu_failure = is_gpu_failure(&error);
            reporter.log(&format!(
                "[whisper.cpp] GPU decode failed ({error}). Retrying on the CPU…"
            ));
            log::warn!("whisper-cli GPU decode failed, retrying on CPU: {error}");
            if let (DeviceChoice::Gpu(_), Some(gpu), true) = (choice, &gpu_usable, gpu_failure) {
                if let Err(save_error) = settings.save_gpu_disabled_for(Some(&gpu.identity())) {
                    log::warn!("Could not remember the failed GPU: {save_error}");
                } else {
                    reporter.log(&format!(
                        "[whisper.cpp] Using the CPU for {} until the graphics driver changes.",
                        gpu.name
                    ));
                }
            }
            match run_cli(
                paths,
                audio_path,
                &output,
                DeviceChoice::Cpu,
                cancel,
                reporter,
                CliOptions::default(),
            ) {
                Ok(segments) => segments,
                Err(Error::Cancelled) => {
                    cleanup_output(&output);
                    return Err(Error::Cancelled);
                }
                Err(error) if is_start_failure(&error) => {
                    cleanup_output(&output);
                    return fallback_in_process(paths, audio_path, cancel, reporter, error);
                }
                Err(error) => {
                    cleanup_output(&output);
                    return Err(error);
                }
            }
        }
        Err(error) => {
            cleanup_output(&output);
            return Err(error);
        }
    };
    match repair_repetitions(paths, audio_path, &output, cancel, reporter, &segments) {
        Ok(Some(repaired)) => segments = repaired,
        Ok(None) => {}
        Err(Error::Cancelled) => {
            cleanup_output(&output);
            return Err(Error::Cancelled);
        }
        Err(error) => {
            reporter.log(&format!(
                "[whisper.cpp] Warning: repetition repair failed ({error}); keeping the original transcript."
            ));
            log::warn!("whisper.cpp repetition repair failed: {error}");
        }
    }
    cleanup_output(&output);
    if let DeviceChoice::Gpu(_) = choice
        && let Some(gpu) = &gpu_usable
        && preferred != Some(gpu.id)
        && let Err(error) = settings.save_gpu_device(Some(gpu.id))
    {
        log::warn!("Could not remember the GPU device: {error}");
    }
    reporter.progress(1.0);
    reporter.log(&format!(
        "[whisper.cpp] Decoded {} segment(s)",
        segments.len()
    ));
    Ok(segments)
}

fn fallback_in_process(
    paths: &Paths,
    audio_path: &Path,
    cancel: &CancelToken,
    reporter: &Arc<dyn Reporter>,
    error: Error,
) -> Result<Vec<Segment>> {
    // The release selftest must prove the shipped primary path. If the CLI
    // cannot start there, report the failure instead of quietly decoding with
    // the linked fallback.
    if std::env::var_os("EASYSCRIBE_SELFTEST_STRICT_CLI").is_some() {
        return Err(error);
    }
    reporter.log(&format!(
        "[Warning] The bundled whisper-cli could not start ({error}). Using the slower in-process engine."
    ));
    let reason: String = error.to_string().chars().take(140).collect();
    reporter.detail(&format!("Built-in engine: {reason}"));
    log::warn!("whisper-cli could not start, using the in-process engine: {error}");
    crate::whispercpp::transcribe_in_process(paths, audio_path, cancel, reporter)
}

fn cleanup_output(output: &Path) {
    let _ = std::fs::remove_file(output);
    let _ = std::fs::remove_file(output.with_extension(""));
    let _ = std::fs::remove_file(output.with_extension("json"));
}

/// A spawn failure or a Windows loader error means the CLI (or a DLL beside
/// it) could not be loaded. Unlike a decode failure this is safe to retry with
/// the linked in-process engine.
fn is_start_failure(error: &Error) -> bool {
    matches!(error, Error::WhisperCpp(message) if message.starts_with("whisper-cli could not start"))
}

/// Only a Vulkan-specific failure should disable the GPU for later runs. A
/// disk-full, bad input or parse error must not.
fn is_gpu_failure(error: &Error) -> bool {
    matches!(error, Error::WhisperCppGpu(_))
}

/// Signals that the failure came from ggml-vulkan rather than from the model,
/// the audio file or the output path. `run_cli` adds the marker only for a GPU
/// run, so a CPU failure can never disable the GPU.
fn gpu_specific_failure(snippet: &str) -> bool {
    let text = snippet.to_ascii_lowercase();
    [
        "ggml_vulkan: error",
        "ggml_vulkan: failed",
        "vk::",
        "vk_error",
        "vkresult",
        "device lost",
        "failed to initialize the backend",
        "whisper_backend_init_gpu: failed",
    ]
    .iter()
    .any(|needle| text.contains(needle))
}

fn is_gpu_crash(code: Option<i32>) -> bool {
    matches!(
        code,
        // 0xC0000005 access violation, 0xC0000409 fast fail/abort,
        // 0xC000001D illegal instruction.
        Some(-1073741819 | -1073740791 | -1073741587)
    )
}

fn is_loader_failure(code: Option<i32>) -> bool {
    matches!(
        code,
        // 0xC0000135, 0xC0000139, 0xC000007B, 0xC0000142
        Some(-1073741515 | -1073741511 | -1073741701 | -1073741502)
    )
}

#[derive(Clone, Copy)]
enum DeviceChoice {
    /// The probe could not run; let whisper-cli pick its default device.
    Auto,
    Cpu,
    Gpu(i32),
}

/// Extra whisper-cli arguments used by the repetition repair. The normal
/// transcription leaves these at their defaults so its output stays identical
/// to v3.2.3.
#[derive(Clone, Copy, Default)]
struct CliOptions {
    offset_ms: Option<u64>,
    duration_ms: Option<u64>,
    max_context: Option<i32>,
    entropy_thold: Option<f32>,
}

fn run_cli(
    paths: &Paths,
    audio_path: &Path,
    output: &Path,
    choice: DeviceChoice,
    cancel: &CancelToken,
    reporter: &Arc<dyn Reporter>,
    options: CliOptions,
) -> Result<Vec<Segment>> {
    cancel.check()?;
    // whisper.cpp writes `<stem>.json`; older builds also touch `<stem>`.
    // Clean both so a failure cannot leave a stale file to be parsed.
    let output_stem = output.with_extension("");
    cleanup_output(output);

    let mut command = Command::new(&paths.whispercpp_bin);
    command
        .arg("-m")
        .arg(&paths.whispercpp_model)
        .arg("-f")
        .arg(audio_path)
        .arg("-bs")
        .arg(BEAM_SIZE.to_string())
        .arg("-t")
        .arg(crate::config::num_threads().to_string())
        .arg("-pp")
        .arg("--output-json")
        .arg("--output-file")
        .arg(&output_stem);
    if let Some(offset) = options.offset_ms {
        command.arg("-ot").arg(offset.to_string());
    }
    if let Some(duration) = options.duration_ms {
        command.arg("-d").arg(duration.to_string());
    }
    if let Some(max_context) = options.max_context {
        command.arg("-mc").arg(max_context.to_string());
    }
    if let Some(entropy) = options.entropy_thold {
        command.arg("-et").arg(format!("{entropy:.2}"));
    }
    match choice {
        DeviceChoice::Gpu(device) => {
            command.arg("-dev").arg(device.to_string());
            reporter.log(&format!("[whisper.cpp] Using Vulkan device {device}"));
        }
        DeviceChoice::Cpu => {
            // -ng stops whisper.cpp from using the GPU, but ggml still
            // enumerates Vulkan backends. Disable the backend as well so a
            // broken driver cannot crash the CPU retry during enumeration.
            command.arg("-ng").env("GGML_DISABLE_VULKAN", "1");
        }
        DeviceChoice::Auto => {}
    }
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(CREATE_NO_WINDOW);
    }

    let mut child = command.spawn().map_err(|e| {
        Error::WhisperCpp(format!(
            "whisper-cli could not start ({}): {e}",
            paths
                .whispercpp_bin
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("whisper-cli.exe")
        ))
    })?;
    #[cfg(windows)]
    crate::win_job::assign(&child);
    let Some(stderr) = child.stderr.take() else {
        kill_and_reap(&mut child);
        return Err(Error::WhisperCpp(
            "whisper-cli started without a stderr pipe".into(),
        ));
    };
    let (line_tx, line_rx) = mpsc::channel::<String>();
    let reader = match std::thread::Builder::new()
        .name("whisper-cli-stderr".into())
        .spawn(move || {
            // Read bytes, not lines(): one invalid UTF-8 byte (a non-ASCII
            // path in a log line) would otherwise stop the reader and hide
            // both progress and the error that follows.
            let mut reader = BufReader::new(stderr);
            let mut line = Vec::new();
            loop {
                line.clear();
                match reader.read_until(b'\n', &mut line) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        let text = String::from_utf8_lossy(&line)
                            .trim_end_matches(&['\r', '\n'][..])
                            .to_owned();
                        if line_tx.send(text).is_err() {
                            break;
                        }
                    }
                }
            }
        }) {
        Ok(reader) => reader,
        Err(error) => {
            kill_and_reap(&mut child);
            return Err(Error::WhisperCpp(format!(
                "Could not read whisper-cli progress: {error}"
            )));
        }
    };
    let mut tail = VecDeque::new();

    let status = loop {
        if cancel.is_cancelled() {
            stop_child(&mut child, reader);
            return Err(Error::Cancelled);
        }
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                drain_stderr(&line_rx, reporter, &mut tail);
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(error) => {
                stop_child(&mut child, reader);
                return Err(Error::WhisperCpp(format!(
                    "Could not wait for whisper-cli: {error}"
                )));
            }
        }
    };
    let _ = reader.join();
    drain_stderr(&line_rx, reporter, &mut tail);

    if !status.success() {
        let snippet = tail
            .iter()
            .rev()
            .take(20)
            .rev()
            .cloned()
            .collect::<Vec<_>>()
            .join("\n");
        if is_loader_failure(status.code()) {
            return Err(Error::WhisperCpp(format!(
                "whisper-cli could not start because a required DLL is missing or broken (exit {:?}).\n\n{snippet}",
                status.code()
            )));
        }
        let message = format!(
            "whisper-cli exited with code {:?}.\n\n{snippet}",
            status.code()
        );
        if matches!(choice, DeviceChoice::Gpu(_))
            && (gpu_specific_failure(&snippet) || is_gpu_crash(status.code()))
        {
            return Err(Error::WhisperCppGpu(message));
        }
        return Err(Error::WhisperCpp(message));
    }
    if !output.is_file() {
        return Err(Error::WhisperCpp(format!(
            "whisper-cli completed but did not produce {}",
            output
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("its JSON output")
        )));
    }
    let segments = parse_output(output)?;
    cancel.check()?;
    Ok(segments)
}

/// A run of the same sentence. `start`/`end` cover the repeated text itself.
#[derive(Debug, Clone, PartialEq)]
struct Repetition {
    start: f64,
    end: f64,
    count: usize,
}

#[derive(Debug, Clone)]
struct Sentence {
    normalized: String,
    start: f64,
    end: f64,
}

/// A long sentence repeated more than this many times is treated as a decoder
/// loop. Short interjections ("No.", "Yes.") are allowed a higher count because
/// they can be genuine speech.
const MAX_LONG_REPEATS: usize = 4;
const MAX_SHORT_REPEATS: usize = 8;
const MIN_LONG_SENTENCE_WORDS: usize = 3;

/// Find the longest run of identical consecutive sentences in `segments`.
/// Timestamps inside one segment are distributed by character count; this is
/// only used to choose the repair range, so an approximation is enough.
fn find_repetition(segments: &[Segment]) -> Option<Repetition> {
    let sentences = split_sentences(segments);
    let mut best: Option<Repetition> = None;
    let mut run_start = 0usize;
    let mut index = 1usize;
    while index <= sentences.len() {
        let same = index < sentences.len()
            && !sentences[index].normalized.is_empty()
            && sentences[index].normalized == sentences[run_start].normalized;
        if same {
            index += 1;
            continue;
        }
        let count = index - run_start;
        let words = sentences[run_start].normalized.split_whitespace().count();
        let limit = if words >= MIN_LONG_SENTENCE_WORDS {
            MAX_LONG_REPEATS
        } else {
            MAX_SHORT_REPEATS
        };
        if count > limit {
            let candidate = Repetition {
                start: sentences[run_start].start,
                end: sentences[index - 1].end,
                count,
            };
            let longer = best.as_ref().is_none_or(|current| {
                candidate.end - candidate.start > current.end - current.start
            });
            if longer {
                best = Some(candidate);
            }
        }
        run_start = index;
        index += 1;
    }
    best
}

/// The longest run of identical consecutive sentences, if it looks like a
/// decoder loop. Exposed for the release selftest; the app uses it internally
/// to choose the CPU repair range.
pub fn repeated_sentence(segments: &[Segment]) -> Option<(usize, f64, f64)> {
    find_repetition(segments).map(|found| (found.count, found.start, found.end))
}

fn split_sentences(segments: &[Segment]) -> Vec<Sentence> {
    let mut sentences = Vec::new();
    for segment in segments {
        let chars: Vec<char> = segment.text.trim().chars().collect();
        if chars.is_empty() {
            continue;
        }
        let total = chars.len().max(1) as f64;
        let mut start_char = 0usize;
        let mut index = 0usize;
        while index < chars.len() {
            let boundary = matches!(chars[index], '.' | '!' | '?')
                && (index + 1 == chars.len()
                    || chars
                        .get(index + 1)
                        .is_some_and(|next| next.is_whitespace()));
            if boundary {
                push_sentence(
                    &mut sentences,
                    &chars[start_char..=index],
                    segment,
                    start_char,
                    index + 1,
                    total,
                );
                start_char = index + 1;
                while start_char < chars.len() && chars[start_char].is_whitespace() {
                    start_char += 1;
                }
                index = start_char;
                continue;
            }
            index += 1;
        }
        if start_char < chars.len() {
            push_sentence(
                &mut sentences,
                &chars[start_char..],
                segment,
                start_char,
                chars.len(),
                total,
            );
        }
    }
    sentences
}

fn push_sentence(
    sentences: &mut Vec<Sentence>,
    chars: &[char],
    segment: &Segment,
    start_char: usize,
    end_char: usize,
    total: f64,
) {
    let text: String = chars.iter().collect();
    let normalized = normalize_sentence(&text);
    if normalized.is_empty() {
        return;
    }
    let span = (segment.end - segment.start).max(0.0);
    sentences.push(Sentence {
        normalized,
        start: segment.start + span * (start_char as f64 / total),
        end: segment.start + span * (end_char as f64 / total),
    });
}

fn normalize_sentence(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut pending_space = false;
    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '\'' || ch == '’' {
            if pending_space && !out.is_empty() {
                out.push(' ');
            }
            out.extend(ch.to_lowercase());
            pending_space = false;
        } else {
            pending_space = true;
        }
    }
    out
}

/// Repair a detected repetition by re-decoding only its time range on the CPU
/// with context carry-over disabled. The GPU path is fast but can produce this
/// failure on some drivers; the CPU decode of a short range is cheap and gives
/// the correct text. Any failure keeps the original transcript instead of
/// losing the whole file.
fn repair_repetitions(
    paths: &Paths,
    audio_path: &Path,
    output: &Path,
    cancel: &CancelToken,
    reporter: &Arc<dyn Reporter>,
    segments: &[Segment],
) -> Result<Option<Vec<Segment>>> {
    let Some(repetition) = find_repetition(segments) else {
        return Ok(None);
    };
    let total = crate::audio::wav_duration(audio_path).unwrap_or(repetition.end);
    let padding = 1.5;
    let start = (repetition.start - padding).max(0.0);
    let end = (repetition.end + padding).min(total.max(repetition.end));
    let span = (end - start).max(0.0);
    if span <= 0.0 {
        return Ok(None);
    }
    if span > 600.0 {
        reporter.log(
            "[whisper.cpp] A repeated section is too long to repair automatically; keeping the original transcript.",
        );
        log::warn!(
            "whisper.cpp repetition spans {:.1}s; skipping automatic repair",
            span
        );
        return Ok(None);
    }

    reporter.log(&format!(
        "[whisper.cpp] Repetition detected ({} repeats over {:.1}s); re-decoding that part on the CPU…",
        repetition.count, span
    ));
    log::warn!(
        "whisper.cpp repetition detected: {} repeats over {:.1}s",
        repetition.count,
        span
    );

    let repair_output = output.with_file_name(format!(
        "{}_repair.json",
        output
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("whisper")
    ));
    let options = CliOptions {
        offset_ms: Some((start * 1000.0).round() as u64),
        duration_ms: Some((span * 1000.0).round() as u64),
        max_context: Some(0),
        entropy_thold: Some(2.8),
    };
    let repaired = match run_cli(
        paths,
        audio_path,
        &repair_output,
        DeviceChoice::Cpu,
        cancel,
        reporter,
        options,
    ) {
        Ok(repaired) => repaired,
        Err(Error::Cancelled) => {
            cleanup_output(&repair_output);
            return Err(Error::Cancelled);
        }
        Err(error) => {
            cleanup_output(&repair_output);
            reporter.log(&format!(
                "[whisper.cpp] Warning: the CPU repair failed ({error}); keeping the original transcript."
            ));
            log::warn!("whisper.cpp CPU repair failed: {error}");
            return Ok(None);
        }
    };
    cleanup_output(&repair_output);

    if repaired.is_empty() || find_repetition(&repaired).is_some() {
        reporter.log(
            "[whisper.cpp] Warning: the CPU repair did not remove the repetition; keeping the original transcript.",
        );
        log::warn!("whisper.cpp CPU repair still contained a repetition; keeping the original");
        return Ok(None);
    }

    let mut combined: Vec<Segment> = segments
        .iter()
        .filter(|segment| segment.end <= start || segment.start >= end)
        .cloned()
        .collect();
    combined.extend(
        repaired
            .into_iter()
            .filter(|segment| segment.end > start && segment.start < end),
    );
    combined.sort_by(|a, b| {
        a.start
            .partial_cmp(&b.start)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    reporter.log(&format!(
        "[whisper.cpp] Repaired {:.1}s of repeated output on the CPU.",
        span
    ));
    log::info!(
        "whisper.cpp repaired {:.1}s of repeated output on the CPU",
        span
    );
    Ok(Some(combined))
}

/// Stop a child for cancellation/error. If it cannot be reaped within five
/// seconds (a GPU driver stuck in kernel mode), leave the reader thread
/// detached; the Windows job object reaps the process when the app exits.
fn stop_child(child: &mut std::process::Child, reader: std::thread::JoinHandle<()>) {
    let reaped = kill_and_reap(child);
    if reaped {
        let _ = reader.join();
    }
}

fn drain_stderr(
    lines: &Receiver<String>,
    reporter: &Arc<dyn Reporter>,
    tail: &mut VecDeque<String>,
) {
    while let Ok(line) = lines.try_recv() {
        describe_stderr(&line, reporter);
        push_tail(tail, line);
    }
}

fn push_tail(tail: &mut VecDeque<String>, line: String) {
    tail.push_back(line);
    if tail.len() > 40 {
        tail.pop_front();
    }
}

fn describe_stderr(line: &str, reporter: &Arc<dyn Reporter>) {
    static PROGRESS: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    let progress = PROGRESS.get_or_init(|| {
        Regex::new(r"whisper_print_progress_callback:\s*progress\s*=\s*(\d+)%")
            .expect("valid whisper progress regex")
    });
    if let Some(caps) = progress.captures(line)
        && let Ok(percent) = caps[1].parse::<f64>()
    {
        reporter.progress((percent / 100.0).clamp(0.0, 1.0));
    }
    if line.contains("whisper_backend_init_gpu: using ")
        && let Some(engine) = line.split("using ").nth(1)
    {
        reporter.log(&format!("[whisper.cpp] Engine: {}", engine.trim()));
    } else if line.contains("whisper_backend_init_gpu: no GPU found") {
        reporter.log("[whisper.cpp] Engine: CPU (no GPU found)");
    }
}

fn parse_output(path: &Path) -> Result<Vec<Segment>> {
    #[derive(Deserialize)]
    struct Root {
        transcription: Vec<Entry>,
    }
    #[derive(Deserialize)]
    struct Entry {
        #[serde(default)]
        text: String,
        #[serde(default)]
        offsets: Offsets,
    }
    #[derive(Deserialize, Default)]
    struct Offsets {
        #[serde(default)]
        from: i64,
        #[serde(default)]
        to: i64,
    }

    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::WhisperCpp(format!("Could not read whisper-cli output: {e}")))?;
    let root: Root = serde_json::from_str(&text)
        .map_err(|e| Error::WhisperCpp(format!("Could not parse whisper-cli output: {e}")))?;
    Ok(root
        .transcription
        .into_iter()
        .filter_map(|entry| {
            let text = entry.text.trim();
            if text.is_empty() {
                return None;
            }
            Some(Segment {
                start: entry.offsets.from as f64 / 1000.0,
                end: entry.offsets.to as f64 / 1000.0,
                text: text.to_owned(),
            })
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_dedicated_gpu_before_integrated() {
        let devices = vec![
            CliGpu {
                id: 0,
                name: "Intel Iris".into(),
                details: "uma: 1 | fp16: 1 | bf16: 0".into(),
            },
            CliGpu {
                id: 1,
                name: "RTX 5070".into(),
                details: "uma: 0 | fp16: 1 | bf16: 0".into(),
            },
        ];
        assert_eq!(pick_gpu(&devices, None, None).unwrap().id, 1);
        assert_eq!(pick_gpu(&devices, Some(0), None).unwrap().id, 0);
        // A failed preferred device is filtered; the other GPU is used.
        let failed = devices[1].identity();
        assert_eq!(
            pick_gpu(&devices, Some(1), Some(failed.as_str()))
                .unwrap()
                .id,
            0
        );
    }

    #[test]
    fn prefers_matrix_cores_within_the_same_class() {
        let devices = vec![
            CliGpu {
                id: 0,
                name: "Old".into(),
                details: "uma: 0 | fp16: 1 | matrix cores: none".into(),
            },
            CliGpu {
                id: 2,
                name: "New".into(),
                details: "uma: 0 | fp16: 1 | matrix cores: 1".into(),
            },
        ];
        assert_eq!(pick_gpu(&devices, None, None).unwrap().id, 2);
    }

    #[test]
    fn parses_ggml_vulkan_device_lines() {
        let stderr = "\
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = NVIDIA GeForce RTX 5070 (NVIDIA) | uma: 0 | fp16: 1 | bf16: 0 | matrix cores: 1
";
        let devices = parse_devices(stderr);
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].name, "NVIDIA GeForce RTX 5070 (NVIDIA)");
        assert!(devices[0].details.contains("uma: 0"));
    }

    #[test]
    fn classifies_loader_exit_codes_as_start_failures() {
        for code in [-1073741515, -1073741511, -1073741701, -1073741502] {
            assert!(is_loader_failure(Some(code)), "{code}");
        }
        assert!(!is_loader_failure(Some(3)));
        assert!(!is_loader_failure(None));
    }

    #[test]
    fn only_gpu_messages_are_gpu_failures() {
        assert!(is_gpu_failure(&Error::WhisperCppGpu(
            "ggml_vulkan: vk::DeviceLostError".into()
        )));
        assert!(!is_gpu_failure(&Error::WhisperCpp(
            "whisper-cli completed but did not produce its JSON output".into()
        )));
        assert!(!is_gpu_failure(&Error::Cancelled));
        assert!(gpu_specific_failure(
            "ggml_vulkan: vk::DeviceLostError\nwhisper_backend_init_gpu: failed"
        ));
        assert!(is_gpu_crash(Some(-1073741819)));
        assert!(!is_gpu_crash(Some(3)));
    }

    #[test]
    fn rejects_output_without_a_transcription_key() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.json");
        std::fs::write(&path, "{}").unwrap();
        assert!(parse_output(&path).is_err());
    }

    #[test]
    fn maps_millisecond_offsets_and_trims_text() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ok.json");
        std::fs::write(
            &path,
            r#"{"transcription":[{"text":" hello ","offsets":{"from":1500,"to":2500}}]}"#,
        )
        .unwrap();
        let segments = parse_output(&path).unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, "hello");
        assert_eq!(segments[0].start, 1.5);
        assert_eq!(segments[0].end, 2.5);
    }

    #[cfg(unix)]
    #[test]
    fn cancellation_kills_the_child_process() {
        use std::os::unix::fs::PermissionsExt;
        use std::time::Instant;

        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("whisper-cli");
        std::fs::write(&bin, "#!/bin/sh\nexec sleep 30\n").unwrap();
        let mut permissions = std::fs::metadata(&bin).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&bin, permissions).unwrap();

        let mut paths = Paths::at(dir.path().to_path_buf()).unwrap();
        paths.whispercpp_bin = bin;
        paths.whispercpp_model = dir.path().join("model.bin");
        let output = paths.temp_dir.join("out.json");
        let cancel = CancelToken::new();
        let worker_cancel = cancel.clone();
        let worker_paths = paths.clone();
        let reporter: Arc<dyn Reporter> = Arc::new(crate::report::SilentReporter);
        let handle = std::thread::spawn(move || {
            run_cli(
                &worker_paths,
                Path::new("audio.wav"),
                &output,
                DeviceChoice::Cpu,
                &worker_cancel,
                &reporter,
                CliOptions::default(),
            )
        });
        std::thread::sleep(Duration::from_millis(150));
        let started = Instant::now();
        cancel.cancel();
        let result = handle.join().unwrap();
        assert!(matches!(result, Err(Error::Cancelled)));
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[cfg(unix)]
    #[test]
    fn repairs_a_repeated_run_on_the_cpu_and_splices_the_result() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("whisper-cli");
        let script = r#"#!/bin/sh
out=""
repair=0
prev=""
for arg in "$@"; do
  if [ "$prev" = "--output-file" ]; then out="$arg"; fi
  if [ "$arg" = "-ot" ]; then repair=1; fi
  prev="$arg"
done
if [ "$repair" = "1" ]; then
  cat > "$out.json" <<'JSON'
{"transcription":[
 {"text":" first repaired line.","offsets":{"from":9000,"to":30000}},
 {"text":" second repaired line.","offsets":{"from":30000,"to":60000}}
]}
JSON
else
  cat > "$out.json" <<'JSON'
{"transcription":[
 {"text":" Intro.","offsets":{"from":0,"to":5000}},
 {"text":" Looping sentence here.","offsets":{"from":10000,"to":20000}},
 {"text":" Looping sentence here.","offsets":{"from":20000,"to":30000}},
 {"text":" Looping sentence here.","offsets":{"from":30000,"to":40000}},
 {"text":" Looping sentence here.","offsets":{"from":40000,"to":50000}},
 {"text":" Looping sentence here.","offsets":{"from":50000,"to":60000}},
 {"text":" Looping sentence here.","offsets":{"from":60000,"to":70000}},
 {"text":" Outro.","offsets":{"from":72000,"to":75000}}
]}
JSON
fi
"#;
        std::fs::write(&bin, script).unwrap();
        let mut permissions = std::fs::metadata(&bin).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&bin, permissions).unwrap();

        let mut paths = Paths::at(dir.path().to_path_buf()).unwrap();
        paths.whispercpp_bin = bin;
        paths.whispercpp_model = dir.path().join("model.bin");
        let output = paths.temp_dir.join("out.json");
        let cancel = CancelToken::new();
        let reporter: Arc<dyn Reporter> = Arc::new(crate::report::SilentReporter);
        let segments = run_cli(
            &paths,
            Path::new("audio.wav"),
            &output,
            DeviceChoice::Cpu,
            &cancel,
            &reporter,
            CliOptions::default(),
        )
        .unwrap();
        assert!(find_repetition(&segments).is_some());

        let repaired = repair_repetitions(
            &paths,
            Path::new("audio.wav"),
            &output,
            &cancel,
            &reporter,
            &segments,
        )
        .unwrap()
        .expect("repair was not attempted");
        assert!(find_repetition(&repaired).is_none());
        let texts: Vec<&str> = repaired.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(
            texts,
            vec![
                "Intro.",
                "first repaired line.",
                "second repaired line.",
                "Outro."
            ]
        );
        assert!((repaired[1].start - 9.0).abs() < 0.01);
        assert!((repaired[2].end - 60.0).abs() < 0.01);
    }

    fn segment(start: f64, end: f64, text: &str) -> Segment {
        Segment::new(start, end, text)
    }

    #[test]
    fn detects_a_long_sentence_repeated_more_than_four_times() {
        let repeated = "The quarterly report is on the table by the door.";
        let segments: Vec<Segment> = (0..6)
            .map(|index| segment(index as f64 * 10.0, index as f64 * 10.0 + 10.0, repeated))
            .collect();
        let found = find_repetition(&segments).expect("loop not detected");
        assert_eq!(found.count, 6);
        assert!((found.start - 0.0).abs() < 0.01);
        assert!((found.end - 60.0).abs() < 0.01);
    }

    #[test]
    fn accepts_four_long_repeats_and_up_to_eight_short_interjections() {
        let long: Vec<Segment> = (0..4)
            .map(|index| segment(index as f64, index as f64 + 1.0, "This is a long sentence."))
            .collect();
        assert!(find_repetition(&long).is_none());

        let short: Vec<Segment> = (0..8)
            .map(|index| segment(index as f64, index as f64 + 1.0, "Yes."))
            .collect();
        assert!(find_repetition(&short).is_none());

        let too_many: Vec<Segment> = (0..9)
            .map(|index| segment(index as f64, index as f64 + 1.0, "Yes."))
            .collect();
        assert!(find_repetition(&too_many).is_some());
    }

    #[test]
    fn repetition_matching_ignores_case_and_punctuation_but_not_words() {
        let segments = vec![
            segment(0.0, 1.0, "Thank you very much."),
            segment(1.0, 2.0, "thank you very much"),
            segment(2.0, 3.0, "Thank you very much!"),
            segment(3.0, 4.0, "Thank you very much?"),
            segment(4.0, 5.0, "Thank you very much."),
            segment(5.0, 6.0, "A different sentence."),
            segment(6.0, 7.0, "A different sentence."),
        ];
        let found = find_repetition(&segments).expect("normalised repeats not detected");
        assert_eq!(found.count, 5);

        let distinct = vec![
            segment(0.0, 1.0, "One two three"),
            segment(1.0, 2.0, "One two four"),
            segment(2.0, 3.0, "One two three"),
            segment(3.0, 4.0, "One two four"),
            segment(4.0, 5.0, "One two three"),
            segment(5.0, 6.0, "One two four"),
        ];
        assert!(find_repetition(&distinct).is_none());
    }

    #[test]
    fn repetition_inside_one_segment_uses_character_positions() {
        let text = "A sentence that repeats. A sentence that repeats. A sentence that repeats. A sentence that repeats. A sentence that repeats.";
        let segments = vec![segment(10.0, 30.0, text)];
        let found = find_repetition(&segments).expect("in-segment loop not detected");
        assert_eq!(found.count, 5);
        assert!(found.start >= 10.0 && found.end <= 30.0);
        assert!(found.end - found.start > 10.0);
    }
}
