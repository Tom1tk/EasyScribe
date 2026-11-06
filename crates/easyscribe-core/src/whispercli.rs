//! File transcription through the bundled `whisper-cli` child process.
//!
//! The Python v3.2.3 app ran the Vulkan build of whisper.cpp this way. Keeping
//! that proven path means a GPU driver crash only kills the child, Stop is a
//! process kill, and the output comes from exactly the same binary and flags
//! as the stable release. The in-process `whispercpp` engine remains the
//! fallback when no bundled binary is present (development and Linux tests).

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
    let regions =
        crate::whispercpp::speech_regions(paths, &samples, cancel, reporter.as_ref(), false)?;
    // The child process reads the WAV itself. Drop our copy before starting it
    // so a long file does not hold hundreds of megabytes beside the model.
    drop(samples);
    if regions.is_empty() {
        if peak < 0.01 {
            reporter.progress(1.0);
            reporter.log("[whisper.cpp] No speech found. Nothing to transcribe.");
            return Ok(Vec::new());
        }
        // Silero can miss quiet speech, music or unusual languages. A nonzero
        // signal must still reach whisper-cli instead of becoming an empty
        // transcript.
        reporter.log(
            "[whisper.cpp] Warning: speech detection found no regions, but the audio is not silent; decoding the whole file.",
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
    let first = run_cli(paths, audio_path, &output, choice, cancel, reporter);
    let segments = match first {
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
        "[whisper.cpp] The bundled whisper-cli could not start ({error}). Using the in-process engine."
    ));
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

fn run_cli(
    paths: &Paths,
    audio_path: &Path,
    output: &Path,
    choice: DeviceChoice,
    cancel: &CancelToken,
    reporter: &Arc<dyn Reporter>,
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
    let mut tail = Vec::new();

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

/// Stop a child for cancellation/error. If it cannot be reaped within five
/// seconds (a GPU driver stuck in kernel mode), leave the reader thread
/// detached; the Windows job object reaps the process when the app exits.
fn stop_child(child: &mut std::process::Child, reader: std::thread::JoinHandle<()>) {
    let reaped = kill_and_reap(child);
    if reaped {
        let _ = reader.join();
    }
}

fn drain_stderr(lines: &Receiver<String>, reporter: &Arc<dyn Reporter>, tail: &mut Vec<String>) {
    while let Ok(line) = lines.try_recv() {
        describe_stderr(&line, reporter);
        push_tail(tail, line);
    }
}

fn push_tail(tail: &mut Vec<String>, line: String) {
    tail.push(line);
    if tail.len() > 40 {
        tail.remove(0);
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
            )
        });
        std::thread::sleep(Duration::from_millis(150));
        let started = Instant::now();
        cancel.cancel();
        let result = handle.join().unwrap();
        assert!(matches!(result, Err(Error::Cancelled)));
        assert!(started.elapsed() < Duration::from_secs(2));
    }
}
