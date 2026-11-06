//! File transcription through the bundled `whisper-cli` child process.
//!
//! The Python v3.2.3 app ran the Vulkan build of whisper.cpp this way. Keeping
//! that proven path means a GPU driver crash only kills the child, Stop is a
//! process kill, and the output comes from exactly the same binary and flags
//! as the stable release. The in-process `whispercpp` engine remains the
//! fallback when no bundled binary is present (development and Linux tests).

use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver};
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

/// Ask whisper-cli which Vulkan devices it sees, without loading the model.
/// An empty file is enough: the CLI lists devices, then rejects the model.
pub fn probe_devices(paths: &Paths) -> Vec<CliGpu> {
    let empty = paths
        .temp_dir
        .join(format!("gpu_probe_{}.bin", uuid::Uuid::new_v4().simple()));
    if std::fs::write(&empty, []).is_err() {
        return Vec::new();
    }
    let mut command = Command::new(&paths.whispercpp_bin);
    command
        .current_dir(&paths.whispercpp_dir)
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
    let devices = match run_probe(command, Duration::from_secs(30)) {
        Some(output) => parse_devices(&String::from_utf8_lossy(&output)),
        None => Vec::new(),
    };
    let _ = std::fs::remove_file(&empty);
    devices
}

fn run_probe(mut command: Command, timeout: Duration) -> Option<Vec<u8>> {
    let mut child = command.spawn().ok()?;
    let started = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => {
                let output = child.wait_with_output().ok()?;
                return Some(output.stderr);
            }
            Ok(None) if started.elapsed() >= timeout => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(20)),
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    }
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
/// then Vulkan's own order. A remembered preferred index wins when present.
pub fn pick_gpu(devices: &[CliGpu], preferred: Option<i32>) -> Option<CliGpu> {
    if let Some(preferred) = preferred
        && let Some(device) = devices.iter().find(|device| device.id == preferred)
    {
        return Some(device.clone());
    }
    devices
        .iter()
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
    let regions =
        crate::whispercpp::speech_regions(paths, &samples, cancel, reporter.as_ref(), false)?;
    if regions.is_empty() {
        reporter.progress(1.0);
        reporter.log("[whisper.cpp] No speech found. Nothing to transcribe.");
        return Ok(Vec::new());
    }

    let mut settings = crate::settings::Settings::load(paths.settings_file.clone());
    let devices = probe_devices(paths);
    for device in &devices {
        reporter.log(&format!(
            "[whisper.cpp] GPU {}: {} | {}",
            device.id, device.name, device.details
        ));
    }
    let preferred = settings.load_gpu_device();
    let gpu = pick_gpu(&devices, preferred);
    if let Some(gpu) = &gpu
        && preferred != Some(gpu.id)
        && let Err(error) = settings.save_gpu_device(Some(gpu.id))
    {
        log::warn!("Could not remember the GPU device: {error}");
    }
    let disabled = settings.load_gpu_disabled_for();
    let force_cpu = std::env::var_os("EASYSCRIBE_WHISPER_CPU").is_some();
    let gpu_usable = gpu
        .as_ref()
        .filter(|gpu| !force_cpu && disabled.as_deref() != Some(gpu.identity().as_str()));
    if force_cpu {
        reporter.log("[whisper.cpp] Engine: CPU (forced by EASYSCRIBE_WHISPER_CPU)");
    } else if let Some(gpu) = &gpu_usable {
        reporter.log(&format!(
            "[whisper.cpp] Engine: Vulkan (device {}: {})",
            gpu.id, gpu.name
        ));
    } else if gpu.is_some() {
        reporter.log("[whisper.cpp] Engine: CPU (this GPU failed before)");
    } else {
        reporter.log("[whisper.cpp] Engine: CPU (no Vulkan device found)");
    }

    let output = paths
        .temp_dir
        .join(format!("whispercpp_{}.json", uuid::Uuid::new_v4().simple()));
    let first = run_cli(
        paths,
        audio_path,
        &output,
        gpu_usable.as_ref().map(|gpu| gpu.id),
        cancel,
        reporter,
    );
    let segments = match first {
        Ok(segments) => segments,
        Err(Error::Cancelled) => {
            let _ = std::fs::remove_file(&output);
            return Err(Error::Cancelled);
        }
        Err(error) if gpu_usable.is_some() => {
            let gpu = gpu_usable.expect("checked above");
            reporter.log(&format!(
                "[whisper.cpp] GPU decode failed ({error}). Retrying on the CPU…"
            ));
            log::warn!("whisper-cli GPU decode failed, retrying on CPU: {error}");
            if let Err(save_error) = settings.save_gpu_disabled_for(Some(&gpu.identity())) {
                log::warn!("Could not remember the failed GPU: {save_error}");
            } else {
                reporter.log(&format!(
                    "[whisper.cpp] Using the CPU for {} until the graphics driver changes.",
                    gpu.name
                ));
            }
            run_cli(paths, audio_path, &output, None, cancel, reporter)?
        }
        Err(error) => return Err(error),
    };
    let _ = std::fs::remove_file(&output);
    reporter.progress(1.0);
    reporter.log(&format!(
        "[whisper.cpp] Decoded {} segment(s)",
        segments.len()
    ));
    Ok(segments)
}

fn run_cli(
    paths: &Paths,
    audio_path: &Path,
    output: &Path,
    gpu_device: Option<i32>,
    cancel: &CancelToken,
    reporter: &Arc<dyn Reporter>,
) -> Result<Vec<Segment>> {
    cancel.check()?;
    let output_stem = output.with_extension("");
    let _ = std::fs::remove_file(output);
    let _ = std::fs::remove_file(output_stem.with_extension("json"));

    let mut command = Command::new(&paths.whispercpp_bin);
    command
        .current_dir(&paths.whispercpp_dir)
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
    match gpu_device {
        Some(device) => {
            command.arg("-dev").arg(device.to_string());
            reporter.log(&format!("[whisper.cpp] Using Vulkan device {device}"));
        }
        None => {
            command.arg("-ng");
        }
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
            "Could not start {}: {e}",
            paths.whispercpp_bin.display()
        ))
    })?;
    let stderr = child.stderr.take();
    let (line_tx, line_rx) = mpsc::channel::<String>();
    let reader = stderr.map(|stderr| {
        std::thread::Builder::new()
            .name("whisper-cli-stderr".into())
            .spawn(move || {
                for line in BufReader::new(stderr)
                    .lines()
                    .map_while(std::result::Result::ok)
                {
                    if line_tx.send(line).is_err() {
                        break;
                    }
                }
            })
    });
    let reader = reader.transpose().ok().flatten();
    let mut tail = Vec::new();

    let status = loop {
        if cancel.is_cancelled() {
            let _ = child.kill();
            let _ = child.wait();
            if let Some(reader) = reader {
                let _ = reader.join();
            }
            return Err(Error::Cancelled);
        }
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                drain_stderr(&line_rx, reporter, &mut tail);
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                if let Some(reader) = reader {
                    let _ = reader.join();
                }
                return Err(Error::WhisperCpp(format!(
                    "Could not wait for whisper-cli: {error}"
                )));
            }
        }
    };
    if let Some(reader) = reader {
        let _ = reader.join();
    }
    drain_stderr(&line_rx, reporter, &mut tail);

    if !status.success() {
        let snippet = tail
            .iter()
            .rev()
            .take(12)
            .rev()
            .cloned()
            .collect::<Vec<_>>()
            .join("\n");
        return Err(Error::WhisperCpp(format!(
            "whisper-cli exited with code {:?}.\n\n{snippet}",
            status.code()
        )));
    }
    if !output.is_file() {
        return Err(Error::WhisperCpp(format!(
            "whisper-cli completed but did not produce {}",
            output.display()
        )));
    }
    parse_output(output)
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
        Regex::new(r"whisper_print_progress_callback: progress = (\d+)%")
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
        #[serde(default)]
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
        assert_eq!(pick_gpu(&devices, None).unwrap().id, 1);
        assert_eq!(pick_gpu(&devices, Some(0)).unwrap().id, 0);
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
        assert_eq!(pick_gpu(&devices, None).unwrap().id, 2);
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
                None,
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
