//! Audio extraction and probing through ffmpeg/ffprobe, ported from
//! `src/ffmpeg_wrapper.py`. The exact command-line arguments and the
//! `-protocol_whitelist file` restriction are part of the offline guarantee.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::report::CancelToken;
use crate::{Error, Paths, Reporter, Result};

/// Offline/GDPR: let ffmpeg read local files only. Must come before `-i`.
const LOCAL_FILES_ONLY: [&str; 2] = ["-protocol_whitelist", "file"];

#[derive(Debug, Clone, Default, PartialEq)]
pub struct AudioInfo {
    pub codec_name: Option<String>,
    pub sample_rate: Option<u32>,
    pub channels: Option<u32>,
    pub duration_seconds: Option<f64>,
}

/// Run ffprobe and return the first audio stream's basic properties.
pub fn probe_audio(paths: &Paths, input: &Path) -> Option<AudioInfo> {
    if !ffprobe_exists(paths) {
        return None;
    }
    let mut command = Command::new(&paths.ffprobe_bin);
    command
        .arg("-v")
        .arg("error")
        .arg("-select_streams")
        .arg("a:0")
        .arg("-show_entries")
        .arg("stream=codec_name,sample_rate,channels,duration")
        .arg("-of")
        .arg("json")
        .args(LOCAL_FILES_ONLY)
        .arg(input)
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    hide_window(&mut command);
    let mut child = command.spawn().ok()?;
    #[cfg(windows)]
    crate::win_job::assign(&child);
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return None;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(_) => return None,
        }
    }
    let output = child.wait_with_output().ok()?;
    let value: Value = serde_json::from_slice(&output.stdout).ok()?;
    let stream = value.get("streams")?.get(0)?;
    Some(AudioInfo {
        codec_name: stream
            .get("codec_name")
            .and_then(Value::as_str)
            .map(str::to_owned),
        sample_rate: stream
            .get("sample_rate")
            .and_then(Value::as_str)
            .and_then(|s| s.parse().ok()),
        channels: stream
            .get("channels")
            .and_then(Value::as_u64)
            .map(|c| c as u32),
        duration_seconds: stream
            .get("duration")
            .and_then(Value::as_str)
            .and_then(|s| s.parse().ok()),
    })
}

/// A WAV that is already mono 16 kHz PCM s16le can be copied without ffmpeg.
fn is_already_suitable_wav(input: &Path, info: &AudioInfo) -> bool {
    if input
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("wav"))
        != Some(true)
    {
        return false;
    }
    info.codec_name.as_deref() == Some("pcm_s16le")
        && info.sample_rate == Some(16_000)
        && info.channels == Some(1)
}

/// The user-facing duration format used in logs: `4m 05s`, `1h 02m 03s`.
pub fn fmt_duration(seconds: f64) -> String {
    let total = seconds.max(0.0) as u64;
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    if h > 0 {
        format!("{h}h {m:02}m {s:02}s")
    } else {
        format!("{m}m {s:02}s")
    }
}

/// Total duration from an ffmpeg stderr header line.
pub fn parse_duration(line: &str) -> Option<f64> {
    static DURATION: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(r"Duration:\s*(\d+):(\d+):([\d.]+)").expect("valid duration regex")
    });
    let captures = DURATION.captures(line)?;
    let h: f64 = captures[1].parse().ok()?;
    let m: f64 = captures[2].parse().ok()?;
    let s: f64 = captures[3].parse().ok()?;
    Some(h * 3600.0 + m * 60.0 + s)
}

/// Current encode position from an ffmpeg `time=` progress line.
pub fn parse_time(line: &str) -> Option<f64> {
    static TIME: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(r"time=(\d+):(\d+):([\d.]+)").expect("valid time regex")
    });
    let captures = TIME.captures(line)?;
    let h: f64 = captures[1].parse().ok()?;
    let m: f64 = captures[2].parse().ok()?;
    let s: f64 = captures[3].parse().ok()?;
    Some(h * 3600.0 + m * 60.0 + s)
}

/// Convert `input` to a temporary mono 16 kHz PCM WAV. Streams ffmpeg progress
/// to the reporter and polls the cancel token every 100 ms.
pub fn extract_audio(
    paths: &Paths,
    input: &Path,
    cancel: &CancelToken,
    reporter: &dyn Reporter,
) -> Result<PathBuf> {
    cancel.check()?;
    validate_ffmpeg(paths)?;

    let log = |message: String| {
        log::info!("{message}");
        reporter.log(&message);
    };

    // Probe once: the suitability test and the displayed input properties
    // used to start ffprobe twice for every non-suitable WAV.
    let info = probe_audio(paths, input);
    if let Some(info) = &info
        && is_already_suitable_wav(input, info)
    {
        log(format!(
            "[ffmpeg] Already mono 16 kHz PCM WAV — copying directly: {}",
            filename(input)
        ));
        let temp = paths.temp_wav();
        std::fs::copy(input, &temp)?;
        return Ok(temp);
    }

    if let Some(info) = info {
        let duration = info
            .duration_seconds
            .map(fmt_duration)
            .unwrap_or_else(|| "unknown duration".into());
        log(format!(
            "[ffmpeg] Input: codec={}, sample_rate={} Hz, channels={}, duration={}",
            info.codec_name.as_deref().unwrap_or("?"),
            info.sample_rate
                .map(|v| v.to_string())
                .unwrap_or_else(|| "?".into()),
            info.channels
                .map(|v| v.to_string())
                .unwrap_or_else(|| "?".into()),
            duration
        ));
    } else {
        log("[ffmpeg] Could not probe audio stream — proceeding anyway".into());
    }

    let temp_wav = paths.temp_wav();
    log(format!(
        "[ffmpeg] Extracting audio → mono 16 kHz PCM WAV: {}",
        filename(input)
    ));
    log(format!("[ffmpeg] Temp file: {}", filename(&temp_wav)));

    let mut command = Command::new(&paths.ffmpeg_bin);
    command
        .arg("-y")
        .args(LOCAL_FILES_ONLY)
        .arg("-i")
        .arg(input)
        .arg("-ac")
        .arg("1")
        .arg("-ar")
        .arg("16000")
        .arg("-vn")
        .arg("-acodec")
        .arg("pcm_s16le")
        .arg("-stats")
        .arg(&temp_wav)
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    hide_window(&mut command);

    let mut child = command.spawn().map_err(|e| {
        let _ = std::fs::remove_file(&temp_wav);
        Error::FfmpegExtraction(format!("Could not start ffmpeg: {e}"))
    })?;
    #[cfg(windows)]
    crate::win_job::assign(&child);
    let stderr = child.stderr.take().expect("piped stderr");

    // One reader thread decodes stderr and sends complete lines; ffmpeg ends
    // progress updates with \r, so split on both \r and \n.
    let (tx, rx) = mpsc::channel::<String>();
    let reader = std::thread::spawn(move || read_lines(stderr, tx));

    let mut lines: Vec<String> = Vec::new();
    let mut total_duration = 0.0_f64;
    let mut last_progress_log = Instant::now() - Duration::from_secs(2);
    let mut last_progress_value: Option<String> = None;
    let mut failure: Option<Error> = None;

    loop {
        if cancel.is_cancelled() {
            log::info!("Cancel requested — terminating ffmpeg");
            let _ = child.kill();
            let _ = child.wait();
            let _ = std::fs::remove_file(&temp_wav);
            return Err(Error::Cancelled);
        }
        while let Ok(line) = rx.try_recv() {
            handle_line(
                &line,
                &mut lines,
                &mut total_duration,
                &mut last_progress_log,
                &mut last_progress_value,
                &log,
            );
        }
        match child.try_wait() {
            Ok(Some(status)) => {
                if !status.success() {
                    failure = Some(Error::FfmpegExtraction(format!(
                        "ffmpeg exited with code {}.\n\n{}",
                        status.code().unwrap_or(-1),
                        snippet(&lines)
                    )));
                }
                break;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(100)),
            Err(e) => {
                failure = Some(Error::FfmpegExtraction(format!("ffmpeg wait failed: {e}")));
                let _ = child.kill();
                break;
            }
        }
    }

    let _ = reader.join();
    while let Ok(line) = rx.try_recv() {
        handle_line(
            &line,
            &mut lines,
            &mut total_duration,
            &mut last_progress_log,
            &mut last_progress_value,
            &log,
        );
    }
    if let Some(error) = failure {
        let _ = std::fs::remove_file(&temp_wav);
        return Err(error);
    }

    let size_kb = std::fs::metadata(&temp_wav)
        .map(|m| m.len() / 1024)
        .unwrap_or(0);
    log(format!(
        "[ffmpeg] Extraction complete — {size_kb} KB written to {}",
        filename(&temp_wav)
    ));
    Ok(temp_wav)
}

fn handle_line(
    line: &str,
    lines: &mut Vec<String>,
    total_duration: &mut f64,
    last_progress_log: &mut Instant,
    last_progress_value: &mut Option<String>,
    log: &impl Fn(String),
) {
    lines.push(line.to_owned());
    if *total_duration == 0.0
        && let Some(d) = parse_duration(line)
    {
        *total_duration = d;
        log(format!("[ffmpeg] Media duration: {}", fmt_duration(d)));
    }
    if let Some(pos) = parse_time(line) {
        if *total_duration > 0.0 && last_progress_log.elapsed() >= Duration::from_secs(1) {
            *last_progress_log = Instant::now();
            let pct = (pos / *total_duration * 100.0).min(100.0);
            let message = format!(
                "[ffmpeg] Extracting… {} / {}  ({pct:.0}%)",
                fmt_duration(pos),
                fmt_duration(*total_duration)
            );
            if last_progress_value.as_deref() != Some(&message) {
                *last_progress_value = Some(message.clone());
                log(message);
            }
        }
    } else if !line.is_empty()
        && !line.starts_with("ffmpeg version")
        && !line.to_ascii_lowercase().contains("encoder")
    {
        let lower = line.to_ascii_lowercase();
        if ["error", "warn", "invalid", "no such", "stream #"]
            .iter()
            .any(|k| lower.contains(k))
        {
            log(format!("[ffmpeg] {line}"));
        }
    }
}

fn read_lines(mut stream: impl Read, tx: mpsc::Sender<String>) {
    let mut buffer = Vec::new();
    let mut chunk = [0_u8; 4096];
    loop {
        match stream.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => {
                buffer.extend_from_slice(&chunk[..n]);
                let mut start = 0;
                let mut i = 0;
                while i < buffer.len() {
                    if buffer[i] == b'\n' || buffer[i] == b'\r' {
                        let mut end = i;
                        if buffer[i] == b'\r' && buffer.get(i + 1) == Some(&b'\n') {
                            end = i;
                            i += 1;
                        }
                        let line = String::from_utf8_lossy(&buffer[start..end])
                            .trim_end()
                            .to_string();
                        let _ = tx.send(line);
                        start = i + 1;
                    }
                    i += 1;
                }
                buffer.drain(..start);
            }
            Err(_) => break,
        }
    }
    if !buffer.is_empty() {
        let _ = tx.send(String::from_utf8_lossy(&buffer).trim_end().to_string());
    }
}

fn snippet(lines: &[String]) -> String {
    let joined = lines.join("\n");
    let chars: Vec<char> = joined.chars().collect();
    let start = chars.len().saturating_sub(800);
    chars[start..].iter().collect::<String>().trim().to_string()
}

pub fn validate_ffmpeg(paths: &Paths) -> Result<()> {
    let mut missing = Vec::new();
    if !ffmpeg_exists(paths) {
        missing.push(format!(
            "ffmpeg not found at: {}",
            paths.ffmpeg_bin.display()
        ));
    }
    if !ffprobe_exists(paths) {
        missing.push(format!(
            "ffprobe not found at: {}",
            paths.ffprobe_bin.display()
        ));
    }
    if missing.is_empty() {
        Ok(())
    } else {
        Err(Error::FfmpegNotFound(missing.join("\n")))
    }
}

fn ffmpeg_exists(paths: &Paths) -> bool {
    // A bare "ffmpeg" (development fallback) is resolved from PATH.
    paths.ffmpeg_bin.components().count() == 1 || paths.ffmpeg_bin.is_file()
}

fn ffprobe_exists(paths: &Paths) -> bool {
    paths.ffprobe_bin.components().count() == 1 || paths.ffprobe_bin.is_file()
}

fn filename(path: &Path) -> String {
    path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("?")
        .to_owned()
}

#[cfg(windows)]
fn hide_window(command: &mut Command) {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    command.creation_flags(CREATE_NO_WINDOW);
}

#[cfg(not(windows))]
fn hide_window(_command: &mut Command) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_duration_and_time() {
        assert_eq!(
            parse_duration("  Duration: 00:04:05.67, start: 0.000000"),
            Some(245.67)
        );
        assert_eq!(
            parse_time("size=1kB time=00:00:02.50 bitrate=..."),
            Some(2.5)
        );
        assert_eq!(parse_duration("nope"), None);
    }

    #[test]
    fn formats_durations_like_the_logs() {
        assert_eq!(fmt_duration(245.0), "4m 05s");
        assert_eq!(fmt_duration(3723.0), "1h 02m 03s");
    }
}
