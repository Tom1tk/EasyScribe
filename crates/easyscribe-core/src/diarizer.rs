//! Speaker diarization and its time estimate, ported from `src/diarizer.py`.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::audio::{load_wav_float32, wav_duration};
use crate::ffmpeg::fmt_duration;
use crate::local_diarize::{Config as DiarizeConfig, LocalDiarizer};
use crate::model::{Segment, Turn};
use crate::report::CancelToken;
use crate::settings::Settings;
use crate::transcript::speech_seconds;
use crate::{Error, Paths, Reporter, Result};

// Fitted against the Python multi-thread pipeline. Rust now passes the same
// thread count to both native models again; re-fit these only from a real-PC
// run of a fixed 10+ minute file, not from the 23 s CI fixture.
pub const SEC_PER_AUDIO_SEC: f64 = 0.045;
pub const SEC_PER_SPEECH_SEC: f64 = 0.165;
pub const START_SECONDS: f64 = 3.0;

#[derive(Debug, Clone)]
pub struct DiarizationEngine {
    paths: Paths,
}

impl DiarizationEngine {
    pub fn new(paths: Paths) -> Self {
        Self { paths }
    }

    pub fn is_available(&self) -> bool {
        self.paths.diarization_available()
    }

    /// Rough time to find the speakers, adjusted by this PC's measured speed.
    pub fn estimate_seconds(&self, audio_sec: f64, speech_sec: Option<f64>) -> f64 {
        let settings = Settings::load(&self.paths.settings_file);
        model_seconds(audio_sec, speech_sec) * settings.load_speaker_speed().unwrap_or(1.0)
    }

    /// Learn this PC's speed from a finished run (moving average).
    pub fn save_measured_speed(&self, audio_path: &Path, segments: &[Segment], took_sec: f64) {
        let Ok(duration) = wav_duration(audio_path) else {
            return;
        };
        let expected = model_seconds(duration, Some(speech_seconds(segments)));
        if expected <= 0.0 {
            return;
        }
        let factor = (took_sec / expected).clamp(0.2, 5.0);
        let mut settings = Settings::load(&self.paths.settings_file);
        let previous = settings.load_speaker_speed();
        let value = previous.map(|p| (p + factor) / 2.0).unwrap_or(factor);
        let _ = settings.save_speaker_speed(value);
    }

    /// Run diarization on a 16 kHz mono WAV.
    pub fn diarize(
        &self,
        audio_path: &Path,
        cancel: &CancelToken,
        reporter: Arc<dyn Reporter>,
    ) -> Result<Vec<Turn>> {
        self.diarize_inner(audio_path, cancel, reporter, false, None)
    }

    /// `keep_engine_alive` is used by the short-lived worker. Dropping the
    /// native diarizer can be slow or crash; once the turns are computed the
    /// worker sends them and exits, so leaking the engine is safer than losing
    /// a valid result during teardown.
    fn diarize_inner(
        &self,
        audio_path: &Path,
        cancel: &CancelToken,
        reporter: Arc<dyn Reporter>,
        keep_engine_alive: bool,
        compute_seconds: Option<&mut f64>,
    ) -> Result<Vec<Turn>> {
        if !self.is_available() {
            return Err(Error::Diarization("The speaker models are missing.".into()));
        }
        let duration = wav_duration(audio_path).map_err(|e| {
            Error::Diarization(format!("Could not load audio for diarization: {e}"))
        })?;
        let estimate = fmt_estimate(self.estimate_seconds(duration, None));
        reporter.log(&format!(
            "[Diarize] Estimated wait: {estimate} for {} of audio (rough estimate). Please wait…",
            fmt_duration(duration)
        ));
        log::info!(
            "Running diarization on: {} (estimate {estimate})",
            audio_path.display()
        );
        reporter.log("[Diarize] Finding the parts with speech…");

        if cancel.is_cancelled() {
            return Err(Error::Cancelled);
        }
        let (samples, _) = load_wav_float32(audio_path).map_err(|e| {
            Error::Diarization(format!("Could not load audio for diarization: {e}"))
        })?;

        let config = DiarizeConfig {
            num_clusters: -1,
            threshold: 0.5,
            min_duration_on: 0.3,
            min_duration_off: 0.5,
            provider: "cpu".into(),
            num_threads: crate::config::num_threads() as i32,
        };
        let mut engine = LocalDiarizer::new(
            &self.paths.separator_model,
            &self.paths.embedding_model,
            config,
        )?;

        let progress_reporter = reporter.clone();
        let progress_cancel = cancel.clone();
        let progress = Box::new(move |done: i32, total: i32| -> i32 {
            if progress_cancel.is_cancelled() {
                return 1;
            }
            if total > 0 {
                let pct = (100 * done / total) as f64;
                progress_reporter.progress(pct / 100.0);
                if done * 10 / total != (done - 1) * 10 / total.max(1) {
                    progress_reporter.log(&format!("[Diarize] Progress: {pct:.0}%"));
                }
            }
            0
        });

        let compute_started = Instant::now();
        let result = match engine.compute(samples, Some(progress)) {
            Ok(segments) => {
                let turns: Vec<Turn> = segments
                    .into_iter()
                    .map(|s| Turn {
                        start: s.start as f64,
                        end: s.end as f64,
                        speaker: format!("SPEAKER_{:02}", s.speaker),
                    })
                    .collect();
                let speakers: std::collections::BTreeSet<&str> =
                    turns.iter().map(|t| t.speaker.as_str()).collect();
                reporter.log(&format!(
                    "[Diarize] Found {} speaker(s) across {} turn(s)",
                    speakers.len(),
                    turns.len()
                ));
                log::info!(
                    "Diarization complete: {} speakers, {} turns",
                    speakers.len(),
                    turns.len()
                );
                Ok(turns)
            }
            Err(_) if cancel.is_cancelled() => Err(Error::Cancelled),
            Err(e) if e.to_string().contains("No segments") => Ok(Vec::new()),
            Err(e) => Err(Error::Diarization(format!(
                "Speaker diarization failed: {e}"
            ))),
        };
        let compute_elapsed = compute_started.elapsed().as_secs_f64();
        reporter.log(&format!(
            "[Diarize] Native compute took {:.1}s",
            compute_elapsed
        ));
        log::info!(
            "Diarization compute: {:.3}s for {:.1}s of audio ({} threads)",
            compute_elapsed,
            duration,
            crate::config::num_threads()
        );
        if let Some(slot) = compute_seconds {
            *slot = compute_elapsed;
        }
        // The worker exits immediately after writing the result; skipping the
        // native destructor avoids losing that result to a teardown crash.
        if keep_engine_alive {
            std::mem::forget(engine);
        }
        result
    }
}

/// Run diarization in a child process of the same executable. A crash or a
/// Stop cannot take the UI down, and the child's memory is reclaimed when it
/// exits. The worker protocol is newline-delimited JSON on stdout.
pub fn diarize_in_child(
    paths: &Paths,
    audio_path: &Path,
    cancel: &CancelToken,
    reporter: Arc<dyn Reporter>,
) -> Result<(Vec<Turn>, f64)> {
    let exe = std::env::current_exe().map_err(Error::Io)?;
    let mut command = Command::new(&exe);
    command
        .arg("--worker")
        .arg("diarize")
        .arg("--wav")
        .arg(audio_path)
        .env("EASYSCRIBE_BASE_DIR", &paths.base_dir)
        .env("EASYSCRIBE_DATA_DIR", &paths.data_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000);
    }
    let mut child = command
        .spawn()
        .map_err(|e| Error::Diarization(format!("Could not start the diarization worker: {e}")))?;
    #[cfg(windows)]
    crate::win_job::assign(&child);

    let Some(stdout) = child.stdout.take() else {
        kill_and_reap(&mut child);
        return Err(Error::Diarization(
            "The diarization worker has no stdout".into(),
        ));
    };
    let Some(stderr) = child.stderr.take() else {
        kill_and_reap(&mut child);
        return Err(Error::Diarization(
            "The diarization worker has no stderr".into(),
        ));
    };
    let (tx, rx) = std::sync::mpsc::channel::<WorkerMessage>();
    let stdout_reader = match spawn_stdout_reader(stdout, tx.clone()) {
        Ok(reader) => reader,
        Err(error) => {
            kill_and_reap(&mut child);
            return Err(Error::Diarization(format!(
                "Could not read the diarization worker: {error}"
            )));
        }
    };
    let stderr_reader = match spawn_stderr_reader(stderr, tx.clone(), reporter.clone()) {
        Ok(reader) => reader,
        Err(error) => {
            kill_and_reap(&mut child);
            let _ = stdout_reader.join();
            return Err(Error::Diarization(format!(
                "Could not read the diarization log: {error}"
            )));
        }
    };
    drop(tx);

    let deadline = std::env::var("EASYSCRIBE_DIARIZE_TIMEOUT_SEC")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|seconds| *seconds > 0)
        .map(|seconds| Instant::now() + Duration::from_secs(seconds));

    let mut turns: Option<(Vec<Turn>, f64)> = None;
    let status = loop {
        if cancel.is_cancelled() {
            stop_worker(&mut child, stdout_reader, stderr_reader);
            return Err(Error::Cancelled);
        }
        if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            stop_worker(&mut child, stdout_reader, stderr_reader);
            return Err(Error::Diarization(
                "The diarization worker took too long and was stopped.".into(),
            ));
        }
        while let Ok(message) = rx.try_recv() {
            match message {
                WorkerMessage::Progress(fraction) => reporter.progress(fraction.clamp(0.0, 1.0)),
                // Turns are complete as soon as they arrive. Do not wait for
                // the child to exit: a teardown crash after the result must not
                // discard a valid answer.
                WorkerMessage::Turns { turns, took_sec } => {
                    stop_worker(&mut child, stdout_reader, stderr_reader);
                    return Ok((turns, took_sec));
                }
                WorkerMessage::Error(message) | WorkerMessage::Malformed(message) => {
                    stop_worker(&mut child, stdout_reader, stderr_reader);
                    return Err(Error::Diarization(message));
                }
                WorkerMessage::Log => {}
            }
        }
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => std::thread::sleep(Duration::from_millis(50)),
            Err(error) => {
                stop_worker(&mut child, stdout_reader, stderr_reader);
                return Err(Error::Diarization(format!(
                    "Could not wait for the diarization worker: {error}"
                )));
            }
        }
    };

    let _ = stdout_reader.join();
    let _ = stderr_reader.join();
    let mut error = None;
    while let Ok(message) = rx.try_recv() {
        match message {
            WorkerMessage::Progress(fraction) => reporter.progress(fraction.clamp(0.0, 1.0)),
            WorkerMessage::Turns {
                turns: value,
                took_sec,
            } => turns = Some((value, took_sec)),
            WorkerMessage::Error(message) | WorkerMessage::Malformed(message) => {
                error = Some(message)
            }
            WorkerMessage::Log => {}
        }
    }

    if let Some(message) = error {
        return Err(Error::Diarization(message));
    }
    if let Some(turns) = turns {
        return Ok(turns);
    }
    let code = status.code().map_or_else(
        || "terminated by a signal".to_owned(),
        |code| code.to_string(),
    );
    Err(Error::Diarization(format!(
        "The diarization worker stopped with exit code {code}."
    )))
}

fn spawn_stdout_reader(
    stdout: std::process::ChildStdout,
    tx: std::sync::mpsc::Sender<WorkerMessage>,
) -> std::io::Result<std::thread::JoinHandle<()>> {
    std::thread::Builder::new()
        .name("DiarizeStdout".into())
        .spawn(move || {
            let mut reader = BufReader::new(stdout);
            let mut line = Vec::new();
            loop {
                line.clear();
                match reader.read_until(b'\n', &mut line) {
                    Ok(0) => break,
                    Ok(_) => {
                        let text = String::from_utf8_lossy(&line)
                            .trim_end_matches(&['\r', '\n'][..])
                            .to_owned();
                        if text.trim().is_empty() {
                            continue;
                        }
                        let message = match serde_json::from_str::<Value>(&text) {
                            Ok(value) => WorkerMessage::parse(value),
                            Err(error) => WorkerMessage::Malformed(format!(
                                "The diarization worker sent invalid JSON ({error}): {text}"
                            )),
                        };
                        if tx.send(message).is_err() {
                            break;
                        }
                    }
                    Err(error) => {
                        let _ = tx.send(WorkerMessage::Malformed(format!(
                            "The diarization worker output could not be read: {error}"
                        )));
                        break;
                    }
                }
            }
        })
}

fn spawn_stderr_reader(
    stderr: std::process::ChildStderr,
    tx: std::sync::mpsc::Sender<WorkerMessage>,
    reporter: Arc<dyn Reporter>,
) -> std::io::Result<std::thread::JoinHandle<()>> {
    std::thread::Builder::new()
        .name("DiarizeStderr".into())
        .spawn(move || {
            let mut reader = BufReader::new(stderr);
            let mut line = Vec::new();
            loop {
                line.clear();
                match reader.read_until(b'\n', &mut line) {
                    Ok(0) => break,
                    Ok(_) => {
                        let text = String::from_utf8_lossy(&line)
                            .trim_end_matches(&['\r', '\n'][..])
                            .to_owned();
                        if !text.is_empty() {
                            reporter.log(&text);
                        }
                        let _ = tx.send(WorkerMessage::Log);
                    }
                    Err(error) => {
                        reporter.log(&format!("[Diarize] Could not read worker log: {error}"));
                        let _ = tx.send(WorkerMessage::Malformed(format!(
                            "The diarization worker log could not be read: {error}"
                        )));
                        break;
                    }
                }
            }
        })
}

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

fn stop_worker(
    child: &mut std::process::Child,
    stdout: std::thread::JoinHandle<()>,
    stderr: std::thread::JoinHandle<()>,
) {
    if kill_and_reap(child) {
        let _ = stdout.join();
        let _ = stderr.join();
    }
}

enum WorkerMessage {
    Progress(f64),
    Turns { turns: Vec<Turn>, took_sec: f64 },
    Error(String),
    Malformed(String),
    Log,
}

impl WorkerMessage {
    fn parse(value: Value) -> Self {
        if let Some(progress) = value.get("progress").and_then(Value::as_f64) {
            return Self::Progress(progress);
        }
        if let Some(turns) = value.get("turns") {
            let took_sec = value.get("took_sec").and_then(Value::as_f64).unwrap_or(0.0);
            return match serde_json::from_value::<Vec<Turn>>(turns.clone()) {
                Ok(turns) => Self::Turns { turns, took_sec },
                Err(error) => {
                    Self::Malformed(format!("The diarization worker sent bad turns: {error}"))
                }
            };
        }
        if let Some(message) = value.get("error").and_then(Value::as_str) {
            return Self::Error(message.to_owned());
        }
        Self::Malformed("The diarization worker sent an unknown message.".into())
    }
}

/// The hidden worker entry point (`easyscribe.exe --worker diarize --wav …`).
/// It writes only JSON lines to stdout; diagnostics go to stderr.
pub fn run_diarize_worker(paths: &Paths, wav: &Path) -> i32 {
    struct JsonReporter;
    impl Reporter for JsonReporter {
        fn progress(&self, fraction: f64) {
            write_worker_line(&serde_json::json!({
                "progress": fraction.clamp(0.0, 1.0)
            }));
        }
        fn log(&self, line: &str) {
            eprintln!("{line}");
        }
    }

    let mut compute_seconds = 0.0_f64;
    let result = DiarizationEngine::new(paths.clone()).diarize_inner(
        wav,
        &CancelToken::new(),
        Arc::new(JsonReporter),
        true,
        Some(&mut compute_seconds),
    );
    match result {
        Ok(turns) => {
            write_worker_line(&serde_json::json!({
                "turns": turns,
                "took_sec": compute_seconds,
            }));
            0
        }
        Err(error) => {
            write_worker_line(&serde_json::json!({ "error": error.to_string() }));
            1
        }
    }
}

fn write_worker_line(value: &Value) {
    let mut stdout = std::io::stdout().lock();
    let _ = serde_json::to_writer(&mut stdout, value);
    let _ = stdout.write_all(b"\n");
    let _ = stdout.flush();
}

fn model_seconds(audio_sec: f64, speech_sec: Option<f64>) -> f64 {
    let speech = speech_sec.map(|s| s.min(audio_sec)).unwrap_or(audio_sec);
    START_SECONDS + SEC_PER_AUDIO_SEC * audio_sec + SEC_PER_SPEECH_SEC * speech
}

/// "About 4 minutes" style estimate, rounded the way a person would say it.
pub fn fmt_estimate(seconds: f64) -> String {
    if seconds < 50.0 {
        return "less than a minute".into();
    }
    let minutes = (seconds / 60.0).round_ties_even() as i64;
    if minutes <= 1 {
        return "about 1 minute".into();
    }
    if minutes < 60 {
        return format!("about {minutes} minutes");
    }
    let mut hours = minutes / 60;
    let mut rest = minutes % 60;
    rest = 5 * ((rest as f64 / 5.0).round_ties_even() as i64);
    if rest == 60 {
        hours += 1;
        rest = 0;
    }
    let hours_text = if hours == 1 {
        "1 hour".to_string()
    } else {
        format!("{hours} hours")
    };
    if rest == 0 {
        format!("about {hours_text}")
    } else {
        format!("about {hours_text} {rest} minutes")
    }
}

/// The labels/clips/shares delivered to the speaker naming panel.
#[derive(Debug, Clone)]
pub struct SpeakerPreview {
    pub speaker_map: std::collections::HashMap<String, String>,
    pub clips: std::collections::HashMap<String, PathBuf>,
    pub shares: std::collections::HashMap<String, f64>,
}

/// A blocking callback implemented by the UI: show the panel, wait for names.
pub trait SpeakerNaming: Send + Sync {
    fn ask(&self, preview: &mut SpeakerPreview) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn bench_diarization_thread_counts() {
        let wav = std::env::var("EASYSCRIBE_BENCH_WAV").unwrap();
        let paths = Paths::discover().unwrap();
        let mut reference: Option<Vec<Turn>> = None;
        for threads in ["1", "4", "8"] {
            // SAFETY: this ignored benchmark is run alone.
            unsafe { std::env::set_var("EASYSCRIBE_NUM_THREADS", threads) };
            let engine = DiarizationEngine::new(paths.clone());
            let started = Instant::now();
            let turns = engine
                .diarize(
                    Path::new(&wav),
                    &CancelToken::new(),
                    Arc::new(crate::report::SilentReporter),
                )
                .unwrap();
            println!(
                "threads={threads} compute={:.2}s turns={}",
                started.elapsed().as_secs_f64(),
                turns.len()
            );
            if let Some(reference) = &reference {
                assert_eq!(reference, &turns, "turn output changed with thread count");
            } else {
                reference = Some(turns);
            }
        }
        // SAFETY: see above.
        unsafe { std::env::remove_var("EASYSCRIBE_NUM_THREADS") };
    }

    #[test]
    fn estimates_read_like_sentences() {
        assert_eq!(fmt_estimate(10.0), "less than a minute");
        assert_eq!(fmt_estimate(60.0), "about 1 minute");
        assert_eq!(fmt_estimate(4.0 * 60.0), "about 4 minutes");
        assert_eq!(fmt_estimate(60.0 * 60.0), "about 1 hour");
        assert_eq!(fmt_estimate(65.0 * 60.0), "about 1 hour 5 minutes");
    }
}
