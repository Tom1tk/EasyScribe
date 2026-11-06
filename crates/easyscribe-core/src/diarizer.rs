//! Speaker diarization and its time estimate, ported from `src/diarizer.py`.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::Value;
use sherpa_rs::diarize::{Diarize, DiarizeConfig};

use crate::audio::{load_wav_float32, wav_duration};
use crate::ffmpeg::fmt_duration;
use crate::model::{Segment, Turn};
use crate::report::CancelToken;
use crate::settings::Settings;
use crate::transcript::speech_seconds;
use crate::{Error, Paths, Reporter, Result};

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

        let started = Instant::now();
        let config = DiarizeConfig {
            num_clusters: Some(-1),
            threshold: Some(0.5),
            min_duration_on: Some(0.0),
            min_duration_off: Some(0.0),
            provider: Some("cpu".into()),
            debug: false,
        };
        let mut engine = Diarize::new(
            &self.paths.separator_model,
            &self.paths.embedding_model,
            config,
        )
        .map_err(|e| Error::Diarization(format!("Speaker diarization failed: {e}")))?;

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

        let segments = match engine.compute(samples, Some(progress)) {
            Ok(segments) => segments,
            Err(_) if cancel.is_cancelled() => return Err(Error::Cancelled),
            Err(e) if e.to_string().contains("No segments") => Vec::new(),
            Err(e) => {
                return Err(Error::Diarization(format!(
                    "Speaker diarization failed: {e}"
                )));
            }
        };
        let _ = started;

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
}

/// Run diarization in a child process of the same executable. A crash or a
/// Stop cannot take the UI down, and the child's memory is reclaimed when it
/// exits. The worker protocol is newline-delimited JSON on stdout.
pub fn diarize_in_child(
    paths: &Paths,
    audio_path: &Path,
    cancel: &CancelToken,
    reporter: Arc<dyn Reporter>,
) -> Result<Vec<Turn>> {
    let exe = std::env::current_exe().map_err(Error::Io)?;
    let mut command = Command::new(&exe);
    command
        .arg("--worker")
        .arg("diarize")
        .arg("--wav")
        .arg(audio_path)
        .arg("--threads")
        .arg(crate::config::num_threads().to_string())
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
        let _ = child.kill();
        let _ = child.wait();
        return Err(Error::Diarization(
            "The diarization worker has no stdout".into(),
        ));
    };
    let Some(stderr) = child.stderr.take() else {
        let _ = child.kill();
        let _ = child.wait();
        return Err(Error::Diarization(
            "The diarization worker has no stderr".into(),
        ));
    };
    let (tx, rx) = std::sync::mpsc::channel::<WorkerMessage>();
    let stdout_reader = {
        let tx = tx.clone();
        match std::thread::Builder::new()
            .name("DiarizeStdout".into())
            .spawn(move || {
                for line in BufReader::new(stdout)
                    .lines()
                    .map_while(std::result::Result::ok)
                {
                    if line.trim().is_empty() {
                        continue;
                    }
                    let message = match serde_json::from_str::<Value>(&line) {
                        Ok(value) => WorkerMessage::parse(value),
                        Err(error) => WorkerMessage::Malformed(format!(
                            "The diarization worker sent invalid JSON ({error}): {line}"
                        )),
                    };
                    if tx.send(message).is_err() {
                        break;
                    }
                }
            }) {
            Ok(reader) => reader,
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(Error::Diarization(format!(
                    "Could not read the diarization worker: {error}"
                )));
            }
        }
    };
    let stderr_reader = {
        let tx = tx.clone();
        let reporter = reporter.clone();
        match std::thread::Builder::new()
            .name("DiarizeStderr".into())
            .spawn(move || {
                for line in BufReader::new(stderr)
                    .lines()
                    .map_while(std::result::Result::ok)
                {
                    reporter.log(&line);
                    let _ = tx.send(WorkerMessage::Log);
                }
            }) {
            Ok(reader) => reader,
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = stdout_reader.join();
                return Err(Error::Diarization(format!(
                    "Could not read the diarization log: {error}"
                )));
            }
        }
    };
    drop(tx);

    let mut turns: Option<Vec<Turn>> = None;
    let status = loop {
        if cancel.is_cancelled() {
            let _ = child.kill();
            let _ = child.wait();
            let _ = stdout_reader.join();
            let _ = stderr_reader.join();
            return Err(Error::Cancelled);
        }
        while let Ok(message) = rx.try_recv() {
            match message {
                WorkerMessage::Progress(fraction) => reporter.progress(fraction.clamp(0.0, 1.0)),
                WorkerMessage::Turns(value) => turns = Some(value),
                WorkerMessage::Error(message) | WorkerMessage::Malformed(message) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    let _ = stdout_reader.join();
                    let _ = stderr_reader.join();
                    return Err(Error::Diarization(message));
                }
                WorkerMessage::Log => {}
            }
        }
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => std::thread::sleep(Duration::from_millis(50)),
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = stdout_reader.join();
                let _ = stderr_reader.join();
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
            WorkerMessage::Turns(value) => turns = Some(value),
            WorkerMessage::Error(message) | WorkerMessage::Malformed(message) => {
                error = Some(message)
            }
            WorkerMessage::Log => {}
        }
    }

    if let Some(message) = error {
        return Err(Error::Diarization(message));
    }
    if !status.success() {
        return Err(Error::Diarization(format!(
            "The diarization worker stopped with exit code {:?}.",
            status.code()
        )));
    }
    turns.ok_or_else(|| {
        Error::Diarization("The diarization worker finished without a result.".into())
    })
}

enum WorkerMessage {
    Progress(f64),
    Turns(Vec<Turn>),
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
            return match serde_json::from_value::<Vec<Turn>>(turns.clone()) {
                Ok(turns) => Self::Turns(turns),
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

    let result = DiarizationEngine::new(paths.clone()).diarize(
        wav,
        &CancelToken::new(),
        Arc::new(JsonReporter),
    );
    match result {
        Ok(turns) => {
            write_worker_line(&serde_json::json!({ "turns": turns }));
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
    fn estimates_read_like_sentences() {
        assert_eq!(fmt_estimate(10.0), "less than a minute");
        assert_eq!(fmt_estimate(60.0), "about 1 minute");
        assert_eq!(fmt_estimate(4.0 * 60.0), "about 4 minutes");
        assert_eq!(fmt_estimate(60.0 * 60.0), "about 1 hour");
        assert_eq!(fmt_estimate(65.0 * 60.0), "about 1 hour 5 minutes");
    }
}
