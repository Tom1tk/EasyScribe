//! Speaker diarization and its time estimate, ported from `src/diarizer.py`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use sherpa_rs::diarize::{Diarize, DiarizeConfig};

use crate::audio::{load_wav_float32, wav_duration};
use crate::ffmpeg::fmt_duration;
use crate::model::{Segment, Turn};
use crate::settings::Settings;
use crate::{Error, Paths, Reporter, Result};
use crate::report::CancelToken;
use crate::transcript::speech_seconds;

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
        let Ok(duration) = wav_duration(audio_path) else { return };
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
        let duration = wav_duration(audio_path)
            .map_err(|e| Error::Diarization(format!("Could not load audio for diarization: {e}")))?;
        let estimate = fmt_estimate(self.estimate_seconds(duration, None));
        reporter.log(&format!(
            "[Diarize] Estimated wait: {estimate} for {} of audio (rough estimate). Please wait…",
            fmt_duration(duration)
        ));
        log::info!("Running diarization on: {} (estimate {estimate})", audio_path.display());
        reporter.log("[Diarize] Finding the parts with speech…");

        if cancel.is_cancelled() {
            return Err(Error::Cancelled);
        }
        let (samples, _) = load_wav_float32(audio_path)
            .map_err(|e| Error::Diarization(format!("Could not load audio for diarization: {e}")))?;

        let started = Instant::now();
        let config = DiarizeConfig {
            num_clusters: Some(-1),
            threshold: Some(0.5),
            min_duration_on: Some(0.0),
            min_duration_off: Some(0.0),
            provider: Some("cpu".into()),
            debug: false,
        };
        let mut engine = Diarize::new(&self.paths.separator_model, &self.paths.embedding_model, config)
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
            Err(e) => return Err(Error::Diarization(format!("Speaker diarization failed: {e}"))),
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
        let speakers: std::collections::BTreeSet<&str> = turns.iter().map(|t| t.speaker.as_str()).collect();
        reporter.log(&format!(
            "[Diarize] Found {} speaker(s) across {} turn(s)",
            speakers.len(),
            turns.len()
        ));
        log::info!("Diarization complete: {} speakers, {} turns", speakers.len(), turns.len());
        Ok(turns)
    }
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
    let minutes = (seconds / 60.0).round() as i64;
    if minutes <= 1 {
        return "about 1 minute".into();
    }
    if minutes < 60 {
        return format!("about {minutes} minutes");
    }
    let mut hours = minutes / 60;
    let mut rest = minutes % 60;
    rest = 5 * ((rest as f64 / 5.0).round() as i64);
    if rest == 60 {
        hours += 1;
        rest = 0;
    }
    let hours_text = if hours == 1 { "1 hour".to_string() } else { format!("{hours} hours") };
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
