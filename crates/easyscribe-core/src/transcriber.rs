//! File transcription orchestration: engine selection, VAD segmentation,
//! diarization, transcript writing. Ported from `src/transcriber.py`.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crate::audio::{load_wav_float32, wav_duration};
use crate::diarizer::{DiarizationEngine, SpeakerNaming, SpeakerPreview};
use crate::ffmpeg::fmt_duration;
use crate::model::{Segment, Status, Turn};
use crate::report::CancelToken;
use crate::sherpa_engine::{self, TranscriptionEngine};
use crate::transcript::{
    assign_speakers, build_diarized_transcript, build_plain_transcript, extract_speaker_clip,
    speech_seconds, write_transcript,
};
use crate::vad::pad_and_clamp;
use crate::{Error, Paths, Reporter, Result, whispercpp};

#[derive(Debug, Clone)]
pub struct TranscribeRequest<'a> {
    pub audio_path: &'a Path,
    pub output_path: &'a Path,
    pub add_timestamps: bool,
    pub diarize: bool,
}

/// Transcribe `audio_path` into `output_path`; returns the timed lines.
pub fn transcribe(
    paths: &Paths,
    engine: &TranscriptionEngine,
    request: &TranscribeRequest<'_>,
    cancel: &CancelToken,
    reporter: &Arc<dyn Reporter>,
    naming: Option<&dyn SpeakerNaming>,
) -> Result<Vec<Segment>> {
    cancel.check()?;
    reporter.status(Status::Transcribing);
    log::info!(
        "Transcribing: {} → {}",
        request.audio_path.display(),
        request.output_path.display()
    );
    reporter.log(&format!(
        "[Transcribe] Starting: {}",
        file_name(request.audio_path)
    ));
    let duration = wav_duration(request.audio_path)
        .map_err(|e| Error::Transcription(format!("Could not load audio: {e}")))?;
    reporter.log(&format!(
        "[Transcribe] Audio duration: {}",
        fmt_duration(duration)
    ));

    let started = Instant::now();
    let (raw_segments, engine_name) = if whispercpp::is_available(paths) {
        reporter.log(&format!(
            "[Transcribe] engine: whisper.cpp (beam_size={}), timestamps: {}, speakers: {}",
            whispercpp::BEAM_SIZE,
            on_off(request.add_timestamps),
            on_off(request.diarize)
        ));
        let segments = whispercpp::transcribe_file(paths, request.audio_path, cancel, reporter)
            .map_err(|e| match e {
                Error::Cancelled => Error::Cancelled,
                other => Error::Transcription(other.to_string()),
            })?;
        for segment in &segments {
            reporter.log(&segment.text);
        }
        (segments, "whisper.cpp")
    } else {
        engine.ensure_loaded(reporter.as_ref())?;
        let (samples, sample_rate) = load_wav_float32(request.audio_path)
            .map_err(|e| Error::Transcription(format!("Could not load audio: {e}")))?;
        reporter.log(&format!(
            "[Transcribe] engine: sherpa-onnx (CPU), timestamps: {}, speakers: {}",
            on_off(request.add_timestamps),
            on_off(request.diarize)
        ));
        reporter.log("[Transcribe] Detecting speech segments (VAD)…");
        let mut detector = engine.file_vad()?;
        let bounds = sherpa_engine::file_segments(&mut detector, &samples, cancel)?;
        let padded = pad_and_clamp(&bounds, samples.len());
        let num_chunks = padded.len();
        reporter.log(&format!(
            "[Transcribe] Processing {num_chunks} speech segment(s)…"
        ));
        let mut segments = Vec::new();
        for (index, (start, end)) in padded.iter().copied().enumerate() {
            if cancel.is_cancelled() {
                return Err(Error::Cancelled);
            }
            let seg_start = start as f64 / sample_rate as f64;
            let seg_end = end as f64 / sample_rate as f64;
            let text = engine.recognize(&samples[start..end])?;
            if !text.is_empty() {
                reporter.log(&text);
                segments.push(Segment::new(seg_start, seg_end, text));
            }
            reporter.progress((index + 1) as f64 / num_chunks as f64);
            let elapsed = started.elapsed().as_secs_f64();
            let speed = if elapsed > 0.0 {
                seg_end / elapsed
            } else {
                0.0
            };
            reporter.log(&format!(
                "[Progress] segment {}/{} — {} / {} — {speed:.1}x realtime",
                index + 1,
                num_chunks,
                fmt_duration(seg_end),
                fmt_duration(duration)
            ));
        }
        (segments, "sherpa-onnx")
    };

    let word_count: usize = raw_segments
        .iter()
        .map(|s| s.text.split_whitespace().count())
        .sum();
    reporter.log(&format!(
        "[Transcribe] Complete — {} segments, {} words in {}",
        raw_segments.len(),
        word_count,
        fmt_duration(started.elapsed().as_secs_f64())
    ));
    let _ = engine_name;

    let transcript_text = if request.diarize && !raw_segments.is_empty() {
        match speaker_transcript(
            paths,
            request.audio_path,
            &raw_segments,
            request.add_timestamps,
            cancel,
            reporter,
            naming,
        ) {
            Ok(text) => text,
            Err(Error::Cancelled) => return Err(Error::Cancelled),
            Err(e) => {
                log::warn!("Diarization failed, falling back to plain: {e}");
                reporter.log(&format!(
                    "[Diarize] Warning: {e}. Writing the transcript without speaker names."
                ));
                build_plain_transcript(&raw_segments, request.add_timestamps)
            }
        }
    } else {
        build_plain_transcript(&raw_segments, request.add_timestamps)
    };

    reporter.status(Status::WritingTranscript);
    reporter.log(&format!(
        "[Output] Writing transcript to: {}",
        request.output_path.display()
    ));
    write_transcript(request.output_path, &transcript_text)?;
    let size_kb = std::fs::metadata(request.output_path)
        .map(|m| m.len() / 1024)
        .unwrap_or(0);
    reporter.progress(1.0);
    reporter.log(&format!(
        "[Done] Saved: {}  ({} KB)",
        request.output_path.display(),
        size_kb
    ));
    Ok(raw_segments)
}

/// Find speakers and build the speaker-labelled text. If `naming` is provided,
/// it is called after diarization with clips and shares.
pub fn speaker_transcript(
    paths: &Paths,
    audio_path: &Path,
    raw_segments: &[Segment],
    add_timestamps: bool,
    cancel: &CancelToken,
    reporter: &Arc<dyn Reporter>,
    naming: Option<&dyn SpeakerNaming>,
) -> Result<String> {
    reporter.status(Status::IdentifyingSpeakers);
    let started = Instant::now();
    let engine = DiarizationEngine::new(paths.clone());
    let turns: Vec<Turn> = engine.diarize(audio_path, cancel, reporter.clone())?;
    engine.save_measured_speed(audio_path, raw_segments, started.elapsed().as_secs_f64());
    let assigned = assign_speakers(raw_segments, &turns);

    let mut speaker_map: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    let mut talk: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    for turn in &turns {
        if !speaker_map.contains_key(&turn.speaker) {
            speaker_map.insert(
                turn.speaker.clone(),
                format!("Speaker {}", speaker_map.len() + 1),
            );
        }
        *talk.entry(turn.speaker.clone()).or_insert(0.0) += (turn.end - turn.start).max(0.0);
    }

    if let Some(naming) = naming
        && !speaker_map.is_empty()
    {
        reporter.status(Status::NamingSpeakers);
        let total: f64 = talk.values().sum::<f64>().max(f64::MIN_POSITIVE);
        let shares = speaker_map
            .keys()
            .map(|speaker| {
                (
                    speaker.clone(),
                    talk.get(speaker).copied().unwrap_or(0.0) / total,
                )
            })
            .collect::<std::collections::HashMap<_, _>>();
        let mut clips = std::collections::HashMap::new();
        for speaker in speaker_map.keys() {
            let clip = paths.temp_dir.join(format!("spk_clip_{speaker}.wav"));
            if extract_speaker_clip(audio_path, &turns, speaker, &clip, 5.0) {
                clips.insert(speaker.clone(), clip);
            }
        }
        let mut preview = SpeakerPreview {
            speaker_map: speaker_map.clone(),
            clips: clips.clone(),
            shares,
        };
        let result = naming.ask(&mut preview);
        speaker_map = preview.speaker_map;
        for clip in clips.values() {
            let _ = std::fs::remove_file(clip);
        }
        cancel.check()?;
        result?;
    }

    Ok(build_diarized_transcript(
        &assigned,
        &speaker_map,
        add_timestamps,
    ))
}

/// Add speakers to an existing transcript and write it to `output_path`.
pub fn add_speakers_to_file(
    paths: &Paths,
    audio_path: &Path,
    raw_segments: &[Segment],
    output_path: &Path,
    cancel: &CancelToken,
    reporter: &Arc<dyn Reporter>,
    naming: Option<&dyn SpeakerNaming>,
) -> Result<()> {
    if raw_segments.is_empty() {
        return Err(Error::Transcription(
            "The transcript has no text to add speakers to.".into(),
        ));
    }
    let text = speaker_transcript(
        paths,
        audio_path,
        raw_segments,
        true,
        cancel,
        reporter,
        naming,
    )?;
    reporter.status(Status::WritingTranscript);
    reporter.log(&format!(
        "[Output] Writing transcript to: {}",
        output_path.display()
    ));
    write_transcript(output_path, &text)?;
    reporter.progress(1.0);
    reporter.log(&format!("[Done] Saved: {}", output_path.display()));
    Ok(())
}

/// Seconds covered by the raw transcript lines (used by the time estimate).
pub fn transcript_speech_seconds(segments: &[Segment]) -> f64 {
    speech_seconds(segments)
}

fn on_off(value: bool) -> &'static str {
    if value { "on" } else { "off" }
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("?")
        .to_owned()
}
