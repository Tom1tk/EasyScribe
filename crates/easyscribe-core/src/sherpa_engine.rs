//! sherpa-onnx wrappers: the Whisper ONNX recognizer, Silero VAD, and
//! speaker diarization. These use `sherpa-rs`, which pins sherpa-onnx and
//! downloads/builds its native library.

use std::sync::Mutex;

use sherpa_rs::silero_vad::{SileroVad, SileroVadConfig};
use sherpa_rs::whisper::{WhisperConfig, WhisperRecognizer};

use crate::config::VAD_SAMPLE_RATE;
use crate::report::CancelToken;
use crate::vad::{
    VAD_MAX_SPEECH_SEC, VAD_MIN_SILENCE_SEC, VAD_MIN_SPEECH_SEC, VAD_PAD_SEC, VAD_THRESHOLD,
};
use crate::{Error, Paths, Reporter, Result};

/// Lazy-loading Whisper ONNX recognizer (sherpa-onnx fallback and live mode).
pub struct TranscriptionEngine {
    paths: Paths,
    recognizer: Mutex<Option<WhisperRecognizer>>,
}

impl TranscriptionEngine {
    pub fn new(paths: Paths) -> Self {
        Self {
            paths,
            recognizer: Mutex::new(None),
        }
    }

    pub fn paths(&self) -> &Paths {
        &self.paths
    }

    /// Validate and load the model once. Concurrent callers block on the lock.
    pub fn ensure_loaded(&self, reporter: &dyn Reporter) -> Result<()> {
        let mut guard = self.recognizer.lock().expect("recognizer mutex poisoned");
        if guard.is_some() {
            return Ok(());
        }
        let missing = self.paths.validate_models();
        if !missing.is_empty() {
            return Err(Error::ModelNotFound(format!(
                "Model validation failed:\n{}",
                missing.join("\n")
            )));
        }
        reporter.status(crate::Status::LoadingModel);
        log::info!("Loading model, provider=cpu");
        let config = WhisperConfig {
            encoder: self.paths.whisper_encoder.to_string_lossy().into_owned(),
            decoder: self.paths.whisper_decoder.to_string_lossy().into_owned(),
            tokens: self.paths.whisper_tokens.to_string_lossy().into_owned(),
            language: "en".into(),
            bpe_vocab: None,
            tail_paddings: None,
            provider: Some("cpu".into()),
            num_threads: Some(crate::config::num_threads() as i32),
            debug: false,
        };
        let recognizer = WhisperRecognizer::new(config)
            .map_err(|e| Error::ModelNotFound(format!("Could not load model: {e}")))?;
        *guard = Some(recognizer);
        log::info!("Model loaded: provider=cpu");
        Ok(())
    }

    /// Decode one chunk of 16 kHz mono audio. The engine must be loaded.
    pub fn recognize(&self, samples: &[f32]) -> Result<String> {
        let mut guard = self.recognizer.lock().expect("recognizer mutex poisoned");
        let recognizer = guard
            .as_mut()
            .ok_or_else(|| Error::ModelNotFound("speech model is not loaded".into()))?;
        let result = recognizer.transcribe(VAD_SAMPLE_RATE, samples);
        Ok(result.text.trim().to_owned())
    }

    /// Build a file-mode VAD detector.
    pub fn file_vad(&self) -> Result<SileroVad> {
        build_vad(&self.paths, 60.0)
    }
}

/// Build a Silero VAD detector with the shared tuning.
pub fn build_vad(paths: &Paths, buffer_seconds: f32) -> Result<SileroVad> {
    build_vad_with(paths, buffer_seconds, VAD_THRESHOLD, VAD_MIN_SPEECH_SEC)
}

/// Build a VAD detector with explicit tuning. The pre-whisper speech gate uses
/// a slightly more sensitive threshold so quiet leading words are not trimmed.
pub fn build_vad_with(
    paths: &Paths,
    buffer_seconds: f32,
    threshold: f32,
    min_speech_duration: f32,
) -> Result<SileroVad> {
    if !paths.vad_model.is_file() {
        return Err(Error::ModelNotFound(format!(
            "Missing: {}",
            paths.vad_model.display()
        )));
    }
    let config = SileroVadConfig {
        model: paths.vad_model.to_string_lossy().into_owned(),
        min_silence_duration: VAD_MIN_SILENCE_SEC,
        min_speech_duration,
        max_speech_duration: VAD_MAX_SPEECH_SEC,
        threshold,
        sample_rate: VAD_SAMPLE_RATE,
        window_size: 512,
        provider: Some("cpu".into()),
        num_threads: Some(1),
        debug: false,
    };
    SileroVad::new(config, buffer_seconds)
        .map_err(|e| Error::Transcription(format!("Could not create VAD: {e}")))
}

/// Feed the whole file through the VAD in 512-sample windows and return
/// `(start_sample, end_sample)` for every speech segment.
pub fn file_segments(
    detector: &mut SileroVad,
    samples: &[f32],
    cancel: &CancelToken,
) -> Result<Vec<(usize, usize)>> {
    let window = crate::config::VAD_CHUNK_SAMPLES;
    let mut bounds = Vec::new();
    let mut pos = 0;
    while pos < samples.len() {
        cancel.check()?;
        let end = (pos + window).min(samples.len());
        let mut chunk = samples[pos..end].to_vec();
        if chunk.len() < window {
            chunk.resize(window, 0.0);
        }
        detector.accept_waveform(chunk);
        pos = end;
        drain_vad(detector, &mut bounds);
    }
    detector.flush();
    drain_vad(detector, &mut bounds);
    Ok(bounds)
}

fn drain_vad(detector: &mut SileroVad, bounds: &mut Vec<(usize, usize)>) {
    while !detector.is_empty() {
        let segment = detector.front();
        let start = segment.start.max(0) as usize;
        let end = start + segment.samples.len();
        bounds.push((start, end));
        detector.pop();
    }
}

/// Padding is applied separately, in `vad::pad_and_clamp`.
#[allow(dead_code)]
pub const PAD_SECONDS: f32 = VAD_PAD_SEC;
