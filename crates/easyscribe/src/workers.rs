//! Background workers. They own every long-running operation and communicate
//! with the UI only through [`Event`] values.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::Sender;
use std::time::{Duration, Instant};

use easyscribe_core::diarizer::SpeakerNaming;
use easyscribe_core::live::{LiveEvent, LiveTranscriber};
use easyscribe_core::mic::MicRecorder;
use easyscribe_core::{
    CancelToken, Error, Paths, Reporter, Segment, TranscriptionEngine, audio, ffmpeg, outputs,
    transcriber,
};
use rune::Waker;

use crate::app::{Event, FileState, RecordMode, UiReporter};

#[derive(Clone, Debug)]
pub struct BatchRequest {
    pub files: Vec<PathBuf>,
    pub output_folder: Option<PathBuf>,
    pub timestamps: bool,
    pub diarize: bool,
    pub batch_recording: Option<PathBuf>,
    pub overrides: HashMap<PathBuf, PathBuf>,
}

#[derive(Debug)]
pub struct BatchDone {
    pub total: usize,
    pub saved: Vec<PathBuf>,
    pub failed: Vec<PathBuf>,
    pub cancelled: bool,
    pub fatal: Option<String>,
    pub recording_segments: Option<Vec<Segment>>,
    pub output_folder: Option<PathBuf>,
}

#[derive(Clone, Debug)]
pub struct RecordRequest {
    pub mic: Option<String>,
    pub mode: RecordMode,
    pub dest_dir: PathBuf,
}

#[derive(Debug)]
pub struct RecordDone {
    pub audio: Option<PathBuf>,
    pub transcript: Option<PathBuf>,
    pub error: Option<String>,
    pub mic_error: bool,
    pub model_error: bool,
}

#[derive(Debug)]
pub struct FindDone {
    pub output: Option<PathBuf>,
    pub cancelled: bool,
    pub error: Option<String>,
    pub took: f64,
}

fn reporter(tx: Sender<Event>, waker: Option<Waker>) -> Arc<dyn Reporter> {
    Arc::new(UiReporter { tx, waker })
}

pub fn spawn_batch(
    paths: Paths,
    engine: Arc<TranscriptionEngine>,
    request: BatchRequest,
    cancel: CancelToken,
    tx: Sender<Event>,
    waker: Option<Waker>,
) {
    let reporter = reporter(tx.clone(), waker.clone());
    let error_tx = tx.clone();
    let error_waker = waker.clone();
    let error_files = request.files.clone();
    let error_output = request.output_folder.clone();
    let spawn = std::thread::Builder::new()
        .name("TranscriptionWorker".into())
        .spawn(move || {
            crate::platform::lower_thread_priority();
            let total = request.files.len();
            let mut saved = Vec::new();
            let mut failed = Vec::new();
            let mut cancelled = false;
            let mut fatal: Option<String> = None;
            let mut recording_segments = None;
            let mut output_folder: Option<PathBuf> = None;

            for (index, input) in request.files.iter().enumerate() {
                if cancel.is_cancelled() {
                    cancelled = true;
                    break;
                }
                let _ = tx.send(Event::FileState(input.clone(), FileState::Working));
                let label = if total > 1 {
                    format!(
                        "File {} of {}: {}",
                        index + 1,
                        total,
                        outputs::shorten(
                            input.file_name().and_then(|n| n.to_str()).unwrap_or("?"),
                            70
                        )
                    )
                } else {
                    outputs::shorten(
                        input.file_name().and_then(|n| n.to_str()).unwrap_or("?"),
                        80,
                    )
                };
                let _ = tx.send(Event::BatchLabel(label));
                reporter.log("");
                reporter.log(&"─".repeat(50));
                reporter.log(&format!(
                    "[File {}/{}] {}",
                    index + 1,
                    total,
                    input.file_name().and_then(|n| n.to_str()).unwrap_or("?")
                ));

                reporter.status(easyscribe_core::Status::ExtractingAudio);
                reporter.progress(0.0);
                let temp = match ffmpeg::extract_audio(&paths, input, &cancel, reporter.as_ref()) {
                    Ok(temp) => temp,
                    Err(Error::Cancelled) => {
                        reporter.log("[Cancelled] Operation stopped by user");
                        let _ = tx.send(Event::FileState(input.clone(), FileState::Stopped));
                        cancelled = true;
                        break;
                    }
                    Err(error @ (Error::FfmpegNotFound(_) | Error::ModelNotFound(_))) => {
                        reporter.log(&format!("[Fatal] {error}"));
                        let _ = tx.send(Event::FileState(input.clone(), FileState::Failed));
                        failed.push(input.clone());
                        fatal = Some(error.to_string());
                        break;
                    }
                    Err(error) => {
                        reporter.log(&format!(
                            "[Error] {}: {error}",
                            input.file_name().and_then(|n| n.to_str()).unwrap_or("?")
                        ));
                        let _ = tx.send(Event::FileState(input.clone(), FileState::Failed));
                        failed.push(input.clone());
                        continue;
                    }
                };

                let output = request.overrides.get(input).cloned().unwrap_or_else(|| {
                    if request.batch_recording.is_some() {
                        outputs::best_quality_output(input)
                    } else {
                        outputs::transcript_output(input, request.output_folder.as_deref())
                    }
                });
                if let Some(parent) = output.parent() {
                    output_folder = Some(parent.to_path_buf());
                }
                let transcribe = transcriber::TranscribeRequest {
                    audio_path: &temp,
                    output_path: &output,
                    add_timestamps: request.timestamps,
                    diarize: request.diarize,
                };
                let naming = crate::app::UiNaming {
                    tx: tx.clone(),
                    waker: waker.clone(),
                    cancel: cancel.clone(),
                };
                let result = transcriber::transcribe(
                    &paths,
                    &engine,
                    &transcribe,
                    &cancel,
                    &reporter,
                    Some(&naming),
                );
                let _ = std::fs::remove_file(&temp);
                match result {
                    Ok(segments) => {
                        saved.push(output.clone());
                        if request.batch_recording.is_some() {
                            recording_segments = Some(segments);
                        }
                        let _ = tx.send(Event::FileState(input.clone(), FileState::Done));
                    }
                    Err(Error::Cancelled) => {
                        reporter.log("[Cancelled] Operation stopped by user");
                        let _ = tx.send(Event::FileState(input.clone(), FileState::Stopped));
                        cancelled = true;
                        break;
                    }
                    Err(error @ (Error::FfmpegNotFound(_) | Error::ModelNotFound(_))) => {
                        reporter.log(&format!("[Fatal] {error}"));
                        let _ = tx.send(Event::FileState(input.clone(), FileState::Failed));
                        failed.push(input.clone());
                        fatal = Some(error.to_string());
                        break;
                    }
                    Err(
                        error @ (Error::FfmpegExtraction(_)
                        | Error::Transcription(_)
                        | Error::WriteTranscript(_)),
                    ) => {
                        reporter.log(&format!(
                            "[Error] {}: {error}",
                            input.file_name().and_then(|n| n.to_str()).unwrap_or("?")
                        ));
                        let _ = tx.send(Event::FileState(input.clone(), FileState::Failed));
                        failed.push(input.clone());
                    }
                    Err(error) => {
                        reporter.log(&format!(
                            "[Unexpected Error] {}: {error}",
                            input.file_name().and_then(|n| n.to_str()).unwrap_or("?")
                        ));
                        let _ = tx.send(Event::FileState(input.clone(), FileState::Failed));
                        failed.push(input.clone());
                    }
                }
            }
            if !failed.is_empty() && !cancelled && fatal.is_none() && failed.len() < total {
                reporter.log("");
                reporter.log(&format!(
                    "[Summary] Completed with {} error(s) out of {total} file(s)",
                    failed.len()
                ));
            }
            if let Some(waker) = &waker {
                waker();
            }
            let _ = tx.send(Event::BatchDone(BatchDone {
                total,
                saved,
                failed,
                cancelled,
                fatal,
                recording_segments,
                output_folder,
            }));
            if let Some(waker) = waker {
                waker();
            }
        });
    if let Err(error) = spawn {
        let _ = error_tx.send(Event::BatchDone(BatchDone {
            total: error_files.len(),
            saved: Vec::new(),
            failed: error_files,
            cancelled: false,
            fatal: Some(format!("Could not start transcription: {error}")),
            recording_segments: None,
            output_folder: error_output,
        }));
        if let Some(waker) = error_waker {
            waker();
        }
    }
}

pub fn spawn_recording(
    paths: Paths,
    engine: Arc<TranscriptionEngine>,
    request: RecordRequest,
    stop: CancelToken,
    tx: Sender<Event>,
    waker: Option<Waker>,
) {
    let reporter = reporter(tx.clone(), waker.clone());
    let error_tx = tx.clone();
    let error_waker = waker.clone();
    let spawn = std::thread::Builder::new()
        .name("RecordingWorker".into())
        .spawn(move || {
            crate::platform::lower_thread_priority();
            let mut recorder = MicRecorder::new();
            let live_mode = request.mode == RecordMode::Live;
            if live_mode {
                reporter.status(easyscribe_core::Status::LoadingModel);
                if let Err(error) = engine.ensure_loaded(reporter.as_ref()) {
                    let model_error = matches!(error, Error::ModelNotFound(_));
                    finish_error(
                        &tx,
                        &waker,
                        format!(
                            "A program file is missing. Please download EasyScribe again.\n{error}"
                        ),
                        false,
                        model_error,
                    );
                    return;
                }
            }

            if let Err(error) = recorder.start(
                request.mic.as_deref(),
                &paths.recordings_work_dir,
                live_mode,
            ) {
                finish_error(
                    &tx,
                    &waker,
                    format!(
                        "Could not record: {error}\nCheck that a microphone is connected, and that Windows allows apps to use it (Settings, Privacy, Microphone)."
                    ),
                    true,
                    false,
                );
                return;
            }
            let _ = tx.send(Event::RecordStarted);
            reporter.status(easyscribe_core::Status::Recording);
            reporter.log(if live_mode {
                "[Record] Starting live recording…"
            } else {
                "[Record] Starting recording (best quality transcript after the stop)…"
            });

            let mut live = LiveTranscriber::new();
            let mut live_rx = None;
            let live_cancel = CancelToken::new();
            let mut lines: Vec<String> = Vec::new();
            if live_mode
                && let Some(queue) = recorder.vad_queue() {
                    let (live_tx, rx) = std::sync::mpsc::channel();
                    match live.start(engine.clone(), queue, live_cancel.clone(), live_tx) {
                        Ok(()) => {
                            live_rx = Some(rx);
                            reporter.log("[Record] Listening. Select Stop recording when done");
                        }
                        Err(error) => {
                            reporter.log(&format!("[Error] Live transcription failed: {error}"));
                        }
                    }
                }

            let started = Instant::now();
            let mut last_level = Instant::now();
            let mut last_timer = Instant::now();
            while !stop.is_cancelled() {
                if let Some(rx) = &live_rx {
                    while let Ok(event) = rx.try_recv() {
                        match event {
                            LiveEvent::Text(text) => {
                                lines.push(text.clone());
                                let _ = tx.send(Event::LiveText(text));
                            }
                            LiveEvent::Overflow => {
                                let _ = tx.send(Event::LiveOverflow);
                            }
                            LiveEvent::Error(error) => {
                                reporter.log(&format!("[Error] Live transcription: {error}"));
                            }
                        }
                    }
                }
                if last_level.elapsed() >= Duration::from_millis(100) {
                    last_level = Instant::now();
                    let _ = tx.send(Event::RecordTick {
                        seconds: started.elapsed().as_secs_f64(),
                        level: recorder.get_level(),
                    });
                    if let Some(waker) = &waker {
                        waker();
                    }
                }
                if last_timer.elapsed() >= Duration::from_millis(500) {
                    last_timer = Instant::now();
                    let _ = tx.send(Event::RecordTick {
                        seconds: started.elapsed().as_secs_f64(),
                        level: recorder.get_level(),
                    });
                    if let Some(waker) = &waker {
                        waker();
                    }
                }
                std::thread::sleep(Duration::from_millis(20));
            }

            if live_mode {
                reporter.status(easyscribe_core::Status::Finishing);
                live_cancel.cancel();
                live.stop();
                if let Some(rx) = live_rx {
                    while let Ok(event) = rx.try_recv() {
                        if let LiveEvent::Text(text) = event {
                            lines.push(text.clone());
                            let _ = tx.send(Event::LiveText(text));
                        }
                    }
                }
            }
            reporter.status(easyscribe_core::Status::SavingRecording);
            let session_stem = recorder.session_stem();
            let (wav, saved_dir) = match recorder.stop(true, Some(&request.dest_dir)) {
                Ok(Some(wav)) => {
                    let folder = wav
                        .parent()
                        .map(Path::to_path_buf)
                        .unwrap_or_else(|| request.dest_dir.clone());
                    (Some(wav), folder)
                }
                // The WAV could not be written. Keep the live transcript in
                // the recordings work folder, where the raw PCM also lives.
                Ok(None) => (None, paths.recordings_work_dir.clone()),
                Err(error) => {
                    finish_error(
                        &tx,
                        &waker,
                        format!("Could not save the recording: {error}"),
                        false,
                        false,
                    );
                    return;
                }
            };
            if let Some(wav) = &wav {
                if wav.parent() != Some(request.dest_dir.as_path()) {
                    reporter.log(&format!(
                        "[Warning] Could not save to {}. The recording is in {}",
                        request.dest_dir.display(),
                        wav.parent().map(|p| p.display().to_string()).unwrap_or_default()
                    ));
                }
                reporter.log(&format!("[Done] Audio saved: {}", wav.display()));
            }
            let mut transcript = None;
            let mut error: Option<String> = None;
            if live_mode && !lines.is_empty() {
                let target = match &wav {
                    Some(wav) => outputs::transcript_output(wav, None),
                    None => saved_dir.join(format!(
                        "{}.txt",
                        session_stem.as_deref().unwrap_or("recording")
                    )),
                };
                let text = format!("{}\n", lines.join("\n"));
                match std::fs::write(&target, text) {
                    Ok(()) => {
                        transcript = Some(target.clone());
                        reporter.log(&format!("[Done] Transcript saved: {}", target.display()));
                    }
                    Err(write_error) => {
                        error = Some(format!("Could not save the transcript: {write_error}"));
                        reporter.log(&format!("[Error] Could not write transcript: {write_error}"));
                    }
                }
            }
            let done = RecordDone {
                audio: wav,
                transcript,
                error,
                mic_error: false,
                model_error: false,
            };
            let _ = tx.send(Event::RecordDone(done));
            if let Some(waker) = waker {
                waker();
            }
        });
    if let Err(error) = spawn {
        let _ = error_tx.send(Event::RecordDone(RecordDone {
            audio: None,
            transcript: None,
            error: Some(format!("Could not start recording: {error}")),
            mic_error: false,
            model_error: false,
        }));
        if let Some(waker) = error_waker {
            waker();
        }
    }
}

fn finish_error(
    tx: &Sender<Event>,
    waker: &Option<Waker>,
    error: String,
    mic_error: bool,
    model_error: bool,
) {
    let _ = tx.send(Event::RecordDone(RecordDone {
        audio: None,
        transcript: None,
        error: Some(error),
        mic_error,
        model_error,
    }));
    if let Some(waker) = waker {
        waker();
    }
}

#[allow(clippy::too_many_arguments)]
pub fn spawn_find_speakers(
    paths: Paths,
    audio: PathBuf,
    segments: Vec<Segment>,
    output: PathBuf,
    cancel: CancelToken,
    tx: Sender<Event>,
    waker: Option<Waker>,
    naming: Option<Arc<dyn SpeakerNaming>>,
) {
    let reporter = reporter(tx.clone(), waker.clone());
    let error_tx = tx.clone();
    let error_waker = waker.clone();
    let spawn = std::thread::Builder::new()
        .name("SpeakersWorker".into())
        .spawn(move || {
            crate::platform::lower_thread_priority();
            let started = Instant::now();
            let result = transcriber::add_speakers_to_file(
                &paths,
                &audio,
                &segments,
                &output,
                &cancel,
                &reporter,
                naming.as_deref(),
            );
            let took = started.elapsed().as_secs_f64();
            let done = match result {
                Ok(()) => FindDone {
                    output: Some(output),
                    cancelled: false,
                    error: None,
                    took,
                },
                Err(Error::Cancelled) => {
                    reporter.log("[Cancelled] Operation stopped by user");
                    FindDone {
                        output: None,
                        cancelled: true,
                        error: None,
                        took,
                    }
                }
                Err(error) => {
                    reporter.log(&format!("[Error] {error}"));
                    FindDone {
                        output: None,
                        cancelled: false,
                        error: Some(error.to_string()),
                        took,
                    }
                }
            };
            let _ = tx.send(Event::FindDone(done));
            if let Some(waker) = waker {
                waker();
            }
        });
    if let Err(error) = spawn {
        let _ = error_tx.send(Event::FindDone(FindDone {
            output: None,
            cancelled: false,
            error: Some(format!("Could not start finding speakers: {error}")),
            took: 0.0,
        }));
        if let Some(waker) = error_waker {
            waker();
        }
    }
}

#[allow(dead_code)]
fn _duration(path: &std::path::Path) -> Option<f64> {
    audio::wav_duration(path).ok()
}
