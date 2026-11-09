//! Application state, event flow and the user-action handlers.
//!
//! The Rune view is a pure function of this state. Background workers send
//! [`Event`] values over an `mpsc` channel; `poll` applies them on the UI thread.
//! No worker ever touches the view or the state directly.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, Sender, SyncSender};
use std::time::Duration;

use easyscribe_core::diarizer::{DiarizationEngine, SpeakerNaming, SpeakerPreview};
use easyscribe_core::{
    CancelToken, Error, Paths, Reporter, Segment, Status, TranscriptionEngine, audio, ffmpeg,
    logger, mic, outputs, payload, recovery, rename, settings::Settings, transcriber,
};
use rune::{TextEdit, Waker};

use crate::dialogs;
use crate::theme::{Theme, visible_rows};
use crate::workers;

pub const LISTENING: &str = "Listening. The words appear here as people speak.";
pub const IDLE_LIVE: &str = "Select Start recording. The words appear here as people speak.";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Files,
    Record,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UiState {
    Idle,
    Running,
    Recording,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileState {
    Waiting,
    Working,
    Done,
    Failed,
    Stopped,
}

impl FileState {
    pub fn pill(self, t: &Theme) -> (&'static str, rune::Color, rune::Color) {
        match self {
            Self::Waiting => ("Waiting", t.muted, t.surface_alt),
            Self::Working => ("Working", t.teal_ink, t.teal_soft),
            Self::Done => ("Done", t.green_ink, t.green_soft),
            Self::Failed => ("Failed", t.crimson, t.crimson_soft),
            Self::Stopped => ("Stopped", t.muted, t.surface_alt),
        }
    }
}

#[derive(Clone, Debug)]
pub struct FileEntry {
    pub path: PathBuf,
    pub state: FileState,
    /// Snapshot-only overrides so the visual QA scenes can mirror the screenshots.
    pub display_name: Option<String>,
    pub display_size: Option<u64>,
}

impl FileEntry {
    pub fn new(path: PathBuf) -> Self {
        // Stat once when the file is added, not on every render: progress
        // updates rebuild the view many times per second.
        let display_size = std::fs::metadata(&path).ok().map(|metadata| metadata.len());
        Self {
            path,
            state: FileState::Waiting,
            display_name: None,
            display_size,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordMode {
    Best,
    Live,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tone {
    Muted,
    Ink,
    Teal,
    Amber,
    Coral,
    Green,
    Crimson,
}

impl Tone {
    pub fn color(self, t: &Theme) -> rune::Color {
        match self {
            Self::Muted => t.muted,
            Self::Ink => t.ink,
            Self::Teal => t.teal_ink,
            Self::Amber => t.amber_ink,
            Self::Coral => t.coral_ink,
            Self::Green => t.green_ink,
            Self::Crimson => t.crimson,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BarMode {
    Idle,
    Determinate,
    Busy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MessageKind {
    Info,
    Error,
}

#[derive(Clone, Debug)]
pub struct Activity {
    pub status: Status,
    pub title: String,
    pub tone: Tone,
    pub detail: String,
    pub bar: BarMode,
    pub progress: f32,
    pub bar_color: rune::Color,
    pub message: Option<(String, MessageKind)>,
    pub stop_visible: bool,
    pub stop_enabled: bool,
    pub open_transcript: bool,
    pub open_folder: bool,
    pub try_again: bool,
    pub best_row: bool,
}

impl Activity {
    pub fn new(t: &Theme) -> Self {
        Self {
            status: Status::Ready,
            title: "Ready".into(),
            tone: Tone::Muted,
            detail: String::new(),
            bar: BarMode::Idle,
            progress: 0.0,
            bar_color: t.teal,
            message: None,
            stop_visible: false,
            stop_enabled: false,
            open_transcript: false,
            open_folder: false,
            try_again: false,
            best_row: false,
        }
    }

    pub fn apply_status(&mut self, status: Status, t: &Theme, running: bool) {
        self.status = status;
        let (title, tone, bar, color) = match status {
            Status::Ready => ("Ready", Tone::Muted, BarMode::Idle, t.teal),
            Status::LoadingModel => ("Getting ready", Tone::Ink, BarMode::Busy, t.teal),
            Status::ExtractingAudio => ("Reading the audio", Tone::Ink, BarMode::Busy, t.teal),
            Status::Transcribing => ("Transcribing", Tone::Teal, BarMode::Determinate, t.teal),
            Status::SlowFallback => (
                "Transcribing (slow fallback)",
                Tone::Amber,
                BarMode::Determinate,
                t.amber,
            ),
            Status::IdentifyingSpeakers => {
                ("Finding the speakers", Tone::Amber, BarMode::Busy, t.amber)
            }
            Status::NamingSpeakers => (
                "Waiting for speaker names",
                Tone::Amber,
                BarMode::Idle,
                t.amber,
            ),
            Status::WritingTranscript => (
                "Saving the transcript",
                Tone::Teal,
                BarMode::Determinate,
                t.teal,
            ),
            Status::Recording => ("Recording", Tone::Coral, BarMode::Busy, t.coral),
            Status::Finishing => (
                "Finishing the last words",
                Tone::Coral,
                BarMode::Busy,
                t.coral,
            ),
            Status::SavingRecording => {
                ("Saving the recording", Tone::Coral, BarMode::Busy, t.coral)
            }
            Status::Cancelling => ("Stopping…", Tone::Muted, BarMode::Busy, t.faint),
            Status::Done => ("Done", Tone::Green, BarMode::Idle, t.green),
            Status::Cancelled => ("Stopped", Tone::Muted, BarMode::Idle, t.faint),
            Status::Failed => (
                "Something went wrong",
                Tone::Crimson,
                BarMode::Idle,
                t.crimson,
            ),
        };
        self.title = title.into();
        self.tone = tone;
        self.bar = bar;
        self.bar_color = color;
        if matches!(status, Status::Transcribing | Status::WritingTranscript) {
            self.progress = 0.0;
        }
        self.message = match status {
            Status::SlowFallback => Some((
                "The bundled whisper-cli could not start, so EasyScribe is using its built-in engine. This can take many times longer.".into(),
                MessageKind::Info,
            )),
            Status::LoadingModel if running => Some((
                "Loading the speech model. This can take a minute.".into(),
                MessageKind::Info,
            )),
            Status::IdentifyingSpeakers if running => Some((
                "This step can take a while for long recordings.".into(),
                MessageKind::Info,
            )),
            Status::NamingSpeakers if running => {
                Some(("Type each speaker's name below.".into(), MessageKind::Info))
            }
            _ => None,
        };
        self.stop_visible = running;
        self.stop_enabled = running;
        self.open_transcript = false;
        self.open_folder = false;
        self.try_again = false;
        self.best_row = false;
    }
}

#[derive(Clone, Debug)]
pub struct FindCard {
    pub estimate: String,
    pub basis: String,
}

#[derive(Clone, Debug)]
pub struct MicChoice {
    pub label: String,
    pub device: Option<String>,
}

impl MicChoice {
    pub fn default_device() -> Self {
        Self {
            label: "Default microphone".into(),
            device: None,
        }
    }
}

pub struct NamingRow {
    pub raw: String,
    pub default_name: String,
    pub share: Option<f64>,
    pub clip: Option<PathBuf>,
    pub edit: TextEdit,
}

pub struct NamingState {
    pub rows: Vec<NamingRow>,
    pub reply: Option<SyncSender<NamingReply>>,
}

#[derive(Debug)]
pub enum NamingReply {
    Save(HashMap<String, String>),
    Skip,
    Cancel,
}

#[derive(Debug)]
pub enum Event {
    Status(Status),
    Progress(f64),
    Log(String),
    BatchLabel(String),
    FileState(PathBuf, FileState),
    BatchDone(workers::BatchDone),
    RecordStarted,
    RecordTick { seconds: f64, level: f32 },
    RecordDone(workers::RecordDone),
    LiveText(String),
    LiveOverflow,
    AskNames(SpeakerPreview, SyncSender<NamingReply>),
    FindDone(workers::FindDone),
    Preparing { stage: String, progress: f32 },
    Prepared(payload::Prepared),
    PrepareFailed(String),
}

/// First-launch progress, shown in place of the app until the runtime cache is
/// complete. Cached launches never see it.
#[derive(Clone, Debug)]
pub struct PreparingState {
    pub stage: String,
    pub progress: f32,
    pub error: Option<String>,
}

pub struct AppState {
    pub paths: Paths,
    pub theme: Theme,
    pub window_height: f32,
    pub mode: Mode,
    pub ui: UiState,
    pub files: Vec<FileEntry>,
    pub output_folder: Option<PathBuf>,
    pub timestamps: bool,
    pub name_speakers: bool,
    pub diarization_available: bool,
    pub drop_note: Option<String>,
    pub action_hint: Option<(String, bool)>,
    pub files_over: bool,

    pub mic_choices: Vec<MicChoice>,
    pub selected_mic: usize,
    pub mic_menu_open: bool,
    pub record_mode: RecordMode,
    pub recordings_dir: Option<PathBuf>,
    pub selected_diarize_models: bool,

    pub timer_seconds: f64,
    pub level: f32,
    pub record_saving: bool,
    pub live_text: String,
    pub live_has_text: bool,

    pub activity: Option<Activity>,
    pub log: Vec<String>,
    pub details_open: bool,
    pub privacy_open: bool,

    pub last_output_folder: Option<PathBuf>,
    pub last_transcript: Option<PathBuf>,
    pub last_recording: Option<PathBuf>,
    pub last_segments: Vec<Segment>,
    pub batch_recording: Option<PathBuf>,
    pub last_batch_was_recording: bool,

    pub rename_visible: bool,
    pub rename_edit: TextEdit,
    pub rename_note: String,
    pub rename_note_tone: Tone,
    pub rename_targets: Vec<PathBuf>,
    pub rename_old_stem: String,

    pub naming: Option<NamingState>,
    pub find_card: Option<FindCard>,

    pub visible_rows: usize,
    pub tx: Sender<Event>,
    pub rx: Receiver<Event>,
    pub waker: Option<Waker>,
    pub cancel: Option<CancelToken>,
    pub record_stop: Option<CancelToken>,
    pub engine: Arc<TranscriptionEngine>,
    pub settings: Settings,
    pub preparing: Option<PreparingState>,
}

pub struct Init {
    pub paths: Paths,
    pub window_height: f32,
    /// Resolve the single-exe payload in `setup`, showing a splash meanwhile.
    pub preparing: bool,
}

impl AppState {
    pub fn new(init: Init) -> Self {
        let settings = Settings::load(init.paths.settings_file.clone());
        let recordings_dir = settings.load_recordings_dir();
        let mut mic_choices = vec![MicChoice::default_device()];
        let mut seen = HashSet::new();
        for device in mic::list_input_devices() {
            let label = outputs::shorten(device.name.trim(), 48);
            if label.is_empty() || !seen.insert(label.clone()) {
                continue;
            }
            mic_choices.push(MicChoice {
                label,
                device: Some(device.name),
            });
        }
        let (tx, rx) = mpsc::channel();
        let engine = Arc::new(TranscriptionEngine::new(init.paths.clone()));
        let theme = Theme::default();
        Self {
            diarization_available: init.paths.diarization_available(),
            selected_diarize_models: init.paths.diarization_available(),
            visible_rows: visible_rows(init.window_height),
            paths: init.paths,
            theme,
            window_height: init.window_height,
            mode: Mode::Files,
            ui: UiState::Idle,
            files: Vec::new(),
            output_folder: None,
            timestamps: false,
            name_speakers: false,
            drop_note: None,
            action_hint: None,
            files_over: false,
            mic_choices,
            selected_mic: 0,
            mic_menu_open: false,
            record_mode: RecordMode::Best,
            recordings_dir,
            timer_seconds: 0.0,
            level: 0.0,
            record_saving: false,
            live_text: IDLE_LIVE.into(),
            live_has_text: false,
            activity: None,
            log: Vec::new(),
            details_open: false,
            privacy_open: false,
            last_output_folder: None,
            last_transcript: None,
            last_recording: None,
            last_segments: Vec::new(),
            batch_recording: None,
            last_batch_was_recording: false,
            rename_visible: false,
            rename_edit: TextEdit::new(""),
            rename_note: String::new(),
            rename_note_tone: Tone::Muted,
            rename_targets: Vec::new(),
            rename_old_stem: String::new(),
            naming: None,
            find_card: None,
            tx,
            rx,
            waker: None,
            cancel: None,
            record_stop: None,
            engine,
            settings,
            preparing: if init.preparing {
                Some(PreparingState {
                    stage: "Preparing EasyScribe".into(),
                    progress: 0.0,
                    error: None,
                })
            } else {
                None
            },
        }
    }

    pub fn set_waker(&mut self, waker: Waker) {
        self.waker = Some(waker);
    }

    /// Unpack and verify the single-exe payload on a worker thread. Progress
    /// updates wake the splash; completion swaps in the real paths.
    pub fn begin_prepare(&mut self) {
        if self.preparing.is_none() {
            return;
        }
        if let Some(preparing) = &mut self.preparing {
            preparing.error = None;
            preparing.progress = 0.0;
            preparing.stage = "Preparing EasyScribe".into();
        }
        let tx = self.tx.clone();
        let waker = self.waker.clone();
        std::thread::spawn(move || {
            let mut last = -1.0_f32;
            let result = payload::prepare_with_progress(|stage, progress| {
                if progress - last >= 0.003 || progress >= 1.0 {
                    last = progress;
                    let _ = tx.send(Event::Preparing {
                        stage: stage.to_owned(),
                        progress,
                    });
                    if let Some(wake) = &waker {
                        wake();
                    }
                }
            });
            let _ = match result {
                Ok(prepared) => tx.send(Event::Prepared(prepared)),
                Err(error) => tx.send(Event::PrepareFailed(error.to_string())),
            };
            if let Some(wake) = &waker {
                wake();
            }
        });
    }

    /// Swap the provisional paths for the extracted runtime, then run the
    /// checks the Python app ran between config and the GUI.
    fn adopt_prepared(&mut self, prepared: payload::Prepared) {
        let paths = match Paths::for_runtime(prepared.runtime_dir, prepared.data_dir) {
            Ok(paths) => paths,
            Err(error) => {
                self.fail_preparing(error.to_string());
                return;
            }
        };

        startup(&paths);
        recover_before_ui(&paths);
        if let Some(message) = startup_error(&paths) {
            dialogs::error("EasyScribe cannot start", &message);
            std::process::exit(1);
        }
        if let Some(problem) = crate::platform::processor_problem() {
            dialogs::error("EasyScribe cannot start", &problem);
            std::process::exit(1);
        }

        self.paths = paths;
        self.engine = Arc::new(TranscriptionEngine::new(self.paths.clone()));
        self.settings = Settings::load(self.paths.settings_file.clone());
        self.recordings_dir = self.settings.load_recordings_dir();
        self.diarization_available = self.paths.diarization_available();
        self.selected_diarize_models = self.diarization_available;
        self.mic_menu_open = false;
        self.selected_mic = 0;
        self.refresh_microphones();
        self.preparing = None;
    }

    fn refresh_microphones(&mut self) {
        let mut mic_choices = vec![MicChoice::default_device()];
        let mut seen = HashSet::new();
        for device in mic::list_input_devices() {
            let label = outputs::shorten(device.name.trim(), 48);
            if label.is_empty() || !seen.insert(label.clone()) {
                continue;
            }
            mic_choices.push(MicChoice {
                label,
                device: Some(device.name),
            });
        }
        self.mic_choices = mic_choices;
    }

    fn fail_preparing(&mut self, message: String) {
        if let Some(preparing) = &mut self.preparing {
            preparing.stage = "EasyScribe could not prepare its files".into();
            preparing.error = Some(message);
        }
    }

    pub fn idle(&self) -> bool {
        self.ui == UiState::Idle
    }

    pub fn effective_recordings_dir(&self) -> PathBuf {
        self.recordings_dir
            .clone()
            .unwrap_or_else(|| self.paths.default_recordings_dir.clone())
    }

    pub fn selected_mic_device(&self) -> Option<String> {
        self.mic_choices
            .get(self.selected_mic)
            .and_then(|m| m.device.clone())
    }

    pub fn poll(&mut self) -> bool {
        let mut changed = false;
        while let Ok(event) = self.rx.try_recv() {
            changed = true;
            self.apply(event);
        }
        changed
    }

    pub fn apply(&mut self, event: Event) {
        let t = self.theme.clone();
        match event {
            Event::Status(status) => {
                if let Some(activity) = &mut self.activity {
                    let running = self.ui == UiState::Running;
                    activity.apply_status(status, &t, running);
                }
            }
            Event::Progress(value) => {
                if let Some(activity) = &mut self.activity {
                    activity.progress = value.clamp(0.0, 1.0) as f32;
                    if activity.bar != BarMode::Idle {
                        activity.bar = BarMode::Determinate;
                    }
                }
            }
            Event::Log(line) => self.log.push(line),
            Event::BatchLabel(label) => {
                if let Some(activity) = &mut self.activity {
                    activity.detail = label;
                }
            }
            Event::FileState(path, state) => {
                if let Some(file) = self.files.iter_mut().find(|f| f.path == path) {
                    file.state = state;
                }
            }
            Event::BatchDone(done) => self.finish_batch(done),
            Event::Preparing { stage, progress } => {
                if let Some(preparing) = &mut self.preparing {
                    preparing.stage = stage;
                    preparing.progress = progress.clamp(0.0, 1.0);
                }
            }
            Event::Prepared(prepared) => self.adopt_prepared(prepared),
            Event::PrepareFailed(error) => self.fail_preparing(error),
            Event::RecordStarted => {
                self.ui = UiState::Recording;
                self.record_saving = false;
                self.timer_seconds = 0.0;
                if let Some(activity) = &mut self.activity {
                    activity.apply_status(Status::Recording, &t, false);
                }
            }
            Event::RecordTick { seconds, level } => {
                self.timer_seconds = seconds;
                self.level = level;
            }
            Event::RecordDone(done) => self.finish_recording(done),
            Event::LiveText(text) => {
                self.log.push(text.trim().to_owned());
                if !self.live_has_text {
                    self.live_has_text = true;
                    self.live_text.clear();
                }
                if !self.live_text.is_empty() {
                    self.live_text.push('\n');
                }
                self.live_text.push_str(text.trim());
            }
            Event::LiveOverflow => {
                self.log.push("[Warning] transcription falling behind. Some audio was skipped in the live view; the full recording is still saved".into());
            }
            Event::AskNames(preview, reply) => self.show_naming(preview, reply),
            Event::FindDone(done) => self.finish_find(done),
        }
    }

    fn finish_batch(&mut self, done: workers::BatchDone) {
        let t = self.theme.clone();
        self.ui = UiState::Idle;
        self.cancel = None;
        self.last_output_folder = done.output_folder.clone();
        let saved_count = done.saved.len();
        let total = done.total.max(1);
        let failed_count = done.failed.len();
        let recording_retry = self
            .last_batch_was_recording
            .then_some(self.batch_recording.as_ref())
            .flatten()
            .is_some_and(|audio| audio.is_file());
        self.last_transcript = if saved_count == 1 {
            done.saved.first().cloned()
        } else {
            None
        };
        if let Some(segments) = done.recording_segments {
            self.last_segments = segments;
        }
        if self.batch_recording.is_some() {
            self.last_batch_was_recording = true;
        }
        let mut activity = self.activity.take().unwrap_or_else(|| Activity::new(&t));
        activity.stop_visible = false;
        activity.stop_enabled = false;
        activity.best_row = false;
        activity.bar = BarMode::Idle;
        activity.progress = 0.0;
        activity.open_transcript = false;
        activity.open_folder = false;
        activity.try_again = false;
        activity.message = None;

        if let Some(fatal) = done.fatal {
            activity.title = "EasyScribe could not continue".into();
            activity.tone = Tone::Crimson;
            activity.detail = "A program file is missing. Please download EasyScribe again.".into();
            activity.message = Some((fatal, MessageKind::Error));
            activity.open_transcript = saved_count == 1;
            activity.open_folder = saved_count > 0 || recording_retry;
            activity.try_again = false;
            if saved_count > 0 {
                // The fatal error is about the remaining files; work that
                // reached disk is still a successful result.
                activity.bar_color = t.green;
                activity.bar = BarMode::Determinate;
                activity.progress = 1.0;
            } else {
                activity.bar_color = t.crimson;
            }
        } else if done.cancelled {
            activity.title = "Stopped".into();
            activity.tone = Tone::Muted;
            activity.bar_color = if saved_count > 0 { t.green } else { t.faint };
            if saved_count > 0 {
                activity.bar = BarMode::Determinate;
                activity.progress = 1.0;
                activity.detail = format!(
                    "{} saved before you stopped.",
                    outputs::plural(saved_count, "transcript")
                );
                activity.open_transcript = saved_count == 1;
                activity.open_folder = true;
            } else {
                activity.detail = "No transcript was saved.".into();
            }
            if recording_retry {
                // The recording exists: the retry action for a recording is
                // the accurate pipeline, not "hide this card".
                activity.best_row = true;
                activity.open_folder = true;
            } else {
                activity.try_again = true;
            }
        } else if failed_count >= total {
            activity.title = "Transcription failed".into();
            activity.tone = Tone::Crimson;
            activity.bar_color = t.crimson;
            activity.detail.clear();
            activity.message = Some((failure_message(&done.failed), MessageKind::Error));
            if recording_retry {
                activity.best_row = true;
                activity.open_folder = true;
            } else {
                activity.try_again = true;
            }
        } else if failed_count > 0 {
            activity.title = format!("{saved_count} of {total} transcripts ready");
            activity.tone = Tone::Amber;
            activity.bar_color = t.green;
            activity.bar = BarMode::Determinate;
            activity.progress = 1.0;
            activity.detail = format!(
                "Saved in {}",
                outputs::shorten(
                    &done
                        .output_folder
                        .as_deref()
                        .map(|p| p.display().to_string())
                        .unwrap_or_default(),
                    90
                )
            );
            activity.message = Some((failure_message(&done.failed), MessageKind::Error));
            activity.open_transcript = saved_count == 1;
            activity.open_folder = saved_count > 0 || recording_retry;
            if recording_retry {
                activity.best_row = true;
            } else {
                activity.try_again = true;
            }
        } else {
            activity.title = if saved_count == 1 {
                "Transcript ready".into()
            } else {
                format!("{saved_count} transcripts ready")
            };
            activity.tone = Tone::Green;
            activity.bar_color = t.green;
            activity.bar = BarMode::Determinate;
            activity.progress = 1.0;
            activity.detail = format!(
                "Saved in {}",
                outputs::shorten(
                    &done
                        .output_folder
                        .as_deref()
                        .map(|p| p.display().to_string())
                        .unwrap_or_default(),
                    90
                )
            );
            activity.open_transcript = saved_count == 1;
            activity.open_folder = saved_count > 0;
        }
        self.activity = Some(activity);
        self.prepare_rename();
        if self.last_batch_was_recording && failed_count == 0 && !done.cancelled {
            self.offer_find_speakers();
        } else {
            self.find_card = None;
        }
    }

    fn finish_recording(&mut self, done: workers::RecordDone) {
        let t = self.theme.clone();
        self.ui = UiState::Idle;
        self.record_stop = None;
        self.record_saving = false;
        self.level = 0.0;
        let mut activity = self.activity.take().unwrap_or_else(|| Activity::new(&t));
        activity.stop_visible = false;
        activity.stop_enabled = false;
        activity.open_transcript = false;
        activity.open_folder = false;
        activity.try_again = false;
        activity.best_row = false;
        activity.message = None;
        activity.bar = BarMode::Idle;
        activity.progress = 0.0;

        if done.cancelled {
            activity.title = "Recording cancelled".into();
            activity.tone = Tone::Muted;
            activity.detail = "Nothing was saved.".into();
            activity.bar_color = t.muted;
            self.activity = Some(activity);
            self.rename_visible = false;
            self.find_card = None;
            if !self.live_has_text {
                self.live_text = IDLE_LIVE.into();
            }
            return;
        }

        if let Some(error) = &done.error
            && (done.audio.is_none() || done.mic_error || done.model_error)
        {
            activity.title = "Could not record".into();
            activity.tone = Tone::Crimson;
            activity.message = Some((error.clone(), MessageKind::Error));
            activity.bar_color = t.crimson;
            self.activity = Some(activity);
            self.rename_visible = false;
            self.find_card = None;
            if !self.live_has_text {
                self.live_text = IDLE_LIVE.into();
            }
            return;
        }

        let Some(audio_path) = done.audio.clone() else {
            activity.title = "Could not record".into();
            activity.tone = Tone::Crimson;
            activity.message = Some((
                done.error
                    .unwrap_or_else(|| "The recording was not saved.".into()),
                MessageKind::Error,
            ));
            self.activity = Some(activity);
            return;
        };

        self.last_recording = Some(audio_path.clone());
        self.last_output_folder = audio_path.parent().map(Path::to_path_buf);
        self.batch_recording = Some(audio_path.clone());
        if let Some(transcript) = done.transcript.clone() {
            self.last_transcript = Some(transcript);
        }
        self.last_batch_was_recording = true;

        if self.record_mode == RecordMode::Best && !done.model_error {
            // The recording result is not shown: run the accurate pipeline now.
            self.activity = Some(activity);
            self.start_best_quality(audio_path);
            return;
        }

        activity.title = "Recording saved".into();
        activity.tone = Tone::Green;
        activity.bar_color = t.green;
        activity.bar = BarMode::Determinate;
        activity.progress = 1.0;
        let folder = outputs::shorten(
            &audio_path
                .parent()
                .map(|p| p.display().to_string())
                .unwrap_or_default(),
            60,
        );
        activity.detail = if done.transcript.is_some() {
            format!("The audio and the live transcript are in {folder}.")
        } else if self.record_mode == RecordMode::Live {
            format!(
                "No speech was heard live, so there is no live transcript. The audio is in {folder}."
            )
        } else {
            format!("The audio is in {folder}.")
        };
        activity.open_transcript = done.transcript.is_some();
        activity.open_folder = true;
        activity.best_row = true;
        if let Some(error) = done.error {
            activity.message = Some((error, MessageKind::Error));
        }
        self.activity = Some(activity);
        self.find_card = None;
        self.prepare_rename();
    }

    fn finish_find(&mut self, done: workers::FindDone) {
        let t = self.theme.clone();
        self.ui = UiState::Idle;
        self.cancel = None;
        let mut activity = self.activity.take().unwrap_or_else(|| Activity::new(&t));
        activity.stop_visible = false;
        activity.stop_enabled = false;
        activity.best_row = false;
        activity.progress = 0.0;
        activity.bar = BarMode::Idle;
        activity.message = None;
        activity.open_transcript = false;
        activity.open_folder = false;
        activity.try_again = false;

        if let Some(output) = done.output {
            self.last_transcript = Some(output.clone());
            self.last_output_folder = output.parent().map(Path::to_path_buf);
            activity.title = "Speakers added".into();
            activity.tone = Tone::Green;
            activity.bar_color = t.green;
            activity.bar = BarMode::Determinate;
            activity.progress = 1.0;
            activity.detail = format!(
                "{} · took {}",
                outputs::shorten(
                    output
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("transcript"),
                    70
                ),
                outputs::fmt_took(done.took)
            );
            activity.message = Some((
                "The first transcript without names is kept too.".into(),
                MessageKind::Info,
            ));
            activity.open_transcript = true;
            activity.open_folder = true;
            self.find_card = None;
        } else if done.cancelled {
            activity.title = "Stopped".into();
            activity.tone = Tone::Muted;
            activity.bar_color = t.faint;
            activity.detail = "No names were added. Your transcript is unchanged.".into();
            activity.open_transcript = self.transcript_exists();
            activity.open_folder = self.last_output_folder.is_some();
            self.find_card = None;
            self.offer_find_speakers();
        } else {
            activity.title = "Could not find the speakers".into();
            activity.tone = Tone::Crimson;
            activity.bar_color = t.crimson;
            activity.detail = "Your transcript is unchanged.".into();
            activity.message = Some((
                format!(
                    "{}\nSelect \"Show details\" to see why.",
                    done.error.unwrap_or_else(|| "Unknown error".into())
                ),
                MessageKind::Error,
            ));
            activity.open_transcript = self.transcript_exists();
            activity.open_folder = self.last_output_folder.is_some();
            self.find_card = None;
            self.offer_find_speakers();
        }
        self.activity = Some(activity);
        self.prepare_rename();
    }

    fn show_naming(&mut self, preview: SpeakerPreview, reply: SyncSender<NamingReply>) {
        let t = self.theme.clone();
        let mut raw_keys: Vec<String> = preview.speaker_map.keys().cloned().collect();
        raw_keys.sort();
        let rows = raw_keys
            .into_iter()
            .enumerate()
            .map(|(index, raw)| {
                let default_name = preview
                    .speaker_map
                    .get(&raw)
                    .cloned()
                    .unwrap_or_else(|| raw.clone());
                let mut edit = TextEdit::new("");
                edit.focused = index == 0;
                NamingRow {
                    share: preview.shares.get(&raw).copied(),
                    clip: preview.clips.get(&raw).cloned(),
                    edit,
                    raw,
                    default_name,
                }
            })
            .collect();
        self.naming = Some(NamingState {
            rows,
            reply: Some(reply),
        });
        self.ui = UiState::Running;
        if let Some(activity) = &mut self.activity {
            activity.apply_status(Status::NamingSpeakers, &t, true);
        }
    }

    pub fn save_naming(&mut self) {
        crate::platform::stop_playback();
        let Some(mut naming) = self.naming.take() else {
            return;
        };
        let mut map = HashMap::new();
        for row in &naming.rows {
            let typed = row.edit.text.trim();
            map.insert(
                row.raw.clone(),
                if typed.is_empty() {
                    row.default_name.clone()
                } else {
                    typed.to_owned()
                },
            );
        }
        if let Some(reply) = naming.reply.take() {
            let _ = reply.send(NamingReply::Save(map));
        }
    }

    pub fn skip_naming(&mut self) {
        crate::platform::stop_playback();
        if let Some(mut naming) = self.naming.take()
            && let Some(reply) = naming.reply.take()
        {
            let _ = reply.send(NamingReply::Skip);
        }
    }

    pub fn cancel_naming(&mut self) {
        crate::platform::stop_playback();
        if let Some(mut naming) = self.naming.take()
            && let Some(reply) = naming.reply.take()
        {
            let _ = reply.send(NamingReply::Cancel);
        }
    }

    pub fn prepare_rename(&mut self) {
        self.rename_visible = false;
        self.rename_targets.clear();
        let record_target = if self.mode == Mode::Record {
            self.last_recording.as_ref().map(|audio| {
                let targets = rename::recording_set(audio);
                (audio.clone(), targets)
            })
        } else {
            None
        };
        if let Some((audio, targets)) = record_target
            && !targets.is_empty()
        {
            let stem = audio
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
            self.rename_old_stem = stem.clone();
            self.rename_edit = TextEdit::new(stem);
            self.rename_note = if targets.len() > 1 {
                "Renames the recording and its transcripts together.".into()
            } else {
                "Renames the recording.".into()
            };
            self.rename_note_tone = Tone::Muted;
            self.rename_targets = targets;
            self.rename_visible = true;
            return;
        }
        if let Some(transcript) = self.last_transcript.clone()
            && transcript.exists()
        {
            let stem = transcript
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
            self.rename_old_stem = stem.clone();
            self.rename_edit = TextEdit::new(stem);
            self.rename_note = if self.mode == Mode::Record {
                "Renames the recording.".into()
            } else {
                "Renames the transcript. Your original file is not changed.".into()
            };
            self.rename_note_tone = Tone::Muted;
            self.rename_targets = vec![transcript];
            self.rename_visible = true;
        }
    }

    pub fn commit_rename(&mut self) {
        if !self.idle() || self.rename_targets.is_empty() {
            return;
        }
        let name = self.rename_edit.text.trim().to_owned();
        let problem = rename::check_name(&name);
        if !problem.is_empty() {
            self.rename_note = problem.into();
            self.rename_note_tone = Tone::Crimson;
            return;
        }
        match rename::rename_all(&self.rename_targets, &self.rename_old_stem, &name) {
            Ok(map) => {
                if let Some(rec) = &self.last_recording
                    && let Some(new) = map.get(rec)
                {
                    self.last_recording = Some(new.clone());
                }
                if let Some(txt) = &self.last_transcript
                    && let Some(new) = map.get(txt)
                {
                    self.last_transcript = Some(new.clone());
                }
                if let Some(rec) = &self.batch_recording
                    && let Some(new) = map.get(rec)
                {
                    self.batch_recording = Some(new.clone());
                }
                self.log.push(format!(
                    "[Rename] {} -> {name} ({} files)",
                    self.rename_old_stem,
                    map.len()
                ));
                self.rename_note = format!("✓ Renamed to {name}");
                self.rename_note_tone = Tone::Green;
                self.rename_old_stem = name;
                self.rename_targets = map.values().cloned().collect();
                self.rename_targets.sort();
            }
            Err(rename::RenameError::Exists(file)) => {
                self.rename_note = format!(
                    "\"{file}\" already exists in that folder. Please choose another name."
                );
                self.rename_note_tone = Tone::Crimson;
            }
            Err(rename::RenameError::Invalid(_)) | Err(rename::RenameError::Io(_)) => {
                self.rename_note = "Could not rename. If the file is open in another program, close it and try again.".into();
                self.rename_note_tone = Tone::Crimson;
            }
        }
    }

    fn transcript_exists(&self) -> bool {
        self.last_transcript
            .as_ref()
            .map(|p| p.exists())
            .unwrap_or(false)
    }

    pub fn offer_find_speakers(&mut self) {
        if !self.diarization_available {
            self.find_card = None;
            return;
        }
        let Some(audio) = self.last_recording.clone() else {
            self.find_card = None;
            return;
        };
        let Some(transcript) = self.last_transcript.clone() else {
            self.find_card = None;
            return;
        };
        if self.last_segments.is_empty() || !transcript.exists() {
            self.find_card = None;
            return;
        }
        let Ok(duration) = audio::wav_duration(&audio) else {
            self.find_card = None;
            return;
        };
        let speech = transcriber::transcript_speech_seconds(&self.last_segments).min(duration);
        let engine = DiarizationEngine::new(self.paths.clone());
        let estimate = outputs::capitalize_first(&easyscribe_core::diarizer::fmt_estimate(
            engine.estimate_seconds(duration, Some(speech)),
        ));
        let basis = format!(
            "Based on {} of audio, with {} of speech. You can keep using your computer and stop at any time.",
            outputs::fmt_minutes(duration),
            outputs::fmt_minutes(speech)
        );
        self.find_card = Some(FindCard { estimate, basis });
    }

    pub fn hide_activity(&mut self) {
        self.activity = None;
        self.rename_visible = false;
        self.find_card = None;
        self.naming = None;
    }

    pub fn start_batch(&mut self, request: workers::BatchRequest) {
        if !self.idle() {
            return;
        }
        let t = self.theme.clone();
        self.ui = UiState::Running;
        self.log.clear();
        self.cancel = Some(CancelToken::new());
        self.action_hint = None;
        self.rename_visible = false;
        self.find_card = None;
        for file in &mut self.files {
            if request.files.iter().any(|p| p == &file.path) {
                file.state = FileState::Waiting;
            }
        }
        let mut activity = Activity::new(&t);
        activity.apply_status(Status::ExtractingAudio, &t, true);
        self.activity = Some(activity);
        self.last_batch_was_recording = request.batch_recording.is_some();
        self.batch_recording = request.batch_recording.clone();
        if request.batch_recording.is_none() {
            self.last_segments.clear();
        }
        workers::spawn_batch(
            self.paths.clone(),
            self.engine.clone(),
            request,
            self.cancel.clone().unwrap(),
            self.tx.clone(),
            self.waker.clone(),
        );
        self.wake();
    }

    pub fn start_best_quality(&mut self, audio: PathBuf) {
        if !audio.is_file() {
            if let Some(activity) = &mut self.activity {
                activity.message = Some((
                    "The recording was moved or deleted.".into(),
                    MessageKind::Error,
                ));
            }
            return;
        }
        let output = outputs::best_quality_output(&audio);
        let mut overrides = HashMap::new();
        overrides.insert(audio.clone(), output);
        let request = workers::BatchRequest {
            files: vec![audio.clone()],
            output_folder: None,
            timestamps: true,
            diarize: false,
            batch_recording: Some(audio),
            overrides,
        };
        self.start_batch(request);
    }

    pub fn toggle_recording(&mut self) {
        match self.ui {
            UiState::Idle => self.start_recording(),
            UiState::Recording if !self.record_saving => self.stop_recording(),
            _ => {}
        }
    }

    fn start_recording(&mut self) {
        if !self.idle() {
            return;
        }
        let t = self.theme.clone();
        self.ui = UiState::Recording;
        self.mic_menu_open = false;
        self.log.clear();
        self.timer_seconds = 0.0;
        self.level = 0.0;
        self.record_saving = false;
        self.live_has_text = false;
        self.live_text = LISTENING.into();
        self.last_transcript = None;
        self.last_segments.clear();
        self.last_recording = None;
        self.rename_visible = false;
        self.find_card = None;
        let mut activity = Activity::new(&t);
        activity.apply_status(Status::LoadingModel, &t, false);
        activity.title = "Getting ready".into();
        activity.bar = BarMode::Busy;
        activity.bar_color = t.coral;
        activity.stop_visible = false;
        self.activity = Some(activity);
        let stop = CancelToken::new();
        self.record_stop = Some(stop.clone());
        let request = workers::RecordRequest {
            mic: self.selected_mic_device(),
            mode: self.record_mode,
            dest_dir: self.effective_recordings_dir(),
        };
        workers::spawn_recording(
            self.paths.clone(),
            self.engine.clone(),
            request,
            stop,
            self.tx.clone(),
            self.waker.clone(),
        );
        self.wake();
    }

    fn stop_recording(&mut self) {
        if let Some(stop) = &self.record_stop {
            stop.cancel();
        }
        self.record_saving = true;
        if let Some(activity) = &mut self.activity {
            let status = if self.record_mode == RecordMode::Live {
                Status::Finishing
            } else {
                Status::SavingRecording
            };
            activity.apply_status(status, &self.theme, false);
        }
    }

    pub fn cancel_job(&mut self) {
        let t = self.theme.clone();
        if let Some(cancel) = &self.cancel {
            cancel.cancel();
        }
        self.cancel_naming();
        if let Some(activity) = &mut self.activity {
            activity.apply_status(Status::Cancelling, &t, true);
            activity.stop_enabled = false;
        }
    }

    pub fn start_find_speakers(&mut self) {
        if !self.idle() {
            return;
        }
        let Some(audio) = self.last_recording.clone() else {
            return;
        };
        if !audio.is_file() {
            if let Some(activity) = &mut self.activity {
                activity.message = Some((
                    "The recording was moved or deleted.".into(),
                    MessageKind::Error,
                ));
            }
            return;
        }
        let Some(transcript) = self.last_transcript.clone() else {
            return;
        };
        let Some(output) = Some(outputs::speakers_output(&transcript)) else {
            return;
        };
        let t = self.theme.clone();
        self.ui = UiState::Running;
        self.log.clear();
        self.cancel = Some(CancelToken::new());
        self.rename_visible = false;
        self.find_card = None;
        let mut activity = self.activity.take().unwrap_or_else(|| Activity::new(&t));
        activity.apply_status(Status::IdentifyingSpeakers, &t, true);
        activity.detail = outputs::shorten(
            audio
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("recording"),
            80,
        );
        self.activity = Some(activity);
        let cancel = self.cancel.clone().unwrap();
        let naming: Option<Arc<dyn SpeakerNaming>> = if self.diarization_available {
            Some(Arc::new(UiNaming {
                tx: self.tx.clone(),
                waker: self.waker.clone(),
                cancel: cancel.clone(),
            }))
        } else {
            None
        };
        workers::spawn_find_speakers(
            self.paths.clone(),
            audio,
            self.last_segments.clone(),
            output,
            cancel,
            self.tx.clone(),
            self.waker.clone(),
            naming,
        );
        self.wake();
    }

    pub fn add_files(&mut self, paths: Vec<PathBuf>) {
        if self.ui != UiState::Idle || self.mode != Mode::Files {
            return;
        }
        let mut skipped: Vec<PathBuf> = Vec::new();
        let mut accepted: Vec<PathBuf> = Vec::new();
        for path in paths {
            if path.is_file() && outputs::is_supported_media(&path) {
                accepted.push(path);
            } else {
                skipped.push(path);
            }
        }
        if !accepted.is_empty() {
            self.hide_activity();
            for file in &mut self.files {
                file.state = FileState::Waiting;
            }
            let mut existing: HashSet<PathBuf> =
                self.files.iter().map(|f| f.path.clone()).collect();
            for path in accepted {
                if existing.insert(path.clone()) {
                    self.files.push(FileEntry::new(path));
                }
            }
        }
        if skipped.is_empty() {
            if !self.files.is_empty() {
                self.drop_note = None;
            }
        } else {
            let refs: Vec<&Path> = skipped.iter().map(PathBuf::as_path).collect();
            self.drop_note = Some(format!(
                "Skipped {}. EasyScribe reads audio and video files: {}.",
                outputs::names_summary(&refs),
                outputs::supported_extensions_display()
            ));
        }
    }

    pub fn remove_file(&mut self, index: usize) {
        if self.ui != UiState::Idle {
            return;
        }
        if index < self.files.len() {
            self.files.remove(index);
        }
    }

    pub fn clear_files(&mut self) {
        if self.ui != UiState::Idle {
            return;
        }
        self.files.clear();
        self.drop_note = None;
        self.action_hint = None;
        self.hide_activity();
    }

    pub fn choose_files(&mut self) {
        if self.ui != UiState::Idle || self.mode != Mode::Files {
            return;
        }
        let files = dialogs::pick_files();
        if !files.is_empty() {
            self.add_files(files);
        }
    }

    pub fn choose_output_folder(&mut self) {
        if self.ui != UiState::Idle {
            return;
        }
        if let Some(folder) = dialogs::pick_folder("Choose where to save transcripts", None) {
            self.output_folder = Some(folder);
        }
    }

    pub fn choose_recordings_folder(&mut self) {
        if self.ui != UiState::Idle {
            return;
        }
        let current = self.effective_recordings_dir();
        if let Some(folder) =
            dialogs::pick_folder("Choose where to save recordings", Some(current.as_path()))
            && self.settings.save_recordings_dir(Some(&folder)).is_ok()
        {
            self.recordings_dir = Some(folder);
        }
    }

    pub fn reset_recordings_folder(&mut self) {
        if self.ui != UiState::Idle {
            return;
        }
        if self.settings.save_recordings_dir(None).is_ok() {
            self.recordings_dir = None;
        }
    }

    pub fn start(&mut self) {
        if !self.idle() {
            return;
        }
        if self.files.is_empty() {
            self.action_hint = Some(("Add at least one recording first.".into(), true));
            return;
        }
        let free_at = self
            .output_folder
            .as_deref()
            .or_else(|| self.files[0].path.parent());
        if let Some(free) = free_at.and_then(outputs::free_space)
            && free < easyscribe_core::config::MIN_FREE_DISK_BYTES
        {
            let mb = free / (1024 * 1024);
            let text = format!(
                "Only {mb} MB of disk space is free.\nTranscription may fail. Continue anyway?"
            );
            if !dialogs::confirm("Low disk space", &text) {
                return;
            }
        }
        let request = workers::BatchRequest {
            files: self.files.iter().map(|f| f.path.clone()).collect(),
            output_folder: self.output_folder.clone(),
            timestamps: self.timestamps,
            diarize: self.name_speakers && self.diarization_available,
            batch_recording: None,
            overrides: HashMap::new(),
        };
        self.start_batch(request);
    }

    pub fn open_transcript(&mut self) {
        let Some(path) = self.last_transcript.clone() else {
            self.open_folder();
            return;
        };
        if !path.exists() {
            self.open_folder();
            return;
        }
        if let Err(error) = crate::platform::open_path(&path) {
            dialogs::error(
                "Could not open",
                &format!("Could not open:\n{}\n\n{error}", path.display()),
            );
        }
    }

    pub fn open_folder(&mut self) {
        let folder = self
            .last_output_folder
            .clone()
            .or_else(|| self.output_folder.clone())
            .or_else(|| Some(self.effective_recordings_dir()));
        if let Some(folder) = folder
            && folder.exists()
        {
            if let Err(error) = crate::platform::open_path(&folder) {
                dialogs::error(
                    "Could not open",
                    &format!("Could not open:\n{}\n\n{error}", folder.display()),
                );
            }
            return;
        }
        dialogs::info("No folder yet", "There is no folder to open yet.");
    }

    pub fn open_logs_folder(&mut self) {
        let folder = self.paths.logs_dir.clone();
        if let Err(error) = crate::platform::open_path(&folder) {
            dialogs::error(
                "Could not open",
                &format!("Could not open:\n{}\n\n{error}", folder.display()),
            );
        }
    }

    pub fn play_clip(&self, clip: Option<&PathBuf>) {
        if let Some(clip) = clip {
            crate::platform::play_wav(clip);
        }
    }

    pub fn switch_mode(&mut self, mode: Mode) {
        if !self.idle() {
            return;
        }
        if self.mode != mode {
            self.mode = mode;
            self.hide_activity();
            self.mic_menu_open = false;
        }
    }

    pub fn mode_summary_files(&self) -> String {
        let timestamps = if self.timestamps {
            "Timestamps on"
        } else {
            "Timestamps off"
        };
        let speakers = if self.name_speakers {
            "speaker names on"
        } else {
            "speaker names off"
        };
        let saved = match &self.output_folder {
            Some(folder) => format!(
                "saved in {}",
                outputs::shorten(&folder.display().to_string(), 50)
            ),
            None => "saved next to each recording".into(),
        };
        format!("{timestamps}, {speakers}, {saved}.")
    }

    pub fn mic_summary(&self) -> String {
        let mic = self
            .mic_choices
            .get(self.selected_mic)
            .map(|m| m.label.clone())
            .unwrap_or_else(|| "Default microphone".into());
        outputs::shorten(&mic, 50)
    }

    pub fn record_mode_summary(&self) -> &'static str {
        match self.record_mode {
            RecordMode::Best => "Best quality transcript after recording.",
            RecordMode::Live => "Show words as I speak.",
        }
    }

    pub fn rename_note_color(&self) -> rune::Color {
        self.rename_note_tone.color(&self.theme)
    }

    /// Escape closes the Privacy overlay; every other key goes to the focus system.
    pub fn handle_key(&mut self, key: &rune::KeyEvent) -> bool {
        if key.key == rune::Key::Escape && self.privacy_open {
            self.privacy_open = false;
            return true;
        }
        false
    }

    pub fn wake(&self) {
        if let Some(waker) = &self.waker {
            waker();
        }
    }
}

/// Reports worker progress back to the UI thread.
pub struct UiReporter {
    pub tx: Sender<Event>,
    pub waker: Option<Waker>,
}

impl UiReporter {
    fn wake(&self) {
        if let Some(waker) = &self.waker {
            waker();
        }
    }
}

impl Reporter for UiReporter {
    fn status(&self, status: Status) {
        let _ = self.tx.send(Event::Status(status));
        self.wake();
    }

    fn progress(&self, fraction: f64) {
        let _ = self.tx.send(Event::Progress(fraction));
        self.wake();
    }

    fn log(&self, line: &str) {
        let _ = self.tx.send(Event::Log(line.to_owned()));
        self.wake();
    }
}

/// The worker-side half of the embedded naming panel.
pub struct UiNaming {
    pub tx: Sender<Event>,
    pub waker: Option<Waker>,
    pub cancel: CancelToken,
}

impl SpeakerNaming for UiNaming {
    fn ask(&self, preview: &mut SpeakerPreview) -> easyscribe_core::Result<()> {
        let (reply_tx, reply_rx) = mpsc::sync_channel(1);
        if self
            .tx
            .send(Event::AskNames(preview.clone(), reply_tx))
            .is_err()
        {
            return Err(Error::Cancelled);
        }
        if let Some(waker) = &self.waker {
            waker();
        }
        loop {
            match reply_rx.recv_timeout(Duration::from_millis(200)) {
                Ok(NamingReply::Save(map)) => {
                    preview.speaker_map = map;
                    return Ok(());
                }
                Ok(NamingReply::Skip) => return Ok(()),
                Ok(NamingReply::Cancel) => return Err(Error::Cancelled),
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    if self.cancel.is_cancelled() {
                        return Err(Error::Cancelled);
                    }
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => return Err(Error::Cancelled),
            }
        }
    }
}

fn failure_message(failed: &[PathBuf]) -> String {
    if failed.is_empty() {
        return String::new();
    }
    let refs: Vec<&Path> = failed.iter().map(PathBuf::as_path).collect();
    format!(
        "Could not transcribe {}. The file may be damaged or have no sound. Select \"Show details\" to see why.",
        outputs::names_summary(&refs)
    )
}

/// Startup cleanup and legacy import (audit finding 6).
pub fn startup(paths: &Paths) {
    if let Err(error) = logger::setup(&paths.logs_dir) {
        eprintln!("EasyScribe: could not start file logging: {error}");
    }
    logger::clean_temp_files(&paths.temp_dir);
    if let Err(error) = crate::legacy::import(paths) {
        log::warn!("Legacy data import failed: {error}");
    }
}

/// Ask for crash recovery once per orphan, before the main window.
pub fn recover_before_ui(paths: &Paths) {
    let mut dirs = vec![paths.recordings_work_dir.clone()];
    dirs.extend(crate::legacy::recordings_dirs(paths));
    let mut seen = HashSet::new();
    for dir in dirs {
        if !seen.insert(dir.clone()) {
            continue;
        }
        for orphan in recovery::scan_for_orphans(&dir) {
            let duration = orphan.duration_seconds();
            let name = orphan
                .pcm_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("recording");
            let text = format!(
                "An unfinished recording was found:\n\n  File: {name}\n  Duration: {}m {}s\n\nRecover it as a .wav file now?\n(Choosing No keeps the raw recording for next time.)",
                (duration as u64) / 60,
                (duration as u64) % 60
            );
            if !dialogs::confirm("Unfinished Recording Found", &text) {
                continue;
            }
            let dest = Settings::load(paths.settings_file.clone())
                .load_recordings_dir()
                .unwrap_or_else(|| paths.default_recordings_dir.clone());
            match recovery::handle_orphan(
                &orphan.pcm_path,
                &orphan.json_path,
                &orphan.metadata,
                true,
                Some(&dest),
            ) {
                Ok(Some(wav)) => {
                    dialogs::info(
                        "Recovery Complete",
                        &format!("Saved to:\n{}", wav.display()),
                    );
                }
                Ok(None) => {}
                Err(error) => {
                    dialogs::error(
                        "Recovery Failed",
                        &format!("Could not recover {name}:\n{error}"),
                    );
                }
            }
        }
    }
}

/// Startup dependency check from `main.py`: missing ffmpeg/model files abort before the GUI.
pub fn startup_error(paths: &Paths) -> Option<String> {
    let mut errors: Vec<String> = Vec::new();
    if ffmpeg::validate_ffmpeg(paths).is_err() {
        if !paths.ffmpeg_bin.is_file() {
            errors.push(format!(
                "ffmpeg.exe not found.\nExpected at: {}\n\nThe bundled ffmpeg folder may be missing.",
                paths.ffmpeg_bin.display()
            ));
        }
        if !paths.ffprobe_bin.is_file() {
            errors.push(format!(
                "ffprobe.exe not found.\nExpected at: {}",
                paths.ffprobe_bin.display()
            ));
        }
    }
    let missing = paths.validate_models();
    if !missing.is_empty() {
        let mut block = String::from("Model files missing or incomplete:\n");
        for line in missing {
            block.push_str(&format!("  • {line}\n"));
        }
        errors.push(block.trim_end().to_owned());
    }
    if errors.is_empty() {
        None
    } else {
        Some(format!(
            "Some program files are missing. Please download EasyScribe again.\n\n{}",
            errors.join("\n\n")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_state() -> AppState {
        let dir = tempfile::tempdir().expect("temp dir");
        let paths = Paths::at(dir.path().to_path_buf()).expect("paths");
        // Leak the temp dir for the duration of the test process; the path must
        // outlive the state and its workers.
        std::mem::forget(dir);
        AppState::new(Init {
            paths,
            window_height: 820.0,
            preparing: false,
        })
    }

    #[test]
    fn start_without_files_sets_the_hint() {
        let mut state = test_state();
        state.start();
        assert_eq!(
            state.action_hint.as_ref().map(|h| h.0.as_str()),
            Some("Add at least one recording first.")
        );
    }

    #[test]
    fn mode_switch_is_blocked_while_running() {
        let mut state = test_state();
        state.ui = UiState::Running;
        state.switch_mode(Mode::Record);
        assert_eq!(state.mode, Mode::Files);
        state.ui = UiState::Idle;
        state.switch_mode(Mode::Record);
        assert_eq!(state.mode, Mode::Record);
    }

    #[test]
    fn naming_panel_focuses_the_first_speaker() {
        let mut state = test_state();
        let mut speaker_map = HashMap::new();
        speaker_map.insert("SPEAKER_00".to_owned(), "Speaker 1".to_owned());
        speaker_map.insert("SPEAKER_01".to_owned(), "Speaker 2".to_owned());
        let preview = SpeakerPreview {
            speaker_map,
            clips: HashMap::new(),
            shares: HashMap::new(),
        };
        let (tx, _rx) = std::sync::mpsc::sync_channel(1);
        state.show_naming(preview, tx);
        let naming = state.naming.as_ref().expect("naming panel");
        assert!(naming.rows[0].edit.focused);
        assert!(!naming.rows[1].edit.focused);
    }

    #[test]
    fn escape_closes_only_the_privacy_overlay() {
        let mut state = test_state();
        let escape = rune::KeyEvent {
            key: rune::Key::Escape,
            mods: rune::Mods::default(),
        };
        assert!(!state.handle_key(&escape));
        state.privacy_open = true;
        assert!(state.handle_key(&escape));
        assert!(!state.privacy_open);
    }

    #[test]
    fn files_result_offers_the_rename_row_for_one_transcript() {
        let mut state = test_state();
        let dir = tempfile::tempdir().unwrap();
        let transcript = dir.path().join("Team meeting.txt");
        std::fs::write(&transcript, "text").unwrap();
        state.ui = UiState::Running;
        state.activity = Some(Activity::new(&state.theme));
        state.finish_batch(workers::BatchDone {
            total: 1,
            saved: vec![transcript.clone()],
            failed: Vec::new(),
            cancelled: false,
            fatal: None,
            recording_segments: None,
            output_folder: Some(dir.path().to_path_buf()),
        });
        assert_eq!(state.ui, UiState::Idle);
        let activity = state.activity.as_ref().unwrap();
        assert_eq!(activity.title, "Transcript ready");
        assert!(activity.open_transcript && activity.open_folder);
        assert!(state.rename_visible);
        assert_eq!(state.last_transcript.as_deref(), Some(transcript.as_path()));
    }

    #[test]
    fn rename_commits_all_targets_and_reports_success() {
        let mut state = test_state();
        let dir = tempfile::tempdir().unwrap();
        let audio = dir.path().join("recording_1.wav");
        let transcript = dir.path().join("recording_1.txt");
        std::fs::write(&audio, "wav").unwrap();
        std::fs::write(&transcript, "text").unwrap();
        state.mode = Mode::Record;
        state.last_recording = Some(audio.clone());
        state.last_transcript = Some(transcript.clone());
        state.prepare_rename();
        assert!(state.rename_visible);
        state.rename_edit.text = "Team meeting".into();
        state.commit_rename();
        assert!(dir.path().join("Team meeting.wav").exists());
        assert!(dir.path().join("Team meeting.txt").exists());
        assert_eq!(state.rename_note_tone, Tone::Green);
        assert_eq!(
            state.last_recording,
            Some(dir.path().join("Team meeting.wav"))
        );

        // A second rename in the same panel must use the new stem, not the
        // recording's original name.
        state.rename_edit.text = "Team meeting final".into();
        state.commit_rename();
        assert_eq!(state.rename_note_tone, Tone::Green);
        assert!(dir.path().join("Team meeting final.wav").exists());
        assert!(dir.path().join("Team meeting final.txt").exists());
        assert_eq!(
            state.last_recording,
            Some(dir.path().join("Team meeting final.wav"))
        );
    }
}
