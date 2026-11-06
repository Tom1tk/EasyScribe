//! Shared value types used across the core.

/// One timed line produced by a transcription engine.
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

impl Segment {
    pub fn new(start: f64, end: f64, text: impl Into<String>) -> Self {
        Self {
            start,
            end,
            text: text.into(),
        }
    }
}

/// A transcript line after speaker diarization.
#[derive(Debug, Clone, PartialEq)]
pub struct AssignedSegment {
    pub speaker: String,
    pub text: String,
    pub start: f64,
    pub end: f64,
}

/// One diarization turn.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Turn {
    pub start: f64,
    pub end: f64,
    pub speaker: String,
}

impl Turn {
    pub fn new(start: f64, end: f64, speaker: impl Into<String>) -> Self {
        Self {
            start,
            end,
            speaker: speaker.into(),
        }
    }
}

/// How much of the talking one speaker did, in `0.0..=1.0`.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerShare {
    pub speaker: String,
    pub share: f64,
}

/// The internal status names the Python app sends to the UI. The UI maps
/// these to plain-language titles and colours (see `REFERENCE_UI.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Ready,
    LoadingModel,
    ExtractingAudio,
    Transcribing,
    IdentifyingSpeakers,
    NamingSpeakers,
    WritingTranscript,
    Recording,
    Finishing,
    SavingRecording,
    Cancelling,
    Done,
    Cancelled,
    Failed,
}

impl Status {
    /// The wire name used by logs and the Python app.
    pub fn name(self) -> &'static str {
        match self {
            Status::Ready => "Ready",
            Status::LoadingModel => "Loading Model",
            Status::ExtractingAudio => "Extracting Audio",
            Status::Transcribing => "Transcribing",
            Status::IdentifyingSpeakers => "Identifying Speakers",
            Status::NamingSpeakers => "Naming Speakers",
            Status::WritingTranscript => "Writing Transcript",
            Status::Recording => "Recording",
            Status::Finishing => "Finishing",
            Status::SavingRecording => "Saving Recording",
            Status::Cancelling => "Cancelling…",
            Status::Done => "Done",
            Status::Cancelled => "Cancelled",
            Status::Failed => "Failed",
        }
    }
}
