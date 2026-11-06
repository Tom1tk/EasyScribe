//! Error types. The variants mirror the Python exceptions so the UI can show
//! the same plain-language messages and make the same decisions.

#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The user pressed Stop. Every stage raises this same error.
    #[error("cancelled")]
    Cancelled,

    #[error("model files are missing:\n{0}")]
    ModelNotFound(String),

    #[error("{0}")]
    Transcription(String),

    #[error("{0}")]
    WhisperCpp(String),

    #[error("{0}")]
    Diarization(String),

    #[error("{0}")]
    FfmpegNotFound(String),

    #[error("{0}")]
    FfmpegExtraction(String),

    #[error("{0}")]
    Settings(String),

    #[error("{0}")]
    Payload(String),

    #[error("{0}")]
    WriteTranscript(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[error(transparent)]
    Wav(#[from] hound::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// True when this error is the user's Stop rather than a failure.
    pub fn is_cancelled(&self) -> bool {
        matches!(self, Error::Cancelled)
    }
}
