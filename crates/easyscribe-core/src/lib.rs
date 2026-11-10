//! EasyScribe core: all non-UI behaviour, ported from the Python v3.2.3 app.
//!
//! The UI crate owns no domain logic. Everything here is testable headlessly.

pub mod audio;
pub mod config;
pub mod diarizer;
pub mod error;
pub mod ffmpeg;
pub mod live;
pub mod logger;
pub mod mic;
pub mod offline;
pub mod outputs;
pub mod payload;
pub mod recovery;
pub mod rename;
pub mod settings;
pub mod sherpa_engine;
pub mod transcriber;
pub mod transcript;
pub mod vad;
pub mod whispercli;
pub mod whispercpp;

pub use config::{APP_NAME, APP_VERSION, Paths};
pub use error::{Error, Result};
pub use model::{AssignedSegment, Segment, SpeakerShare, Status, Turn};
pub use report::{CancelToken, PauseToken, Reporter, SilentReporter};
pub use sherpa_engine::TranscriptionEngine;

mod model;
mod report;
#[cfg(windows)]
mod win_job;
