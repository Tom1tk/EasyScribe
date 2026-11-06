//! Cancellation and progress reporting shared by every long-running job.
//!
//! The Python app used a `threading.Event` and three callbacks. Rust workers
//! get the same shape: a cloneable token plus a `Reporter` trait. The UI
//! implements the reporter by sending events over a channel; tests use the
//! recording reporter below.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::model::Status;

/// A cheap, cloneable cancellation flag.
#[derive(Debug, Clone, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }

    pub fn check(&self) -> crate::Result<()> {
        if self.is_cancelled() {
            Err(crate::Error::Cancelled)
        } else {
            Ok(())
        }
    }
}

/// Everything a long job reports. Default methods make it easy for tests and
/// for callers that only care about one channel.
pub trait Reporter: Send + Sync {
    fn status(&self, _status: Status) {}
    fn progress(&self, _fraction: f64) {}
    fn log(&self, _line: &str) {}
}

/// A reporter that does nothing.
pub struct SilentReporter;

impl Reporter for SilentReporter {}
