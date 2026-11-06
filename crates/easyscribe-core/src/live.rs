//! Live microphone transcription: Silero VAD chunks a microphone queue, a
//! decoder thread turns each chunk into text through the shared recognizer.
//! Ported from `src/live_transcriber.py`.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use crate::mic::AudioQueue;
use crate::report::CancelToken;
use crate::sherpa_engine::{self, TranscriptionEngine};
use crate::vad::pad_live_segment;
use crate::{Error, Paths, Result};

pub const DECODE_QUEUE_MAX: usize = 8;
pub const STOP_TIMEOUT: Duration = Duration::from_secs(60);
const HISTORY_SAMPLES: usize = crate::vad::PAD_SAMPLES + 16 * crate::config::VAD_CHUNK_SAMPLES;

/// sherpa-onnx is not safe to call from two native sessions at once. The old
/// per-session lock allowed a detached decoder from a timed-out Stop to run
/// concurrently with the next recording's VAD and recognizer. One process-wide
/// lock serialises every live VAD/recognizer call instead.
static ONNX_LOCK: Mutex<()> = Mutex::new(());
/// True from the moment a decoder thread starts until it exits. A new live
/// session must wait, so a detached decoder cannot race a fresh engine.
static DECODER_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Acquire the live ONNX lock, recovering from a panic in a previous native
/// call instead of poisoning every later session.
fn onnx_guard() -> std::sync::MutexGuard<'static, ()> {
    ONNX_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[derive(Debug, Clone)]
pub enum LiveEvent {
    Text(String),
    Overflow,
    Error(String),
}

/// Answer a pair of requests: start initialises both threads, and a new
/// recording can begin after stop.
pub struct LiveTranscriber {
    vad_thread: Option<std::thread::JoinHandle<()>>,
    decode_thread: Option<std::thread::JoinHandle<()>>,
    decode_done: Option<mpsc::Receiver<()>>,
}

impl LiveTranscriber {
    pub fn new() -> Self {
        Self {
            vad_thread: None,
            decode_thread: None,
            decode_done: None,
        }
    }

    pub fn start(
        &mut self,
        engine: Arc<TranscriptionEngine>,
        mic_queue: Arc<AudioQueue>,
        cancel: CancelToken,
        events: mpsc::Sender<LiveEvent>,
    ) -> Result<()> {
        if DECODER_ACTIVE.swap(true, Ordering::SeqCst) {
            return Err(Error::Transcription(
                "The previous live transcript is still finishing. Wait a moment and try again."
                    .into(),
            ));
        }
        if let Err(error) = engine.ensure_loaded(&crate::report::SilentReporter) {
            DECODER_ACTIVE.store(false, Ordering::SeqCst);
            return Err(error);
        }
        let decode = Arc::new(DecodeQueue::new(DECODE_QUEUE_MAX));
        let vad_done = Arc::new(Mutex::new(false));

        let (decode_done_tx, decode_done_rx) = mpsc::channel();
        let decode_handle = {
            let engine = engine.clone();
            let vad_done = vad_done.clone();
            let decode = decode.clone();
            let events = events.clone();
            match std::thread::Builder::new()
                .name("LiveDecoder".into())
                .spawn(move || {
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        decode_loop(engine, decode, vad_done, events);
                    }));
                    DECODER_ACTIVE.store(false, Ordering::SeqCst);
                    let _ = decode_done_tx.send(());
                    if result.is_err() {
                        log::error!("Live decoder thread panicked");
                    }
                }) {
                Ok(handle) => handle,
                Err(e) => {
                    DECODER_ACTIVE.store(false, Ordering::SeqCst);
                    return Err(Error::Transcription(format!(
                        "Could not start live decoder: {e}"
                    )));
                }
            }
        };

        let vad_handle = {
            let paths = engine.paths().clone();
            let vad_done_thread = vad_done.clone();
            match std::thread::Builder::new()
                .name("LiveVAD".into())
                .spawn(move || vad_loop(paths, mic_queue, decode, cancel, vad_done_thread, events))
            {
                Ok(handle) => handle,
                Err(e) => {
                    // Let the decoder observe vad_done and exit instead of
                    // waiting forever for a VAD thread that never started.
                    *vad_done.lock().expect("vad_done poisoned") = true;
                    let _ = decode_handle.join();
                    DECODER_ACTIVE.store(false, Ordering::SeqCst);
                    return Err(Error::Transcription(format!(
                        "Could not start live VAD: {e}"
                    )));
                }
            }
        };

        self.decode_thread = Some(decode_handle);
        self.decode_done = Some(decode_done_rx);
        self.vad_thread = Some(vad_handle);
        log::info!("LiveTranscriber started");
        Ok(())
    }

    /// The caller must have cancelled the token first. Wait for the decoder
    /// first: the VAD's final flush needs the same ONNX lock, so joining VAD
    /// before the decoder could block forever. If the native decode ignores
    /// the deadline, detach both threads; the process-wide lock and
    /// `DECODER_ACTIVE` keep the next session from racing them.
    pub fn stop(&mut self) {
        let timed_out = if let Some(done) = self.decode_done.take() {
            done.recv_timeout(STOP_TIMEOUT).is_err()
        } else {
            false
        };
        if let Some(thread) = self.decode_thread.take() {
            if timed_out {
                log::warn!(
                    "Live decoder did not stop within {} s; detaching it",
                    STOP_TIMEOUT.as_secs()
                );
                drop(thread);
            } else if thread.join().is_err() {
                log::warn!("Live decoder thread panicked");
            }
        }
        if let Some(thread) = self.vad_thread.take() {
            if timed_out {
                // The VAD thread may be waiting for the ONNX lock held by the
                // detached decoder. Drop the handle; it exits once the
                // decoder releases the lock.
                log::warn!("Live VAD thread did not finish; detaching it");
                drop(thread);
            } else if thread.join().is_err() {
                log::warn!("Live VAD thread panicked");
            }
        }
        log::info!("LiveTranscriber stopped");
    }
}

impl Default for LiveTranscriber {
    fn default() -> Self {
        Self::new()
    }
}

struct DecodeQueue {
    inner: Mutex<VecDeque<Vec<f32>>>,
    ready: Condvar,
    capacity: usize,
}

impl DecodeQueue {
    fn new(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(VecDeque::with_capacity(capacity)),
            ready: Condvar::new(),
            capacity,
        }
    }

    /// Returns false when an older segment had to be dropped.
    fn push_drop_oldest(&self, item: Vec<f32>) -> bool {
        let mut queue = self.inner.lock().expect("decode queue poisoned");
        let mut dropped = false;
        if queue.len() >= self.capacity {
            queue.pop_front();
            dropped = true;
        }
        queue.push_back(item);
        self.ready.notify_one();
        !dropped
    }

    fn pop_timeout(&self, timeout: Duration) -> Option<Vec<f32>> {
        let mut queue = self.inner.lock().expect("decode queue poisoned");
        if queue.is_empty() {
            let (guard, _) = self
                .ready
                .wait_timeout(queue, timeout)
                .expect("decode queue poisoned");
            queue = guard;
        }
        queue.pop_front()
    }
}

fn vad_loop(
    paths: Paths,
    mic_queue: Arc<AudioQueue>,
    decode: Arc<DecodeQueue>,
    cancel: CancelToken,
    vad_done: Arc<Mutex<bool>>,
    events: mpsc::Sender<LiveEvent>,
) {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut detector = match sherpa_engine::build_vad(&paths, 30.0) {
            Ok(detector) => detector,
            Err(e) => {
                let _ = events.send(LiveEvent::Error(e.to_string()));
                return;
            }
        };
        let mut state = LiveState::default();
        while !cancel.is_cancelled() {
            if let Some(chunk) = mic_queue.pop_timeout(Duration::from_millis(100)) {
                state.feed(&mut detector, chunk, &decode, &events);
            }
        }
        while let Some(chunk) = mic_queue.try_pop() {
            state.feed(&mut detector, chunk, &decode, &events);
        }
        {
            let _guard = onnx_guard();
            detector.flush();
        }
        state.drain(&mut detector, &decode, &events);
    }));
    if result.is_err() {
        log::error!("Live VAD thread failed");
    }
    *vad_done.lock().expect("vad_done poisoned") = true;
}

#[derive(Default)]
struct LiveState {
    history: Vec<f32>,
    history_start: usize,
    prev_segment_end: usize,
    overflow_notified: bool,
}

impl LiveState {
    fn feed(
        &mut self,
        detector: &mut sherpa_rs::silero_vad::SileroVad,
        chunk: Vec<f32>,
        decode: &DecodeQueue,
        events: &mpsc::Sender<LiveEvent>,
    ) {
        self.history.extend_from_slice(&chunk);
        if self.history.len() > HISTORY_SAMPLES {
            let trim = self.history.len() - HISTORY_SAMPLES;
            self.history.drain(..trim);
            self.history_start += trim;
        }
        {
            let _guard = onnx_guard();
            detector.accept_waveform(chunk);
        }
        self.drain(detector, decode, events);
    }

    fn drain(
        &mut self,
        detector: &mut sherpa_rs::silero_vad::SileroVad,
        decode: &DecodeQueue,
        events: &mpsc::Sender<LiveEvent>,
    ) {
        while !detector.is_empty() {
            let segment = detector.front();
            let seg_start = segment.start.max(0) as usize;
            let samples = pad_live_segment(
                &self.history,
                self.history_start,
                self.prev_segment_end,
                seg_start,
                &segment.samples,
            );
            self.prev_segment_end = seg_start + segment.samples.len();
            if !decode.push_drop_oldest(samples) && !self.overflow_notified {
                self.overflow_notified = true;
                log::warn!("Live decode queue full — dropped oldest segment");
                let _ = events.send(LiveEvent::Overflow);
            }
            detector.pop();
        }
    }
}

fn decode_loop(
    engine: Arc<TranscriptionEngine>,
    decode: Arc<DecodeQueue>,
    vad_done: Arc<Mutex<bool>>,
    events: mpsc::Sender<LiveEvent>,
) {
    loop {
        if let Some(samples) = decode.pop_timeout(Duration::from_millis(100)) {
            let _guard = onnx_guard();
            match engine.recognize(&samples) {
                Ok(text) if !text.is_empty() => {
                    let _ = events.send(LiveEvent::Text(text));
                }
                Ok(_) => {}
                Err(e) => {
                    let _ = events.send(LiveEvent::Error(e.to_string()));
                    return;
                }
            }
        } else if *vad_done.lock().expect("vad_done poisoned") {
            return;
        }
    }
}
