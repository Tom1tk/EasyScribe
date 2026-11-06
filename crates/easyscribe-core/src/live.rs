//! Live microphone transcription: Silero VAD chunks a microphone queue, a
//! decoder thread turns each chunk into text through the shared recognizer.
//! Ported from `src/live_transcriber.py`.

use std::collections::VecDeque;
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use crate::mic::AudioQueue;
use crate::report::CancelToken;
use crate::sherpa_engine::{self, TranscriptionEngine};
use crate::vad::pad_live_segment;
use crate::{Paths, Result};

pub const DECODE_QUEUE_MAX: usize = 8;
pub const STOP_TIMEOUT: Duration = Duration::from_secs(60);
const HISTORY_SAMPLES: usize = crate::vad::PAD_SAMPLES + 16 * crate::config::VAD_CHUNK_SAMPLES;

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
}

impl LiveTranscriber {
    pub fn new() -> Self {
        Self { vad_thread: None, decode_thread: None }
    }

    pub fn start(
        &mut self,
        engine: Arc<TranscriptionEngine>,
        mic_queue: Arc<AudioQueue>,
        cancel: CancelToken,
        events: mpsc::Sender<LiveEvent>,
    ) -> Result<()> {
        engine.ensure_loaded(&crate::report::SilentReporter)?;
        let decode = Arc::new(DecodeQueue::new(DECODE_QUEUE_MAX));
        let vad_done = Arc::new(Mutex::new(false));
        let onnx_lock = Arc::new(Mutex::new(()));

        let decode_handle = {
            let engine = engine.clone();
            let vad_done = vad_done.clone();
            let onnx_lock = onnx_lock.clone();
            let decode = decode.clone();
            let events = events.clone();
            std::thread::Builder::new()
                .name("LiveDecoder".into())
                .spawn(move || decode_loop(engine, decode, vad_done, events, onnx_lock))
                .map_err(|e| crate::Error::Transcription(format!("Could not start live decoder: {e}")))?
        };

        let vad_handle = {
            let paths = engine.paths().clone();
            let vad_done = vad_done.clone();
            let onnx_lock = onnx_lock.clone();
            std::thread::Builder::new()
                .name("LiveVAD".into())
                .spawn(move || vad_loop(paths, mic_queue, decode, cancel, vad_done, onnx_lock, events))
                .map_err(|e| crate::Error::Transcription(format!("Could not start live VAD: {e}")))?
        };

        self.decode_thread = Some(decode_handle);
        self.vad_thread = Some(vad_handle);
        log::info!("LiveTranscriber started");
        Ok(())
    }

    /// The caller must have cancelled the token first. Joins VAD, then decoder.
    pub fn stop(&mut self) {
        if let Some(thread) = self.vad_thread.take() {
            if thread.join().is_err() {
                log::warn!("Live VAD thread panicked");
            }
        }
        if let Some(thread) = self.decode_thread.take() {
            if thread.join().is_err() {
                log::warn!("Live decoder thread panicked");
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
            let (guard, _) = self.ready.wait_timeout(queue, timeout).expect("decode queue poisoned");
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
    onnx_lock: Arc<Mutex<()>>,
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
                state.feed(&mut detector, chunk, &decode, &onnx_lock, &events);
            }
        }
        while let Some(chunk) = mic_queue.try_pop() {
            state.feed(&mut detector, chunk, &decode, &onnx_lock, &events);
        }
        {
            let _guard = onnx_lock.lock().expect("onnx lock poisoned");
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
        onnx_lock: &Mutex<()>,
        events: &mpsc::Sender<LiveEvent>,
    ) {
        self.history.extend_from_slice(&chunk);
        if self.history.len() > HISTORY_SAMPLES {
            let trim = self.history.len() - HISTORY_SAMPLES;
            self.history.drain(..trim);
            self.history_start += trim;
        }
        {
            let _guard = onnx_lock.lock().expect("onnx lock poisoned");
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
    onnx_lock: Arc<Mutex<()>>,
) {
    loop {
        if let Some(samples) = decode.pop_timeout(Duration::from_millis(100)) {
            let _guard = onnx_lock.lock().expect("onnx lock poisoned");
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
