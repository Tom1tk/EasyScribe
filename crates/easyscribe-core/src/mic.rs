//! Microphone capture, crash-safe PCM writing and level metering.
//! The Python app used sounddevice + a writer thread; this is the cpal port.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Sample, SampleFormat, Stream};
use serde::Serialize;

use crate::audio::AntiAliasResampler;
use crate::config::VAD_SAMPLE_RATE;
use crate::recovery::{RecordingMetadata, recover_pcm_to_wav};
use crate::report::PauseToken;
use crate::{Error, Result};

pub const PCM_SAMPLE_RATE: u32 = 16_000;
pub const CHUNK_SAMPLES: usize = 512;
const FSYNC_EVERY: u64 = 50;
const VAD_QUEUE_CHUNKS: usize = 60 * 16_000 / 512;

#[derive(Debug, Clone)]
pub struct MicDevice {
    pub name: String,
}

/// List every input device in the host's order. Errors mean "no devices".
pub fn list_input_devices() -> Vec<MicDevice> {
    let host = cpal::default_host();
    match host.input_devices() {
        Ok(devices) => devices
            .filter_map(|device| device.description().ok())
            .map(|description| MicDevice {
                name: description.name().to_owned(),
            })
            .collect(),
        Err(e) => {
            log::warn!("Could not enumerate microphones: {e}");
            Vec::new()
        }
    }
}

/// A bounded queue of 512-sample float chunks, newest dropped when full.
pub struct AudioQueue {
    inner: Mutex<VecDeque<Vec<f32>>>,
    ready: Condvar,
    capacity: usize,
    dropped: AtomicBool,
    high_water: AtomicBool,
}

impl AudioQueue {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(VecDeque::with_capacity(capacity)),
            ready: Condvar::new(),
            capacity,
            dropped: AtomicBool::new(false),
            high_water: AtomicBool::new(false),
        }
    }

    pub fn push_drop_newest(&self, chunk: Vec<f32>) {
        let mut queue = self.inner.lock().expect("audio queue poisoned");
        let depth = queue.len();
        if depth >= self.capacity * 3 / 4 && !self.high_water.swap(true, Ordering::SeqCst) {
            log::warn!(
                "VAD queue depth {depth}/{}: live transcription is falling behind",
                self.capacity
            );
        } else if depth < self.capacity / 2 {
            self.high_water.store(false, Ordering::SeqCst);
        }
        if depth >= self.capacity {
            if !self.dropped.swap(true, Ordering::SeqCst) {
                log::warn!("VAD queue full — dropping audio from live transcription");
            }
            return;
        }
        queue.push_back(chunk);
        self.ready.notify_one();
    }

    pub fn pop_timeout(&self, timeout: Duration) -> Option<Vec<f32>> {
        let mut queue = self.inner.lock().expect("audio queue poisoned");
        if queue.is_empty() {
            let (guard, _) = self
                .ready
                .wait_timeout(queue, timeout)
                .expect("audio queue poisoned");
            queue = guard;
        }
        queue.pop_front()
    }

    pub fn try_pop(&self) -> Option<Vec<f32>> {
        self.inner.lock().expect("audio queue poisoned").pop_front()
    }
}

#[derive(Serialize)]
struct Sidecar {
    sample_rate: u32,
    channels: u16,
    bit_depth: u16,
    samples_written: u64,
}

/// A running recording. Keeping the stream here also keeps the callback alive.
pub struct MicRecorder {
    stream: Option<Stream>,
    vad_queue: Option<Arc<AudioQueue>>,
    pcm_tx: Option<SyncSender<Vec<i16>>>,
    writer: Option<std::thread::JoinHandle<()>>,
    writer_error: Arc<std::sync::Mutex<Option<String>>>,
    stop: Arc<AtomicBool>,
    samples_written: Arc<AtomicU64>,
    pcm_drops: Arc<AtomicU64>,
    peak: Arc<AtomicU32>,
    pcm_path: Option<PathBuf>,
    json_path: Option<PathBuf>,
    session_stem: Option<String>,
    feed_vad: bool,
}

impl Default for MicRecorder {
    fn default() -> Self {
        Self::new()
    }
}

impl MicRecorder {
    pub fn new() -> Self {
        Self {
            stream: None,
            vad_queue: None,
            pcm_tx: None,
            writer: None,
            writer_error: Arc::new(std::sync::Mutex::new(None)),
            stop: Arc::new(AtomicBool::new(false)),
            samples_written: Arc::new(AtomicU64::new(0)),
            pcm_drops: Arc::new(AtomicU64::new(0)),
            peak: Arc::new(AtomicU32::new(0)),
            pcm_path: None,
            json_path: None,
            session_stem: None,
            feed_vad: true,
        }
    }

    pub fn vad_queue(&self) -> Option<Arc<AudioQueue>> {
        self.vad_queue.clone()
    }

    pub fn session_stem(&self) -> Option<String> {
        self.session_stem.clone()
    }

    /// Peak amplitude (0.0..=1.0) since the previous call, then reset.
    pub fn get_level(&self) -> f32 {
        let bits = self.peak.swap(0, Ordering::SeqCst);
        f32::from_bits(bits)
    }

    /// Recorded duration in seconds. It follows the samples actually written,
    /// so it stops while capture is paused and always matches the WAV length.
    pub fn recorded_seconds(&self) -> f64 {
        self.samples_written.load(Ordering::SeqCst) as f64 / PCM_SAMPLE_RATE as f64
    }

    pub fn is_recording(&self) -> bool {
        self.stream.is_some()
    }

    /// Start capturing. `device_name == None` means the system default.
    pub fn start(
        &mut self,
        device_name: Option<&str>,
        output_dir: &Path,
        feed_vad: bool,
        pause: PauseToken,
    ) -> Result<()> {
        if self.stream.is_some() {
            return Err(Error::Transcription(
                "a recording is already running".into(),
            ));
        }
        std::fs::create_dir_all(output_dir)?;
        let host = cpal::default_host();
        let device = match device_name {
            Some(name) => host
                .input_devices()
                .map_err(|e| Error::Transcription(format!("Could not list microphones: {e}")))?
                .find(|d| {
                    d.description()
                        .map(|desc| desc.name() == name)
                        .unwrap_or(false)
                })
                .ok_or_else(|| Error::Transcription(format!("Microphone not found: {name}")))?,
            None => host
                .default_input_device()
                .ok_or_else(|| Error::Transcription("No microphone was found.".into()))?,
        };

        let config = choose_config(&device)?;
        let sample_rate = config.sample_rate() as u32;
        let channels = config.channels() as usize;

        let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();
        let suffix = uuid::Uuid::new_v4().simple().to_string();
        let stem = format!("recording_{timestamp}_{}", &suffix[..8]);
        let pcm_path = output_dir.join(format!("{stem}.pcm"));
        let json_path = output_dir.join(format!("{stem}.json"));

        self.stop = Arc::new(AtomicBool::new(false));
        self.samples_written = Arc::new(AtomicU64::new(0));
        self.pcm_drops = Arc::new(AtomicU64::new(0));
        self.peak = Arc::new(AtomicU32::new(0));
        // A previous failed start can leave an error behind; it must not be
        // reported against this recording.
        *self.writer_error.lock().expect("writer error poisoned") = None;
        self.pcm_path = Some(pcm_path.clone());
        self.json_path = Some(json_path.clone());
        self.session_stem = Some(stem);
        self.feed_vad = feed_vad;

        let (pcm_tx, pcm_rx) = mpsc::sync_channel::<Vec<i16>>(256);
        let (ready_tx, ready_rx) = mpsc::sync_channel::<std::result::Result<(), String>>(1);
        let samples_written = self.samples_written.clone();
        let stop = self.stop.clone();
        let writer_json = json_path.clone();
        let pcm_for_writer = pcm_path.clone();
        let writer_error = self.writer_error.clone();
        let writer = std::thread::Builder::new()
            .name("PCMWriter".into())
            .spawn(move || {
                writer_loop(
                    pcm_rx,
                    pcm_for_writer,
                    writer_json,
                    samples_written,
                    stop,
                    writer_error,
                    ready_tx,
                )
            })
            .map_err(|e| Error::Transcription(format!("Could not start PCM writer: {e}")))?;
        // Do not open the microphone until the PCM file exists. A full or
        // read-only recordings folder must fail here, not after the meeting.
        match ready_rx.recv_timeout(Duration::from_secs(5)) {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                self.stop.store(true, Ordering::SeqCst);
                drop(pcm_tx);
                let _ = writer.join();
                let _ = std::fs::remove_file(&pcm_path);
                let _ = std::fs::remove_file(&json_path);
                return Err(Error::Transcription(format!(
                    "The raw audio file could not be created: {error}"
                )));
            }
            Err(_) => {
                self.stop.store(true, Ordering::SeqCst);
                drop(pcm_tx);
                let _ = writer.join();
                let _ = std::fs::remove_file(&pcm_path);
                let _ = std::fs::remove_file(&json_path);
                return Err(Error::Transcription(
                    "The PCM writer did not start in time.".into(),
                ));
            }
        }

        let vad_queue = if feed_vad {
            Some(Arc::new(AudioQueue::new(VAD_QUEUE_CHUNKS)))
        } else {
            None
        };
        let peak = self.peak.clone();
        let pcm_drops = self.pcm_drops.clone();
        let pcm_tx_for_callback = pcm_tx.clone();
        let vad_for_callback = vad_queue.clone();
        let pause_for_callback = pause.clone();

        let mut resampler = (sample_rate != VAD_SAMPLE_RATE)
            .then(|| AntiAliasResampler::new(sample_rate, VAD_SAMPLE_RATE));
        let stream = build_stream(&device, &config, move |data| {
            // Keep the device stream open (closing and reopening it could
            // change the format or fail), but append nothing while paused.
            if pause_for_callback.is_paused() {
                peak.store(0, Ordering::Relaxed);
                return;
            }
            let mut mono = Vec::with_capacity(data.len() / channels.max(1));
            for frame in data.chunks(channels.max(1)) {
                // A stereo interface may put the only active mic on either
                // channel; average instead of taking channel 0.
                mono.push(frame.iter().copied().sum::<f32>() / frame.len() as f32);
            }
            if let Some(resampler) = resampler.as_mut() {
                mono = resampler.process(&mono);
            }
            let mut pcm = Vec::with_capacity(mono.len());
            let mut vad = Vec::with_capacity(mono.len());
            let mut local_peak = 0.0_f32;
            for value in mono {
                let clamped = value.clamp(-1.0, 1.0);
                let sample = (clamped * 32768.0).clamp(-32768.0, 32767.0) as i16;
                local_peak = local_peak.max((sample as i32).unsigned_abs() as f32 / 32768.0);
                pcm.push(sample);
                if vad_for_callback.is_some() {
                    vad.push(sample as f32 / 32768.0);
                }
            }
            if local_peak > 0.0 {
                let _ = peak.fetch_max(local_peak.to_bits(), Ordering::Relaxed);
            }
            if !pcm.is_empty() {
                match pcm_tx_for_callback.try_send(pcm) {
                    Ok(()) => {}
                    Err(mpsc::TrySendError::Full(_)) => {
                        pcm_drops.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(mpsc::TrySendError::Disconnected(_)) => {}
                }
            }
            if let (Some(queue), true) = (&vad_for_callback, !vad.is_empty()) {
                queue.push_drop_newest(vad);
            }
        });

        let stream = match stream {
            Ok(stream) => stream,
            Err(e) => {
                // The callback (and its sender) is gone, but the local sender
                // is still alive: without dropping it, `join` waits forever.
                drop(pcm_tx);
                self.stop.store(true, Ordering::SeqCst);
                let _ = writer.join();
                let _ = std::fs::remove_file(&pcm_path);
                let _ = std::fs::remove_file(&json_path);
                return Err(e);
            }
        };
        if let Err(e) = stream.play() {
            drop(stream);
            drop(pcm_tx);
            self.stop.store(true, Ordering::SeqCst);
            let _ = writer.join();
            let _ = std::fs::remove_file(&pcm_path);
            let _ = std::fs::remove_file(&json_path);
            return Err(Error::Transcription(format!(
                "Could not start the microphone: {e}"
            )));
        }

        self.stream = Some(stream);
        self.vad_queue = vad_queue;
        self.pcm_tx = Some(pcm_tx);
        self.writer = Some(writer);
        log::info!("MicRecorder started: {}", pcm_path.display());
        Ok(())
    }

    /// Stop capture and convert the PCM to WAV. `dest_dir` is tried first;
    /// on failure the WAV is written next to the raw file.
    pub fn stop(
        &mut self,
        convert_to_wav: bool,
        dest_dir: Option<&Path>,
    ) -> Result<Option<PathBuf>> {
        if let Some(stream) = self.stream.take() {
            let _ = stream.pause();
            drop(stream);
        }
        self.stop.store(true, Ordering::SeqCst);
        self.pcm_tx.take();
        if let Some(writer) = self.writer.take() {
            let _ = writer.join();
        }
        let drops = self.pcm_drops.load(Ordering::Relaxed);
        if drops > 0 {
            log::warn!(
                "PCM writer queue was full for {drops} audio block(s); the recording may have small gaps"
            );
        }
        if let Some(error) = self
            .writer_error
            .lock()
            .expect("writer error poisoned")
            .take()
        {
            let written = self.samples_written.load(Ordering::SeqCst);
            if written == 0 {
                return Err(Error::Transcription(format!(
                    "The raw audio could not be written: {error}"
                )));
            }
            // Keep the valid part of the recording. Python still converts the
            // PCM after a writer error, and a partial WAV is better than none.
            log::warn!(
                "PCM writer stopped after {written} samples ({error}); saving the partial recording"
            );
        }
        if !convert_to_wav {
            return Ok(None);
        }
        let written = self.samples_written.load(Ordering::SeqCst);
        let Some(pcm_path) = self.pcm_path.clone() else {
            return Ok(None);
        };
        let Some(json_path) = self.json_path.clone() else {
            return Ok(None);
        };
        if written == 0 {
            // Paused (or stopped) before a single sample was kept. There is
            // nothing to convert, and an empty "recording" is not an error.
            let _ = std::fs::remove_file(&pcm_path);
            let _ = std::fs::remove_file(&json_path);
            log::info!("Recording stopped before any audio was captured");
            return Ok(None);
        }
        let metadata = RecordingMetadata {
            sample_rate: PCM_SAMPLE_RATE,
            channels: 1,
            bit_depth: 16,
            samples_written: written,
        };
        let wav_name = pcm_path
            .with_extension("wav")
            .file_name()
            .unwrap_or_default()
            .to_owned();
        let mut folders: Vec<PathBuf> = Vec::new();
        if let Some(dest) = dest_dir
            && dest != pcm_path.parent().unwrap_or(Path::new("."))
        {
            folders.push(dest.to_path_buf());
        }
        folders.push(pcm_path.parent().unwrap_or(Path::new(".")).to_path_buf());
        let mut last_error: Option<String> = None;
        for folder in folders {
            let wav_path = folder.join(&wav_name);
            if let Err(e) = std::fs::create_dir_all(&folder) {
                log::error!("Recording folder {} is unusable: {e}", folder.display());
                last_error = Some(format!("{}: {e}", folder.display()));
                continue;
            }
            match recover_pcm_to_wav(&pcm_path, &metadata, &wav_path) {
                Ok(()) => {
                    let _ = std::fs::remove_file(&pcm_path);
                    let _ = std::fs::remove_file(&json_path);
                    log::info!("Recording saved: {}", wav_path.display());
                    return Ok(Some(wav_path));
                }
                Err(e) => {
                    log::error!("PCM→WAV conversion to {} failed: {e}", folder.display());
                    last_error = Some(format!("{}: {e}", folder.display()));
                }
            }
        }
        // Never swallow the reason: the UI shows it, and the PCM/sidecar pair
        // stays in the recordings folder so crash recovery can retry it.
        Err(Error::Recovery(format!(
            "The recording could not be converted to WAV. The raw audio is kept in {}.{}",
            pcm_path
                .parent()
                .map(|p| p.display().to_string())
                .unwrap_or_default(),
            last_error
                .map(|e| format!("\nLast error: {e}"))
                .unwrap_or_default()
        )))
    }
}

fn supported_format(format: SampleFormat) -> bool {
    matches!(
        format,
        SampleFormat::F32
            | SampleFormat::F64
            | SampleFormat::I8
            | SampleFormat::U8
            | SampleFormat::I16
            | SampleFormat::U16
            | SampleFormat::I24
            | SampleFormat::U24
            | SampleFormat::I32
            | SampleFormat::U32
            | SampleFormat::I64
            | SampleFormat::U64
    )
}

fn format_rank(format: SampleFormat) -> u8 {
    match format {
        SampleFormat::F32 => 0,
        SampleFormat::I16 => 1,
        SampleFormat::I24 => 2,
        SampleFormat::I32 => 3,
        SampleFormat::F64 => 4,
        _ => 5,
    }
}

fn choose_config(device: &cpal::Device) -> Result<cpal::SupportedStreamConfig> {
    let mut best: Option<((u8, u8, u8, u32), cpal::SupportedStreamConfig)> = None;
    if let Ok(ranges) = device.supported_input_configs() {
        for range in ranges {
            if range.channels() < 1 || !supported_format(range.sample_format()) {
                continue;
            }
            let min = range.min_sample_rate();
            let max = range.max_sample_rate();
            // Prefer the native 16 kHz stream. Otherwise use 48 kHz rather
            // than whichever range happened to be enumerated first.
            let rate = if min <= VAD_SAMPLE_RATE && max >= VAD_SAMPLE_RATE {
                VAD_SAMPLE_RATE
            } else if min <= 48_000 && max >= 48_000 {
                48_000
            } else {
                max.min(96_000)
            };
            let Some(config) = range.try_with_sample_rate(rate) else {
                continue;
            };
            let score = (
                u8::from(rate != VAD_SAMPLE_RATE),
                u8::from(config.channels() != 1),
                format_rank(config.sample_format()),
                rate,
            );
            if best.as_ref().is_none_or(|(current, _)| score < *current) {
                best = Some((score, config));
            }
        }
    }
    if let Some((_, config)) = best {
        return Ok(config);
    }
    device
        .default_input_config()
        .map_err(|e| Error::Transcription(format!("Could not open the microphone: {e}")))
}

fn build_stream(
    device: &cpal::Device,
    supported: &cpal::SupportedStreamConfig,
    mut on_samples: impl FnMut(&[f32]) + Send + 'static,
) -> Result<Stream> {
    let config = supported.config();
    let sample_format = supported.sample_format();
    let err_fn = |e| log::warn!("microphone stream error: {e}");
    let stream = match sample_format {
        SampleFormat::F32 => device.build_input_stream(
            config,
            move |data: &[f32], _| on_samples(data),
            err_fn,
            None,
        ),
        SampleFormat::I16 => device.build_input_stream(
            config,
            move |data: &[i16], _| {
                let converted: Vec<f32> = data.iter().map(|v| *v as f32 / 32768.0).collect();
                on_samples(&converted)
            },
            err_fn,
            None,
        ),
        SampleFormat::U16 => device.build_input_stream(
            config,
            move |data: &[u16], _| {
                let converted: Vec<f32> = data
                    .iter()
                    .map(|v| (*v as f32 - 32768.0) / 32768.0)
                    .collect();
                on_samples(&converted)
            },
            err_fn,
            None,
        ),
        SampleFormat::I32 => device.build_input_stream(
            config,
            move |data: &[i32], _| {
                let converted: Vec<f32> =
                    data.iter().map(|v| *v as f32 / 2_147_483_648.0).collect();
                on_samples(&converted)
            },
            err_fn,
            None,
        ),
        SampleFormat::I8 => device.build_input_stream(
            config,
            move |data: &[i8], _| {
                let converted: Vec<f32> = data.iter().map(|v| *v as f32 / 128.0).collect();
                on_samples(&converted)
            },
            err_fn,
            None,
        ),
        SampleFormat::U8 => device.build_input_stream(
            config,
            move |data: &[u8], _| {
                let converted: Vec<f32> =
                    data.iter().map(|v| (*v as f32 - 128.0) / 128.0).collect();
                on_samples(&converted)
            },
            err_fn,
            None,
        ),
        SampleFormat::I24 => device.build_input_stream(
            config,
            move |data: &[cpal::I24], _| {
                let converted: Vec<f32> = data.iter().map(|v| f32::from_sample(*v)).collect();
                on_samples(&converted)
            },
            err_fn,
            None,
        ),
        SampleFormat::U24 => device.build_input_stream(
            config,
            move |data: &[cpal::U24], _| {
                let converted: Vec<f32> = data.iter().map(|v| f32::from_sample(*v)).collect();
                on_samples(&converted)
            },
            err_fn,
            None,
        ),
        SampleFormat::I64 => device.build_input_stream(
            config,
            move |data: &[i64], _| {
                let converted: Vec<f32> = data
                    .iter()
                    .map(|v| *v as f32 / 9_223_372_036_854_775_808.0)
                    .collect();
                on_samples(&converted)
            },
            err_fn,
            None,
        ),
        SampleFormat::U32 => device.build_input_stream(
            config,
            move |data: &[u32], _| {
                let converted: Vec<f32> = data
                    .iter()
                    .map(|v| ((*v as f64 - 2_147_483_648.0) / 2_147_483_648.0) as f32)
                    .collect();
                on_samples(&converted)
            },
            err_fn,
            None,
        ),
        SampleFormat::U64 => device.build_input_stream(
            config,
            move |data: &[u64], _| {
                let converted: Vec<f32> = data
                    .iter()
                    .map(|v| {
                        ((*v as f64 - 9_223_372_036_854_775_808.0) / 9_223_372_036_854_775_808.0)
                            as f32
                    })
                    .collect();
                on_samples(&converted)
            },
            err_fn,
            None,
        ),
        SampleFormat::F64 => device.build_input_stream(
            config,
            move |data: &[f64], _| {
                let converted: Vec<f32> = data.iter().map(|v| *v as f32).collect();
                on_samples(&converted)
            },
            err_fn,
            None,
        ),
        other => {
            return Err(Error::Transcription(format!(
                "unsupported microphone sample format: {other:?}"
            )));
        }
    };
    stream.map_err(|e| Error::Transcription(format!("Could not open the microphone: {e}")))
}

fn writer_loop(
    rx: Receiver<Vec<i16>>,
    pcm_path: PathBuf,
    json_path: PathBuf,
    samples_written: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    error_slot: Arc<std::sync::Mutex<Option<String>>>,
    ready: SyncSender<std::result::Result<(), String>>,
) {
    use std::io::Write;
    let file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&pcm_path);
    let mut file = match file {
        Ok(file) => file,
        Err(e) => {
            log::error!("PCM writer error: {e}");
            if let Ok(mut slot) = error_slot.lock() {
                *slot = Some(e.to_string());
            }
            let _ = ready.send(Err(e.to_string()));
            return;
        }
    };
    // Make the sidecar exist from the first second, so a crash or power loss
    // leaves a recoverable pair instead of an orphan PCM.
    write_sidecar(&json_path, 0);
    if ready.send(Ok(())).is_err() {
        return;
    }
    let mut chunks = 0_u64;
    loop {
        match rx.recv_timeout(Duration::from_millis(200)) {
            Ok(chunk) => {
                if let Err(e) = write_pcm_chunk(&mut file, &chunk) {
                    record_writer_error(&error_slot, &e);
                    return;
                }
                let total = samples_written.fetch_add(chunk.len() as u64, Ordering::SeqCst)
                    + chunk.len() as u64;
                chunks += 1;
                if chunks.is_multiple_of(FSYNC_EVERY) {
                    let _ = file.flush();
                    let _ = file.sync_data();
                    write_sidecar(&json_path, total);
                }
            }
            Err(RecvTimeoutError::Timeout) => {
                if stop.load(Ordering::SeqCst) {
                    while let Ok(chunk) = rx.try_recv() {
                        if let Err(e) = write_pcm_chunk(&mut file, &chunk) {
                            record_writer_error(&error_slot, &e);
                            return;
                        }
                        samples_written.fetch_add(chunk.len() as u64, Ordering::SeqCst);
                    }
                    break;
                }
            }
            Err(RecvTimeoutError::Disconnected) => {
                while let Ok(chunk) = rx.try_recv() {
                    if let Err(e) = write_pcm_chunk(&mut file, &chunk) {
                        record_writer_error(&error_slot, &e);
                        return;
                    }
                    samples_written.fetch_add(chunk.len() as u64, Ordering::SeqCst);
                }
                break;
            }
        }
    }
    let total = samples_written.load(Ordering::SeqCst);
    let _ = file.flush();
    let _ = file.sync_data();
    write_sidecar(&json_path, total);
}

fn write_pcm_chunk(file: &mut std::fs::File, chunk: &[i16]) -> std::io::Result<()> {
    use std::io::Write;
    let mut bytes = Vec::with_capacity(chunk.len() * 2);
    for sample in chunk {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    file.write_all(&bytes)
}

fn record_writer_error(error_slot: &std::sync::Mutex<Option<String>>, error: &std::io::Error) {
    log::error!("PCM writer error: {error}");
    if let Ok(mut slot) = error_slot.lock() {
        *slot = Some(error.to_string());
    }
}

fn write_sidecar(json_path: &Path, samples_written: u64) {
    let sidecar = Sidecar {
        sample_rate: PCM_SAMPLE_RATE,
        channels: 1,
        bit_depth: 16,
        samples_written,
    };
    let Ok(text) = serde_json::to_string(&sidecar) else {
        return;
    };
    let tmp = json_path.with_extension(format!("json.tmp-{}", std::process::id()));
    let result = (|| -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(&tmp)?;
        file.write_all(text.as_bytes())?;
        file.sync_all()?;
        drop(file);
        // Windows rename does not replace an existing file.
        let _ = std::fs::remove_file(json_path);
        std::fs::rename(&tmp, json_path)
    })();
    if let Err(e) = result {
        let _ = std::fs::remove_file(&tmp);
        log::warn!("JSON sidecar write failed: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queue_drops_newest_when_full() {
        let q = AudioQueue::new(1);
        q.push_drop_newest(vec![1.0]);
        q.push_drop_newest(vec![2.0]);
        assert_eq!(q.try_pop(), Some(vec![1.0]));
        assert_eq!(q.try_pop(), None);
    }

    #[test]
    fn stop_writes_the_recording_to_the_chosen_folder() {
        let dir = tempfile::tempdir().unwrap();
        let work = dir.path().join("work");
        let dest = dir.path().join("dest");
        std::fs::create_dir_all(&work).unwrap();
        let stem = "recording_20260101_000000";
        let pcm_path = work.join(format!("{stem}.pcm"));
        let json_path = work.join(format!("{stem}.json"));
        let samples: Vec<i16> = (0..1600).map(|i| (i % 100) as i16).collect();
        let mut raw = Vec::with_capacity(samples.len() * 2);
        for sample in &samples {
            raw.extend_from_slice(&sample.to_le_bytes());
        }
        std::fs::write(&pcm_path, raw).unwrap();
        std::fs::write(
            &json_path,
            serde_json::to_string(&RecordingMetadata {
                sample_rate: PCM_SAMPLE_RATE,
                channels: 1,
                bit_depth: 16,
                samples_written: samples.len() as u64,
            })
            .unwrap(),
        )
        .unwrap();

        let mut recorder = MicRecorder::new();
        recorder.pcm_path = Some(pcm_path.clone());
        recorder.json_path = Some(json_path.clone());
        recorder.samples_written = Arc::new(AtomicU64::new(samples.len() as u64));
        recorder.session_stem = Some(stem.into());

        // This also guards the Windows-only `sync_all` failure: the WAV must
        // be reopened for writing before FlushFileBuffers.
        let wav = recorder
            .stop(true, Some(&dest))
            .unwrap()
            .expect("the recording should be converted");
        assert_eq!(wav.parent(), Some(dest.as_path()));
        assert!(!pcm_path.exists());
        assert!(!json_path.exists());
        let reader = hound::WavReader::open(&wav).unwrap();
        assert_eq!(reader.spec().channels, 1);
        assert_eq!(reader.spec().sample_rate, PCM_SAMPLE_RATE);
        assert_eq!(reader.duration() as usize, samples.len());
    }

    #[test]
    fn stop_before_any_audio_discards_the_empty_pair() {
        let dir = tempfile::tempdir().unwrap();
        let work = dir.path().join("work");
        let dest = dir.path().join("dest");
        std::fs::create_dir_all(&work).unwrap();
        let stem = "recording_20260101_000000";
        let pcm_path = work.join(format!("{stem}.pcm"));
        let json_path = work.join(format!("{stem}.json"));
        std::fs::write(&pcm_path, []).unwrap();
        std::fs::write(
            &json_path,
            serde_json::to_string(&RecordingMetadata {
                sample_rate: PCM_SAMPLE_RATE,
                channels: 1,
                bit_depth: 16,
                samples_written: 0,
            })
            .unwrap(),
        )
        .unwrap();

        let mut recorder = MicRecorder::new();
        recorder.pcm_path = Some(pcm_path.clone());
        recorder.json_path = Some(json_path.clone());
        recorder.samples_written = Arc::new(AtomicU64::new(0));
        recorder.session_stem = Some(stem.into());
        let result = recorder.stop(true, Some(&dest)).unwrap();
        assert!(result.is_none());
        assert!(!pcm_path.exists());
        assert!(!json_path.exists());
        assert!(!dest.exists());
    }
}
