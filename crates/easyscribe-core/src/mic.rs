//! Microphone capture, crash-safe PCM writing and level metering.
//! The Python app used sounddevice + a writer thread; this is the cpal port.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream};
use serde::Serialize;

use crate::audio::LinearResampler;
use crate::config::VAD_SAMPLE_RATE;
use crate::recovery::{RecordingMetadata, recover_pcm_to_wav};
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
}

impl AudioQueue {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(VecDeque::with_capacity(capacity)),
            ready: Condvar::new(),
            capacity,
            dropped: AtomicBool::new(false),
        }
    }

    pub fn push_drop_newest(&self, chunk: Vec<f32>) {
        let mut queue = self.inner.lock().expect("audio queue poisoned");
        if queue.len() >= self.capacity {
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
    pcm_tx: Option<Sender<Vec<i16>>>,
    writer: Option<std::thread::JoinHandle<()>>,
    writer_error: Arc<std::sync::Mutex<Option<String>>>,
    stop: Arc<AtomicBool>,
    samples_written: Arc<AtomicU64>,
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

    pub fn is_recording(&self) -> bool {
        self.stream.is_some()
    }

    /// Start capturing. `device_name == None` means the system default.
    pub fn start(
        &mut self,
        device_name: Option<&str>,
        output_dir: &Path,
        feed_vad: bool,
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
        let pcm_path = output_dir.join(format!("recording_{timestamp}.pcm"));
        let json_path = output_dir.join(format!("recording_{timestamp}.json"));

        self.stop = Arc::new(AtomicBool::new(false));
        self.samples_written = Arc::new(AtomicU64::new(0));
        self.peak = Arc::new(AtomicU32::new(0));
        // A previous failed start can leave an error behind; it must not be
        // reported against this recording.
        *self.writer_error.lock().expect("writer error poisoned") = None;
        self.pcm_path = Some(pcm_path.clone());
        self.json_path = Some(json_path.clone());
        self.session_stem = Some(format!("recording_{timestamp}"));
        self.feed_vad = feed_vad;

        let (pcm_tx, pcm_rx) = mpsc::channel::<Vec<i16>>();
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
                )
            })
            .map_err(|e| Error::Transcription(format!("Could not start PCM writer: {e}")))?;

        let vad_queue = if feed_vad {
            Some(Arc::new(AudioQueue::new(VAD_QUEUE_CHUNKS)))
        } else {
            None
        };
        let peak = self.peak.clone();
        let pcm_tx_for_callback = pcm_tx.clone();
        let vad_for_callback = vad_queue.clone();

        let mut resampler = (sample_rate != VAD_SAMPLE_RATE)
            .then(|| LinearResampler::new(sample_rate, VAD_SAMPLE_RATE));
        let stream = build_stream(&device, &config, move |data| {
            let mut mono = Vec::with_capacity(data.len() / channels.max(1));
            for frame in data.chunks(channels.max(1)) {
                mono.push(frame[0]);
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
                let _ = pcm_tx_for_callback.send(pcm);
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
        if let Some(error) = self
            .writer_error
            .lock()
            .expect("writer error poisoned")
            .take()
        {
            return Err(Error::Transcription(format!(
                "The raw audio could not be written: {error}"
            )));
        }
        if !convert_to_wav {
            return Ok(None);
        }
        let Some(pcm_path) = self.pcm_path.clone() else {
            return Ok(None);
        };
        let Some(json_path) = self.json_path.clone() else {
            return Ok(None);
        };
        let metadata = RecordingMetadata {
            sample_rate: PCM_SAMPLE_RATE,
            channels: 1,
            bit_depth: 16,
            samples_written: self.samples_written.load(Ordering::SeqCst),
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
        for folder in folders {
            let wav_path = folder.join(&wav_name);
            if std::fs::create_dir_all(&folder).is_err() {
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
                }
            }
        }
        Ok(None)
    }
}

fn choose_config(device: &cpal::Device) -> Result<cpal::SupportedStreamConfig> {
    let mut fallback: Option<cpal::SupportedStreamConfig> = None;
    if let Ok(ranges) = device.supported_input_configs() {
        for range in ranges {
            if range.channels() < 1 {
                continue;
            }
            if range.min_sample_rate() <= VAD_SAMPLE_RATE
                && range.max_sample_rate() >= VAD_SAMPLE_RATE
            {
                if let Some(config) = range.try_with_sample_rate(VAD_SAMPLE_RATE) {
                    if config.channels() == 1 {
                        return Ok(config);
                    }
                    fallback.get_or_insert(config);
                }
            } else {
                fallback.get_or_insert(range.with_max_sample_rate());
            }
        }
    }
    if let Some(config) = fallback {
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
    _stop: Arc<AtomicBool>,
    error_slot: Arc<std::sync::Mutex<Option<String>>>,
) {
    use std::io::Write;
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&pcm_path);
    let mut file = match file {
        Ok(file) => file,
        Err(e) => {
            log::error!("PCM writer error: {e}");
            if let Ok(mut slot) = error_slot.lock() {
                *slot = Some(e.to_string());
            }
            return;
        }
    };
    let mut chunks = 0_u64;
    for chunk in rx {
        let mut bytes = Vec::with_capacity(chunk.len() * 2);
        for sample in &chunk {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        if let Err(e) = file.write_all(&bytes) {
            log::error!("PCM writer error: {e}");
            if let Ok(mut slot) = error_slot.lock() {
                *slot = Some(e.to_string());
            }
            return;
        }
        let total =
            samples_written.fetch_add(chunk.len() as u64, Ordering::SeqCst) + chunk.len() as u64;
        chunks += 1;
        if chunks.is_multiple_of(FSYNC_EVERY) {
            let _ = file.flush();
            let _ = file.sync_data();
            write_sidecar(&json_path, total);
        }
    }
    let total = samples_written.load(Ordering::SeqCst);
    let _ = file.flush();
    let _ = file.sync_data();
    write_sidecar(&json_path, total);
}

fn write_sidecar(json_path: &Path, samples_written: u64) {
    let sidecar = Sidecar {
        sample_rate: PCM_SAMPLE_RATE,
        channels: 1,
        bit_depth: 16,
        samples_written,
    };
    if let Ok(text) = serde_json::to_string(&sidecar)
        && let Err(e) = std::fs::write(json_path, text)
    {
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
}
