//! Crash recovery for raw microphone recordings, ported from `src/recovery.py`.
//!
//! The recorder writes int16 PCM plus a JSON sidecar (`samples_written` lets
//! recovery drop a torn final write). Declining recovery keeps both files;
//! accepting converts them to WAV and removes the pair only on success.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordingMetadata {
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
    #[serde(default = "default_channels")]
    pub channels: u16,
    #[serde(default = "default_bit_depth")]
    pub bit_depth: u16,
    #[serde(default)]
    pub samples_written: u64,
}

fn default_sample_rate() -> u32 { 16_000 }
fn default_channels() -> u16 { 1 }
fn default_bit_depth() -> u16 { 16 }

impl Default for RecordingMetadata {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            channels: 1,
            bit_depth: 16,
            samples_written: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Orphan {
    pub pcm_path: PathBuf,
    pub json_path: PathBuf,
    pub metadata: RecordingMetadata,
}

impl Orphan {
    pub fn duration_seconds(&self) -> f64 {
        let sr = self.metadata.sample_rate;
        if sr == 0 { 0.0 } else { self.metadata.samples_written as f64 / sr as f64 }
    }
}

/// Find `recording_*.pcm` + `.json` pairs, sorted by PCM filename.
pub fn scan_for_orphans(output_dir: &Path) -> Vec<Orphan> {
    if !output_dir.exists() {
        return Vec::new();
    }
    let Ok(entries) = std::fs::read_dir(output_dir) else { return Vec::new() };
    let mut pcm_paths: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.extension().and_then(|e| e.to_str()).map(|e| e.eq_ignore_ascii_case("pcm")).unwrap_or(false)
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("recording_"))
                    .unwrap_or(false)
        })
        .collect();
    pcm_paths.sort();

    let mut orphans = Vec::new();
    for pcm_path in pcm_paths {
        let json_path = pcm_path.with_extension("json");
        if !json_path.exists() {
            continue;
        }
        match std::fs::read_to_string(&json_path)
            .map_err(Error::from)
            .and_then(|text| serde_json::from_str::<RecordingMetadata>(&text).map_err(Error::from))
        {
            Ok(metadata) => orphans.push(Orphan { pcm_path, json_path, metadata }),
            Err(e) => log::warn!("Could not read {}: {e}", json_path.display()),
        }
    }
    orphans
}

/// Convert raw int16 PCM to WAV, keeping only `samples_written` frames.
pub fn recover_pcm_to_wav(pcm_path: &Path, metadata: &RecordingMetadata, wav_path: &Path) -> Result<()> {
    let byte_count = (metadata.samples_written as usize).saturating_mul(2);
    let raw = std::fs::read(pcm_path)?;
    let raw = &raw[..byte_count.min(raw.len())];
    let spec = hound::WavSpec {
        channels: metadata.channels.max(1),
        sample_rate: metadata.sample_rate.max(1),
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(wav_path, spec)?;
    for chunk in raw.chunks_exact(2) {
        writer.write_sample(i16::from_le_bytes([chunk[0], chunk[1]]))?;
    }
    writer.finalize()?;
    log::info!(
        "Recovered {} → {} ({} bytes)",
        pcm_path.file_name().and_then(|n| n.to_str()).unwrap_or("?"),
        wav_path.file_name().and_then(|n| n.to_str()).unwrap_or("?"),
        raw.len()
    );
    Ok(())
}

/// Apply the user's choice. `recover == false` deliberately leaves both files
/// alone so the prompt reappears next launch. Returns the WAV path on success.
pub fn handle_orphan(
    pcm_path: &Path,
    json_path: &Path,
    metadata: &RecordingMetadata,
    recover: bool,
    dest_dir: Option<&Path>,
) -> Result<Option<PathBuf>> {
    if !recover {
        return Ok(None);
    }
    let mut wav_path = pcm_path.with_extension("wav");
    if let Some(dest) = dest_dir {
        if let Err(e) = std::fs::create_dir_all(dest) {
            log::error!("Could not create {}; recovering next to the raw file: {e}", dest.display());
            recover_pcm_to_wav(pcm_path, metadata, &wav_path)?;
        } else {
            let candidate = dest.join(wav_path.file_name().unwrap_or_default());
            match recover_pcm_to_wav(pcm_path, metadata, &candidate) {
                Ok(()) => wav_path = candidate,
                Err(e) => {
                    log::error!("Could not recover into {}, keeping it next to the raw file: {e}", dest.display());
                    recover_pcm_to_wav(pcm_path, metadata, &wav_path)?;
                }
            }
        }
    } else {
        recover_pcm_to_wav(pcm_path, metadata, &wav_path)?;
    }
    let _ = std::fs::remove_file(pcm_path);
    let _ = std::fs::remove_file(json_path);
    Ok(Some(wav_path))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_orphan(dir: &Path, samples_written: u64, extra: usize) -> (PathBuf, PathBuf) {
        let pcm = dir.join("recording_20260101_000000.pcm");
        let json = dir.join("recording_20260101_000000.json");
        let total = samples_written as usize + extra;
        let mut raw = Vec::with_capacity(total * 2);
        for _ in 0..total {
            raw.extend_from_slice(&[0x01, 0x02]);
        }
        std::fs::write(&pcm, raw).unwrap();
        std::fs::write(
            &json,
            serde_json::to_string(&RecordingMetadata {
                sample_rate: 16_000,
                channels: 1,
                bit_depth: 16,
                samples_written,
            })
            .unwrap(),
        )
        .unwrap();
        (pcm, json)
    }

    #[test]
    fn recovery_truncates_to_samples_written() {
        let dir = tempfile::tempdir().unwrap();
        let (pcm, _) = write_orphan(dir.path(), 1000, 200);
        let wav = dir.path().join("recording_20260101_000000.wav");
        let metadata = RecordingMetadata { samples_written: 1000, ..Default::default() };
        recover_pcm_to_wav(&pcm, &metadata, &wav).unwrap();
        let reader = hound::WavReader::open(&wav).unwrap();
        assert_eq!(reader.spec().channels, 1);
        assert_eq!(reader.spec().sample_rate, 16_000);
        assert_eq!(reader.spec().bits_per_sample, 16);
        assert_eq!(reader.duration() as u64, 1000);
    }

    #[test]
    fn declining_keeps_both_files() {
        let dir = tempfile::tempdir().unwrap();
        let (pcm, json) = write_orphan(dir.path(), 100, 0);
        let meta = RecordingMetadata { samples_written: 100, ..Default::default() };
        assert_eq!(handle_orphan(&pcm, &json, &meta, false, None).unwrap(), None);
        assert!(pcm.exists());
        assert!(json.exists());
    }

    #[test]
    fn accepting_recovers_and_cleans_up() {
        let dir = tempfile::tempdir().unwrap();
        let (pcm, json) = write_orphan(dir.path(), 100, 0);
        let meta = RecordingMetadata { samples_written: 100, ..Default::default() };
        let wav = handle_orphan(&pcm, &json, &meta, true, None).unwrap().unwrap();
        assert!(wav.exists());
        assert!(!pcm.exists());
        assert!(!json.exists());
        assert_eq!(hound::WavReader::open(&wav).unwrap().duration(), 100);
    }

    #[test]
    fn scan_finds_pairs_and_skips_missing_json() {
        let dir = tempfile::tempdir().unwrap();
        write_orphan(dir.path(), 10, 0);
        std::fs::write(dir.path().join("recording_20260101_000001.pcm"), b"x").unwrap();
        let found = scan_for_orphans(dir.path());
        assert_eq!(found.len(), 1);
    }
}
