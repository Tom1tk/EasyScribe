//! WAV loading and small audio helpers shared by every engine.

use std::path::Path;

use crate::{Error, Result};

/// Length of a WAV file in seconds, read from its header.
pub fn wav_duration(path: &Path) -> Result<f64> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.sample_rate == 0 {
        return Ok(0.0);
    }
    Ok(reader.duration() as f64 / spec.sample_rate as f64)
}

/// Load a WAV as mono `f32` samples, using the same integer scaling as the
/// Python app (`/32768`, `/2147483648`).
pub fn load_wav_float32(path: &Path) -> Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;
    let mut samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => match spec.bits_per_sample {
            8 => reader
                .samples::<i8>()
                .map(|s| s.map(|v| v as f32 / 128.0))
                .collect::<std::result::Result<_, _>>()?,
            9..=16 => reader
                .samples::<i16>()
                .map(|s| s.map(|v| v as f32 / 32768.0))
                .collect::<std::result::Result<_, _>>()?,
            17..=24 => reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / 8_388_608.0))
                .collect::<std::result::Result<_, _>>()?,
            25..=32 => reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / 2_147_483_648.0))
                .collect::<std::result::Result<_, _>>()?,
            bits => {
                return Err(Error::Transcription(format!(
                    "unsupported WAV sample width: {bits} bits"
                )));
            }
        },
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<_, _>>()?,
    };
    if channels > 1 {
        let mono: Vec<f32> = samples
            .chunks_exact(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect();
        samples = mono;
    }
    Ok((samples, spec.sample_rate))
}

/// Write mono 16-bit PCM to a WAV file.
pub fn write_mono_i16(path: &Path, sample_rate: u32, samples: &[i16]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for sample in samples {
        writer.write_sample(*sample)?;
    }
    writer.finalize()?;
    Ok(())
}

/// Linear resampling for devices that cannot open at 16 kHz. The reference app
/// asks PortAudio for 16 kHz and lets the host convert; this is the portable
/// fallback and is only used when the device rejects 16 kHz.
pub struct LinearResampler {
    ratio: f64,
    position: f64,
    previous: f32,
    has_previous: bool,
}

impl LinearResampler {
    pub fn new(from_rate: u32, to_rate: u32) -> Self {
        Self {
            ratio: from_rate as f64 / to_rate as f64,
            position: 0.0,
            previous: 0.0,
            has_previous: false,
        }
    }

    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        if (self.ratio - 1.0).abs() < f64::EPSILON {
            return input.to_vec();
        }
        let mut out = Vec::with_capacity((input.len() as f64 / self.ratio) as usize + 2);
        let mut pos = self.position;
        while pos < input.len() as f64 {
            let index = pos.floor() as usize;
            let frac = pos - index as f64;
            let a = if index == 0 {
                if self.has_previous {
                    self.previous
                } else {
                    input[0]
                }
            } else {
                input[index - 1]
            };
            let b = input.get(index).copied().unwrap_or(a);
            out.push(a + (b - a) * frac as f32);
            pos += self.ratio;
        }
        self.position = pos - input.len() as f64;
        if let Some(last) = input.last() {
            self.previous = *last;
            self.has_previous = true;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_round_trip_and_duration() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.wav");
        let samples: Vec<i16> = (0..1600).map(|i| (i % 100) as i16).collect();
        write_mono_i16(&path, 16_000, &samples).unwrap();
        assert!((wav_duration(&path).unwrap() - 0.1).abs() < 1e-9);
        let (loaded, rate) = load_wav_float32(&path).unwrap();
        assert_eq!(rate, 16_000);
        assert_eq!(loaded.len(), samples.len());
        assert!((loaded[50] - 50.0 / 32768.0).abs() < 1e-6);
    }

    #[test]
    fn eight_bit_and_24_bit_wavs_are_scaled_correctly() {
        let dir = tempfile::tempdir().unwrap();
        let eight = dir.path().join("eight.wav");
        let mut writer = hound::WavWriter::create(
            &eight,
            hound::WavSpec {
                channels: 1,
                sample_rate: 16_000,
                bits_per_sample: 8,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        writer.write_sample(-128_i8).unwrap();
        writer.write_sample(0_i8).unwrap();
        writer.write_sample(127_i8).unwrap();
        writer.finalize().unwrap();
        let (loaded, _) = load_wav_float32(&eight).unwrap();
        assert!((loaded[0] + 1.0).abs() < 1e-6);
        assert_eq!(loaded[1], 0.0);
        assert!((loaded[2] - 127.0 / 128.0).abs() < 1e-6);

        let twenty_four = dir.path().join("24.wav");
        let mut writer = hound::WavWriter::create(
            &twenty_four,
            hound::WavSpec {
                channels: 1,
                sample_rate: 16_000,
                bits_per_sample: 24,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        writer.write_sample(-8_388_608_i32).unwrap();
        writer.write_sample(0_i32).unwrap();
        writer.write_sample(4_194_304_i32).unwrap();
        writer.finalize().unwrap();
        let (loaded, _) = load_wav_float32(&twenty_four).unwrap();
        assert!((loaded[0] + 1.0).abs() < 1e-6);
        assert_eq!(loaded[1], 0.0);
        assert!((loaded[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn resampler_roughly_preserves_length() {
        let mut r = LinearResampler::new(48_000, 16_000);
        let input: Vec<f32> = (0..4800).map(|i| (i as f32 / 4800.0).sin()).collect();
        let out = r.process(&input);
        assert!((out.len() as i64 - 1600).abs() <= 2, "{}", out.len());
    }
}
