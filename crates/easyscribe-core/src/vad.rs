//! Shared VAD tuning and segment-bound helpers, ported from `src/vad.py`.
//! The actual Silero model runs through `sherpa_engine`; the padding rules
//! live here so file and live mode can never drift apart.

use crate::config::{VAD_CHUNK_SAMPLES, VAD_SAMPLE_RATE};

pub const VAD_THRESHOLD: f32 = 0.5;
pub const VAD_MIN_SILENCE_SEC: f32 = 0.5;
pub const VAD_MIN_SPEECH_SEC: f32 = 0.25;
pub const VAD_PAD_SEC: f32 = 0.2;
pub const VAD_MAX_SPEECH_SEC: f32 = 20.0;
pub const PAD_SAMPLES: usize = (VAD_PAD_SEC * VAD_SAMPLE_RATE as f32) as usize;

const _: () = assert!(
    VAD_PAD_SEC < VAD_MIN_SILENCE_SEC / 2.0,
    "VAD_PAD_SEC must stay below half of VAD_MIN_SILENCE_SEC so padding on adjacent segments can never overlap"
);

/// Expand each `(start, end)` by `PAD_SAMPLES` on both sides, clamped so
/// padded segments never overlap each other or leave `[0, total_samples)`.
pub fn pad_and_clamp(bounds: &[(usize, usize)], total_samples: usize) -> Vec<(usize, usize)> {
    let mut padded = Vec::with_capacity(bounds.len());
    for (i, (b_start, b_end)) in bounds.iter().copied().enumerate() {
        let prev_end = if i > 0 { bounds[i - 1].1 } else { 0 };
        let next_start = if i + 1 < bounds.len() { bounds[i + 1].0 } else { total_samples };
        let c_start = prev_end.max(b_start.saturating_sub(PAD_SAMPLES));
        let c_end = next_start.min(b_end.saturating_add(PAD_SAMPLES)).min(total_samples);
        padded.push((c_start, c_end));
    }
    padded
}

/// Prepend up to `PAD_SAMPLES` of preceding audio from a rolling `history`
/// buffer, never reaching past the previous segment or the oldest sample.
pub fn pad_live_segment(
    history: &[f32],
    history_start: usize,
    prev_segment_end: usize,
    seg_start: usize,
    seg_samples: &[f32],
) -> Vec<f32> {
    let pad_start = prev_segment_end.max(seg_start.saturating_sub(PAD_SAMPLES)).max(history_start);
    if pad_start >= seg_start {
        return seg_samples.to_vec();
    }
    let lead_from = pad_start.saturating_sub(history_start);
    let lead_to = seg_start.saturating_sub(history_start).min(history.len());
    let mut out = Vec::with_capacity(lead_to.saturating_sub(lead_from) + seg_samples.len());
    if lead_from < lead_to {
        out.extend_from_slice(&history[lead_from..lead_to]);
    }
    out.extend_from_slice(seg_samples);
    out
}

/// Split a long buffer into the 512-sample windows Silero expects.
pub fn windows(samples: &[f32]) -> impl Iterator<Item = &[f32]> {
    samples.chunks(VAD_CHUNK_SAMPLES)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pads_and_clamps_at_edges() {
        let bounds = [(10_000, 20_000), (25_000, 30_000)];
        let out = pad_and_clamp(&bounds, 32_000);
        assert_eq!(out[0].0, 10_000 - PAD_SAMPLES);
        assert_eq!(out[0].1, 20_000 + PAD_SAMPLES);
        assert_eq!(out[1].0, 25_000 - PAD_SAMPLES);
        assert_eq!(out[1].1, 32_000);

        let out = pad_and_clamp(&[(0, 100)], 100);
        assert_eq!(out, vec![(0, 100)]);
    }

    #[test]
    fn adjacent_segments_never_overlap() {
        let bounds = [(1000, 2000), (9000, 10_000)];
        let out = pad_and_clamp(&bounds, 12_000);
        assert!(out[0].1 <= out[1].0, "{out:?}");
    }

    #[test]
    fn padding_live_never_before_history_or_previous_turn() {
        let history: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let seg = vec![9.0, 10.0];
        let out = pad_live_segment(&history, 500, 600, 900, &seg);
        assert_eq!(out.len(), 300 + 2);
        assert_eq!(out[0], 100.0);
        assert_eq!(&out[out.len() - 2..], &[9.0, 10.0]);
    }
}
