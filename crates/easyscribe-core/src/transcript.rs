//! Transcript formatting and speaker assignment. This is a line-for-line port
//! of the pure helpers in Python's `transcriber.py` and `diarizer.py`.

use std::collections::HashMap;
use std::path::Path;

use crate::model::{AssignedSegment, Segment, Turn};
use crate::{Error, Result};

pub const BLOCK_GAP_SEC: f64 = 2.0;

/// `HH:MM:SS`, as used in transcript timestamps.
pub fn format_hms(seconds: f64) -> String {
    let total = seconds.max(0.0) as u64;
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    format!("{h:02}:{m:02}:{s:02}")
}

/// Join raw segments into the plain transcript. Without timestamps every
/// segment is joined with one space. With timestamps, a new block starts when
/// the silence before a segment is at least 2.0 s; blocks are separated by a
/// blank line.
pub fn build_plain_transcript(segments: &[Segment], add_timestamps: bool) -> String {
    if segments.is_empty() {
        return String::new();
    }
    if !add_timestamps {
        return segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
    }

    let mut blocks: Vec<(f64, Vec<&str>)> = Vec::new();
    let mut block_start = segments[0].start;
    let mut block_texts: Vec<&str> = Vec::new();
    let mut prev_end = 0.0;

    for seg in segments {
        if !block_texts.is_empty() && (seg.start - prev_end) >= BLOCK_GAP_SEC {
            blocks.push((block_start, std::mem::take(&mut block_texts)));
            block_start = seg.start;
        }
        block_texts.push(seg.text.as_str());
        prev_end = seg.end;
    }
    if !block_texts.is_empty() {
        blocks.push((block_start, block_texts));
    }

    blocks
        .into_iter()
        .map(|(ts, texts)| format!("[{}]\n{}", format_hms(ts), texts.join(" ")))
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Join diarized segments into a speaker-labelled transcript, following the
/// same block rules as the Python implementation.
pub fn build_diarized_transcript(
    assigned: &[AssignedSegment],
    speaker_map: &HashMap<String, String>,
    add_timestamps: bool,
) -> String {
    if assigned.is_empty() {
        return String::new();
    }
    let label = |raw: &str| {
        speaker_map
            .get(raw)
            .cloned()
            .unwrap_or_else(|| raw.to_owned())
    };
    let mut parts: Vec<String> = Vec::new();

    if !add_timestamps {
        let mut block_label = label(&assigned[0].speaker);
        let mut block_texts: Vec<&str> = vec![&assigned[0].text];
        for seg in &assigned[1..] {
            let next = label(&seg.speaker);
            if next == block_label {
                block_texts.push(&seg.text);
            } else {
                parts.push(format!("[{block_label}]\n{}", block_texts.join(" ")));
                block_label = next;
                block_texts = vec![&seg.text];
            }
        }
        parts.push(format!("[{block_label}]\n{}", block_texts.join(" ")));
    } else {
        let mut block_label = label(&assigned[0].speaker);
        let mut block_start = assigned[0].start;
        let mut block_texts: Vec<&str> = vec![&assigned[0].text];
        let mut prev_end = assigned[0].end;
        for seg in &assigned[1..] {
            let next = label(&seg.speaker);
            let new_block = next != block_label || (seg.start - prev_end) >= BLOCK_GAP_SEC;
            if new_block {
                parts.push(format!(
                    "[{}] [{block_label}]\n{}",
                    format_hms(block_start),
                    block_texts.join(" ")
                ));
                block_label = next;
                block_start = seg.start;
                block_texts = vec![&seg.text];
            } else {
                block_texts.push(&seg.text);
            }
            prev_end = seg.end;
        }
        parts.push(format!(
            "[{}] [{block_label}]\n{}",
            format_hms(block_start),
            block_texts.join(" ")
        ));
    }

    parts.join("\n\n")
}

/// Assign each transcript segment to the speaker whose turn overlaps it most.
/// No overlap at all falls back to `SPEAKER_00`, exactly as before.
pub fn assign_speakers(segments: &[Segment], turns: &[Turn]) -> Vec<AssignedSegment> {
    segments
        .iter()
        .map(|seg| {
            let mut best_speaker = "SPEAKER_00".to_string();
            let mut best_overlap = 0.0_f64;
            for turn in turns {
                let overlap = seg.end.min(turn.end) - seg.start.max(turn.start);
                if overlap > best_overlap {
                    best_overlap = overlap;
                    best_speaker = turn.speaker.clone();
                }
            }
            AssignedSegment {
                speaker: best_speaker,
                text: seg.text.clone(),
                start: seg.start,
                end: seg.end,
            }
        })
        .collect()
}

/// Seconds covered by the transcript lines, overlaps counted once.
pub fn speech_seconds(segments: &[Segment]) -> f64 {
    let mut sorted: Vec<&Segment> = segments.iter().collect();
    sorted.sort_by(|a, b| a.start.total_cmp(&b.start));
    let mut total = 0.0;
    let mut reach = 0.0;
    for seg in sorted {
        let start = seg.start.max(reach);
        if seg.end > start {
            total += seg.end - start;
            reach = seg.end;
        }
    }
    total
}

/// Write a transcript with a single trailing newline.
pub fn write_transcript(output_path: &Path, text: &str) -> Result<()> {
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut payload = String::with_capacity(text.len() + 1);
    payload.push_str(text);
    payload.push('\n');
    std::fs::write(output_path, payload).map_err(|e| {
        if e.kind() == std::io::ErrorKind::PermissionDenied {
            Error::WriteTranscript(format!(
                "Cannot write to output file.\nThe file may be open in another program: {}\n\n{e}",
                output_path.display()
            ))
        } else {
            Error::WriteTranscript(format!(
                "Failed to write transcript: {}\n\n{e}",
                output_path.display()
            ))
        }
    })
}

/// Extract up to `max_duration` seconds of a speaker's longest turn from a WAV
/// file. Returns `false` if the speaker has no turn or the WAV cannot be read.
pub fn extract_speaker_clip(
    audio_path: &Path,
    turns: &[Turn],
    speaker: &str,
    out_path: &Path,
    max_duration: f64,
) -> bool {
    let best = turns
        .iter()
        .filter(|t| t.speaker == speaker)
        .map(|t| (t.start, t.end))
        .max_by(|a, b| (a.1 - a.0).total_cmp(&(b.1 - b.0)));
    let Some((best_start, best_end)) = best else {
        return false;
    };
    let clip_end = best_end.min(best_start + max_duration);
    match write_wav_slice(audio_path, best_start, clip_end, out_path) {
        Ok(()) => true,
        Err(e) => {
            log::warn!("Could not extract speaker clip for {speaker}: {e}");
            false
        }
    }
}

/// Copy a time range of an existing WAV verbatim (keeping channels, width and
/// rate), as Python's `_extract_speaker_clip` does. Stream the requested
/// frames instead of loading the whole recording into memory.
fn write_wav_slice(input: &Path, start: f64, end: f64, output: &Path) -> Result<()> {
    let mut reader = hound::WavReader::open(input)?;
    let spec = reader.spec();
    let sr = spec.sample_rate as f64;
    let start_frame = (start.max(0.0) * sr) as u64;
    let frames = ((end - start) * sr).max(0.0) as u64;
    let channels = spec.channels as u64;
    let offset = usize::try_from(start_frame.saturating_mul(channels)).unwrap_or(usize::MAX);
    let count = usize::try_from(frames.saturating_mul(channels)).unwrap_or(usize::MAX);
    let mut writer = hound::WavWriter::create(output, spec)?;

    match spec.sample_format {
        hound::SampleFormat::Int if spec.bits_per_sample <= 16 => {
            for sample in reader.samples::<i16>().skip(offset).take(count) {
                writer.write_sample(sample?)?;
            }
        }
        hound::SampleFormat::Int => {
            for sample in reader.samples::<i32>().skip(offset).take(count) {
                writer.write_sample(sample?)?;
            }
        }
        hound::SampleFormat::Float => {
            for sample in reader.samples::<f32>().skip(offset).take(count) {
                writer.write_sample(sample?)?;
            }
        }
    }
    writer.finalize()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(start: f64, end: f64, text: &str) -> Segment {
        Segment::new(start, end, text)
    }

    #[test]
    fn plain_empty() {
        assert_eq!(build_plain_transcript(&[], false), "");
        assert_eq!(build_plain_transcript(&[], true), "");
    }

    #[test]
    fn plain_no_timestamps_joins_with_space() {
        let segs = [seg(0.0, 1.0, "Hello"), seg(1.0, 2.0, "world.")];
        assert_eq!(build_plain_transcript(&segs, false), "Hello world.");
    }

    #[test]
    fn plain_timestamps_single_block() {
        let segs = [seg(0.0, 1.0, "Hello"), seg(1.5, 2.5, "world.")];
        assert_eq!(
            build_plain_transcript(&segs, true),
            "[00:00:00]\nHello world."
        );
    }

    #[test]
    fn plain_timestamps_splits_on_gap() {
        let segs = [seg(0.0, 1.0, "Hello."), seg(10.0, 11.0, "Goodbye.")];
        assert_eq!(
            build_plain_transcript(&segs, true),
            "[00:00:00]\nHello.\n\n[00:00:10]\nGoodbye."
        );
    }

    #[test]
    fn diarized_groups_consecutive_speakers() {
        let assigned = vec![
            AssignedSegment {
                speaker: "SPEAKER_00".into(),
                text: "Hello,".into(),
                start: 0.0,
                end: 1.0,
            },
            AssignedSegment {
                speaker: "SPEAKER_00".into(),
                text: "how are you?".into(),
                start: 1.0,
                end: 2.0,
            },
            AssignedSegment {
                speaker: "SPEAKER_01".into(),
                text: "Fine, thanks.".into(),
                start: 2.0,
                end: 3.0,
            },
        ];
        let map = HashMap::from([
            ("SPEAKER_00".to_string(), "Speaker 1".to_string()),
            ("SPEAKER_01".to_string(), "Speaker 2".to_string()),
        ]);
        assert_eq!(
            build_diarized_transcript(&assigned, &map, false),
            "[Speaker 1]\nHello, how are you?\n\n[Speaker 2]\nFine, thanks."
        );
    }

    #[test]
    fn diarized_timestamps_split_on_speaker_and_gap() {
        let assigned = vec![
            AssignedSegment {
                speaker: "SPEAKER_00".into(),
                text: "Hello.".into(),
                start: 0.0,
                end: 1.0,
            },
            AssignedSegment {
                speaker: "SPEAKER_00".into(),
                text: "Still talking.".into(),
                start: 1.2,
                end: 2.0,
            },
            AssignedSegment {
                speaker: "SPEAKER_01".into(),
                text: "Reply.".into(),
                start: 2.0,
                end: 3.0,
            },
            AssignedSegment {
                speaker: "SPEAKER_01".into(),
                text: "More.".into(),
                start: 10.0,
                end: 11.0,
            },
        ];
        let map = HashMap::from([
            ("SPEAKER_00".to_string(), "Speaker 1".to_string()),
            ("SPEAKER_01".to_string(), "Speaker 2".to_string()),
        ]);
        assert_eq!(
            build_diarized_transcript(&assigned, &map, true),
            "[00:00:00] [Speaker 1]\nHello. Still talking.\n\n\
             [00:00:02] [Speaker 2]\nReply.\n\n\
             [00:00:10] [Speaker 2]\nMore."
        );
    }

    #[test]
    fn diarized_unmapped_speaker_keeps_raw_label() {
        let assigned = vec![AssignedSegment {
            speaker: "SPEAKER_05".into(),
            text: "Hi.".into(),
            start: 0.0,
            end: 1.0,
        }];
        assert_eq!(
            build_diarized_transcript(&assigned, &HashMap::new(), false),
            "[SPEAKER_05]\nHi."
        );
    }

    #[test]
    fn assign_picks_max_overlap() {
        let segs = [seg(0.0, 2.0, "Hello"), seg(5.0, 7.0, "world")];
        let turns = [
            Turn::new(0.0, 1.5, "SPEAKER_00"),
            Turn::new(1.5, 4.0, "SPEAKER_01"),
            Turn::new(4.5, 8.0, "SPEAKER_02"),
        ];
        let assigned = assign_speakers(&segs, &turns);
        assert_eq!(assigned[0].speaker, "SPEAKER_00");
        assert_eq!(assigned[1].speaker, "SPEAKER_02");
    }

    #[test]
    fn assign_without_overlap_defaults_to_first_speaker() {
        let assigned = assign_speakers(
            &[seg(100.0, 101.0, "lonely")],
            &[Turn::new(0.0, 1.0, "SPEAKER_03")],
        );
        assert_eq!(assigned[0].speaker, "SPEAKER_00");
    }

    #[test]
    fn hms_rounds_down_and_counts_hours() {
        assert_eq!(format_hms(0.0), "00:00:00");
        assert_eq!(format_hms(3661.9), "01:01:01");
    }
}
