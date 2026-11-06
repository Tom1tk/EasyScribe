//! Presentation helpers and output naming rules shared by the UI and tests.
//!
//! These mirror the small formatting functions in `src/gui.py` exactly
//! (`_shorten`, `_fmt_size`, `_fmt_minutes`, `_fmt_took`, `_plural`, timer),
//! so the visible copy cannot drift from the oracle.

use std::path::{Path, PathBuf};

use crate::config::SUPPORTED_EXTENSIONS;

/// `_shorten`: keep the beginning and end, with a single-character ellipsis.
pub fn shorten(text: &str, limit: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= limit {
        return text.to_owned();
    }
    let keep = limit.saturating_sub(1);
    let head = keep * 2 / 3;
    let tail = keep - head;
    let mut out = String::with_capacity(limit + 3);
    out.extend(chars.iter().take(head));
    out.push('…');
    out.extend(chars.iter().skip(chars.len() - tail));
    out
}

/// `_fmt_size`: no decimals for bytes, one decimal for KB/MB/GB.
pub fn fmt_size(bytes: u64) -> String {
    let mut size = bytes as f64;
    for unit in ["bytes", "KB", "MB", "GB"] {
        if size < 1024.0 || unit == "GB" {
            return if unit == "bytes" {
                format!("{size:.0} {unit}")
            } else {
                format!("{size:.1} {unit}")
            };
        }
        size /= 1024.0;
    }
    format!("{size:.1} GB")
}

/// `_fmt_minutes`: `under 1 min` or the rounded minute count.
pub fn fmt_minutes(seconds: f64) -> String {
    if seconds < 60.0 {
        "under 1 min".into()
    } else {
        format!("{} min", (seconds / 60.0).round() as i64)
    }
}

/// `_fmt_took`: `6 min 12 s` or `12 s`.
pub fn fmt_took(seconds: f64) -> String {
    let total = seconds.max(0.0).round() as i64;
    let minutes = total / 60;
    let secs = total % 60;
    if minutes > 0 {
        format!("{minutes} min {secs} s")
    } else {
        format!("{secs} s")
    }
}

/// Recording timer: `MM:SS`, or `H:MM:SS` past the hour.
pub fn fmt_timer(seconds: f64) -> String {
    let total = seconds.max(0.0) as u64;
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m:02}:{s:02}")
    }
}

/// `_plural(n, word)`.
pub fn plural(n: usize, word: &str) -> String {
    if n == 1 {
        format!("{n} {word}")
    } else {
        format!("{n} {word}s")
    }
}

/// The estimate is lowercase in `diarizer`; the Find card capitalises it.
pub fn capitalize_first(text: &str) -> String {
    let mut chars = text.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

/// True when the file has one of the supported media suffixes (case-insensitive).
pub fn is_supported_media(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| SUPPORTED_EXTENSIONS.iter().any(|x| x.eq_ignore_ascii_case(e)))
        .unwrap_or(false)
}

/// The uppercase extension list used in the skipped-files note.
pub fn supported_extensions_display() -> String {
    let mut list: Vec<String> = SUPPORTED_EXTENSIONS.iter().map(|e| e.to_ascii_uppercase()).collect();
    list.sort();
    list.join(", ")
}

/// Ordinary Files-mode output: `<input stem>.txt` in `folder`, else beside input.
pub fn transcript_output(input: &Path, folder: Option<&Path>) -> PathBuf {
    let stem = input.file_stem().unwrap_or_default();
    let name = format!("{}.txt", stem.to_string_lossy());
    match folder {
        Some(folder) => folder.join(name),
        None => input.with_file_name(name),
    }
}

/// Best-quality recording output, with exactly one collision suffix level.
pub fn best_quality_output(audio: &Path) -> PathBuf {
    let base = audio.with_extension("txt");
    if base.exists() {
        let stem = audio.file_stem().unwrap_or_default().to_string_lossy();
        audio.with_file_name(format!("{stem} (best quality).txt"))
    } else {
        base
    }
}

/// Find-speakers output: `<transcript stem> (speakers).txt`, overwriting a copy.
pub fn speakers_output(transcript: &Path) -> PathBuf {
    let stem = transcript.file_stem().unwrap_or_default().to_string_lossy();
    transcript.with_file_name(format!("{stem} (speakers).txt"))
}

/// Up to three shortened failure/skip names, then `and N more`.
pub fn names_summary(paths: &[&Path]) -> String {
    let names: Vec<String> = paths
        .iter()
        .take(3)
        .map(|p| {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or_else(|| p.to_str().unwrap_or("?"));
            shorten(name, 40)
        })
        .collect();
    let mut text = names.join(", ");
    if paths.len() > 3 {
        text.push_str(&format!(" and {} more", paths.len() - 3));
    }
    text
}

/// Free bytes on the volume containing `path` (walks up to the nearest existing
/// ancestor). `None` when the platform cannot answer.
pub fn free_space(path: &Path) -> Option<u64> {
    let mut current = path;
    loop {
        if current.exists() {
            return fs2::available_space(current).ok();
        }
        current = current.parent()?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shortening_keeps_both_ends() {
        assert_eq!(shorten("short", 10), "short");
        let out = shorten("abcdefghijklmnopqrstuvwxyz", 10);
        assert_eq!(out.chars().count(), 10);
        assert!(out.starts_with("abcdef") && out.ends_with("xyz"), "{out}");
        assert!(out.contains('…'));
    }

    #[test]
    fn sizes_match_python() {
        assert_eq!(fmt_size(500), "500 bytes");
        assert_eq!(fmt_size(1536), "1.5 KB");
        assert_eq!(fmt_size(29_569_024), "28.2 MB");
        assert_eq!(fmt_size(145_000_000), "138.3 MB");
    }

    #[test]
    fn time_helpers_match_python() {
        assert_eq!(fmt_minutes(59.0), "under 1 min");
        assert_eq!(fmt_minutes(27.0 * 60.0), "27 min");
        assert_eq!(fmt_took(372.4), "6 min 12 s");
        assert_eq!(fmt_took(7.2), "7 s");
        assert_eq!(fmt_timer(5.0), "00:05");
        assert_eq!(fmt_timer(65.0), "01:05");
        assert_eq!(fmt_timer(3661.0), "1:01:01");
    }

    #[test]
    fn output_names_follow_the_rules() {
        let base = Path::new("/tmp/Team meeting.m4a");
        assert_eq!(transcript_output(base, None), Path::new("/tmp/Team meeting.txt"));
        assert_eq!(
            transcript_output(base, Some(Path::new("/out"))),
            Path::new("/out/Team meeting.txt")
        );
        assert_eq!(
            speakers_output(Path::new("/out/Team meeting.txt")),
            Path::new("/out/Team meeting (speakers).txt")
        );
    }

    #[test]
    fn supported_extensions_are_case_insensitive() {
        assert!(is_supported_media(Path::new("a.MP4")));
        assert!(is_supported_media(Path::new("a.flac")));
        assert!(!is_supported_media(Path::new("a.txt")));
        assert!(supported_extensions_display().starts_with("AAC, AVI"));
    }

    #[test]
    fn failure_names_limit_to_three() {
        let paths: Vec<PathBuf> = (0..5).map(|i| PathBuf::from(format!("file{i}.wav"))).collect();
        let refs: Vec<&Path> = paths.iter().map(PathBuf::as_path).collect();
        assert_eq!(names_summary(&refs), "file0.wav, file1.wav, file2.wav and 2 more");
    }
}
