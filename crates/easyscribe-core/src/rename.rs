//! The Name box's file operations, ported from `src/rename.py`.
//!
//! A recording's files share its stem (`<stem>.wav`, `<stem>.txt`,
//! `<stem> (best quality).txt`, ...). They are renamed together: either every
//! file is renamed or none is.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use regex::Regex;

pub const MAX_NAME: usize = 120;

fn bad_chars() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| Regex::new(r#"[\\/:*?"<>|\x00-\x1f]"#).expect("valid name regex"))
}

fn reserved() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)^(con|prn|aux|nul|com[1-9]|lpt[1-9])$").expect("valid reserved regex")
    })
}

/// `""` if `name` can be a Windows file name, else what is wrong with it.
pub fn check_name(name: &str) -> &'static str {
    if name.is_empty() {
        return "Type a name first.";
    }
    if bad_chars().is_match(name) {
        return "A name cannot have these characters: \\ / : * ? \" < > |";
    }
    if reserved().is_match(name) || name.ends_with('.') {
        return "Windows does not allow this name. Please choose another.";
    }
    if name.chars().count() > MAX_NAME {
        return "Please use a shorter name (at most 120 characters).";
    }
    ""
}

/// The recording and the transcripts made from it, in its folder, sorted.
pub fn recording_set(audio: &Path) -> Vec<PathBuf> {
    let Some(parent) = audio.parent() else {
        return Vec::new();
    };
    let Some(stem) = audio.file_stem().and_then(|s| s.to_str()) else {
        return Vec::new();
    };
    let Ok(entries) = std::fs::read_dir(parent) else {
        return Vec::new();
    };
    let mut set: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .filter(|p| {
            let ext = p
                .extension()
                .and_then(|e| e.to_str())
                .map(str::to_ascii_lowercase);
            let matches_ext = matches!(ext.as_deref(), Some("wav" | "txt"));
            let file_stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            matches_ext && (file_stem == stem || file_stem.starts_with(&format!("{stem} (")))
        })
        .collect();
    set.sort();
    set
}

/// Windows paths are case-insensitive, so `Meeting.wav` -> `meeting.wav` is a
/// rename of the same file, not a collision with an existing one.
fn same_file_name(a: &Path, b: &Path) -> bool {
    #[cfg(windows)]
    {
        a.as_os_str()
            .to_string_lossy()
            .eq_ignore_ascii_case(&b.as_os_str().to_string_lossy())
    }
    #[cfg(not(windows))]
    {
        a == b
    }
}

/// Rename each path's leading `old_stem` to `new_stem`.
///
/// Returns the old→new map. A taken new name changes nothing and returns
/// `FileExistsError`. Any failure undoes the renames already done and returns
/// the OS error.
pub fn rename_all(
    paths: &[PathBuf],
    old_stem: &str,
    new_stem: &str,
) -> Result<BTreeMap<PathBuf, PathBuf>, RenameError> {
    let mut plan: Vec<(PathBuf, PathBuf)> = Vec::with_capacity(paths.len());
    for path in paths {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| RenameError::Invalid(path.display().to_string()))?;
        if !name.starts_with(old_stem) {
            return Err(RenameError::Invalid(name.to_owned()));
        }
        let rest = &name[old_stem.len()..];
        let new_path = path.with_file_name(format!("{new_stem}{rest}"));
        plan.push((path.clone(), new_path));
    }

    if let Some((_, taken)) = plan
        .iter()
        .find(|(old, new)| new != old && new.exists() && !same_file_name(old, new))
    {
        return Err(RenameError::Exists(
            taken
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default()
                .to_owned(),
        ));
    }

    let mut done: Vec<(PathBuf, PathBuf)> = Vec::new();
    for (old, new) in &plan {
        if new != old
            && let Err(e) = std::fs::rename(old, new)
        {
            for (done_old, done_new) in done.iter().rev() {
                if done_new != done_old {
                    let _ = std::fs::rename(done_new, done_old);
                }
            }
            return Err(RenameError::Io(e));
        }
        done.push((old.clone(), new.clone()));
    }

    Ok(plan.into_iter().collect())
}

#[derive(Debug, thiserror::Error)]
pub enum RenameError {
    #[error("a file named \"{0}\" already exists")]
    Exists(String),
    #[error("could not rename {0}")]
    Invalid(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    const STEM: &str = "recording_20261021_143207";
    const SET: [&str; 4] = [
        "recording_20261021_143207.wav",
        "recording_20261021_143207.txt",
        "recording_20261021_143207 (best quality).txt",
        "recording_20261021_143207 (best quality) (speakers).txt",
    ];
    const OTHERS: [&str; 3] = [
        "recording_20261021_143207_2.wav",
        "recording_20261021_150000.txt",
        "recording_20261021_143207.json",
    ];

    fn folder() -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("temp dir");
        for name in SET.iter().chain(OTHERS.iter()) {
            std::fs::write(dir.path().join(name), name).expect("write fixture");
        }
        dir
    }

    #[test]
    fn accepts_plain_names() {
        assert_eq!(check_name("Team meeting 21 Oct"), "");
    }

    #[test]
    fn rejects_windows_problems() {
        for bad in [
            "",
            "a/b",
            "a:b",
            "say \"hi\"",
            "what?",
            "CON",
            "lpt1",
            "name.",
        ] {
            assert!(!check_name(bad).is_empty(), "{bad:?} should be rejected");
        }
        assert!(!check_name(&"x".repeat(200)).is_empty());
    }

    #[test]
    fn finds_the_whole_recording_set() {
        let dir = folder();
        let files = recording_set(&dir.path().join(format!("{STEM}.wav")));
        let names: Vec<_> = files
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap())
            .collect();
        let mut expected = SET.to_vec();
        expected.sort();
        assert_eq!(names, expected);
    }

    #[test]
    fn renames_all_files_together() {
        let dir = folder();
        let files = recording_set(&dir.path().join(format!("{STEM}.wav")));
        rename_all(&files, STEM, "Team meeting").expect("rename");
        let names: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert!(names.contains(&"Team meeting (best quality) (speakers).txt".to_string()));
        assert!(names.contains(&"Team meeting.wav".to_string()));
        for old in SET {
            assert!(!names.contains(&old.to_string()));
        }
        for other in OTHERS {
            assert!(names.contains(&other.to_string()));
        }
    }

    #[test]
    fn taken_name_changes_nothing() {
        let dir = folder();
        std::fs::write(dir.path().join("Team meeting.txt"), "other").unwrap();
        let files = recording_set(&dir.path().join(format!("{STEM}.wav")));
        assert!(matches!(
            rename_all(&files, STEM, "Team meeting"),
            Err(RenameError::Exists(_))
        ));
        for old in SET {
            assert!(dir.path().join(old).exists());
        }
        assert_eq!(
            std::fs::read_to_string(dir.path().join("Team meeting.txt")).unwrap(),
            "other"
        );
    }

    #[test]
    fn transcript_only_rename() {
        let dir = folder();
        let t = dir.path().join(format!("{STEM}.txt"));
        rename_all(&[t], STEM, "Notes").unwrap();
        assert!(dir.path().join("Notes.txt").exists());
        assert!(dir.path().join(format!("{STEM}.wav")).exists());
    }
}
