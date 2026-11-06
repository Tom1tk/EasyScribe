//! Remembered choices, stored as JSON in `recordings/settings.json`.
//! A missing or broken file means "use the defaults"; unknown keys survive a
//! save, exactly as the Python implementation preserves them.

use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

use crate::{Error, Result};

const RECORDINGS_DIR_KEY: &str = "recordings_dir";
const SPEAKER_SPEED_KEY: &str = "speaker_speed_factor";
const GPU_DEVICE_KEY: &str = "whisper_gpu_device";
const GPU_DISABLED_KEY: &str = "whisper_gpu_disabled_for";

#[derive(Debug, Clone)]
pub struct Settings {
    path: PathBuf,
    data: Map<String, Value>,
}

impl Settings {
    pub fn load(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let data = match std::fs::read_to_string(&path) {
            Ok(text) => match serde_json::from_str::<Value>(&text) {
                Ok(Value::Object(map)) => map,
                Ok(_) => {
                    backup_broken(&path, &text);
                    Map::new()
                }
                Err(e) => {
                    log::warn!("Could not read settings, using defaults: {e}");
                    backup_broken(&path, &text);
                    Map::new()
                }
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Map::new(),
            Err(e) => {
                log::warn!("Could not read settings, using defaults: {e}");
                Map::new()
            }
        };
        Self { path, data }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The chosen recordings folder, or `None` when unset/missing.
    pub fn load_recordings_dir(&self) -> Option<PathBuf> {
        let value = self.data.get(RECORDINGS_DIR_KEY)?.as_str()?;
        if value.is_empty() {
            return None;
        }
        let folder = PathBuf::from(value);
        if folder.is_dir() {
            Some(folder)
        } else {
            log::warn!(
                "Chosen recordings folder is missing, using the default: {}",
                folder.display()
            );
            None
        }
    }

    pub fn save_recordings_dir(&mut self, folder: Option<&Path>) -> Result<()> {
        self.save(
            RECORDINGS_DIR_KEY,
            folder.map(|f| Value::String(f.display().to_string())),
        )
    }

    /// How much slower (>1) or faster (<1) than expected this PC found speakers.
    pub fn load_speaker_speed(&self) -> Option<f64> {
        self.data
            .get(SPEAKER_SPEED_KEY)?
            .as_f64()
            .filter(|v| *v > 0.0)
    }

    pub fn save_speaker_speed(&mut self, factor: f64) -> Result<()> {
        let rounded = (factor * 1000.0).round_ties_even() / 1000.0;
        self.save(SPEAKER_SPEED_KEY, Some(Value::from(rounded)))
    }

    /// The GPU index chosen for whisper.cpp, when the user has one.
    pub fn load_gpu_device(&self) -> Option<i32> {
        self.data
            .get(GPU_DEVICE_KEY)?
            .as_i64()
            .and_then(|v| i32::try_from(v).ok())
            .filter(|v| *v >= 0)
    }

    pub fn save_gpu_device(&mut self, device: Option<i32>) -> Result<()> {
        self.save(GPU_DEVICE_KEY, device.map(Value::from))
    }

    /// The GPU identity that failed before, so later runs start on the CPU.
    pub fn load_gpu_disabled_for(&self) -> Option<String> {
        self.data
            .get(GPU_DISABLED_KEY)?
            .as_str()
            .filter(|v| !v.is_empty())
            .map(str::to_owned)
    }

    pub fn save_gpu_disabled_for(&mut self, identity: Option<&str>) -> Result<()> {
        self.save(
            GPU_DISABLED_KEY,
            identity.map(|v| Value::String(v.to_owned())),
        )
    }

    fn save(&mut self, key: &str, value: Option<Value>) -> Result<()> {
        // Workers write other keys through their own `Settings` instance
        // (speaker speed, a failed GPU). Reload before merging so a folder
        // change cannot clobber those values, matching `settings._save`.
        let mut data = std::fs::read_to_string(&self.path)
            .ok()
            .and_then(|text| serde_json::from_str::<Value>(&text).ok())
            .and_then(|value| match value {
                Value::Object(map) => Some(map),
                _ => None,
            })
            .unwrap_or_default();
        data.remove(key);
        if let Some(value) = value {
            data.insert(key.to_string(), value);
        }
        self.data = data.clone();
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let text = serde_json::to_string_pretty(&Value::Object(data))?;
        let temp = self
            .path
            .with_extension(format!("json.{}.tmp", uuid::Uuid::new_v4().simple()));
        let write = (|| -> std::io::Result<()> {
            use std::io::Write;
            let mut file = std::fs::File::create(&temp)?;
            file.write_all(text.as_bytes())?;
            file.sync_all()?;
            drop(file);
            std::fs::rename(&temp, &self.path)
        })();
        if let Err(e) = write {
            let _ = std::fs::remove_file(&temp);
            log::warn!("Could not save settings: {e}");
            return Err(Error::Io(e));
        }
        Ok(())
    }
}

/// Keep a malformed settings file beside the original before defaults are
/// written over it. The audit's "settings recovery" improvement.
fn backup_broken(path: &Path, _text: &str) {
    let stamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
    let backup = path.with_file_name(format!(
        "{}.broken-{stamp}",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("settings.json")
    ));
    match std::fs::rename(path, &backup) {
        Ok(()) => log::warn!("Moved unreadable settings to {}", backup.display()),
        Err(e) => log::warn!("Could not back up unreadable settings: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_file_means_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let s = Settings::load(dir.path().join("settings.json"));
        assert_eq!(s.load_recordings_dir(), None);
        assert_eq!(s.load_speaker_speed(), None);
    }

    #[test]
    fn round_trips_recordings_and_speaker_speed() {
        let dir = tempfile::tempdir().unwrap();
        let chosen = dir.path().join("Recordings");
        std::fs::create_dir_all(&chosen).unwrap();
        let path = dir.path().join("settings.json");
        let mut s = Settings::load(&path);
        s.save_recordings_dir(Some(&chosen)).unwrap();
        s.save_speaker_speed(1.23456).unwrap();

        let s2 = Settings::load(&path);
        assert_eq!(s2.load_recordings_dir().as_deref(), Some(chosen.as_path()));
        assert_eq!(s2.load_speaker_speed(), Some(1.235));
    }

    #[test]
    fn missing_chosen_folder_falls_back_to_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        std::fs::write(&path, r#"{"recordings_dir":"/definitely/not/here"}"#).unwrap();
        assert_eq!(Settings::load(&path).load_recordings_dir(), None);
    }

    #[test]
    fn remembers_a_failed_gpu_and_a_chosen_device() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let mut s = Settings::load(&path);
        s.save_gpu_device(Some(1)).unwrap();
        s.save_gpu_disabled_for(Some("GPU A|8589934592")).unwrap();
        let s2 = Settings::load(&path);
        assert_eq!(s2.load_gpu_device(), Some(1));
        assert_eq!(
            s2.load_gpu_disabled_for().as_deref(),
            Some("GPU A|8589934592")
        );
        let mut s3 = s2;
        s3.save_gpu_disabled_for(None).unwrap();
        assert_eq!(Settings::load(&path).load_gpu_disabled_for(), None);
    }

    #[test]
    fn a_stale_instance_does_not_clobber_worker_keys() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let mut stale = Settings::load(&path);
        let mut worker = Settings::load(&path);
        worker.save_speaker_speed(1.5).unwrap();
        stale.save_recordings_dir(None).unwrap();
        assert_eq!(Settings::load(&path).load_speaker_speed(), Some(1.5));
    }

    #[test]
    fn preserves_unknown_keys() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        std::fs::write(&path, r#"{"other": 7, "recordings_dir": null}"#).unwrap();
        let mut s = Settings::load(&path);
        s.save_speaker_speed(2.0).unwrap();
        let value: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(value["other"], 7);
    }
}
