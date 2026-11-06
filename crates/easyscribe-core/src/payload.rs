//! The single-exe payload: a compressed archive appended to the executable,
//! extracted once into a per-user runtime cache.
//!
//! This replaces the Python installer/launcher completely. The exe is a normal
//! program plus an overlay; Windows ignores the overlay, the app reads its own
//! footer and reuses the extracted runtime on every later launch.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{Error, Result};

pub const FOOTER_MAGIC: &[u8; 8] = b"ESCFTR01";
pub const TRAILER_MAGIC: &[u8; 8] = b"ESCEOF01";
pub const FOOTER_SIZE: usize = 128;
pub const CONTAINER_VERSION: u16 = 1;
pub const RUNTIME_MARKER: &str = ".complete";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Footer {
    pub version: u16,
    pub flags: u16,
    pub payload_offset: u64,
    pub payload_size: u64,
    pub manifest_offset: u64,
    pub manifest_size: u64,
    pub payload_sha256: [u8; 32],
    pub app_version: String,
    pub build_id: String,
}

impl Footer {
    pub fn to_bytes(&self) -> [u8; FOOTER_SIZE] {
        let mut bytes = [0_u8; FOOTER_SIZE];
        bytes[0..8].copy_from_slice(FOOTER_MAGIC);
        bytes[8..10].copy_from_slice(&self.version.to_le_bytes());
        bytes[10..12].copy_from_slice(&self.flags.to_le_bytes());
        bytes[12..16].copy_from_slice(&(FOOTER_SIZE as u32).to_le_bytes());
        bytes[16..24].copy_from_slice(&self.payload_offset.to_le_bytes());
        bytes[24..32].copy_from_slice(&self.payload_size.to_le_bytes());
        bytes[32..40].copy_from_slice(&self.manifest_offset.to_le_bytes());
        bytes[40..48].copy_from_slice(&self.manifest_size.to_le_bytes());
        bytes[48..80].copy_from_slice(&self.payload_sha256);
        copy_padded(&mut bytes[80..96], self.app_version.as_bytes());
        copy_padded(&mut bytes[96..112], self.build_id.as_bytes());
        let crc = crc32fast::hash(&bytes[0..112]);
        bytes[112..116].copy_from_slice(&crc.to_le_bytes());
        bytes[120..128].copy_from_slice(TRAILER_MAGIC);
        bytes
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != FOOTER_SIZE
            || &bytes[0..8] != FOOTER_MAGIC
            || &bytes[120..128] != TRAILER_MAGIC
        {
            return Err(Error::Payload("not an EasyScribe payload".into()));
        }
        let version = u16::from_le_bytes([bytes[8], bytes[9]]);
        let flags = u16::from_le_bytes([bytes[10], bytes[11]]);
        let footer_size = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        if footer_size != FOOTER_SIZE {
            return Err(Error::Payload("unsupported payload footer size".into()));
        }
        if version != CONTAINER_VERSION {
            return Err(Error::Payload(format!(
                "unsupported payload format version {version}"
            )));
        }
        let crc = u32::from_le_bytes(bytes[112..116].try_into().unwrap());
        if crc != crc32fast::hash(&bytes[0..112]) {
            return Err(Error::Payload("payload footer is damaged".into()));
        }
        let payload_offset = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        let payload_size = u64::from_le_bytes(bytes[24..32].try_into().unwrap());
        let manifest_offset = u64::from_le_bytes(bytes[32..40].try_into().unwrap());
        let manifest_size = u64::from_le_bytes(bytes[40..48].try_into().unwrap());
        if payload_size == 0 || manifest_size == 0 {
            return Err(Error::Payload("payload footer has empty sizes".into()));
        }
        if manifest_offset
            .checked_add(manifest_size)
            .is_none_or(|end| end > payload_size)
        {
            return Err(Error::Payload(
                "payload manifest is outside the archive".into(),
            ));
        }
        let mut hash = [0_u8; 32];
        hash.copy_from_slice(&bytes[48..80]);
        let app_version = read_padded(&bytes[80..96]);
        let build_id = read_padded(&bytes[96..112]);
        Ok(Self {
            version,
            flags,
            payload_offset,
            payload_size,
            manifest_offset,
            manifest_size,
            payload_sha256: hash,
            app_version,
            build_id,
        })
    }

    pub fn cache_key(&self) -> String {
        let digest = self
            .payload_sha256
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        format!("v1-{}", &digest[..32])
    }
}

fn copy_padded(target: &mut [u8], text: &[u8]) {
    let len = target.len().min(text.len());
    target[..len].copy_from_slice(&text[..len]);
}

fn read_padded(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|b| *b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

/// Largest signed overlay we search for a footer. Authenticode certificates
/// are normally well under a megabyte.
const SIGNATURE_SCAN: u64 = 1024 * 1024;

/// Read the footer at the end of `exe`, if there is one. A code-signed
/// executable has its certificate appended after the footer, so the footer is
/// not necessarily the final 128 bytes.
pub fn read_footer(exe: &Path) -> Result<Option<Footer>> {
    let mut file = match File::open(exe) {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(Error::Io(e)),
    };
    let len = file.metadata()?.len();
    if len < FOOTER_SIZE as u64 {
        return Ok(None);
    }
    let tail_len = len.min(SIGNATURE_SCAN);
    let tail_start = len - tail_len;
    file.seek(SeekFrom::Start(tail_start))?;
    let mut tail = vec![0_u8; tail_len as usize];
    file.read_exact(&mut tail)?;

    let mut found_magic = false;
    for pos in (0..=tail.len().saturating_sub(FOOTER_SIZE)).rev() {
        if &tail[pos..pos + 8] != FOOTER_MAGIC {
            continue;
        }
        found_magic = true;
        let footer_end = tail_start + (pos + FOOTER_SIZE) as u64;
        let footer = match Footer::from_bytes(&tail[pos..pos + FOOTER_SIZE]) {
            Ok(footer) => footer,
            Err(_) => continue,
        };
        let expected_end = footer
            .payload_offset
            .checked_add(footer.payload_size)
            .and_then(|end| end.checked_add(FOOTER_SIZE as u64));
        if expected_end == Some(footer_end) {
            return Ok(Some(footer));
        }
    }
    if found_magic {
        return Err(Error::Payload(
            "This EasyScribe file is damaged or incomplete. Download it again.".into(),
        ));
    }
    Ok(None)
}

#[derive(Debug, Clone)]
pub struct Prepared {
    pub runtime_dir: PathBuf,
    pub data_dir: PathBuf,
}

/// Whether this launch will have to unpack the appended payload. Cached
/// launches and development builds return `false` so they go straight to the
/// main window without a splash.
pub fn needs_prepare() -> Result<bool> {
    if std::env::var_os("EASYSCRIBE_PAYLOAD_DIR").is_some() {
        return Ok(false);
    }
    let exe = std::env::current_exe().map_err(Error::Io)?;
    let Some(footer) = read_footer(&exe)? else {
        return Ok(false);
    };
    let data_dir = data_root();
    let runtime = runtime_root(&data_dir).join(footer.cache_key());
    Ok(!cache_complete(&runtime, &footer))
}

/// Resolve the runtime and data directories, extracting the appended payload
/// on first launch. Development can bypass the payload with
/// `EASYSCRIBE_PAYLOAD_DIR`.
pub fn prepare() -> Result<Prepared> {
    prepare_with_progress(|_, _| {})
}

/// [`prepare`] with a coarse progress callback `(stage, fraction)` for a
/// first-run splash. `fraction` is 0..1 across verification and extraction.
pub fn prepare_with_progress<F: FnMut(&str, f32)>(mut progress: F) -> Result<Prepared> {
    let data_dir = data_root();
    if let Some(dir) = std::env::var_os("EASYSCRIBE_PAYLOAD_DIR") {
        return Ok(Prepared {
            runtime_dir: PathBuf::from(dir),
            data_dir,
        });
    }
    let exe = std::env::current_exe().map_err(Error::Io)?;
    let Some(footer) = read_footer(&exe)? else {
        if cfg!(debug_assertions) {
            // Development checkout: models/assets live beside the Cargo project.
            return Ok(Prepared {
                runtime_dir: crate::config::dev_base_dir(),
                data_dir,
            });
        }
        return Err(Error::Payload(
            "This EasyScribe file is damaged or incomplete. Download it again.".into(),
        ));
    };

    let runtime_root = runtime_root(&data_dir);
    let runtime = runtime_root.join(footer.cache_key());
    if cache_complete(&runtime, &footer) {
        cleanup_old_runtimes(&runtime_root, &runtime);
        return Ok(Prepared {
            runtime_dir: runtime,
            data_dir,
        });
    }
    std::fs::create_dir_all(&runtime_root)?;
    progress("Preparing EasyScribe", 0.0);
    let _lock = CacheLock::acquire(&runtime_root.join(".lock"))?;
    if cache_complete(&runtime, &footer) {
        cleanup_old_runtimes(&runtime_root, &runtime);
        return Ok(Prepared {
            runtime_dir: runtime,
            data_dir,
        });
    }
    cleanup_stale_staging(&runtime_root);
    extract_to_with_progress(&exe, &footer, &runtime_root, &runtime, &mut progress)?;
    cleanup_old_runtimes(&runtime_root, &runtime);
    progress("Ready", 1.0);
    Ok(Prepared {
        runtime_dir: runtime,
        data_dir,
    })
}

/// `%LOCALAPPDATA%\EasyScribe`, or a portable `EasyScribe-data` folder when
/// an `EasyScribe.portable` marker sits next to the exe.
pub fn data_root() -> PathBuf {
    if let Some(dir) = std::env::var_os("EASYSCRIBE_DATA_DIR") {
        return PathBuf::from(dir);
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
        && dir.join("EasyScribe.portable").exists()
    {
        return dir.join("EasyScribe-data");
    }
    dirs::data_local_dir()
        .or_else(dirs::home_dir)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(crate::APP_NAME)
}

pub fn runtime_root(data_dir: &Path) -> PathBuf {
    std::env::var_os("EASYSCRIBE_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| data_dir.join("runtime"))
}

fn cache_complete(runtime: &Path, footer: &Footer) -> bool {
    let marker = runtime.join(RUNTIME_MARKER);
    let Ok(text) = std::fs::read_to_string(&marker) else {
        return false;
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
        return false;
    };
    let expected = footer.cache_key();
    if value.get("payload_sha256").and_then(|v| v.as_str())
        != Some(expected.trim_start_matches("v1-"))
        || value.get("app_version").and_then(|v| v.as_str()) != Some(&footer.app_version)
        || value.get("payload_format").and_then(|v| v.as_u64()) != Some(footer.version as u64)
    {
        return false;
    }
    let required = [
        "manifest.json",
        "models/silero_vad.onnx",
        "models/diarization/segmentation.onnx",
        "models/diarization/embedding.onnx",
        "models/whisper/turbo-encoder.int8.onnx",
        "models/whisper/turbo-decoder.int8.onnx",
        "models/whisper/turbo-tokens.txt",
        "whispercpp/ggml-large-v3-turbo-q5_0.bin",
        "whispercpp/whisper-cli.exe",
        "ffmpeg/ffmpeg.exe",
        "ffmpeg/ffprobe.exe",
        "assets/EasyScribe.ico",
        "assets/EasyScribe.png",
        "assets/selftest_speech.wav",
        "assets/selftest_speech.txt",
        "licenses/THIRD-PARTY-NOTICES.txt",
        "LICENSE",
    ];
    required.iter().all(|rel| {
        std::fs::metadata(runtime.join(rel))
            .map(|metadata| metadata.is_file() && metadata.len() > 0)
            .unwrap_or(false)
    })
}

fn extract_to_with_progress<F: FnMut(&str, f32)>(
    exe: &Path,
    footer: &Footer,
    runtime_root: &Path,
    runtime: &Path,
    progress: &mut F,
) -> Result<()> {
    let staging = runtime_root.join(format!(
        "{}.tmp-{}-{}",
        runtime
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("runtime"),
        std::process::id(),
        uuid::Uuid::new_v4().simple()
    ));
    if staging.exists() {
        let _ = std::fs::remove_dir_all(&staging);
    }
    std::fs::create_dir_all(&staging)?;
    let result = (|| -> Result<()> {
        progress("Checking the app files", 0.02);
        let mut file = File::open(exe)?;
        let mut hasher = Sha256::new();
        file.seek(SeekFrom::Start(footer.payload_offset))?;
        let mut remaining = footer.payload_size;
        let mut buffer = vec![0_u8; 1024 * 1024];
        let mut hashed = 0_u64;
        while remaining > 0 {
            let take = remaining.min(buffer.len() as u64) as usize;
            file.read_exact(&mut buffer[..take])?;
            hasher.update(&buffer[..take]);
            remaining -= take as u64;
            hashed += take as u64;
            progress(
                "Checking the app files",
                0.02 + 0.43 * (hashed as f32 / footer.payload_size.max(1) as f32),
            );
        }
        if hasher.finalize().as_slice() != footer.payload_sha256 {
            return Err(Error::Payload(
                "The app files could not be verified. Download EasyScribe again.".into(),
            ));
        }
        file.seek(SeekFrom::Start(footer.payload_offset))?;
        progress("Unpacking the app files", 0.45);
        let take = file.take(footer.payload_size);
        extract_zip_with_progress(take, &staging, progress)?;
        let marker = serde_json::json!({
            "payload_sha256": footer.cache_key().trim_start_matches("v1-"),
            "app_version": footer.app_version,
            "payload_format": footer.version,
            "build_id": footer.build_id,
        });
        let temp_marker = staging.join(".complete.tmp");
        std::fs::write(&temp_marker, serde_json::to_string_pretty(&marker)?)?;
        std::fs::rename(&temp_marker, staging.join(RUNTIME_MARKER))?;
        Ok(())
    })();

    if result.is_err() {
        let _ = std::fs::remove_dir_all(&staging);
        return result;
    }
    if runtime.exists() {
        // `cache_complete` was false, so this is a partial or corrupted cache
        // (for example a marker from an interrupted older build). Never throw
        // away the freshly extracted staging copy in its favour.
        if let Err(e) = std::fs::remove_dir_all(runtime) {
            let _ = std::fs::remove_dir_all(&staging);
            return Err(Error::Payload(format!(
                "The old EasyScribe runtime could not be replaced. Close any other EasyScribe windows and try again. {e}"
            )));
        }
    }
    std::fs::rename(&staging, runtime)?;
    Ok(())
}

fn extract_zip_with_progress<R: Read + Seek, F: FnMut(&str, f32)>(
    reader: R,
    dest: &Path,
    progress: &mut F,
) -> Result<()> {
    let mut archive = zip::ZipArchive::new(reader)
        .map_err(|e| Error::Payload(format!("The app files are damaged: {e}")))?;
    let total = archive.len().max(1);
    for index in 0..archive.len() {
        progress(
            "Unpacking the app files",
            0.45 + 0.5 * (index as f32 / total as f32),
        );
        let mut entry = archive
            .by_index(index)
            .map_err(|e| Error::Payload(format!("The app files are damaged: {e}")))?;
        let Some(path) = entry.enclosed_name() else {
            return Err(Error::Payload(
                "The app files contain an unsafe path.".into(),
            ));
        };
        let path = sanitize_relative(&path)?;
        let out = dest.join(&path);
        if entry.is_dir() {
            std::fs::create_dir_all(&out)?;
            continue;
        }
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = File::create(&out)?;
        std::io::copy(&mut entry, &mut file)?;
    }
    Ok(())
}

/// Remove extracted runtimes from earlier versions after a successful launch
/// of the current one. Windows keeps files of a running older copy locked, so a
/// failed removal is harmless and retried on the next launch.
fn cleanup_old_runtimes(runtime_root: &Path, keep: &Path) {
    let Ok(entries) = std::fs::read_dir(runtime_root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path == keep || !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !name.starts_with("v1-") || !path.join(RUNTIME_MARKER).is_file() {
            continue;
        }
        if std::fs::remove_dir_all(&path).is_ok() {
            log::info!("Removed old EasyScribe runtime {}", path.display());
        }
    }
}

/// Remove half-written staging folders left by a crashed or killed first run.
/// The caller holds the extraction lock, so every existing `.tmp-` folder is
/// stale. They never contain a complete runtime, so deleting them is safe.
fn cleanup_stale_staging(runtime_root: &Path) {
    let Ok(entries) = std::fs::read_dir(runtime_root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if name.starts_with("v1-") && name.contains(".tmp-") && path.is_dir() {
            let _ = std::fs::remove_dir_all(&path);
        }
    }
}

fn sanitize_relative(path: &Path) -> Result<PathBuf> {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => out.push(part),
            Component::CurDir => {}
            _ => {
                return Err(Error::Payload(
                    "The app files contain an unsafe path.".into(),
                ));
            }
        }
    }
    if out.as_os_str().is_empty() {
        return Err(Error::Payload(
            "The app files contain an empty path.".into(),
        ));
    }
    Ok(out)
}

struct CacheLock {
    /// Holding the open file holds the OS lock. `fs2` releases it when the
    /// process exits, so a crash never leaves a stale lock behind.
    _file: File,
}

impl CacheLock {
    fn acquire(path: &Path) -> Result<Self> {
        use fs2::FileExt;

        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(path)?;
        let deadline = Instant::now() + Duration::from_secs(300);
        loop {
            match file.try_lock_exclusive() {
                Ok(()) => return Ok(Self { _file: file }),
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    if Instant::now() >= deadline {
                        return Err(Error::Payload(
                            "EasyScribe is already getting ready in another window.".into(),
                        ));
                    }
                    std::thread::sleep(Duration::from_millis(200));
                }
                Err(e) => return Err(Error::Io(e)),
            }
        }
    }
}

#[derive(Serialize, Deserialize)]
struct Manifest {
    schema: u32,
    app: String,
    app_version: String,
    payload_format: u16,
    target: String,
    built_utc: String,
    build_id: String,
    total_uncompressed: u64,
    files: Vec<ManifestFile>,
}

#[derive(Serialize, Deserialize)]
struct ManifestFile {
    path: String,
    size: u64,
    method: String,
    sha256: String,
}

/// Build the appendable archive for `payload_dir`. Returns the container bytes
/// and manifest (used by the pack tool and by tests).
pub fn build_container(payload_dir: &Path, app_version: &str, build_id: &str) -> Result<Vec<u8>> {
    let mut files: Vec<PathBuf> = Vec::new();
    collect_files(payload_dir, &mut files)?;
    files.sort();
    if files.is_empty() {
        return Err(Error::Payload("the payload directory is empty".into()));
    }

    let mut manifest_files = Vec::new();
    let mut total = 0_u64;
    for file in &files {
        let relative = file
            .strip_prefix(payload_dir)
            .map_err(|_| Error::Payload("bad payload path".into()))?;
        let relative = relative.to_string_lossy().replace('\\', "/");
        let size = std::fs::metadata(file)?.len();
        total += size;
        let method = if size > 8 * 1024 * 1024 {
            "store"
        } else {
            "deflate"
        };
        manifest_files.push(ManifestFile {
            path: relative,
            size,
            method: method.into(),
            sha256: sha256_file(file)?,
        });
    }
    let manifest = Manifest {
        schema: 1,
        app: crate::APP_NAME.into(),
        app_version: app_version.into(),
        payload_format: CONTAINER_VERSION,
        target: "windows-x86_64".into(),
        built_utc: chrono::Utc::now().to_rfc3339(),
        build_id: build_id.into(),
        total_uncompressed: total,
        files: manifest_files,
    };
    let manifest_json = serde_json::to_vec_pretty(&manifest)?;

    let mut cursor = std::io::Cursor::new(Vec::new());
    {
        let mut zip = zip::ZipWriter::new(&mut cursor);
        let stored = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored)
            .large_file(true);
        zip.start_file("manifest.json", stored)
            .map_err(|e| Error::Payload(e.to_string()))?;
        zip.write_all(&manifest_json)?;
        for file in &files {
            let relative = file
                .strip_prefix(payload_dir)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            let size = std::fs::metadata(file)?.len();
            let options = if size > 8 * 1024 * 1024 {
                zip::write::SimpleFileOptions::default()
                    .compression_method(zip::CompressionMethod::Stored)
                    .large_file(true)
            } else {
                zip::write::SimpleFileOptions::default()
                    .compression_method(zip::CompressionMethod::Deflated)
                    .compression_level(Some(3))
                    .large_file(true)
            };
            zip.start_file(relative, options)
                .map_err(|e| Error::Payload(e.to_string()))?;
            let mut input = File::open(file)?;
            std::io::copy(&mut input, &mut zip)?;
        }
        zip.finish().map_err(|e| Error::Payload(e.to_string()))?;
    }
    Ok(cursor.into_inner())
}

fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') || name.ends_with(".tmp") || name.ends_with(".pdb") {
            continue;
        }
        let metadata = std::fs::symlink_metadata(&path)?;
        if metadata.file_type().is_symlink() {
            return Err(Error::Payload(format!(
                "payload contains a symlink: {}",
                path.display()
            )));
        }
        if metadata.is_dir() {
            collect_files(&path, out)?;
        } else if metadata.is_file() {
            out.push(path);
        }
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

/// Append `container` to a copy of `binary` and write `out`.
pub fn pack(
    binary: &Path,
    payload_dir: &Path,
    out: &Path,
    app_version: &str,
    build_id: &str,
) -> Result<Footer> {
    let container = build_container(payload_dir, app_version, build_id)?;
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::copy(binary, out)?;
    let payload_offset = std::fs::metadata(out)?.len();
    let mut file = OpenOptions::new().write(true).open(out)?;
    file.seek(SeekFrom::End(0))?;
    file.write_all(&container)?;
    let payload_size = container.len() as u64;
    let payload_sha256: [u8; 32] = Sha256::digest(&container).into();
    let manifest = {
        let mut archive = zip::ZipArchive::new(std::io::Cursor::new(&container))
            .map_err(|e| Error::Payload(e.to_string()))?;
        let mut entry = archive
            .by_name("manifest.json")
            .map_err(|e| Error::Payload(e.to_string()))?;
        let mut bytes = Vec::new();
        entry.read_to_end(&mut bytes)?;
        bytes
    };
    let manifest_offset = find_manifest_offset(&container)?;
    let footer = Footer {
        version: CONTAINER_VERSION,
        flags: 0,
        payload_offset,
        payload_size,
        manifest_offset,
        manifest_size: manifest.len() as u64,
        payload_sha256,
        app_version: app_version.into(),
        build_id: build_id.into(),
    };
    file.write_all(&footer.to_bytes())?;
    file.flush()?;
    Ok(footer)
}

fn find_manifest_offset(container: &[u8]) -> Result<u64> {
    // The manifest is the first entry. Its local header ends after the name
    // and extra fields; find the exact data offset so `manifest_offset` is a
    // real location inside the container.
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(container))
        .map_err(|e| Error::Payload(e.to_string()))?;
    let entry = archive
        .by_name("manifest.json")
        .map_err(|e| Error::Payload(e.to_string()))?;
    Ok(entry.data_start())
}

/// Stream the payload archive to disk instead of building it in memory.
///
/// The archive is written to a temporary file beside `out`, hashed while
/// reading it back, then appended to the copied executable. This keeps memory
/// flat for the multi-gigabyte Windows payload (audit finding 12).
pub fn pack_streaming(
    binary: &Path,
    payload_dir: &Path,
    out: &Path,
    app_version: &str,
    build_id: &str,
) -> Result<Footer> {
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let temp = out.with_file_name(format!(
        ".easyscribe-payload-{}.tmp",
        uuid::Uuid::new_v4().simple()
    ));
    let result = (|| -> Result<Footer> {
        write_container_file(&temp, payload_dir, app_version, build_id)?;
        let payload_size = std::fs::metadata(&temp)?.len();
        let payload_sha256 = sha256_file_bytes(&temp)?;
        let (manifest_offset, manifest_size) = inspect_container(&temp)?;

        std::fs::copy(binary, out)?;
        let payload_offset = std::fs::metadata(out)?.len();
        let mut output = OpenOptions::new().append(true).open(out)?;
        let mut archive = File::open(&temp)?;
        std::io::copy(&mut archive, &mut output)?;
        output.flush()?;
        let footer = Footer {
            version: CONTAINER_VERSION,
            flags: 0,
            payload_offset,
            payload_size,
            // The footer stores the manifest position relative to the payload
            // archive, matching `pack` and the packaging reference.
            manifest_offset,
            manifest_size,
            payload_sha256,
            app_version: app_version.into(),
            build_id: build_id.into(),
        };
        output.write_all(&footer.to_bytes())?;
        output.flush()?;
        Ok(footer)
    })();
    let _ = std::fs::remove_file(&temp);
    result
}

fn write_container_file(
    path: &Path,
    payload_dir: &Path,
    app_version: &str,
    build_id: &str,
) -> Result<()> {
    let mut files: Vec<PathBuf> = Vec::new();
    collect_files(payload_dir, &mut files)?;
    files.sort();
    if files.is_empty() {
        return Err(Error::Payload("the payload directory is empty".into()));
    }
    let mut manifest_files = Vec::with_capacity(files.len());
    let mut total = 0_u64;
    for file in &files {
        let relative = file
            .strip_prefix(payload_dir)
            .map_err(|_| Error::Payload("bad payload path".into()))?;
        let relative = relative.to_string_lossy().replace('\\', "/");
        let size = std::fs::metadata(file)?.len();
        total += size;
        manifest_files.push(ManifestFile {
            path: relative,
            size,
            method: if size > 8 * 1024 * 1024 {
                "store"
            } else {
                "deflate"
            }
            .into(),
            sha256: sha256_file(file)?,
        });
    }
    let manifest = Manifest {
        schema: 1,
        app: crate::APP_NAME.into(),
        app_version: app_version.into(),
        payload_format: CONTAINER_VERSION,
        target: "windows-x86_64".into(),
        built_utc: chrono::Utc::now().to_rfc3339(),
        build_id: build_id.into(),
        total_uncompressed: total,
        files: manifest_files,
    };
    let manifest_json = serde_json::to_vec_pretty(&manifest)?;
    let output = File::create(path)?;
    let mut zip = zip::ZipWriter::new(output);
    let stored = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Stored)
        .large_file(true);
    zip.start_file("manifest.json", stored)
        .map_err(|e| Error::Payload(e.to_string()))?;
    zip.write_all(&manifest_json)?;
    for file in &files {
        let relative = file
            .strip_prefix(payload_dir)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        let size = std::fs::metadata(file)?.len();
        let options = if size > 8 * 1024 * 1024 {
            zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored)
                .large_file(true)
        } else {
            zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated)
                .compression_level(Some(3))
                .large_file(true)
        };
        zip.start_file(relative, options)
            .map_err(|e| Error::Payload(e.to_string()))?;
        let mut input = File::open(file)?;
        std::io::copy(&mut input, &mut zip)?;
    }
    zip.finish().map_err(|e| Error::Payload(e.to_string()))?;
    Ok(())
}

fn inspect_container(path: &Path) -> Result<(u64, u64)> {
    let file = File::open(path)?;
    let mut archive = zip::ZipArchive::new(file).map_err(|e| Error::Payload(e.to_string()))?;
    let entry = archive
        .by_name("manifest.json")
        .map_err(|e| Error::Payload(e.to_string()))?;
    Ok((entry.data_start(), entry.size()))
}

fn sha256_file_bytes(path: &Path) -> Result<[u8; 32]> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finalize().into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn footer_round_trip() {
        let footer = Footer {
            version: 1,
            flags: 0,
            payload_offset: 1234,
            payload_size: 5678,
            manifest_offset: 12,
            manifest_size: 34,
            payload_sha256: [7; 32],
            app_version: "3.2.3".into(),
            build_id: "abc".into(),
        };
        let bytes = footer.to_bytes();
        let parsed = Footer::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.payload_size, 5678);
        assert_eq!(parsed.app_version, "3.2.3");
        assert_eq!(parsed.cache_key().len(), 35);
    }

    #[test]
    fn old_runtimes_are_removed_but_current_survives() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let current = root.join("v1-current");
        let old = root.join("v1-old");
        let unrelated = root.join("notes");
        std::fs::create_dir_all(&current).unwrap();
        std::fs::create_dir_all(&old).unwrap();
        std::fs::create_dir_all(&unrelated).unwrap();
        std::fs::write(current.join(RUNTIME_MARKER), "{}").unwrap();
        std::fs::write(old.join(RUNTIME_MARKER), "{}").unwrap();
        cleanup_old_runtimes(root, &current);
        assert!(current.exists());
        assert!(!old.exists());
        assert!(unrelated.exists());
    }

    #[test]
    fn streaming_pack_appends_a_verifiable_container() {
        let dir = tempfile::tempdir().unwrap();
        let payload = dir.path().join("payload");
        std::fs::create_dir_all(payload.join("models")).unwrap();
        std::fs::write(payload.join("models/a.txt"), b"hello").unwrap();
        std::fs::write(payload.join("run.bin"), vec![3_u8; 5000]).unwrap();
        // Files over 8 MB take the stored (no compression level) path.
        std::fs::write(payload.join("big.bin"), vec![7_u8; 9 * 1024 * 1024]).unwrap();
        let binary = dir.path().join("EasyScribe.exe");
        std::fs::write(&binary, b"fake windows launcher").unwrap();
        let out = dir.path().join("out.exe");
        let footer = pack_streaming(&binary, &payload, &out, "3.2.3", "test").unwrap();
        assert_eq!(
            footer.payload_size,
            std::fs::metadata(&out).unwrap().len()
                - 128
                - std::fs::metadata(&binary).unwrap().len()
        );
        let parsed = read_footer(&out).unwrap().unwrap();
        assert_eq!(parsed.payload_sha256, footer.payload_sha256);
        let bytes = std::fs::read(&out).unwrap();
        assert_eq!(
            &bytes[footer.payload_offset as usize..footer.payload_offset as usize + 2],
            b"PK"
        );
        // The footer stores the manifest data position relative to the payload.
        let manifest = footer.payload_offset as usize + footer.manifest_offset as usize;
        assert!(
            manifest as u64 + footer.manifest_size <= footer.payload_offset + footer.payload_size
        );
        let data = &bytes[manifest..manifest + footer.manifest_size as usize];
        let value: serde_json::Value = serde_json::from_slice(data).unwrap();
        assert_eq!(value["app_version"], "3.2.3");
    }

    #[test]
    fn cache_requires_every_runtime_file_to_be_present_and_non_empty() {
        let footer = Footer {
            version: CONTAINER_VERSION,
            flags: 0,
            payload_offset: 0,
            payload_size: 1,
            manifest_offset: 0,
            manifest_size: 1,
            payload_sha256: [9; 32],
            app_version: "3.2.3".into(),
            build_id: "test".into(),
        };
        let dir = tempfile::tempdir().unwrap();
        let runtime = dir.path().join("runtime");
        for rel in [
            "manifest.json",
            "models/silero_vad.onnx",
            "models/diarization/segmentation.onnx",
            "models/diarization/embedding.onnx",
            "models/whisper/turbo-encoder.int8.onnx",
            "models/whisper/turbo-decoder.int8.onnx",
            "models/whisper/turbo-tokens.txt",
            "whispercpp/ggml-large-v3-turbo-q5_0.bin",
            "whispercpp/whisper-cli.exe",
            "ffmpeg/ffmpeg.exe",
            "ffmpeg/ffprobe.exe",
            "assets/EasyScribe.ico",
            "assets/EasyScribe.png",
            "assets/selftest_speech.wav",
            "assets/selftest_speech.txt",
            "licenses/THIRD-PARTY-NOTICES.txt",
            "LICENSE",
        ] {
            let path = runtime.join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, b"x").unwrap();
        }
        let marker = serde_json::json!({
            "payload_sha256": footer.cache_key().trim_start_matches("v1-"),
            "app_version": "3.2.3",
            "payload_format": 1,
        });
        std::fs::write(runtime.join(RUNTIME_MARKER), marker.to_string()).unwrap();
        assert!(cache_complete(&runtime, &footer));

        let gguf = runtime.join("whispercpp/ggml-large-v3-turbo-q5_0.bin");
        std::fs::remove_file(&gguf).unwrap();
        assert!(!cache_complete(&runtime, &footer));

        std::fs::write(&gguf, b"").unwrap();
        assert!(!cache_complete(&runtime, &footer));
        std::fs::write(&gguf, b"x").unwrap();

        let wrong_version = serde_json::json!({
            "payload_sha256": footer.cache_key().trim_start_matches("v1-"),
            "app_version": "3.2.2",
            "payload_format": 1,
        });
        std::fs::write(runtime.join(RUNTIME_MARKER), wrong_version.to_string()).unwrap();
        assert!(!cache_complete(&runtime, &footer));
    }

    #[test]
    fn read_footer_allows_an_appended_signature() {
        let dir = tempfile::tempdir().unwrap();
        let payload = dir.path().join("payload");
        std::fs::create_dir_all(&payload).unwrap();
        std::fs::write(payload.join("a.txt"), b"hello").unwrap();
        let binary = dir.path().join("EasyScribe.exe");
        std::fs::write(&binary, b"fake windows launcher").unwrap();
        let out = dir.path().join("out.exe");
        let footer = pack_streaming(&binary, &payload, &out, "3.2.3", "signed").unwrap();
        let mut bytes = std::fs::read(&out).unwrap();
        // Authenticode appends a WIN_CERTIFICATE table after the footer.
        bytes.extend_from_slice(&[0xAA; 4096]);
        std::fs::write(&out, bytes).unwrap();
        let parsed = read_footer(&out).unwrap().unwrap();
        assert_eq!(parsed.payload_sha256, footer.payload_sha256);
    }

    #[test]
    fn container_extracts_and_verifies() {
        let dir = tempfile::tempdir().unwrap();
        let payload = dir.path().join("payload");
        std::fs::create_dir_all(payload.join("models")).unwrap();
        std::fs::write(payload.join("models/a.txt"), b"hello").unwrap();
        std::fs::write(payload.join("run.bin"), vec![0_u8; 100]).unwrap();
        std::fs::write(payload.join("big.bin"), vec![0_u8; 9 * 1024 * 1024]).unwrap();
        let container = build_container(&payload, "3.2.3", "test").unwrap();
        let mut archive = zip::ZipArchive::new(std::io::Cursor::new(&container)).unwrap();
        assert!(archive.by_name("manifest.json").is_ok());
        assert!(archive.by_name("models/a.txt").is_ok());
        assert_eq!(
            archive.by_name("big.bin").unwrap().compression(),
            zip::CompressionMethod::Stored
        );
    }
}
