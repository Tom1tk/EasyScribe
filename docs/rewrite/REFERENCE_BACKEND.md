# EasyScribe Reference Backend — Engineering Specification

**Status:** implementation reference for the Rust rewrite  
**Reference implementation:** `/home/user/EasyScribe` (Python), git commit `83235166a7a0f317d1c33498eb791dc3fd383d17`, `APP_VERSION = "3.2.3"`  
**Scope:** the non-GUI backend: configuration, logging, offline enforcement, settings, audio extraction, VAD, whisper.cpp, sherpa-onnx transcription, diarization, microphone capture, live transcription, crash recovery, renaming, and the process entry point. GUI-originated orchestration rules that affect backend outputs (file naming, callbacks, status strings) are included because they are part of the observable contract.  
**Not in scope:** visual layout, customtkinter widgets, screen transitions, `win_paint`, installer/launcher logic, packaging.

All `src/...` and `tests/...` line references point at the Python reference repository. Rust code must reproduce the behavior, not the Python structure.

---

## 0. How to use this document

### 0.1 Parity rules

1. **P0 — externally observable behavior is normative.** File names, file bytes, transcript text, log text, subprocess argument vectors, progress values, callback ordering, cancellation points, and exception categories/messages must match.
2. **P1 — internal structure is free.** Rust module names, threading primitives, channels, and type names may differ. The invariants in this document are normative.
3. **P2 — platform mappings are normative.** Windows-specific calls have a documented cross-platform equivalent in §20. On Windows the reference behavior wins; on other platforms the mapping must preserve semantics.
4. **Defects are recorded, not hidden.** Where the Python implementation contains a latent defect (for example same-second recording filename collisions), this document marks it **DEFECT**. The default parity choice is to reproduce observable behavior; a Rust deviation must be explicitly approved and recorded in the compatibility section.
5. **No Python files may be modified during the rewrite.** This document is derived by reading, not changing, the reference.

### 0.2 Reference line notation

- `src/foo.py:12-34` means lines 12 through 34 inclusive in the reference file.
- `tests/test_foo.py:10-20` means the relevant assertions.
- “GUI” means `src/gui.py`; it is not a backend module, but its backend-facing contract is included in §18.

### 0.3 Source snapshot facts

- Python 3.13 virtual environment, `sherpa-onnx==1.13.8` pinned in `requirements.txt` (the local `.venv` used for inspection reports 1.13.2; API calls in the source are the pin’s API).
- `requirements.txt` pins `numpy==2.4.6`, `sounddevice==0.5.6`, `customtkinter==6.0.0`, `tkinterdnd2==0.6.3`, `pyinstaller==6.22.3`.
- All model inference is local. No module imports an HTTP client; the offline guard is described in §5.

---

## 1. System architecture

### 1.1 Data flow: file transcription

```
selected media file
        │
        ▼
extract_audio()  (src/ffmpeg_wrapper.py)
  ├─ already mono 16 kHz PCM s16le WAV? ──► copy to temp/<uuid>.wav
  └─ otherwise ffmpeg ────────────────────► temp/<uuid>.wav
        │
        ▼
TranscriptionEngine.transcribe()  (src/transcriber.py)
  ├─ whispercpp_wrapper.is_available()?
  │     yes ─► whisper-cli whole file ─► temp/whispercpp_<uuid>.json ─► segments
  │     no  ─► sherpa-onnx:
  │             load WAV float32
  │             Silero VAD ─► speech bounds
  │             pad_and_clamp ─► chunks
  │             OfflineRecognizer per chunk ─► segments
  │
  ├─ optional diarization (sherpa-onnx child process)
  │     turns ─► assign_speakers() ─► speaker names
  │
  ├─ transcript builders
  └─ _write_transcript() ─► UTF-8 .txt (always trailing LF)
```

### 1.2 Data flow: microphone recording

```
sounddevice InputStream 16 kHz mono int16, blocksize 512
        │ audio callback
        ├─► VAD queue (float32 [-1,1]) ─► LiveTranscriber (optional)
        │                                   ├─ Silero VAD (30 s buffer)
        │                                   ├─ pad_live_segment (leading only)
        │                                   ├─ decode queue (max 8, drop oldest)
        │                                   └─ sherpa recognizer ─► on_segment(text)
        │
        └─► PCM queue ─► PCMWriter thread
                           recording_<YYYYMMDD_HHMMSS>.pcm  (raw int16 LE)
                           recording_<YYYYMMDD_HHMMSS>.json (sidecar)
                           fsync + sidecar update every 50 chunks (~1.6 s)
        │
        ▼ Stop
LiveTranscriber.stop()  (flush in-progress speech, drain queue)
MicRecorder.stop()      (convert PCM → WAV via recovery.recover_pcm_to_wav,
                         write to chosen folder or fall back next to PCM,
                         delete .pcm/.json only after WAV succeeds)
        │
        ├─ live transcript: <session_stem>.txt (one segment per line)
        └─ record-only: optionally immediately run the file pipeline
           (“best quality transcript”) with timestamps on, diarization off
```

### 1.3 Thread and process model

| Component | Execution | Lifetime / naming | Notes |
|---|---|---|---|
| GUI main thread | Tk mainloop | process lifetime | Marshals worker callbacks with `after(0, ...)` (`src/gui.py:1979-1999`) |
| Transcription worker | daemon thread `TranscriptionWorker` | per batch (`src/gui.py:1431-1434`) | Sequential files |
| Recording worker | daemon thread `RecordingWorker` | per recording (`src/gui.py:1453-1456`) | Owns MicRecorder/LiveTranscriber |
| Speakers worker | daemon thread `SpeakersWorker` | per “Find speakers” (`src/gui.py:1908-1911`) | `add_speakers_to_file` |
| PCM writer | daemon thread `PCMWriter` (`src/mic_recorder.py:101-106`) | recording lifetime | Raw PCM + sidecar |
| Live VAD | daemon thread `LiveVAD` (`src/live_transcriber.py:81-88`) | live recording | Owns VAD, history |
| Live decoder | daemon thread `LiveDecoder` (`src/live_transcriber.py:75-88`) | live recording | Owns recognizer |
| Diarization worker | separate process, `multiprocessing.get_context("spawn")` (`src/diarizer.py:163-170`) | per diarization | Child runs sherpa pipeline; parent can terminate it |

Locks:

- `TranscriptionEngine._lock` serializes lazy model loading (`src/transcriber.py:313-338`).
- `LiveTranscriber` passes one `threading.Lock` (`onnx_lock`) shared by the VAD and decode loops because sherpa-onnx’s VAD and recognizer share an onnxruntime environment; concurrent `Run()` calls can abort the process (`src/live_transcriber.py:69-73`, `153-155`, `171-173`, `195-199`). Rust must serialize both through the same mutex.
- `MicRecorder` has no explicit queue lock; Python’s `queue.Queue` is thread-safe. Rust must use `crossbeam-channel`/`std::sync::mpsc` or a mutex-protected deque.

### 1.4 Startup order

`src/main.py:12-14` and `89-136`:

1. `import offline_guard; offline_guard.install()` **before any other application import**. This sets environment variables and monkey-patches sockets.
2. `import config` — resolves `BASE_DIR`, creates `logs/` and `temp/` at import time (`src/config.py:119-123`).
3. `setup_logging()` — rotates old logs, opens `logs/EasyScribe_<timestamp>.log`, installs stdout handler only when not frozen (`src/logger.py:29-68`).
4. `_cleanup_temp_files()` — deletes `temp/*.wav` and `temp/*.json`; registers the same function with `atexit` (`src/main.py:30-47`, `98-99`).
5. `RECORDING_WORK_DIR.mkdir(parents=True, exist_ok=True)` (`src/main.py:101-102`). This directory is `BASE_DIR/recordings`.
6. Offer crash recovery for `recording_*.pcm` + `.json` in `RECORDING_WORK_DIR`, into `load_recordings_dir() or DEFAULT_RECORDINGS_DIR`; any exception is logged as a warning and startup continues (`src/main.py:104-110`).
7. Check dependencies: ffmpeg, ffprobe, and sherpa model files (`src/main.py:112-127`). Missing files show a Tk error dialog and exit code 1.
8. Import `gui.TranscriberApp`, construct it, run `mainloop()` (`src/main.py:129-136`).
9. In `__main__`, call `multiprocessing.freeze_support()` before `main()` so a frozen child process runs the spawn target rather than opening a second window (`src/main.py:139-144`).

### 1.5 Shutdown

- `atexit` deletes `temp/*.wav` and `temp/*.json` again (`src/main.py:99`).
- Interrupted microphone recordings are deliberately **not** in `temp/`; they live in `recordings/` and survive a crash for recovery (`src/main.py:33-38`, `src/config.py:63-68`).
- No explicit cleanup of `temp/*.bin`; `whispercpp_wrapper._best_gpu()` deletes its probe file in `finally`, but a leaked probe file would survive cleanup (`src/whispercpp_wrapper.py:144-161`).

---

## 2. Constants and path resolution (`src/config.py`)

### 2.1 App identity

| Constant | Exact value | Line |
|---|---|---|
| `APP_NAME` | `"EasyScribe"` | `src/config.py:14` |
| `APP_VERSION` | `"3.2.3"` | `src/config.py:15` |

`APP_VERSION` is the single source of truth for the Python release. The Rust crate version and UI title must match the release under test.

### 2.2 Base directory

`get_base_dir()` (`src/config.py:20-29`):

- Frozen (`getattr(sys, "frozen", False)` true): `Path(sys.executable).parent.resolve()` — the directory containing `EasyScribe.exe`.
- Development: `Path(__file__).parent.parent.resolve()` — project root (parent of `src/`).
- `BASE_DIR` is computed once at import (`src/config.py:32`).

Rust mapping:

- Frozen/bundled: `std::env::current_exe()?.parent()`, canonicalized.
- Development/test: an explicit `CARGO_MANIFEST_DIR`/resource-root or the executable’s parent, **not** the current working directory. The Rust binary must behave the same regardless of the directory it is launched from.
- `resolve()` follows symlinks and returns an absolute path. Rust `canonicalize()` fails if the path does not exist; for the executable path it exists. For development, canonicalize the manifest/resource root.

### 2.3 Documents folder

`_documents_dir()` (`src/config.py:37-60`):

1. On Windows, call `SHGetKnownFolderPath` with GUID `FDD39AD0-238F-46AF-ADB4-6C85480369C7` (Documents), flags 0, token `None`. Use `uuid.UUID(...).bytes_le` (little-endian GUID byte order). On success, return `Path(path_ptr.value)` and free the string with `CoTaskMemFree`.
2. Any exception is swallowed and the fallback is used.
3. Fallback: `Path.home() / "Documents"` if that directory exists, otherwise `Path.home()`.

Rust mapping: `windows` crate `SHGetKnownFolderPath(&FOLDERID_Documents, 0, None)` and `CoTaskMemFree`; or a maintained `known-folders`/`dirs` crate that uses the same API. Non-Windows: `home_dir()/Documents` if it is a directory, else `home_dir()`. Do not use an environment variable that a user can spoof if the known-folder API is available.

### 2.4 User data paths

| Constant | Value | Line | Notes |
|---|---|---|---|
| `RECORDING_WORK_DIR` | `BASE_DIR / "recordings"` | `src/config.py:65` | In-progress PCM + JSON; launcher preserves this directory on update |
| `SETTINGS_FILE` | `RECORDING_WORK_DIR / "settings.json"` | `src/config.py:68` | Lives inside the preserved directory |
| `DEFAULT_RECORDINGS_DIR` | `_documents_dir() / APP_NAME` | `src/config.py:70` | e.g. `C:\Users\me\Documents\EasyScribe`; not created at import |

Important: `RECORDING_WORK_DIR` is created by `main()` (`src/main.py:102`), not by `config.py`. `DEFAULT_RECORDINGS_DIR` is created lazily by `MicRecorder.stop()`/recovery (`src/mic_recorder.py:152`, `src/recovery.py:81`).

### 2.5 Model and tool paths

| Constant | Exact path | Line |
|---|---|---|
| `WHISPER_ENCODER` | `BASE_DIR/models/whisper/turbo-encoder.int8.onnx` | `src/config.py:81-84` |
| `WHISPER_DECODER` | `BASE_DIR/models/whisper/turbo-decoder.int8.onnx` | `src/config.py:81-84` |
| `WHISPER_TOKENS` | `BASE_DIR/models/whisper/turbo-tokens.txt` | `src/config.py:81-84` |
| `VAD_MODEL_PATH` | `BASE_DIR/models/silero_vad.onnx` | `src/config.py:88` |
| `DIARIZATION_SEGMENTATION_MODEL` | `BASE_DIR/models/diarization/segmentation.onnx` | `src/config.py:97-99` |
| `DIARIZATION_EMBEDDING_MODEL` | `BASE_DIR/models/diarization/embedding.onnx` | `src/config.py:97-99` |
| `FFMPEG_BIN` | `BASE_DIR/ffmpeg/ffmpeg.exe` | `src/config.py:103-105` |
| `FFPROBE_BIN` | `BASE_DIR/ffmpeg/ffprobe.exe` | `src/config.py:103-105` |
| `WHISPERCPP_BIN` | `BASE_DIR/whispercpp/whisper-cli.exe` on Windows, `BASE_DIR/whispercpp/whisper-cli` otherwise | `src/config.py:112-114` |
| `WHISPERCPP_MODEL` | `BASE_DIR/whispercpp/ggml-large-v3-turbo-q5_0.bin` | `src/config.py:114` |
| `APP_ICON` | `assets/EasyScribe.ico` under `sys._MEIPASS` (frozen) or `BASE_DIR` | `src/config.py:75-77` |
| `APP_ICON_PNG` | `assets/EasyScribe.png` under the same resource root | `src/config.py:75-77` |

Cross-platform note: `FFMPEG_BIN` and `FFPROBE_BIN` are hard-coded to `.exe` in the reference. See §20.2 for the required mapping; do not silently change Windows behavior.

### 2.6 Runtime constants

| Constant | Exact value | Line | Notes |
|---|---|---|---|
| `VAD_SAMPLE_RATE` | `16000` | `src/config.py:92` | All engines and mic use 16 kHz |
| `VAD_CHUNK_SAMPLES` | `512` | `src/config.py:93` | 32 ms at 16 kHz |
| `WHISPERCPP_BEAM_SIZE` | `5` | `src/config.py:115` | Passed as `-bs 5` |
| `NUM_THREADS` | `min(8, max(4, (os.cpu_count() or 4) // 2))` | `src/config.py:139` | Logical CPUs; examples: 8→4, 16→8, 32→8, 1→4 |
| `MIN_FREE_DISK_BYTES` | `512 * 1024 * 1024` (536,870,912) | `src/config.py:143` | GUI warns below this |
| `MAX_LOG_FILES` | `10` | `src/config.py:144` | Rotation target |
| `SUPPORTED_EXTENSIONS` | see below | `src/config.py:127-132` | Used by GUI file selection/drop only |

`SUPPORTED_EXTENSIONS` exact set (lowercase with dot):

```
.mp4 .mkv .mov .avi .webm
.mp3 .wav .m4a .flac .ogg .opus .aac
```

It is a `frozenset[str]`; GUI checks `p.suffix.lower() in SUPPORTED_EXTENSIONS` (`src/gui.py:2027-2029`). The extractor itself does not enforce the list; ffmpeg decides.

`NUM_THREADS` is computed from the process’s CPU count at import. Rust should use `std::thread::available_parallelism()` with the same formula and a fallback of 4. Note the reference uses `os.cpu_count()`, which counts all logical processors; `available_parallelism` may respect affinity/cgroups. If exact parity on a given machine matters, use the platform CPU count API that matches `os.cpu_count()` (Windows `GetSystemInfo`, Linux `sysconf(_SC_NPROCESSORS_ONLN)` / `/proc/cpuinfo`, macOS `sysctlbyname("hw.logicalcpu")`) and fall back to 4.

### 2.7 Directories created at import

`src/config.py:119-123`:

```
LOGS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)
```

If these fail, import fails before logging is configured. Rust must create them early and report a fatal startup error; it must not silently fall back to a temp directory, because paths are part of the contract.

---

## 3. Common primitives (`src/common.py`)

### 3.1 `CancelledError`

`class CancelledError(RuntimeError)` (`src/common.py:15-16`). Raised by every cancellable stage; this is the **only** cancellation type. An empty message is allowed. Rust equivalent:

```rust
#[derive(Debug)]
pub struct CancelledError(pub String);
impl std::fmt::Display ... // message exactly as supplied
impl std::error::Error ...
```

Do not map cancellation to `std::io::ErrorKind::Interrupted`; GUI code distinguishes it by type.

### 3.2 `fmt_duration(seconds) -> str`

`src/common.py:19-22`:

```python
h = int(seconds // 3600)
m = int((seconds % 3600) // 60)
s = int(seconds % 60)
return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"
```

Exact behavior:

- No rounding; fractional seconds are truncated.
- `0` → `"0m 00s"`.
- `65` → `"1m 05s"`.
- `3723.5` → `"1h 02m 03s"`.
- Without hours, minutes are **not** zero-padded (`"1m 05s"`, not `"01m 05s"`).
- With hours, minutes and seconds are zero-padded to at least 2 digits; hours are not padded.
- Inputs are nonnegative in all call sites. For negative values the Python floor-division behavior is unusual; Rust may clamp to 0 or reproduce floor semantics, but no backend should pass negatives.

Rust signature: `pub fn fmt_duration(seconds: f64) -> String`.

### 3.3 `wav_duration(wav_path) -> float`

`src/common.py:25-28`:

```python
with wave.open(str(wav_path), "rb") as wf:
    return wf.getnframes() / float(wf.getframerate())
```

- Reads only the header; no samples.
- Uses the WAV header’s frame count, which includes all channels; for mono 16 kHz it is seconds.
- Errors propagate: `FileNotFoundError`, `wave.Error`, `OSError`, `ZeroDivisionError` if frame rate is zero (unlikely), etc.
- Callers wrap it in stage-specific errors:
  - `TranscriptionEngine.transcribe` → `TranscriptionError("Could not load audio: {exc}")` (`src/transcriber.py:369-372`).
  - `DiarizationEngine.diarize` → `DiarizationError("Could not load audio for diarization: {exc}")` (`src/diarizer.py:149-152`).
  - `save_measured_speed` catches all and returns (`src/diarizer.py:76-79`).

Rust: parse the RIFF/WAVE header or use a small WAV reader; compute `frames as f64 / sample_rate as f64`. Reject non-PCM/unsupported formats consistently with Python `wave` (which supports PCM only). The function must not decode audio.

### 3.4 `load_wav_float32(wav_path) -> (np.ndarray, int)`

`src/common.py:31-46`:

1. Open with `wave.open(..., "rb")`.
2. Read `n_ch`, `sampwidth`, `sr`, and all frames.
3. Sample-width mapping:
   - `sampwidth == 2` → `np.int16`, scale `32768.0`
   - `sampwidth == 4` → `np.int32`, scale `2147483648.0`
   - anything else → `np.int8`, scale `128.0`
4. `samples = np.frombuffer(raw, dtype).astype(np.float32) / scale`.
5. If `n_ch > 1`: `samples.reshape(-1, n_ch).mean(axis=1)` (mono downmix by arithmetic mean, after scaling).
6. Return `(samples, sample_rate)`.

Exact semantics and caveats:

- The result is normalized approximately to `[-1.0, 1.0)`: `32767/32768`, `-32768/32768 = -1.0`.
- For 8-bit PCM, Python’s `wave` gives unsigned bytes (0..255), but this code interprets them as signed int8 and scales by 128. That is a **DEFECT** for 8-bit PCM. All real engine inputs are 16-bit PCM (ffmpeg’s `-acodec pcm_s16le`), so the branch is not exercised in production.
- For 24-bit PCM (`sampwidth == 3`) the code treats the 3-byte stream as int8, producing three times as many samples with wrong values. **DEFECT**; production never produces 24-bit.
- For 32-bit float WAV (`sampwidth == 4`, format 3), `wave` still reports 4 bytes but the data is IEEE float; this code interprets it as int32. **DEFECT** for float WAV; production never produces it.
- Multi-channel downmix uses the mean of scaled floats, not integer sum/division.
- The array dtype is `float32`; `mean` on float32 stays float32 in current NumPy. Rust should accumulate in f32 or f64? To match bit-for-bit, use f32 arithmetic and divide by channel count as f32. Most inputs are mono, so this only matters for unusual WAVs.
- No resampling; the sample rate is returned and callers use it for `seg_start / sr`.

Rust signature: `pub fn load_wav_float32(path: &Path) -> Result<(Vec<f32>, u32), WavError>`.

**Recommended Rust policy:** support exactly 8/16/32-bit integer PCM with the Python mapping for 16/32 and the documented 8-bit behavior if parity is required; reject 24-bit and IEEE-float WAVs with a clear `UnsupportedWav` error unless a compatibility mode is requested. Since all internal WAVs are `pcm_s16le`, this does not affect production paths. Record the deviation in the compatibility section if 8/24-bit behavior differs.

---

## 4. Logging (`src/logger.py`)

### 4.1 Log rotation

`_rotate_logs()` (`src/logger.py:19-27`):

```python
log_files = sorted(LOGS_DIR.glob(f"{APP_NAME}_*.log"), key=lambda p: p.stat().st_mtime)
while len(log_files) >= MAX_LOG_FILES:
    try:
        log_files.pop(0).unlink()
    except OSError:
        break
```

- Pattern: `EasyScribe_*.log` in `LOGS_DIR`.
- Sort ascending by modification time; oldest first.
- While count is **greater than or equal to** 10, delete the oldest. After this, at most 9 files remain; adding the new log gives at most 10.
- If deleting fails (permissions, file locked), stop immediately; do not try younger files.
- If `stat()` fails during sorting, the exception propagates and logging setup fails. Rust should handle this more defensively but preserve the deletion policy.
- Ties in mtime: Python’s sort is stable and the glob order is filesystem-dependent. Rust should use `(mtime, file_name)` as a deterministic tie-break; this is not externally important but makes tests stable.

### 4.2 `setup_logging() -> Logger`

`src/logger.py:29-68`:

1. Call `_rotate_logs()`.
2. Timestamp: `datetime.now().strftime("%Y%m%d_%H%M%S")`.
3. File: `LOGS_DIR / f"{APP_NAME}_{timestamp}.log"` → `logs/EasyScribe_20261106_084312.log`.
4. Format: `"%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"`.
5. Date format: `"%Y-%m-%d %H:%M:%S"`.
6. Handlers:
   - `logging.FileHandler(log_file, encoding="utf-8")` — default append mode; creates the file immediately.
   - `logging.StreamHandler(sys.stdout)` **only if not frozen**.
7. `logging.basicConfig(level=logging.DEBUG, format=..., datefmt=..., handlers=..., force=True)`.
8. Get `logging.getLogger(APP_NAME)` and log:
   - `"=" * 60`
   - `"EasyScribe v3.2.3 starting"`
   - `"=" * 60`
   - `"Log file    : {log_file}"`
   - `"BASE_DIR    : {BASE_DIR}"`
   - `"Python      : {sys.version}"`
   - `"Platform    : {platform.platform()}"`
   - `"Frozen      : {True|False}"`
9. Return the `EasyScribe` logger.

Observable format example:

```
2026-11-06 08:43:12 [INFO    ] transcriber: [Transcribe] Starting: meeting.mp3
```

- `%(levelname)-8s` pads to 8 characters: `INFO    `, `WARNING `, `ERROR   `, `DEBUG   `, `CRITICAL`.
- Logger names are Python module names (`config`, `transcriber`, `diarizer`, `mic_recorder`, `live_transcriber`, `whispercpp_wrapper`, `ffmpeg_wrapper`, `recovery`, `settings`, `gui`), **not** prefixed with `EasyScribe`.
- All levels are enabled (`DEBUG`); most code uses `info`, some `debug`, warnings/errors as noted.

Rust mapping:

- Use `log` + a file writer (or `tracing-appender`). Reproduce the exact line format and timestamp format; do not add ANSI color or target prefixes.
- File append mode and UTF-8. On Windows, open with sharing that permits reading? Python’s `FileHandler` uses standard `open` with default sharing; Rust `OpenOptions` similar.
- Frozen/bundled builds should not install a stdout handler (GUI/no console); debug builds may mirror to stderr.
- Thread-safe writes must not interleave partial lines; use a mutex or a single writer thread.
- The “Python:”, “Platform:”, “Frozen:” values are diagnostic only; Rust should log equivalent `Rust:`, OS version, and bundled flag, or keep the labels and provide equivalent values if log parity is tested. Record the choice.

### 4.3 Log file lifecycle

- A new file is created on every process start, even if the previous process crashed.
- Same-second starts use the same filename because the timestamp has one-second resolution; `FileHandler` appends. **DEFECT** for log separation; Rust should either append (parity) or add a collision suffix. Document the choice.
- Rotation runs before opening the new file. It does not compress or archive.
- No log is written to the user’s Documents folder; logs stay in the install directory.

---

## 5. Offline guard (`src/offline_guard.py`)

### 5.1 Purpose

EasyScribe is GDPR/offline software: all models are bundled and user files never leave the machine. Python cannot prevent a native library from opening a socket, but the guard makes Python-level networking fail closed and sets standard opt-out environment variables. The reference docstring explicitly states the limitation for subprocesses (`src/offline_guard.py:22-24`).

### 5.2 Error type

`class NetworkBlockedError(ConnectionRefusedError)` (`src/offline_guard.py:55-56`).

- Subclass of `ConnectionRefusedError`, therefore `OSError`.
- This lets library code that catches `OSError`/`URLError` degrade gracefully; `urllib.request.urlopen` ends up raising `URLError` (an `OSError` subclass), which the test accepts (`tests/test_offline_guard.py:58-68`).

Rust equivalent: `OfflineBlocked` implementing `std::error::Error`, with a conversion/compatibility path to `std::io::ErrorKind::ConnectionRefused` if any code expects an I/O error. Do not use `panic!` in release for a library-triggered attempt; return/propagate an error.

### 5.3 Environment variables

`_OFFLINE_ENV` (`src/offline_guard.py:40-52`) exact insertion order:

| Key | Value |
|---|---|
| `DO_NOT_TRACK` | `"1"` |
| `HF_HUB_OFFLINE` | `"1"` |
| `HF_HUB_DISABLE_TELEMETRY` | `"1"` |
| `TRANSFORMERS_OFFLINE` | `"1"` |
| `HF_DATASETS_OFFLINE` | `"1"` |
| `NO_PROXY` | `"*"` |
| `no_proxy` | `"*"` |

`install()` sets each with `os.environ[key] = value`, overwriting existing values. The comment explains `NO_PROXY=*`: a local proxy at `127.0.0.1:port` would pass the loopback check and forward traffic out; `*` makes urllib/requests bypass every proxy, so the request hits the DNS/connect block.

Rust: set these at the very start of `main`, before spawning any threads or loading native libraries. In Rust 2024, `std::env::set_var` is `unsafe` because environment mutation is not thread-safe; call it before thread creation and wrap in a clearly documented `unsafe` block. On Windows environment names are case-insensitive; setting both `NO_PROXY` and `no_proxy` is intentional and must be preserved.

### 5.4 Host and address classification

`_is_local_host(host)` (`src/offline_guard.py:59-72`):

1. `None` → `True`.
2. `bytes` → decode ASCII with `errors="replace"`.
3. Non-`str` → `False`.
4. Normalize:
   - `host.strip()` — remove leading/trailing whitespace.
   - `.strip("[]")` — remove **all** leading/trailing `[` and `]` characters (not just one pair).
   - `.split("%", 1)[0]` — drop an IPv6 zone identifier.
   - `.lower()`.
5. Allowed names: `""`, `"localhost"`, `"localhost."`, `"ip6-localhost"` → `True`.
6. Otherwise `ipaddress.ip_address(host).is_loopback`; `ValueError` → `False`.

Consequences:

- `127.0.0.0/8` and `::1` are allowed.
- IPv4-mapped loopback (`::ffff:127.0.0.1`) is `is_loopback` in supported Python versions; Rust should use the `ipnet`/`std::net::IpAddr::is_loopback` semantics and explicitly test mapped addresses.
- A hostname that merely resolves to loopback is **not** allowed because DNS lookup itself is blocked unless the name is in the literal allowlist.
- Empty string is allowed.

`_is_allowed_address(family, address)` (`src/offline_guard.py:75-80`):

- `family == AF_UNIX` (using `getattr(socket, "AF_UNIX", object())`) → `True`.
- If `address` is a non-empty tuple → `_is_local_host(address[0])`.
- Otherwise → `False`.

### 5.5 Patched socket APIs

`install()` (`src/offline_guard.py:89-144`) is idempotent via module flag `_installed` (`src/offline_guard.py:36`, `89-93`). It saves the original bound methods and replaces:

| API | Behavior |
|---|---|
| `socket.socket.connect(self, address)` | If address not allowed, raise `NetworkBlockedError`; else original |
| `socket.socket.connect_ex(self, address)` | Same check; raises instead of returning an errno (deliberate; tests enforce) |
| `socket.socket.sendto(self, data, *args)` | `address = args[-1] if args else None`; check family/address; else original |
| `socket.getaddrinfo(host, *args, **kwargs)` | If host not local, raise `NetworkBlockedError`; else original |
| `socket.gethostbyname(host)` | Same |
| `socket.gethostbyname_ex(host)` | Same |

`_blocked(what)` (`src/offline_guard.py:83-86`):

```python
msg = f"Network access blocked (EasyScribe runs offline): {what}"
logger.warning(msg)
return NetworkBlockedError(msg)
```

`what` is exactly `f"connect to {address!r}"`, `f"send to {address!r}"`, or `f"DNS lookup for {host!r}"`.

`install()` also logs `"Offline guard active: outbound network access is blocked"` (`src/offline_guard.py:144`). In `main.py` this log is lost because logging is not configured yet; `main()` logs the flag again after `setup_logging()` (`src/main.py:93-96`).

`is_installed() -> bool` (`src/offline_guard.py:147-153`) returns the module-level `_installed` flag. It does not re-check the patches or environment. `main()` uses it to log either the active or the “NOT active” message (`src/main.py:93-96`). The docstring notes that `install()` may run before logging, so its own info line is normally lost.


### 5.6 Test-enforced constraints

`tests/test_offline_guard.py` enforces:

1. `install()` is idempotent (`tests/test_offline_guard.py:45-46`).
2. Blocked:
   - `socket.create_connection(("1.1.1.1", 443))`
   - `socket.socket().connect(("8.8.8.8", 53))`
   - `socket.socket().connect_ex(("8.8.8.8", 53))`
   - UDP `sendto(b"x", ("8.8.8.8", 53))`
   - `getaddrinfo("huggingface.co", 443)`
   - `gethostbyname("example.com")`
   - `connect(("example.com", 80))`
   - `urllib.request.urlopen("https://example.com")` raises `OSError`
   (`tests/test_offline_guard.py:48-68`)
3. Allowed:
   - loopback TCP connect and `getaddrinfo("localhost", port)`
   - `socket.socketpair()`
   (`tests/test_offline_guard.py:70-90`)
4. ffmpeg source checks (`tests/test_offline_guard.py:92-96`):
   - exact line `_LOCAL_FILES_ONLY: tuple[str, ...] = ("-protocol_whitelist", "file")`
   - exactly **two** occurrences of `*_LOCAL_FILES_ONLY,` (ffprobe and ffmpeg)
   - the ffmpeg command contains `*_LOCAL_FILES_ONLY,\n        "-i"` — whitelist before `-i`.
5. AST scan of every `src/*.py` and `launcher/*.py` (`tests/test_offline_guard.py:98-115`): no import whose top-level module is in
   `{requests, urllib, urllib3, http, httpx, aiohttp, ftplib, smtplib, webbrowser, huggingface_hub, socketserver, xmlrpc}`.
   `socket` and `ipaddress` are allowed.

### 5.7 Rust implementation strategy

Rust’s `std::net` has no global hook comparable to monkey-patching Python’s `socket.socket`. The rewrite must therefore enforce offline behavior architecturally:

1. **No network capability in the dependency graph.** Do not depend on `reqwest`, `hyper`, `ureq`, `curl`, `isahc`, `surf`, `awc`, `quinn`, `h2`, `tokio` with `net` feature, `async-std` with networking, `socket2` for outbound use, TLS crates, or DNS resolvers. Use `cargo deny`/`cargo tree` in CI with a deny-list. Be careful: some GUI/OS crates may pull `http`/`url` types transitively; the policy should ban socket-capable crates, not every crate that defines a URL type.
2. **A single `offline` module** owns process startup: set the environment variables and expose a `NetworkBlocked` error. All application code is forbidden from importing `std::net`/`tokio::net`/`socket2`; enforce with `#![deny(unused_imports)]` plus a CI source scan. There should be no socket code to guard.
3. **Local IPC must use pipes**, not TCP. The diarization child should communicate over stdin/stdout or anonymous/named pipes. If a future component needs a Unix domain socket, allow `AF_UNIX` only.
4. **Subprocess restrictions are explicit:**
   - ffmpeg/ffprobe argv must include `-protocol_whitelist file` exactly once each and before the input (`-i` for ffmpeg; positional input for ffprobe). Reject input paths containing a URI scheme (`://`) as defense in depth, even though ffmpeg’s whitelist already blocks network protocols.
   - whisper-cli receives only local filesystem paths and a local model path; do not pass URLs.
   - sherpa-onnx/onnxruntime models are local files; do not enable any remote model resolution.
5. **CI tests to replace the Python source/AST checks:**
   - A test that builds the ffmpeg command and asserts the exact `-protocol_whitelist file` position/count.
   - A test that the ffprobe command contains it before the input.
   - A workspace scan for banned network APIs/crates; fail on `std::net::TcpStream`, `UdpSocket`, `TcpListener`, `tokio::net`, `reqwest`, `hyper`, `ureq`, `curl`, `socket2`, `quinn`, etc.
   - A test that no URL is accepted as a media path (if the defense-in-depth check is implemented).
6. **Residual risk:** native libraries (ffmpeg, whisper.cpp, sherpa-onnx) could still perform network I/O through their own C/C++ stacks. Python cannot stop this either. Document that the guard covers the application and its immediate subprocess arguments, not a maliciously modified native binary. Optional OS-level sandboxing (Linux seccomp, Windows AppContainer/WFP) is defense in depth, not required for parity; if added, it must not break loopback `socketpair`-style internals.

### 5.8 What the guard does *not* do

- It does not block file system access outside the app.
- It does not sandbox subprocesses.
- It does not prevent a native library from using raw syscalls or its own HTTP stack.
- It does not firewall the machine.
- It does not affect `socket.socketpair()` on Windows if implemented through loopback/AF_UNIX; the reference intentionally allows loopback because Python internals need it (`src/offline_guard.py:10-13`).

---

## 6. Settings (`src/settings.py`)

### 6.1 File and keys

- File: `config.SETTINGS_FILE` = `<BASE_DIR>/recordings/settings.json` (`src/config.py:68`).
- Keys:
  - `"recordings_dir"` — string path or absent.
  - `"speaker_speed_factor"` — positive JSON number or absent.
- Missing or broken file means defaults; never fatal (`src/settings.py:1-7`).

### 6.2 `_load() -> dict`

`src/settings.py:21-29`:

1. `SETTINGS_FILE.read_text(encoding="utf-8")`, `json.loads`.
2. `FileNotFoundError` → `{}` (no warning).
3. `OSError` or `ValueError` → log warning `"Could not read settings, using defaults: {exc}"` → `{}`.
4. If the parsed value is not a `dict` → `{}`.
5. Otherwise return it unchanged (extra keys preserved).

Rust: read the whole file as UTF-8; on `NotFound` return empty; on I/O/parse error log the same warning text (with the underlying error display) and return empty; parse `serde_json::Value`, require an object.

### 6.3 `load_recordings_dir() -> Path | None`

`src/settings.py:32-45`:

- `value = data.get("recordings_dir")`.
- If not a non-empty `str` → `None`.
- `folder = Path(value)` (no `expanduser`, no `resolve`).
- If `folder.is_dir()` → return it.
- Otherwise log warning `"Chosen recordings folder is missing, using the default: {folder}"` → `None`.

A relative path is accepted relative to the process working directory; the GUI normally stores an absolute path. A path that exists but is a file → `None`.

### 6.4 `save_recordings_dir(folder: Path | None) -> None`

`src/settings.py:48-50`: `_save("recordings_dir", str(folder) if folder is not None else None)`.

- `None` removes the key.
- `str(folder)` uses platform path separators; a Windows path is stored with backslashes and JSON escaping.

### 6.5 `load_speaker_speed() -> float | None`

`src/settings.py:53-56`:

```python
value = _load().get("speaker_speed_factor")
return float(value) if isinstance(value, (int, float)) and value > 0 else None
```

- Accepts `int` and `float`; **`bool` is a subclass of `int` in Python**, so JSON `true` becomes `1.0` and is accepted. This is a **DEFECT/quirk**. Rust `serde_json` distinguishes booleans; to match, explicitly accept `Value::Bool(true)` as `1.0` and `false` as `None` if parity is required.
- Rejects zero, negative, strings, `null`, arrays/objects.
- `NaN` compares false (`NaN > 0` is false) → `None`. `Infinity > 0` is true → `inf`; Python’s `json.dumps` can write `Infinity` (non-standard), while `serde_json` cannot. See §6.8.

### 6.6 `save_speaker_speed(factor: float) -> None`

`src/settings.py:59-60`: `_save("speaker_speed_factor", round(factor, 3))`.

- Python `round(x, 3)` uses round-half-even on the decimal representation; the result is a float. Example: `round(1.2345, 3)` is `1.234` or `1.235` depending on binary representation; do not assume decimal arithmetic. For parity, implement Python’s algorithm or use a Rust decimal rounding helper with the same tie rule and test the values the app produces.
- The JSON writer emits the shortest round-trippable representation (`1.235`, `1.0`, etc.).

### 6.7 `_save(key, value) -> None`

`src/settings.py:63-72`:

1. `data = {k: v for k, v in _load().items() if k != key}` — preserves the insertion order of existing keys; removes the key being set.
2. If `value is not None`, append `data[key] = value`.
3. `SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)`.
4. `SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")`.
5. Any `OSError` → log warning `"Could not save settings: {exc}"`. Do not raise.

Formatting:

- `json.dumps(..., indent=2)` with default `ensure_ascii=True`, `separators=(',', ': ')` and newline `\n`.
- No trailing newline.
- Example:

```json
{
  "recordings_dir": "C:\\Users\\me\\Documents\\EasyScribe",
  "speaker_speed_factor": 1.235
}
```

- Non-ASCII paths are written as `\uXXXX` escapes because `ensure_ascii=True`; Rust `serde_json` writes UTF-8 by default. To match bytes exactly, configure the serializer to escape non-ASCII, or accept a documented formatting difference (the parsed value is identical). File bytes are externally observable; prefer matching.
- The write is **not atomic**: a crash mid-write can truncate/corrupt the file. Rust should write a temporary file and rename for durability; the resulting successful-file bytes are the same. Mark this as an approved robustness improvement, or write directly if byte-level crash behavior must match (it cannot be observed reliably).
- If the parent cannot be created or the file cannot be written, the app continues with defaults next launch.

### 6.8 Python JSON quirks to decide

- `json.loads` accepts `NaN`, `Infinity`, `-Infinity` by default. `serde_json` rejects them. Such values are not produced by this app except via a hand-edited file or an infinite `speaker_speed_factor`; reject them with a warning/default, or enable a permissive parser. Document the choice.
- Duplicate keys: Python’s parser keeps the last value; serde_json also keeps the last by default. Fine.
- Very large integers: Python arbitrary precision; Rust `serde_json::Number` preserves i64/u64/f64. `int` conversion for `samples_written` etc. elsewhere may overflow; see §15.

### 6.9 Tests

`tests/test_recordings_folder.py:63-77` monkey-patches `settings.SETTINGS_FILE` to a temporary path and asserts:

- no file → `None`;
- save an existing folder → load returns the same `Path`;
- save `None` → `None`;
- save a missing folder → load returns `None`;
- write `"{not json"` → load returns `None`.

Rust must make the settings path injectable for tests (e.g. a `SettingsStore` with a `PathBuf` field or a test-only override), because the Python test relies on module-global replacement.

---

## 7. On-disk formats

### 7.1 Raw microphone PCM (`.pcm`)

- File name: `recording_<YYYYMMDD_HHMMSS>.pcm` (`src/mic_recorder.py:92-94`).
- Content: headerless signed 16-bit little-endian mono samples at 16,000 Hz.
- Written by appending `chunk.tobytes()` from a NumPy `int16` array (`src/mic_recorder.py:179`, `186-198`).
- No WAV header, no length field. Length is derived from the sidecar.
- On little-endian platforms (all supported Windows/x86-64/ARM64), `int16` bytes are little-endian. Rust must write `i16::to_le_bytes()`; do not rely on native endianness if cross-platform parity matters.

### 7.2 JSON sidecar (`.json`)

- File name: same stem as the PCM, `.json` (`src/mic_recorder.py:94`).
- Content (compact `json.dumps`, no newline):

```json
{"sample_rate": 16000, "channels": 1, "bit_depth": 16, "samples_written": 123456}
```

- Key insertion order: `sample_rate`, `channels`, `bit_depth`, `samples_written` (`src/mic_recorder.py:139-144`, `209-214`).
- `samples_written` is the number of **mono samples** written to the PCM, incremented by `len(raw) // 2` per chunk (`src/mic_recorder.py:193`).
- `bit_depth` is written but **not read** by recovery; recovery assumes 16-bit/2 bytes (`src/recovery.py:42-53`).
- Updated:
  - after every 50 chunks (with `flush` + `fsync`) (`src/mic_recorder.py:195-198`);
  - once after the writer loop ends (`src/mic_recorder.py:200-202`).
- 50 chunks × 512 samples = 25,600 samples = 1.6 s at 16 kHz. Crash loss is at most ~1.6 s of audio plus the current unsynced chunk.
- The sidecar write is not atomic; a crash during `write_text` can leave invalid JSON. Recovery skips invalid JSON and leaves the pair for the next launch (`src/recovery.py:30-35`).

### 7.3 WAV files

All engine-facing WAVs are mono 16 kHz PCM signed 16-bit little-endian (`pcm_s16le`), format tag 1, standard RIFF/WAVE.

Producers:

- `ffmpeg -ac 1 -ar 16000 -vn -acodec pcm_s16le` (`src/ffmpeg_wrapper.py:181-192`).
- Fast path `shutil.copy2` of an input already validated as `pcm_s16le`, 16000 Hz, 1 channel (`src/ffmpeg_wrapper.py:126-134`, `158-163`).
- `recovery.recover_pcm_to_wav` writes a WAV with `nchannels`/`sampwidth=2`/`framerate` from metadata and exactly `samples_written * 2` bytes of PCM (`src/recovery.py:40-55`).
- `_extract_speaker_clip` copies the original WAV parameters and a slice of frames (`src/transcriber.py:151-160`).

Writers must not add extra chunks (LIST/INFO) or a different `fmt` layout if byte-for-byte parity of generated WAVs is required. Python’s `wave` module writes a minimal canonical header: `RIFF`, size, `WAVE`, `fmt ` (16 bytes, PCM), `data`, size. The ffmpeg output may include additional chunks (e.g. `LIST`) depending on build/input; do not normalize it.

### 7.4 Transcript (`.txt`)

- Encoding: UTF-8.
- Newlines: LF (`\n`) only; opened with `newline="\n"` (`src/transcriber.py:177`).
- Always ends with exactly one trailing `\n` appended by `_write_transcript` (`src/transcriber.py:177-179`). Builders do not add a trailing newline; if a caller passes text already ending in `\n`, the file ends with two.
- Empty transcript: the file contains a single `"\n"`.
- Content rules are in §11.2–§11.4.
- No JSON sidecar for transcripts, no metadata header, no BOM.

### 7.5 `settings.json`

See §6.7. Path: `<BASE_DIR>/recordings/settings.json`.

### 7.6 Logs

- Path: `<BASE_DIR>/logs/EasyScribe_<YYYYMMDD_HHMMSS>.log`.
- UTF-8, append mode, LF on all platforms? Python text mode on Windows translates `\n` to `\r\n` by default because `FileHandler` opens with `newline=None` (universal newlines translate on write). Rust should check: for byte-level parity on Windows, use `\r\n`? The reference log is a diagnostic file; more important is readable content. Document: Python’s `FileHandler` uses `open(..., 'a', encoding='utf-8')` with default newline translation, so Windows logs contain CRLF. If exact log bytes matter, use CRLF on Windows; otherwise use LF consistently and note the deviation.
- Rotation: up to 10 files (see §4.1).
- No timestamps in the file name beyond seconds.

### 7.7 Temporary files

All under `config.TEMP_DIR` = `<BASE_DIR>/temp` (`src/config.py:120`):

| Pattern | Producer | Cleanup |
|---|---|---|
| `<uuid4hex>.wav` | `extract_audio` fast path or ffmpeg (`src/ffmpeg_wrapper.py:161`, `177`) | GUI deletes after each file (`src/gui.py:1561-1566`); `main._cleanup_temp_files` at startup/exit; ffmpeg deletes on cancel/failure |
| `whispercpp_<uuid4hex>.json` | whisper-cli output (`src/whispercpp_wrapper.py:208-209`) | always `unlink(missing_ok=True)` in `finally` (`src/whispercpp_wrapper.py:290-291`) |
| `whispercpp_<uuid4hex>` (stem) | `--output-file` base; whisper-cli may create only `.json` in normal use | not explicitly deleted by the wrapper; cleanup patterns delete `*.json` but not extensionless files. **DEFECT/leak risk** if whisper-cli creates other extensions. |
| `spk_clip_<raw_speaker>.wav` | speaker naming (`src/transcriber.py:230`) | deleted in `finally` after callback (`src/transcriber.py:235-240`); startup cleanup catches leftovers |
| `probe_<uuid4hex>.bin` | GPU probe (`src/whispercpp_wrapper.py:147`) | `finally` in `_best_gpu` (`src/whispercpp_wrapper.py:160-161`); **not** matched by `main` cleanup (`*.wav`, `*.json`) |

UUIDs are `uuid.uuid4().hex`: 32 lowercase hex characters, no hyphens.

`_cleanup_temp_files` (`src/main.py:30-47`) globs only the top level of `temp/`, patterns `*.wav` and `*.json`, and ignores individual unlink errors. It does not recurse and does not delete `.bin`/extensionless files.

### 7.8 Output naming rules (GUI contract)

The backend receives explicit `output_path`; the GUI computes names. Rust backend APIs must accept an output path and not invent a different naming scheme.

1. **Files tab, default** (`src/gui.py:2156-2160`):
   - If an override exists for the input path, use it.
   - Else folder = chosen output folder, or the input file’s parent.
   - Name = `input_file.stem + ".txt"`.
   - Existing files are overwritten (open `"w"`).
2. **Best-quality transcript after a recording** (`src/gui.py:1792-1816`):
   - Folder = the recording’s parent (not the Files tab’s output folder).
   - Name = `audio.stem + ".txt"`.
   - If that exists, name = `audio.stem + " (best quality).txt"`.
   - If the second name also exists, it is overwritten (no further suffixing).
   - Options: timestamps **on**, diarization **off**.
   - This is run through the normal file pipeline with `_batch_recording = audio`; on success the timed segments are retained for “Find speakers”.
3. **Find speakers** (`src/gui.py:1895-1911`):
   - `output_path = transcript.with_name(f"{transcript.stem} (speakers).txt")`.
   - Always timestamps (via `add_speakers_to_file`), even if the source transcript had none.
   - Existing output is overwritten.
   - Examples:
     - `meeting.txt` → `meeting (speakers).txt`
     - `meeting (best quality).txt` → `meeting (best quality) (speakers).txt`
4. **Live recording transcript** (`src/gui.py:1709-1717`):
   - `saved_dir / f"{session_stem}.txt"` where `session_stem` is the PCM stem `recording_<YYYYMMDD_HHMMSS>`.
   - Content = each non-empty `on_segment` text on its own line, joined with `"\n"`, plus one trailing `"\n"`.
   - Only written if `transcript_lines` is non-empty and `session_stem` is available.
   - `saved_dir` is the WAV’s parent; if WAV saving fell back next to the PCM, the transcript goes there too.
5. **Record-only**:
   - No live transcript.
   - After the WAV is saved, the GUI immediately starts the best-quality file pipeline (§7.8.2), so the user gets `recording_...txt` or `recording_... (best quality).txt`.
6. **Rename set** (`src/rename.py:31-38`):
   - All `.wav`/`.txt` files in the folder whose stem equals the audio stem **or** starts with `audio.stem + " ("`.
   - Thus `<stem>.wav`, `<stem>.txt`, `<stem> (best quality).txt`, `<stem> (speakers).txt`, `<stem> (best quality) (speakers).txt` form one set.
   - `recording_..._2.wav` and `recording_....json` are excluded.
   - Renaming applies the same prefix replacement to every member.

No other suffix rules exist. `output_path` directories are created lazily by `_write_transcript` (`src/transcriber.py:173`).

### 7.9 Default folders

- Default recordings: `Documents/EasyScribe` (`src/config.py:70`).
- In-progress recordings: `<install>/recordings` (`src/config.py:65`).
- Startup recovery scans **only** `<install>/recordings` (`RECORDING_WORK_DIR`), not the user’s chosen output folder; a failed `MicRecorder.stop` leaves the raw pair there for recovery (`src/main.py:104-110`).
- Chosen recordings folder: persisted in `settings.json`; if missing on load, default is used (`src/settings.py:32-45`).
- `MicRecorder.stop(dest_dir=...)`: tries the chosen folder first; on any conversion failure, falls back to the PCM’s parent (`src/mic_recorder.py:145-163`). See §13.6.
- Recovery: tries the chosen/default recordings folder; on `OSError`, falls back next to the PCM (`src/recovery.py:78-91`).

---

## 8. FFmpeg wrapper (`src/ffmpeg_wrapper.py`)

### 8.1 Exceptions

| Type | Base | Raised when |
|---|---|---|
| `FFmpegNotFoundError` | `RuntimeError` | `validate_ffmpeg()` finds a missing binary (`src/ffmpeg_wrapper.py:33-34`, `83-90`) |
| `FFmpegExtractionError` | `RuntimeError` | ffmpeg returns nonzero, or any unexpected exception during extraction (`src/ffmpeg_wrapper.py:36-37`, `262-281`) |

`validate_ffmpeg()` messages, in order, joined by `"\n"`:

```
ffmpeg.exe not found at: {FFMPEG_BIN}
ffprobe.exe not found at: {FFPROBE_BIN}
```

`exists()` is used, not `is_file()`; a directory with that name passes.

### 8.2 `_CREATE_NO_WINDOW`

`_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0` (`src/ffmpeg_wrapper.py:28`). Passed as `creationflags` to every subprocess. See §20.2.

### 8.3 `_parse_duration(stderr_text) -> float`

`src/ffmpeg_wrapper.py:42-48`:

- Regex: `r"Duration:\s*(\d+):(\d+):([\d.]+)"`.
- Returns `h * 3600 + mn * 60 + float(s)`; no match → `0.0`.
- First match anywhere in the text.
- Examples: `"  Duration: 01:02:03.50, start"` → `3723.5`; `"no duration here"` → `0.0`.
- `N/A` does not match → `0.0`.
- Does not handle negative durations.
- A syntactically matched but non-numeric seconds field (for example `"00:00:1.2.3"`) reaches `float()` and raises `ValueError`. In `extract_audio` the first duration parse happens inside the stderr reader thread, so the exception kills that thread rather than failing extraction cleanly. **DEFECT/edge**; Rust should treat a malformed duration as unknown (`0.0`) and log a warning.

### 8.4 `_iter_lines(stream) -> Iterator[str]`

`src/ffmpeg_wrapper.py:51-69`:

1. Read chunks of 4096 bytes using `stream.read1(4096)` if available, else `stream.read(4096)`.
2. Accumulate bytes.
3. Replace `b"\r\n"` with `b"\n"`, then `b"\r"` with `b"\n"` (order matters: CRLF must not become two newlines).
4. Split on `b"\n"`; yield each complete line decoded UTF-8 with `errors="replace"`, then `.rstrip()`.
5. At EOF, yield any remaining buffer (also decoded and `.rstrip()`), even without a trailing newline.
6. Empty stream yields nothing. Consecutive newlines yield empty strings (the caller may filter).

Tests (`tests/test_ffmpeg_wrapper.py:51-67`) enforce:

- ffmpeg `-stats` progress lines ending in `\r` are split into separate lines; times `[10.0, 50.0, 100.0]`.
- A line split across arbitrary pipe-read sizes (1, 3, 7, 64 bytes) comes out identical.
- Empty stream → `[]`.

Rust: do not use `BufRead::lines()` directly; it only splits `\n` and will merge `\r` progress updates. Implement a byte buffer and CR/LF normalization exactly.

### 8.5 `_parse_time(line) -> float`

`src/ffmpeg_wrapper.py:72-78`:

- Regex: `r"time=(\d+):(\d+):([\d.]+)"`.
- Returns seconds or `-1.0` if absent.
- Matches the first `time=` on the line.
- Does not accept a leading `-` in the timestamp; ffmpeg’s negative start times are ignored (`-1.0`).
- A malformed numeric seconds field raises `ValueError` in the reader thread. **DEFECT/edge**; Rust should return `-1.0` and log a warning.

### 8.6 `_LOCAL_FILES_ONLY`

`src/ffmpeg_wrapper.py:93-96`:

```python
_LOCAL_FILES_ONLY: tuple[str, ...] = ("-protocol_whitelist", "file")
```

This must appear:

- once in the ffprobe command, before the positional input path;
- once in the ffmpeg command, before `-i`.

Tests grep for exactly two uses (`tests/test_offline_guard.py:92-96`). Rust should define a `const LOCAL_FILES_ONLY: [&str; 2]` and a command builder test.

### 8.7 `probe_audio(input_path) -> dict`

`src/ffmpeg_wrapper.py:99-123`:

```
FFPROBE_BIN
  -v error
  -select_streams a:0
  -show_entries stream=codec_name,sample_rate,channels,duration
  -of json
  -protocol_whitelist file
  <input_path>
```

- If `FFPROBE_BIN` does not exist → return `{}` (no exception, no log).
- `subprocess.run(..., capture_output=True, timeout=30, creationflags=...)`.
- Decode stdout UTF-8 with `errors="replace"`; `json.loads`; `data.get("streams", [])`; return `streams[0]` if any.
- Any exception (missing file, timeout, bad JSON, no audio stream, permission) → `logger.debug("ffprobe failed for {input_path.name}: {exc}")` and return `{}`.
- The returned dict has string keys and values as produced by ffprobe (`sample_rate` is typically a number, `channels` a number, `duration` a string or absent).
- It does not check `returncode`; a nonzero ffprobe with JSON on stdout would still parse if valid.

Rust: `Command::output` with a 30 s timeout. `std::process::Command::output` has no timeout; use a wait loop with `try_wait`, kill the child on timeout, then return `{}`. Preserve the debug log text. `-protocol_whitelist file` is placed after `-of json` and before the input, exactly as above.

### 8.8 `_is_already_suitable_wav(input_path) -> bool`

`src/ffmpeg_wrapper.py:126-134`:

```python
if input_path.suffix.lower() != ".wav": return False
info = probe_audio(input_path)
return (info.get("codec_name") == "pcm_s16le"
        and str(info.get("sample_rate", "0")) == "16000"
        and str(info.get("channels", "0")) == "1")
```

- Requires exact codec string `pcm_s16le`.
- Converts values to strings, so numeric `16000` and string `"16000"` both pass; `16000.0` would fail (`"16000.0"`).
- Missing keys default to `"0"` and fail.
- Case-sensitive `.wav` check is lowercased first, so `.WAV` passes; `.wave` does not.

### 8.9 `extract_audio(input_path, cancel_event, log_callback=None) -> Path`

`src/ffmpeg_wrapper.py:137-281`.

#### Preconditions and fast path

1. If `cancel_event.is_set()` → `CancelledError("Cancelled before audio extraction started")` (`148-149`).
2. `validate_ffmpeg()` (`151`).
3. Define `log(msg)` = `logger.info(msg)` + `log_callback(msg)` if provided (`153-156`).
4. If `_is_already_suitable_wav(input_path)` (`159`):
   - Log `"[ffmpeg] Already mono 16 kHz PCM WAV — copying directly: {input_path.name}"`.
   - `temp_wav = TEMP_DIR / f"{uuid.uuid4().hex}.wav"`.
   - `shutil.copy2(input_path, temp_wav)`.
   - Return `temp_wav`.
   - No duration/probe logging, no progress, no try/except around `copy2`. A copy failure propagates as a raw `OSError` (not `FFmpegExtractionError`). **DEFECT/unwrapped error**; Rust should wrap it in `FFmpegExtractionError` unless exact error taxonomy is required.

#### Probe logging

5. `probe = probe_audio(input_path)` (`166`).
6. If `probe` is truthy:
   - `codec = probe.get("codec_name", "?")`
   - `rate = probe.get("sample_rate", "?")`
   - `chans = probe.get("channels", "?")`
   - `dur_raw = probe.get("duration", "")`
   - `dur_str = fmt_duration(float(dur_raw)) if dur_raw else "unknown duration"`
   - Log `"[ffmpeg] Input: codec={codec}, sample_rate={rate} Hz, channels={chans}, duration={dur_str}"`.
   - If `dur_raw` is nonempty but not float-convertible, `float()` raises **outside** the main try block and propagates raw. **DEFECT**; Rust should catch and use `"unknown duration"` or wrap.
7. Else log `"[ffmpeg] Could not probe audio stream — proceeding anyway"`.

#### Command

`src/ffmpeg_wrapper.py:181-192` exact argv order:

```
FFMPEG_BIN
  -y
  -protocol_whitelist file
  -i <input_path>
  -ac 1
  -ar 16000
  -vn
  -acodec pcm_s16le
  -stats
  <temp_wav>
```

- `temp_wav = TEMP_DIR / f"{uuid.uuid4().hex}.wav"`.
- Logs:
  - `"[ffmpeg] Extracting audio → mono 16 kHz PCM WAV: {input_path.name}"`
  - `"[ffmpeg] Temp file: {temp_wav.name}"`
- `logger.debug("ffmpeg cmd: " + " ".join(cmd))` (`194`). Note this is a naive join without quoting; paths with spaces appear ambiguous but the actual `argv` is exact.
- `stdout=DEVNULL`, `stderr=PIPE`, `creationflags`. No stdin redirection (inherits); no `text=True`.
- `-stats` writes progress to stderr with `\r`; `-y` overwrites the temp file (UUID makes collision unlikely).
- `-vn` disables video; `-acodec pcm_s16le` after `-vn`.

#### Live stderr reader

`src/ffmpeg_wrapper.py:206-242`:

- `stderr_lines: list[str]`, `total_duration: [0.0]`, `last_progress_log: [0.0]` shared with the thread.
- Reader thread iterates `_iter_lines(proc.stderr)`:
  - Append each line.
  - If `total_duration[0] == 0.0`, parse `_parse_duration(line)`; if `> 0`, set and log `"[ffmpeg] Media duration: {fmt_duration(d)}"`.
  - Parse `pos = _parse_time(line)`.
  - If `pos >= 0 and total_duration[0] > 0`:
    - `now = time.monotonic()`.
    - If `now - last_progress_log[0] >= 1.0`:
      - update last;
      - `pct = min(pos / total_duration[0] * 100, 100)`;
      - log `"[ffmpeg] Extracting… {fmt_duration(pos)} / {fmt_duration(total_duration[0])}  ({pct:.0f}%)"` (two spaces before the parenthesis).
  - Else if `line` is nonempty and does **not** start with `"ffmpeg version"` and does **not** contain `"encoder"` case-insensitively:
    - If the lowercased line contains any of `"error"`, `"warn"`, `"invalid"`, `"no such"`, `"stream #"`, log `"[ffmpeg] {line}"`.
- Reader thread is daemon.
- Only the first positive duration is recorded. If ffmpeg prints duration after the first progress line, progress may be suppressed until it is found.
- Progress log is throttled to at most one per second; the first progress line can log immediately because `last_progress_log` starts at 0.0.
- `pct` is capped at 100 but not rounded before formatting; `.0f` uses Python’s round-half-even formatting (Rust’s `format!("{:.0}")` rounds half-to-even? Rust formatting uses round-half-to-even for `Display` floats; verify with tests).
- Progress goes only to the log callback (`log`), **not** to a numeric progress callback. The GUI leaves the progress bar at 0 during extraction; status is “Extracting Audio”.

#### Cancellation polling

`src/ffmpeg_wrapper.py:244-257`:

- While `proc.poll() is None`, check every `0.1` s.
- If cancel:
  - `logger.info("Cancel requested — terminating ffmpeg")`.
  - `proc.terminate()`; `proc.wait(timeout=5)`; on any exception `proc.kill()` (no wait after kill).
  - `reader_thread.join(timeout=2)`.
  - `temp_wav.unlink(missing_ok=True)`.
  - Raise `CancelledError("Audio extraction cancelled by user")`.
- Latency: up to 100 ms + terminate/5 s + 2 s join.

#### Completion and errors

`src/ffmpeg_wrapper.py:259-281`:

- `reader_thread.join(timeout=5)`; `return_code = proc.returncode`.
- If nonzero:
  - `stderr_text = "\n".join(stderr_lines)`;
  - `snippet = stderr_text[-800:].strip()` (last 800 **characters**, after UTF-8 replacement);
  - `logger.error("ffmpeg failed (rc={rc}):\n{snippet}")`;
  - unlink temp if exists;
  - raise `FFmpegExtractionError(f"ffmpeg exited with code {return_code}.\n\n{snippet}")`.
- Success:
  - `size_kb = temp_wav.stat().st_size // 1024`;
  - log `"[ffmpeg] Extraction complete — {size_kb:,} KB written to {temp_wav.name}"` (`:,` thousands separators);
  - return `temp_wav`.
- Exception handling:
  - `CancelledError` and `FFmpegExtractionError` re-raise unchanged.
  - Any other `Exception`:
    - unlink temp if exists;
    - raise `FFmpegExtractionError(f"Unexpected error during audio extraction: {exc}")` with `from exc`.
  - `KeyboardInterrupt`/`SystemExit` (BaseException, not Exception) are not caught and do not clean temp.

#### Temp file lifecycle

- On success, the caller owns the temp WAV. The GUI deletes it in `finally` after each file (`src/gui.py:1561-1566`).
- On failure/cancel, the wrapper deletes it.
- Startup/exit cleanup also removes leftovers (`src/main.py:30-47`).
- The fast path copy is not inside the cleanup wrapper; if the process crashes after copy, startup cleanup removes it.

### 8.10 FFmpeg command examples

Input `C:\Users\me\My Meeting.mp4`:

```
C:\EasyScribe\ffmpeg\ffmpeg.exe
-y
-protocol_whitelist
file
-i
C:\Users\me\My Meeting.mp4
-ac
1
-ar
16000
-vn
-acodec
pcm_s16le
-stats
C:\EasyScribe\temp\3f2a...<32 hex>.wav
```

Each token is a separate `argv` element; no shell quoting. Rust `Command::arg` must be used, never a shell string.

### 8.11 Tests

`tests/test_ffmpeg_wrapper.py:51-80` covers `_iter_lines`, `_parse_duration`, `_parse_time`. There is no unit test for the ffmpeg subprocess itself; end-to-end coverage is manual/CI.

---

## 9. VAD (`src/vad.py`)

### 9.1 Constants

`src/vad.py:24-39`:

| Constant | Value | Meaning |
|---|---|---|
| `VAD_THRESHOLD` | `0.5` | Silero speech probability threshold |
| `VAD_MIN_SILENCE_SEC` | `0.5` | Silence needed to close a segment |
| `VAD_MIN_SPEECH_SEC` | `0.25` | Minimum speech duration |
| `VAD_PAD_SEC` | `0.2` | Segment padding |
| `PAD_SAMPLES` | `int(0.2 * 16000)` = `3200` | Padding in samples |
| `VAD_SAMPLE_RATE` (config) | `16000` | |
| `VAD_CHUNK_SAMPLES` (config) | `512` | Feed window |

Invariant, asserted at import (`src/vad.py:34-37`):

```python
assert VAD_PAD_SEC < VAD_MIN_SILENCE_SEC / 2
```

`0.2 < 0.25` is true. Test `tests/test_vad_bounds.py:45-46` re-asserts it. Rust must enforce this at compile time or startup:

```rust
const _: () = assert!(VAD_PAD_SEC < VAD_MIN_SILENCE_SEC / 2.0);
```

The comment explains the purpose: two adjacent segments are separated by at least `VAD_MIN_SILENCE_SEC`; padding each by less than half the minimum silence guarantees padded segments cannot overlap.

### 9.2 `build_vad(buffer_size_in_seconds=None)`

`src/vad.py:42-59`:

- Lazy `import sherpa_onnx`.
- Config:

```python
sherpa_onnx.VadModelConfig(
    silero_vad=sherpa_onnx.SileroVadModelConfig(
        model=str(config.VAD_MODEL_PATH),
        threshold=0.5,
        min_silence_duration=0.5,
        min_speech_duration=0.25,
    ),
    sample_rate=16000,
)
```

- If `buffer_size_in_seconds is None`: `sherpa_onnx.VoiceActivityDetector(vad_config)`.
  - The pybind signature default is `buffer_size_in_seconds=60` (verified in the installed 1.13.x binding). So file mode uses a **60-second** internal circular buffer.
- Else: `VoiceActivityDetector(vad_config, buffer_size_in_seconds=buffer_size_in_seconds)`.
  - Live mode passes `30` (`src/live_transcriber.py:118`).
- No `num_threads`, `debug`, `window_size`, or provider settings; library defaults apply.

Rust: bind the sherpa-onnx C API `SherpaOnnxVadModelConfig` / `SherpaOnnxVoiceActivityDetector`. Set `silero_vad.model`, `threshold`, `min_silence_duration`, `min_speech_duration`, `sample_rate`; pass buffer size 60 by default for file mode and 30 for live mode. Do not substitute a different VAD unless output parity is proven; the model is `models/silero_vad.onnx`.

### 9.3 `file_segments(vad_detector, samples, cancel_event) -> list[(start_sample, end_sample)]`

`src/vad.py:62-92`:

```
window = 512
bounds = []
pos = 0
while pos < len(samples):
    if cancel_event.is_set(): raise CancelledError("VAD segmentation cancelled by user")
    end = min(pos + window, len(samples))
    chunk = samples[pos:end]
    if len(chunk) < window:
        chunk = np.pad(chunk, (0, window - len(chunk)))   # zeros, same dtype
    vad_detector.accept_waveform(chunk)
    pos = end
    while not vad_detector.empty():
        seg = vad_detector.front
        bounds.append((seg.start, seg.start + len(seg.samples)))
        vad_detector.pop()

vad_detector.flush()
while not vad_detector.empty():
    seg = vad_detector.front
    bounds.append((seg.start, seg.start + len(seg.samples)))
    vad_detector.pop()
return bounds
```

Exact details:

- Cancellation is checked once per 512-sample window (32 ms), **before** feeding that window. Cancellation after the last window but before `flush()` is not observed.
- The final partial window is right-padded with zeros to exactly 512 samples. This slightly lengthens the processed timeline; VAD may produce a final segment including those zeros.
- `seg.start` is the absolute input sample index since the detector was created (“Start sample index relative to the processed audio timeline” in the C API), not relative to the current buffer.
- `seg.samples` is used only for its length; `seg.start + len(seg.samples)` is the end.
- Segments are appended in detector order. The detector’s `front` property is valid only until the next call to any method; copy values before `pop`.
- `flush()` processes buffered audio and emits the final segment(s).
- Empty input (`len(samples)==0`) skips the loop, calls `flush()`, drains any segments (normally none), returns `[]`.
- The function does not merge overlapping/adjacent bounds, sort them, or filter zero-length segments.
- The input is a mono float32 NumPy array. `np.pad` preserves dtype and pads with 0.0.

Rust: feed `&[f32]` slices of 512; pad the final one with zeros. Drain after each feed and after flush. Use the same cancellation message. Preserve segment order.

### 9.4 `pad_and_clamp(bounds, total_samples) -> list[(start, end)]`

`src/vad.py:95-107`:

```python
padded = []
for i, (b_start, b_end) in enumerate(bounds):
    prev_end = bounds[i - 1][1] if i > 0 else 0
    next_start = bounds[i + 1][0] if i + 1 < len(bounds) else total_samples
    c_start = max(prev_end, b_start - PAD_SAMPLES)
    c_end = min(next_start, b_end + PAD_SAMPLES)
    padded.append((c_start, c_end))
return padded
```

Key points:

- Pads **both** sides by 3200 samples (0.2 s).
- Clamps against the **original** previous end and next start, not against already-padded values.
- First segment’s lower bound is 0; last segment’s upper bound is `total_samples`.
- Never extends beyond `[0, total_samples]` because `prev_end >= 0` and `next_start <= total_samples` for well-formed sorted bounds.
- Does not merge, drop, or reorder.
- For well-formed bounds with gaps ≥ `VAD_MIN_SILENCE_SEC` (8000 samples), padded segments cannot overlap because `PAD_SAMPLES (3200) < 4000`.
- If bounds are unsorted or overlapping, the formula can produce `c_end < c_start` or overlap; the function does not validate. Callers only pass detector output, which is sorted and non-overlapping.

Tests `tests/test_vad_bounds.py:49-113` enforce:

- single segment pads both sides and clamps at 0/total;
- adjacent segments at exactly `8000` samples gap do not overlap;
- far-apart segments;
- start clamped to 0;
- end clamped to total;
- ten adjacent segments;
- empty bounds → `[]`;
- output length equals input length;
- `0 <= start <= end <= total`;
- sorted and non-overlapping (`padded[i].end <= padded[i+1].start`).

Rust signature: `pub fn pad_and_clamp(bounds: &[(i64, i64)], total_samples: i64) -> Vec<(i64, i64)>` (or `usize` with checked arithmetic; sample indices are nonnegative and fit in `i64`/`usize` for any realistic file). Use saturating subtraction for `b_start - PAD_SAMPLES` to avoid underflow on unsigned types; Python’s `max(prev_end, b_start - 3200)` with nonnegative `b_start` would conceptually clamp at 0 anyway.

### 9.5 `pad_live_segment(history, history_start, prev_segment_end, seg_start, seg_samples) -> np.ndarray`

`src/vad.py:110-128`:

```python
pad_start = max(prev_segment_end, seg_start - PAD_SAMPLES, history_start)
if pad_start >= seg_start:
    return seg_samples
lead_in = history[pad_start - history_start : seg_start - history_start]
return np.concatenate([lead_in, seg_samples])
```

- `history` is a rolling float32 buffer whose first element is absolute sample index `history_start`.
- `prev_segment_end` is the **original** end (not padded end) of the previous segment; this prevents leading padding from reaching into the previous segment.
- `history_start` prevents reading older samples than buffered.
- Only the **leading** edge is padded in live mode; no trailing padding.
- If no padding is available (`pad_start >= seg_start`), returns the **same array object**, not a copy. Rust may return a borrowed slice or a new `Vec`; callers only read.
- Slicing uses absolute indices: `history[pad_start - history_start : seg_start - history_start]`. If `seg_start > history_start + len(history)` (should not happen), NumPy truncates the slice at the end; it does not panic.
- `lead_in` and `seg_samples` are concatenated in that order; dtype float32.
- Negative indices cannot occur because `pad_start >= history_start`; if `seg_start < history_start`, then `pad_start >= history_start > seg_start` → no padding.

Tests `tests/test_live_transcriber.py:35-83` enforce:

- history only after `seg_start` → unchanged;
- normal padding uses exactly `PAD_SAMPLES` preceding samples;
- `prev_segment_end` clamps the lead-in;
- `history_start` clamps the lead-in.

### 9.6 VAD mode differences

| Aspect | File mode | Live mode |
|---|---|---|
| Detector buffer | 60 s (library default) | 30 s |
| Feed window | 512 samples, zero-padded final | 512-sample mic chunks (may be other sizes in tests) |
| Padding | Both sides via `pad_and_clamp` | Leading only via `pad_live_segment` |
| History | Whole file in memory | Rolling `_HISTORY_SAMPLES = 11392` samples (0.712 s) |
| Cancellation | Checked per window | Stop event ends loop after draining queue |
| Segment queue | Direct list | Bounded decode queue max 8, drop oldest |

### 9.7 Rust notes

- `sherpa-rs`/C API segment `start` must be read before `pop`; the Python binding’s `front` is a reference valid only until the next call.
- If the Rust binding exposes `SpeechSegment` by value/clone, that is fine, but do not retain a borrowed reference across `pop`.
- Keep the VAD detector and recognizer behind the same mutex in live mode; see §1.3 and §14.
- `PAD_SAMPLES` must be exactly 3200, not recomputed with a different float rounding. Use `const PAD_SAMPLES: usize = 3200;` and assert `PAD_SAMPLES as f64 / 16000.0 == VAD_PAD_SEC`.
- The zero-pad final window is part of the processed timeline; do not drop the padded tail before VAD.

---

## 10. whisper.cpp wrapper (`src/whispercpp_wrapper.py`)

### 10.1 Purpose

File transcription uses the bundled `whisper-cli` when both the binary and GGUF model exist (`is_available()`). It runs one subprocess for the whole file; whisper.cpp does its own 30-second windowing and timestamping. The wrapper parses `--output-json` into `(start_sec, end_sec, text)` segments, the same shape as the sherpa VAD loop, so transcript formatting and diarization assignment are unchanged (`src/whispercpp_wrapper.py:1-11`).

Implementation note: `REWRITE_PLAN.md` proposes `whisper-rs`. That is acceptable only if its segmentation/timestamps and the `Engine:`/`GPU:`/`CPU:` log information are verified equivalent to the reference; otherwise invoke the bundled `whisper-cli` with the exact argv in §10.10.

### 10.2 Exceptions

| Type | Base | Raised when |
|---|---|---|
| `WhisperCppNotFoundError` | `RuntimeError` | binary/model missing in `validate_whispercpp()` |
| `WhisperCppError` | `RuntimeError` | nonzero exit, missing JSON, or unexpected error |

`validate_whispercpp()` messages, joined by `"\n"`, in order:

```
whisper-cli not found at: {WHISPERCPP_BIN}
Whisper GGUF model not found at: {WHISPERCPP_MODEL}
```

Uses `is_file()`, unlike `validate_ffmpeg`’s `exists()`.

`is_available()` (`src/whispercpp_wrapper.py:55-57`): `WHISPERCPP_BIN.is_file() and WHISPERCPP_MODEL.is_file()`.

### 10.3 Exact regexes

`src/whispercpp_wrapper.py:46-52`:

```python
_USING_RE = re.compile(r"whisper_backend_init_gpu: using (\S+) backend")
_NO_GPU_RE = re.compile(r"whisper_backend_init_gpu: no GPU found")   # defined but unused
_VULKAN_DEV_RE = re.compile(r"ggml_vulkan: (\d+) = (.+?) \| (.*)")
_SYSTEM_INFO_RE = re.compile(r"system_info: n_threads = (\d+) / (\d+) \|(.*)")
_PROGRESS_RE = re.compile(r"whisper_print_progress_callback: progress = (\d+)%")
```

- `_NO_GPU_RE` is dead code. Do not rely on it; “no GPU found” is inferred from the absence of a `using ... backend` line.
- `_VULKAN_DEV_RE` requires a literal `" | "` between the device name and details. A device line without details does not match and is ignored.
- The device name group is non-greedy (`(.+?)`), so it stops at the first `" | "`.
- `_SYSTEM_INFO_RE` requires `system_info: n_threads = N / M |`; group 3 is everything after the first pipe.
- `_PROGRESS_RE` matches the exact whisper.cpp progress prefix; other progress formats are ignored.

### 10.4 `_cpu_name() -> str`

`src/whispercpp_wrapper.py:70-85`:

1. Windows:
   - `winreg.OpenKey(HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")`
   - `QueryValueEx(key, "ProcessorNameString")[0]`, `str(...).strip()`.
   - Returns it immediately.
2. Non-Windows:
   - Read `/proc/cpuinfo` as text (default encoding).
   - Find the first line starting with `"model name"`; return the part after the first `":"`, stripped.
   - If no line, fall through.
3. Any exception: fall through.
4. `platform.processor() or "unknown CPU"`.

Rust mapping:

- Windows: registry `HKEY_LOCAL_MACHINE\HARDWARE\DESCRIPTION\System\CentralProcessor\0`, value `ProcessorNameString` (`winreg` crate or `windows` crate). Strip whitespace.
- Linux: read `/proc/cpuinfo`, find `model name`, split at first `:`.
- macOS/other: `sysctl -n machdep.cpu.brand_string` (or `std::env` fallback) and finally a generic `"unknown CPU"`.
- Never make a network call.
- CPU name is only for logs; exact failure text is not critical, but the log line format is.

### 10.5 `_describe_devices(lines) -> list[str]`

`src/whispercpp_wrapper.py:88-122`. Input is the accumulated stderr lines (including the current `system_info` line).

Algorithm:

1. `using` = first match group 1 of `_USING_RE`, or `None`.
2. `vulkan` = dict `{int(index): (name, details)}` for every `_VULKAN_DEV_RE` match; later duplicate index overwrites earlier.
3. `info` = first `_SYSTEM_INFO_RE` match object, or `None`.
4. `out = []`.
5. If `using` starts with `"Vulkan"`:
   - `suffix = using[len("Vulkan"):]` (e.g. `"0"`).
   - `name, details = vulkan.get(suffix, (None, ""))`.
   - `out.append(f"Engine: Vulkan (GPU {using})")`.
   - `out.append(f"GPU: {name or 'name not reported'}" + (f" | {details}" if details else ""))`.
   - If the device line is missing entirely, `name is None` → `name not reported`, and details empty → no suffix.
6. Elif `using` is truthy:
   - `out.append(f"Engine: {using}")` (e.g. `Engine: CPU` if whisper.cpp says `using CPU backend`).
7. Else:
   - `out.append("Engine: CPU (no GPU found)")`.
8. For each `(idx, (name, details))` in `sorted(vulkan.items())`:
   - If `f"Vulkan{idx}" != using`, append `f"Other Vulkan device {idx} (not used): {name} | {details}"`.
9. `cpu = f"CPU: {_cpu_name()}"`.
10. If `info`:
    - `features = [f.split("=")[0].split()[-1] for f in info.group(3).split("|") if f.strip().endswith("= 1")]`.
    - `cpu += f" | {info.group(1)} of {info.group(2)} threads"`.
    - If `features`: `cpu += " | " + " ".join(features)`.
11. Append `cpu` to `out`; return.

Exact test expectations (`tests/test_whispercpp_wrapper.py:86-121`):

Input lines:

```
ggml_vulkan: Found 2 Vulkan devices:
ggml_vulkan: 0 = NVIDIA GeForce RTX 5070 (NVIDIA) | uma: 0 | fp16: 1 | matrix cores: NV_coopmat2
ggml_vulkan: 1 = AMD Radeon(TM) Graphics (AMD proprietary driver) | uma: 1 | fp16: 1
whisper_backend_init_gpu: device 0: CPU (type: 0)
whisper_backend_init_gpu: device 1: Vulkan0 (type: 1)
whisper_backend_init_gpu: using Vulkan0 backend
system_info: n_threads = 8 / 16 | WHISPER : COREML = 0 | CPU : SSE3 = 1 | AVX2 = 1 | AVX512 = 0 |
```

Output:

```
Engine: Vulkan (GPU Vulkan0)
GPU: NVIDIA GeForce RTX 5070 (NVIDIA) | uma: 0 | fp16: 1 | matrix cores: NV_coopmat2
Other Vulkan device 1 (not used): AMD Radeon(TM) Graphics (AMD proprietary driver) | uma: 1 | fp16: 1
CPU: <cpu name> | 8 of 16 threads | SSE3 AVX2
```

CPU-only input:

```
whisper_backend_init_gpu: device 0: CPU (type: 0)
whisper_backend_init_gpu: no GPU found
system_info: ...
```

Output:

```
Engine: CPU (no GPU found)
CPU: <cpu name> | ...
```

No device name line (`using Vulkan0` only) → second line `GPU: name not reported`.

Feature parsing details:

- Split group 3 on `"|"`; trim each; keep only pieces ending in `"= 1"`.
- Take `piece.split("=")[0].split()[-1]` → the token immediately before `=`.
- Example: `" CPU : SSE3 = 1 "` → `"SSE3"`, `" AVX2 = 1 "` → `"AVX2"`, `" AVX512 = 0 "` excluded.
- If a piece is malformed (`"= 1"` with no token), `split()[-1]` raises `IndexError`; the reader thread would die. **DEFECT**; Rust should parse defensively and skip malformed features, or match and propagate consistently.
- `_describe_devices` is called every time a `system_info` line is seen, passing all lines accumulated so far. If multiple `system_info` lines occur, device lines repeat in the log.

### 10.6 `_pick_gpu(lines) -> int | None`

`src/whispercpp_wrapper.py:125-141`:

1. `devices = [(int(idx), details) for each _VULKAN_DEV_RE match]`.
2. No devices → `None`.
3. Ranking key for `(idx, details)`:
   ```python
   ("uma: 1" not in details,          # True = dedicated, preferred
    "matrix cores: none" not in details,  # True = has matrix cores (or unspecified), preferred
    -idx)                            # larger key wins; -idx means smaller index wins on tie
   ```
4. Return `max(devices, key=rank)[0]`.

Consequences:

- Dedicated GPU (no `uma: 1`) beats integrated (`uma: 1`).
- If both are dedicated, one with matrix cores (or no “matrix cores: none” text) wins.
- If still tied, lowest Vulkan index wins (because `-idx` is larger for smaller idx).
- A device line lacking `matrix cores` entirely is considered as having matrix cores, because `"matrix cores: none" not in details` is `True`. This is intentional according to the code comment? It can mis-rank; document exact.
- Test `tests/test_whispercpp_wrapper.py:123-135`:
  - integrated `uma: 1` index 0, dedicated index 1 → 1.
  - only integrated → 0.
  - no device lines → `None`.
  - two dedicated with `matrix cores: none` → 0 (tie by lowest index).

### 10.7 `_best_gpu() -> int | None`

`src/whispercpp_wrapper.py:144-161`:

- Decorated `@functools.lru_cache(maxsize=1)`; the first result is cached for the process lifetime.
- Creates `TEMP_DIR / f"probe_{uuid.uuid4().hex}.bin"`.
- `TEMP_DIR.mkdir(parents=True, exist_ok=True)`.
- Writes an empty file (`b""`).
- Runs:
  ```
  WHISPERCPP_BIN -m <empty_model> -f <empty_model>
  ```
  `capture_output=True`, `timeout=30`, `creationflags`.
- The binary lists Vulkan devices, then fails/stops at the empty model. No model is loaded.
- Decodes stderr UTF-8 replace, splits lines, passes to `_pick_gpu`.
- Any exception (missing binary, timeout, permissions) → `logger.warning("GPU probe failed, whisper.cpp picks the device: {exc}")` → `None`.
- `finally`: `empty_model.unlink(missing_ok=True)`.
- No `-t`, `-bs`, `-dev`, `-pp` in the probe.

Rust: use `OnceLock<Option<i32>>` (or `OnceCell`) instead of `lru_cache`. Use the same probe file and cleanup. On timeout, kill the child before returning.

### 10.8 `_parse_progress_line(line) -> int | None`

`src/whispercpp_wrapper.py:164-166`: first `_PROGRESS_RE` match → `int(group(1))`; else `None`.

Tests: `"whisper_print_progress_callback: progress = 100%"` → 100; `" ... = 42%"` → 42; unrelated → `None`.

### 10.9 `_parse_output_json(json_path) -> list[(float, float, str)]`

`src/whispercpp_wrapper.py:169-181`:

1. `json.loads(path.read_text(encoding="utf-8"))`.
2. `for entry in data.get("transcription", [])`:
   - `text = entry.get("text", "").strip()`.
   - If empty → skip. This removes whitespace-only entries.
   - `offsets = entry.get("offsets", {})`.
   - `start = offsets.get("from", 0) / 1000.0`.
   - `end = offsets.get("to", 0) / 1000.0`.
   - Append `(start, end, text)`.
3. Return.

Notes:

- Whisper JSON offsets are milliseconds; divide by 1000.0 to seconds.
- Missing offsets default to 0.0; missing transcription key → empty list.
- Negative offsets are not validated.
- `text` is stripped at both ends; internal whitespace/punctuation preserved.
- `entry` need not be a dict; a non-dict raises `AttributeError`, caught by the outer `transcribe_file` as `WhisperCppError` (unexpected).
- The JSON file may contain `timestamps` strings; they are ignored.
- Test (`tests/test_whispercpp_wrapper.py:40-81`) expects:
  ```
  [(0.0, 5.42, "Hello, my name is John."),
   (6.38, 11.16, "Sure, I would be happy.")]
  ```
  and skips the blank third entry.

### 10.10 `transcribe_file(audio_path, cancel_event, log_callback=None, progress_callback=None) -> list[(float, float, str)]`

`src/whispercpp_wrapper.py:186-291`.

#### Preconditions

- If `cancel_event.is_set()` → `CancelledError("Cancelled before transcription started")`.
- `validate_whispercpp()`.

#### Output paths

- `out_json = TEMP_DIR / f"whispercpp_{uuid.uuid4().hex}.json"`.
- `out_stem = out_json.with_suffix("")` → `TEMP_DIR / f"whispercpp_{uuid.uuid4().hex}"` (no extension).
- `--output-file` receives `out_stem`; whisper-cli appends `.json` for `--output-json`.

#### Command

`src/whispercpp_wrapper.py:211-221` exact order:

```
WHISPERCPP_BIN
  -m <WHISPERCPP_MODEL>
  -f <audio_path>
  -bs 5
  -t <NUM_THREADS>
  [ -dev <gpu> ]          # only if _best_gpu() is Some
  -pp
  --output-json
  --output-file <out_stem>
```

- `_best_gpu()` is evaluated during list construction; it runs the probe once per process and may take ~0.1 s (comment).
- `-dev` is omitted entirely when `_best_gpu()` returns `None`.
- No `-l`/language, no `--threads` long form, no `-ng`, no `--no-prints`.
- `logger.debug("whisper-cli cmd: " + " ".join(cmd))`.
- Log `"[whisper.cpp] Transcribing with beam_size=5: {audio_path.name}"`.

#### Process and stderr reader

`src/whispercpp_wrapper.py:227-254`:

- `stdout=DEVNULL` (stdout carries transcript text; the JSON file is authoritative; an unread pipe would fill).
- `stderr=PIPE`, `creationflags`.
- Reader thread iterates `for raw in proc.stderr` — **line splitting on `\n` only**, not `\r`. whisper.cpp progress lines end in `\n`, unlike ffmpeg, so this is sufficient.
- `line = raw.decode("utf-8", errors="replace").rstrip()`.
- Append to `stderr_lines`.
- If `_SYSTEM_INFO_RE.search(line)`: for each text in `_describe_devices(stderr_lines)`, log `"[whisper.cpp] {text}"`.
- `pct = _parse_progress_line(line)`; if not `None` and `progress_callback` is set, call `progress_callback(pct / 100.0)`.
  - Progress is emitted for every line matching, including duplicates and 0/100.
  - No explicit monotonicity enforcement.
- `_describe_devices` is invoked on the same line that matches system info; it sees all prior lines plus the current one.
- Reader thread is daemon.

#### Cancellation

`src/whispercpp_wrapper.py:256-266`:

- While `proc.poll() is None`, check every 0.1 s.
- If cancel:
  - `logger.info("Cancel requested — terminating whisper-cli")`.
  - `proc.terminate()`; `proc.wait(timeout=5)`; on exception `proc.kill()` (no wait).
  - `reader_thread.join(timeout=2)`.
  - Raise `CancelledError("Transcription cancelled by user")`.
  - The `finally` deletes `out_json` (partial or absent).
- No separate temp WAV cleanup here; the audio is the caller’s temp file.

#### Completion

`src/whispercpp_wrapper.py:268-284`:

- `reader_thread.join(timeout=5)`.
- If return code nonzero:
  - `stderr_text = "\n".join(stderr_lines)`.
  - `snippet = stderr_text[-800:].strip()`.
  - `logger.error("whisper-cli failed (rc={rc}):\n{snippet}")`.
  - Raise `WhisperCppError(f"whisper-cli exited with code {rc}.\n\n{snippet}")`.
- If `out_json` does not exist:
  - Raise `WhisperCppError(f"whisper-cli completed but did not produce {out_json.name}")`.
- Parse output; log `"[whisper.cpp] Decoded {n} segment(s)"`.
- If `progress_callback`: call `progress_callback(1.0)` (even if the last line already reported 100%).
- Return segments.

#### Exception handling / cleanup

`src/whispercpp_wrapper.py:286-291`:

- `CancelledError` and `WhisperCppError` re-raise unchanged.
- Any other `Exception` → `WhisperCppError(f"Unexpected error during whisper.cpp transcription: {exc}")` with `from exc`.
- `finally`: `out_json.unlink(missing_ok=True)`. It does **not** delete `out_stem` or any other files whisper-cli may have created. See §7.7.

### 10.11 Device logging and GPU selection semantics

- Never assume Vulkan was used because the binary was built with Vulkan. The only ground truth is `whisper_backend_init_gpu: using ... backend` (`src/whispercpp_wrapper.py:93-97`).
- A CPU-only build may still print `device N: Vulkan...` listing lines; those are backend listings, not proof of use.
- The real GPU name comes only from `ggml_vulkan: N = <name> | <details>`.
- `_best_gpu()` picks a device for `-dev` before whisper-cli starts, based on the probe. This can differ from the device whisper.cpp would choose by default (it would use device 0).
- The probe is cached; changing GPUs/drivers requires restarting the app.

### 10.12 Tests

`tests/test_whispercpp_wrapper.py:40-203` enforces:

- JSON parsing/skipping blank text and ms→s conversion;
- `_describe_devices` Vulkan, CPU-only, unnamed GPU;
- `_pick_gpu` dedicated/integrated/tie;
- `_parse_progress_line`;
- `is_available` false when missing, `validate_whispercpp` message contents;
- pre-cancelled `transcribe_file` raises `CancelledError`.

The test imports `whispercpp_wrapper.CancelledError`, which is the object imported from `common` at module scope (`src/whispercpp_wrapper.py:27`); Rust should expose the same cancellation type across modules.

---

## 11. Transcription engine (`src/transcriber.py`)

### 11.1 Exceptions

| Type | Base | Meaning |
|---|---|---|
| `ModelNotFoundError` | `RuntimeError` | required sherpa model files missing or recognizer construction failed |
| `TranscriptionError` | `RuntimeError` | audio load failure, output write failure, whisper.cpp failure, empty input for speaker addition |

`CancelledError` is defined in `common` and propagates unchanged.

### 11.2 `_format_hms(seconds) -> str`

`src/transcriber.py:40-44`:

```python
h = int(seconds // 3600)
m = int((seconds % 3600) // 60)
s = int(seconds % 60)
return f"{h:02d}:{m:02d}:{s:02d}"
```

- Truncates fractional seconds; no rounding.
- Hours are zero-padded to at least 2 digits; 100 hours → `"100:00:00"`.
- `59.999` → `"00:00:59"`.
- Used for block timestamps only.

### 11.3 `_BLOCK_GAP_SEC`

`_BLOCK_GAP_SEC: float = 2.0` (`src/transcriber.py:47`). A gap `start - prev_end >= 2.0` starts a new block in both plain timed and diarized timed output.

### 11.4 `_build_plain_transcript(raw_segments, add_timestamps) -> str`

`src/transcriber.py:50-76`.

Input: list of `(start_sec, end_sec, text)`.

- Empty list → `""` (regardless of `add_timestamps`).
- `add_timestamps=False`:
  - Return `" ".join(text for _, _, text in raw_segments)`.
  - All segments are joined with a single space; no filtering of empty text; no trailing space.
- `add_timestamps=True`:
  1. `blocks = []`, `block_start = raw_segments[0][0]`, `block_texts = []`, `prev_end = 0.0`.
  2. For each `(start, end, text)`:
     - If `block_texts` is nonempty **and** `(start - prev_end) >= 2.0`:
       - append `(block_start, block_texts)` to `blocks`;
       - `block_start = start`;
       - `block_texts = []`.
     - Append `text` to `block_texts`.
     - `prev_end = end`.
  3. Append final `(block_start, block_texts)` if nonempty.
  4. Return `"\n\n".join(f"[{_format_hms(ts)}]\n" + " ".join(texts) for ts, texts in blocks)`.
- Important: the first segment never triggers a split because `block_texts` is empty.
- A gap is computed from the **previous segment’s end**, not the block start.
- A negative gap (overlap) does not split.
- Exactly `2.0` splits (`>=`).
- Segments within a block are joined with a single space, including across speaker turns (plain mode has no speaker info).
- Block timestamps are the first segment’s start in that block, not the first word’s actual onset beyond truncation.

Tests (`tests/test_transcript_builders.py:35-55`):

- empty → `""` for both flags;
- no timestamps → `"Hello world."`;
- timestamps single block → `"[00:00:00]\nHello world."`;
- gap → `"[00:00:00]\nHello.\n\n[00:00:10]\nGoodbye."` (the test uses 0.0 and 10.0; both format to `00:00:00` and `00:00:10`).

### 11.5 `_build_diarized_transcript(assigned, speaker_map, add_timestamps) -> str`

`src/transcriber.py:79-128`.

Input: `assigned` = list of `(speaker_raw, text, start_sec, end_sec)`; `speaker_map` = raw label → display name.

- Empty `assigned` → `""`.
- `add_timestamps=False`:
  1. `block_label = speaker_map.get(assigned[0][0], assigned[0][0])`.
  2. `block_texts = [assigned[0][1]]`.
  3. For each subsequent `(speaker_raw, text, _s, _e)`:
     - `label = speaker_map.get(speaker_raw, speaker_raw)`.
     - If `label == block_label`: append text.
     - Else: append `f"[{block_label}]\n" + " ".join(block_texts)` to `parts`; set `block_label = label`; `block_texts = [text]`.
  4. Append the final block.
  5. Return `"\n\n".join(parts)`.
  - **No gap splitting in this mode**; only speaker changes split blocks. A same-speaker segment after a 10-minute gap stays in the same block.
  - Unmapped raw labels fall back to the raw label (e.g. `[SPEAKER_05]`).
- `add_timestamps=True`:
  1. `block_label = mapped first`; `block_start = assigned[0][2]`; `block_texts = [first text]`; `prev_end = assigned[0][3]`.
  2. For each subsequent `(speaker_raw, text, start, end)`:
     - `label = mapped`.
     - `new_block = (label != block_label) or ((start - prev_end) >= 2.0)`.
     - If new block:
       - append `f"[{_format_hms(block_start)}] [{block_label}]\n" + " ".join(block_texts)`;
       - `block_label = label`;
       - `block_start = start`;
       - `block_texts = [text]`.
     - Else append text.
     - **Always** `prev_end = end` (even when a new block started).
  3. Append final block with its start/label.
  4. Return `"\n\n".join(parts)`.
  - Splits on speaker change **or** gap ≥ 2.0.
  - The `prev_end` update outside the `if` means a speaker change followed immediately by another segment uses the immediately previous segment’s end for the gap test, as expected.
  - Block timestamp is the first segment’s start in that block.

Tests (`tests/test_transcript_builders.py:61-99`):

- empty → `""`;
- no timestamps groups consecutive same speaker, splits on change:
  ```
  [Speaker 1]
  Hello, how are you?

  [Speaker 2]
  Fine, thanks.
  ```
- timestamps splits on speaker change and gap:
  ```
  [00:00:00] [Speaker 1]
  Hello. Still talking.

  [00:00:02] [Speaker 2]
  Reply.

  [00:00:10] [Speaker 2]
  More.
  ```
- unmapped speaker → `[SPEAKER_05]\nHi.`

### 11.6 Speaker labels and mapping

- Diarization turns use raw labels `SPEAKER_00`, `SPEAKER_01`, ... (two-digit minimum; see §12.7).
- `speaker_transcript` builds `speaker_map` in order of **first appearance in `turns`**:
  - First raw speaker → `"Speaker 1"`, next new raw speaker → `"Speaker 2"`, etc.
  - The numeric suffix of the raw label does not determine the display number. If `SPEAKER_02` appears first, it becomes `"Speaker 1"`.
- The GUI lets the user replace values with nonempty stripped names; empty fields keep defaults (`src/gui.py:433-441`).
- No sanitization/escaping of names: a name containing `]`, newline, etc. is inserted verbatim into `[label]`. If transcript-format injection matters, validate at the GUI; the backend does not.
- `_build_diarized_transcript` accepts any map values; unmapped labels use the raw label.
- `assign_speakers` can produce raw `SPEAKER_00` even when no turn has that label (no-overlap default), which then appears unmapped. See §12.6.

### 11.7 `_extract_speaker_clip(audio_path, turns, speaker, out_path, max_duration=5.0) -> bool`

`src/transcriber.py:131-164`:

1. Find the turn for `speaker` with the longest duration:
   - `best_start = best_end = None`, `best_dur = 0.0`.
   - For each `(t_start, t_end, t_speaker)`:
     - If `t_speaker == speaker`, `dur = t_end - t_start`; if `dur > best_dur` (strict), update.
   - If no turn with positive duration → `False` (zero-length turns do not qualify).
2. `clip_end = min(best_end, best_start + 5.0)`.
3. Open `audio_path` with `wave`:
   - `n_ch`, `sampwidth`, `sr`.
   - `setpos(int(best_start * sr))`.
   - `readframes(int((clip_end - best_start) * sr))`.
4. Open `out_path` `"wb"` with the same channels/width/rate; write raw frames.
5. Return `True`.
6. Any exception → `logger.warning(f"Could not extract speaker clip for {speaker}: {exc}")` → `False`.

Notes:

- The clip is at most 5.0 seconds from the start of the longest turn.
- `int()` truncates start and frame count; a fractional start rounds down.
- No padding or fade; starts exactly at the turn start.
- Does not create the output directory (`TEMP_DIR` exists).
- The output WAV is a valid WAV with the original format; for production it is mono 16 kHz s16le.
- If `best_start` is near the end and `clip_end` equals `best_start`, writes a zero-frame WAV and returns `True`.
- Used only for the naming UI; failure is nonfatal.

### 11.8 `_write_transcript(output_path, transcript_text, log_callback) -> None`

`src/transcriber.py:170-188`:

1. `output_path.parent.mkdir(parents=True, exist_ok=True)`.
2. `log_callback(f"[Output] Writing transcript to: {output_path}")`.
3. `logger.info(f"Writing transcript: {output_path}")`.
4. `open(output_path, "w", encoding="utf-8", newline="\n")`.
5. Write `transcript_text`, then `"\n"`.
6. `PermissionError` → raise `TranscriptionError(f"Cannot write to output file.\nThe file may be open in another program: {output_path}\n\n{exc}")`.
7. `OSError` → raise `TranscriptionError(f"Failed to write transcript: {output_path}\n\n{exc}")`.
8. Other exceptions propagate.

Details:

- The trailing newline is always added, even for empty text.
- Existing file is truncated.
- Parent directories are created; if creation fails, the `OSError` from `mkdir` is **not** inside the try block? In Python, line 173 is before the `try` at 176; a `mkdir` `OSError` propagates raw, not wrapped. **DEFECT/edge**; Rust should wrap all write-path errors in `TranscriptionError` for consistency unless exact taxonomy is required.
- `newline="\n"` means no platform translation; LF on Windows too.
- The log callback is called before the write; if writing fails, the log still says “Writing transcript to”.
- No atomic write; a crash can leave a partial file.

### 11.9 `speaker_transcript(...) -> str`

`src/transcriber.py:191-242`.

Signature:

```python
speaker_transcript(
    audio_path: Path,
    raw_segments: list[tuple[float, float, str]],
    add_timestamps: bool,
    cancel_event: threading.Event,
    status_callback: Callable[[str], None],
    progress_callback: Callable[[float], None],
    log_callback: Callable[[str], None],
    speaker_name_callback: Callable[[dict, dict, dict], None] | None = None,
) -> str
```

Sequence:

1. Lazy import `DiarizationEngine`, `save_measured_speed` (`208`).
2. `status_callback("Identifying Speakers")`.
3. `started = time.monotonic()`.
4. `engine = DiarizationEngine()`.
5. `turns = engine.diarize(audio_path, log_callback, cancel_event, progress_callback)`.
6. `save_measured_speed(audio_path, raw_segments, time.monotonic() - started)` — learns PC speed even if naming is cancelled later.
7. `assigned = engine.assign_speakers(raw_segments, turns)`.
8. Build `speaker_map` and `talk`:
   - Iterate `turns` in order:
     - If `spk not in speaker_map`: `speaker_map[spk] = f"Speaker {len(speaker_map) + 1}"`.
     - `talk[spk] = talk.get(spk, 0.0) + max(0.0, t_end - t_start)`.
9. If `speaker_name_callback is not None and speaker_map`:
   - `status_callback("Naming Speakers")`.
   - `total_talk = sum(talk.values()) or 1.0`.
   - `shares = {spk: talk.get(spk, 0.0) / total_talk for spk in speaker_map}`.
   - `clips = {}`.
   - For each `raw_spk in speaker_map` (insertion order):
     - `clip_path = TEMP_DIR / f"spk_clip_{raw_spk}.wav"`.
     - If `_extract_speaker_clip(...)` returns true: `clips[raw_spk] = clip_path`.
   - `try: speaker_name_callback(speaker_map, clips, shares)`
     `finally:` for each clip path, `unlink(missing_ok=True)`, catch `OSError` and ignore.
10. Return `_build_diarized_transcript(assigned, speaker_map, add_timestamps)`.

Callback contract:

- Called synchronously on the worker thread; it must block until the user finishes.
- `speaker_map` is mutable; the callback should set values for names. Empty/default values remain `"Speaker N"`.
- `clips` maps raw label → path for speakers with a successfully extracted positive-length turn; missing entries mean “No sample”.
- `shares` values are in `[0,1]` and sum to 1.0 (denominator `sum(talk) or 1.0`); if all talk zero, shares are 0.0.
- The callback may raise `CancelledError` (GUI does when the user cancels); it propagates, and the `finally` deletes clips.
- The callback runs before transcript formatting; if it returns, formatting uses the possibly-mutated map.
- If `speaker_map` is empty, the callback is not called.
- Diarization progress is whatever `DiarizationEngine.diarize` emits; there is no separate progress during naming. The GUI shows “Waiting for speaker names” with a paused bar.

### 11.10 `add_speakers_to_file(...) -> None`

`src/transcriber.py:245-269`.

Signature mirrors `speaker_transcript` plus `output_path`:

```python
add_speakers_to_file(
    audio_path, raw_segments, output_path, cancel_event,
    status_callback, progress_callback, log_callback,
    speaker_name_callback=None,
) -> None
```

1. If `not raw_segments`: raise `TranscriptionError("The transcript has no text to add speakers to.")`.
2. `text = speaker_transcript(..., add_timestamps=True, ...)` — **always timestamps**, regardless of the original transcript’s formatting.
3. `status_callback("Writing Transcript")`.
4. `_write_transcript(output_path, text, log_callback)`.
5. `progress_callback(1.0)`.
6. `log_callback(f"[Done] Saved: {output_path}")`.
7. No return value.

Differences from `TranscribeEngine.transcribe(diarize=True)`:

- A diarization failure is **raised**, not hidden with a plain fallback (`src/transcriber.py:257-259`).
- The output is always timestamped.
- The source transcript is not modified; a new file is written (GUI picks the name).

### 11.11 `validate_model_directory() -> list[str]`

`src/transcriber.py:275-286`:

- If `VAD_MODEL_PATH` is not a file → `"Missing: {path}"`.
- For each of `WHISPER_ENCODER`, `WHISPER_DECODER`, `WHISPER_TOKENS` in that order, if not a file → `"Missing: {path}"`.
- Returns the list; caller joins with newlines and/or formats.
- Does **not** check diarization models or whisper.cpp files.
- `main._check_dependencies` calls this and aborts startup if nonempty (`src/main.py:79-84`, `115-127`), even if whisper.cpp is available. This is deliberate: live mode/VAD/diarization need sherpa.

### 11.12 `_build_recognizer_config()`

`src/transcriber.py:289-304`:

```python
sherpa_onnx.OfflineModelConfig(
    whisper=sherpa_onnx.OfflineWhisperModelConfig(
        encoder=str(WHISPER_ENCODER),
        decoder=str(WHISPER_DECODER),
        language="en",
        task="transcribe",
    ),
    tokens=str(WHISPER_TOKENS),
    provider="cpu",
    num_threads=NUM_THREADS,
)
return sherpa_onnx.OfflineRecognizerConfig(model_config=model_cfg)
```

- Fixed language `"en"`, task `"transcribe"`.
- Provider `"cpu"`; no GPU provider.
- No `decoding_method`, `hotwords`, `tail_paddings`, `debug`, etc.; defaults.
- The model is Whisper ONNX large-v3-turbo INT8 (`turbo-encoder.int8.onnx`, `turbo-decoder.int8.onnx`, `turbo-tokens.txt`).

### 11.13 `TranscriptionEngine`

`src/transcriber.py:310-473`.

#### Construction

```python
def __init__(self):
    self._recognizer = None
    self._lock = threading.Lock()
```

Lazy; no model loaded at startup. Safe to construct before models exist.

#### `_ensure_model_loaded(status_callback)`

`src/transcriber.py:317-338`:

1. Acquire `self._lock`.
2. If `self._recognizer is not None`: return.
3. `missing = validate_model_directory()`; if nonempty → `ModelNotFoundError("Model validation failed:\n" + "\n".join(missing))`.
4. `status_callback("Loading Model")`; `logger.info("Loading model, provider=cpu")`.
5. `from sherpa_onnx.lib._sherpa_onnx import OfflineRecognizer`.
6. `cfg = _build_recognizer_config()`; `self._recognizer = OfflineRecognizer(cfg)`; `logger.info("Model loaded: provider=cpu")`.
7. Catch `RuntimeError` → `ModelNotFoundError(f"Could not load model: {exc}")`; other exceptions propagate (ImportError, OSError, etc.).
8. The lock covers the entire load, so concurrent callers block and then see the loaded recognizer.

#### `get_recognizer(status_callback)`

`src/transcriber.py:340-343`: ensure loaded, return `self._recognizer`.

Used by live recording before `MicRecorder.start` (`src/gui.py:1671-1672`), so model loading can block recording start (status “Loading Model”).

#### `transcribe(...)`

Signature `src/transcriber.py:345-356`:

```python
def transcribe(
    self,
    audio_path: Path,
    output_path: Path,
    add_timestamps: bool,
    cancel_event: threading.Event,
    status_callback: Callable[[str], None],
    progress_callback: Callable[[float], None],
    log_callback: Callable[[str], None],
    diarize: bool = False,
    speaker_name_callback: Callable[[dict, dict, dict], None] | None = None,
) -> list[tuple[float, float, str]]:
```

Sequence in detail:

1. If cancel set → `CancelledError("Cancelled before transcription started")`.
2. `status_callback("Transcribing")`.
3. `logger.info(f"Transcribing: {audio_path.name} → {output_path.name}")` (Unicode arrow `→`).
4. `log_callback(f"[Transcribe] Starting: {audio_path.name}")`.
5. `duration = wav_duration(audio_path)`; any exception → `TranscriptionError(f"Could not load audio: {exc}")`.
6. `log_callback(f"[Transcribe] Audio duration: {fmt_duration(duration)}")`.
7. `infer_start = time.monotonic()`.
8. `raw_segments: list[tuple[float, float, str]]`.
9. **Engine selection**: `if whispercpp_wrapper.is_available():`
   - Log:
     ```
     [Transcribe] engine: whisper.cpp (beam_size=5), timestamps: {on|off}, speakers: {on|off}
     ```
   - `try: raw_segments = whispercpp_wrapper.transcribe_file(audio_path, cancel_event, log_callback, progress_callback)`.
   - Catch `WhisperCppError` → `TranscriptionError(str(exc))` with cause. (`CancelledError` propagates.)
   - For each segment, `log_callback(text)` (raw text, no prefix) (`394-395`). This is the UI details log only; the code does **not** call `logger.info(text)` for segments, so file logs never contain transcript text (privacy requirement).
10. **Else sherpa**:
    - `self._ensure_model_loaded(status_callback)`.
    - `samples, sr = load_wav_float32(audio_path)`; any exception → `TranscriptionError(f"Could not load audio: {exc}")`.
    - Log:
      ```
      [Transcribe] engine: sherpa-onnx (CPU), timestamps: {on|off}, speakers: {on|off}
      ```
    - Log `"[Transcribe] Detecting speech segments (VAD)…"`.
    - `vad_detector = vad.build_vad()` (60 s default buffer).
    - `bounds = vad.file_segments(vad_detector, samples, cancel_event)`.
    - `padded_bounds = vad.pad_and_clamp(bounds, len(samples))`.
    - `num_chunks = len(padded_bounds)`.
    - Log `"[Transcribe] Processing {num_chunks} speech segment(s)…"`.
    - `raw_segments = []`.
    - For `i, (c_start, c_end)`:
      - If cancel set → `CancelledError("Transcription cancelled by user")` (checked before each segment; not during `decode_stream`).
      - `seg_start = c_start / sr`, `seg_end = c_end / sr` (floats).
      - `stream = self._recognizer.create_stream()`.
      - `stream.accept_waveform(sr, samples[c_start:c_end])`.
      - `self._recognizer.decode_stream(stream)`.
      - `text = stream.result.text.strip()`.
      - If nonempty: append `(seg_start, seg_end, text)` and `log_callback(text)` (UI-only, not written to the file log).
      - `progress_callback((i + 1) / num_chunks)` — always called, even if text was empty. If `num_chunks == 0`, not called here.
      - `elapsed = time.monotonic() - infer_start`.
      - `speed = seg_end / elapsed if elapsed > 0 else 0`.
      - Log:
        ```
        [Progress] segment {i+1}/{num_chunks} — {fmt_duration(seg_end)} / {fmt_duration(duration)} — {speed:.1f}x realtime
        ```
    - `del samples` (frees ~700 MB for a 3-hour file before diarization loads its own copy).
11. After either engine:
    - `segment_count = len(raw_segments)`.
    - `word_count = sum(len(text.split()) for _, _, text in raw_segments)` — Python `str.split()` splits on arbitrary whitespace and ignores leading/trailing; Rust `split_whitespace().count()` equivalent.
    - `total_elapsed = time.monotonic() - infer_start`.
    - Log:
      ```
      [Transcribe] Complete — {segment_count} segments, {word_count} words in {fmt_duration(total_elapsed)}
      ```
12. **Diarization**:
    - If `diarize and raw_segments`:
      - `try: transcript_text = speaker_transcript(...)` with the same callbacks and `add_timestamps`.
      - `except CancelledError: raise`.
      - `except Exception as exc:` log warning `"Diarization failed, falling back to plain: {exc}"`; `log_callback(f"[Diarize] Warning: {exc}. Writing the transcript without speaker names.")`; `transcript_text = _build_plain_transcript(raw_segments, add_timestamps)`.
      - Note: diarization progress will have overwritten the transcription progress; on fallback, the final write sets 1.0.
    - Else: `transcript_text = _build_plain_transcript(raw_segments, add_timestamps)`.
13. **Write**:
    - `status_callback("Writing Transcript")`.
    - `_write_transcript(output_path, transcript_text, log_callback)`.
    - `size_kb = output_path.stat().st_size // 1024`.
    - `progress_callback(1.0)`.
    - `log_callback(f"[Done] Saved: {output_path}  ({size_kb} KB)")` (two spaces before `(`).
    - Return `raw_segments`.

Return value: timed lines (including empty-text filtering at the engine level), in segment order. Used by the GUI to allow “Find speakers” without retranscribing (`src/gui.py:1532-1534`).

Error propagation table:

| Condition | Result |
|---|---|
| Pre-cancel | `CancelledError("Cancelled before transcription started")` |
| WAV duration/load failure | `TranscriptionError("Could not load audio: ...")` |
| Missing sherpa models | `ModelNotFoundError("Model validation failed:\n...")` |
| Recognizer construction `RuntimeError` | `ModelNotFoundError("Could not load model: ...")` |
| whisper.cpp nonzero/missing JSON | `WhisperCppError` → `TranscriptionError(str(exc))` |
| VAD cancel | `CancelledError("VAD segmentation cancelled by user")` |
| Per-segment cancel | `CancelledError("Transcription cancelled by user")` |
| Diarization cancel | `CancelledError` (propagates, no plain fallback) |
| Other diarization failure | logged, plain transcript written, function succeeds |
| Output write failure | `TranscriptionError` (unless mkdir fails raw) |

### 11.14 Progress and status contract

| Stage | Status callback | Progress callback |
|---|---|---|
| Load model (sherpa only) | `"Loading Model"` | none |
| Start transcribe | `"Transcribing"` | — |
| whisper.cpp | `"Transcribing"` | `pct/100` per progress line; final `1.0` |
| sherpa VAD | `"Transcribing"` | none during VAD |
| sherpa chunks | `"Transcribing"` | `(i+1)/num_chunks` after each chunk |
| After chunks | `"Transcribing"` | no reset |
| Diarization | `"Identifying Speakers"` | `pct/100` from child (0..1) |
| Naming | `"Naming Speakers"` | no updates; last value likely 1.0 |
| Write | `"Writing Transcript"` | `1.0` after write |
| Completion | no `"Done"` status from engine | final progress `1.0` |

The GUI sets final `"Done"`/`"Cancelled"`/`"Failed"` after the worker returns (`src/gui.py:1568-1581`). The engine itself only uses the statuses above.

### 11.15 Rust API shape

Suggested:

```rust
pub struct TranscriptionEngine { recognizer: Option<...>, lock: Mutex<()> }

pub struct TranscribeRequest<'a> {
    pub audio_path: &'a Path,
    pub output_path: &'a Path,
    pub add_timestamps: bool,
    pub cancel: &'a CancelToken,
    pub status: &'a mut dyn FnMut(&str),
    pub progress: &'a mut dyn FnMut(f64),
    pub log: &'a mut dyn FnMut(&str),
    pub diarize: bool,
    pub speaker_naming: Option<&'a mut dyn SpeakerNaming>,
}
```

- Use `f64` for timestamps and progress; `f32` for audio.
- The speaker-naming callback must be able to mutate a `HashMap<String,String>` and block. Prefer a trait:

```rust
pub trait SpeakerNaming {
    fn name_speakers(
        &mut self,
        map: &mut HashMap<String, String>,
        clips: &HashMap<String, PathBuf>,
        shares: &HashMap<String, f64>,
    ) -> Result<(), CancelledError>;
}
```

- The GUI adapter sends a message to the UI thread and blocks on a channel/condvar; it polls the cancel token and returns `CancelledError` if set.
- Callbacks are `FnMut` and invoked on the worker thread; the UI adapter must marshal to the event loop. In Rust, `Send` bounds are required if moving the callback to a worker thread; alternatively run the pipeline on a dedicated thread and use channels to the UI.

---

## 12. Diarizer (`src/diarizer.py`)

### 12.1 Constants and time model

`src/diarizer.py:31-40`:

| Constant | Value | Meaning |
|---|---|---|
| `_POLL_SECONDS` | `0.2` | Poll interval for child messages/cancel |
| `_SEC_PER_AUDIO_SEC` | `0.045` | Speech-finding cost per second of audio |
| `_SEC_PER_SPEECH_SEC` | `0.165` | Embedding/clustering cost per second of speech |
| `_START_SECONDS` | `3.0` | Fixed startup cost |

`_model_seconds(audio_sec, speech_sec)` (`src/diarizer.py:54-56`):

```python
speech = audio_sec if speech_sec is None else min(speech_sec, audio_sec)
return 3.0 + 0.045 * audio_sec + 0.165 * speech
```

### 12.2 `speech_seconds(segments) -> float`

`src/diarizer.py:43-51`:

```python
total, reach = 0.0, 0.0
for start, end, _text in sorted(segments):
    start = max(start, reach)
    if end > start:
        total += end - start
        reach = end
return total
```

- Sorts tuples lexicographically by `(start, end, text)`; for normal segments this is start order.
- Computes the **union length** of the timed segments: overlaps are counted once.
- If a segment is fully inside the previous reach, it contributes 0 and does not move `reach`.
- If segments have gaps, gaps are excluded.
- Negative or inverted segments (`end <= start`) contribute 0 and do not move `reach`.
- `reach` starts at 0, so a segment starting at negative time contributes only its positive part.
- The result is used for estimates and speed learning, not for transcript text.

### 12.3 `estimate_seconds(audio_sec, speech_sec=None) -> float`

`src/diarizer.py:59-67`:

- `_model_seconds(audio_sec, speech_sec) * (load_speaker_speed() or 1.0)`.
- `load_speaker_speed()` is read on every call; a missing/invalid setting → 1.0.
- `speech_sec=None` means all audio is treated as speech.
- Clamps `speech_sec` to `audio_sec` (if larger).
- Used by:
  - `diarize` log estimate with `speech_sec=None` (`src/diarizer.py:154`);
  - GUI “Find speakers” card with `speech_seconds(segments)` clamped to `audio_sec` (`src/gui.py:1880-1885`).

### 12.4 `save_measured_speed(audio_path, segments, took_sec) -> None`

`src/diarizer.py:70-82`:

1. Try `expected = _model_seconds(wav_duration(audio_path), speech_seconds(segments))`; any exception → return silently.
2. `factor = min(5.0, max(0.2, took_sec / expected))`.
   - Clamp to `[0.2, 5.0]`.
   - `expected` could be 0? `_START_SECONDS=3` ensures ≥3 for valid audio, so no division by zero unless weird negative inputs.
3. `previous = load_speaker_speed()`.
4. `save_speaker_speed(factor if previous is None else (previous + factor) / 2)`.
   - Arithmetic mean of previous and current, not weighted.
   - `save_speaker_speed` rounds to 3 decimals.
5. Called from `speaker_transcript` after diarization succeeds and before naming (`src/transcriber.py:214`); thus a naming cancel still updates the estimate.

### 12.5 `fmt_estimate(seconds) -> str`

`src/diarizer.py:85-99`:

```python
if seconds < 50:
    return "less than a minute"
minutes = round(seconds / 60)
if minutes <= 1:
    return "about 1 minute"
if minutes < 60:
    return f"about {minutes} minutes"
hours, rest = divmod(minutes, 60)
rest = 5 * round(rest / 5)
if rest == 60:
    hours, rest = hours + 1, 0
hours_text = "1 hour" if hours == 1 else f"{hours} hours"
return f"about {hours_text}" + (f" {rest} minutes" if rest else "")
```

Exact quirks:

- `seconds < 50` includes all negatives.
- `round()` is **banker’s rounding (round-half-even)** on a float:
  - 90 s = 1.5 min → `round(1.5) = 2` → “about 2 minutes”.
  - 150 s = 2.5 min → `round(2.5) = 2` → “about 2 minutes”.
  - 210 s = 3.5 min → `round(3.5) = 4` → “about 4 minutes”.
  - 270 s = 4.5 min → `round(4.5) = 4` → “about 4 minutes”.
- `minutes <= 1` includes 0 and 1; 50–89 s generally → “about 1 minute” (except 90 → 2).
- `minutes < 60`: plural always `"minutes"`, even 2.
- For ≥60: `divmod`; `rest` is the remaining minutes (0..59). `rest = 5 * round(rest / 5)` rounds to nearest 5 with banker’s rounding:
  - rest 2.5 → round(0.5)=0 → 0;
  - rest 7.5 → round(1.5)=2 → 10;
  - rest 57.5 → round(11.5)=12 → 60 → carry to hour.
- Final: `"about 1 hour"`, `"about 2 hours"`, `"about 1 hour 30 minutes"`, `"about 2 hours 5 minutes"`.
- `rest == 0` omits minutes.
- Uses `"1 hour"` singular, `"{hours} hours"` plural (even 0? 0 only if seconds <60, but <50 returns earlier; 50-59 -> minutes 1 -> earlier; so no).
- No “less than a minute” vs “under a minute” variation: exact `"less than a minute"`.
- Rust: use `round_ties_even()` (stable since Rust 1.77) for both rounds, then integer arithmetic. Do not use `f64::round()` (half away from zero) or you will change 150 s / 270 s outputs.

Examples computed from the reference:

| seconds | output |
|---:|---|
| 0 | less than a minute |
| 30 | less than a minute |
| 49 | less than a minute |
| 50 | about 1 minute |
| 89 | about 1 minute |
| 90 | about 2 minutes |
| 149 | about 2 minutes |
| 150 | about 2 minutes |
| 151 | about 3 minutes |
| 3599 | about 1 hour |
| 3600 | about 1 hour |
| 5400 | about 1 hour 30 minutes |

### 12.6 `DiarizationEngine.assign_speakers(segments, turns) -> list[(speaker, text, start, end)]`

`src/diarizer.py:186-217` (staticmethod):

```python
result = []
for seg_start, seg_end, text in segments:
    best_speaker = "SPEAKER_00"
    best_overlap = 0.0
    for turn_start, turn_end, speaker in turns:
        overlap = min(seg_end, turn_end) - max(seg_start, turn_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = speaker
    result.append((best_speaker, text, seg_start, seg_end))
return result
```

- Picks the turn with the greatest positive time overlap.
- Ties: strict `>` keeps the **first** turn with the maximum overlap (turn list order).
- No positive overlap (including just touching at a boundary, overlap 0) → `"SPEAKER_00"`, even if no such speaker exists.
- Invalid/negative turn durations produce negative overlap, never selected.
- Output preserves segment order and text/times exactly.
- Tests (`tests/test_transcript_builders.py:105-120`):
  - segment 0–2 with turns 0–1.5 (1.5 s) and 1.5–4 (0.5 s) → `SPEAKER_00`;
  - segment 5–7 with turn 4.5–8 → `SPEAKER_02`;
  - no-overlap segment → `SPEAKER_00`.

### 12.7 `DiarizationEngine.is_available()`

`src/diarizer.py:113-128`:

- Lazy imports `DIARIZATION_SEGMENTATION_MODEL`, `DIARIZATION_EMBEDDING_MODEL` from config.
- Returns true iff both are `is_file()`.
- If false, logs debug:
  ```
  Diarization models not found at {segmentation} / {embedding} (checkbox will be disabled)
  ```
- Does not import `sherpa_onnx`; safe at startup.
- No checks for model validity, matching versions, or readability beyond `is_file`.

### 12.8 `DiarizationEngine.diarize(...) -> list[(start, end, speaker)]`

Signature `src/diarizer.py:130-136`:

```python
diarize(
    audio_path: Path,
    log_callback: Callable[[str], None],
    cancel_event: threading.Event | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> list[tuple[float, float, str]]
```

Sequence:

1. If `not self.is_available()` → `DiarizationError("The speaker models are missing.")`.
2. `audio_sec = wav_duration(audio_path)`; exception → `DiarizationError(f"Could not load audio for diarization: {exc}")`.
3. `est_str = fmt_estimate(estimate_seconds(audio_sec))` (no speech_sec; treats all audio as speech).
4. Log:
   ```
   [Diarize] Estimated wait: {est_str} for {fmt_duration(audio_sec)} of audio (rough estimate). Please wait…
   ```
   (Unicode ellipsis `…`.)
5. `logger.info(f"Running diarization on: {audio_path.name} (estimate {est_str})")`.
6. Log `"[Diarize] Finding the parts with speech…"`.
7. Create a spawn context:
   - `ctx = multiprocessing.get_context("spawn")`.
   - `recv_conn, send_conn = ctx.Pipe(duplex=False)` — first is receive-only, second send-only.
   - `proc = ctx.Process(target=_run_pipeline, args=(str(audio_path), config.NUM_THREADS, send_conn), daemon=True)`.
   - `proc.start()`.
   - `send_conn.close()` in the parent; the child holds its own copy. EOF occurs when the child exits.
8. `try: turns = _wait_for_turns(proc, recv_conn, cancel_event, log_callback, progress_callback)`.
9. `finally: _stop_process(proc); recv_conn.close()`.
10. `speaker_set = {t[2] for t in turns}`.
11. Log `"[Diarize] Found {len(speaker_set)} speaker(s) across {len(turns)} turn(s)"`.
12. `logger.info("Diarization complete: {n} speakers, {m} turns")`.
13. Return turns.

Important:

- Cancellation is only observed in `_wait_for_turns` after the child is started. If the event is already set, the child is still spawned, then immediately terminated in the loop/finally.
- The child is daemonic; if the parent exits, it is terminated.
- The parent does not load the audio; only the child does.
- `is_available` is checked before `wav_duration`; missing models win over a bad audio path.
- No progress is reported during the first ~20% (speech finding), per comment (`src/diarizer.py:160`).

### 12.9 Parent/child IPC protocol

`_wait_for_turns` (`src/diarizer.py:225-263`):

- `logged_step = -1`.
- Loop forever:
  1. If `cancel_event` is not `None` and set:
     - `logger.info("Cancel requested — ending the diarization process")`.
     - Raise `CancelledError("Speaker identification cancelled by user")`.
  2. `if not conn.poll(0.2):`
     - If `not proc.is_alive() and not conn.poll()`:
       - Raise `DiarizationError(f"Speaker identification stopped unexpectedly (exit code {proc.exitcode}).")`.
     - Else continue.
  3. `try: kind, value = conn.recv()`; `EOFError` → same `DiarizationError` from `None`.
  4. Dispatch:
     - `"progress"`:
       - If `progress_callback`: `progress_callback(value / 100.0)`.
       - If `value // 10 > logged_step`: `logged_step = value // 10`; `log_callback(f"[Diarize] Progress: {value}%")`.
         - `logged_step` starts -1; 0 → 0 > -1 logs; 9 → 0 not > 0; 10 logs; 19 no; 20 logs; etc.
         - If progress jumps 0→100, logs 0 then 100.
     - `"done"`: return `value` (list of turns).
     - `"error"`: `logger.error("Diarization failed in child process: {value}")`; raise `DiarizationError(f"Speaker diarization failed: {value}")`.
     - Unknown `kind`: ignored (loop continues); no log.
  5. Repeat.

Parent termination `_stop_process` (`src/diarizer.py:266-273`):

```python
if proc.is_alive():
    proc.terminate()
    proc.join(timeout=5)
    if proc.is_alive():
        proc.kill()
proc.join(timeout=5)
```

- On Windows, `terminate()` is `TerminateProcess`; on POSIX, SIGTERM.
- `kill()` is SIGKILL/strong termination.
- The second `join(5)` also reaps a process that was already dead.
- If the child ignores SIGTERM, wait up to 5 s, then SIGKILL, then join up to 5 s.
- `_stop_process` is called even after `"done"`; it may kill a still-exiting child, but the parent already has the data.

Child `_run_pipeline` (`src/diarizer.py:308-334`):

1. `pipeline = _build_pipeline(num_threads)`.
2. `samples, _sr = load_wav_float32(Path(audio_path))`.
3. `last_pct = [-1]`.
4. Callback `_progress(num_done, num_total)`:
   - `pct = int(100 * num_done / num_total) if num_total > 0 else 0`.
   - If `pct != last_pct[0]`: set, `conn.send(("progress", pct))`.
   - Return `0` (sherpa-onnx expects 0 to continue; the parent cancels by killing the process, not by a return value).
5. `result = pipeline.process(samples, callback=_progress)`.
6. `turns = [(seg.start, seg.end, f"SPEAKER_{seg.speaker:02d}") for seg in result.sort_by_start_time()]`.
   - `seg.start`/`seg.end` are seconds (float).
   - `:02d` zero-pads to at least 2 digits; speaker 7 → `SPEAKER_07`; speaker 100 → `SPEAKER_100`.
7. `conn.send(("done", turns))`.
8. `except Exception as exc`: `conn.send(("error", f"{type(exc).__name__}: {exc}"))`.
9. `finally: conn.close()`.

Child pipeline config `_build_pipeline` (`src/diarizer.py:281-305`):

```python
seg_cfg = sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
    pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
        model=str(DIARIZATION_SEGMENTATION_MODEL)
    ),
    num_threads=num_threads,
    provider="cpu",
)
emb_cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
    model=str(DIARIZATION_EMBEDDING_MODEL),
    num_threads=num_threads,
    provider="cpu",
)
clu_cfg = sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=0.5)
diar_cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
    segmentation=seg_cfg, embedding=emb_cfg, clustering=clu_cfg
)
return sherpa_onnx.OfflineSpeakerDiarization(diar_cfg)
```

- `num_clusters=-1`: automatic speaker count.
- `threshold=0.5`: clustering threshold.
- CPU only; no CUDA/DirectML.
- The child loads mono float32 samples via `load_wav_float32`; `_sr` is ignored (sherpa’s diarization pipeline assumes 16 kHz; production WAVs are 16 kHz).
- No VAD model is passed; the segmentation model finds speech.

### 12.10 Progress/cancellation details

- Progress values sent by the child are integer percentages; parent converts to `0.0..1.0`.
- The child suppresses duplicate percentages; progress is nondecreasing in normal sherpa operation (test asserts sorted).
- The test `tests/test_diarizer_process.py:32-39` asserts:
  - turns nonempty,
  - progress values nonempty,
  - values sorted ascending,
  - last value exactly `1.0`.
- Cancellation test (`tests/test_diarizer_process.py:42-61`):
  - 20 copies of the test speech (~8 min) ensures work >1 min.
  - Sets cancel on the first progress callback.
  - Asserts `CancelledError` and elapsed <60 s.
  - Because the first progress arrives after speech finding, cancellation is not instantaneous; the process is terminated by `_stop_process`.
- If the child exits unexpectedly with no message, parent error includes `exit code {proc.exitcode}`.
- If the child sends `("error", "RuntimeError: ...")`, parent error is `"Speaker diarization failed: RuntimeError: ..."`.

### 12.11 Rust implementation

Two viable designs:

1. **Keep a child process** (recommended for parity): spawn the current executable with a hidden `--diarize-worker` argument, pass audio path and thread count, and exchange newline-delimited JSON over stdin/stdout/pipes. The parent polls the pipe with a 200 ms timeout, checks a cancellation token, and sends progress/done/error messages. On cancel, terminate the child (Windows `TerminateProcess`; Unix SIGTERM, wait 5 s, then SIGKILL). This preserves crash isolation and the “kill to cancel” semantics.
2. **Run in-process on a worker thread** with a native cancellation flag if the sherpa-rs binding supports safe cancellation. The reference explicitly says sherpa-onnx ignores stop requests (`src/diarizer.py:1-8`); in-process cancellation may not be possible. If used, document the divergence and ensure a hung native call cannot freeze the UI.

For a child process:

- Serialize samples? No: pass the WAV path; the child loads it, matching the reference.
- `NUM_THREADS` passed.
- JSON messages:
  ```json
  {"kind":"progress","value":42}
  {"kind":"done","turns":[{"start":0.0,"end":1.2,"speaker":"SPEAKER_00"}]}
  {"kind":"error","message":"RuntimeError: ..."}
  ```
- Child stderr should be captured/logged or inherited? The reference child uses multiprocessing, not stderr; native sherpa logs may go to stderr and are not captured by the parent. Rust can inherit or redirect to the log; document.
- Ensure the child does not open a GUI window: detect the worker argument before initializing the UI, like `multiprocessing.freeze_support()` (`src/main.py:139-144`).
- Do not use TCP for IPC (offline guard).

### 12.12 Diarization error strings

Exact messages:

- `"The speaker models are missing."`
- `f"Could not load audio for diarization: {exc}"`
- `f"Speaker identification stopped unexpectedly (exit code {proc.exitcode})."`
- `f"Speaker diarization failed: {value}"`
- `"Speaker identification cancelled by user"` (CancelledError)
- Child error value format: `f"{type(exc).__name__}: {exc}"`.

---

## 13. Microphone recorder (`src/mic_recorder.py`)

### 13.1 Constants

`src/mic_recorder.py:25-33`:

| Constant | Value | Note |
|---|---|---|
| `_SAMPLE_RATE` | `16000` | |
| `_CHANNELS` | `1` | |
| `_CHUNK_SAMPLES` | `512` | `blocksize` |
| `_FSYNC_EVERY` | `50` | chunks between flush/fsync/sidecar writes |
| `_VAD_QUEUE_MAXSIZE` | `60 * 16000 // 512` = `1875` | ~60 s of 512-sample chunks |

### 13.2 Device enumeration

`MicRecorder.list_devices()` (`src/mic_recorder.py:59-67`):

```python
import sounddevice as sd
result = []
for i, d in enumerate(sd.query_devices()):
    if d["max_input_channels"] > 0:
        result.append({"index": i, "name": d["name"]})
return result
```

- Returns **all** devices with at least one input channel, in `sounddevice.query_devices()` order.
- `index` is the global device index (including output-only devices in the numbering); it is not renumbered to the filtered list.
- `name` is passed through as returned; may be non-ASCII, duplicate, or empty.
- No default-device marker; the GUI adds `"Default microphone"` mapped to `None`.
- Any sounddevice error propagates to the GUI, which logs a warning and offers only the default (`src/gui.py:2005-2015`).

### 13.3 GUI device labels

`src/gui.py:2001-2017`:

1. Start with `["Default microphone"]` mapped to `None`.
2. For each device:
   - `label = _shorten(str(d["name"]).strip(), 48)`.
   - If label not already in the map: append and map to the **first** index with that label.
3. Duplicates are collapsed by displayed label; later duplicates are ignored. On Windows, MME/DirectSound/WASAPI often expose the same mic multiple times; the first (usually most compatible) wins.
4. `_shorten(text, 48)` uses middle ellipsis: `len <= 48` unchanged; else keep 47, head 31 chars, `"…"`, last 16 chars.
5. If a real device is named exactly `"Default microphone"` (after trimming), it is skipped because the label collides with the default.
6. On enumeration failure: log warning `"Could not enumerate microphones: {exc}"`; only default remains.

At record time (`src/gui.py:1451-1452`): `device_index = self._mic_index_map.get(mic_label)`; a missing label → `None` (default). Passed to `MicRecorder.start`.

### 13.4 `MicRecorder.__init__`

`src/mic_recorder.py:45-57`:

```python
self._stream = None
self._vad_queue = queue.Queue(maxsize=1875)
self._pcm_queue = queue.Queue()          # unbounded
self._writer_thread = None
self._stop_event = threading.Event()
self._pcm_path = None
self._json_path = None
self._samples_written = 0
self._vad_drop_logged = False
self._feed_vad = True
self._peak = 0.0
```

- `_vad_queue` is bounded; `put_nowait` drops the **newest** chunk when full.
- `_pcm_queue` is unbounded; the writer is expected to keep up.
- `_peak` is the max absolute sample since last `get_level()`, normalized.
- `_feed_vad` controls whether audio is also sent to the VAD queue; record-only sets false.

`get_queue() -> queue.Queue` (`src/mic_recorder.py:69-71`) returns the `_vad_queue` object itself (bounded to 1875 chunks). The live transcriber consumes it; the PCM writer does not. Callers must not put into it. `get_level()`, `get_session_stem()`, `start()`, and `stop()` are described below.

### 13.5 `start(device_index, output_dir, feed_vad=True)`

`src/mic_recorder.py:83-117`:

1. `import sounddevice as sd`.
2. `output_dir.mkdir(parents=True, exist_ok=True)`.
3. `ts = datetime.now().strftime("%Y%m%d_%H%M%S")`.
4. `_pcm_path = output_dir / f"recording_{ts}.pcm"`.
5. `_json_path = output_dir / f"recording_{ts}.json"`.
6. Reset `_samples_written=0`, `_vad_drop_logged=False`, `_feed_vad=feed_vad`, `_peak=0.0`, `_stop_event.clear()`.
7. Start writer thread:
   - target `_pcm_writer_loop`, daemon, name `"PCMWriter"`.
8. Create `sd.InputStream`:
   - `samplerate=16000`
   - `channels=1`
   - `dtype="int16"`
   - `blocksize=512`
   - `device=device_index` (`None` = default)
   - `callback=self._audio_callback`
9. `self._stream.start()`.
10. `logger.info(f"MicRecorder started: {self._pcm_path.name}")`.

Order matters: the writer thread starts **before** the stream. If stream creation fails, the writer thread exists with `_stop_event` clear and no audio; the exception propagates to the recording worker, which logs and finishes, but does not call `stop()` in the error path (see `src/gui.py:1737-1744`), leaving a daemon writer thread until process exit. **DEFECT/leak**; Rust should clean up on stream-open failure.

`_pcm_path`/`_json_path` are set before the thread; `_pcm_writer_loop` returns immediately if `_pcm_path is None` (not the case here).

No check for an existing same-second file: opening `"ab"` appends to an existing PCM; the sidecar is overwritten with the new session’s `samples_written`, corrupting recovery. **DEFECT** (see §24).

### 13.6 `stop(convert_to_wav=True, dest_dir=None) -> Path | None`

`src/mic_recorder.py:119-163`:

1. If `_stream is not None`: `stop()`, `close()`, set `None`.
2. `_stop_event.set()`.
3. If `_writer_thread is not None`: `join(timeout=5)`, set `None`.
4. If `not convert_to_wav or _pcm_path is None`: return `None`.
5. `metadata`:
   ```python
   {
       "sample_rate": 16000,
       "channels": 1,
       "bit_depth": 16,
       "samples_written": self._samples_written,
   }
   ```
6. `wav_name = self._pcm_path.with_suffix(".wav").name`.
7. `folders = [self._pcm_path.parent]`.
8. If `dest_dir is not None and dest_dir != self._pcm_path.parent`: insert `dest_dir` at index 0 (try chosen first).
9. For each folder:
   - `wav_path = folder / wav_name`.
   - `try`: `folder.mkdir(parents=True, exist_ok=True)`; `recover_pcm_to_wav(self._pcm_path, metadata, wav_path)`.
   - `except Exception as exc`: log `"PCM→WAV conversion to {folder} failed: {exc}"`; `continue`.
   - On success: unlink `.pcm` (`missing_ok=True`); unlink `.json` if set (`missing_ok=True`); `logger.info("Recording saved: {wav_path}")`; return `wav_path`.
10. If all fail: return `None`.

Details:

- The WAV is written from exactly `samples_written * 2` bytes (truncates any unsynced tail).
- Raw files are deleted **only after** a successful WAV write; on total failure they remain for recovery.
- If the chosen dest equals the PCM parent (`Path` equality), it is not tried twice.
- If dest is unusable (e.g. a file), the first attempt fails, then the fallback writes next to the PCM. The GUI logs a warning if `wav_path.parent != dest_dir` (`src/gui.py:1703-1706`).
- Existing WAV at the target is overwritten.
- `stop()` can be called without `start()`: `_pcm_path is None` → returns `None`.
- If the writer thread did not finish within 5 s, `_samples_written` may lag; the PCM is truncated to that lagging count, losing the tail. **DEFECT/edge**; Rust should join until the writer is done or report an error.
- `_pcm_path`/`_json_path` remain set after stop; `get_session_stem()` still returns the stem (useful to GUI before stop; after stop it still returns it).
- `convert_to_wav=False` leaves raw PCM/JSON and returns `None`; the GUI never uses this today.

### 13.7 Audio callback

`src/mic_recorder.py:165-179`:

```python
def _audio_callback(self, indata, frames, time_info, status):
    if status:
        logger.warning(f"sounddevice status: {status}")
    chunk = indata[:, 0].copy()  # int16, shape (frames,)
    if chunk.size:
        self._peak = max(self._peak, float(np.abs(chunk.astype(np.int32)).max()) / 32768.0)
    if self._feed_vad:
        try:
            self._vad_queue.put_nowait(chunk.astype(np.float32) / 32768.0)
        except queue.Full:
            if not self._vad_drop_logged:
                self._vad_drop_logged = True
                logger.warning("VAD queue full — dropping audio from live transcription")
    self._pcm_queue.put(chunk.tobytes())
```

- `indata` is shape `(frames, channels)`; only channel 0 is used (mono stream, so fine).
- `.copy()` detaches from the PortAudio buffer.
- Peak calculation casts int16 → int32 before `abs` to avoid `abs(-32768)` overflow; max 32768 → 1.0.
- VAD gets float32 in `[-1.0, 1.0)`; conversion happens per chunk.
- If the VAD queue is full, the **newest** chunk is dropped; the first drop logs once per session, subsequent drops are silent. The PCM path still receives every chunk.
- `_pcm_queue.put` is unbounded and never drops.
- `status` warnings (input overflow etc.) are logged on every callback where status is truthy; can be noisy.
- No sample-rate conversion: the stream requests 16 kHz; PortAudio/OS handles conversion or fails.
- Rust callback must avoid blocking/allocating too much; use a lock-free ring buffer for PCM and VAD, or copy into a preallocated buffer. Preserve the exact int16→float scaling and peak semantics.

### 13.8 PCM writer

`src/mic_recorder.py:181-204`:

```python
if self._pcm_path is None: return
chunk_count = 0
try:
    with open(self._pcm_path, "ab") as pcm_file:
        while not self._stop_event.is_set() or not self._pcm_queue.empty():
            try:
                raw = self._pcm_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            pcm_file.write(raw)
            self._samples_written += len(raw) // 2
            chunk_count += 1
            if chunk_count % 50 == 0:
                pcm_file.flush()
                os.fsync(pcm_file.fileno())
                self._flush_json()
        pcm_file.flush()
        os.fsync(pcm_file.fileno())
    self._flush_json()
except Exception as exc:
    logger.error(f"PCM writer error: {exc}")
```

- Opens append binary; if the file exists, appends.
- Drains queue after stop set until empty.
- `len(raw)//2` assumes int16 and an even byte count; chunks are always 512×2 bytes normally; an odd/empty chunk would undercount.
- Every 50 chunks: flush, `os.fsync`, write sidecar.
- Final flush/fsync, then sidecar after closing.
- On any exception, logs and exits; the audio callback continues queueing, memory grows; `stop()` joins up to 5 s and proceeds, potentially with a partial file.
- `os.fsync` is expensive but gives crash safety at ~1.6 s granularity.
- No WAV header; recovery handles it.

`_flush_json` (`src/mic_recorder.py:206-218`):

```python
data = {"sample_rate": 16000, "channels": 1, "bit_depth": 16,
        "samples_written": self._samples_written}
self._json_path.write_text(json.dumps(data), encoding="utf-8")
```

- Compact JSON with spaces after `:`/`,`; no newline.
- Catches all exceptions and logs `"JSON sidecar write failed: {exc}"`.
- If `_json_path is None`, returns.
- Not atomic.

### 13.9 `get_level() -> float`

`src/mic_recorder.py:78-81`:

```python
peak, self._peak = self._peak, 0.0
return peak
```

- Returns the maximum normalized absolute sample since the previous call, then resets to 0.
- Not thread-safe in a formal sense; Python GIL makes the tuple swap atomic enough. Rust use `AtomicU32` (bit-cast f32) with `swap(0)` or a mutex.
- GUI polls every 100 ms and displays `min(1.0, level ** 0.5)` (`src/gui.py:2362-2370`), i.e. square root for visibility.
- When recording stops, GUI sets the bar to 0 (`src/gui.py:2363-2366`).

### 13.10 `get_session_stem()`

`src/mic_recorder.py:73-76`: `self._pcm_path.stem if _pcm_path is not None else None`.

- Before `start()`: `None`.
- After `start()`: `"recording_YYYYMMDD_HHMMSS"`.
- After `stop()`: same (path object retained).

### 13.11 Crash-safety contract

- At any kill, the `.pcm` may contain up to the last written chunk, but the `.json` `samples_written` is only guaranteed up to the last 50-chunk fsync.
- Recovery truncates the PCM to `samples_written * 2` bytes, so unsynced tail audio is **discarded**, not recovered. This is intentional to avoid a WAV with an unknown length.
- If the process is killed between writing PCM and fsyncing the sidecar, the last up-to-1.6 s is lost. If killed after fsync but before sidecar write, recovery truncates to the previous count; same.
- The `.json` and `.pcm` are separate files; a crash can leave an invalid/partial JSON. Recovery skips the pair if JSON is invalid, preserving it for a future launch (the user can manually fix it).
- Declining recovery leaves both files untouched so the prompt reappears (`src/recovery.py:69-78`).

### 13.12 Rust notes

- Use `cpal` (cross-platform) or a PortAudio binding. If using `cpal`, choose the default input device or the enumerated device; request 1 channel, 16 kHz, and convert from the device’s native sample format/rate to i16 16 kHz. `sounddevice`/PortAudio may perform host-level conversion; cpal does not automatically resample in all backends. Implement a resampler (e.g. linear or rubato) and document that it is an addition required for cross-platform behavior; on Windows WASAPI shared mode, matching the reference may require requesting 16 kHz with `AUTOCONVERTPCM` or using the device mix format and resampling.
- Device indexes: cpal’s `input_devices()` order may differ from PortAudio’s. The GUI must map labels to stable device IDs, not assume numeric index parity. On Windows, preserve the “first duplicate label wins” rule.
- Callback: no blocking, no `println!`, no allocation in the real-time thread. Send fixed-size `[i16; 512]` blocks over a bounded ring; if the ring is full, follow the same drop policy (VAD newest dropped, PCM never dropped).
- Writer: use `BufWriter`, `flush`, `sync_all` every 50 blocks; sidecar temp+rename (or direct write) with the exact JSON.
- Peak: atomic `u32` bit pattern; `fetch_max` with CAS or mutex.
- Clean up the writer/stream on any start failure.

---

## 14. Live transcriber (`src/live_transcriber.py`)

### 14.1 Constants and queues

`src/live_transcriber.py:28-35`:

| Constant | Value | Note |
|---|---|---|
| `_DECODE_QUEUE_MAXSIZE` | `8` | segments waiting for the recognizer |
| `_STOP_TIMEOUT_SEC` | `60.0` | total join budget |
| `_HISTORY_SAMPLES` | `PAD_SAMPLES + 16 * VAD_CHUNK_SAMPLES` = `3200 + 8192` = `11392` | 0.712 s rolling history |
| `_HISTORY` purpose | cover `PAD_SAMPLES` + VAD detection lag | comment `src/live_transcriber.py:33-35` |

Queues:

- `mic_queue`: from `MicRecorder.get_queue()`, max 1875, float32 chunks, **drops newest** when full.
- `decode_queue`: max 8, segment arrays, **drops oldest** when full (`_put_dropping_oldest`).
- `vad_done` event: set by VAD thread after final flush; decode thread exits when queue empty and this is set.

### 14.2 `LiveTranscriber.start(...)`

`LiveTranscriber.__init__` (`src/live_transcriber.py:50-52`) only stores `_vad_thread = None` and `_decode_thread = None`; no model is loaded and no thread starts until `start()`.

`src/live_transcriber.py:54-89`:

```python
start(
    recognizer,
    mic_queue: queue.Queue,
    cancel_event: threading.Event,
    on_segment: Callable[[str], None],
    on_overflow: Callable[[], None] | None = None,
) -> None
```

1. `decode_queue = queue.Queue(maxsize=8)`.
2. `vad_done = threading.Event()`.
3. `onnx_lock = threading.Lock()` shared by both loops.
4. Create decode thread:
   - target `_decode_loop`
   - args `(recognizer, decode_queue, vad_done, on_segment, onnx_lock)`
   - daemon, name `"LiveDecoder"`.
5. Create VAD thread:
   - target `_vad_loop`
   - args `(mic_queue, decode_queue, cancel_event, vad_done, on_overflow, onnx_lock)`
   - daemon, name `"LiveVAD"`.
6. **Start decode thread first**, then VAD thread.
7. `logger.info("LiveTranscriber started")`.

The order ensures the decoder is ready before segments arrive; even if VAD starts first, the queue would buffer up to 8.

### 14.3 `LiveTranscriber.stop(timeout=60.0)`

`src/live_transcriber.py:91-106`:

```python
deadline = time.monotonic() + timeout
for attr in ("_vad_thread", "_decode_thread"):
    thread = getattr(self, attr)
    if thread is not None:
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if thread.is_alive():
            logger.warning(f"{thread.name} did not finish within {timeout:.0f} s")
        setattr(self, attr, None)
logger.info("LiveTranscriber stopped")
```

- Caller must have set `cancel_event` first; `stop()` does not set it.
- Joins VAD first, then decoder, sharing one deadline.
- If VAD consumes the whole 60 s, decoder gets 0 and may be left running; it logs a warning and drops the handle. The daemon thread may later call `on_segment` after the UI thinks recording stopped. **DEFECT/edge**; Rust should use a single cancellation token and report timeout.
- `timeout` is formatted with `.0f` in the warning (Python round-half-even).
- After stop, `_vad_thread`/`_decode_thread` are `None`; a new `start()` can be called for a new recording.

### 14.4 VAD loop (`_vad_loop`)

`src/live_transcriber.py:108-177`.

Initialization:

```python
detector = vad.build_vad(buffer_size_in_seconds=30)
history = np.zeros(0, dtype=np.float32)
history_start = 0
prev_segment_end = 0
overflow_notified = False
```

- `detector` buffer 30 s; a continuous speech segment longer than the buffer is handled by sherpa (likely forced at buffer boundaries); the wrapper does not add its own split.
- `history` starts empty; `history_start=0` absolute.
- `prev_segment_end=0`.
- `overflow_notified` suppresses repeated overflow logs/callbacks.

`push_ready_segments()`:

```python
while not detector.empty():
    segment = detector.front
    seg_start = segment.start
    samples = vad.pad_live_segment(history, history_start, prev_segment_end, seg_start, segment.samples)
    prev_segment_end = seg_start + len(segment.samples)
    if not _put_dropping_oldest(decode_queue, samples) and not overflow_notified:
        overflow_notified = True
        logger.warning("Live decode queue full — dropped oldest segment")
        if on_overflow is not None:
            on_overflow()
    detector.pop()
```

- Drags all currently ready segments.
- `prev_segment_end` is the original end, not padded; since padding is leading-only, this prevents overlap with the previous segment.
- If `_put_dropping_oldest` returns False (it had to drop), notify once.
- `detector.pop()` after copying/using `segment.samples`; the Python binding’s reference is invalid after pop.

`feed(chunk)`:

```python
history = np.concatenate([history, chunk])
if len(history) > _HISTORY_SAMPLES:
    trim = len(history) - _HISTORY_SAMPLES
    history = history[trim:]
    history_start += trim
with onnx_lock:
    detector.accept_waveform(chunk)
    push_ready_segments()
```

- The history trim happens before VAD; `history_start` advances by the trimmed count.
- `_HISTORY_SAMPLES=11392` is enough for 3200 pad + 8192 (16 chunks) of margin.
- `accept_waveform` + `push_ready_segments` are inside the same `onnx_lock` so the VAD and recognizer never run concurrently.
- `np.concatenate` allocates a new array each chunk; Rust should use a ring buffer for efficiency, preserving the same logical window.

Main loop:

```python
while not cancel_event.is_set():
    try:
        chunk = mic_queue.get(timeout=0.1)
    except queue.Empty:
        continue
    feed(chunk)
```

- Polls the mic queue every 100 ms; cancellation is observed within ~100 ms plus the current feed.
- If cancel is set, exit loop and drain:

```python
while True:
    try:
        feed(mic_queue.get_nowait())
    except queue.Empty:
        break
with onnx_lock:
    detector.flush()
    push_ready_segments()
```

- Drains everything currently queued (may include audio captured after Stop was pressed but before the mic stream is stopped).
- Feeds it through VAD (and history) before flushing.
- `flush()` emits a segment for speech still in progress.
- `push_ready_segments` pushes final segments.
- Any exception in the try block is caught by `except Exception: logger.exception("Live VAD thread failed")`; the flush/drain code only runs on normal cancel, **not** after an exception. `finally: vad_done.set()` always runs.
- No `CancelledError` is raised; live stop is graceful, not an error.

Test `tests/test_live_transcriber.py:186-206` simulates this:

- Fake detector only emits on `flush()`.
- Mic queue receives 1600 samples before start, then 1600 + 800 while VAD is running; Stop is set.
- `stop(timeout=10)` must drain all queued audio and flush; expected decoded sample count `"4000"` (1600 + 1600 + 800 = 4000). This proves queued audio is not lost.

### 14.5 Decode loop (`_decode_loop`)

`src/live_transcriber.py:179-202`:

```python
while True:
    try:
        samples = decode_queue.get(timeout=0.1)
    except queue.Empty:
        if vad_done.is_set():
            return
        continue
    with onnx_lock:
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, samples)
        recognizer.decode_stream(stream)
        text = stream.result.text.strip()
    if text:
        on_segment(text)
```

- Polls every 100 ms; exits only when queue empty **and** `vad_done` set.
- Recognizer calls serialized with VAD via `onnx_lock`.
- `on_segment` called outside the lock, on the decoder thread, with stripped text.
- Empty/whitespace-only results are dropped.
- No timestamps are produced or passed; the GUI appends each text as a line.
- No exception handling: a recognizer error kills the decoder thread; `vad_done` may later be set, but no error is surfaced. **DEFECT/edge**; Rust should send an error to the UI and stop cleanly.
- `accept_waveform` uses `config.VAD_SAMPLE_RATE` (16000), not a device rate.

### 14.6 `_put_dropping_oldest(q, item) -> bool`

`src/live_transcriber.py:205-222`:

```python
try:
    q.put_nowait(item)
    return True
except queue.Full:
    try:
        q.get_nowait()
    except queue.Empty:
        pass
    try:
        q.put_nowait(item)
    except queue.Full:
        pass
    return False
```

- On full: discard the oldest queued segment, then insert the new one.
- Returns `False` whenever the first put failed, even if the second put also failed.
- If the queue is still full after dropping one (race with another producer; not expected with a single VAD thread), the new item is dropped too.
- Test `tests/test_live_transcriber.py:109-131` uses a slow fake recognizer and 50 puts; asserts at least one drop and `qsize() <= 8`.

### 14.7 Stop/flush ordering contract

GUI (`src/gui.py:1694-1702`):

1. `_stop_recording_event.wait()` returns when the user presses Stop.
2. If live: status `"Finishing"`, then `self._live_transcriber.stop()`; this joins VAD (drains + flushes) and decoder (finishes queue).
3. Then `self._mic_recorder.stop(...)` stops the audio stream and converts PCM to WAV.
4. The live transcript is written from `transcript_lines` collected by `on_segment`.

Consequences:

- Audio captured between the Stop press and `MicRecorder.stop()` is appended to the PCM/WAV but may not be transcribed live, because the VAD thread drains only what is in `mic_queue` at the moment it sees the cancel. This is a deliberate trade-off; the WAV is complete.
- Live transcript has no timestamps; it is a sequence of non-empty segment texts, one per line.
- If the decoder is slow, `stop()` waits up to 60 s; the UI shows “Finishing the last words”.
- If `on_overflow` fired, the GUI logs:
  `"[Warning] transcription falling behind. Some audio was skipped in the live view; the full recording is still saved"` (`src/gui.py:1664-1668`).

### 14.8 Rust notes

- Use `Arc<AtomicBool>` for cancel/stop and `crossbeam-channel::bounded(8)` for decode. Implement `try_send`; on full, `try_recv` oldest then `try_send` again; return false if the first send failed.
- Use a shared `Mutex<SherpaEngine>` for VAD+recognizer calls. A `std::sync::Mutex` is fine; do not hold it while calling user callbacks.
- History: a `VecDeque<f32>` with absolute `history_start` or a ring buffer; trim to 11,392 samples before each VAD feed. Implement `pad_live_segment` with the same max/clamp and slice.
- Stop: drain the mic channel with `try_recv` until empty, feed, flush, then signal `vad_done`; join VAD then decoder with one deadline.
- Decoder must surface errors via an error channel; do not silently kill the thread.
- The callback `on_segment(String)` may be called from the decoder thread; UI adapter must send to the event loop.
- Keep the exact overflow log text (em dash) and on_overflow once per session.

---

## 15. Recovery (`src/recovery.py`)

### 15.1 `scan_for_orphans(output_dir) -> list[(pcm_path, json_path, metadata)]`

`src/recovery.py:17-37`:

1. If `output_dir` does not exist → `[]`.
2. `for pcm_path in sorted(output_dir.glob("recording_*.pcm")):` — only top-level, case-sensitive glob on POSIX (Windows glob semantics may be case-insensitive).
3. `json_path = pcm_path.with_suffix(".json")`.
4. If JSON does not exist → skip (the PCM is left alone; no prompt).
5. `json.loads(json_path.read_text(encoding="utf-8"))`; any exception → `logger.warning("Could not read {json.name}: {exc}")`; skip.
6. Append `(pcm_path, json_path, metadata)`.
7. Return sorted by full path/filename (because iteration is sorted and the list is appended in order).

Notes:

- It does not validate that metadata is a dict or contains keys.
- A `.json` with invalid JSON causes the pair to be skipped and left on disk; the user cannot recover it through the UI, but files are not deleted.
- `recording_*.pcm` matches exactly the prefix; other PCM names are ignored.
- A `.pcm` without `.json` is ignored forever (unless a sidecar appears).

### 15.2 `recover_pcm_to_wav(pcm_path, metadata, wav_path) -> None`

`src/recovery.py:40-55`:

1. `sample_rate = int(metadata.get("sample_rate", 16000))`.
2. `channels = int(metadata.get("channels", 1))`.
3. `samples_written = int(metadata.get("samples_written", 0))`.
4. `byte_count = samples_written * 2` (assumes int16).
5. `raw = pcm_path.read_bytes()[:byte_count]`.
6. Open `wav_path` `"wb"`:
   - `setnchannels(channels)`
   - `setsampwidth(2)`
   - `setframerate(sample_rate)`
   - `writeframes(raw)`
7. `logger.info(f"Recovered {pcm_path.name} → {wav_path.name} ({len(raw):,} bytes)")`.

Notes:

- Truncates to `samples_written * 2`; extra bytes in the PCM (unsynced tail) are ignored. This is the crash-safety contract.
- If `samples_written` exceeds the actual file length, the WAV is shorter than the metadata claims (the slice just returns what exists).
- If `samples_written` is negative, Python slicing `[:negative]` returns all but the last N bytes; **DEFECT/edge**. Validate nonnegative in Rust.
- `channels` is used in the WAV header, but `samples_written` is a mono sample count; for channels >1 the frame count/interpretation would be off. Production sidecars are channels=1.
- `bit_depth` is ignored; always writes 16-bit.
- Does not create parent directories (callers do).
- Overwrites an existing WAV.
- Errors propagate (`read_bytes`, `wave.Error`, `OSError`, `TypeError`, `ValueError` from int conversion).

### 15.3 `_handle_orphan(pcm_path, json_path, metadata, recover, dest_dir=None) -> Path | None`

`src/recovery.py:58-91`:

1. If `not recover`: return `None`; **do not touch either file** (declining must not delete the only copy; the prompt reappears next launch).
2. `wav_path = pcm_path.with_suffix(".wav")`.
3. If `dest_dir is not None`:
   - `try`:
     - `dest_dir.mkdir(parents=True, exist_ok=True)`.
     - `recover_pcm_to_wav(pcm_path, metadata, dest_dir / wav_path.name)`.
     - `wav_path = dest_dir / wav_path.name`.
   - `except OSError as exc`: log `"Could not recover into {dest_dir}, keeping it next to the raw file: {exc}"`; `recover_pcm_to_wav(pcm_path, metadata, wav_path)` (next to PCM).
   - Only `OSError` is caught; `ValueError`/`TypeError` from bad metadata propagate.
4. Else: `recover_pcm_to_wav(pcm_path, metadata, wav_path)`.
5. `pcm_path.unlink(missing_ok=True)`.
6. `json_path.unlink(missing_ok=True)`.
7. Return `wav_path`.

Notes:

- If the destination write succeeds but unlinking fails, the exception propagates after the WAV exists; raw files may remain. A later recovery could overwrite/duplicate.
- If the destination write partially succeeds and then raises `OSError`, the fallback writes next to the PCM; a partial/complete file may remain in the destination.
- If `dest_dir` is the PCM parent, it still goes through the destination branch (mkdir no-op, write same path), then deletes raw files. Fine.
- `json_path` is required by `scan_for_orphans`, but `_handle_orphan` accepts any path and deletes it.
- Declining returns `None` and leaves both files (test `tests/test_recovery.py:69-78`).
- Accepting deletes both and returns the WAV (test `tests/test_recovery.py:81-94`).

### 15.4 `prompt_and_recover(output_dir, dest_dir=None) -> None`

`src/recovery.py:94-145`:

1. Lazy import `tkinter` and `messagebox`.
2. `orphans = scan_for_orphans(output_dir)`; if empty return.
3. Create `root = tk.Tk()`; `root.withdraw()`; `root.update()`.
4. For each orphan:
   - `samples = int(metadata.get("samples_written", 0))`.
   - `sr = int(metadata.get("sample_rate", 16000))`.
   - `duration_sec = samples / sr if sr > 0 else 0`.
   - `duration_str = f"{int(duration_sec // 60)}m {int(duration_sec % 60)}s"` — no zero padding; 65 s → `"1m 5s"`; 3600 s → `"60m 0s"`.
   - `answer = messagebox.askyesno("Unfinished Recording Found", "...", parent=root)` with exact text:
     ```
     An unfinished recording was found:

       File: {pcm_path.name}
       Duration: {duration_str}

     Recover it as a .wav file now?
     (Choosing No keeps the raw recording for next time.)
     ```
   - `try: wav_path = _handle_orphan(..., answer, dest_dir)`.
   - `except Exception as exc`: log `"Recovery failed for {pcm_path.name}: {exc}"`; show error dialog `"Recovery Failed"` / `"Could not recover {pcm_path.name}:\n{exc}"`; continue.
   - If `wav_path is not None`: show info dialog `"Recovery Complete"` / `"Saved to:\n{wav_path}"`.
5. `root.destroy()`.

- Called before the main GUI window (`src/main.py:104-110`); a withdrawn Tk root exists only for dialogs.
- `root.update()` is needed to make dialogs appear correctly.
- If `messagebox`/Tk fails, the exception propagates to `main`’s broad `except Exception` around recovery, which logs a warning and continues; a partially created root may remain.
- Each orphan gets its own yes/no; no “recover all”.
- The prompt is modal per orphan; declining leaves the files and they prompt again next launch.
- The destination is `load_recordings_dir() or DEFAULT_RECORDINGS_DIR` from main. If the chosen folder is missing, settings returns `None` → default.

### 15.5 Rust notes

- Use the native GUI dialog API of the chosen Rust UI framework; exact strings are part of the UX contract.
- Run recovery before the main window, with the same startup ordering.
- Scanning must be deterministic (sort by file name).
- Validate metadata types and nonnegative `samples_written`; preserve the “decline keeps files” invariant.
- Write WAV with a standard 16-bit PCM header; use a temp file + rename for the destination if possible, but keep the fallback semantics.
- Recovery dialogs must not block the main app if the user is absent? The reference blocks; parity.

---

## 16. Rename (`src/rename.py`)

### 16.1 Constants

`src/rename.py:13-15`:

```python
_BAD_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_RESERVED = re.compile(r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])$", re.IGNORECASE)
MAX_NAME = 120
```

### 16.2 `check_name(name) -> str`

`src/rename.py:18-28`:

1. `if not name: return "Type a name first."`
2. If `_BAD_CHARS.search(name)`: return `'A name cannot have these characters: \\ / : * ? " < > |'`.
3. If `_RESERVED.match(name)` or `name.endswith(".")`: return `"Windows does not allow this name. Please choose another."`.
4. If `len(name) > 120`: return `f"Please use a shorter name (at most {MAX_NAME} characters)."`.
5. Else `""` (valid).

Exact behavior/quirk:

- Empty string only; a string of spaces is **valid** at this API level. GUI trims the entry before calling (`src/gui.py:2295`), so spaces become empty and fail with “Type a name first.”
- Bad character set: backslash, forward slash, colon, asterisk, question mark, double quote, `<`, `>`, pipe, and control characters U+0000–U+001F.
- Does not reject other Windows-invalid characters: trailing spaces, leading spaces, names containing Unicode control beyond 0x1F, etc.
- Reserved regex is anchored and case-insensitive: only exact `CON`, `PRN`, `AUX`, `NUL`, `COM1`–`COM9`, `LPT1`–`LPT9`.
  - `CON.txt` is **allowed** by this function, though Windows refuses it. **DEFECT/gap**.
  - `COM0`, `COM10`, `LPT0`, `LPT10` are allowed (regex only 1–9).
- `name.endswith(".")` rejects any trailing period, including `"."` and `".."`.
- `len(name)` counts Unicode code points, not bytes; 120 emoji are allowed.
- Returns an error string, never raises.
- The exact returned string is `A name cannot have these characters: \ / : * ? " < > |` (one backslash, then a space, then the other characters). The Python source escapes the backslash as `\\` in the string literal.

Rust:

```rust
pub const MAX_NAME: usize = 120;
pub fn check_name(name: &str) -> Result<(), NameError>
```

or return the same strings. Use `name.chars().count()` for length. Regex/character checks can be manual; reject `\\ / : * ? " < > |` and `'\u{0}'..='\u{1f}'`. Case-insensitive exact reserved match via ASCII lowercase. Preserve the `CON.txt` gap unless an approved fix; if fixing, document.

### 16.3 `recording_set(audio) -> list[Path]`

`src/rename.py:31-38`:

```python
stem = audio.stem
return sorted(
    p for p in audio.parent.iterdir()
    if p.is_file() and p.suffix.lower() in (".wav", ".txt")
    and (p.stem == stem or p.stem.startswith(stem + " ("))
)
```

- Scans the audio’s parent directory (non-recursive).
- Files only.
- Suffixes case-insensitive `.wav`/`.txt`.
- Stem equals the audio stem, or starts with `<stem> (`.
- Examples for stem `recording_20261021_143207`:
  - included: `.wav`, `.txt`, ` (best quality).txt`, ` (speakers).txt`, ` (best quality) (speakers).txt`
  - excluded: `_2.wav` (stem `..._2`), `.json`, another recording’s `.txt`.
- Returns sorted by full path (Python `Path` ordering). On Windows, `WindowsPath` comparison is case-insensitive? Python’s `PureWindowsPath` ordering uses casefolded parts; verify. Rust should sort by the same effective name ordering on Windows; a simple byte sort may differ for mixed case. Document.
- If the audio file does not exist, its parent may still be iterated; the set may include other files with the same stem.
- Does not include `.pcm`/`.json`, so an in-progress recording cannot be renamed via this set.

### 16.4 `rename_all(paths, old_stem, new_stem) -> dict[Path, Path]`

`src/rename.py:41-62`:

```python
plan = {p: p.with_name(new_stem + p.name[len(old_stem):]) for p in paths}
taken = [new for old, new in plan.items() if new.exists() and new != old]
if taken:
    raise FileExistsError(taken[0].name)
done = {}
try:
    for old, new in plan.items():
        if new != old:
            old.rename(new)
        done[old] = new
except OSError:
    for old, new in done.items():
        if new != old:
            new.rename(old)
    raise
return done
```

- The new name is `new_stem` + the suffix/remainder after `old_stem` in each file’s name. This assumes every path starts with `old_stem`; if not, the remainder is wrong.
- Checks all targets before renaming: if any target exists and differs from its source, raise `FileExistsError(target.name)` and change nothing.
- `plan` preserves the input order (dict insertion order); `taken[0]` is the first conflict in that order.
- Renames are sequential. If a rename raises `OSError`, it attempts to undo every already-completed rename in `done` (insertion order), then re-raises the original exception. If an undo fails, that exception replaces the original.
- Unchanged paths (`new == old`) are included in the returned map but not touched.
- Returns `{old_path: new_path}` for every input (including unchanged).
- Does not validate `new_stem`; GUI calls `check_name` first.
- Does not create directories; all files are in the same folder.
- Does not handle a target that is another source being renamed away: `new.exists()` is true, so it rejects, even if a sequential rename could work. No cycle resolution.
- Case-only rename: Python `WindowsPath` equality is case-insensitive; on Windows, `new.exists()` true and `new == old` may be true, allowing the rename. On POSIX, `new != old` and `new.exists()` true (if a differently-cased file exists) → `FileExistsError`. Rust must replicate per-platform semantics; see §20.9.
- If `paths` is empty: plan/taken/done empty, returns `{}`.
- If `old_stem` is empty: `p.name[0:]` = full name; new = `new_stem + full_name`; `with_name` may produce a valid name. GUI prevents empty via `check_name`.

Tests `tests/test_rename.py:46-105`:

- `check_name` accepts `"Team meeting 21 Oct"`; rejects `""`, `"a/b"`, `"a:b"`, `'say "hi"'`, `"what?"`, `"CON"`, `"lpt1"`, `"name."`, `"x"*200`.
- `recording_set` includes exactly the four set files, excludes `_2.wav`, another recording `.txt`, `.json`.
- `rename_all` renames all four together; returns old→new mappings; contents preserved.
- Taken name → `FileExistsError`, nothing changes, existing target content untouched.
- Failure on the 3rd rename → all files restored to original names.
- Transcript-only rename works and leaves the WAV.

### 16.5 Rust notes

- Use `std::fs::rename`; preflight all targets with `Path::exists()`.
- Preserve rollback. Log rollback failures; do not mask the original error if possible (Python does mask, but this is a robustness improvement).
- On Windows, compare paths case-insensitively for the `new == old` check and for existence conflicts, matching `WindowsPath` semantics. A simple `to_string_lossy().to_lowercase()` is not fully correct for Unicode; use `CompareStringOrdinal` or `Path::eq` with Windows semantics if exact. Document.
- Return a `Vec<(PathBuf, PathBuf)>` or `HashMap` preserving order; GUI updates its `_last_*` paths by lookup.
- Rename atomicity: each rename is atomic on the same volume; across volumes not possible (same folder, so same volume).

---

## 17. Main entry point (`src/main.py`)

### 17.1 `_cleanup_temp_files()`

`src/main.py:30-47`:

```python
for pattern in ("*.wav", "*.json"):
    try:
        for path in TEMP_DIR.glob(pattern):
            try:
                path.unlink()
            except OSError:
                pass
    except Exception:
        pass
```

- Only top-level `temp/`.
- `.wav` first, then `.json`.
- Individual unlink errors ignored; glob-level errors ignored.
- Does not delete `.bin`, extensionless, or subdirectories.
- Called at startup and via `atexit`; not called after each job (GUI deletes its own temp WAV).

### 17.2 `_set_taskbar_identity()`

`src/main.py:50-64`:

- Windows only; return immediately otherwise.
- `ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("EasyScribe.App")`.
- Any exception ignored (cosmetic).
- Must run before the first window is created so the taskbar uses the window icon rather than a cached exe icon.

### 17.3 `_check_dependencies() -> list[str]`

`src/main.py:67-86`:

1. If `not FFMPEG_BIN.exists()`:
   ```
   ffmpeg.exe not found.
   Expected at: {FFMPEG_BIN}

   The bundled ffmpeg folder may be missing.
   ```
2. If `not FFPROBE_BIN.exists()`:
   ```
   ffprobe.exe not found.
   Expected at: {FFPROBE_BIN}
   ```
3. `model_errors = validate_model_directory()`; if nonempty:
   ```
   Model files missing or incomplete:
     • {error1}
     • {error2}
   ```
   Each line prefixed with two spaces and a bullet `•` (U+2022), joined by `\n`.
4. Return errors list; empty means OK.

Does not check:

- `WHISPERCPP_BIN`/model (optional fallback);
- diarization models (checkbox disabled if absent);
- write permissions;
- free disk (GUI checks before batch).

### 17.4 `main()`

`src/main.py:89-136`:

1. `_set_taskbar_identity()`.
2. `setup_logging()`.
3. `log = logging.getLogger(APP_NAME)`.
4. If `offline_guard.is_installed()`: log `"Offline guard active: outbound network access is blocked"`; else log error `"Offline guard is NOT active"`.
5. `_cleanup_temp_files()`.
6. `atexit.register(_cleanup_temp_files)`.
7. `RECORDING_WORK_DIR.mkdir(parents=True, exist_ok=True)`.
8. Recovery:
   ```python
   try:
       from recovery import prompt_and_recover
       from settings import load_recordings_dir
       prompt_and_recover(RECORDING_WORK_DIR, load_recordings_dir() or DEFAULT_RECORDINGS_DIR)
   except Exception as exc:
       log.warning(f"Recovery check failed: {exc}")
   ```
9. Log `"Checking dependencies…"`.
10. `errors = _check_dependencies()`.
11. If errors:
    - Create `tk.Tk()`, withdraw, update.
    - `messagebox.showerror(f"{APP_NAME} cannot start", "Some program files are missing. Please download EasyScribe again.\n\n" + "\n\n".join(errors), parent=root)`.
    - `root.destroy()`.
    - `log.error(f"Startup aborted — missing dependencies: {errors}")`.
    - `sys.exit(1)`.
12. Log `"Dependencies OK — launching GUI"`.
13. `from gui import TranscriberApp`; `app = TranscriberApp()`; `app.mainloop()`.
14. Log `"Application exited normally"`.

`__main__` (`src/main.py:139-144`):

```python
multiprocessing.freeze_support()
main()
```

- `freeze_support()` is required for PyInstaller + spawn; in a source run it is harmless.
- `main()` does not catch `SystemExit` from the dependency failure.
- `atexit` cleanup runs on normal exit and `sys.exit`; not on a hard kill.
- The recovery dialog happens **before** dependency validation, so a missing-model install can still recover audio.
- If recovery hangs (modal dialog), startup waits.
- Logging is not configured when `offline_guard.install()` logs, hence the duplicate flag log in main.

### 17.5 Rust startup mapping

1. Install offline environment/no-network policy **before** any other initialization.
2. Resolve base/resource dirs, create `logs/`, `temp/`.
3. Initialize logger.
4. Clean temp files; register shutdown cleanup.
5. Create `recordings/`.
6. Scan/recover orphans with dialogs.
7. Validate ffmpeg/ffprobe/sherpa models; show fatal error and exit 1 if missing.
8. Start UI.
9. Child-process mode: detect a hidden worker argument **before** UI initialization (equivalent to `freeze_support`), run the diarization worker, exit.

---

## 18. GUI ↔ backend contract

This section captures behavior that backend callbacks must support for the app to behave identically, even if the Rust UI is different.

### 18.1 Status strings

`src/gui.py:139-163` maps internal status strings to UI text/colors/mode. Backend code must emit these exact strings:

| Status string | UI title | Progress mode | Emitted by |
|---|---|---|---|
| `"Ready"` | Ready | pause | GUI initial |
| `"Loading Model"` | Getting ready | busy | `TranscriptionEngine._ensure_model_loaded` |
| `"Extracting Audio"` | Reading the audio | busy | GUI before `extract_audio` |
| `"Transcribing"` | Transcribing | bar | `TranscriptionEngine.transcribe` |
| `"Identifying Speakers"` | Finding the speakers | busy | `speaker_transcript` |
| `"Naming Speakers"` | Waiting for speaker names | pause | `speaker_transcript` |
| `"Writing Transcript"` | Saving the transcript | bar | `transcribe`, `add_speakers_to_file` |
| `"Recording"` | Recording | busy | GUI recording worker |
| `"Finishing"` | Finishing the last words | busy | GUI live stop |
| `"Saving Recording"` | Saving the recording | busy | GUI record-only stop |
| `"Cancelling…"` | Stopping… | busy | GUI cancel button |
| `"Done"` | Done | pause | GUI final |
| `"Cancelled"` | Stopped | pause | GUI final |
| `"Failed"` | Something went wrong | pause | GUI final |

Hints (`src/gui.py:159-163`):

- `"Loading Model"` → `"Loading the speech model. This can take a minute."`
- `"Identifying Speakers"` → `"This step can take a while for long recordings."`
- `"Naming Speakers"` → `"Type each speaker's name below."`

Unknown status strings fall back to `(status, INK, TEAL, "pause")` (`src/gui.py:2316-2318`).

### 18.2 Progress callback

- Type: `Callable[[float], None]`, value nominally `0.0..1.0`.
- GUI clamps with `max(0.0, min(1.0, float(value)))`, sets the bar, and displays `int(round(value * 100))%` when running (`src/gui.py:2335-2341`).
- Python `round` is banker’s; for exact percent text, use half-even.
- Sources:
  - whisper.cpp: each progress line /100; final 1.0.
  - sherpa: `(chunk_index+1)/num_chunks`; final 1.0.
  - diarization: child percent /100; progress logs every 10%.
  - extraction: no numeric callback; GUI sets 0.0 before extraction.
- The GUI starts an indeterminate bar for “busy” statuses and stops it when a numeric progress arrives (`src/gui.py:2335-2338`, `2343-2354`).

### 18.3 Log callback

- Type: `Callable[[str], None]`.
- Called from worker threads; GUI marshals each string to the Tk main thread and appends a line (`src/gui.py:1985-1986`, `2406-2410`).
- Strings are appended as-is; no timestamp is added by the GUI. The log file has timestamps; the UI log does not.
- Backend log lines use bracketed tags: `[ffmpeg]`, `[whisper.cpp]`, `[Transcribe]`, `[Progress]`, `[Diarize]`, `[Output]`, `[Done]`, `[Error]`, `[Fatal]`, `[Warning]`, `[Cancelled]`, `[Record]`.
- `logger.info` calls often duplicate the same message to the file; the callback is for the UI only.

### 18.4 Speaker naming callback

Already specified in §11.9. GUI adapter details:

- Called on the worker thread; it creates `done_event`, schedules `_show_naming_panel` on the UI thread with `after(0, show)`, then waits:
  ```python
  while not done_event.wait(0.2):
      if self._cancel_event.is_set():
          self.after(0, self._close_naming_panel)
          raise CancelledError("Cancelled while naming speakers")
  ```
  (`src/gui.py:1818-1837`)
- If showing the panel raises, it logs and sets `done_event` so the worker never hangs (`src/gui.py:1823-1829`).
- Panel:
  - Title `"We found N speaker(s)"` (singular/plural logic).
  - Helper text exact (`src/gui.py:353-358`).
  - Each speaker row: default label chip, “Play sample” or disabled “No sample”, name entry, share label `"{round(share*100)}% of the talking"`.
  - More than 4 speakers → scrollable frame with height 4*62.
  - Buttons: `"Skip names"` (subtle) and `"Save transcript with names"` (amber).
  - Enter in any entry confirms.
  - Confirm applies each nonempty `entry.get().strip()` to the map; empty leaves default.
  - Playback uses `winsound.PlaySound(path, SND_FILENAME | SND_ASYNC)` on Windows; no-op elsewhere.
  - On finish, stops playback and calls `on_done` (which closes the panel and sets `done_event`).
- Cancellation closes the panel and raises `CancelledError("Cancelled while naming speakers")`; `speaker_transcript` propagates; clips are deleted.

### 18.5 Batch behavior

`_transcription_worker` (`src/gui.py:1487-1585`):

- For each file in order:
  - If cancel set before the file: break.
  - Set batch label (`File i of N: name` if N>1, else shortened name).
  - Mark file `"Working"`.
  - Log separator `\n` + 50 `─` (U+2500) + `\n`; log `[File i/N] name`.
  - `temp_wav = None`.
  - `try`:
    - status `"Extracting Audio"`, progress 0.0.
    - `temp_wav = extract_audio(input_file, cancel_event, log_callback)`.
    - If cancel set after extraction: raise `CancelledError("Cancelled")`.
    - `output_path = _resolve_output_path(input_file)`; `_last_output_folder = output_path.parent`.
    - `segments = engine.transcribe(...)` with options timestamps/diarize and speaker naming callback if diarize.
    - Append output to saved; if `_batch_recording` set, store `_last_segments = segments`.
    - Mark file `"Done"`.
  - `except CancelledError`: log `[Cancelled] Operation stopped by user`; mark `"Stopped"`; break.
  - `except (FFmpegNotFoundError, ModelNotFoundError)`: log fatal; mark `"Failed"`; set `fatal`; append failed; **break entire batch**.
  - `except (FFmpegExtractionError, TranscriptionError)`: log error; mark `"Failed"`; append failed; **continue** next file.
  - `except Exception`: log unexpected; mark `"Failed"`; append failed; continue.
  - `finally`: delete `temp_wav` if it exists; ignore `OSError`.
- Final status:
  - If `cancelled`: `"Cancelled"`.
  - Else if no failed: `"Done"`.
  - Else if `len(failed) == total` or `fatal`: `"Failed"`.
  - Else `"Done"` and log `[Summary] Completed with {n} error(s) out of {total} file(s)`.
- Then schedule `_finish_batch` on UI thread.

`_finish_batch` (`src/gui.py:1587-1649`):

- Set UI idle; `_last_transcript = saved[0] if len(saved)==1 else None`.
- If saved, `_last_output_folder = saved[-1].parent`.
- UI titles/messages:
  - Cancelled: “Stopped”; detail “N transcript(s) saved before you stopped.” or “No transcript was saved.”
  - Fatal: “EasyScribe could not continue”; detail “A program file is missing. Please download EasyScribe again.”
  - No failed: “Transcript ready” or “N transcripts ready”; detail “Saved in …”.
  - Some failed: “Transcription failed” or “N of M transcripts ready”; message “Could not transcribe ... The file may be damaged or have no sound. Select "Show details" to see why.”
- Progress bar 1.0 if saved else 0.0.
- Retry shown for failed or cancelled (except fatal).
- For a recording batch (`_batch_recording` not None), “full”/best-quality button shown if retry and not fatal and recording exists; if status Done and saved, `_offer_speakers()`.

### 18.6 “Find speakers” workflow

- Offered only if `_diarization_available` and there is a recording, timed segments, and a transcript (`src/gui.py:1876-1893`).
- Estimate uses `wav_duration(audio)`, `speech_seconds(segments)` clamped to audio, `estimate_seconds(audio_sec, speech_sec)`, `fmt_estimate`; first character uppercased for display.
- On click:
  - output path as §7.8.3.
  - Clears cancel event, clears log, sets UI running, shows activity, batch label audio name.
  - Starts `_speakers_worker` with `add_speakers_to_file`.
- `_speakers_worker`:
  - Status Done/Failed/Cancelled; on `CancelledError` log `[Cancelled] Operation stopped by user`; on other exceptions log error and status Failed with message.
  - Measures elapsed; updates status; schedules finish.
- `_finish_speakers`:
  - Done: sets `_last_transcript` to new path, output folder, title “Speakers added”, detail name + `· took {_fmt_took(took)}`, message “The first transcript without names is kept too.”, progress 1.0, shows results.
  - Cancelled: title “Stopped”; detail “No names were added. Your transcript is unchanged.”; keeps original `_last_transcript`.
  - Failed: title “Could not find the speakers”; detail “Your transcript is unchanged.”; error message.
  - Re-offers speakers card on non-done.

### 18.7 Recording workflow

`_recording_worker` (`src/gui.py:1655-1747`):

1. `transcript_lines = []`, `outcome = {"transcript": None, "audio": None, "error": None}`.
2. `on_segment(text)`: append to list, log text, schedule live box append.
3. `on_overflow()`: log warning (exact text in §14.7).
4. If live: `recognizer = engine.get_recognizer(status_callback)` **before** recording starts; this may load the model (status “Loading Model”) and blocks the start.
5. `dest_dir = _recordings_folder()` (chosen or default).
6. `RECORDING_WORK_DIR.mkdir`.
7. `mic_recorder.start(device_index, RECORDING_WORK_DIR, feed_vad=live)`.
8. status `"Recording"`; schedule timer/level; `mic_queue`, `session_stem`.
9. If live: `live_transcriber.start(recognizer, mic_queue, stop_event, on_segment, on_overflow)`.
10. Log `"[Record] Listening. Select Stop recording when done"`.
11. Wait `_stop_recording_event`.
12. If live: status `"Finishing"`; `live_transcriber.stop()`.
13. Else status `"Saving Recording"`.
14. `wav_path = mic_recorder.stop(convert_to_wav=True, dest_dir=dest_dir)`.
15. If `wav_path` and its parent != dest_dir: log `"[Warning] Could not save to {dest_dir}. The recording is in {wav_path.parent}"`.
16. `saved_dir = wav_path.parent if wav_path else RECORDING_WORK_DIR`.
17. If `transcript_lines` and `session_stem`:
    - `txt_path = saved_dir / f"{session_stem}.txt"`.
    - `write_text("\n".join(transcript_lines) + "\n", encoding="utf-8")`.
    - On `OSError`: `outcome["error"] = f"Could not save the transcript: {exc}"`; log.
    - On success: `outcome["transcript"] = txt_path`; log `[Done] Transcript saved: {txt_path}`.
18. If `wav_path`: `outcome["audio"] = wav_path`; log `[Done] Audio saved: {wav_path}`; `_last_output_folder = saved_dir`.
19. status `"Done"`.
20. `except ModelNotFoundError`: log fatal; outcome error `"A program file is missing. Please download EasyScribe again.\n{exc}"`; status `"Failed"`.
21. `except Exception`: log; outcome error:
    ```
    Could not record: {exc}
    Check that a microphone is connected, and that Windows allows apps to use it (Settings, Privacy, Microphone).
    ```
    status `"Failed"`.
22. `finally`: schedule `_finish_recording`.

`_finish_recording` (`src/gui.py:1749-1790`):

- Stop timer, reset button, idle, stop progress, clear percent.
- Set `_last_transcript`/`_last_recording` from outcome.
- If error and no audio: title “Could not record”, message error, bar 0; if no live text reset live box; return.
- If `not live` and audio exists: immediately `_on_full_transcript()` (best-quality pipeline); return. This means the “Recording saved” screen is skipped for record-only; the file pipeline takes over.
- Else (live):
  - Title “Recording saved”.
  - Detail:
    - record-only (unreachable here because handled above): `"The audio is in {where}."`
    - live with transcript: `"The audio and the live transcript are in {where}."`
    - live without transcript: `"No speech was heard live, so there is no live transcript. The audio is in {where}."`
  - Show error if any, bar 1.0, results (transcript button if transcript; full/best-quality button because audio exists).

### 18.8 Thread marshalling requirement

Every backend callback is invoked on a worker thread. The UI must marshal:

- `status_callback` → `after(0, ...)`
- `progress_callback` → `after(0, ...)`
- `log_callback` → `after(0, ...)`
- `on_segment` → `after(0, ...)`
- speaker naming → `after(0, show panel)` + blocking event.

Rust: use a UI event channel (`Sender<UiEvent>`) polled on the main thread; never touch UI state from a worker. The speaker naming callback needs a oneshot channel back to the worker and must poll cancellation while waiting.

---

## 19. Error taxonomy, cancellation, and cleanup

### 19.1 Exception classes

| Exception | Base(s) | Module | Typical trigger |
|---|---|---|---|
| `CancelledError` | `RuntimeError` | `common` | user cancel at any stage |
| `FFmpegNotFoundError` | `RuntimeError` | `ffmpeg_wrapper` | ffmpeg/ffprobe missing |
| `FFmpegExtractionError` | `RuntimeError` | `ffmpeg_wrapper` | ffmpeg nonzero/unexpected |
| `WhisperCppNotFoundError` | `RuntimeError` | `whispercpp_wrapper` | cli/model missing |
| `WhisperCppError` | `RuntimeError` | `whispercpp_wrapper` | cli nonzero/missing JSON/unexpected |
| `ModelNotFoundError` | `RuntimeError` | `transcriber` | sherpa model files missing/load failure |
| `TranscriptionError` | `RuntimeError` | `transcriber` | audio load/write/whisper failure |
| `DiarizationError` | `RuntimeError` | `diarizer` | models missing/child failure |
| `NetworkBlockedError` | `ConnectionRefusedError` (`OSError`) | `offline_guard` | blocked socket/DNS |

Built-in exceptions also matter:

- `FileNotFoundError`, `PermissionError`, `OSError` from file operations.
- `FileExistsError` from `rename_all` (target taken).
- `ValueError`/`json.JSONDecodeError` (subclass) from parsing.
- `wave.Error` from malformed WAV.
- `subprocess.TimeoutExpired` (caught in probe/GPU probe; not propagated).
- `ImportError` if sherpa-onnx is missing (not wrapped in transcription).

### 19.2 Cancellation matrix

| Stage | Check granularity | Action on cancel | Exception message | Cleanup |
|---|---|---|---|---|
| `extract_audio` pre-check | once | raise | `Cancelled before audio extraction started` | none |
| ffmpeg run | 100 ms poll | SIGTERM/terminate, wait 5 s, kill; join reader 2 s | `Audio extraction cancelled by user` | temp WAV unlinked |
| `whisper_transcribe` pre-check | once | raise | `Cancelled before transcription started` | none |
| whisper-cli run | 100 ms poll | SIGTERM/terminate, wait 5 s, kill; join reader 2 s | `Transcription cancelled by user` | temp JSON unlinked in `finally` |
| `vad.file_segments` | per 512-sample window | raise | `VAD segmentation cancelled by user` | none |
| sherpa chunk loop | before each chunk | raise | `Transcription cancelled by user` | samples freed by drop; no output written |
| `DiarizationEngine.diarize` | 200 ms poll | terminate child, join 5 s, kill, join 5 s | `Speaker identification cancelled by user` | pipes/process closed |
| speaker naming callback | 200 ms poll in GUI | close panel; raise | `Cancelled while naming speakers` | clips unlinked in `finally` |
| live transcriber | 100 ms queue poll + drain/flush | no exception; graceful stop | — | threads joined up to 60 s |

Additional rules:

- Pre-cancel checks run before validation; a pre-cancelled call may not raise `FFmpegNotFoundError`/`WhisperCppNotFoundError`.
- Cancellation does not write a transcript; the output file is untouched/no file.
- A cancelled diarization inside `transcribe(diarize=True)` propagates and does **not** write a plain fallback (only non-cancel exceptions fall back).
- A cancelled `add_speakers_to_file` leaves the original transcript untouched; the output `(speakers).txt` is not written (unless cancellation happens during writing, which is not checked).
- The GUI’s cancel button sets `_cancel_event` and status `"Cancelling…"`; worker catches `CancelledError` and marks the file `"Stopped"`/final `"Cancelled"`.

### 19.3 Cleanup matrix

| Situation | Temp WAV | Temp JSON | Speaker clips | Raw PCM/JSON | Output transcript |
|---|---|---|---|---|---|
| ffmpeg success | caller deletes after job | — | — | — | — |
| ffmpeg failure/cancel | wrapper deletes | — | — | — | — |
| whisper success | caller deletes after job | wrapper deletes | — | — | written |
| whisper failure/cancel | caller deletes after job | wrapper deletes | — | — | not written |
| diarization failure (non-cancel) | caller deletes after job | — | callback finally deletes | — | plain transcript written |
| naming cancel | caller deletes after job | — | finally deletes | — | not written |
| write failure | caller deletes | — | already deleted | — | partial/none; error |
| recording stop success | — | — | — | deleted | live txt if any |
| recording stop failure | — | — | — | kept for recovery | live txt may still write if WAV failed? `saved_dir` work dir; yes if transcript_lines |
| crash | may remain in temp | may remain | may remain | remains in recordings | may be partial |

### 19.4 Error message formatting

- Messages often include platform paths and exception strings; preserve `Path` display (native separators, no quoting).
- `\n\n` separates context from the underlying error in several messages (`FFmpegExtractionError`, `TranscriptionError` write, `WhisperCppError`).
- Snippets from stderr are the **last 800 characters**, not lines.
- No exception messages are localized; English only.
- The GUI displays exceptions via `str(exc)` in error boxes and logs; Rust’s `Display` must match for dialog parity.

---

## 20. Platform-specific behavior and cross-platform plan

### 20.1 Reference platform

The production target is Windows 10/11 x64 (PyInstaller `--windowed`). The Python code contains explicit `sys.platform == "win32"` branches. Linux/macOS are used for development/tests but are not the shipping target.

### 20.2 Subprocess creation and binary names

| Concern | Windows reference | Non-Windows reference | Rust requirement |
|---|---|---|---|
| Hide console | `creationflags=0x08000000` (`CREATE_NO_WINDOW`) | `0` | Windows: `std::os::windows::process::CommandExt::creation_flags(0x08000000)`; Unix: no flag |
| ffmpeg | `ffmpeg/ffmpeg.exe` | still `ffmpeg/ffmpeg.exe` (would need a build change) | Resolve `ffmpeg` on Unix/macOS, `ffmpeg.exe` on Windows; keep bundled layout `<base>/ffmpeg/` |
| ffprobe | `ffmpeg/ffprobe.exe` | still `.exe` | Same mapping |
| whisper-cli | `whispercpp/whisper-cli.exe` | `whispercpp/whisper-cli` (`config.py:113`) | Keep the existing platform branch; ensure executable bit on Unix |
| Python spawn child | `multiprocessing` re-execs `EasyScribe.exe`; `freeze_support()` | same (spawn) | Hidden `--diarize-worker` arg; detect before UI |
| Shell | none; `subprocess.Popen(list)` | none | `Command::arg`, never `sh -c`/`cmd /c` |

`CREATE_NO_WINDOW` is 0x08000000; `DETACHED_PROCESS` is 0x00000008; do not confuse them.

### 20.3 Documents folder

- Reference uses `SHGetKnownFolderPath(FOLDERID_Documents)` on Windows, fallback `~/Documents` or `~` (`src/config.py:37-60`).
- Rust: use the Windows known-folder API (not `%USERPROFILE%\Documents` blindly, because OneDrive can relocate it). On macOS, `~/Documents`; on Linux, XDG `xdg-user-dir DOCUMENTS` if available, else `~/Documents` if it exists, else `~`.
- The Rust app must not create the Documents/EasyScribe folder until a recording is saved or recovery needs it.

### 20.4 CPU name

- Windows: registry `HKEY_LOCAL_MACHINE\HARDWARE\DESCRIPTION\System\CentralProcessor\0`, `ProcessorNameString` (`src/whispercpp_wrapper.py:73-78`).
- Linux: `/proc/cpuinfo`, first `model name` (`src/whispercpp_wrapper.py:79-81`).
- Fallback: `platform.processor()` or `"unknown CPU"`.
- Rust: `winreg` crate / `windows` registry API; Linux `/proc/cpuinfo`; macOS `sysctl -n machdep.cpu.brand_string`; fallback `"unknown CPU"`.
- Do not use WMI (slow, may require COM initialization) unless cached.

### 20.5 Taskbar identity

- `SetCurrentProcessExplicitAppUserModelID("EasyScribe.App")` before the first window (`src/main.py:50-64`).
- Rust: `windows` crate `SetCurrentProcessExplicitAppUserModelID` (shell32), called before creating the main window; ignore errors.

### 20.6 Opening files/folders and playing clips

- `_open_path` (`src/gui.py:205-216`):
  - Windows `os.startfile(path)`.
  - macOS `subprocess.Popen(["open", path])`.
  - Linux `subprocess.Popen(["xdg-open", path])`.
  - Errors: log warning; show message box `"Could not open"` / `"Could not open:\n{path}\n\n{exc}"`.
- Clip playback (`src/gui.py:415-431`):
  - Windows `winsound.PlaySound(path, SND_FILENAME | SND_ASYNC)`; stop with `PlaySound(None, 0)`.
  - Non-Windows reference is a no-op (the button still exists but playback does nothing).
- Rust: use `open`/`xdg-open`/`start` via `Command`; for audio, use a small decoder (rodio/symphonia) or platform APIs. On non-Windows, decide whether to implement playback (improvement) or keep no-op for parity. Document.

### 20.7 Audio capture

- Reference uses `sounddevice` (PortAudio) with `samplerate=16000`, `channels=1`, `dtype=int16`, `blocksize=512`, `device` = global index or `None`.
- Windows host APIs (MME, DirectSound, WASAPI, WDM-KS) expose duplicates; GUI dedupes by name.
- Rust with `cpal`:
  - Enumerate input devices; build labels; dedupe identically.
  - Prefer WASAPI shared mode on Windows; request the device’s default config and resample to 16 kHz mono, or request 16 kHz if supported.
  - Convert all sample formats (i16, u16, f32, etc.) to i16 with the same full-scale mapping as PortAudio; clipping at i16 bounds.
  - Block size 512 at 16 kHz = 32 ms. If the backend’s buffer size differs, accumulate/resample into 512-sample chunks before the VAD/PCM pipeline so downstream behavior is unchanged.
  - Device disconnect/stream error: propagate to the recording worker’s error path with the same user-facing message.
- Exact mic gain/level: peak = max(abs(i16)) / 32768; do not apply AGC or normalization.

### 20.8 Multiprocessing / child process

- Python uses `spawn`; on Windows, the child re-executes the frozen exe and `freeze_support()` dispatches to the target. On POSIX, spawn also starts a fresh interpreter and imports the module.
- Rust: implement a hidden subcommand, e.g. `EasyScribe.exe --internal-diarize-worker <args>`; parse it at the top of `main` before any UI/logging? Logging can be initialized in the child to a separate file or stderr. The child must not create a window.
- Parent/child IPC: pipes; JSON lines or a binary framing. Do not use TCP.
- Termination: Windows `TerminateProcess`; Unix SIGTERM then SIGKILL after 5 s. The Rust standard `Child::kill()` sends SIGKILL on Unix; to match, use `libc::kill(pid, SIGTERM)` first.
- `daemon` semantics: if the parent exits, the child should be terminated. Rust can use a job object on Windows (`AssignProcessToJobObject` with kill-on-close) or a watchdog pipe; on Unix, `prctl(PR_SET_PDEATHSIG)` or a watchdog. Document.

### 20.9 Paths, case, and filenames

- Windows paths are case-insensitive; Python `WindowsPath` equality/ordering is case-insensitive. POSIX paths are case-sensitive.
- `rename_all` must treat case-only renames as allowed on Windows. On POSIX, a differently-cased existing file is a conflict.
- `recording_set` sorted order may differ under case-insensitive ordering; use a platform-aware sort key.
- `Path.with_suffix`, `Path.stem`, and `Path.name` semantics:
  - `foo.tar.gz` stem `foo.tar`, suffix `.gz`.
  - `.hidden` stem `.hidden`, suffix empty.
  - `recording_...pcm` with_suffix(`.json`) → `.json`; with_suffix(`""`) removes `.json`.
  - Rust `Path::file_stem`, `extension`, `with_extension` mostly match; be careful with trailing dots and multi-dot names. Add unit tests for the exact names used.
- Long paths (>260 chars): Python on Windows may use long-path support depending on manifest/registry; Rust should use `\\?\`-prefixed paths or enable long-path awareness. The reference may fail; document.
- Invalid filename characters: `check_name` covers the common set; Windows also rejects trailing spaces/dots and reserved names with extensions. See §16.2.
- Unicode normalization: Windows NTFS preserves the input form; Python `str` does not normalize. Rust should not normalize; use `OsString`/wide APIs.

### 20.10 Newlines and text encoding

- Python `open(..., encoding="utf-8", newline="\n")` writes LF on all platforms for transcripts.
- Logs via `FileHandler` use default newline translation → CRLF on Windows. Decide whether to match CRLF in logs; transcripts must be LF.
- Settings/sidecar JSON via `write_text` uses default newline translation? `Path.write_text` opens with `newline=None`, which translates `\n` to `os.linesep` on Windows. So `settings.json` and the PCM sidecar may contain CRLF on Windows. The JSON parser accepts either. For byte parity, use CRLF on Windows for these files; for cross-platform consistency, LF. Document the choice; parsed content is identical.
- WAV/PCM are binary; no translation.
- UTF-8 BOM is never written.

### 20.11 Time and formatting

- `time.monotonic()` → Rust `Instant` for elapsed/progress throttling; never wall clock.
- `datetime.now().strftime("%Y%m%d_%H%M%S")` → local time for filenames and logs. Rust use `chrono::Local` with the same format; beware DST/clock changes. Filenames can repeat/duplicate around DST fallback; Python does too.
- `%.0f`, `%.1f`, `%02d`, thousands separators:
  - Python’s float formatting uses round-half-even; Rust’s `format!` generally uses round-half-even for decimal formatting, but verify edge cases with tests (e.g. `0.5`, `1.5`, `2.5` at `.0f`).
  - Thousands separator `:,` in `"{size_kb:,} KB"`; Rust’s formatting has no built-in comma grouping, so use a helper (a small manual function or a `thousands`-style crate). Match a comma every 3 digits, no locale awareness.
- `round()` in Python is banker’s; use `round_ties_even` in Rust for `fmt_estimate`, percent labels, and speed estimates. For `round(factor, 3)`, use a decimal-aware half-even implementation.

### 20.12 Windows-specific GUI/backend hooks outside this spec

These are GUI-only but affect startup/behavior; the Rust UI must account for them:

- `win_paint.install(self, C.BG)` before widgets to avoid black flash (`src/gui.py:561-563`).
- `transitions.Veil` screen fade; disabled outside Windows (`src/transitions.py`; test `tests/test_transitions.py:143-145`).
- `_flash_taskbar` uses `FlashWindow` on Windows (`src/gui.py:1863-1871`).
- `_ui_family` uses “Segoe UI” on Windows, else default; `_mono_family` “Consolas” vs “DejaVu Sans Mono” (`src/gui.py:173-181`).
- Drag-and-drop via `tkinterdnd2`; optional.
- Window sizing based on screen height and DPI (`src/gui.py:546-558`).

These do not change backend outputs; replicate the behavior only if the Rust UI aims for visual parity.

### 20.13 Cross-platform acceptance criteria

- On Windows: all reference behavior, paths, subprocess flags, dialogs, device enumeration, and file names match.
- On Linux/macOS: the app builds and runs; ffmpeg/ffprobe/whisper-cli names resolve; Documents folder mapping works; mic capture produces 16 kHz mono; transcripts/WAVs are byte-compatible; no network access.
- Tests must run on all three where feasible; model-dependent tests can be gated on model presence.

---

## 21. Rust implementation checklist

### 21.1 Suggested module layout

Aligned with `REWRITE_PLAN.md` §Architecture:

```
crates/easyscribe/
  src/main.rs             # startup order, payload bootstrap, headless worker mode
  src/app.rs              # state machine + worker events
  src/workers/            # file batch, record, live, diarize
  src/ui/                 # Rune screens, theme, dialogs, event loop

crates/easyscribe-core/   # no UI; all domain logic, testable
  src/config.rs            # constants, paths, resource resolution
  src/logging.rs           # rotating file logger, exact format
  src/offline.rs           # env vars, network policy, blocked error
  src/common.rs            # CancelledError, fmt_duration, WAV I/O
  src/settings.rs          # settings.json
  src/ffmpeg.rs            # probe, extract, stderr parser
  src/vad.rs               # constants, detector wrapper, bounds, padding
  src/whispercpp.rs        # engine wrapper, device parsing, JSON
  src/transcriber.rs       # engine, transcript builders, speaker pipeline
  src/diarizer.rs          # time model, child process protocol, assignment
  src/mic.rs               # capture, PCM writer, level
  src/live.rs              # live VAD/decode threads
  src/recovery.rs          # orphan scan/conversion/dialogs
  src/rename.rs            # name validation/set/rename
  src/payload.rs           # single-exe payload extraction/locating (plan-specific)

crates/easyscribe-pack/    # build tool: append compressed payload
```

If the primary file engine is implemented with `whisper-rs` instead of invoking the bundled `whisper-cli`, it must still reproduce the reference’s externally visible behavior: beam size 5, the same timestamp/segment semantics, the `-dev`/device selection rules, and the `Engine:`/`GPU:`/`CPU:` log lines derived from whisper.cpp’s stderr (or a verified equivalent). See §10.12 and §21.5.

### 21.2 Required public API surface

At minimum, expose equivalents of:

- `fmt_duration`, `wav_duration`, `load_wav_float32`
- `validate_ffmpeg`, `probe_audio`, `extract_audio`
- `validate_whispercpp`, `is_available`, `transcribe_file`
- `validate_model_directory`, `TranscriptionEngine::{new, get_recognizer, transcribe}`
- `build_vad`, `file_segments`, `pad_and_clamp`, `pad_live_segment`
- `DiarizationEngine::{is_available, diarize, assign_speakers}`, `speech_seconds`, `estimate_seconds`, `save_measured_speed`, `fmt_estimate`
- `MicRecorder::{list_devices, start, stop, get_queue, get_level, get_session_stem}`
- `LiveTranscriber::{start, stop}`
- `scan_for_orphans`, `recover_pcm_to_wav`, `_handle_orphan` (internal), `prompt_and_recover`
- `check_name`, `recording_set`, `rename_all`
- settings load/save functions
- setup/rotate logging
- offline install/check

### 21.3 Invariants to encode as tests

1. `PAD_SAMPLES == 3200`; `VAD_PAD_SEC < VAD_MIN_SILENCE_SEC / 2`; `_HISTORY_SAMPLES == 11392`; `_VAD_QUEUE_MAXSIZE == 1875`.
2. `NUM_THREADS` formula for CPU counts 1,2,4,6,8,10,12,16,32.
3. `fmt_duration` table (0, 59.9, 60, 65, 3599, 3600, 3723.5).
4. `fmt_estimate` table incl. 90/150/210/270/3599/5400.
5. Transcript builder golden strings from `tests/test_transcript_builders.py`.
6. `pad_and_clamp` well-formedness for empty/single/adjacent/far/apart/end cases.
7. `pad_live_segment` four clamp cases.
8. `_parse_output_json` blank-skip and ms→s.
9. `_describe_devices` three cases and `_pick_gpu` four cases.
10. `_iter_lines` CR/LF and split-read behavior; `_parse_duration`/`_parse_time`.
11. FFmpeg argv exact (including whitelist position/count).
12. whisper argv exact with/without `-dev`.
13. Settings load/save edge cases (missing, broken, missing folder, None).
14. Recovery truncation and decline/accept invariants.
15. Rename set/rename/rollback/taken.
16. Offline: no banned network imports/crates; argv whitelist.
17. Mic stop destination/fallback and WAV frame count.
18. Diarization assignment max overlap/default.
19. Child protocol progress/done/error and cancel termination.
20. Live stop drains queued audio and flushes.

### 21.4 Golden-file tests

- WAV from PCM: fixed PCM bytes + metadata → compare full file bytes (header canonical) or frame count/format.
- Transcripts: exact UTF-8 bytes, including trailing LF and block separators.
- Settings: exact JSON bytes (indent, key order, escaping) or parsed equality with a documented formatting deviation.
- Logs: regex line format; rotation count.
- Temp cleanup: create files of each pattern and assert which remain.

### 21.5 Compatibility deviations log

Create `docs/rewrite/COMPATIBILITY.md` (or a section in this file) recording every intentional difference. Candidates from this document:

- Same-second recording filename collision (fix with unique suffix vs reproduce corruption).
- 8/24-bit/float WAV handling (reject vs reproduce wrong samples).
- 32-bit float WAV.
- `check_name` reserved names with extensions (`CON.txt`) and trailing spaces.
- Non-atomic settings/sidecar/transcript writes (fix vs reproduce partial files).
- `rename_all` rollback error masking.
- `_cpu_name` / `_describe_devices` malformed-line behavior.
- Live decoder swallowed errors.
- Mic writer join timeout losing tail.
- Python `round` vs Rust formatting differences.
- Non-Windows clip playback (no-op vs implement).
- Log/settings CRLF vs LF.
- `settings` NaN/Infinity/bool quirks.
- `_shorten` edge cases for limit ≤1.
- `DiarizationEngine` child spawn details vs alternative in-process implementation.

### 21.6 Definition of done

- All pure-function tests ported and passing.
- Golden transcript/WAV/JSON tests byte-identical (or approved deviations).
- End-to-end file transcription on a bundled model produces the same transcript text/format as Python on the same input/engine.
- Diarization produces the same turn assignment for the same audio/model (speaker labels may depend on clustering; compare assignment shape and labels).
- Live recording produces a valid 16 kHz mono WAV and a transcript with one segment per line.
- Cancel at each stage leaves no stray output and the expected exception category/message.
- Recovery prompt recovers a killed recording to the chosen/default folder and keeps files on decline.
- Offline checks pass; no network capability in the dependency tree.
- Windows and at least one Unix platform build and pass smoke tests.

---

## 22. Test matrix

| Test file | What it locks down | Backend sections |
|---|---|---|
| `tests/test_transcript_builders.py` | plain/diarized formatting, block gaps, speaker fallback, max-overlap assignment | §11.4–§11.6, §12.6 |
| `tests/test_vad_bounds.py` | pad invariant, pad_and_clamp well-formedness | §9.1, §9.4 |
| `tests/test_live_transcriber.py` | live padding clamps, decode queue drop-oldest, stop flush/drain | §9.5, §14.4–§14.6 |
| `tests/test_ffmpeg_wrapper.py` | CR/LF line splitting, chunked reads, duration/time parsing | §8.3–§8.5 |
| `tests/test_whispercpp_wrapper.py` | JSON parse, device lines, GPU rank, progress regex, availability, pre-cancel | §10.3–§10.10 |
| `tests/test_offline_guard.py` | socket/DNS blocking, loopback, ffmpeg whitelist source, no network imports | §5.4–§5.7, §8.6 |
| `tests/test_recovery.py` | PCM truncation, decline keeps files, accept deletes pair | §15.2–§15.3 |
| `tests/test_recordings_folder.py` | settings behavior, MicRecorder dest/fallback, recovery dest | §6, §13.6, §15.3 |
| `tests/test_rename.py` | name validation, recording set, rename/rollback/taken | §16 |
| `tests/test_diarization.py` | end-to-end VAD+Whisper+diarization, `[Speaker` present | §11.13, §12 |
| `tests/test_diarizer_process.py` | child progress monotonic/1.0, fast cancel | §12.8–§12.10 |
| `tests/test_version.py` | version consistency (launcher/workflow/assets/specs) | §2.1 (adapt to Cargo/version files) |
| `tests/test_launcher_safety.py` | launcher/install safety (not backend; keep separate) | — |
| `tests/test_transitions.py` | Windows screen fade (not backend) | — |

`tests/run_all.py` runs every `test_*.py` except `test_diarization.py` and `test_diarizer_process.py`, which need real models/audio (`tests/run_all.py:19`, `22-35`). Rust CI should have a fast suite (no models) and an optional model suite.

Test harness conventions:

- Python tests insert `src/` into `sys.path` and use a local `_check(name, fn)` that catches exceptions, prints `OK`/`FAIL`, and exits 1 if any failure.
- Rust can use `#[test]`/`cargo test`, but the assertions and golden strings must remain identical.
- Tests monkey-patch module globals (e.g. `settings.SETTINGS_FILE`, `whispercpp_wrapper.WHISPERCPP_BIN`, `vad.build_vad`). Rust needs injectable paths/factories or test-only setters to preserve testability.

---

## Appendix A. Exact format strings and magic values

### A.1 Durations

```
fmt_duration:          "{h}h {m:02d}m {s:02d}s"  if h else "{m}m {s:02d}s"
_format_hms:           "{h:02d}:{m:02d}:{s:02d}"
recovery duration:     "{int(sec//60)}m {int(sec%60)}s"
```

### A.2 FFmpeg logs

```
[ffmpeg] Already mono 16 kHz PCM WAV — copying directly: {name}
[ffmpeg] Input: codec={codec}, sample_rate={rate} Hz, channels={chans}, duration={dur}
[ffmpeg] Could not probe audio stream — proceeding anyway
[ffmpeg] Extracting audio → mono 16 kHz PCM WAV: {name}
[ffmpeg] Temp file: {temp.name}
[ffmpeg] Media duration: {dur}
[ffmpeg] Extracting… {pos} / {total}  ({pct:.0f}%)
[ffmpeg] {line}
[ffmpeg] Extraction complete — {size_kb:,} KB written to {name}
```

### A.3 whisper.cpp logs

```
[whisper.cpp] Transcribing with beam_size={n}: {name}
[whisper.cpp] Engine: Vulkan (GPU Vulkan{idx})
[whisper.cpp] GPU: {name or "name not reported"}[ | {details}]
[whisper.cpp] Engine: {using}
[whisper.cpp] Engine: CPU (no GPU found)
[whisper.cpp] Other Vulkan device {idx} (not used): {name} | {details}
[whisper.cpp] CPU: {cpu_name}[ | {n} of {total} threads][ | {features}]
[whisper.cpp] Decoded {n} segment(s)
```

### A.4 Transcription logs

```
[Transcribe] Starting: {name}
[Transcribe] Audio duration: {dur}
[Transcribe] engine: whisper.cpp (beam_size=5), timestamps: {on|off}, speakers: {on|off}
[Transcribe] engine: sherpa-onnx (CPU), timestamps: {on|off}, speakers: {on|off}
[Transcribe] Detecting speech segments (VAD)…
[Transcribe] Processing {n} speech segment(s)…
[Progress] segment {i}/{n} — {seg_end} / {duration} — {speed:.1f}x realtime
[Transcribe] Complete — {segments} segments, {words} words in {dur}
[Output] Writing transcript to: {path}
[Done] Saved: {path}  ({kb} KB)
```

### A.5 Diarization logs

```
[Diarize] Estimated wait: {est} for {dur} of audio (rough estimate). Please wait…
[Diarize] Finding the parts with speech…
[Diarize] Progress: {pct}%
[Diarize] Found {n} speaker(s) across {m} turn(s)
[Diarize] Warning: {exc}. Writing the transcript without speaker names.
```

### A.6 Recording logs

```
MicRecorder started: {name}
Recording saved: {path}
[Record] Listening. Select Stop recording when done
[Record] Starting live recording…
[Record] Starting recording (best quality transcript after the stop)…
[Done] Audio saved: {path}
[Done] Transcript saved: {path}
[Warning] Could not save to {dest}. The recording is in {parent}
[Warning] transcription falling behind. Some audio was skipped in the live view; the full recording is still saved
```

### A.7 Errors

```
ffmpeg.exe not found at: {path}
ffprobe.exe not found at: {path}
ffmpeg exited with code {rc}.\n\n{snippet}
Unexpected error during audio extraction: {exc}
whisper-cli not found at: {path}
Whisper GGUF model not found at: {path}
whisper-cli exited with code {rc}.\n\n{snippet}
whisper-cli completed but did not produce {name}
Unexpected error during whisper.cpp transcription: {exc}
Model validation failed:\n{errors}
Could not load model: {exc}
Could not load audio: {exc}
The transcript has no text to add speakers to.
Cannot write to output file.\nThe file may be open in another program: {path}\n\n{exc}
Failed to write transcript: {path}\n\n{exc}
The speaker models are missing.
Could not load audio for diarization: {exc}
Speaker identification stopped unexpectedly (exit code {code}).
Speaker diarization failed: {value}
Speaker identification cancelled by user
Network access blocked (EasyScribe runs offline): {what}
```

### A.8 Special characters

- Em dash `—` (U+2014) in many log messages.
- Ellipsis `…` (U+2026) in `[ffmpeg] Extracting…`, `[Transcribe] Detecting…`, `[Diarize] … Please wait…`.
- Arrow `→` (U+2192) in `Transcribing: {a} → {b}`, `[ffmpeg] Extracting audio → …`, `Recovered {a} → {b}`.
- Bullet `•` (U+2022) in dependency error.
- Box-drawing `─` (U+2500) ×50 in batch log separators.
- Middle ellipsis `…` (U+2026) in `_shorten`.
- Arrow `→` (U+2192) in `"PCM→WAV conversion to {folder} failed"`.
- `·` (U+00B7) in `_finish_speakers` detail: `"{name} · took {took}"`.

---

## Appendix B. Quick line-reference index

| Topic | Lines |
|---|---|
| App metadata, base dir, documents, paths, constants | `src/config.py:14-144` |
| CancelledError, duration, WAV load | `src/common.py:15-46` |
| Log rotation/setup | `src/logger.py:19-68` |
| Offline env/host/patch/install | `src/offline_guard.py:36-153` |
| Settings keys/load/save | `src/settings.py:17-72` |
| FFmpeg exceptions/parsers/commands/extract | `src/ffmpeg_wrapper.py:28-281` |
| VAD constants/build/segments/pad | `src/vad.py:24-128` |
| whisper regex/device/GPU/JSON/transcribe | `src/whispercpp_wrapper.py:32-291` |
| Transcript builders/speaker clips/write/engine | `src/transcriber.py:29-473` |
| Diarizer constants/time model/engine/protocol | `src/diarizer.py:32-334` |
| Mic constants/devices/start/stop/callback/writer | `src/mic_recorder.py:25-218` |
| Live constants/start/stop/VAD/decode/drop | `src/live_transcriber.py:28-222` |
| Recovery scan/WAV/orphan/prompt | `src/recovery.py:17-145` |
| Rename validation/set/rename | `src/rename.py:13-62` |
| Main cleanup/taskbar/deps/startup | `src/main.py:12-144` |
| GUI status table | `src/gui.py:139-163` |
| GUI batch worker | `src/gui.py:1487-1649` |
| GUI recording worker | `src/gui.py:1655-1790` |
| GUI speaker naming | `src/gui.py:311-445`, `1818-1973` |
| GUI output naming | `src/gui.py:1792-1816`, `1895-1911`, `2156-2160` |
| GUI mic labels | `src/gui.py:2001-2017` |
| GUI progress/status helpers | `src/gui.py:2315-2354` |

---

## Appendix C. Reference test line index

| Test | Lines |
|---|---|
| transcript builders | `tests/test_transcript_builders.py:35-133` |
| VAD bounds | `tests/test_vad_bounds.py:45-113` |
| live transcriber | `tests/test_live_transcriber.py:35-216` |
| ffmpeg parsers | `tests/test_ffmpeg_wrapper.py:51-80` |
| whisper wrapper | `tests/test_whispercpp_wrapper.py:40-203` |
| offline guard | `tests/test_offline_guard.py:45-115` |
| recovery | `tests/test_recovery.py:54-101` |
| recordings folder/settings | `tests/test_recordings_folder.py:63-119` |
| rename | `tests/test_rename.py:46-105` |
| diarization E2E | `tests/test_diarization.py:21-59` |
| diarizer process/progress/cancel | `tests/test_diarizer_process.py:32-61` |
| run all | `tests/run_all.py:19-35` |

---

## Appendix D. Decisions the Rust implementer must make explicitly

These are not ambiguities in the Python source; they are areas where a literal port would preserve a defect or where Rust’s platform model requires a choice. Record each in the compatibility log.

1. **Same-second recording collision** (`src/mic_recorder.py:92-94`, `186`): reproduce append-to-existing-PCM corruption, or add a unique suffix while keeping the `recording_YYYYMMDD_HHMMSS` prefix. Recommended: unique suffix (`_2`, `_3`, …) and a warning; update `recording_set`/rename behavior if the suffix is outside the `" ("` convention (or use `recording_<ts> (2)` to stay inside it).
2. **WAV widths other than 16-bit** (`src/common.py:39-41`): reproduce the int8/scale-128 behavior for 8-bit and int32 for 32-bit, reject 24-bit/float, or implement correct decoding. Recommended: correct decoding plus a documented deviation; production always 16-bit.
3. **Non-atomic writes** (settings, sidecar, transcript): reproduce direct writes or use temp+rename. Recommended: atomic writes; crash behavior improves, success bytes identical.
4. **`check_name` gaps** (`src/rename.py:13-28`): allow `CON.txt`/trailing spaces like Python, or reject them correctly. Recommended: reject Windows-invalid names fully; document the difference.
5. **`rename_all` rollback masking** (`src/rename.py:57-61`): preserve original error vs report rollback failure. Recommended: report both, prioritize original.
6. **Case-only rename** (§20.9): match Python’s Windows case-insensitive equality; implement platform-aware comparison.
7. **Python `round` half-even**: use `round_ties_even`; do not use Rust’s default `round()` for `fmt_estimate`, percentages, or speed rounding.
8. **`_describe_devices` malformed features** (`src/whispercpp_wrapper.py:117`): reproduce IndexError/thread death or parse defensively. Recommended: parse defensively and log a warning.
9. **Live decoder errors** (`src/live_transcriber.py:195-202`): reproduce silent thread death or surface an error. Recommended: surface via an error channel and stop recording cleanly.
10. **Mic writer join timeout** (`src/mic_recorder.py:131-134`): reproduce tail loss or wait/report. Recommended: wait for the writer with a longer timeout and fail the save rather than truncate silently.
11. **Log/settings CRLF** (§20.10): match Windows newline translation or use LF consistently. Recommended: LF for machine-readable JSON/transcripts; CRLF for logs only if byte parity is required.
12. **Non-Windows clip playback**: reference is a no-op; implement playback or keep no-op. Recommended: implement with a small decoder; document.
13. **Diarization child vs in-process**: reference uses a killable child because sherpa ignores cancellation. Keep a child for parity and crash isolation.
14. **GPU probe cache**: reference caches for process lifetime; keep or re-probe per job. Recommended: keep `OnceLock` for parity and startup cost.
15. **`settings` bool/NaN/Infinity quirks**: match Python’s permissive JSON or reject nonstandard values. Recommended: reject NaN/Infinity with a warning; accept bool as 1.0 only if a compatibility test requires it.

---

*End of specification.*
