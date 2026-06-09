# EasyScribe v2.0 — sherpa-onnx rewrite, live transcription, single-exe distribution

## Context

v1.x shipped faster-whisper + ctranslate2 + CUDA DLL stack (nvidia-*-cu12) in a two-ZIP distribution. After five debug builds fighting onnxruntime CUDA DLL collisions on Windows, the decision was made to:
- Replace all inference with **sherpa-onnx** (already used for diarization)
- Use **Vulkan** GPU provider instead of CUDA (no DLL bundling, works on any GPU)
- Ship as a **single .exe** (extracts once to AppData, instant subsequent runs)
- Add **live microphone transcription** (VAD-chunked, no network required)
- Offer **two model variants**: Whisper ONNX (distil-large-v3) and Parakeet CTC 0.6B

This work goes on a new branch: `v2`.

---

## Decisions confirmed by user

| Topic | Decision |
|---|---|
| Inference backend | sherpa-onnx for everything (transcription + diarization + VAD) |
| GPU | Vulkan provider — any GPU, no DLL bundling. CPU fallback. |
| Distribution | Single .exe: 7-zip SFX → tiny launcher → extracts to AppData on first run |
| Audio input | Microphone only |
| Language | English only |
| Variants | Whisper ONNX (distil-large-v3) and Parakeet CTC (0.6B). Same features in both. |
| Default output dir | `%USERPROFILE%\Documents\EasyScribe Recordings\` (created on first use) |
| Diarization in live mode | No — file-based only (live adds too much complexity in v2.0) |

---

## File inventory

### Delete
- `src/cuda_setup.py` — Vulkan/CPU need no PATH/DLL management
- `hooks/hook-ctranslate2.py`
- `hooks/hook-faster_whisper.py`
- `EasyScribe.spec` — replaced by two variant specs

### Create
- `src/mic_recorder.py` — sounddevice capture + crash-safe PCM writer
- `src/live_transcriber.py` — Silero VAD loop + OfflineRecognizer for live mode
- `src/vulkan_probe.py` — ctypes-based Vulkan GPU enumeration (no pip dep)
- `src/recovery.py` — scan for orphaned .pcm files, recover to .wav
- `launcher/launcher.py` — tiny AppData installer/launcher (compiled separately)
- `launcher/launcher.spec` — PyInstaller ONEFILE spec for the launcher only
- `EasyScribe_whisper.spec` — PyInstaller ONEDIR for Whisper variant
- `EasyScribe_parakeet.spec` — PyInstaller ONEDIR for Parakeet variant

### Modify
- `src/config.py` — new paths, version, model variant, remove HF env vars
- `src/transcriber.py` — replace WhisperModel with sherpa-onnx OfflineRecognizer
- `src/diarizer.py` — path constant update only; Vulkan provider option
- `src/gui.py` — Record button, mic dropdown, live recording state machine
- `src/main.py` — remove cuda_setup import, add orphaned PCM recovery on startup
- `src/ffmpeg_wrapper.py` — no logic changes
- `requirements.txt` — drop faster-whisper; add sounddevice; sherpa-onnx already present
- `.github/workflows/build-release.yml` — two matrix jobs, no CUDA, 7-zip SFX assembly
- `CLAUDE.md` — v2.0 packaging rules

---

## 1. `src/config.py`

**Key changes:**
- `APP_VERSION = "2.0.0"`
- `BASE_DIR` logic unchanged — when frozen, `Path(sys.executable).parent` = the AppData install dir (`%LOCALAPPDATA%\EasyScribe\2.0.0\`)
- Add `DEFAULT_OUTPUT_DIR = Path.home() / "Documents" / "EasyScribe Recordings"` — created on first use
- Add `MODEL_VARIANT: str` — read from `BASE_DIR / "models" / "variant.json"` at startup (contains `{"variant": "whisper"}` or `{"variant": "parakeet"}`); falls back to env var `EASYSCRIBE_MODEL_VARIANT` in dev
- New model path constants (Whisper): `WHISPER_ENCODER`, `WHISPER_DECODER`, `WHISPER_TOKENS` all under `BASE_DIR / "models" / "whisper"`
- New model path constants (Parakeet): `PARAKEET_MODEL`, `PARAKEET_TOKENS` under `BASE_DIR / "models" / "parakeet"`
- VAD constants: `VAD_SAMPLE_RATE = 16000`, `VAD_CHUNK_SAMPLES = 512`
- Remove: `HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_OFFLINE`, `HF_HUB_OFFLINE` env var setup
- Remove: `WHISPER_BEAM_SIZE`, `WHISPER_VAD_FILTER`, `WHISPER_VAD_MIN_SILENCE_MS`, `REQUIRED_MODEL_FILES`

---

## 2. `src/transcriber.py` (major rewrite)

**Drop:** `WhisperModel`, ctranslate2, `list_gpus()` using ctranslate2, `_detect_device()`.

**Add:** `list_gpus()` calls `vulkan_probe.detect_vulkan_gpus()`.

**`_resolve_provider(preferred_gpu_index)` function:**
- `-1` → `"cpu"`
- Any GPU detected → `"vulkan"`, with Vulkan fallback: catch `RuntimeError` from `OfflineRecognizer(config)` construction and retry with `"cpu"`

**`_build_recognizer_config(provider)` function:** returns `sherpa_onnx.OfflineRecognizerConfig` based on `config.MODEL_VARIANT`:

```
Whisper:   OfflineWhisperModelConfig(encoder, decoder, language="en", task="transcribe")
Parakeet:  OfflineNemoCtcModelConfig(model=PARAKEET_MODEL) + tokens=PARAKEET_TOKENS
Both wrapped in OfflineModelConfig(provider=provider, num_threads=4)
```

**Chunked file transcription (replaces faster-whisper's lazy generator):**
- Read full WAV → numpy float32 array
- Split into 30-second chunks with 1-second overlap
- Per chunk: `stream = recognizer.create_stream()` → `stream.accept_waveform(16000, chunk)` → `recognizer.decode_stream(stream)` → extract `(start, end, text)`
- `start`/`end` = chunk offset + `stream.result.timestamps[0/−1]` (word-level from Whisper ONNX); for Parakeet, use CTC token timings; fallback: `start=chunk_offset, end=chunk_offset+chunk_duration`
- `progress_callback(i / num_chunks)` after each chunk
- Cancellation: check `cancel_event` between chunks

**Everything downstream unchanged:** `_build_plain_transcript`, `_build_diarized_transcript`, `_extract_speaker_clip`, `assign_speakers` all consume `list[tuple[float, float, str]]` — no changes needed.

**`validate_model_directory()`:** check for variant-specific files from `variant.json`, not for HF cache layout.

---

## 3. `src/vulkan_probe.py` (new, ~60 lines)

Use ctypes against `vulkan-1.dll` (present on any Windows machine with a GPU driver):
1. `WinDLL("vulkan-1.dll")` — if `OSError`, return `[]`
2. Create a minimal `VkInstance` with `vkCreateInstance`
3. Enumerate physical devices with `vkEnumeratePhysicalDevices`
4. Get device name string per device with `vkGetPhysicalDeviceProperties`
5. Destroy instance, return `[{"index": i, "name": name}]`

Cache the result in a module-level variable. Called once at GUI init to populate the GPU dropdown.

---

## 4. `src/mic_recorder.py` (new, ~120 lines)

**`MicRecorder` class:**
- `list_devices() -> list[dict]` — static, calls `sounddevice.query_devices()`, returns input devices
- `start(device_index, output_dir)` — opens `sounddevice.InputStream` at 16kHz mono int16, chunk=512; spawns PCM writer
- PCM writer: opens `output_dir/recording_YYYYMMDD_HHMMSS.pcm` (binary append) + companion `.json`; every 50 chunks writes updated `{"sample_rate":16000,"channels":1,"bit_depth":16,"samples_written":N}` and calls `os.fsync()`
- `get_queue() -> queue.Queue` — the mic → VAD data pipe
- `stop(convert_to_wav=True)` — flushes, optionally converts PCM → WAV using stdlib `wave`, deletes `.pcm`+`.json`

**Note:** `sounddevice.InputStream` callback puts `(frames * 2)` raw bytes onto the queue as `numpy.frombuffer(..., dtype=int16)`. The VAD loop converts to float32 by dividing by 32768.0.

---

## 5. `src/live_transcriber.py` (new, ~80 lines)

**`LiveTranscriber` class — one method `start(recognizer, mic_queue, cancel_event, on_segment)`:**

Runs in its own daemon thread:
```python
vad_model = sherpa_onnx.get_default_vad_model()  # bundled in sherpa_onnx package data
vad_config = sherpa_onnx.VadModelConfig(
    silero_vad=sherpa_onnx.SileroVadModelConfig(
        model=vad_model, threshold=0.5,
        min_silence_duration=0.5, min_speech_duration=0.25
    ),
    sample_rate=16000
)
vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=30)

while not cancel_event.is_set():
    try:
        chunk = mic_queue.get(timeout=0.1)  # float32 numpy array, 512 samples
    except queue.Empty:
        continue
    vad.accept_waveform(chunk)
    while not vad.empty():
        samples = vad.front.samples
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, samples)
        recognizer.decode_stream(stream)
        text = stream.result.text.strip()
        if text:
            on_segment(text)
        vad.pop()
```

**Note:** Verify `sherpa_onnx.get_default_vad_model()` API exists in the installed version. If not, bundle `silero_vad.onnx` explicitly from sherpa-onnx's package data directory and reference it via `config.VAD_MODEL_PATH`.

---

## 6. `src/recovery.py` (new, ~60 lines)

- `scan_for_orphans(output_dir) -> list[tuple[Path, Path, dict]]` — matches `recording_*.pcm` + `recording_*.json` pairs
- `recover_pcm_to_wav(pcm_path, metadata, wav_path)` — wraps raw bytes in a valid WAV header using `wave.open()`, truncates to `samples_written * 2` bytes
- Called from `main.py` before GUI launch; shows `tkinter.messagebox.askyesno()` dialog per orphan found

---

## 7. `src/gui.py` changes

**Options row — new mic dropdown:**
- Row added below existing GPU dropdown
- Populated by `MicRecorder.list_devices()` at startup
- Stored as `self._mic_device_index: int`

**Actions row — new Record button:**
- `[Transcribe] [Record] [Cancel] [Open Output Folder]`
- Record button: label toggles `"Record"` ↔ `"Stop Recording"`, command toggles `_on_record` ↔ `_on_stop_recording`

**New UI state `"recording"`:**
- Disables: Transcribe, file selection, output folder selector, Cancel
- Record shows "Stop Recording"
- Status colour: `"#E91E63"` (red-pink)

**`_recording_worker()` thread:**
1. `engine._ensure_model_loaded()` (shared model reuse)
2. `mic_recorder.start(device_index, DEFAULT_OUTPUT_DIR)` — crash-safe write begins
3. `live_transcriber.start(engine._recognizer, mic_queue, cancel_event, on_segment)` — VAD loop begins
4. Wait on `stop_recording_event`
5. `live_transcriber.stop()`, `mic_recorder.stop(convert_to_wav=True)`
6. Write accumulated transcript to `output_dir/recording_YYYYMMDD_HHMMSS.txt`
7. `self.after(0, lambda: self._set_ui_state("idle"))`

**`on_segment` callback:** calls `self._safe_append_log(text)` AND appends to an in-memory `list[str]` for final .txt write.

---

## 8. `src/main.py` changes

- Remove `import cuda_setup`
- Before GUI: `recovery.scan_for_orphans(config.DEFAULT_OUTPUT_DIR)` → show dialogs
- `DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)` on startup

---

## 9. `launcher/launcher.py` (new, ~80 lines)

Compiled as a separate PyInstaller ONEFILE (~5 MB), no ML deps:

```python
VERSION = "2.0.0"
INSTALL_DIR = Path(os.environ["LOCALAPPDATA"]) / "EasyScribe" / VERSION
MAIN_EXE = INSTALL_DIR / "EasyScribe.exe"

# 1. If already installed → launch directly, exit
# 2. Otherwise: find app.bundle adjacent to launcher.exe
#    (placed there by the 7-zip SFX before running launcher.exe)
# 3. Extract app.bundle (a zip) to INSTALL_DIR with progress
# 4. Launch MAIN_EXE via subprocess.Popen, exit
```

The `EASYSCRIBE_LAUNCHER_DIR` env var is NOT needed — `DEFAULT_OUTPUT_DIR` is fixed as `Documents/EasyScribe Recordings`, not relative to launcher location.

---

## 10. Distribution: 7-zip SFX assembly (in CI)

```
EasyScribe-v2.0.0-whisper.exe
  = [7zSD.sfx] + [config.txt] + [payload.7z]
                                      └── launcher.exe  (~5 MB)
                                      └── app.bundle    (~700 MB compressed)
                                               └── EasyScribe.exe + _internal/ + models/
```

`config.txt` (ASCII only — no BOM):
```
;!@Install@!UTF-8!
Title="EasyScribe"
ExtractPath="%TEMP%\EasyScribe_Setup"
RunProgram="launcher.exe"
;!@InstallEnd@!
```

CI steps (after PyInstaller ONEDIR build):
```powershell
# 1. Compress main app + models + ffmpeg into app.bundle
Compress-Archive -Path "dist\EasyScribe\*" -DestinationPath "app.bundle" -CompressionLevel Optimal

# 2. Create 7z payload
7z a payload.7z dist\launcher\launcher.exe app.bundle

# 3. Concatenate SFX + config + payload → final exe
copy /b 7zSD.sfx + config.txt + payload.7z EasyScribe-v2.0.0-whisper.exe
```

---

## 11. GitHub Actions CI restructure

**Strategy: two matrix jobs** (`variant: [whisper, parakeet]`)

Each job:
1. Checkout, Python 3.11
2. Cache + download ffmpeg (key unchanged)
3. Cache venv — new key `venv-win64-py3.11-v2` (drops nvidia packages, simpler)
4. Install: `pip install sherpa-onnx sounddevice customtkinter tkinterdnd2 pyinstaller`
5. Cache + download model (variant-specific key + URL — see §12)
6. Cache + download diarization models (key unchanged)
7. Build `launcher.exe`: `pyinstaller launcher/launcher.spec --noconfirm`
8. Build main app: `pyinstaller EasyScribe_${{ matrix.variant }}.spec --noconfirm`
9. Assemble: copy models + ffmpeg into `dist\EasyScribe\`; compress to `app.bundle`
10. Create 7-zip SFX → `EasyScribe-${{ github.event.inputs.tag }}-${{ matrix.variant }}.exe`
11. Publish to GitHub Release (draft)

**No `cuda` input flag** — no more CUDA/CPU build variants.

---

## 12. Model download URLs (to be verified against sherpa-onnx releases)

**Whisper ONNX (distil-large-v3 English):**
```
https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-distil-en-large-v3.tar.bz2
```
Expected files: `encoder.int8.onnx` (~400 MB), `decoder.int8.onnx` (~150 MB), `vocab.txt`
⚠️ Verify this URL exists before implementing — sherpa-onnx model naming conventions may differ.

**Parakeet CTC 0.6B English:**
```
https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-ctc-0.6b-en.tar.bz2
```
Expected files: `model.int8.onnx` (~600 MB), `tokens.txt`
⚠️ Verify this URL exists before implementing. Alternative: `parakeet-tdt-0.6b-en`.

**Diarization (unchanged):**
- Segmentation: `https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2`
- Embedding: `https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx`

---

## 13. PyInstaller spec changes (both variants)

**Remove from both specs:**
- `_collect_nvidia_dlls()` function
- `collect_all('faster_whisper')`, `collect_all('ctranslate2')`
- `hiddenimports`: ctranslate2, faster_whisper, huggingface_hub, torch, torchaudio
- nvidia DLL binaries entries

**Add to both specs:**
- `collect_all('sounddevice')` — needed for `_sounddevice.pyd` + `portaudio_x64.dll`
- `hiddenimports`: `sounddevice`, `sounddevice._sounddevice`
- Datas entry for `models/variant.json` — written at build time by CI (bakes in the variant)

**Variant-specific datas:**
- Whisper spec: `("models/whisper/", "models/whisper")`
- Parakeet spec: `("models/parakeet/", "models/parakeet")`

**Both retain:** `collect_all('sherpa_onnx')`, `ffmpeg` datas, `customtkinter` datas, `tkinterdnd2` datas.

**`launcher/launcher.spec`:**
- ONEFILE, console=True
- Entry: `launcher/launcher.py`
- Excludes: everything except stdlib (pathlib, os, sys, zipfile, subprocess, shutil)
- No sherpa-onnx, no customtkinter, no sounddevice

---

## 14. Risk flags

| Risk | Mitigation |
|---|---|
| sherpa-onnx distil-large-v3 ONNX model may not exist under that exact name | Verify URL against [sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases) before implementing CI step |
| Parakeet CTC ONNX accuracy may differ from NeMo reference | Run WER test on sample audio before shipping |
| Vulkan provider may not be available in the sherpa-onnx PyPI wheel | Confirm `provider="vulkan"` is supported in `sherpa_onnx >= 1.10.0`; if not, CPU only for v2.0 and revisit |
| `sherpa_onnx.get_default_vad_model()` API may not exist | Check installed version API; fallback: bundle `silero_vad.onnx` explicitly |
| sounddevice `portaudio_x64.dll` may not be collected by `collect_all` | Verify after first ONEDIR build; add explicit binaries entry if missing |
| 7-zip SFX will trigger SmartScreen (unsigned exe) | Document in release notes — same situation as current builds |
| AppData partial extraction if user closes launcher mid-extract | Launcher detects `MAIN_EXE` missing after extraction completes → shows error, cleans partial state |

---

## 15. Verification

1. **sherpa-onnx transcription test (dev):** run `test_transcription.py` (adapt existing `test_diarization.py` pattern) against `test_speech.wav`, verify non-empty output for both Whisper and Parakeet configs.
2. **VAD loop test:** feed 10 seconds of pre-recorded audio (converted to 512-sample float32 chunks) through `LiveTranscriber`; verify `on_segment` fires at least once with non-empty text.
3. **Crash recovery test:** write 1000 PCM chunks to a `.pcm` file + `.json`, call `recover_pcm_to_wav`, verify output is a valid WAV with `wave.open()`.
4. **Vulkan probe test:** call `detect_vulkan_gpus()` — must return a list (possibly empty) without raising.
5. **Full CI build:** both matrix jobs pass, two SFX `.exe` files produced, sizes in expected range (~700 MB each).
6. **First-run test (Windows VM):** run SFX on a clean VM — launcher runs, AppData dir created, `EasyScribe.exe` launches.
7. **Repeat-run test:** run same SFX again — launcher detects existing install, launches in <5 seconds.
8. **Live transcription test:** record 30 seconds of speech, verify transcript appears in log box, `.wav` saved to Documents/EasyScribe Recordings.
9. **Crash recovery UX test:** start recording, kill process via Task Manager, relaunch, verify recovery dialog.

---

## Branch

New branch: `v2` branched from `sherpa-onnx-diarization` (which has the current working CPU diarization).
