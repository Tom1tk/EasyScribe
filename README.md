# Transcriber7

A portable, fully offline Windows desktop application for transcribing any media file to plain text using [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) and the `mobiuslabsgmbh/faster-whisper-large-v3-turbo` model.

- **No internet required at runtime** — works completely offline
- **No installer** — copy the folder to any Windows machine or USB stick and run
- **No Python required** on the target machine — everything is bundled
- **GPU-accelerated** on NVIDIA GPUs, automatic CPU fallback

---

## Requirements (build machine only)

| Requirement | Notes |
|---|---|
| Windows 10/11 64-bit | Build and target platform |
| Python 3.10+ | Must be on PATH (`python --version`) |
| Model snapshot files | See below |
| Internet access | Only needed during build to download ffmpeg and Python packages |

---

## Before Building: Place Model Files

You need the local snapshot of `mobiuslabsgmbh/faster-whisper-large-v3-turbo`.

Create this folder structure in the project root:

```
models\
  faster-whisper-large-v3-turbo\
    config.json
    model.bin
    preprocessor_config.json
    tokenizer.json
    vocabulary.json
```

Download the model from Hugging Face:

```powershell
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('mobiuslabsgmbh/faster-whisper-large-v3-turbo', local_dir='models/faster-whisper-large-v3-turbo', ignore_patterns=['*.gitattributes'])"
```

Or download manually from: https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo

---

## Building

### CPU-only build (default)

```batch
build_windows.bat
```

### GPU build (CUDA 11.8)

```batch
build_windows.bat --cuda 11.8
```

### GPU build (CUDA 12.1)

```batch
build_windows.bat --cuda 12.1
```

### GPU build (CUDA 12.4 — latest)

```batch
build_windows.bat --cuda 12.4
```

> **PowerShell tip:** Run the build script from **Command Prompt** (`cmd.exe`), not PowerShell.
> If you must use PowerShell, note that `>` is a redirect operator — never write
> `pip install torch>=2.3.0` unquoted. The build script avoids this problem by not
> embedding version specifiers in the pip call.

The build script will:

1. Check Python is installed
2. Validate model files exist
3. **Auto-download ffmpeg** (Windows static build from GitHub) if not already present
4. Create a Python virtual environment
5. Install all Python dependencies (including PyTorch)
6. Build the portable executable with PyInstaller
7. Copy model files and ffmpeg into the output folder
8. Create `logs/` and `temp/` runtime directories

**Output:** `dist\PortableTranscriber\`

---

## Running

Double-click `dist\PortableTranscriber\PortableTranscriber.exe`.

No installation required. To deploy to another machine, copy the entire `PortableTranscriber\` folder.

---

## Usage

1. Click **Select File(s)** or drag and drop media files onto the drop zone
2. Optionally click **Select Folder** to choose where transcripts are saved
   (defaults to the same folder as each input file)
3. Choose a device from the **Device** dropdown (auto-detected GPUs are listed)
4. Optionally tick **Include timestamps** for time-coded output
5. Click **Transcribe**
6. Watch the progress bar and log output
7. Click **Open Output Folder** when done

### Batch mode

Select multiple files at once — each gets its own `.txt` transcript. If one file fails, transcription continues for the remaining files.

### Cancel

Click **Cancel** at any time to stop the current job cleanly.

---

## Output

Transcripts are saved as UTF-8 plain text files:

- Plain text (default): `filename.txt`
- Timestamped: `filename.txt` with lines like `[00:00:01.234 --> 00:00:04.567] Hello world`

---

## Supported Input Formats

| Video | Audio |
|---|---|
| mp4, mkv, mov, avi, webm | mp3, wav, m4a, flac, ogg, opus, aac |

---

## Portable Folder Layout

```
PortableTranscriber\
  PortableTranscriber.exe
  models\
    faster-whisper-large-v3-turbo\
      config.json
      model.bin
      preprocessor_config.json
      tokenizer.json
      vocabulary.json
  ffmpeg\
    ffmpeg.exe
    ffprobe.exe
  logs\              <- log files written here at runtime
  temp\              <- temporary WAV files (auto-cleaned)
  (PyInstaller DLLs and .pyd files alongside the exe)
```

---

## Swapping the Model

To use a different Faster Whisper model:

1. Place the model snapshot files in a new subfolder under `models\`, e.g.:
   `models\faster-whisper-medium\`

2. Edit `src\config.py`, line:
   ```python
   MODEL_FOLDER_NAME = "faster-whisper-large-v3-turbo"
   ```
   Change to match your folder name.

3. Rebuild with `build_windows.bat`.

---

## Architecture

```
src/
  main.py            Entry point; validates deps, launches GUI
  config.py          Path resolution, constants, offline env vars
  logger.py          Rotating log file setup
  ffmpeg_wrapper.py  Subprocess ffmpeg with cancellation polling
  transcriber.py     Faster Whisper engine (lazy load, GPU/CPU detection)
  gui.py             CustomTkinter UI with threaded worker
hooks/
  hook-ctranslate2.py  PyInstaller hook for ctranslate2 dynamic libs
```

### Offline guarantee

Three independent layers prevent any network access:

1. **Environment variables** set in `config.py` before any library import:
   `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, etc.

2. **Explicit API argument**: `WhisperModel(..., local_files_only=True)`

3. **Local path**: the model is always loaded from a directory path string,
   never from a model alias like `"large-v3-turbo"` that could trigger a download

### GPU selection

The Device dropdown is populated at startup by querying `torch.cuda.device_count()`.
On systems with multiple NVIDIA GPUs, each is listed by name. Selecting a different
GPU unloads the cached model so it reloads on the correct device on the next run.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| "Model files missing" on startup | Copy model files to `models\faster-whisper-large-v3-turbo\` next to the exe |
| "ffmpeg.exe not found" on startup | Copy `ffmpeg.exe` and `ffprobe.exe` to `ffmpeg\` next to the exe |
| Antivirus flags the exe | PyInstaller executables may trigger false positives; add an exclusion |
| Very slow transcription | No NVIDIA GPU detected; CPU mode is slower by design |
| Out of memory on large files | Try a smaller model, or use CPU mode |
| "Access denied" writing transcript | Close the output .txt file in any other program |

---

## License

This project is released under the MIT License.

FFmpeg is included under the GPL v3 license. See https://ffmpeg.org/legal.html

The Whisper model weights are subject to the MIT License.
See https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo
