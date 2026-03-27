# Transcriber7

A portable, fully offline Windows desktop application for transcribing any media file to plain text using [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) (`faster-whisper-large-v3-turbo`) with optional speaker identification powered by [pyannote.audio](https://github.com/pyannote/pyannote-audio).

- **No internet required at runtime** — works completely offline
- **No installer** — copy the folder to any Windows machine or USB stick and run
- **No Python required** on the target machine — everything is bundled
- **GPU-accelerated** on NVIDIA GPUs, automatic CPU fallback
- **Speaker diarization** — identify who said what, with optional name assignment

---

## Features

### Transcription
Converts any audio or video file to a plain UTF-8 text file. Uses the `faster-whisper-large-v3-turbo` model with voice activity detection (VAD) to skip silence and avoid hallucinations.

### Timestamps
Group output into natural-pause blocks, each headed with a `[HH:MM:SS]` timestamp. Blocks break when there is a ~2-second gap in speech.

### Speaker Identification
Uses pyannote.audio's `speaker-diarization-3.1` pipeline to detect and separate speakers. The transcript is formatted with `[Speaker N]` headers at each speaker change. When combined with timestamps, headers include both time and speaker: `[00:01:23] [Speaker 1]`.

### Speaker Naming
After diarization completes, a popup lets you name each speaker. For each detected speaker you can play a short audio sample (to identify whose voice it is), then type a custom name. Names replace the generic `Speaker 1` labels in the output file.

---

## Usage

1. Click **Select File(s)** or drag and drop media files onto the drop zone
2. Optionally click **Select Folder** to choose where transcripts are saved
   (defaults to the same folder as each input file)
3. Choose a device from the **Device** dropdown (auto-detected GPUs are listed)
4. Tick options as needed:
   - **Include timestamps** — adds `[HH:MM:SS]` block headers
   - **Identify speakers** — runs speaker diarization (requires bundled models)
5. Click **Transcribe**
6. If speaker identification is enabled, a popup will appear when diarization is done — play samples and enter names, then click **Use These Names**
7. Click **Open Output Folder** when done

### Batch mode

Select multiple files at once — each gets its own `.txt` transcript. If one file fails, transcription continues for the remaining files.

### Cancel

Click **Cancel** at any time to stop the current job cleanly.

---

## Output Examples

**Plain text:**
```
Hello, this is the first sentence. And here is another.
```

**With timestamps:**
```
[00:00:01]
Hello, this is the first sentence.

[00:00:14]
After a pause, the next block starts here.
```

**With speakers:**
```
[Speaker 1]
Hello, this is the first sentence.

[Speaker 2]
And I'm replying here.
```

**With speakers + timestamps:**
```
[00:00:01] [Alice]
Hello, this is the first sentence.

[00:00:14] [Bob]
After a pause, the next block starts here.
```

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
    hf_cache\
      hub\
        models--pyannote--speaker-diarization-3.1\
        models--pyannote--segmentation-3.0\
        models--pyannote--wespeaker-voxceleb-resnet34-LM\
  ffmpeg\
    ffmpeg.exe
    ffprobe.exe
  logs\              <- log files written here at runtime
  temp\              <- temporary WAV files (auto-cleaned)
  (PyInstaller DLLs and .pyd files alongside the exe)
```

---

## Requirements (build machine only)

| Requirement | Notes |
|---|---|
| Windows 10/11 64-bit | Build and target platform |
| Python 3.11 | Must be on PATH |
| Model files | See CI workflow for download commands |
| Internet access | Only needed during build |

The GitHub Actions CI workflow handles all model downloads, dependency installs, and packaging automatically on every tagged release.

---

## Architecture

```
src/
  main.py            Entry point; validates deps, launches GUI
  config.py          Path resolution, constants, offline env vars
  logger.py          Rotating log file setup
  ffmpeg_wrapper.py  Subprocess ffmpeg with cancellation polling
  transcriber.py     Faster Whisper engine (lazy load, GPU/CPU detection)
  diarizer.py        pyannote.audio speaker diarization engine
  gui.py             CustomTkinter UI with threaded worker
```

### Offline guarantee

Three independent layers prevent any network access at runtime:

1. **Environment variables** set in `config.py` before any library import:
   `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, etc.

2. **Explicit API argument**: `WhisperModel(..., local_files_only=True)`

3. **Local path loading**: models are always loaded from absolute local paths,
   never from hub alias strings

---

## Troubleshooting

| Problem | Solution |
|---|---|
| "Model files missing" on startup | Copy Whisper model files to `models\faster-whisper-large-v3-turbo\` next to the exe |
| "ffmpeg.exe not found" | Copy `ffmpeg.exe` and `ffprobe.exe` to `ffmpeg\` next to the exe |
| "Identify speakers" checkbox is greyed out | Diarization models not bundled; use a release build from CI |
| Antivirus flags the exe | PyInstaller executables may trigger false positives; add an exclusion |
| Very slow transcription | No NVIDIA GPU detected; CPU mode is slower by design |
| Out of memory on large files | Try CPU mode |
| "Access denied" writing transcript | Close the output .txt file in any other program |

---

## License

This project is released under the MIT License.

FFmpeg is included under the GPL v3 license. See https://ffmpeg.org/legal.html

Whisper model weights: MIT License — https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo

pyannote.audio models are subject to their own license terms. See https://huggingface.co/pyannote
