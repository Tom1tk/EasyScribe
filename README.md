# EasyScribe

A portable, fully offline Windows desktop application for transcribing media files and live microphone audio to plain text. Uses [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) for all inference — transcription, speaker diarization, and voice activity detection — with GPU acceleration via Vulkan (any GPU, no CUDA required).

- **No internet required at runtime** — works completely offline
- **Single .exe installer** — runs on any Windows 10/11 machine, no setup wizard
- **No Python required** on the target machine — everything is bundled
- **GPU-accelerated** on any Vulkan-capable GPU (NVIDIA, AMD, Intel), automatic CPU fallback
- **Live microphone transcription** — real-time VAD-chunked transcription with crash-safe recording
- **Speaker diarization** — identify who said what, with optional name assignment

---

## Installation

**No installer, no admin rights required.**

1. Place `EasyScribe-v2.0.0-whisper.exe` (or `-parakeet.exe`) anywhere — a local folder, a USB stick, a shared network drive.
2. Double-click. On first run it extracts an `EasyScribe\` folder next to itself (~30 seconds). EasyScribe launches automatically.
3. On subsequent runs the same `.exe` detects the existing `EasyScribe\` folder and launches in under a second.

Transcripts and recordings are saved to `EasyScribe\recordings\` next to the `.exe`.

To move the app, move both the `.exe` and the `EasyScribe\` folder together.

> **SmartScreen warning:** PyInstaller executables are unsigned. Click "More info → Run anyway" to proceed.

---

## Model Variants

Two builds are released. They are otherwise identical in features.

| Variant | Model | Notes |
|---|---|---|
| `whisper` | Whisper ONNX distil-large-v3 (English, int8) | Higher accuracy, ~700 MB |
| `parakeet` | Parakeet TDT 0.6B v3 int8 (English) | Faster inference, ~700 MB |

---

## Features

### File Transcription
Converts any audio or video file to a plain UTF-8 `.txt` file. Audio is decoded via bundled ffmpeg, resampled to 16 kHz mono, then processed in 30-second overlapping chunks.

### Live Microphone Recording
Record directly from any input device. Voice activity detection (Silero VAD) automatically segments speech — only non-silent segments are transcribed. Recording writes crash-safe `.pcm` + `.json` sidecar files; if the app closes unexpectedly, the next launch offers to recover the audio.

### Timestamps
Group output into natural-pause blocks, each headed with a `[HH:MM:SS]` timestamp.

### Speaker Identification
Offline speaker diarization via sherpa-onnx (pyannote segmentation-3.0 ONNX + WeSpeaker ResNet34-LM embedding). Detects and separates speakers; output is formatted with `[Speaker N]` headers at each speaker change. Speaker identification is available for file transcription only (not live recording).

### Speaker Naming
After diarization completes, a popup lets you name each speaker. Play a short audio clip to identify each voice, then type a name. Names replace the generic `Speaker N` labels in the output file.

---

## Usage

### Transcribe a file

1. Click **Select File(s)** or drag and drop media files onto the drop zone
2. Optionally click **Select Folder** to choose where transcripts are saved
   (defaults to `Documents\EasyScribe Recordings\`)
3. Choose a device from the **Device** dropdown — Vulkan-capable GPUs are listed; select **CPU** to force CPU mode
4. Tick options as needed:
   - **Include timestamps** — adds `[HH:MM:SS]` block headers
   - **Identify speakers** — runs speaker diarization
5. Click **Transcribe**
6. If speaker identification is enabled, a popup appears when diarization completes — play samples, enter names, then click **Use These Names**
7. Click **Open Output Folder** when done

### Record from microphone

1. Choose a microphone from the **Mic** dropdown
2. Click **Record** — the button turns red and shows **Stop Recording**
3. Speak; transcribed segments appear in the log box in real time
4. Click **Stop Recording** — the final transcript is saved to `Documents\EasyScribe Recordings\`

### Batch mode
Select multiple files at once — each gets its own `.txt` transcript. If one file fails, transcription continues for the remaining files.

### Cancel
Click **Cancel** at any time to stop the current transcription job cleanly.

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

## Layout after first run

The `.exe` extracts an `EasyScribe\` folder next to itself:

```
<wherever you placed the .exe>
  EasyScribe-v2.0.0-whisper.exe   <- the launcher; keep this to re-run or move the app
  EasyScribe\
    EasyScribe.exe
    _internal\               <- PyInstaller runtime (DLLs, .pyd files)
    models\
      whisper\               <- (whisper variant only)
        distil-large-v3-encoder.int8.onnx
        distil-large-v3-decoder.int8.onnx
        distil-large-v3-tokens.txt
      parakeet\              <- (parakeet variant only)
        encoder.int8.onnx
        decoder.int8.onnx
        joiner.int8.onnx
        tokens.txt
      diarization\
        segmentation.onnx
        embedding.onnx
      silero_vad.onnx
      variant.json           <- baked in at build time: {"variant": "whisper"}
    ffmpeg\
      ffmpeg.exe
      ffprobe.exe
    recordings\              <- transcripts and mic recordings saved here
    logs\                    <- rotating log files
    temp\                    <- temporary WAV files (auto-cleaned)
```

To move the app to another machine or USB stick, copy both the `.exe` and the `EasyScribe\` folder.

---

## Requirements (build machine only)

| Requirement | Notes |
|---|---|
| Windows 10/11 64-bit | Build and target platform |
| Python 3.11 | Must be on PATH |
| Internet access | Only needed during build — CI downloads models |

The GitHub Actions CI workflow handles all model downloads, dependency installs, and packaging automatically on every tagged release.

---

## Architecture

```
src/
  main.py            Entry point; creates output dir, runs orphan recovery, launches GUI
  config.py          Path resolution, constants, model variant detection
  logger.py          Rotating log file setup
  ffmpeg_wrapper.py  Subprocess ffmpeg with cancellation polling
  transcriber.py     sherpa-onnx OfflineRecognizer (Vulkan/CPU, Whisper or Parakeet)
  vulkan_probe.py    ctypes-based Vulkan GPU enumeration (no pip dependency)
  diarizer.py        sherpa-onnx speaker diarization engine
  mic_recorder.py    sounddevice capture + crash-safe PCM writer
  live_transcriber.py  Silero VAD loop + OfflineRecognizer for live mode
  recovery.py        Orphaned .pcm file scanner and WAV recovery
  gui.py             CustomTkinter UI with threaded workers

launcher/
  launcher.py        AppData extract-once installer (compiled separately, ~5 MB)
  launcher.spec      PyInstaller ONEFILE spec for launcher
```

### GPU acceleration

`provider="vulkan"` is passed to `sherpa_onnx.OfflineRecognizerConfig`. Vulkan uses the system GPU driver — no additional DLLs need to be bundled. If Vulkan initialisation fails (e.g. on a headless machine), the engine transparently retries with `provider="cpu"`.

### Offline guarantee

All models are loaded from absolute local paths inside the install directory. No HuggingFace Hub, no network calls, no telemetry.

### Distribution

```
EasyScribe-v2.0.0-whisper.exe
  = [7zSD.sfx] + [config.txt] + [payload.7z]
                                    ├── launcher.exe  (~5 MB)
                                    └── app.bundle    (zip: EasyScribe.exe + _internal/ + models/)
```

The SFX extracts to `%TEMP%\EasyScribe_Setup` and runs `launcher.exe`, which copies the app to AppData and then deletes the temp files.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| SmartScreen blocks the exe | Click "More info → Run anyway" — the exe is unsigned but safe |
| "Model files missing" on startup | Re-run the installer; extraction may have been interrupted |
| "ffmpeg.exe not found" | Re-run the installer |
| No GPU listed in Device dropdown | No Vulkan-capable GPU detected; CPU mode will be used |
| "Identify speakers" checkbox is greyed out | Diarization models not found; re-run the installer |
| Mic dropdown shows no devices | No audio input devices found; check Windows sound settings |
| Recovery dialog on launch | A previous recording was interrupted; choose Recover to save the audio or Delete to discard |
| Antivirus flags the exe | PyInstaller executables may trigger false positives; add an exclusion |
| Slow transcription | CPU mode is slower by design; a Vulkan-capable GPU significantly improves speed |

---

## License

This project is released under the MIT License.

FFmpeg is included under the GPL v3 license. See https://ffmpeg.org/legal.html

sherpa-onnx and its bundled models (Whisper ONNX, Parakeet TDT, pyannote segmentation-3.0 ONNX export, WeSpeaker ResNet34-LM embedding, Silero VAD) are subject to their own license terms. See https://github.com/k2-fsa/sherpa-onnx
