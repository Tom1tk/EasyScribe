# EasyScribe

A portable, fully offline Windows desktop application for transcribing media files and live microphone audio to plain text. Speaker diarization, voice activity detection, and live transcription run on [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (CPU). File transcription uses [whisper.cpp](https://github.com/ggml-org/whisper.cpp) with beam search, accelerated on any Vulkan-capable GPU (NVIDIA, AMD, Intel) when available — falling back to the sherpa-onnx CPU engine if not bundled.

- **No internet required at runtime** — works completely offline
- **Single .exe installer** — runs on any Windows 10/11 machine, no setup wizard
- **No Python required** on the target machine — everything is bundled
- **GPU-accelerated file transcription** — whisper.cpp with beam search on any Vulkan GPU, CPU fallback
- **Live microphone transcription** — real-time VAD-chunked transcription with crash-safe recording
- **Speaker diarization** — identify who said what, with optional name assignment

---

## Installation

**No installer, no admin rights required.**

1. Place `EasyScribe-<version>-whisper.exe` anywhere — a local folder, a USB stick, a shared network drive.
2. Double-click. On first run, the installer GUI appears — pick an install location (or accept the default) and click **Install**. This extracts an `EasyScribe\` folder there (~30 seconds), creates **Desktop** and **Start Menu** shortcuts, then launches EasyScribe.
3. From then on, launch EasyScribe from the **Desktop or Start Menu shortcut** — it starts the installed `EasyScribe.exe` directly, in under a second. Re-running the big `.exe` still works (it detects the existing install and relaunches it), but is slower since it re-extracts itself first each time.

File transcripts are saved next to each recording (or in the folder you choose). Live recordings and their transcripts are saved to `EasyScribe\recordings\`.

To move the app, move both the `.exe` and the `EasyScribe\` folder together.

> **SmartScreen warning:** PyInstaller executables are unsigned. Click "More info → Run anyway" to proceed.

---

## Features

### File Transcription
Converts any audio or video file to a plain UTF-8 `.txt` file. Audio is decoded via bundled ffmpeg and resampled to 16 kHz mono. If `whisper-cli` (whisper.cpp) is bundled, it transcribes the whole file with beam search (`beam_size=5`), using a Vulkan GPU if one is detected or CPU otherwise — the device actually used is shown in the log. Otherwise, EasyScribe falls back to the sherpa-onnx VAD+greedy engine, processing the file in speech segments.

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

1. Select **Transcribe files** at the top
2. Select **Choose files**, or drag audio or video files onto the drop area
3. Optionally select **Change folder** to choose where transcripts are saved
   (by default, each transcript goes next to its recording)
4. Tick options as needed:
   - **Add timestamps** (blue): adds `[HH:MM:SS]` paragraph headers
   - **Name the speakers** (amber): finds who is talking
5. Select **Transcribe N files**. Progress shows below, in the colour of the current step
6. If you chose **Name the speakers**, a window opens when the speakers are found. Play each sample, type the names, then select **Save names**
7. Select **Open transcript** or **Open folder** when it is done

### Record from microphone

1. Select **Record live** at the top (the app turns coral for recording)
2. Choose a microphone from the **Microphone** list
3. Select **Start recording**. A timer shows how long you have recorded
4. Speak. The words appear in **Live transcript** as you talk
5. Select **Stop recording**. The audio and the transcript are saved in `recordings/`

### Batch mode
Add more than one file. Each gets its own `.txt` transcript and its own status (Waiting, Working, Done, Failed). If one file fails, the others continue, and **Try again** lets you run the batch again.

### Stop
Select **Stop** next to the progress bar to stop the current job cleanly.

### Colours
Each colour means one thing everywhere in the app: **teal** for files and the main actions, **coral** for recording, **amber** for speakers, **blue** for timestamps, **green** for done and **red** for errors.

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
  EasyScribe-<version>-whisper.exe   <- the launcher; keep this to re-run or move the app
  EasyScribe\
    EasyScribe.exe
    _internal\               <- PyInstaller runtime (DLLs, .pyd files)
    models\
      whisper\
        turbo-encoder.int8.onnx
        turbo-decoder.int8.onnx
        turbo-tokens.txt
      diarization\
        segmentation.onnx
        embedding.onnx
      silero_vad.onnx
    whispercpp\              <- whisper.cpp file-transcription engine (Vulkan/CPU)
      whisper-cli.exe
      ggml-large-v3-turbo-q5_0.bin
      *.dll
    ffmpeg\
      ffmpeg.exe
      ffprobe.exe
    recordings\              <- live mic recordings and their transcripts
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
  config.py          Path resolution and constants
  logger.py          Rotating log file setup
  ffmpeg_wrapper.py  Subprocess ffmpeg with cancellation polling
  transcriber.py     File transcription engine (whisper.cpp if bundled, else sherpa-onnx)
  whispercpp_wrapper.py  Subprocess whisper-cli (whisper.cpp), beam search + Vulkan/CPU
  diarizer.py        sherpa-onnx speaker diarization engine
  mic_recorder.py    sounddevice capture + crash-safe PCM writer
  live_transcriber.py  Silero VAD loop + OfflineRecognizer for live mode
  recovery.py        Orphaned .pcm file scanner and WAV recovery
  gui.py             CustomTkinter UI with threaded workers

  offline_guard.py   Blocks all network access (runs first, see "Offline guarantee")

launcher/
  launcher.py        Extract-once installer (compiled separately, ~5 MB)
  launcher.spec      PyInstaller ONEFILE spec for launcher

assets/
  EasyScribe.ico     App icon (both .exe files, all windows, shortcuts)
  make_icon.py       Regenerates the icon from the logo design (needs Pillow)
  version_resource.py  Windows file-version info for both specs, from APP_VERSION
```

The version is set in one place: `APP_VERSION` in `src/config.py` (keep `VERSION` in `launcher/launcher.py` the same; `tests/test_version.py` checks this). Build a release from **Actions → Build and Publish Release** with the tag `v` + `APP_VERSION`.

### Inference engines

Voice activity detection, speaker diarization, and live microphone transcription run on sherpa-onnx via `provider="cpu"` in `sherpa_onnx.OfflineRecognizerConfig` — sherpa-onnx 1.13.2 has no Vulkan provider, so these stay CPU-only.

File transcription prefers `whisper-cli` (whisper.cpp), built with Vulkan support — a different engine/binary from sherpa-onnx, so it runs on GPU (NVIDIA, AMD, or Intel) when one is available, with automatic CPU fallback. If `whisper-cli` or its model isn't bundled (e.g. a dev checkout before CI bundles it), file transcription falls back to the same sherpa-onnx CPU engine used for live mode.

### Offline guarantee and privacy (GDPR)

EasyScribe never sends data off the computer. Audio, transcripts and speaker names stay on the local disk. There are no accounts, analytics, crash reports or update checks.

How this is enforced:

- **All models are local.** They load from absolute paths inside the install directory. There are no HuggingFace Hub downloads.
- **Network access is blocked in the process.** `src/offline_guard.py` runs first in `main.py`, before any other import. It refuses every socket connection and DNS lookup that is not to loopback, disables proxies (`NO_PROXY=*`) and sets the usual opt-out variables (`HF_HUB_OFFLINE`, `DO_NOT_TRACK`, ...). A blocked attempt is logged.
- **ffmpeg and ffprobe read local files only.** They run with `-protocol_whitelist file`, so a playlist or concat file cannot make them fetch a URL.
- **No network code.** `tests/test_offline_guard.py` fails if a module in `src/` or `launcher/` imports a network library (`urllib`, `http`, `requests`, `webbrowser`, ...).
- **Temporary audio is deleted.** Converted WAV files and whisper.cpp output in the temp folder are removed after each file, at startup and at exit.

What is stored, and where (the app shows this too: select **Privacy** at the bottom of the window):

| Data | Location | Kept until |
|---|---|---|
| Transcripts | Next to each recording, or the folder you choose | You delete them |
| Live recordings | `recordings/` in the install folder | You delete them |
| Technical logs | `logs/` in the install folder. File names and progress only, never transcript text | The last 10 are kept |

To remove all data, delete the transcripts you made and the EasyScribe folder.

### Distribution

```
EasyScribe-<version>-whisper.exe   (launcher.exe, PyInstaller one-file)
  └── app.bundle                   (zip: EasyScribe.exe + _internal/ + models/ + whispercpp/ + ffmpeg/)
```

On first run the launcher asks where to install, extracts `app.bundle` into an `EasyScribe\` folder there, creates Desktop and Start Menu shortcuts and opens the app.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| SmartScreen blocks the exe | Click "More info → Run anyway" — the exe is unsigned but safe |
| "Model files missing" on startup | Re-run the installer; extraction may have been interrupted |
| "ffmpeg.exe not found" | Re-run the installer |
| "Identify speakers" checkbox is greyed out | Diarization models not found; re-run the installer |
| Mic dropdown shows no devices | No audio input devices found; check Windows sound settings |
| Recovery dialog on launch | A previous recording was interrupted; choose Recover to save the audio or Delete to discard |
| Antivirus flags the exe | PyInstaller executables may trigger false positives; add an exclusion |
| Slow transcription | EasyScribe runs on CPU; large files take proportionally longer — this is expected |

---

## License

This project is released under the MIT License.

FFmpeg is included under the GPL v3 license. See https://ffmpeg.org/legal.html

sherpa-onnx and its bundled models (Whisper ONNX large-v3-turbo, pyannote segmentation-3.0 ONNX export, WeSpeaker ResNet34-LM embedding, Silero VAD) are subject to their own license terms. See https://github.com/k2-fsa/sherpa-onnx
