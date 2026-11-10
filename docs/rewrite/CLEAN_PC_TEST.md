# Clean-PC acceptance test (Windows 10/11 x64)

Run on a clean Windows 10 or 11 PC, or a VM with a snapshot. Record pass/fail
and timings in `REWRITE_STATUS.md`. Use invented/public-domain audio only.

## 0. Preconditions

- 64-bit Windows 10/11, x86-64 CPU with AVX2, FMA, F16C and BMI2.
- Download `EasyScribe-v4.0.1.exe` and its `.sha256` from the pre-release;
  verify the hash with `Get-FileHash`.
- Do not install anything system-wide. Keep a snapshot before the first run.

## 1. Unpack

- [ ] Start with about 4 GB free (the 1.92 GB exe plus ~1.7 GB of extracted
      files). The first launch shows the preparing screen, then the main
      window; the runtime appears under
      `%LOCALAPPDATA%\EasyScribe\runtime\v1-<hash>`.
- [ ] Close and start again: the cached launch skips preparation and does not
      rewrite the `.complete` marker.
- [ ] Repeat from the snapshot with about 2 GB free: expect a clear
      "Not enough free disk space" message and no half-unpacked runtime that
      breaks the next start.
- [ ] Start a newer exe while the old runtime still exists: extraction needs up
      to about 5.4 GB free at peak (exe, old runtime and staging copy), and the
      old runtime is removed after the new one launches. Test with about 4 GB
      free to see the clear message instead of a partial unpack.

## 2. Microphone

- [ ] Record 2 min with the default microphone; the WAV and transcript save.
- [ ] During a recording, select **Pause**: the button becomes **Resume**, the
      timer and level meter stop, and the activity says `Recording paused`.
      Wait 10 s, select **Resume**, and finish. The saved WAV length excludes
      the paused 10 s, and the live transcript contains no audio from it.
- [ ] Stop while paused: the audio captured before the pause is saved and the
      paused tail is absent.
- [ ] Pause at 00:00 and stop immediately: the result is `Recording cancelled`
      with `Nothing was saved.`, not a save error.
- [ ] After Stop, "Recording saved" appears and the WAV plays from the chosen
      folder (or the default `Documents\EasyScribe`).
- [ ] Record 2 min with a USB headset; select it in the microphone menu.
- [ ] The microphone menu opens over Step 2 without moving it; clicking
      elsewhere or pressing Escape closes it.
- [ ] Show details on the Record tab: the log appears below Privacy and never
      covers the Start/Stop button. Test this at 100% and 150% scaling and in
      the 600–678 pt short-window layouts.
- [ ] If an earlier build left `recording_*.pcm` and `.json` files in
      `%LOCALAPPDATA%\EasyScribe\recordings`, the next launch offers to recover
      them and writes a playable WAV.
- [ ] Unplug the microphone mid-recording: expect a message and the recording
      kept so far.

## 3. DPI and window

- [ ] Test at 100%, 150% and 200% scaling.
- [ ] Drag the window between monitors with different scaling.
- [ ] No clipped text, no blurry UI, and the naming panel is usable.
- [ ] The EasyScribe logo appears in the window's top-left title-bar icon and
      on the taskbar while the app is running.
- [ ] Click the rename box: a thin caret blinks at the insertion point, and it
      moves as you type or click elsewhere in the name.

## 4. Dialogs and drag and drop

- [ ] Open files with the file dialog.
- [ ] Choose the "Save recordings to" folder.
- [ ] Drop one audio file, then several, then a non-audio file.
- [ ] Use paths with spaces, non-ASCII characters (for example `Müller`) and a
      OneDrive folder.

## 5. Mark of the Web

- [ ] Download from the GitHub release and check SmartScreen.
- [ ] Confirm the README's Unblock tip still applies if Windows blocks it.

## 6. GPU matrix

- [ ] Dedicated GPU: the log says `Engine: Vulkan` and names the dedicated GPU.
- [ ] Integrated GPU only: Vulkan is used, or CPU with a clear reason.
- [ ] No Vulkan driver: the app starts and transcription retries on the CPU.
- [ ] Force a GPU failure (for example disable/roll back the driver after a
      successful run): the next run logs the remembered GPU, uses the CPU, and
      still transcribes.
- [ ] `EASYSCRIBE_WHISPER_CPU=1`: the log says CPU was forced and transcription
      matches the GPU text.

## 7. Speed comparison (the gate for replacing v3.2.3)

Use the same 5–10 min invented-content WAV on one real PC:

- [ ] v3.2.3 with GPU (default).
- [ ] v3.2.3 with CPU (`whisper_gpu_disabled_for` set, or no Vulkan driver).
- [ ] v4.0.1 with GPU (default).
- [ ] v4.0.1 with CPU (`EASYSCRIBE_WHISPER_CPU=1`).

Record wall time and the timings block for each in `REWRITE_STATUS.md`.
Pass: rust CPU is within 10% of v3.2.3 CPU, and rust GPU is within 10% of
v3.2.3 GPU.

## 8. Diarization and Stop

- [ ] Long recording: press Stop during diarization; the worker exits within 1 s.
- [ ] Kill `EasyScribe.exe` in Task Manager: no worker process remains.
- [ ] The speaker transcript matches the in-process result for the test WAV.

## 9. Offline

- [ ] Run with the network unplugged; everything works.
- [ ] A firewall log shows zero outbound connections from EasyScribe,
      `whisper-cli.exe`, `ffmpeg.exe` and the diarization worker.
