# EasyScribe Rune rewrite: audit 3 (independent code audit)

Nov 6, 2026

## Scope

After actioning every finding in `EasyScribe-rewrite-audit2.md`, six independent
read-only audits were run over the finished Rust/Rune product:

1. Core transcription, transcript formatting, outputs, rename, recovery,
   settings, logging, VAD and ffmpeg.
2. Rune UI state, batch/queue handling, recording flow and dialogs.
3. Microphone capture, live transcription, crash recovery and WAV finalisation.
4. Single-exe payload, extraction cache, offline guard and Windows integration.
5. Diarization child process and hidden worker lifecycle.
6. Windows release workflow, packaging and third-party licensing.

The audits were code reviews against the v3.2.3 Python reference. They were
not a substitute for the Windows-only clean-PC and real-PC tests listed at the
end of this document. This file records what was fixed and what remains.

## Fixed after the audit

### Recording and recovery

- **A second EasyScribe window can no longer delete the first one's temporary
  audio or recover its live recording.** A per-user `fs2` lock
  (`%LOCALAPPDATA%\EasyScribe\.instance.lock`) is taken before payload
  preparation and held for the process lifetime. A second GUI instance shows
  "EasyScribe is already running" and exits.
- **`MicRecorder::start` now fails before opening the microphone if the PCM
  file cannot be created.** The writer opens the file with `create_new`, writes
  the initial JSON sidecar, and only then signals readiness. A full or
  read-only recordings folder no longer produces a silent "Recording" state
  with no file.
- **The JSON sidecar exists from the first second and is updated atomically.**
  Each update is written to a temporary file, `sync_all`ed and renamed. A crash
  or power loss leaves a recoverable pair instead of a PCM without metadata.
- **A writer error no longer discards the valid part of a recording.**
  `stop()` saves the PCM prefix up to `samples_written` and logs the writer
  error; it only fails when nothing was written.
- **Recovery validates metadata and is atomic.** `sample_rate == 0`,
  `channels == 0`, a non-16-bit depth or `samples_written == 0` is rejected
  instead of producing an unusable 1 Hz/empty WAV and deleting the PCM. The WAV
  is written to a temporary file, fsynced and renamed. A missing PCM after a
  successful conversion is treated as success; a locked PCM keeps the sidecar
  for the next attempt.
- **PCM filenames include a random suffix** and are opened with `create_new`,
  so two starts in the same second, or two app instances, cannot append into
  each other's recording.
- **The writer loop uses `recv_timeout` plus the stop flag**, so Stop does not
  wait forever for a sender that cpal still holds. Chunks already queued at
  Stop are drained before the file is closed.
- **Sample formats and configuration are broader.** I8, U8, I24, U24, I32,
  U32, I64, U64, F32 and F64 are converted. Unsupported/DSD formats are
  filtered out before a config is chosen. The configuration prefers native
  16 kHz, then 48 kHz, then the nearest supported rate, mono first.
- **Multi-channel capture averages all channels** instead of using channel 0,
  so a stereo interface with the active mic on the right channel is not
  recorded as silence.

### Transcripts, outputs and speaker results

- **Transcript writes are atomic.** The text is written to a temporary file,
  fsynced, and renamed over the old transcript, so a full disk cannot leave an
  empty/partial file and destroy the previous good transcript.
- **Empty diarization is a valid result** and now matches Python: one unnamed
  `SPEAKER_00` transcript is produced instead of "Find speakers" failing.
- **Batch output names are de-duplicated.** `interview.mp3` and `interview.wav`
  (or same-named files from different folders) get ` (2)`, ` (3)` suffixes
  instead of the second transcript silently overwriting the first.
- **Speaker clips stream the requested frames** instead of allocating the whole
  recording once per speaker. The remaining prefix-read cost is listed below.

### Payload and configuration

- **An incomplete extracted tree is never marked complete.** After unzipping,
  the staging folder is checked against the required runtime file list before
  `.complete` is written. The marker is fsynced.
- **Stale staging is cleaned on cached launches**, but only after one hour, so
  a live extraction in another process is not deleted. The cache-hit path no
  longer needs the extraction lock to do this safely.
- **Newer runtimes are protected.** Cleanup only removes a runtime whose
  `created` timestamp is older than the running one; the newest cache is
  never deleted by an older exe.
- **A disk-space preflight runs before extraction** and reports a clear
  "Not enough free disk space" message instead of a half-extracted runtime.
- **Relative `EASYSCRIBE_BASE_DIR`, `DATA_DIR`, `RUNTIME_DIR` and
  `PAYLOAD_DIR` are anchored to the executable**, not to a shortcut's working
  directory.
- **The declared MSRV is now 1.89**, matching the `Take<File>: Seek` API used
  by the extraction path.

### Release pipeline

- **The Rune GGUF cache has its own key and an explicit save** before the model
  folders are freed, so a Python-release cache cannot restore the file into
  the wrong directory and a warm Rune cache is actually populated.
- **The Vulkan loader is ensured beside `whisper-cli.exe` on every run**,
  including cache hits, and the payload check requires it.
- **The standalone whisper-cli build asserts `GGML_NATIVE=OFF` and
  `GGML_AVX2=ON`**, and the app's CMake gate accepts the multi-config
  `CMAKE_BUILD_TYPE:UNINITIALIZED=Release` form. FMA/F16C are implied by AVX2
  on MSVC.
- **`ffmpeg.exe`/`ffprobe.exe` and the four VC++ runtime DLLs are required and
  executed** in the payload, and `scripts/build-windows.ps1` checks the same
  set.
- **The release tag must match the workspace version** (`v<version>-...`), so
  a `v3.2.4` tag cannot silently ship a 3.2.3 binary.
- **The sherpa TTS/espeak link was investigated.** Removing `espeak-ng.lib` and
  `piper_phonemize.lib` made the Windows link fail on
  `piper-phonemize-lexicon.obj` (`espeak_Initialize`), which proves the linked
  sherpa archive does pull espeak. This finding led to the `rust.14` TTS-off
  source build described below; the prebuilt archive is no longer used.

### Version metadata and dependencies

- **`APP_VERSION`, the Windows resource version, and the packer version are
  derived from the workspace `Cargo.toml`** instead of three hardcoded 3.2.3
  values.
- **The offline dependency test now resolves the Windows target** rather than
  the Linux host, so a Windows-only HTTP crate cannot pass unnoticed.

## Final audit follow-up (rust.14 / v4.0.0)

The final audit in `EasyScribe-rewrite-audit2.md` was actioned in commits
`0a5db71`..`09761af` and validated by `v3.2.3-rust.14` (run `42309047043`, from
`09761af`). The same code is now published as the `v4.0.0` pre-release (run
`42356076024`, from `f9e03a2`); the version change only affects metadata and
the window title. `v4.0.0` is a draft pre-release, and `v3.2.3` remains
`Latest`.

1. **GPL-3.0 `espeak-ng` removed from the Rust build.** The prebuilt sherpa
   archive and `SHERPA_LIB_PATH` are gone; `sherpa-rs-sys` builds vendored
   sherpa-onnx 1.12.9 from source with `SHERPA_ONNX_ENABLE_TTS=OFF` and
   `BUILD_SHARED_LIBS=OFF`. The Windows 11 runner has no `wmic`, so CI and
   `scripts/build-windows.ps1` patch `cmake/show-info.cmake` after `cargo fetch`
   for CMake 4. The build asserts the two cache entries and checks
   `sherpa-onnx-core.lib` for `espeak_Initialize`; the log shows
   `SHERPA_ONNX_ENABLE_TTS:BOOL=OFF`, `BUILD_SHARED_LIBS:BOOL=OFF` and
   "sherpa-onnx has no espeak symbols." Main's Python-release notice now
   discloses the component that remains in v3.2.3 (`a11ca32`).
2. **The slow fallback warns.** `Status::SlowFallback` changes the activity
   title to "Transcribing (slow fallback)" and explains that the built-in
   engine can take many times longer; the log carries a warning. The engine
   stays for damaged-payload resilience, which is the audit's accepted
   "warn, don't run silently" option.
3. **Asset-size guard.** The existing 1900 MiB check is stricter than the
   suggested 2.05 GB and now documents the GitHub 2 GiB per-asset limit.
4. **Live ONNX race fixed.** Every live VAD/recognizer call takes one
   process-wide `ONNX_LOCK`. `stop()` waits for the decoder first with the 60 s
   deadline and detaches both threads only on timeout; `DECODER_ACTIVE` rejects
   a new session until a detached decoder exits. The lock also recovers from
   poisoning.
5. **Release notes and docs corrected.** They state 1.92 GB (1.79 GiB), about
   3.7 GB fresh and about 5.4 GB peak update space.

## Residual risks (not fixed in this pass)

These were confirmed or plausible at the audit-3 pass. Items 3 and 17 were
subsequently fixed in `rust.14` and are kept below with their disposition; the
rest remain open.

1. **Real-PC speed and GPU parity are still unmeasured.** CI has no GPU and four
   slow vCPUs. The branch now measures an independent whisper-cli control and
   the in-process engine on the same runner; the 2x budget is a CI guard, not a
   desktop speed proof.
2. **Windows-only behaviour is unverified**: microphone capture, hot-unplug,
   high-DPI scaling, native dialogs, drag/drop, SmartScreen, and the real
   1.7 GB unpack on a nearly full disk. See `CLEAN_PC_TEST.md`. The first manual
   run proved microphone capture and exposed the WAV-save bug, now fixed; the
   rest of the matrix is still open.
3. **Live mode had a decoder/VAD shutdown race — fixed in rust.14.** If a live
   decode was still running when Stop timed out, a later session could use the
   same sherpa engine concurrently. All live VAD/recognizer calls now share one
   process-wide `ONNX_LOCK`, `stop()` waits for the decoder first, and
   `DECODER_ACTIVE` blocks a new session until a detached decoder exits. The
   file and child-diarization paths were already isolated.
4. **Hot-unplug is logged but not surfaced.** A cpal stream error stops the
   samples reaching the file, yet the timer keeps running and Stop may report
   success for the audio captured before the unplug. The fix needs an error
   slot shared with the recording worker and a device-list refresh.
5. **Live errors are not shown.** A corrupt VAD/decoder model can save a WAV
   with no transcript while the UI says "No speech was heard live".
6. **The live VAD queue is bounded in chunks, not samples**, so unusual cpal
   callback sizes change its effective latency, and overflow is only logged.
7. **Window height is frozen at startup on non-Windows**, and the stored height
   is not updated on resize, so the short-window scroll fallback can be wrong.
8. **Minor UI parity gaps** from the UI audit: "Saved in" uses the last file's
   folder even when it failed; a saved live transcript is discarded if the WAV
   conversion fails; Best-quality runs clear the log; speaker rows sort by raw
   ID instead of speaking order; queued file sizes can be stale; legacy install
   candidates are alphabetized. None are data-loss bugs. Stop during model
   loading, the silent recordings-folder save error and the microphone menu
   reopening after a recording were fixed in the 2026-11-09 rebuild.
9. **Whisper-cli uses narrow `argv`.** A non-ASCII or CJK runtime path can still
   fail in the child's ANSI `main`, because the Windows CRT converts `argv`
   with the active code page. The app now falls back in-process only for a
   start/load failure, not for a decode failure caused by the path. A UTF-8
   manifest/`wmain` helper or an ASCII temp copy is the fix.
10. **Speaker-clip extraction still reads the file prefix** for each speaker
    (`.skip(offset)`), so a 3-hour recording costs one full prefix read per
    speaker even though memory is bounded. `WavReader::seek` should be used.
11. **Checksums are pinned only for the GGUF model and the Cargo-vendored
    sherpa source.** ffmpeg, the ONNX models, the Vulkan SDK and the
    whisper.cpp tag are not hash-pinned; cache hits also bypass the two existing
    checks. This is supply-chain hardening, not a functional bug.
12. **Cache-hit integrity is a marker plus non-zero sizes.** Per-file hashes in
    `manifest.json` are written but not verified on a hit. Bit rot or a partial
    external deletion can pass until a model/DLL fails to load.
13. **Recovery still ignores a lone PCM without JSON** in `scan_for_orphans`.
    New recordings always create the sidecar first, so this only affects files
    from older builds or manual deletion.
14. **Settings writes have no cross-process lock.** The single-instance lock
    removes the GUI race; a worker or a second headless run could still
    last-writer-wins.
15. **Rename validation is incomplete for Windows reserved names** (`CON.txt`)
    and uses ASCII-only case folding, so a case-only rename of an accented name
    is reported as a collision.
16. **Diarization remains single-threaded** in `sherpa-rs` 0.6.8, and the
    crate has unchecked null results and `to_str().unwrap()` on model paths.
    The child process isolates crashes, but the binding should be replaced.
17. **GPL-3.0 `espeak-ng` was statically linked into the Windows exe —
    resolved in rust.14 for the Rust build.** The earlier attempt to remove the
    TTS-only libraries failed at link time because `piper-phonemize-lexicon.obj`
    was pulled and referenced `espeak_Initialize`. sherpa-onnx is now compiled
    from source with `SHERPA_ONNX_ENABLE_TTS=OFF`, and CI asserts that
    `sherpa-onnx-core.lib` has no `espeak_Initialize` symbol. The Python v3.2.3
    release still ships the wheel's espeak-ng; main now discloses it and should
    adopt the same TTS-off build when it is next released.
18. **The release workflow does not reject a re-used existing tag.** It
    validates the tag against the workspace version, but a re-dispatch of an
    existing `rust.N` tag can still attach assets to the old tag. Use a new
    tag for every attempt.

## Real-PC test feedback (2026-11-09)

The owner's first test of the `v4.0.0` preview found three user-visible bugs;
all are fixed in `4bab66e`/`17415fb` and the draft was rebuilt.

1. **Every recording reported "The recording was not saved."** `MicRecorder`
   wrote the PCM correctly, but `recover_pcm_to_wav` reopened the final WAV
   read-only and called `sync_all`. Windows `FlushFileBuffers` requires
   `GENERIC_WRITE`, so the conversion failed in the chosen folder and in the
   work folder; `stop` returned `Ok(None)` and the UI lost the reason. The WAV
   is now reopened with write access. A conversion failure is returned with
   the folder and OS error, and the raw PCM/sidecar stay in
   `%LOCALAPPDATA%\EasyScribe\recordings` for recovery. The next launch converts
   a recording left by the first build.
2. **Show details covered Start recording.** The record view and details card
   both grew, so the record view shrank below its card and the utility row was
   drawn over the Start button. The details card no longer grows on the Record
   tab (Files keeps the growing log), the record view cannot shrink below its
   card, and the Record page scrolls when details need more room.
3. **The microphone menu moved the form down.** It is now an `El::anchored`
   popup at the root, under the unchanged selector placement, with an
   outside-click scrim and Escape handling.

Stop during "Getting ready" is also treated as a cancelled recording instead
of a save error, because the worker now checks the stop token before opening
the microphone.

## v4.0.1 usability follow-up (2026-11-10)

The owner asked for three additions after testing the fixed `v4.0.0` preview:
record Pause/Resume, a blinking caret in the rename box, and the EasyScribe
logo in the Windows title bar and taskbar. All three are implemented in
`1fd3f35` and built as the `v4.0.1` pre-release.

1. **Pause/Resume recording.** A `PauseToken` is shared by the UI, the
   recorder and the worker. Pausing keeps the cpal device stream open and the
   audio callback drops every buffer, so resuming cannot renegotiate the
   device format or fail on a busy microphone. Nothing is appended to the PCM,
   the crash-recovery sidecar, the level meter or the live VAD queue while
   paused. The timer is derived from the samples actually written, so it
   freezes at the same value the final WAV will have. Stop while paused saves
   the pre-pause audio. Stopping at `00:00` removes the empty PCM/sidecar pair
   and reports `Recording cancelled` with `Nothing was saved.` instead of a
   save error. The button is hidden while the stop is saving and pause state
   is cleared when the session finishes.
2. **Blinking rename caret.** Rune draws a solid caret for a focused text
   field and defaults its colour to white, which is invisible on the white
   rename field. The rename field hides that caret (transparent) and the view
   draws an ink-coloured line at `Cx::caret("rename-input")`, blinking with a
   1.06 s two-stop cycle. The rect is in window points, so the line follows
   typing, selection and scrolling. It is suppressed under the Privacy and
   microphone popups. Speaker-name fields keep Rune's caret but set it to ink.
3. **Window and taskbar icon.** `build.rs` already embedded
   `assets/EasyScribe.ico` with `winres` as resource id 1, but winit left the
   window class's generic icon in place. `App::on_window` now loads the big
   and small icons from that resource and sends `WM_SETICON`; the hook also
   runs when the native window is recreated after a GPU release.

Residual risks specific to this follow-up:

- The microphone stream stays open while paused, so Windows can still show the
  microphone-in-use indicator and the device is not released for other apps.
  Closing and reopening was rejected because it can change the format or fail
  on resume.
- Samples already queued for the live VAD decoder before Pause may still
  produce a final phrase after the button is pressed. They are at most the
  pre-pause tail; the saved WAV excludes all paused time.
- The blink is an app-drawn rect, so it has no Windows caret/accessibility
  semantics; it exists only while the rename row is visible and focused.
- The title-bar/taskbar icon cannot be inspected by the headless CI; it needs
  the owner's real-Windows check. The Windows code is compiled in the release
  build, and the icon resource is verified by the embedded resource.
- The fast release gate skips the two in-process whisper.cpp fallback decodes
  because they cost about 11 minutes each on the CI runner. Run the workflow
  with `full_selftest=true` before promotion or after changing the whisper
  backend; the shipping CLI, sherpa, VAD, diarization and silence checks are
  still in the fast gate.

## Verification

- `cargo test --workspace`: 73 passed (9 app, 62 core, 2 offline), 2 ignored
  model-backed tests. The `mic::tests` cases cover the full PCM→WAV stop path
  (which would fail on Windows with the old read-only sync) and stopping before
  any samples; the app tests cover pause/resume, pause-while-saving and clearing
  the pause state after finish.
- `cargo clippy --workspace --all-targets -- -D warnings`: clean.
- `cargo fmt --all -- --check`: clean.
- Release selftest after the final-audit changes, 4 threads, 23.3 s real
  speech: whisper-cli cold 44.4 s, in-process 65.3 s, after sherpa 69.2 s,
  diarization in-process 10.5 s, diarization child 10.6 s, silence 0.1 s with
  zero segments.
- `v3.2.3-rust.14` (run `42309047043`, `09761af`, 4 vCPUs, no GPU) passed:
  control 33.5 s, budget 67 s, CLI child 37.6 s, in-process 551.0 s, sherpa
  ONNX 42.6 s, VAD 0.0 s, after-sherpa 541.1 s, diarization 8.5/8.6 s, silence
  0.1 s. AVX2/FMA/F16C/BMI2 and the golden text were verified. The build proved
  `SHERPA_ONNX_ENABLE_TTS:BOOL=OFF`, `BUILD_SHARED_LIBS:BOOL=OFF` and no
  `espeak_Initialize` symbol. The draft asset is 1,921,914,713 bytes, SHA-256
  `f4e79cfb49a1755a4b362bd4467659c65ae553c309afc3f51d5f1c329bc6969c`. It is a
  draft pre-release, not `Latest`, and the superseded `rust.13` draft was
  deleted.
- `v4.0.0` (run `42621260185`, `17415fb`, 4 vCPUs, no GPU) passed with control
  23.8 s, budget 60 s, CLI child 26.9 s, in-process 338.7 s, sherpa ONNX
  33.9 s, VAD 0.0 s, after-sherpa 339.4 s, diarization 5.1/5.2 s and silence
  0.1 s. The asset is 1,921,949,518 bytes, SHA-256
  `d935714b4b3e09b5d8a3f0717276bdb3526c914734f91c93ed54d5fc79a1c28b`, with an
  89-byte `.sha256`. The first `v4.0.0` asset (run `42356076024`, `f9e03a2`,
  SHA-256 `0b0bc737...`) was replaced by this fixed build. `v4.0.0` was
  superseded by `v4.0.1`, and its draft release and tag were deleted.
- `v4.0.1` (run `42795767098`, `1fd3f35`, 4 vCPUs, no GPU) passed with control
  44.8 s, budget 90 s, CLI child 50.8 s, in-process 439.9 s, sherpa ONNX
  47.0 s, VAD 0.1 s, after sherpa 425.4 s, diarization 7.2/7.3 s and silence
  0.1 s. The native selftest (including the golden text), AVX2/FMA/F16C/BMI2,
  the TTS-off cache and the no-espeak symbol check passed. The asset is
  1,921,993,041 bytes, SHA-256
  `54a45ae030df7bbccee096fe8ef9d138af742e00a0636f50f4f4bb334f34af6e`, with
  an 89-byte `.sha256`. It is published as a pre-release at
  <https://github.com/Tom1tk/EasyScribe/releases/tag/v4.0.1>; `v3.2.3` remains
  `Latest`.
- Release turnaround (2026-11-10): the workflow now restores compiled Rust
  dependencies and reuses the whisper.cpp/sherpa-onnx CMake outputs. Run
  `42815563747` completed in **10.6 min** (build 3.5, selftest 2.1, release
  1.1) versus 33.7 min for `v4.0.1`. An 8-second in-process excerpt was tested
  and discarded: whisper pads to a 30 s window, so each short decode still
  took about 11 minutes. The fast gate keeps the CLI full-fixture golden
  check, sherpa load, VAD, diarization and silence; the two in-process
  fallback decodes require `full_selftest=true`.
- `v3.2.3-rust.11` failed at the app link when the espeak exclusion was tried;
  the linker proved `piper-phonemize-lexicon.obj` references `espeak_Initialize`.
  The TTS-off source build in `rust.14` supersedes that workaround. `rust.12`
  and `rust.13` validated the payload and CPU paths recorded in
  `REWRITE_STATUS.md`.

## 2026-11-12 — follow-up: `EasyScribe-v4-audit.md` actioned

The nine findings in `/home/user/EasyScribe-v4-audit.md` (branch
`rune-rewrite`, base `399b038`) are now implemented; the per-item reasoning and
logs are in `REWRITE_JOURNAL.md` session 19.

1. **Diarization threads.** `local_diarize.rs` wraps the sherpa-onnx C API
   with `num_threads = config::num_threads()` for both models, returns errors
   for non-UTF-8 paths, and checks the native result for null. The child
   reports `took_sec` (native compute only) and the parent feeds that to
   `save_measured_speed`. Thread configuration and non-UTF-8 paths have unit
   tests. The estimate constants stay unchanged until the owner measures the
   fixed 10+ minute file on Windows.
2. **Pre-whisper VAD.** `sherpa_engine::has_speech` exits at the first
   confident speech, logs `Checking for speech…`, and records elapsed time at
   `log::info!`. `whispercli::transcribe_file` now reads the WAV through
   `audio::for_each_wav_block` in 16,384-frame blocks, computes the peak in
   the same pass, and feeds `SpeechScanner`, which retains only a partial
   512-sample VAD window; the whole recording is no longer allocated as
   `f32`. `speech_regions` and the in-memory `has_speech` wrapper remain for
   the in-process path. The selftest now checks 30 s of silence followed by
   the fixture (0.4 s locally) and the full gate transcribes that padded 53 s
   file end to end (83.7 s locally); the existing silence fixture still
   returns no speech.
3. **Worker priority.** `lower_thread_priority` and its three calls are gone.
4. **Startup.** `prepare_with_progress` records its wall time; the app logs
   paths-ready, prepare and first-window times. Cleanup runs on a background
   `PayloadCleanup` thread under the cache lock. `HashingReader` hashes the
   payload while the streaming zip reader extracts it, then drains the central
   directory so the digest covers the whole payload; a mismatch deletes
   staging before the completion marker is written. Unit tests prove the
   one-pass hash and corruption detection. Cold/warm/update and Defender
   numbers need the owner's PC; packaging was not changed.
5. **Resampler.** 48 k → 16 k (and 44.1 k → 16 k) now pass through a 63-tap
   Blackman windowed-sinc low-pass at 7.2 kHz before the linear step. The unit
   test measures more than 40 dB rejection of 10 kHz against 1 kHz. The PCM
   writer channel is bounded (256 blocks) with a drop counter/warning, and the
   live VAD queue logs depth at 75%. Same-speech v3.2.3/v4 recordings still
   need the owner.
6. **Release profile.** `codegen-units = 1`, `incremental = false`; the old
   settings live in `release-fast`. Windows build time and the release
   selftest still need a run.
7. **Slow fallback.** The visible status already said the slow path was in
   use; it now also shows a truncated CLI failure reason in the activity
   detail line (`Reporter::detail` → `Event::StatusDetail`) and keeps the full
   reason in the file log. Renaming `whisper-cli.exe` in a test runtime is on
   the owner's checklist.
8. **Repetition loop.** The app normalises sentences, flags more than 4
   identical long repeats (more than 8 short interjections), re-decodes only
   that padded range on CPU with `-ng -ot -d -mc 0 -et 2.8`, and splices the
   absolute-offset result back. Failure or a still-repeating repair keeps the
   original. Unit tests cover the detector, normalisation and a fake CLI
   repair; the selftest fails if the golden decode loops. The A–F GPU matrix
   and three clean GPU runs of the real recording remain required.
9. **Smaller items.** UI log capped at 2,000 lines; `VecDeque` stderr tail;
   `LazyLock` ffmpeg regexes; one ffprobe per input; clear errors for
   non-UTF-8 whisper/VAD paths; live VAD high-water logging. The diarization
   child still inherits `EASYSCRIBE_NUM_THREADS`.

**Verification (Linux, 8 logical CPUs, CPU-only).** `cargo fmt --all --
--check`, `cargo clippy --workspace --all-targets -- -D warnings` and
`cargo test --workspace` passed (10 app + 74 core + 2 offline; 4 ignored:
2 model-backed, 2 manual benchmarks). After the block-streaming change the
full debug selftest passed: cold CLI 43.9 s, in-process 68.4 s, sherpa 8.7 s,
speech gate 0.3 s, padded 30 s-silence transcription 83.7 s, decode after
sherpa 68.7 s, diarization 3.6/4.0 s, silence 0.1 s. The fast debug selftest
passed (CLI 44.6 s, gate 0.4 s, diarization 3.7/4.0 s, silence 0.1 s). The
post-streaming release build took 2m43s incrementally and its selftest passed
(CLI 43.7 s, gate 0.3 s, diarization 3.6/3.8 s, silence 0.1 s). A local
long-file benchmark (file kept outside the repository) showed the expected
multi-thread speedup with identical turn output, and the old full-file VAD
pass became an early-exit gate that returns in milliseconds. CI run
`43221068673` (history cleanup), run `43221226305` (final docs) and run
`43221786609` (pre-streaming docs) passed Linux tests and the Windows
`cargo check`; the post-streaming run is recorded after the push.

**These are not Windows GPU or real-PC gate results.** The owner still needs
to run the clean-PC checklist, the fixed-file before/after comparison and the
GPU matrix before promotion; `v3.2.3` remains `Latest`.
