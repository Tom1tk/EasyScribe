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

## Final audit follow-up (rust.14)

The final audit in `EasyScribe-rewrite-audit2.md` was actioned in commits
`0a5db71`..`09761af` and validated by `v3.2.3-rust.14` (run `42309047043`, from
`09761af`):

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
   1.7 GB unpack on a nearly full disk. See `CLEAN_PC_TEST.md`.
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
   conversion fails; Stop during model loading is visually undone; Best-quality
   runs clear the log; the recordings-folder save error is silent; the
   microphone menu can reopen after recording; speaker rows sort by raw ID
   instead of speaking order; queued file sizes can be stale; legacy install
   candidates are alphabetized. None are data-loss bugs.
9. **Whisper-cli uses narrow `argv`.** A non-ASCII or CJK runtime path can still
   fail in the child's ANSI `main`, because the Windows CRT converts `argv`
   with the active code page. The app now falls back in-process only for a
   start/load failure, not for a decode failure caused by the path. A UTF-8
   manifest/`wmain` helper or an ASCII temp copy is the fix.
10. **Speaker-clip extraction still reads the file prefix** for each speaker
    (`.skip(offset)`), so a 3-hour recording costs one full prefix read per
    speaker even though memory is bounded. `WavReader::seek` should be used.
11. **Checksums are pinned only for sherpa and the GGUF model.** ffmpeg, the
    ONNX models, the Vulkan SDK and the whisper.cpp tag are not hash-pinned;
    cache hits also bypass the two existing checks. This is supply-chain
    hardening, not a functional bug.
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

## Verification

- `cargo test --workspace`: 68 passed (6 app, 60 core, 2 offline), 2 ignored
  model-backed tests.
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
- `v3.2.3-rust.11` failed at the app link when the espeak exclusion was tried;
  the linker proved `piper-phonemize-lexicon.obj` references `espeak_Initialize`.
  The TTS-off source build in `rust.14` supersedes that workaround. `rust.12`
  and `rust.13` validated the payload and CPU paths recorded in
  `REWRITE_STATUS.md`.
