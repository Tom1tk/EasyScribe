# EasyScribe Rune Rewrite — Status

**Updated:** 2026-11-12 (session 21 — Windows audit results and logging follow-up)
**Branch:** `rune-rewrite` · **Worktree:** `/home/user/EasyScribe-rune`
**Base:** `main` @ `8323516` (EasyScribe v3.2.3)
**Version:** `4.0.2` · **Release:** `v4.0.1` pre-release and `v4.0.2` draft pre-release, not `Latest`
**Phase:** 7 — speed, GPU, process isolation and final release
**Overall:** implementation complete. Audit 2's findings are actioned: the
Vulkan `whisper-cli` is back as a child process with CPU retry, diarization is
isolated in a child, flash attention and the no-speech gate are in, and the
selftest measures real speech. The audit-2 follow-up blockers (missing tracked
speech fixture, `Win32_Security`, Linux sherpa link, `whisper-cli` job object,
missing redistributable notices) are fixed. `rust.7` failed in the Vulkan loader
copy, `rust.8` in a multi-config CMake assertion, `rust.9` in a wrong
MSVC-only FMA assertion, `rust.10` on the app's own
`CMAKE_BUILD_TYPE:UNINITIALIZED=Release` cache form, `rust.11` when the
TTS-only sherpa libraries were removed, and `rust.12` because the payload's
expected `whisper-cli` empty-path exit code failed the verification step. That
last attempt proved espeak-ng is genuinely pulled by
`piper-phonemize-lexicon.obj`. The final audit's remedy is now implemented:
sherpa-onnx 1.12.9 is built from source with text-to-speech disabled, so the
GPL-3.0 espeak-ng/piper components are no longer linked. Six independent audits
of the finished product are recorded in `docs/rewrite/AUDIT_3.md`; their
confirmed data-loss, recovery, payload and release-gate fixes are in `d59b8f2`.
The remaining final-audit findings are also actioned: the slow in-process
fallback shows a visible warning, the live-mode ONNX race is serialised
process-wide, the asset-size guard is documented, and the release notes use the
measured 1.92 GB. The final-audit fixes passed as `v3.2.3-rust.14` (run
`42309047043`). The owner's first real-PC test of the `v4.0.0` preview found
three issues: recordings reported "The recording was not saved", Show details
covered Start recording, and the microphone menu pushed the form down. All
three were fixed and rebuilt as `v4.0.0` (run `42621260185`). The owner then
asked for a recording Pause/Resume control, a blinking caret in the rename box
and the EasyScribe logo in the Windows title bar/taskbar. Those are implemented
as `4.0.1` and built in run `42795767098` from `1fd3f35`; the new pre-release
asset is recorded below. `v4.0.0` was superseded. The 2026-11-12 v4
speed/bug audit is actioned in code (see "Audit dispositions"); the only open
gate is the real-PC clean-PC/speed/GPU/microphone checklist and the audit's
same-file/GPU measurements. The first partial Windows run reproduced the
repetition on the RTX 5070 and showed prompt carry-over as the cause; it also
found two logging gaps, both fixed below. `v3.2.3` remains `Latest`. The
Windows agent should start with `docs/rewrite/WINDOWS_AUDIT_HANDOFF.md`, which
records the fresh-build command, every remaining measurement, the A–F matrix,
pass criteria and the results template.

## What is built

- [x] **`easyscribe-core`** — config/paths, errors, cancellation and reporting,
      transcript formatting, rename/recovery/settings/outputs, rotating logs,
      offline guard, ffmpeg/ffprobe with `-protocol_whitelist file`, WAV/VAD
      helpers, sherpa-onnx VAD/Whisper/diarization, whisper.cpp through
      `whisper-rs`, file transcription orchestration, microphone capture with
      raw PCM + JSON sidecar, live transcription, and the single-exe payload.
- [x] **`easyscribe`** — complete Rune UI: Files, Record, activity/results,
      speaker naming, Find speakers, rename, Privacy, details, microphone and
      folder pickers, drag/drop, keyboard focus, first-run splash.
- [x] **`easyscribe-pack`** — streaming ZIP payload builder, 128-byte footer,
      SHA-256 verification, cache keys, extraction lock and cache GC.
- [x] **Windows resources** — icon, version info, per-monitor DPI manifest,
      GUI subsystem for release builds, AppUserModelID.
- [x] **Build/release** — `scripts/build-windows.ps1` and
      `.github/workflows/rune-release.yml` build one self-contained
      `EasyScribe.exe` and publish a pre-release with a `.sha256`.
- [x] **Docs** — README, third-party notices, packaging/UI/backend references.

## Verification completed

_This section records the v4.0.1 release-gate evidence. The current audit
follow-up counts, timings and open Windows checks are in "Audit dispositions"
and "Next actions" below._

- [x] `cargo test --workspace`: **9 app + 62 core + 2 offline = 73 passed**,
      2 ignored (the model-backed long-silence and golden tests, run manually).
- [x] `cargo clippy --workspace --all-targets`: clean.
- [x] `cargo fmt --all -- --check`: clean.
- [x] Real-engine golden: both the in-process engine and the bundled
      `whisper-cli` produce the expected plain-text transcript for
      `assets/selftest_speech.wav` (23.3 s, generated test speech). The clip is
      tracked outside `tests/data/` so CI has it without force-adding an ignored
      `.wav`.
- [x] Long-silence gate test: 40 s of added silence before and after the speech
      fixture still produces the `quarterly results` transcript.
- [x] `--selftest` prints `whisper_print_system_info()` (AVX2/FMA/F16C all 1),
      the Cargo/CMake profile, per-phase whisper timings, and loads the sherpa
      Whisper ONNX model, Silero VAD, whisper.cpp GGUF and diarization models.
- [x] Local release selftest, 4 threads, 23.3 s speech, after the final-audit
      changes: `whisper-cli` cold **44.4 s**, in-process cold **65.3 s**, after
      sherpa **69.2 s**, diarization in-process **10.5 s**, diarization child
      **10.6 s**, silence check **0.1 s** (zero segments, "No speech found").
- [x] Windows CI `v4.0.0` (run `42621260185`, `17415fb`, 4 vCPUs, no GPU):
      the standalone `whisper-cli` control decoded the 23.3 s fixture in
      **23.8 s** and set a **60 s** speech budget (2×, minimum 60 s). The
      bundled CLI child decoded it in **26.9 s** (inside budget), the
      in-process engine **338.7 s**, the sherpa Whisper ONNX model **33.9 s**,
      Silero VAD **0.0 s**, the post-sherpa in-process decode **339.4 s**,
      in-process diarization **5.1 s**, child diarization **5.2 s**, and the
      silence check **0.1 s** with zero segments. System info reported
      AVX2/FMA/F16C/BMI2; the golden text matched on every decode; the build log
      asserted `SHERPA_ONNX_ENABLE_TTS:BOOL=OFF`,
      `BUILD_SHARED_LIBS:BOOL=OFF` and no `espeak_Initialize` symbol; imports,
      subsystem, extraction, cached launch and the draft assets were verified.
      The draft is `v4.0.0` (draft, pre-release, not `Latest`):
      `EasyScribe-v4.0.0.exe` is 1,921,949,518 bytes (1,832.9 MiB, 1.79 GiB),
      SHA-256
      `d935714b4b3e09b5d8a3f0717276bdb3526c914734f91c93ed54d5fc79a1c28b`,
      with an 89-byte `.sha256` asset. The code had previously passed as
      `v3.2.3-rust.14` (run `42309047043`, `09761af`); the `rust.13` and
      `rust.14` drafts were superseded and deleted. This build is superseded
      by `v4.0.1`.
- [x] Windows CI `v4.0.1` (run `42795767098`, `1fd3f35`, 4 vCPUs, no GPU):
      the standalone control decoded the 23.3 s fixture in **44.8 s** and set
      a **90 s** budget (2×). The bundled CLI child decoded in **50.8 s**
      (inside budget), the in-process engine **439.9 s**, the sherpa Whisper
      ONNX model **47.0 s**, Silero VAD **0.1 s**, the post-sherpa in-process
      decode **425.4 s**, in-process diarization **7.2 s**, child diarization
      **7.3 s**, and silence **0.1 s** with zero segments. System info
      reported AVX2/FMA/F16C/BMI2, the native selftest (including the golden
      text) passed, and the build asserted `SHERPA_ONNX_ENABLE_TTS:BOOL=OFF`,
      `BUILD_SHARED_LIBS:BOOL=OFF` and no `espeak_Initialize` symbol. Imports,
      GUI subsystem, extraction, cached launch and the assets were verified.
      `EasyScribe-v4.0.1.exe` is 1,921,993,041 bytes (1,832.9 MiB, 1.79 GiB),
      SHA-256
      `54a45ae030df7bbccee096fe8ef9d138af742e00a0636f50f4f4bb334f34af6e`,
      with an 89-byte `.sha256`. It is a **published pre-release**, not
      `Latest`; `v3.2.3` remains `Latest`, and the superseded `v4.0.0` draft
      and tag were deleted.
- [x] Final-audit fixes verified in `v4.0.0`: TTS-off source build (no
      espeak-ng/piper), the 1900 MiB size guard, the 1.92 GB release notes, the
      `Transcribing (slow fallback)` warning, and the process-wide live ONNX
      lock with decoder-first Stop.
- [x] **CI turnaround (2026-11-10):** the release workflow now keeps
      `target/` for the Rust cache and reuses the whisper.cpp/sherpa-onnx CMake
      outputs instead of deleting and rebuilding them. The normal release gate
      runs the bundled `whisper-cli` on the full fixture, the sherpa model
      load, VAD, both diarization paths and the silence check; the two
      in-process fallback decodes are opt-in with `full_selftest=true`.
      Run `42815563747` measured **10.6 min end to end** (build 3.5 min,
      selftest 2.1 min, release 1.1 min), down from **33.7 min** for `v4.0.1`
      (build 11.3 min, selftest 16.6 min). An 8-second excerpt was tried first
      and still took 680 s because whisper pads to a 30 s encoder window, so it
      was discarded. The published `v4.0.1` asset predates these workflow
      changes; the next build uses them.
- [x] First real-PC test fixes: `recover_pcm_to_wav` now reopens the finished
      WAV for writing before `sync_all` (Windows `FlushFileBuffers` rejects a
      read-only handle, which made every recording report "not saved"); the
      real conversion error is shown and the raw PCM is kept for recovery; Stop
      during "Getting ready" is treated as a cancelled recording; the Record
      details panel no longer shrinks the record view under the utility row and
      scrolls on short windows; the microphone menu is an anchored popup with
      an outside-click scrim and Escape. A regression test covers the full
      PCM→WAV stop path.
- [x] All 18 headless scenes render (the 13 original states plus record-paused,
      record-details, record-live-details, recording-details and
      record-mic-menu).
- [x] UI click/state tests pass, including pause/resume, pause-while-saving,
      the cleared pause state after finish, the cancelled-recording path and
      stopping before any audio was captured.
- [x] v4.0.1 UI additions: Pause/Resume beside Stop with a frozen timer and
      zero level meter; amber `Recording paused` activity and info banner;
      blinking ink caret in the rename box; visible caret in speaker-name
      fields; embedded window icon installed with `WM_SETICON` after native
      creation so the title bar and taskbar use the logo.
- [x] Packed first-run smoke with a 300 MB fake payload: extraction, marker,
      adoption, main window, and cached quick launch.
- [x] Payload ZIP stores >8 MB entries without a compression level; a 9 MB
      regression test covers the previous real-payload failure.
- [x] Flock-based cache lock, stale staging cleanup, incomplete-runtime
      replacement, and signed-exe footer scanning.
- [x] `v3.2.3-rust.5` built from the updated workflow with the GGUF SHA-256
      gate and asserted `GGML_NATIVE=OFF`, `GGML_AVX=ON`, `GGML_AVX2=ON`,
      `CMAKE_BUILD_TYPE=Release`; both selftest decodes passed.
- [x] Audit 3: six independent audits of the finished product, recorded in
      `docs/rewrite/AUDIT_3.md`. Confirmed fixes include the single-instance
      lock, PCM writer handshake and atomic sidecar, partial-recording save,
      sample-format/config coverage, multi-channel averaging, atomic transcript
      writes, empty-diarization parity, batch output de-duplication, payload
      staging validation and marker fsync, stale-staging age gate, newer-runtime
      protection, relative env paths, MSRV 1.89, version metadata derived from
      Cargo, the Rune-only GGUF cache save, the unconditional Vulkan loader, the
      multi-config CMake gate, and (in `rust.14`) the TTS-off sherpa source
      build that removes espeak-ng.
- [x] Audit 3 residual risks are listed in `docs/rewrite/AUDIT_3.md`. The
      unverified Windows items (real-PC speed, microphone, DPI, dialogs,
      drag/drop, SmartScreen, low-disk unpack) remain gated by
      `docs/rewrite/CLEAN_PC_TEST.md`.

## Functional/packaging decisions

| Decision | Reason |
|---|---|
| Full-size updates only | One exe is the product; no delta bundle or installer. |
| Extract once to `%LOCALAPPDATA%\EasyScribe\runtime\v1-<hash>` | whisper.cpp/ffmpeg need real files; later launches reuse the cache. |
| User data separate from runtime | Settings, logs, temp and recovery survive version updates. |
| Old v3 shortcuts not repointed | Avoid touching user data without consent; portable exe runs anywhere. |
| Vulkan `whisper-cli` child process is the primary file engine | Reuses the proven v3.2.3 binary and flags, gives real GPU acceleration, and a driver crash only kills the child. The in-process `whisper-rs` build remains the fallback. |
| Bundled VC++/OpenMP redistributable DLLs beside `whisper-cli.exe` | The Vulkan build uses the dynamic MSVC runtime and OpenMP; a clean PC may not have `vcomp140.dll`. The DLLs live in the payload, not next to the outer exe. |
| Diarization runs in a child process of the same exe | Stop is a process kill, a crash cannot take the UI down, and long recordings free their memory when the worker exits. A Windows job object kills workers if the app is killed. |
| sherpa-onnx **1.12.9**, built from source with TTS off | `sherpa-rs` 0.6.8 binds the 1.12.9 C API; 1.13.x added struct fields and is ABI-incompatible. `SHERPA_ONNX_ENABLE_TTS=OFF` removes the GPL-3.0 espeak-ng/piper components. |
| Static MSVC runtime | The workflow sets `+crt-static` and `CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded`; the build asserts `BUILD_SHARED_LIBS=OFF`. |
| Explicit `advapi32.lib` on Windows | onnxruntime's ETW/telemetry calls leave undefined imports in `sherpa-rs`'s cdylib link. The build adds `-C link-arg=advapi32.lib` alongside the static CRT flags. |
| Naming panel takes over at 820 pt and below | Python clips Save/Skip at the default height; the Rust build hides the Files/Record content while naming so both actions stay reachable. Intentional usability deviation. |
| In-process fallback + child engines | File transcription prefers the `whisper-cli` child and falls back to in-process `whisper-rs` when only the GGUF model exists; diarization is always a child process. |
| Privacy as an in-app modal | Same content/actions as the Python window; documented visual deviation. |
| Flash attention on | Matches whisper-cli's default and the Python release; smaller encode buffer and faster CPU decode. |
| VAD gate on every file | Short files still decode whole when speech exists, preserving the golden transcript; a file with no speech returns an empty transcript instead of "Thank you." |
| Real 23.3 s speech selftest with a golden text and a time budget | The old 5 s silence input was the decoder's worst case and hid the actual cause of the 655 s CI time. |

## Audit dispositions

- **Functional audit** — all high/medium findings fixed: microphone-open
  deadlock, disabled sample playback, stale rename stem, recovery folder,
  settings merge, bundled-ffmpeg validation, case-only rename, ffprobe timeout,
  live transcript fallback, half-even rounding, missing log lines, speech-gate
  cancellation, recovery deletion order, silent-live message, offline guard.
  Long-silence gate regression added.
- **Code-quality audit** — fixed signed-exe footer discovery, close cancels
  batch/find workers, worker-spawn failures report terminal events, live-stop
  deadline, PCM writer errors, atomic/merged settings, recovery folder, case-only
  rename, second rename, cache completeness, missing bundled ffmpeg, truncated
  footer handling, footer validation, manifest offset, cached file sizes,
  8/24-bit WAV scaling, offline install, GUI subsystem, `rodio` WAV decoder,
  and worker-thread isolation for whisper.cpp after sherpa use.
- **Packaging audit** — switched to sherpa 1.12.9, removed the load-time Vulkan
  import, added static CRT, GUI subsystem, native `--selftest`, full runtime
  file checks, import/subsystem checks, cached-launch check and SHA-256 upload.
- **Windows selftest failure and diagnostic hardening** — `v3.2.3-rust.3`
  passed the static link, payload pack, import scan and GUI check, but the
  packed selftest failed in whisper.cpp after 10 m 49 s
  (`whisper_full_with_state: failed to encode`, error -6) while the same model
  and decode pass locally. The selftest now decodes once **before** loading any
  sherpa/ONNX model and once after, with timings in the output, so the next run
  separates an ONNX-interference cause from a Windows whisper.cpp build cause.
  `EASYSCRIBE_WHISPER_TRACE=1` adds model-load, region, thread and abort-callback
  diagnostics. The Windows job now pins and asserts `GGML_AVX`/`GGML_AVX2`,
  wipes cached whisper build directories before compiling, verifies the GGUF
  SHA-256 and bumps the Rust cache key.
- **whisper.cpp Stop fixed (and a likely encode-failure cause removed).**
  whisper-rs 0.16's `set_abort_callback_safe` boxes the closure twice and then
  calls it through the wrong type, so the abort callback reads memory that is
  not the cancel token. A Linux reproduction reads false even for a cancelled
  token; on another layout the same undefined read can produce `true`, which
  aborts the encoder at its final check and yields exactly
  `whisper_full_with_state: failed to encode`. `whispercpp::decode_regions` now
  installs a plain C abort callback with a live `AbortProbe`, restoring
  cooperative Stop and removing the undefined read.
- **`v3.2.3-rust.4` draft passed the native selftest.** The cold whisper decode
  completed in 655.4 s and the post-sherpa decode in 633.6 s — the same ~650 s
  encode that used to fail, now succeeding. That confirms the whisper-rs
  abort-callback read as the Windows-only failure. The run used the older
  workflow file from `main`, so the AVX2 pin/assert, GGUF hash check and trace
  environment did not execute; `main` now carries those changes and
  `v3.2.3-rust.5` is rebuilding with an asserted AVX2 CMake cache.
- **`v3.2.3-rust.5` — workflow gate confirmed.** The run was dispatched from
  `main` but used the updated workflow file (GGUF hash, AVX2 pins and CMake
  assertions all ran). It still took 661.2 s cold and 635.8 s after sherpa for
  5 s of silence. The cause is **still unproven**: the local benchmark in audit 2
  found OpenMP on/off made no difference (38.7 s vs 38.5 s encode), so the
  earlier OpenMP explanation is not supported. `rust.6` was meant to compare the
  independent `whisper-cli` control with the in-process engine on the same
  runner, but it failed before building. `rust.7` now carries that comparison;
  keep the cause open until both numbers exist.
- **`v3.2.3-rust.6` failed before building.** `test-linux` could not link
  `-lonnxruntime`/`-lsherpa-onnx-c-api` because `rust-cache` restored the
  sherpa build-script output but not the download cache it points at.
  `check-windows` failed on `CreateJobObjectW` because `windows-sys` needs
  `Win32_Security`. The release job then failed at the control step with
  `Cannot find path ... tests\data\selftest_speech.wav`, which `.gitignore`
  excludes. All three are fixed in the follow-up commit.
- **`v3.2.3-rust.7` failed at the standalone whisper-cli build.** The new
  `vulkan-1.dll` copy expected the loader under `VULKAN_SDK\Bin`; the SDK
  installs it through `VulkanRT.exe` into `System32`. The build step now copies
  from `System32` first, falls back to `Bin`, and asserts the CLI's
  `GGML_AVX2`/`GGML_FMA`/`GGML_F16C` cache entries. The budget is capped at
  180 s.
- **`v3.2.3-rust.8` failed the same build step on a bad assertion.** The CMake
  generator is multi-config, so `CMAKE_BUILD_TYPE` is empty; asserting
  `CMAKE_BUILD_TYPE=Release` could never pass. The Release configuration is now
  proved by the `bin\Release` output and the `--config Release` build.
- **`v3.2.3-rust.9` failed on an MSVC-invalid FMA assertion.** ggml's CMake
  defines `GGML_FMA`/`GGML_F16C` only for non-MSVC builds; on MSVC they are
  implied by AVX2. The assertion now checks `GGML_NATIVE=OFF` and
  `GGML_AVX2=ON` only, and the CMake command passes `-DGGML_AVX2=ON`.
- **Diarization audit fixes.** The parent now treats the first `turns` message
  as success and stops the worker without waiting for teardown, so a destructor
  crash cannot discard a valid result; the worker leaks its native engine after
  computing the result for the same reason. Reader threads read bytes with
  `from_utf8_lossy`, report I/O errors instead of silently stopping, and use
  bounded kill/reap. `EASYSCRIBE_DIARIZE_TIMEOUT_SEC` gives the selftest a
  deadline. `min_duration_on/off` now match the Python 0.3/0.5 defaults, the
  unused `--threads` plumbing is gone, and the worker parses `args_os` so a
  non-Unicode path cannot panic. Speaker clips are streamed from the WAV instead
  of loading the whole recording once per speaker.
- **Release-pipeline audit fixes.** The GGUF cache has a Rune-only key and is
  explicitly saved before cleanup; the Vulkan loader is ensured beside
  `whisper-cli.exe` on cache hits as well as cold builds; the standalone CLI
  asserts `GGML_AVX2` (not MSVC-only FMA); `ffmpeg`/`ffprobe` are checked in the
  payload.
- **Whisper-cli engine audit fixes.** A spawn or DLL-load failure
  (`0xC0000135`/`0139`/`007B`/`0142`) now falls back directly to the linked
  in-process engine instead of recursing through the dispatcher; the selftest
  sets `EASYSCRIBE_SELFTEST_STRICT_CLI` so the release path still fails if the
  shipped CLI cannot start. Any GPU failure retries the CPU once, but only a
  Vulkan marker or GPU crash code (`0xC0000005`/`0409`/`001D`) is remembered as
  a failed GPU; CPU runs set `GGML_DISABLE_VULKAN=1` so a broken driver cannot
  crash during backend enumeration. The probe now distinguishes “failed” from
  “no GPU”, lets whisper-cli choose on failure, drains stderr while polling,
  honours Cancel, and is skipped when the CPU is forced. The disabled GPU is
  filtered during selection, the preferred index is saved only after a
  successful decode, the WAV buffer is dropped before the child starts, a
  non-silent file with empty VAD bounds is decoded instead of returning an empty
  transcript, stderr survives invalid UTF-8, and all temp JSON/stem files are
  removed. Loader DLLs and notices are now part of the runtime completeness
  check, and `ffmpeg`/`ffprobe` join the kill-on-close job object.
- **Selftest audit fixes.** The CLI failure path no longer recurses through
  `whispercpp::transcribe_file`; the selftest calls `whispercli::transcribe_file`
  directly when the bundled CLI exists and now fails if the speech fixture or
  golden text is missing. The post-sherpa step is back on
  `transcribe_in_process` (the ONNX-interference diagnostic), empty diarization
  turns are errors, VAD failure is reported to the user, the 30 s gap test no
  longer loses its padding, `wav_seconds` no longer divides stereo frames twice,
  and the failure summary is printed as well as logged.
- **Audit 2 follow-up fixes.** Moved the generated fixture to tracked
  `assets/selftest_speech.*` (payload copy is automatic); added
  `Win32_Security`; cached `~/.cache/sherpa-rs` and forced a fresh
  `sherpa-rs-sys` build script on Linux; moved `win_job` into a shared module
  and assigned `whisper-cli` to it; removed `current_dir` so test/development
  binary paths work; cached the Vulkan probe; made the progress regex tolerate
  spaces; killed a worker on a malformed message; and shipped `vulkan-1.dll`
  plus the VC++/Vulkan notices. The GPU path still retries `-ng`, and a CLI
  spawn failure now falls back to the in-process engine instead of a hard file
  error.
- **Audit 2 actioned.** File transcription now runs the bundled Vulkan
  `whisper-cli` as a child process (`whispercli.rs`), probing devices, choosing
  a dedicated GPU, persisting the choice, and retrying `-ng` on failure; the
  in-process engine remains the fallback. Diarization runs in a hidden child
  (`--worker diarize`) with JSON-lines progress/turns, kill-based Stop and a
  Windows kill-on-close job object. Flash attention is on, empty VAD bounds mean
  "No speech found", short files still decode whole, and silence no longer
  reaches the decoder. The selftest uses a 23.3 s real-speech fixture, checks
  the golden text, prints system info/build profile/per-phase timings, and
  enforces a speech budget derived from an independent whisper-cli control on
  the same CI runner. `main`'s copy of the workflow is now a stub that refuses
  to run without `--ref rune-rewrite`.
- **UI audit** — all confirmed issues fixed: the real window height now reaches
  `AppState`; the 600–678 pt page scrolls so the utility row and tall results
  stay reachable; record tiles and microphone menu rows are keyboard-focusable
  radios/options; unsupported drops produce the `Skipped …` note; the
  timestamps/speaker labels toggle their options; Privacy traps focus; file
  rows no longer show `Waiting` before a batch; folded Files/Record summaries
  keep their step titles; `Use default` is a teal link; the file count is 13 pt
  muted; recording-saved details shorten the folder to 60 characters; the
  timer stays coral while saving; the level meter has no 1 px dot at zero; a
  fatal batch keeps a full green bar when an earlier file saved; and the
  speaker-naming panel focuses its first entry.

- **v4 speed/bug audit (2026-11-12).** All nine findings are actioned in
  code. Diarization calls both native models with `config::num_threads()`;
  the child reports native compute time for the per-PC estimate. The
  pre-whisper speech check is now an early-exit `has_speech` gate with a
  visible status line and timing log; the WAV is streamed in 16,384-frame
  blocks and the peak is computed in the same pass, so the gate no longer
  holds the whole recording as `f32`. Worker priority lowering is removed.
  Startup logs process/paths/prepare/first-window times, cleans old runtimes
  on a background thread, and hashes the payload while extracting it instead
  of reading the archive twice. Microphone downsampling has a 63-tap
  anti-alias filter (tested at >40 dB rejection), a bounded PCM queue and
  drop/VAD-depth logging. Release uses `codegen-units = 1`/`incremental =
  false`, with `release-fast` retained for smoke builds. The slow fallback
  shows its failure reason in the activity detail line. Repeated sentences
  are detected and only the affected range is re-decoded on CPU (`-ng`,
  `-ot`/`-d`, `-mc 0`, `-et 2.8`) and spliced back, with the original kept
  on failure; the selftest fails on a decoder loop. The UI log is capped at
  2,000 lines, stderr uses `VecDeque`, ffmpeg regexes are cached, WAV probing
  happens once, non-UTF-8 model paths error clearly, and live queue depth is
  logged. Local Linux verification: fmt/clippy clean; the post-streaming full
  debug selftest passed (CLI cold 43.9 s, in-process 68.4 s, after-sherpa
  68.7 s, diarization 3.6/4.0 s); the 30 s-silence gate check took 0.3 s and
  the padded 53 s file transcribed end to end in 83.7 s. The fast debug
  selftest passed (CLI 44.6 s, gate 0.4 s, diarization 3.7/4.0 s), and the
  post-streaming release build (2m43s incremental) and selftest passed (CLI
  43.7 s, diarization 3.6/3.8 s). A local long-file benchmark (file kept
  outside the repository) showed the expected multi-thread speedup with
  identical turn output, and the full-file VAD pass was replaced by an
  early-exit gate that returns in milliseconds. GitHub Actions run
  `43221786609` passed the Linux tests and Windows `cargo check`; post-streaming
  runs `43224636495` and `43224901423` also passed both jobs. Real Windows
  cold/warm/update timings, the GPU A–F matrix and the clean-PC checks remain
  open.

## Known limitations / residual risk

- Windows GPU device selection, Vulkan failure/CPU retry, DPI, native dialogs,
  drag/drop and a real microphone can only be proven on a real Windows PC or VM.
  `v4.0.1` exercises the CPU path on CI (the runner has no GPU), but not a
  dedicated/integrated GPU or a crashing driver. The owner's first manual test
  found the recording-save, details-layout and microphone-menu bugs now fixed
  here; the v4.0.1 pause/resume, caret and window-icon changes also need the
  owner's real-Windows check. The full clean-PC checklist still needs a rerun.
- The real-PC speed gate from audit 2 and the v4 audit is still open:
  compare the same 10+ minute file on v3.2.3 GPU, v3.2.3 CPU, and the current
  `v4.0.2` build (with and without speakers), then record wall times in this
  file. The first Windows pass already recorded cold/warm/update launch and
  the A–F GPU matrix; the fixed-file comparison, diarization threads and the
  remaining checks are still open. CI has no GPU and its 4 vCPUs are much
  slower than a normal desktop; do not tune from the 23 s fixture.
- The old 655 s CI decode was the in-process engine, not the shipped CLI path.
  `v4.0.1` confirms the CLI child decodes the fixture in 50.8 s against a 44.8 s
  control (the runner was slower than the 23.8 s v4.0.0 run); the in-process
  fallback took 439.9 s and remains around 19× real time. Its exact cause is
  still unproven, so the UI warns instead of running silently.
- The in-process fallback is slower than `whisper-cli` on Windows; it is used
  when the payload has no `whisper-cli` or when the CLI cannot start. The
  shipping primary path is the child process. When it is active, the title
  reads **Transcribing (slow fallback)**, the activity detail line shows a
  short CLI failure reason, and the file log records why the fallback ran.
- The download is about 1.92 GB and every update is full-size. A fresh install
  uses about 3.7 GB (exe plus ~1.7 GB runtime); an update needs up to about
  5.4 GB free temporarily while the old and new runtimes coexist.
- A real ~1.8 GB payload unpack on a nearly full disk still needs a clean-PC
  test; the local fake-payload smoke validates mechanics only.
- Authenticode signing is not enabled. The footer parser tolerates a certificate
  appended after the footer, and the release publishes a SHA-256 file.
- Audit 3 lists the remaining hot-unplug, live-error, non-ASCII path and UI
  parity risks in `docs/rewrite/AUDIT_3.md`. The live decoder/VAD race is fixed
  since `v4.0.0`; hot-unplug reporting is now the highest-risk item not fixed.
- Pause keeps the microphone stream open by design, so Windows can still show
  the microphone-in-use indicator and another app cannot take the device until
  Stop. Samples already queued before Pause may still add one pre-pause phrase
  to the live transcript; the saved WAV excludes all paused time.
- The window icon and blinking caret are compiled into the release but cannot
  be inspected by headless CI; both are on the owner's real-Windows checklist.
- The fast release gate does not run the in-process whisper.cpp fallback on the
  full fixture. Dispatch with `full_selftest=true` after a whisper/backend
  change or before promoting a build; the CLI child, sherpa load, VAD,
  diarization and silence checks remain in the fast gate.

## Next actions

A Windows agent should start with `docs/rewrite/WINDOWS_AUDIT_HANDOFF.md`; the
list below is the summary.

1. **Done.** `v4.0.1` from `1fd3f35` passed the full Windows pipeline
   (run `42795767098`): control 44.8 s, budget 90 s, CLI child 50.8 s,
   in-process 439.9 s, after-sherpa 425.4 s, sherpa ONNX 47.0 s, VAD 0.1 s,
   diarization 7.2/7.3 s, silence 0.1 s; AVX2/FMA/F16C/BMI2, the native
   selftest/golden text, the TTS-off CMake cache and the no-espeak symbol
   check were verified.
2. **Done.** `EasyScribe-v4.0.1.exe` (1,921,993,041 bytes) and its 89-byte
   `.sha256`
   (`54a45ae030df7bbccee096fe8ef9d138af742e00a0636f50f4f4bb334f34af6e`)
   are published as a pre-release at
   <https://github.com/Tom1tk/EasyScribe/releases/tag/v4.0.1>. `v3.2.3`
   remains the stable `Latest` release; the superseded `v4.0.0` draft and tag
   were deleted.
3. **Done.** The GPL-3.0 espeak-ng question is resolved for the Rust build:
   sherpa-onnx is compiled from source with `SHERPA_ONNX_ENABLE_TTS=OFF`, and
   the workflow proves the library has no `espeak_Initialize` symbol. Main's
   Python-release notice now discloses the component that remains in v3.2.3
   (`a11ca32`).
4. Rerun the clean-PC script in `docs/rewrite/CLEAN_PC_TEST.md` (unpack,
   microphone including pause/resume, zero-audio cancel and the save/recovery
   checks, the rename caret and window/taskbar icon, DPI, dialogs/drag-drop,
   SmartScreen, GPU matrix, offline) and the real-PC speed comparison; record
   results here. A failed recording from the first build leaves
   `recording_*.pcm`/`.json` in `%LOCALAPPDATA%\EasyScribe\recordings`; the
   app offers to recover it on launch.
5. Run the v4 audit verification on the owner's Windows PC: one fixed 10+
   minute speech file, v3.2.3 vs this build, cold/warm/update launch, Find
   speakers with 4- and 8-thread settings, and the repetition A–F GPU matrix
   (three runs each). Record the same speech through v3.2.3 and this build
   with the same microphone to compare the new anti-alias resampler's output.
   Check the new `startup:`, `Speech gate took` and `Diarization compute:`
   lines in the file log. Rename `whisper-cli.exe` in a test runtime to
   confirm the visible slow-fallback reason. Record the numbers in this file.
   **The full handoff for the Windows agent, including the fresh-build command,
   A–F script, pass criteria and results template, is
   `docs/rewrite/WINDOWS_AUDIT_HANDOFF.md`.** The first Windows pass is
   recorded below; resume from its open rows rather than repeating the
   completed cold/warm/update and B-matrix checks.
6. Hand over for the decision on replacing v3.2.3. Do not publish this build
   as `Latest` until the real-PC speed, clean-PC checks and GPU checks pass.

### Windows audit results — 2026-11-12, build 9af1e4b5125a0a154602d6a899eee42c7b5e3382 (partial)

Environment: Windows 11 Education 10.0.26200, AMD Ryzen 9 8940HX (16C/32T),
NVIDIA GeForce RTX 5070 Laptop GPU (driver 32.0.15.9613, 2026-04-08) +
AMD Radeon 610M (driver 32.0.21036.11002, 2026-08-20), 31.2 GB RAM,
Defender real-time on, free disk 155 GB.

Exe `EasyScribe-v4.0.2.exe` from run 43242882848 (draft pre-release), SHA-256
`482d4edad9cd30dd0761cbd077909ea0cdbc64841ac1f3904d45ba556213b952`, matches
the `.sha256` asset.

| Check | Result | Evidence |
|---|---|---|
| Build/selftest | pass | `--selftest` exit 0, 383.2 s; release build, AVX2/FMA/F16C/BMI2 on, 8 whisper.cpp threads of 32. Steps: cold decode 2.1 s, in-process decode 161.0 s, sherpa model 39.3 s, VAD 0.0 s, gate 0.2 s, speech after 30 s silence 2.7 s, decode after sherpa 168.7 s, diarization 4.7 s in-process / 3.4 s child, silence 0.1 s |
| Cold launch | 3.46 s | `startup: prepared in 3.460s`; 1.76 GB disk used during unpack |
| Warm launch | 0.87 / 0.90 s | `paths ready in 0.006s, payload prepare 0.001s`; `first window after 0.873s` / `0.903s`; `.complete` marker unchanged |
| Update launch (v4.0.1 → v4.0.2) | 4.06 s | `startup: prepared in 4.058s`; old runtime removed in the same second; peak extra disk 1.72 GB |
| Low-disk unpack | not run | owner declined filling the system disk |
| Fixed file: v3.2.3 / current, GPU / CPU | not run yet | needs GUI runs |
| Diarization 4/8 threads | not run yet | needs GUI runs |
| A–F matrix | B clean 3/3 | see below |
| Repetition repair | not run yet | |
| Same-speech resampler | not run yet | |
| Slow fallback | not run yet | |
| Clean-PC checklist | not run yet | |

A–F matrix on the fixed file (`-bs 5 -t 4 -pp`, q5_0 model unless noted).
"Longest run" is the longest chain of identical consecutive segments; "loop" is
the time those chains cover; "words" is the total word count relative to F-1,
the only run free of a loop on the reference engine path.

| Config | Device | Seconds (3 runs) | Exit | Longest run | Loop | Words vs F-1 |
|---|---|---|---|---|---|---|
| A baseline | RTX 5070 | 51.0 / 41.9 / 39.7 | 0 | 58 (all 3) | 57 s | 113% |
| B `-mc 0` | RTX 5070 | 57.2 / 19.8 / 30.2 | 0 | 1 (all 3) | 0 s | 90% |
| C `-et 2.8` | RTX 5070 | 42.0 / 36.5 / 46.3 | 0 | 4 (all 3) | 3 s | 97% |
| D `-nfa` | RTX 5070 | 45.8 / 43.7 / 45.9 | 0 | 50 (all 3) | 48 s | 107% |
| E f16 model | — | not run | — | — | — | — |
| F `-dev 1` | Radeon 610M | 603.6 (1 run) | 0 | 2 | 0 s | 100% |

- The loop reproduces on this PC, identically on all three A runs.
- B is clean on all three runs, so prompt carry-over is the cause. It also
  produces 10% fewer words than F-1; check B's transcript for dropped speech
  before adopting `-mc 0` on the GPU path.
- C shortens the loop but does not remove it. D does not help, so Vulkan flash
  attention is not the source.
- F ran once (runs 2–3 stopped by the owner); one clean run on the integrated
  GPU is not enough to blame the NVIDIA driver.
- E skipped by owner decision. Driver not updated.

Findings:

1. On a first or update launch the log has only `startup: prepared in ...`.
   The `paths ready` and `first window` lines are written before the logger
   starts, so cold and update first-window times cannot be read from the log.
2. `--selftest` writes only the pass/fail line to `selftest.txt`; the step
   lines go to stdout, which a GUI exe loses. The handoff's
   `Get-Content selftest.txt` shows no step detail; capture stdout with
   `Start-Process -RedirectStandardOutput` instead.

### Follow-up after the first Windows pass — logging fixes (Linux, 2026-11-12)

The partial Windows results above found two measurement gaps. Both are fixed
in code:

1. **Cold/update startup lines are now logged.** `main` installs the file
   logger before `payload::needs_prepare`/`Paths::discover`, using
   `payload::data_root_before_prepare()`. `logger::setup` is idempotent, so
   the later `app::startup` call keeps the same file instead of creating a
   second one. A first or update launch now logs `startup: first window`,
   `startup: paths ready` and `startup: prepared`; a cached launch keeps the
   existing `paths ready` / `first window` pair.
2. **`--selftest` keeps its step detail.** A `SelftestLog` tees every step
   line, timing and failure to `logs/selftest.txt` as well as stdout, so the
   GUI exe no longer loses the step detail that made the Windows selftest
   result hard to diagnose.

Checks:

- `cargo fmt --all -- --check`: clean.
- `cargo clippy --workspace --all-targets -- -D warnings`: clean.
- `cargo test --workspace`: 11 app + 74 core + 2 offline + 1 logger = 88
  passed; 4 ignored (2 model-backed, 2 manual benchmarks). New test
  `crates/easyscribe-core/tests/logger.rs` proves a second setup does not
  create or rotate a log; `selftest_log_keeps_step_lines_in_the_file` proves
  the step lines are written.
- Fast debug selftest passed and its `selftest.txt` now contains every step:
  cold decode 109.0 s (in-process; this development checkout has no bundled
  `whisper-cli`), sherpa 12.2 s, VAD 0.0 s, gate 0.4 s, diarization
  6.0 / 7.8 s, silence 0.2 s. This is a logging check, not a speed comparison.

The repetition A–F result narrows the cause to prompt carry-over: B (`-mc 0`)
was clean on all three runs, but its 10% lower word count than the clean F run
means `-mc 0` is not adopted globally without a word-loss check. The existing
targeted CPU repair remains the fix; its Windows verification on the
reproducing file is still open.

Remaining Windows checks are unchanged: packed-exe selftest on a fresh
`v4.0.2` build, fixed-file `v3.2.3` comparison, diarization 4/8 threads,
app-level repetition repair, same-speech resampler, slow-fallback UI, low-disk
unpack and the clean-PC checklist. `v3.2.3` remains `Latest`.
