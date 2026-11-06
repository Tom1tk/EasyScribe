# EasyScribe Rune Rewrite — Status

**Updated:** 2026-11-06 (session 8 — own audit and rust.10)
**Branch:** `rune-rewrite` · **Worktree:** `/home/user/EasyScribe-rune`
**Base:** `main` @ `8323516` (EasyScribe v3.2.3)
**Phase:** 7 — speed, GPU, process isolation and final release
**Overall:** implementation complete. Audit 2's findings are actioned: the
Vulkan `whisper-cli` is back as a child process with CPU retry, diarization is
isolated in a child, flash attention and the no-speech gate are in, and the
selftest measures real speech. The audit-2 follow-up blockers (missing tracked
speech fixture, `Win32_Security`, Linux sherpa link, `whisper-cli` job object,
missing redistributable notices) are fixed. `rust.7` failed in the Vulkan loader
copy, `rust.8` in a multi-config CMake assertion and `rust.9` in a wrong
MSVC-only FMA assertion; all three are fixed and `v3.2.3-rust.10` is the current
validation run. The independent audit against the CLI engine, diarization
worker, transcript clips and release pipeline is being actioned; the confirmed
fixes are listed below.

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
      `EasyScribe.exe` and publish a draft pre-release with a `.sha256`.
- [x] **Docs** — README, third-party notices, packaging/UI/backend references.

## Verification completed

- [x] `cargo test --workspace`: **6 app + 60 core + 2 offline = 68 passed**,
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
- [x] Local release selftest, 4 threads, 23.3 s speech:
      `whisper-cli` cold **43.9 s**, in-process cold **65.7 s**, after sherpa
      **65.9 s**, diarization in-process **10.6 s**, diarization child
      **10.5 s**, silence check **0.1 s** (zero segments, "No speech found").
- [x] All 13 headless scenes render (`preparing` plus the 12 app states).
- [x] UI click/state tests pass.
- [x] Packed first-run smoke with a 300 MB fake payload: extraction, marker,
      adoption, main window, and cached quick launch.
- [x] Payload ZIP stores >8 MB entries without a compression level; a 9 MB
      regression test covers the previous real-payload failure.
- [x] Flock-based cache lock, stale staging cleanup, incomplete-runtime
      replacement, and signed-exe footer scanning.
- [x] `v3.2.3-rust.5` built from the updated workflow with the GGUF SHA-256
      gate and asserted `GGML_NATIVE=OFF`, `GGML_AVX=ON`, `GGML_AVX2=ON`,
      `CMAKE_BUILD_TYPE=Release`; both selftest decodes passed.
- [x] Audit 2 follow-up: the selftest WAV/TXT now live in tracked `assets/`;
      `windows-sys` has `Win32_Security`; the Linux job restores the sherpa
      runtime cache and re-runs `sherpa-rs-sys`'s build script; `whisper-cli`
      shares the kill-on-close job object; the payload ships the Vulkan loader
      and notices for it and the VC++ redistributables; `current_dir` and the
      repeated GPU probe were removed; a malformed diarization message stops the
      worker immediately.

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
| sherpa-onnx **1.12.9** static libs | `sherpa-rs` 0.6.8 binds the 1.12.9 C API; 1.13.x added struct fields and is ABI-incompatible. |
| Static MSVC runtime | The 1.12.9 static libraries are `/MT`; the workflow sets `+crt-static` and `CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded`. |
| Explicit `advapi32.lib` on Windows | onnxruntime's ETW/telemetry and espeak-ng's registry calls leave six undefined imports in `sherpa-rs`'s cdylib link. The build adds `-C link-arg=advapi32.lib` alongside the static CRT flags. |
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

## Known limitations / residual risk

- Windows GPU device selection, Vulkan failure/CPU retry, DPI, native dialogs,
  drag/drop and a real microphone can only be proven on a real Windows PC or VM.
  `v3.2.3-rust.10` exercises the CPU path on CI (the runner has no GPU), but not
  a dedicated/integrated GPU or a crashing driver.
- The real-PC speed gate from audit 2 is still open: compare the same 5–10 min
  file on v3.2.3 GPU, v3.2.3 CPU and rust.10, then record wall times in this
  file. CI has no GPU and its 4 vCPUs are much slower than a normal desktop.
- The 655 s CI decode cause is unproven. The OpenMP hypothesis was not supported
  by the local benchmark; rust.10's same-runner control measurement is the next
  datapoint.
- The in-process fallback is slower than `whisper-cli` on Windows; it is used
  when the payload has no `whisper-cli` or when the CLI cannot start. The
  shipping primary path is the child process.
- The download is about 1.8 GB and every update is full-size; after first launch
  the extracted runtime takes about another 1.7 GB (roughly 2x the exe).
- A real ~1.8 GB payload unpack on a nearly full disk still needs a clean-PC
  test; the local fake-payload smoke validates mechanics only.
- Authenticode signing is not enabled. The footer parser tolerates a certificate
  appended after the footer, and the release publishes a SHA-256 file.

## Next actions

1. Commit and push these fixes, then dispatch `v3.2.3-rust.10` from
   `rune-rewrite`.
2. Confirm `rune-release.yml@refs/heads/rune-rewrite`, the whisper-cli control
   time, system-info features, golden text, speech budget, both selftest
   engines, the child diarization, imports/subsystem, cached launch and draft
   assets. Record the result (pass or fail) here.
3. Run an independent code audit over the finished Rust/Rune product and fix
   confirmed findings.
4. Verify the draft `EasyScribe-v3.2.3-rust.10.exe` and `.sha256`; delete the
   superseded `rust.4`–`rust.9` drafts if present.
5. Run the clean-PC script in `docs/rewrite/CLEAN_PC_TEST.md` (unpack,
   microphone, DPI, dialogs/drag-drop, SmartScreen, GPU matrix, offline) and the
   real-PC speed comparison; record results here.
6. Hand over for the decision on replacing v3.2.3. Do not publish rust.10 as
   Latest until the real-PC speed and clean-PC checks pass.
