# EasyScribe Rune Rewrite — Status

**Updated:** 2026-11-06 (session 4)
**Branch:** `rune-rewrite` · **Worktree:** `/home/user/EasyScribe-rune`
**Base:** `main` @ `8323516` (EasyScribe v3.2.3)
**Phase:** 6 — packaging, audit and Windows release
**Overall:** implementation complete; functional, code-quality, packaging and UI
audits addressed. Windows release run `42235634403` (`v3.2.3-rust.3`) is the
final validation gate.

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

- [x] `cargo test --workspace`: **6 app + 51 core + 2 offline = 59 passed**,
      1 ignored (the long-silence golden test, run manually).
- [x] `cargo clippy --workspace --all-targets`: clean.
- [x] `cargo fmt --all -- --check`: clean.
- [x] Real-engine golden: whisper.cpp output for
      `tests/data/test_speech.wav` matches the Python v3.2.3 text byte for byte.
- [x] Long-silence gate test: 40 s of added silence before and after the speech
      fixture still produces the `quarterly results` transcript.
- [x] `--selftest` loads the sherpa Whisper ONNX model, Silero VAD,
      whisper.cpp GGUF and diarization models, and decodes a local WAV.
- [x] All 13 headless scenes render (`preparing` plus the 12 app states).
- [x] UI click/state tests pass.
- [x] Packed first-run smoke with a 300 MB fake payload: extraction, marker,
      adoption, main window, and cached quick launch.
- [x] Payload ZIP stores >8 MB entries without a compression level; a 9 MB
      regression test covers the previous real-payload failure.
- [x] Flock-based cache lock, stale staging cleanup, incomplete-runtime
      replacement, and signed-exe footer scanning.
- [x] Windows run `42232368420` built and packed the exe successfully after
      adding `advapi32`; it stopped only at a CI regex that rejected dumpbin's
      `subsystem (Windows GUI)` spelling. The regex is fixed in `370fcc2`.

## Functional/packaging decisions

| Decision | Reason |
|---|---|
| Full-size updates only | One exe is the product; no delta bundle or installer. |
| Extract once to `%LOCALAPPDATA%\EasyScribe\runtime\v1-<hash>` | whisper.cpp/ffmpeg need real files; later launches reuse the cache. |
| User data separate from runtime | Settings, logs, temp and recovery survive version updates. |
| Old v3 shortcuts not repointed | Avoid touching user data without consent; portable exe runs anywhere. |
| CPU-only whisper.cpp in this preview | `whisper-rs` Vulkan links `vulkan-1.dll` at load time, so a machine without a Vulkan driver could not start or fall back to CPU. GPU acceleration can return later as a dynamically loaded path. |
| sherpa-onnx **1.12.9** static libs | `sherpa-rs` 0.6.8 binds the 1.12.9 C API; 1.13.x added struct fields and is ABI-incompatible. |
| Static MSVC runtime | The 1.12.9 static libraries are `/MT`; the workflow sets `+crt-static` and `CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded`. |
| Explicit `advapi32.lib` on Windows | onnxruntime's ETW/telemetry and espeak-ng's registry calls leave six undefined imports in `sherpa-rs`'s cdylib link. The build adds `-C link-arg=advapi32.lib` alongside the static CRT flags. |
| Naming panel takes over at 820 pt and below | Python clips Save/Skip at the default height; the Rust build hides the Files/Record content while naming so both actions stay reachable. Intentional usability deviation. |
| In-process engines | Stop is cooperative; diarization has a cancellation callback. Crash isolation remains optional future hardening. |
| Privacy as an in-app modal | Same content/actions as the Python window; documented visual deviation. |

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

- Windows static link, Vulkan removal, DPI, native dialogs, drag/drop and a real
  microphone can only be proven by Windows CI and a clean Windows VM.
- The local 300 MB fake payload validates mechanics, not the real 1.7 GB
  artifact or free-space behaviour.
- CPU-only whisper.cpp is slower than the Python Vulkan path; functionality and
  transcript output are unchanged.
- In-process diarization can use substantial memory on very long recordings;
  crash isolation is not implemented.
- Authenticode signing is not enabled. The footer parser tolerates a certificate
  appended after the footer, and the release publishes a SHA-256 file.

## Next actions

1. Watch Windows run `42235634403` (`v3.2.3-rust.3`): static link with
   `advapi32`, GUI startup, import/subsystem scan, `--selftest`, cached launch
   and draft upload.
2. Fix any remaining Windows-only failure and re-run with the next tag.
3. Verify the draft `EasyScribe.exe` and `.sha256`, then hand over for
   real-device testing (microphone, DPI, dialogs, drag/drop).
