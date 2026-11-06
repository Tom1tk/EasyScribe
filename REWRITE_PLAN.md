# EasyScribe Rune Rewrite — Plan

**Branch:** `rune-rewrite` · **Worktree:** `/home/user/EasyScribe-rune` · **Base:** `main` @ `8323516` (v3.2.3)
**Started:** 2026-11-06
**Status owner:** `REWRITE_STATUS.md` (read this first after any compaction)
**Journal:** `REWRITE_JOURNAL.md`

## Mission

Rebuild EasyScribe completely in Rust with the Rune native UI framework
(<https://github.com/anomalyco/rune>), replacing the Python/CustomTkinter
implementation and the Python launcher. The result must be **100% functionally
identical** to EasyScribe v3.2.3, with a visual treatment that is recognisably
the same product but more polished (Rune's GPU rendering, springs, typography,
native-feeling focus and hover states).

Two product goals carried over from the existing planning documents:

1. **Fully offline.** No network code, no accounts, no telemetry, no update
   checks; ffmpeg restricted to local files; temp audio cleaned.
2. **No installer, one self-contained Windows `.exe`.** This is the explicit
   direction in `EASYSCRIBE_AUDIT.md` §7.10 ("Restructure the single-exe so
   repeat launches don't re-extract ...") and
   `EASYSCRIBE_ACTION_PLAN.md` §8.3. The Rust rewrite must remove the
   installer/launcher entirely: one `EasyScribe.exe` that embeds the app
   payload (models, ffmpeg, sherpa-onnx, whisper model), extracts it **once**
   to a private cache on first run, and launches straight into the GUI on every
   later run. No install window, no shortcuts requirement, no version compare,
   no `app.bundle`, no `%TEMP%` re-extraction.

## Non-negotiable functional parity checklist (from v3.2.3)

- Transcribe files: add/drag/drop multiple audio/video files, remove/clear,
  supported extensions exactly as before, per-file status/step/progress.
- Options: Add timestamps; Name the speakers (disabled when diarization models
  absent); output folder choice ("same folder as recording" default).
- Plain/Timestamped transcript format exactly (`[HH:MM:SS]` blocks split at
  >= 2.0 s gaps, blank line between blocks; one trailing newline on disk).
- Speaker transcript format exactly (`[Speaker]` / `[HH:MM:SS] [Speaker]`,
  speaker turn + >= 2.0 s gap block rules, overlap-based assignment).
- whisper.cpp file engine when bundled (beam size 5, in-process via
  `whisper-rs`), sherpa-onnx VAD+greedy fallback otherwise. The shipped preview
  is CPU-only; the Vulkan feature remains available but is deliberately not in
  `release-static` because its load-time `vulkan-1.dll` import would break CPU
  fallback on machines without a Vulkan driver.
- Live mode: microphone capture, Silero VAD chunking, words appear as spoken,
  timer + input level meter, stop/flush last words, "Finishing".
- Record mode choice: best-quality-after-recording (record only, then file
  pipeline, timestamps + optional speakers) vs live transcription.
- Crash-safe raw PCM + JSON sidecar; startup recovery prompt; "No" keeps files;
  successful recovery writes WAV and removes sidecars.
- Save recordings folder picker + persistence in `recordings/settings.json`;
  default `Documents/EasyScribe`.
- "Find speakers" after a finished recording/transcript: estimate dialog,
  progress, speaker naming panel with sample playback, shares, Skip/confirm,
  writes `(speakers).txt`; never clobbers the first transcript.
- "Make best quality transcript" after a live recording; writes
  `(best quality).txt`; keeps live transcript.
- Name box after result: renames recording + its transcripts atomically
  (`<stem>.wav`, `<stem>.txt`, `<stem> (*).txt`); files mode renames only the
  new transcript; taken name / failure changes nothing; Enter commits.
- Per-file failure isolation; Stop cancels ffmpeg/whisper/diarization/live.
- Privacy panel content/paths/logs; "Open logs folder"; details log; rotating
  logs (10), no transcript text in logs.
- Window title `EasyScribe 3.2.3`, app icon, close cancellation/recovery.
- All user-facing strings, ordering, spacing and colour meanings preserved;
  visual polish allowed but no workflow/IA changes.
- Windows taskbar AppUserModelID, no black-flash issues (n/a in Rune),
  high-DPI, 1366x768 and larger screens.

## Implementation snapshot (session 3)

The workspace has three crates: `easyscribe-core`, `easyscribe` (Rune UI) and
`easyscribe-pack`. The UI, workers, payload, splash, recovery/rename/settings,
Windows resources and CI are implemented. Verification: 57 tests pass, clippy
and fmt are clean, the real speech fixture matches the Python transcript, the
long-silence gate test passes, all scenes render, and a 300 MB packed payload
passed first-run/cached-launch smoke. Remaining work is Windows CI validation
and the draft release; see `REWRITE_STATUS.md`.

## External plan audit (2026-11-06) — dispositions

An independent audit (`/home/user/EasyScribe-rewrite-audit.md`, @Tom) reviewed
the plan and the first core scaffold. Dispositions, so none of it is lost:

| Audit finding | Decision |
|---|---|
| 1. whisper.cpp runs CPU-only | **Partially accepted, release reversed.** The Vulkan feature exists (`whisper_rs::vulkan::list_devices()`, `gpu_device`, CPU retry), but it is not enabled in `release-static`: `whisper-rs` links `vulkan-1.dll` at load time, so a no-driver machine cannot start at all. The preview ships CPU-only. A future GPU path must load Vulkan dynamically or use a separate process. |
| 2. Engines moved in-process lose Stop/crash isolation | **Partially accepted.** Stop is restored in-process with `FullParams::set_abort_callback_safe` (whisper) and by making the diarization progress callback return non-zero when cancelled. Crash isolation is an optional hardening step: a self-spawned `--worker whisper`/`--worker diarize` child can be added behind the same IPC later. Parity requires Stop, not process isolation; tests must prove Stop interrupts a long decode, not just between chunks. |
| 3. GPU choice dropped | **Implemented behind `vulkan`, not shipped.** Device enumeration/choice and CPU retry are in `whispercpp.rs`; the release feature is CPU-only for the reason above. |
| 4. sherpa-onnx version drift (1.12.9 vs tested 1.13.2) | **Resolved by matching the binding, not forcing 1.13.2.** `sherpa-rs` 0.6.8 generates bindings from the vendored 1.12.9 header; 1.13.x changed struct layouts. The Windows build uses the hash-pinned 1.12.9 static archive. `--selftest` loads the recognizer/VAD/diarizer on the packed exe. |
| 5. Build downloads binaries | **Accepted.** Release builds must not use `download-binaries`. Vendor the sherpa DLLs/headers (or build from a vendored source tarball) and change the offline test to inspect normal runtime dependencies (`cargo tree -e normal`) instead of build dependencies. |
| 6. Existing users' data not found | **Accepted.** Add a one-time legacy import: look beside the exe, in `%LOCALAPPDATA%\Programs\EasyScribe`, and for a v3 marker; copy (never move) `recordings/settings.json`, keep the old logs, and scan legacy `recordings/` for orphan PCM/JSON for the recovery prompt. Declining recovery keeps the old files. |
| 7. No commits yet | **Accepted.** Committed the core scaffold before continuing (assets restored, models ignored). |
| 8. Icons became symlinks | **Fixed.** `assets/EasyScribe.ico` and `.png` are real tracked files again. |
| 9. Model symlinks not ignored | **Fixed.** `/models` and `/whispercpp` added to `.gitignore` without trailing slashes. |
| 10. Stale status file | **Fixed.** `REWRITE_STATUS.md` now reports the core implementation and real test results. |
| 11. Rune on Windows unproven | **Accepted.** Add a Windows cross-check smoke build (Rune window, DPI, dialog, drag/drop) as the first UI milestone and a Windows CI job before release. Headless PNG tests remain the visual safety net on Linux. |
| 12. Payload built in memory | **Accepted.** Stream the payload archive to the output exe with a hashing writer; keep the in-memory builder only for unit tests. |

### Hex-derived improvements to build in

These are deliberate improvements that keep the workflow and offline promise.
They are **not** allowed to break the golden transcript tests; where a change
could alter output (speech gating), it is gated to files with substantial
silence and covered by a regression test.

- **Speech detection before whisper.cpp.** Reuse Silero VAD to trim leading and
  trailing silence and split very long silent gaps, with timestamps offset back
  to the original file. Short/ordinary files keep the single whole-file decode
  path so the byte-identical golden transcript stays byte-identical.
- **CPU retry when the GPU fails.** One retry on CPU, with the GPU/driver
  remembered in `settings.json`; use the CPU for that driver until it changes.
- **Processor floor check.** Check AVX2, FMA, F16C and BMI2 before starting the
  app (x86_64), and show a plain message instead of dying on the first decode.
- **Background priority for long jobs.** Lower the worker thread priority on
  Windows (and `nice` on Unix) so transcription and diarization leave the PC
  responsive.
- **Speech-detection tuning (new gate only).** Try threshold 0.4 and minimum
  speech 160 ms for the pre-whisper gate; keep the existing 0.5 / 250 ms
  constants for live mode and the sherpa fallback until parity tests agree.
- **Settings recovery.** Back up a malformed `settings.json` before replacing
  it, and fall back per field to defaults.

## Architecture

```
Cargo.toml                     workspace
crates/easyscribe/             binary: main, app state, Rune views
  src/main.rs                  entry: payload bootstrap → UI / headless
  src/ui/                      screens, widgets, theme
  src/app.rs                   state machine + worker events
  src/workers/                 background jobs (files, record, live, diarize)
crates/easyscribe-core/        no UI; all domain logic, testable
  config, logging, settings, transcript, rename, recovery, ffmpeg,
  whispercpp, sherpa (C API), vad, diarizer, live, mic, payload
crates/easyscribe-pack/        build tool: append compressed payload to exe
assets/                        icon, splash/logo, fonts if needed
licenses/                      unchanged third-party texts
docs/rewrite/                  reference specs + audit reports
tests/ (in crates)             unit + integration + Rune headless UI tests
.github/workflows/             Rust CI + Windows release build
```

### Inference

- **sherpa-onnx 1.12.9** through `sherpa-rs` 0.6.8 for: offline Whisper
  transcription fallback, Silero VAD, live recognizer, speaker diarization.
  The Windows build links the official 1.12.9 static libraries with the static
  MSVC runtime.
- **whisper.cpp 1.8.3** through `whisper-rs` 0.16 for the primary file engine
  when the GGUF model is present. CPU-only in the release preview; preserve
  beam size 5 and report the engine/thread count in Show details. Whisper
  decode runs on a fresh thread because ONNX Runtime leaves thread-local state
  that makes a later encode on the same thread fail.
- **ffmpeg/ffprobe**: keep the proven subprocess approach and exact arguments,
  including `-protocol_whitelist file`. Locate via payload dir.
- **Audio**: `cpal` capture (device enumeration by name, 16 kHz mono),
  `hound` WAV, `rodio` clip playback.

### Single-exe distribution

`easyscribe.exe` + appended compressed archive footer:
- build normal binary; `easyscribe-pack` appends `payload.zip` (models,
  ffmpeg.exe/ffprobe.exe, sherpa DLLs, whisper GGUF, licenses, assets) plus a
  footer: magic, offset, length, SHA-256, version.
- first run: verify, extract to `%LOCALAPPDATA%\EasyScribe\runtime\<hash>\`
  with atomic `.complete` marker; subsequent runs use it directly. Dev builds
  read `./assets-dev` or `EASYSCRIBE_PAYLOAD_DIR`.
- no installer UI, no shortcuts, no registry, no admin. Moving the exe keeps
  working; cache can be deleted and is recreated.
- Windows resources (icon, version) embedded at link time.

## Milestones

| # | Deliverable | Verification |
|---|---|---|
| 0 | Worktree, logs, reference specs, Rune/FFI research | docs present |
| 1 | Core skeleton: config/log/settings/transcript/rename/recovery/ffmpeg + tests | `cargo test -p easyscribe-core` |
| 2 | sherpa C API + VAD/live/diarization/transcription + real-model smoke test | tests + 20 s clip transcript |
| 3 | Mic capture, PCM writer, recovery, playback | unit + headless |
| 4 | Rune UI: every screen/state, theme, interactions | headless UI tests + PNG snapshots |
| 5 | Worker orchestration: file batch, record, live, find speakers, rename | end-to-end tests |
| 6 | Single-exe payload builder, Windows resources, CI, README | pack/unpack test + `cargo build --release` |
| 7 | Multi-agent audits (functional, UI visual, code, packaging) | audit reports + fixes |
| 8 | Final polish, full test run, commit history | all green |

## Rules for this rewrite

- `REWRITE_STATUS.md` is updated after every work session and before any long
  task; it must always say what is done, what is in progress, what is next,
  known gaps and commands.
- Never claim parity for something not implemented; list gaps honestly.
- Keep the original Python tree untouched in `/home/user/EasyScribe` as the
  behavioural oracle. This worktree is the rewrite.
- One logical change per commit; no `git add -A` until final cleanup; never
  commit models, build output, caches or secrets.
- No network code in the shipped app. No telemetry.
- Every user-visible string should be copied from the Python app unless a
  documented polish change is approved.
