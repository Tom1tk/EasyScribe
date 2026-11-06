# EasyScribe Rune Rewrite — Journal

Newest entries at the bottom. Each entry: date/time, what was done, what broke,
what was decided. `REWRITE_STATUS.md` holds the current snapshot; this file is
the narrative history.

## 2026-11-06 — Session 1

- **Project check.** `/home/user/EasyScribe` is on `main`, clean except
  untracked `rebrand-plan.md`; `git fetch` confirms HEAD == `origin/main`
  (`8323516`, v3.2.3) and fetched tag `v3.2.3`.
- **Read source inventory:** ~6.5k lines Python app + 870-line launcher.
  `src/gui.py` is 2455 lines; `src/transcriber.py` 473; `diarizer.py` 334.
- **Read `CLAUDE.md` + README.** Captured packaging lessons, offline guarantee,
  engine story (whisper.cpp file engine → Vulkan/CPU; sherpa-onnx fallback,
  VAD, live, diarization), model inventory and version history.
- **Created worktree** `/home/user/EasyScribe-rune`, branch `rune-rewrite`,
  based on `main`.
- **Cloned Rune** to `/home/user/rune-ref` (shallow, commit
  `781e2b6d87cd5f32d6f4365956fad4baaf24ee50`). Confirmed hello example builds
  on this Linux VM. Rune has headless rendering (`--snap`, Runtime tests),
  a component crate (`rune-kit`), drag/drop example, and no heading into a
  display requirement for tests.
- **Found single-exe direction** in audit §7.10 and action plan §8.3:
  "Restructure the single-exe so repeat launches don't re-extract the embedded
  ~1 GB bundle". User explicitly asked to pick this up. Design chosen:
  one executable with appended compressed payload, extract once to a cache,
  no installer/launcher.
- **Dispatched four background research agents:** backend spec, UI spec,
  packaging/installer spec, Rune + Rust crates research. Their outputs will be
  `docs/rewrite/REFERENCE_BACKEND.md`, `REFERENCE_UI.md`,
  `REFERENCE_PACKAGING.md`, `TECH_RESEARCH.md`.
- **Wrote** `REWRITE_PLAN.md`, `REWRITE_STATUS.md`, this journal.
- **Next:** consume reference docs, scaffold workspace, port pure logic first.

## 2026-11-06 — Session 2: core done, audit response

- **Core implementation.** Built the whole domain crate: config, errors,
  reporting/cancel, transcript builders, rename, recovery, settings, rotating
  logs, offline guard, ffmpeg, VAD, audio, sherpa engine, diarization,
  whisper.cpp, orchestration, microphone/PCM, live transcription and payload.
  36 unit tests pass.
- **Real-engine proof.** Rust whisper.cpp produced the exact Python transcript
  for `tests/data/test_speech.wav` (byte-identical, including the
  `[00:00:00]` block and trailing newline).
- **Audit received** from @Tom. Took findings 1, 3, 4, 5, 6, 7–12 directly.
  On finding 2, I prefer abort callbacks to self-spawned workers for this
  rewrite: `whisper-rs` 0.16 has `set_abort_callback_safe`, and sherpa's
  diarization progress callback can return non-zero. That restores Stop without
  a second process. Crash isolation remains an optional hardening step, not a
  parity requirement. This is the one place I disagree with the audit's
  recommended implementation, not its concern.
- **Assets/ignore fixed.** Restored the real `.ico`/`.png`; added `/models` and
  `/whispercpp` so the dev symlinks can never be committed.
- **Plan updated** with the six changes and the Hex-derived improvements
  (speech gate, CPU retry, CPU floor check, background priority, settings
  recovery). The speech gate is deliberately conditional so the golden
  transcript stays byte-identical.
- **Two questions left for the user**: full-size updates (no delta bundle) and
  whether to repoint old v3 shortcuts. Both are recorded in `REWRITE_STATUS.md`.
- **Next:** the Rune UI. I will build it as a state machine with channel-driven
  workers, headless scenes for every key screen, and the exact copy deck from
  `REFERENCE_UI.md`.


## 2026-11-06 — Session 3: complete UI, audits, release hardening

- **Rune UI finished.** Files, Record, activity/results, naming, Find speakers,
  rename, Privacy, details, splash, native dialogs and drag/drop. All 12
  original headless scenes plus `preparing` render; click/state tests pass.
- **First-run packaging proven.** Built a 300 MB fake payload under Xvfb:
  first launch verified the hash, extracted to a staging folder, wrote
  `.complete`, renamed into the cache and adopted the runtime. A second launch
  reused the cache without touching the marker. Real payload was not packed
  locally because only ~5 GB disk remained.
- **Real-payload ZIP bug found and fixed.** `zip` rejected a compression level
  on `Stored` entries, so payloads with models over 8 MB failed. Stored entries
  now omit the level; a 9 MB regression test covers it.
- **Payload hardening.** OS-level cache lock (`fs2`), stale staging cleanup,
  incomplete-runtime replacement, signed-exe footer scan (certificate after the
  footer), container/footer field validation, relative manifest offset, and
  full asset/licence checks in `cache_complete`.
- **Functional audit fixes.** Microphone-open deadlock (sender kept alive before
  `join`), disabled Play-sample, stale second-rename stem, recovery folder,
  settings lost-update merge, bundled-ffmpeg validation, case-only Windows
  rename, ffprobe timeout, live transcript fallback when WAV conversion fails,
  half-even rounding, missing details-log lines, speech-gate cancellation,
  recovery sidecar ordering, silent-live message, offline guard install.
- **Long-silence gate test.** Added a pure region-merge test plus an ignored
  real-engine test with 40 s of silence before and after the speech fixture.
  The padded transcript still contains the expected `quarterly results` text.
- **Thread-local engine bug found by the native selftest.** After sherpa loads
  its ONNX Whisper model on a thread, a later whisper.cpp encode on that same
  thread fails with error -6 (`failed to encode`); a fresh thread works. File
  transcription now runs every whisper.cpp decode on a dedicated fresh thread
  and shares the sample buffer via `Arc`, so the isolation costs no extra audio
  memory. Golden output is unchanged.
- **Code-quality audit fixes.** Close now cancels batch/find workers; spawn
  failures send terminal events; live stop has a 60 s decoder deadline; PCM
  writer errors propagate; settings writes are atomic and merge on disk; cached
  file sizes; 8/24-bit WAV scaling; `ShellExecuteW`; GUI subsystem; `rodio`
  WAV decoder.
- **Packaging audit changes.** The earlier plan to use sherpa-onnx 1.13.2
  static libraries was wrong: `sherpa-rs` 0.6.8 generates bindings for 1.12.9,
  and 1.13.x added fields to the Whisper config struct. Switched to the
  hash-pinned 1.12.9 static archive. Removed the `vulkan` feature from
  `release-static`: a load-time `vulkan-1.dll` import would stop the app before
  `main` on machines without a Vulkan driver, making CPU fallback unreachable.
  This preview is CPU-only; GPU acceleration needs a dynamically loaded path.
- **Static CRT.** The 1.12.9 libraries are `/MT`; the workflow and local script
  now set `+crt-static`, `CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded`,
  `CMAKE_POLICY_DEFAULT_CMP0091=NEW` and `GGML_NATIVE=OFF`, and `dumpbin`
  rejects `VCRUNTIME`, `MSVCP`, `VULKAN`, `sherpa` and `onnxruntime` imports.
- **Release selftest.** Added `--selftest`: extraction, Silero VAD,
  sherpa Whisper ONNX, whisper.cpp GGUF decode and diarization model load.
  It returns an exit code and writes `logs/selftest.txt`, so CI proves the
  packed exe actually loads every native engine. CI also checks a cached second
  launch does not re-extract and uploads a `.sha256`.
- **Verification.** 58 tests pass (5 app, 51 core, 2 offline; 1 ignored golden),
  clippy clean, fmt clean. A fresh packed-payload smoke (9 MB fixture with the
  full required file list) extracted on first launch and reused the cache on the
  second. Next: commit/push, put the release workflow on `main`, dispatch the
  Windows draft build.

## 2026-11-06 — Session 4: UI audit, Windows link fix, final build

- **UI audit blockers fixed.** The computed window height (`820/678/600 pt`) was
  never passed to `AppState`, so short-window layout decisions used the
  default. The real height is now stored before `App::new`; headless scenes can
  override it with `EASYSCRIBE_WINDOW_HEIGHT`. At 600–678 pt the page scrolls,
  so the utility row and tall results are reachable where the reference allows
  a last-resort scroll.
- **Keyboard and pointer audit fixes.** Record-mode tiles are focusable radios
  in a focus group; microphone menu rows are focusable options and the selector
  is a labelled combobox. Rejected file drops route through `add_files` so the
  `Skipped …` note appears. The timestamps and speaker labels toggle their
  options. Privacy is a real modal, trapping focus and restoring it on close.
- **Visual audit fixes.** Idle file rows no longer show `Waiting`; folded Files
  and Record summaries keep the `Choose options` / `Record` (and microphone)
  step titles; `Use default` uses the teal link style; the file count is 13 pt
  muted; the recording-saved detail shortens the folder to 60 characters; the
  timer stays coral while saving; the level meter no longer draws a 1 px dot at
  zero; a fatal batch that saved an earlier file keeps a full green bar; and
  the speaker-naming panel focuses its first entry.
- **Windows link failure and fix.** The first release run reached
  `Build the app and packer` and failed while linking `sherpa_rs.dll`: six
  unresolved `advapi32` imports from onnxruntime's ETW telemetry and
  espeak-ng's registry lookup. The 1.12.9 static libs do not carry the
  `advapi32.lib` defaultlib reliably. Both the CI environment and
  `scripts/build-windows.ps1` now pass `-C link-arg=advapi32.lib` with the
  static CRT flags.
- **Verification.** 59 tests pass (6 app, 51 core, 2 offline; 1 ignored golden),
  clippy clean, fmt clean. Re-rendered the 600/678/820 Files, Record and Live
  scenes; short windows scroll and the folded steps read correctly. Dispatched
  Windows run `42232368420` (`v3.2.3-rust.2`) for the static link, selftest,
  import checks and draft pre-release.
