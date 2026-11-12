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
  the speaker-naming panel focuses its first entry. At the default 820 pt
  height the naming panel now takes over the screen as it already did below
  820 pt, because the new folded-step titles pushed Save/Skip past the glass.
- **Windows link failure and fix.** The first release run reached
  `Build the app and packer` and failed while linking `sherpa_rs.dll`: six
  unresolved `advapi32` imports from onnxruntime's ETW telemetry and
  espeak-ng's registry lookup. The 1.12.9 static libs do not carry the
  `advapi32.lib` defaultlib reliably. Both the CI environment and
  `scripts/build-windows.ps1` now pass `-C link-arg=advapi32.lib` with the
  static CRT flags.
- **First Windows run after the link fix.** `v3.2.3-rust.2` built the release
  binary, packed the 1.7 GB payload and reached the import check. The
  `advapi32` link worked. It then failed only on a CI regex: dumpbin prints
  `subsystem (Windows GUI)`, but the check expected no parentheses. The
  corrected check accepts the real output; no application code changed.
- **Naming at the default height.** The new folded-step titles pushed the
  speaker panel's Save/Skip below the 820 pt glass. The naming panel now takes
  the screen at 820 pt as well as below it, so both actions stay visible.
- **Accessibility labels.** The timestamp and speaker checkboxes, rename field
  and per-speaker name fields now expose their visible labels to the
  accessibility tree.
- **Verification.** 59 tests pass (6 app, 51 core, 2 offline; 1 ignored golden),
  clippy clean, fmt clean. Re-rendered the 600/678/820 Files, Record, Live and
  Naming scenes; short windows scroll and the folded steps read correctly.
  Dispatched Windows run `42235634403` (`v3.2.3-rust.3`) for the static link,
  selftest, import checks and draft pre-release.

## 2026-11-06 — Session 5: diagnosing the Windows whisper.cpp encode failure

- **rust.3 result.** The static link with `advapi32` worked, the 1.7 GB payload
  packed, the imports contained no sherpa/ONNX/VC-runtime DLLs, and the exe was
  correctly a GUI binary. The packed `--selftest` then failed after 10 m 49 s in
  `whisper.cpp model and decode` with `whisper_full_with_state: failed to
  encode` (error -6). No beam-search lines were printed, so the first encoder
  pass is where it stopped. Sherpa Whisper and Silero VAD had loaded first;
  diarization still loaded afterwards.
- **Local evidence.** The same model/decoder passes locally with both generic
  (`GGML_AVX*` off) and AVX2 builds; a 5 s silence decode takes about 70 s with
  generic kernels and passes both before and after loading sherpa. A standalone
  reproduction confirmed whisper-rs 0.16's `set_abort_callback_safe` boxes the
  closure twice and then calls it through the wrong type; it reads memory that
  is not the cancel token (false on Linux, where Stop therefore never worked).
  The same undefined read can produce `true` on another layout: whisper.cpp
  then rejects the encode at its final check, after the whole encoder has run,
  which matches the Windows timing and error.
- **Diagnostic changes.** `whispercpp` now installs a raw C abort callback with
  a live `AbortProbe` (fixing Stop), and `EASYSCRIBE_WHISPER_TRACE=1` prints
  model-load time, sample/region count, thread count, per-region `full()` time
  and abort-callback call counts. The selftest runs a cold whisper decode
  **before** any sherpa model, then loads sherpa and decodes again; the first
  pass identifies a Windows build problem, the second an ONNX interference
  problem. The engine is dropped before the second decode to keep memory out of
  the comparison.
- **Windows build hardening.** The workflow now sets `GGML_AVX=ON` and
  `GGML_AVX2=ON` explicitly (matching the Python release's non-native CPU
  build), deletes cached `whisper-rs-sys` build/fingerprint directories before
  compiling, asserts the resulting CMakeCache flags, verifies the GGUF SHA-256
  (`394221…a7e2`), and uses a fresh Rust cache key. The selftest step logs the
  runner's CPU and memory. Local verification: fmt, clippy and all 59 tests
  pass; the traced selftest passes cold and after sherpa.
- **rust.4 confirmation.** `gh workflow run` without `--ref` used the workflow
  file from `main`, so the run built the new `rune-rewrite` code with the old
  CI recipe. It still proved the fix: the cold decode passed in 655.4 s and the
  post-sherpa decode in 633.6 s, where rust.3 failed at 649 s. Sherpa loaded in
  8.1 s, VAD in 0.0 s and diarization in 0.6 s. The draft
  `v3.2.3-rust.4.exe` (1,773.7 MB) and its SHA-256 were uploaded and the
  import/subsystem and cached-launch checks passed. The 655 s also showed the
  cached build was still scalar, so the workflow changes were copied to `main`
  (7f3e964) and `v3.2.3-rust.5` was dispatched with the asserted AVX2 build.

## 2026-11-06 — Session 6: audit 2 — GPU, speed diagnosis, process isolation

- **Audit 2's release-state findings.** The single exe was accepted, but two
  findings blocked replacing v3.2.3: the 655 s CI decode and the missing GPU
  path. Finding 5 clarified that the remembered-GPU code existed but was inert
  because the Vulkan feature was off, and that a driver crash in-process cannot
  be caught.
- **Finding 3 first.** `v3.2.3-rust.5` was dispatched from `main` with the
  updated workflow: the GGUF hash gate and the `GGML_NATIVE=OFF`,
  `GGML_AVX=ON`, `GGML_AVX2=ON`, `CMAKE_BUILD_TYPE=Release` asserts all ran.
  The decode still took 661.2 s cold and 635.8 s after sherpa for 5 s of
  silence. That ruled out the missing AVX2 codegen and pointed at the
  whisper-rs-sys build: it forces `GGML_OPENMP=OFF`, while the Python
  `whisper-cli` build uses OpenMP. Enabling OpenMP in-process would import
  `vcomp140.dll` at process start, which the one-exe contract cannot satisfy.
- **GPU restored (option A).** `whispercli.rs` now runs the bundled Vulkan
  `whisper-cli` as a child process with the reference argv (`-bs 5`, `-t`,
  `-dev`, `-pp`, `--output-json`). It probes devices from whisper-cli's own
  stderr, prefers a dedicated GPU over `uma: 1`, then matrix cores, persists the
  choice with `save_gpu_device`, and on a non-zero exit saves the identity with
  `save_gpu_disabled_for` and retries once with `-ng`. Stop kills the child.
  `EASYSCRIBE_WHISPER_CPU=1` forces the CPU path for clean-PC testing. The
  in-process `whisper-rs` engine remains the fallback and is still selftested.
- **Speed diagnosis.** On the Linux dev box, 23.3 s of real speech with 4
  threads: the independent `whisper-cli` took 43.9 s, the in-process engine
  65.5 s. Flash attention is now on in-process (55 MB encode buffer instead of
  212 MB). CI now runs the same `whisper-cli -ng` control on the same runner and
  derives the selftest budget as 2x that control (minimum 60 s), so a fast
  runner gets a tight gate and a slow runner cannot fail the release on runner
  variance alone.
- **No-speech gate.** Empty VAD bounds now return zero regions; the CLI and
  in-process paths log `No speech found` and return an empty transcript without
  decoding. Short files still run the VAD only for the no-speech decision and
  stay undivided when speech exists, so the golden transcript is unchanged.
  The selftest adds a 5 s silence check: 0.1 s, zero segments, message present.
- **Diarization child.** `--worker diarize --wav <path> --threads N` runs only
  the diarization engine, writes `{"progress":…}` / `{"turns":[…]}` /
  `{"error":…}` JSON lines to stdout, and puts diagnostics on stderr. The parent
  parses the pipe, forwards progress, kills on Stop, and assigns the child to a
  Windows kill-on-close job object (`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`). The
  selftest runs both the in-process engine and the child; local child timing
  10.5 s vs 10.3 s in-process.
- **Selftest now self-diagnosing.** It prints
  `whisper_print_system_info()` and fails unless AVX2/FMA/F16C are 1, prints
  `debug_assertions`/Cargo/CMake profiles, prints `print_timings()` per decode,
  uses the 23.3 s speech fixture with a golden plain-text check, enforces the
  speech budget on the primary path, tests the in-process fallback, and checks
  silence. Local release result: cold 43.9 s, in-process 65.5 s, after sherpa
  44.3 s, diarization 10.3/10.5 s, silence 0.1 s; all passed.
- **Packaging.** The payload now carries `whisper-cli.exe`, `whisper.dll`,
  `ggml*.dll`, the VC++/OpenMP redistributables, the selftest WAV and golden
  text. CI checks every non-system whisper-cli dependency is present, verifies
  the CLI runs, and asserts the CMake profile. The release notes now state the
  ~1.8 GB download, the ~2x disk use and the AVX2/FMA/F16C/BMI2 requirement.
- **Workflow ref hygiene.** `main` now holds a stub `rune-release.yml` that
  fails with the correct `--ref rune-rewrite` command; the real pipeline lives
  only on the branch. `v3.2.3-rust.6` was dispatched with
  `gh workflow run rune-release.yml --ref rune-rewrite`, so the run page shows
  `rune-release.yml@refs/heads/rune-rewrite`.
- **Code audit.** All 64 tests pass (6 app, 56 core, 2 offline; 2 ignored
  model-backed tests run manually), clippy and fmt clean. The ignored golden
  test and the long-silence test pass. `cargo test` exercises the new
  whisper-cli cancellation kill and GPU-ranking unit tests.

## 2026-11-06 — Session 7: audit 2 follow-up and rust.7

- **rust.6 failed before building.** `test-linux` restored the sherpa build
  script output from `rust-cache` but not the shared libraries it points at, so
  the linker could not find `libonnxruntime.so` / `libsherpa-onnx-c-api.so`.
  `check-windows` hit the missing `Win32_Security` feature on
  `CreateJobObjectW`. The release job reached the new control step and then
  failed because `tests/data/selftest_speech.wav` is ignored by git.
- **Tracked fixture.** Moved the generated 23.3 s clip and its golden text to
  `assets/selftest_speech.wav` / `.txt`. The payload already copies `assets/`,
  so the workflow and local build no longer reference an ignored path.
- **CI link fix.** The Linux job now caches `~/.cache/sherpa-rs` and removes the
  restored `sherpa-rs-sys` build/fingerprint directories before `cargo test`, so
  the build script re-downloads or restores the libraries the linker needs.
- **Windows fixes.** Added `Win32_Security`; moved the job object to
  `win_job.rs` and assigned `whisper-cli` as well as the diarization worker;
  shipped `vulkan-1.dll` from the Vulkan SDK so a PC with no driver can start
  `whisper-cli` and decode on the CPU instead of failing at load time. The
  `whisper.cpp` cache key was bumped to include the loader.
- **Robustness.** Removed `current_dir` from the CLI probe/decode so test
  binaries and development paths work; cached the Vulkan probe per process;
  made the progress regex whitespace-tolerant; treated a malformed diarization
  message as a terminal error and killed the worker immediately; cleaned up
  child processes if a pipe or reader thread cannot be created; cleaned temp
  JSON on every error path; and fall back to the in-process engine only when
  `whisper-cli` cannot spawn (missing DLL), not when a decode fails.
- **Notices.** Added entries for the whisper.cpp 1.8.6 Vulkan CLI, the Vulkan
  loader and the VC++ 2022 redistributable DLLs, with license texts.
- **Status honesty.** The status file now records rust.6 as a failure and keeps
  the 655 s cause open; the OpenMP explanation was not supported by the local
  benchmark. The rust.7 control step is the next measurement.

## 2026-11-06 — Session 8: rust.8 CMake gate and the whisper-cli audit

- **rust.8 failed on the CMake assertion, not the build.** The Visual Studio
  generator is multi-config, so `CMAKE_BUILD_TYPE` is empty even when the
  binary is built with `--config Release`. The assertion now checks
  `GGML_NATIVE=OFF` and `GGML_AVX2`/`FMA`/`F16C=ON` in the cache, and the
  Release binary path plus `--config Release` prove the configuration.
- **CLI start/load failures now fall back correctly.** A spawn error or a
  Windows loader exit (`0xC0000135`, `0139`, `007B`, `0142`) calls
  `transcribe_in_process` directly, never the dispatcher that could recurse
  back. The selftest sets `EASYSCRIBE_SELFTEST_STRICT_CLI` so a release build
  still fails if the shipped CLI is unusable.
- **GPU failure memory is narrower.** Every GPU error retries once on the CPU,
  but only a Vulkan marker or a GPU crash code is stored in
  `whisper_gpu_disabled_for`. CPU retries and forced-CPU runs set
  `GGML_DISABLE_VULKAN=1` so a broken driver is not touched during backend
  enumeration. A disabled device is filtered out of the picker, and the
  preferred index is saved only after a successful decode.
- **Probe and process robustness.** The probe distinguishes failure from “no
  GPU”, drains stderr while polling, honours Cancel, and is skipped for forced
  CPU. On probe failure whisper-cli chooses its own device instead of being
  forced to `-ng`. The WAV buffer is dropped before the child starts, a
  non-silent file with no VAD regions is decoded instead of becoming empty,
  stderr reading survives invalid UTF-8, kill/wait is bounded, and all temp
  JSON/stem files are removed. `ffmpeg`/`ffprobe` now join the job object.
- **Payload completeness.** The runtime cache now requires the whisper-cli
  DLLs, the Vulkan loader, the four VC++ runtime DLLs and the new notices; the
  workflow copies those DLLs as required files instead of skipping missing
  ones. `scripts/build-windows.ps1` checks the same set and pins FMA/F16C.
- **Local verification after the changes:** 68 tests pass (6 app, 60 core,
  2 offline; 2 ignored), clippy/fmt clean, release selftest passes (cold CLI
  43.9 s, in-process 65.7 s, after-sherpa 65.9 s, diarization 10.6/10.5 s,
  silence 0.1 s). `v3.2.3-rust.9` is dispatched from `rune-rewrite`.

## 2026-11-06 — Session 9: rust.9 gate and diarization audit fixes

- **rust.9 failed on a wrong CMake assertion, not the build.** On MSVC ggml does
  not define `GGML_FMA`/`GGML_F16C` options; it implies them from AVX2. The
  standalone whisper-cli gate now asserts `GGML_NATIVE=OFF` and `GGML_AVX2=ON`
  only, and passes `-DGGML_AVX2=ON`.
- **Diarization terminal result.** The parent treats the first `{"turns":…}` as
  success and stops the worker immediately, so a crash during native teardown
  cannot discard a valid answer. The worker leaks its native engine after
  computing the result and exits. Reader threads now read raw lines with
  `from_utf8_lossy` and report I/O errors instead of silently stopping; cleanup
  is bounded. `EASYSCRIBE_DIARIZE_TIMEOUT_SEC` bounds the selftest child.
- **Diarization parity.** `min_duration_on/off` now match Python's 0.3/0.5
  defaults, the unused `--threads` plumbing is removed, the worker parses
  `args_os` (non-Unicode paths no longer panic), and speaker clips are streamed
  from the WAV instead of loading the entire recording once per speaker.
- **Release pipeline.** The GGUF cache uses a Rune-only key and is saved
  explicitly before cleanup; the Vulkan loader is ensured beside the CLI on
  cache hits as well as cold builds; `ffmpeg`/`ffprobe` are checked in the
  payload.
- Local verification: 68 tests pass (6 app, 60 core, 2 offline; 2 ignored),
  clippy/fmt clean, release selftest passes (cold CLI 46.1 s, in-process 67.4 s,
  after-sherpa 68.2 s, diarization 10.5/10.5 s, silence 0.1 s).
  `v3.2.3-rust.10` is dispatched from `rune-rewrite`.

## 2026-11-06 — Session 10: rust.11 and the independent audit

- **rust.10 failed on the app's CMake gate, not the build.** The Visual Studio
  generator writes `CMAKE_BUILD_TYPE:UNINITIALIZED=Release`, not
  `CMAKE_BUILD_TYPE:STRING=Release`. The gate now accepts either form and keeps
  asserting `GGML_NATIVE=OFF`, `GGML_AVX=ON` and `GGML_AVX2=ON`.
- **Six independent audits** covered core transcription/outputs, the Rune UI,
  microphone/live recording, the payload/cache/platform layer, diarization and
  the release pipeline. Their confirmed fixes are in `d59b8f2` and summarised in
  `docs/rewrite/AUDIT_3.md`.
- **Recording hardening.** A single-instance lock prevents a second window from
  deleting the first one's temp audio or recovering its live recording. The PCM
  writer now opens the file and writes the initial sidecar before the microphone
  starts, updates the sidecar atomically, drains queued chunks on Stop, keeps the
  valid prefix after a disk error, and supports I8/U8/I24/U24/I32/U32/I64/U64.
  Multi-channel capture averages channels; config selection prefers 16 kHz then
  48 kHz, mono first.
- **Recovery hardening.** Invalid/zero metadata is rejected, the WAV is written
  to a temp file and fsynced before rename, and a missing PCM after conversion
  is success, not a failed recovery.
- **Output hardening.** Transcript writes are atomic; empty diarization produces
  one unnamed speaker (Python parity); batch outputs with the same name get a
  numeric suffix; speaker clips stream frames instead of allocating the file.
- **Payload hardening.** A staging tree is checked against the required files
  before `.complete` is written, the marker is fsynced, stale staging is
  age-gated on cached launches, relative env paths are anchored to the exe, and
  the declared MSRV is 1.89.
- **Release hardening.** The sherpa archive's TTS-only `espeak-ng.lib` and
  `piper_phonemize.lib` are removed from the link set (EasyScribe never calls
  TTS), so the GPL-3.0 static-link risk is removed if the Windows link
  succeeds. The tag must match the workspace version, ffmpeg/ffprobe and the
  VC++ DLLs are executed in the payload, and the GGUF cache is Rune-only.
- Local verification: 68 tests pass, clippy/fmt clean, release selftest passes
  (whisper-cli 48.2 s, in-process 67.9 s, after-sherpa 65.7 s, diarization
  10.4/10.6 s, silence 0.1 s). `v3.2.3-rust.11` is dispatched from `d59b8f2`.

## 2026-11-06 — Session 11: rust.11 and the espeak licensing finding

- **rust.11 failed at the app link, not at compile.** The audit's GPL concern was
  real: removing `espeak-ng.lib`/`piper_phonemize.lib` produced
  `LNK2019: unresolved external symbol espeak_Initialize` from
  `piper-phonemize-lexicon.obj`. The libraries are restored. espeak-ng
  GPL-3.0 is now named in `licenses/THIRD-PARTY-NOTICES.txt` with a source
  offer, and the combined-work licensing decision is recorded as a blocker for
  publishing `Latest`.
- The release tag must now match the workspace version, the app's CMake gate
  accepts `CMAKE_BUILD_TYPE:UNINITIALIZED=Release`, and the offline test checks
  the Windows dependency graph.
- Audit 3 findings and residual risks are in `docs/rewrite/AUDIT_3.md`.
  `v3.2.3-rust.12` is dispatched from the latest HEAD.

## 2026-11-06 — Session 12: rust.12 payload verification

- **rust.12 built and assembled the payload**, then failed in the payload
  verification step because `whisper-cli -m "" -f ""` is expected to exit 1
  (empty paths) and the step inherited that exit code. The check now captures
  the code, fails only on the loader error `-1073741515`, and exits 0 after the
  successful import/ffmpeg checks. `dumpbin`'s own exit code is now checked.
- The GPL-3.0 espeak-ng disclosure is also in the draft release notes.
- `v3.2.3-rust.13` is the validation run.

## 2026-11-06 — Session 13: rust.13 passes the Windows pipeline

- **`v3.2.3-rust.13` (run `42289270909`, `b3c5112`) passed.** The control
  whisper-cli CPU decode of the 23.3 s fixture took 40.8 s and set an 82 s
  budget (2x, minimum 60 s). The bundled CLI child decoded in 37.5 s, the
  in-process engine in 551.0 s, the post-sherpa decode in 538.5 s, sherpa
  Whisper ONNX in 6.1 s, VAD in 0.0 s, in-process diarization in 6.2 s, child
  diarization in 6.3 s, and silence in 0.0 s with zero segments. System info
  showed AVX2/FMA/F16C/BMI2, the golden text matched, and the extraction,
  cached launch, imports and subsystem checks passed.
- **Draft assets verified.** `EasyScribe-v3.2.3-rust.13.exe` is 1,921,180,779
  bytes; its SHA-256 is
  `69ee5c49b15501833699a73518ab9c8ec216f323f9b54035a9cae48ad4d4d991` and the
  `.sha256` asset is 97 bytes. The release is a draft pre-release, not
  `Latest`. The superseded `rust.4` and `rust.5` drafts were deleted; no other
  old drafts remained.
- Still open before `Latest`: the real-PC speed/GPU/microphone/DPI checks in
  `docs/rewrite/CLEAN_PC_TEST.md` and the GPL-3.0 espeak-ng combined-work
  licensing decision.

## 2026-11-06 — Session 14: final audit and rust.14

- **Final-audit finding 1, the GPL-3.0 espeak-ng link, is resolved for the Rust
  build.** The prebuilt sherpa archive and `SHERPA_LIB_PATH` were removed;
  `sherpa-rs-sys` now builds the vendored sherpa-onnx 1.12.9 from source with
  `SHERPA_ONNX_ENABLE_TTS=OFF` and `BUILD_SHARED_LIBS=OFF`. `rust.14` failed
  twice on the way: Windows 11 has no `wmic`, so `cmake/show-info.cmake` passed
  an empty string to `string(REPLACE)` and then an empty list to `list(GET)`
  under CMake 4. The workflow and `scripts/build-windows.ps1` patch both lines
  after `cargo fetch`. The build now asserts the TTS-off cache entries and
  checks `sherpa-onnx-core.lib` for `espeak_Initialize` (the first check was
  too broad and matched WeSpeaker model names). Main's Python-release notice
  now discloses the espeak-ng component that remains in v3.2.3 (`a11ca32`).
- **Final-audit finding 2, the silent slow fallback.** `Status::SlowFallback`
  makes the UI read **Transcribing (slow fallback)** with an explanation that
  the built-in engine can take many times longer; the log carries the same
  warning. The engine stays as a damaged-payload fallback, matching the audit's
  accepted "warn, don't run silently" option.
- **Final-audit finding 3, size headroom.** The existing 1900 MiB exe guard is
  stricter than the suggested 2.05 GB and now documents the GitHub 2 GiB
  release-asset limit.
- **Final-audit finding 4, the live-mode ONNX race.** All live VAD and
  recognizer calls now take one process-wide `ONNX_LOCK`. `stop()` waits for
  the decoder first with the 60 s deadline, then detaches both threads only on
  timeout; `DECODER_ACTIVE` rejects a new live session until a detached decoder
  exits, and a poisoned lock is recovered so one native panic cannot disable
  live mode.
- **Release notes corrected** to 1.92 GB (1.79 GiB), about 3.7 GB fresh and
  about 5.4 GB peak update space; `CLEAN_PC_TEST.md` and the README match.
- **`v3.2.3-rust.14` (run `42309047043`, `09761af`) passed.** Control 33.5 s,
  budget 67 s, CLI child 37.6 s, in-process 551.0 s, sherpa ONNX 42.6 s, VAD
  0.0 s, after-sherpa 541.1 s, diarization 8.5/8.6 s, silence 0.1 s;
  AVX2/FMA/F16C/BMI2 and the golden text verified. `EasyScribe-v3.2.3-rust.14.exe`
  is 1,921,914,713 bytes, SHA-256
  `f4e79cfb49a1755a4b362bd4467659c65ae553c309afc3f51d5f1c329bc6969c`, with a
  97-byte `.sha256`. The draft is a pre-release, not `Latest`; the superseded
  `v3.2.3-rust.13` draft was deleted.
- Local verification after the changes: 68 tests pass (6 app, 60 core, 2
  offline; 2 ignored), clippy and fmt clean, and the release selftest passed
  (CLI 44.4 s, in-process 65.3 s, after-sherpa 69.2 s, diarization 10.5/10.6 s,
  silence 0.1 s; the last full run before the documentation commit).
- Remaining before `Latest`: the real-PC speed/GPU/microphone/DPI/dialog/
  drag-drop/low-disk checks in `docs/rewrite/CLEAN_PC_TEST.md`. The espeak
  licensing decision is no longer a blocker.

## 2026-11-07 — Session 15: the Rust line becomes v4.0.0 (still pre-release)

- **Version 4.0.0.** The owner asked for the Rust build to carry the 4.0.0
  version while staying a pre-release. The workspace version is now `4.0.0`,
  the window title follows `CARGO_PKG_VERSION` (`EasyScribe 4.0.0`), and the
  README and clean-PC checklist use `EasyScribe-v4.0.0.exe`.
- **The release tag no longer needs a hyphen.** `v4.0.0` is accepted alongside
  `v4.0.0-rc.1`; the workflow still creates the release with
  `prerelease: true`, so `v4.0.0` does not become `Latest`. The tag must match
  the workspace version exactly or start with `v4.0.0-`.
- **`v4.0.0` (run `42356076024`, `f9e03a2`) passed the full Windows pipeline.**
  Control 23.1 s, budget 60 s, CLI child 27.7 s, in-process 331.7 s, sherpa
  ONNX 31.9 s, VAD 0.0 s, after-sherpa 315.8 s, diarization 4.7/4.8 s, silence
  0.0 s. AVX2/FMA/F16C/BMI2 and the golden text verified; the build again
  asserted TTS off and no `espeak_Initialize` symbol.
- **Asset.** `EasyScribe-v4.0.0.exe` is 1,921,914,193 bytes, SHA-256
  `0b0bc737ff822385b242230b822852fa0c910106cd5cca8960852c95503b4e62`, with an
  89-byte `.sha256`. The release is a draft pre-release, not `Latest`;
  `v3.2.3` remains `Latest`. The superseded `v3.2.3-rust.14` draft was deleted.
- Local checks: 68 tests pass, clippy and fmt clean. The remaining gate is the
  real-PC clean-PC/speed/GPU/microphone test; `v4.0.0` must not be promoted to
  `Latest` before that passes.

## 2026-11-09 — Session 16: first real-PC test fixes

- **Recordings now save on Windows.** The owner's first v4.0.0 test reported
  "Could not record / The recording was not saved" after Stop. Root cause:
  `recover_pcm_to_wav` reopened the finished WAV read-only before
  `File::sync_all`; Windows `FlushFileBuffers` requires `GENERIC_WRITE`, so
  every conversion failed in both the chosen folder and the work folder. The
  file is now reopened for writing. The conversion error is surfaced instead
  of being swallowed, and the raw PCM/sidecar remain for crash recovery.
- **Stop while "Getting ready" is a cancellation, not a failure.** The worker
  checks the stop token before opening the microphone and reports
  "Recording cancelled / Nothing was saved."; the UI no longer shows a save
  error for a session that never captured audio.
- **Record details no longer overlap.** The details card grew alongside the
  record view, shrinking it below its content so the utility row covered Start
  recording. On Record the details card now keeps its natural height and the
  Record page scrolls when it needs more room; Files keeps the growing log.
  `record-view` also no longer allows itself to shrink below its card.
- **Microphone menu is a popup.** It is mounted at the root with
  `El::anchored` under the selector, so it no longer pushes Step 2 down. An
  outside-click scrim and Escape close it; the placement is unchanged.
- **Regression coverage.** A new `mic::tests` case runs the full PCM→WAV stop
  path into a destination folder (it would fail on Windows with the old
  read-only sync). Four new headless scenes cover record-details,
  record-live-details, recording-details and the open microphone menu.
- **Rebuild.** `v4.0.0` from `17415fb` passed the full Windows pipeline (run
  `42621260185`): control 23.8 s, budget 60 s, CLI child 26.9 s, in-process
  338.7 s, after-sherpa 339.4 s, sherpa ONNX 33.9 s, VAD 0.0 s, diarization
  5.1/5.2 s, silence 0.1 s. The old draft was deleted and recreated with the
  new asset: 1,921,949,518 bytes, SHA-256
  `d935714b4b3e09b5d8a3f0717276bdb3526c914734f91c93ed54d5fc79a1c28b`.
  `v3.2.3` remains `Latest`; `v4.0.0` remains a draft pre-release.

## 2026-11-10 — Session 17: v4.0.1 pause, caret and window icon

- **Pause/Resume while recording.** The owner asked for a Pause button next to
  Stop in the Record tab, changing to Resume while paused. `MicRecorder` takes
  a shared `PauseToken`; the cpal callback keeps the device stream open but
  drops audio while paused, so the format cannot change and resume cannot fail.
  Nothing is appended to the PCM, the crash-recovery sidecar, the level meter
  or the live VAD queue. The timer now follows the written sample count, so it
  freezes while paused and always matches the WAV duration. Stop while paused
  saves the audio captured before the pause; pausing at 00:00 and stopping
  removes the empty pair and reports `Recording cancelled`.
- **Rename caret.** Rune's text-field caret is solid and defaults to white,
  which is invisible on the rename box's white field. The rename box now hides
  the built-in caret and draws a 1.06 s blink line at the insertion point using
  `Cx::caret`; the speaker-name fields get a visible ink caret. The caret is
  read in window points so it follows typing, selection and page scrolling.
- **Window icon.** The executable already embeds `assets/EasyScribe.ico`
  through `winres`, but winit left the class's generic icon in the title bar
  and taskbar. `App::on_window` now calls `WM_SETICON` with the big and small
  icon sizes loaded from resource id 1, which also covers window recreation
  after a GPU release.
- **Version and tests.** The workspace, window title and release docs move to
  `4.0.1`. New tests cover pause/resume, pause-while-saving, clearing the pause
  state after finish, and stopping before any audio. There are now 9 app, 62
  core and 2 offline tests (73 passed, 2 ignored), and 18 headless scenes
  including `record-paused`.
- **Rebuild.** `v4.0.1` from `1fd3f35` passed in run `42795767098` (4 vCPUs,
  no GPU): control 44.8 s, budget 90 s, CLI child 50.8 s, in-process 439.9 s,
  sherpa ONNX 47.0 s, VAD 0.1 s, after-sherpa 425.4 s, diarization 7.2/7.3 s,
  silence 0.1 s. AVX2/FMA/F16C/BMI2, the native selftest and the TTS-off/no
  espeak checks passed. The asset is 1,921,993,041 bytes, SHA-256
  `54a45ae030df7bbccee096fe8ef9d138af742e00a0636f50f4f4bb334f34af6e`.
  It is published as a pre-release (not `Latest`) at
  <https://github.com/Tom1tk/EasyScribe/releases/tag/v4.0.1>; the superseded
  `v4.0.0` draft and tag were deleted, and `v3.2.3` stays `Latest`.
- **Next.** The owner needs to retest pause/resume, the timer/WAV length, the
  zero-audio cancel, the rename caret and the title-bar/taskbar icon, then run
  the clean-PC checklist. Do not promote `v4.0.1` to `Latest` before those
  pass.

## 2026-11-10 — Session 18: release turnaround

- **The 30-minute release was measured before changing it.** Run `42795767098`
  took 33.7 min: 11.3 min build, 16.6 min selftest, 1.4 min release and about
  2 min setup. The selftest's two in-process whisper.cpp decodes took 439.9 s
  and 425.4 s; the shipping `whisper-cli` path took 50.8 s.
- **A shorter in-process clip did not help.** The first attempt trimmed the
  fixture to 8 s. On the next run each short decode still took 680.5 s and
  657.9 s, because whisper.cpp pads every clip to a 30 s encoder window; the
  cost is the fixed encode, not the audio length. That change was discarded.
- **The fallback diagnostics are now opt-in.** The normal release gate runs
  the bundled CLI on the full 23.3 s fixture with the exact golden text, loads
  sherpa, checks VAD, runs both diarization paths and the silence check. The
  two in-process fallback decodes run only when `full_selftest=true` is passed
  to the workflow dispatch. This is the gate to use before promotion or after
  a whisper/backend change.
- **The build cache is now effective.** The workflow no longer deletes
  `target/` before the cache saves, so Swatinem restores compiled dependencies.
  It also stops deleting the whisper.cpp/sherpa-onnx build-script outputs; the
  existing CMake-cache assertions still reject a stale flag configuration.
  The first warm-up run took 15.9 min to build; the next took 3.5 min.
- **Result.** Run `42815563747` completed in **10.6 min**: build 3.5 min,
  selftest 2.1 min, release 1.1 min. That is a 23-minute reduction. The
  workflow changes are after the `v4.0.1` asset, so the owner's current
  download is unchanged; the next build uses the faster path.

## 2026-11-12 — Session 19: v4 speed/bug audit actioned

- **Diarization threads (item 1).** `sherpa-rs` 0.6.8 hardcodes
  `num_threads: 1` for both the segmentation and embedding models. A local
  `local_diarize.rs` now calls `SherpaOnnxCreateOfflineSpeakerDiarization`
  with `config::num_threads()` for both, keeps the existing clustering values
  and progress callback, rejects non-UTF-8 model paths instead of panicking,
  and checks for a null native result. "Find speakers" now passes only the
  native compute time to `save_measured_speed` through the worker protocol
  (`took_sec`) instead of wall time that included process start and model
  load. The estimate constants were not changed without a Windows
  same-file measurement; the new `Diarize: Native compute took …` log line
  makes that measurement possible. A local long-file benchmark (ignored test
  `bench_diarization_thread_counts`, path supplied through
  `EASYSCRIBE_BENCH_WAV`) showed the expected multi-thread speedup with
  identical turn output at every setting.
- **Speech gate (item 2).** `sherpa_engine::has_speech` replaces the
  full-file `speech_regions` pass before `whisper-cli`. It exits at the first
  confirmed speech segment (or after 0.25 s of continuous speech), logs
  "Checking for speech…" and records the gate's elapsed time at info level.
  Silence still costs a full pass, as the audit noted. On a long local test
  file (kept outside the repository), the gate returned speech almost
  immediately instead of scanning the whole file (ignored
  `bench_gate_against_full_vad` test). The
  selftest now prepends 30 s of silence to the speech fixture and asserts the
  gate finds the speech (0.6 s locally).
- **Thread priority (item 3).** Removed `lower_thread_priority` and all three
  worker calls. Python had no equivalent; the children already run at normal
  priority, and the Rust-side file preparation is faster without the
  below-normal setting.
- **Startup (item 4).** `prepare_with_progress` records wall time; the main
  process logs paths-ready, payload-prepare and first-window times. Cleanup of
  stale staging and old runtimes now runs on a `PayloadCleanup` thread after
  the window, guarded by the cache lock. The pre-unzip SHA-256 pass is gone:
  a `HashingReader` feeds the same bytes to SHA-256 and the streaming zip
  reader, and the staging folder is deleted if the final hash does not match.
  A unit test proves the hash covers the whole payload and that a flipped
  byte changes it. Packaging/cold-launch numbers still need the owner's PC.
- **Resampler (item 5).** Microphone capture now uses a 63-tap Blackman
  windowed-sinc low-pass (7.2 kHz) before the linear resample. The unit test
  puts a 10 kHz sine through 48 k → 16 k and measures more than 40 dB of
  rejection against a 1 kHz reference. The PCM writer channel is bounded at
  256 blocks; drops are counted and logged, and VAD queue depth is logged at
  75% (once per crossing). This protects the real-time callback without
  silently losing audio.
- **Release profile (item 6).** The release profile uses `codegen-units = 1`
  and `incremental = false`; `release-fast` keeps the old 16/true settings for
  local smoke builds. The shipped artifact still needs a Windows build-time
  and selftest comparison.
- **Fallback visibility (item 7).** The slow path already switched the status
  to "Transcribing (slow fallback)" and logged the failure; it now also sends
  a short failure reason to the activity's visible detail line through
  `Reporter::detail`, so the reason is not limited to the details panel. The
  file log records "whisper-cli could not start, using the in-process engine".
- **Repetition repair (item 8).** After `parse_json`, the app splits output
  into sentences, normalises case and punctuation, and flags a run of more
  than 4 identical long sentences (more than 8 for short interjections such
  as "Yes."). It re-decodes only `start - 1.5 s .. end + 1.5 s` with
  `-ng -ot -d -mc 0 -et 2.8`, then splices the absolute-offset JSON back into
  the original timeline. A failed or still-repeating repair keeps the original
  transcript. Tests cover the detector, punctuation/case normalisation,
  in-segment repeats and a fake `whisper-cli` that proves the `-ot` run is
  spliced back; the selftest now fails if the golden decode contains a loop.
  The owner's A–F GPU matrix is still required before promotion.
- **Smaller items (item 9).** The UI log is capped at 2,000 lines; the child
  stderr tail uses `VecDeque`; ffmpeg's duration/time regexes are `LazyLock`;
  a non-suitable WAV is probed once instead of twice; whisper/VAD model paths
  return a clear error on non-UTF-8 instead of a lossy path; the live VAD
  queue logs high-water depth. The diarization child still inherits
  `EASYSCRIBE_NUM_THREADS` as the audit noted.
- **Verification (Linux, 8 logical CPUs, CPU-only).** `cargo fmt --all --
  --check`, `cargo clippy --workspace --all-targets -- -D warnings` and
  `cargo test --workspace` are clean (9 app + 73 core + 2 offline passed,
  4 ignored (2 model-backed, 2 manual benchmarks)). The full debug selftest passed: CLI cold
  57.0 s, in-process 90.9 s, sherpa 9.5 s, after-sherpa 93.3 s, diarization
  5.3/6.0 s, silence 0.2 s. The fast selftest with the new gate step passed:
  CLI 55.6 s, sherpa 13.0 s, speech-after-silence gate 0.6 s, diarization
  5.6/5.5 s, silence 0.1 s. The release profile built in 8m35s and its
  selftest passed (release profile asserted; CLI 53.6 s, sherpa 15.0 s, gate
  0.3 s, diarization 5.9/5.2 s, silence 0.1 s). These are Linux CPU numbers,
  not the Windows GPU/real-PC gate.
- **Still open.** Real Windows cold/warm launch and update timings, the same
  10+ minute file on v3.2.3 vs this build (with and without speakers), the
  item 8 A–F GPU matrix and three clean GPU runs, the clean-PC checklist, and
  a Windows release build with the new profile. Do not promote the preview to
  `Latest` until those pass.
