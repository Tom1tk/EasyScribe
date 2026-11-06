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
