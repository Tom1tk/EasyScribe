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

