# EasyScribe Rune Rewrite — Status

**Updated:** 2026-11-06 (session 2)
**Branch:** `rune-rewrite` · **Worktree:** `/home/user/EasyScribe-rune`
**Current phase:** 4 — Rune UI
**Overall:** core logic complete and tested; UI in progress

## Where things stand

- [x] Worktree/branch created from `main` @ `8323516` (v3.2.3).
- [x] Reference specs produced:
      `docs/rewrite/REFERENCE_BACKEND.md`, `REFERENCE_UI.md`,
      `REFERENCE_PACKAGING.md`.
- [x] **`easyscribe-core` implemented** (~3,500 lines, 20 modules): config,
      errors, cancellation/reporting, transcript formatting, rename, recovery,
      settings, logging, offline guard, ffmpeg/ffprobe, VAD helpers, audio,
      sherpa engine/VAD/diarization, whisper.cpp via whisper-rs, file
      transcription orchestration, microphone capture/PCM sidecar, live
      transcription, single-exe payload.
- [x] `cargo test -p easyscribe-core --lib`: **36 passed**.
- [x] **Real-engine parity proven:** whisper.cpp transcription of
      `tests/data/test_speech.wav` in Rust is byte-identical to the Python
      v3.2.3 output (all four paragraphs).
- [x] Assets restored from symlinks; `/models` and `/whispercpp` ignored.
- [x] External plan audit read and folded into `REWRITE_PLAN.md` with explicit
      dispositions.
- [ ] Full Rune UI (in progress).
- [ ] Worker orchestration in the UI crate.
- [ ] `easyscribe-pack` (streaming payload builder) + first-run progress.
- [ ] Windows smoke build / CI.
- [ ] Legacy v3 data import.
- [ ] Full test matrix and multi-agent audits.

## What works right now

```bash
cd /home/user/EasyScribe-rune
cargo test -p easyscribe-core --lib
cargo run -p easyscribe-core --example transcribe -- \
  tests/data/test_speech.wav /tmp/out.txt
```

`models` and `whispercpp` are development symlinks to the oracle tree and are
ignored by git. The Python tree is untouched.

## Open decisions (need the user)

1. **Update size.** Removing the installer/launcher means each release is the
   full ~1.3 GB exe; there is no 74 MB update bundle. Accept, or add a separate
   delta-update path later?
2. **v3 shortcuts.** Existing Desktop/Start-menu shortcuts point at v3. Should
   the first Rust run offer to repoint them (with a click), or leave them alone?

## Known gaps / deviations

- UI is a placeholder until this session's work lands.
- `sherpa-rs` 0.6.8 builds sherpa-onnx 1.12.9 while the Python app tests
  1.13.2. Release must vendor the tested DLL (audit finding 4).
- Windows-specific behaviour (Vulkan device choice, AppUserModelID, DPI,
  native dialogs, drag/drop) has not run on Windows yet; a smoke build is
  scheduled before visual sign-off.
- The Privacy dialog will be an in-app modal card over a scrim rather than a
  separate OS window. Same content/actions; documented visual deviation.
- In-process whisper/diarization rely on abort callbacks for Stop; crash
  isolation (optional self-worker child) is not implemented.

## Commands

```bash
cd /home/user/EasyScribe-rune
cargo test --workspace
cargo run -p easyscribe            # needs a display
cargo run -p easyscribe -- --snap out.png --scene files
cargo check --target x86_64-pc-windows-gnu -p easyscribe   # if toolchain present
```

## Next actions

1. Finish the Rune UI: header/mode switch, Files, Record, activity/results,
   naming panel, Find speakers, rename, Privacy, details.
2. Wire workers through `AppSpec { setup, poll }` and channels; prove Stop.
3. Add legacy import + first-run extraction progress.
4. Implement `easyscribe-pack` streaming + verification.
5. Headless scenes/PNGs, Windows CI, audits.
