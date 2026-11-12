# Windows audit handoff (v4 / `rune-rewrite`)

This is the entry point for the agent running the remaining v4 audit checks on
a real Windows PC. Everything that can be verified on Linux is already done;
this file records what is left, how to obtain a build containing the fixes,
and exactly what to measure and where to write the results.

Do not promote the preview or replace `v3.2.3` as `Latest`. The promotion
decision belongs to the owner after these checks pass.

## 0. State at handoff

- Branch: `rune-rewrite`.
- Code state: all nine findings in `EasyScribe-v4-audit.md` are actioned.
  Item 2 is fully implemented: the pre-whisper gate streams the WAV in
  16,384-frame blocks and computes the peak in the same pass, so it no longer
  allocates the whole file as `f32`.
- Linux verification (CPU only, not Windows GPU/real-PC evidence):
  - `cargo fmt --all -- --check`: clean.
  - `cargo clippy --workspace --all-targets -- -D warnings`: clean.
  - `cargo test --workspace`: 10 app + 74 core + 2 offline passed; 4 ignored
    (2 model-backed, 2 manual benchmarks).
  - Full debug selftest: CLI cold 43.9 s, in-process 68.4 s, sherpa 8.7 s,
    speech gate 0.3 s, padded 30 s-silence transcription 83.7 s, after-sherpa
    68.7 s, diarization 3.6/4.0 s, silence 0.1 s.
  - Post-streaming release build: 2m43s incremental; selftest CLI 43.7 s,
    gate 0.3 s, diarization 3.6/3.8 s, silence 0.1 s.
  - CI runs `43224636495` (streaming), `43224901423` (cancellation test) and
    `43225196187` (docs) passed Linux tests and the Windows `cargo check`.
- The first partial Windows pass (`9af1e4b`, recorded in `REWRITE_STATUS.md`)
  found two logging gaps, now fixed: the logger starts before the payload
  check so cold/update logs contain all startup lines, and `--selftest` writes
  every step to `selftest.txt`. The Windows timings already recorded remain
  valid; re-run the launch checks only to confirm the new lines.
- The current Windows pre-release `v4.0.1` was built before the audit changes.
  Do not use it for the audit; build a fresh exe from this branch (section 2).

The exact commit that contains this file is the handoff commit. Record the
full SHA in the results; do not rely on a short hash written in prose.

## 1. Non-negotiable rules

1. **Confidential recordings and transcripts never enter the repository.**
   Do not commit audio, video, WAV/PCM, JSON sidecars, transcript text,
   screenshots containing transcript text, file logs, shell history or
   `git` commit messages with transcript content. Use a local path variable
   such as `$AuditWav` and record only timings/counts/pass-fail in the repo.
2. **No transcript text in the file log.** The app already follows this rule;
   do not paste transcript text into `REWRITE_STATUS.md`, a commit, or a test
   fixture. Invented fixtures are fine; confidential material is not.
3. **Use `git add <explicit paths>`, never `git add -A`.** The working tree
   may contain local audio and extracted payloads that must not be committed.
4. **Keep `v3.2.3` as `Latest`.** Do not publish the draft release, move the
   `Latest` tag, or bump the version. A failed check is a result, not a reason
   to change the release state.
5. Record Windows build, CPU, GPU, driver version/date, RAM, Defender state
   and free disk space with every measurement. A timing without the machine
   is not useful.

## 2. Get a Windows build containing the audit fixes

### Preferred: dispatch the release workflow

The workflow checks out `rune-rewrite` at dispatch time and builds the
single-file exe. It creates a **draft pre-release**; leave it as a draft.
From the repository root:

```powershell
git fetch origin
git rev-parse origin/rune-rewrite

gh workflow run rune-release.yml --ref rune-rewrite `
  -f tag=v4.0.2 -f full_selftest=false
gh run list --workflow rune-release.yml --limit 1
gh run watch <run-id> --exit-status
```

Use a unique tag such as `v4.0.2` or `v4.0.2-rc.1`; it must be `v4.0.2` or start with `v4.0.2-` because
the workspace version is `4.0.2`. In the workflow log, find the line
`Building <sha>` and confirm it is the handoff commit or a later commit that
contains `perf: stream the pre-whisper gate WAV in bounded blocks` and
`fix: keep cold-start timings and selftest steps in the log`. The
`9af1e4b` build from the first Windows pass predates the logging fix; use a
newer build for the remaining checks.

Download the draft asset:

```powershell
New-Item -ItemType Directory -Force .\audit-build | Out-Null
gh release download v4.0.2 `
  -p "EasyScribe-v4.0.2.exe" `
  -p "EasyScribe-v4.0.2.exe.sha256" `
  -D .\audit-build
```

If `gh release download` cannot see the draft, open the draft release in
GitHub or download its assets through `gh api`; do not publish it.

Verify the hash:

```powershell
$exe = ".\audit-build\EasyScribe-v4.0.2.exe"
$expected = (Get-Content "$exe.sha256" -Raw).Split()[0].ToLower()
$actual = (Get-FileHash $exe -Algorithm SHA256).Hash.ToLower()
if ($actual -ne $expected) { throw "SHA-256 mismatch: $actual" }
$actual
```

### Alternative: local Windows build

`scripts/build-windows.ps1` builds `dist\EasyScribe.exe`. It requires the
payload folders (`models`, `ffmpeg` or `tools\ffmpeg`, `whispercpp` with the
Vulkan `whisper-cli.exe` and its DLLs, `assets`, `licenses`, `LICENSE`) and
the MSVC/CMake prerequisites described at the top of the script. Prefer the
workflow unless the agent already has that payload tree.

## 3. Where the app writes logs and data

- Runtime cache: `%LOCALAPPDATA%\EasyScribe\runtime\v1-<hash>`.
- Data root: `%LOCALAPPDATA%\EasyScribe`.
- File logs: `%LOCALAPPDATA%\EasyScribe\logs\EasyScribe_<timestamp>.log`.
- Selftest result: `%LOCALAPPDATA%\EasyScribe\logs\selftest.txt`.
- Recording work files: `%LOCALAPPDATA%\EasyScribe\recordings`.

The **file log is authoritative** for timing. Use Show details for the
user-visible status, but copy numbers from the log.

Useful log lines:

- `startup: paths ready in ... payload prepare ...`
- `startup: first window after ...`
- `startup: prepared in ...`

On a first or update launch, `first window` is the splash window and
`paths ready` / `prepared` cover the unpack. On a cached launch, `paths ready`
and `first window` are the two launch timings. The logger now starts before
the payload check, so all three lines are present on every launch.

- `[whisper.cpp] Engine: Vulkan (device ...)` / `Engine: CPU ...`
- `Speech gate took ... for ... of audio (speech=...)`
- `[Diarize] Native compute took ...`
- `Diarization compute: ... for ... of audio (N threads)`
- `[whisper.cpp] Repetition detected ...`
- `[whisper.cpp] Repaired ...s of repeated output on the CPU.`
- `whisper-cli could not start, using the in-process engine`

Search without dumping transcript text:

```powershell
$log = Get-ChildItem "$env:LOCALAPPDATA\EasyScribe\logs\EasyScribe_*.log" |
  Sort-Object LastWriteTime -Descending | Select-Object -First 1
Select-String -Path $log.FullName -Pattern `
  'startup:|Speech gate took|Diarization compute|Native compute took|Engine: |Repetition detected|Repaired |whisper-cli could not start' |
  ForEach-Object { $_.Line }
```

## 4. Build/smoke check

- [ ] Run the exe on a clean profile. The window opens; first launch extracts
      the runtime under `%LOCALAPPDATA%\EasyScribe\runtime`.
- [ ] Run the native selftest and require exit code 0:

```powershell
& $exe --selftest
if ($LASTEXITCODE -ne 0) { throw "selftest failed" }
Get-Content "$env:LOCALAPPDATA\EasyScribe\logs\selftest.txt" -Tail 60
```

`selftest.txt` now contains every step line and timing, not only the final
pass/fail line, so the GUI exe does not lose the step detail. If the exe is
started from a console you can still tee stdout, but the file is the record.

- [ ] Record the exe size, the extracted runtime size, the version/build SHA,
      and the Windows/CPU/GPU/driver/RAM/Defender/free-disk details.
- [ ] Confirm the selftest log contains the speech-gate step and the golden
      transcript check (the fixture is invented/public test speech).

## 5. Audit item 4: startup and unpack timings

Measure with Windows Defender real-time protection **on** (the owner's normal
state). Use a stopwatch or `Measure-Command`; record the `startup:` lines too.

- [ ] **Cold first launch.** Start from a snapshot with no
      `%LOCALAPPDATA%\EasyScribe`, or rename that folder aside. Measure from
      process start to the main window, and record:
      `first window` (the splash appears), `paths ready` / `prepared` (the
      unpack is done and the main UI can run), exe size, extracted runtime
      size, free disk before/after.
- [ ] **Warm launch.** Close the app completely and start again. Record the
      `paths ready` / `first window` timings. Confirm the runtime `.complete`
      marker timestamp did not change.
- [ ] **Update launch.** Keep the old runtime, run the new exe, and record the
      prepare/unpack time, peak free disk (the update needs up to about
      5.4 GB temporarily), and whether the old runtime is removed after the
      new one starts.
- [ ] **Low-disk unpack.** From a snapshot with about 2 GB free, confirm the
      clear "Not enough free disk space" message and no half-unpacked runtime
      that breaks the next start.
- [ ] Record the release build time from the workflow run (the audit item 6
      check).

Pass: cold/warm/update complete without a partial runtime; low disk fails
cleanly; the numbers are recorded, not guessed.

## 6. Fixed 10+ minute file: items 1, 2 and 6

Use **one fixed 10+ minute speech file** for every run. Keep it outside the
repository and refer to it only through a local variable:

```powershell
$AuditWav = "C:\path\to\invented-or-owner-supplied-10min.wav"
if (-not (Test-Path $AuditWav)) { throw "set `$AuditWav first" }
```

Do not commit the WAV, its transcript, or its name if it is confidential.
Record only wall times, gate times, diarization times, thread counts and
whether the transcript matched a local reference (and a word-difference count
if it did not).

Run the same file through:

- [ ] `v3.2.3` with GPU (default).
- [ ] `v3.2.3` with CPU (the Python release's GPU-disabled setting, or no
      Vulkan driver).
- [ ] Current build with GPU (default).
- [ ] Current build with CPU (`$env:EASYSCRIBE_WHISPER_CPU = "1"`).
      Clear it afterwards with `Remove-Item Env:EASYSCRIBE_WHISPER_CPU`.
- [ ] Repeat the current-build GPU/CPU runs with speaker diarization on and
      off, if the file has multiple speakers.

For each run record:

| Field | Where it comes from |
|---|---|
| Wall time | stopwatch from Start to completed transcript |
| Engine/device | `[whisper.cpp] Engine:` line |
| Gate time | `Speech gate took ...` |
| Gate result | `speech=true/false` in the same line |
| Diarization wall/compute | `Diarization compute: ...` and `[Diarize] Native compute took ...` |
| Threads | `Diarization compute: ... (N threads)` |
| Output match | compare to a local reference; record identical / N words differ |
| Peak memory (optional) | Task Manager or `Get-Process` sampling; note method |

Diarization thread check (item 1):

- [ ] Set `$env:EASYSCRIBE_NUM_THREADS = "4"` and run Find speakers.
- [ ] Set `$env:EASYSCRIBE_NUM_THREADS = "8"` and run Find speakers.
      Clear it afterwards with `Remove-Item Env:EASYSCRIBE_NUM_THREADS`.
- [ ] Record `[Diarize] Native compute took` and `Diarization compute` for
      each; confirm the turn count/output is identical.
- [ ] Do the same on `v3.2.3` if its setting is available, or record why not.

Pass criteria from the audit:

- Current-build CPU wall time is within 10% of `v3.2.3` CPU.
- Current-build GPU wall time is within 10% of `v3.2.3` GPU.
- The transcript matches the local reference on both engines.
- The gate returns quickly on a file with early speech; the gate time is much
  smaller than the old full-file VAD pass. Record both numbers even if the
  old pass must be inferred from wall time.
- Diarization is faster at 4/8 threads than at the default/1-thread setting,
  with identical turn output.

## 7. Audit item 8: repetition A–F GPU matrix

Use the same fixed file and the bundled `whisper-cli.exe` from the extracted
runtime. Run each configuration **three times**; the fault is not
deterministic.

```powershell
$runtime = Get-ChildItem "$env:LOCALAPPDATA\EasyScribe\runtime" -Directory |
  Sort-Object LastWriteTime -Descending | Select-Object -First 1
$cli   = Join-Path $runtime.FullName "whispercpp\whisper-cli.exe"
$model = Join-Path $runtime.FullName "whispercpp\ggml-large-v3-turbo-q5_0.bin"
$outDir = ".\audit-runs"
New-Item -ItemType Directory -Force $outDir | Out-Null

function Invoke-AuditRun {
  param(
    [string]$Name,
    [string[]]$Extra = @(),
    [string]$ModelPath = $model,
    [int]$Device = 0
  )
  $out = Join-Path $outDir $Name
  $sw = [Diagnostics.Stopwatch]::StartNew()
  & $cli -m $ModelPath -f $AuditWav -bs 5 -t 4 -pp `
    --output-json --output-file $out -dev $Device @Extra 2> "$out.err"
  $code = $LASTEXITCODE
  $sw.Stop()
  [pscustomobject]@{
    Run = $Name
    Seconds = [math]::Round($sw.Elapsed.TotalSeconds, 1)
    Exit = $code
  }
}
```

Matrix (three runs each; use `-1`, `-2`, `-3` in the name):

| Run | Extra arguments | What it tests |
|---|---|---|
| A | none (primary GPU, `-dev 0`) | baseline |
| B | `-mc 0` | prompt carry-over |
| C | `-et 2.8` | stricter loop guard |
| D | `-nfa` | flash attention on Vulkan |
| E | f16 model instead of q5_0 | quantised weights |
| F | `-dev 1` | the other GPU/driver, if present |

Example calls:

```powershell
$results = @()
$results += Invoke-AuditRun "A-1" -ModelPath $model -Device 0
$results += Invoke-AuditRun "B-1" -Extra @("-mc","0") -ModelPath $model -Device 0
$results += Invoke-AuditRun "C-1" -Extra @("-et","2.8") -ModelPath $model -Device 0
$results += Invoke-AuditRun "D-1" -Extra @("-nfa") -ModelPath $model -Device 0
# E: download the f16 model locally, then:
# $f16 = ".\audit-build\ggml-large-v3-turbo.bin"
# $results += Invoke-AuditRun "E-1" -ModelPath $f16 -Device 0
# F, only if a second GPU exists:
# $results += Invoke-AuditRun "F-1" -ModelPath $model -Device 1
$results | Format-Table -AutoSize | Tee-Object "$outDir\timings.txt"
```

For E, download the f16 model once and record its SHA-256; keep it outside the
repo:

```powershell
Invoke-WebRequest `
  -Uri "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin" `
  -OutFile ".\audit-build\ggml-large-v3-turbo.bin"
(Get-FileHash ".\audit-build\ggml-large-v3-turbo.bin" -Algorithm SHA256).Hash
```

For every run record:

- exit code, wall seconds, and the stderr tail (GPU name, driver, `fallbacks`);
- whether the repeated sentence appears and how many times (**count only, no
  transcript text**);
- whether the run is clean on all three attempts;
- `Get-CimInstance Win32_VideoController | Select Name,DriverVersion,DriverDate`
  and the GPU reported in stderr.

**Partial result already measured (2026-11-12, RTX 5070 Laptop):** A reproduced
a 58-segment / 57 s loop on all three runs; B (`-mc 0`) was clean on all three
but produced 10% fewer words than the one clean F run, so do not switch the
whole GPU path to `-mc 0` without checking for dropped speech. C shortened the
loop but did not remove it; D did not help; F was clean once on the integrated
Radeon. The targeted fix is the app-level repair below: it leaves the fast
decode alone and re-decodes only the repeated range on the CPU with `-mc 0`.
The next Windows run must verify that repair on this same reproducing file.

Then verify the app-level repair on the same file:

- [ ] Run the file through the current build on the GPU.
- [ ] The log must show `Repetition detected ...` and
      `Repaired ...s of repeated output on the CPU.` when a loop is detected.
- [ ] The saved transcript must no longer contain the loop; if repair fails,
      the log must say it kept the original transcript.
- [ ] Record the repeat count and repaired span; never paste the sentence.

Interpretation:

- B clean ⇒ prompt carry-over is the cause. The app-level repair uses `-mc 0`
  only for the re-decoded range; a global `-mc 0` needs a word-loss check
  first.
- C clean ⇒ the entropy guard is too loose; consider `-et 2.8`.
- D clean ⇒ Vulkan flash attention is the source.
- E clean ⇒ quantised weights on Vulkan are the source.
- F clean ⇒ one GPU driver is the source.
- If A is still clean on all three runs, record that the loop did not
  reproduce on this machine; do not claim the fault is fixed. Those three
  clean A runs are also the audit's "three clean GPU runs".
- Record the GPU driver version/date. If the owner allows it, update the
  driver once and repeat A before concluding anything about the driver.

## 8. Audit item 5: same-speech resampler comparison

- [ ] Record the same invented sentence (30–60 s) through `v3.2.3` and the
      current build with the **same microphone** and input level.
- [ ] Compare the transcripts. Record pass/fail and a word-difference count;
      do not copy the text into the repo.
- [ ] Repeat once with a USB headset if available.
- [ ] Note the microphone name and Windows input format (44.1 kHz/48 kHz).

Pass: no dropped/aliased words attributable to the new 63-tap anti-alias
resampler, and the transcript is at least as good as `v3.2.3`.

## 9. Audit item 7: slow fallback visibility

Use a short invented clip; the in-process fallback is 10–19x slower.

- [ ] Copy the extracted runtime to a disposable folder, or rename
      `whispercpp\whisper-cli.exe` to `whisper-cli.exe.bak` in the active
      runtime.
- [ ] Start the app and transcribe the clip.
- [ ] Confirm the title/status says the slow fallback is in use and the
      activity detail line shows a short `Built-in engine: ...` reason.
- [ ] Confirm the file log contains
      `whisper-cli could not start, using the in-process engine`.
- [ ] Restore `whisper-cli.exe` and confirm normal GPU/CPU transcription again.
- [ ] Record pass/fail and the timing of the slow run.

## 10. Clean-PC checklist and remaining gates

Run `docs/rewrite/CLEAN_PC_TEST.md` on the clean PC/snapshot and record
pass/fail for every section:

- unpack and low disk;
- microphone, pause/resume, zero-audio cancel, save/recovery, USB headset;
- DPI 100/150/200%, mixed-monitor drag, short windows;
- dialogs, drag/drop, spaces/non-ASCII/OneDrive paths;
- SmartScreen/Mark of the Web;
- GPU matrix (dedicated/integrated/no driver/forced failure/forced CPU);
- diarization and Stop/worker cleanup;
- offline/firewall check.

Also record the real-PC speed comparison from section 6 and the repetition
matrix from section 7. The pass criteria for replacing `v3.2.3` are in
`CLEAN_PC_TEST.md` section 7 and in the audit.

## 11. Record the results

Append a section to `REWRITE_STATUS.md` (do not rewrite history):

```markdown
### Windows audit results — YYYY-MM-DD, build <full-sha>

Environment: Windows <build>, <CPU>, <GPU + driver version/date>, <RAM>,
Defender <on/off>, free disk <GB>.

| Check | Result | Evidence |
|---|---|---|
| Build/selftest | pass/fail | exit code, version, selftest tail |
| Cold launch | ... s | `startup:` lines |
| Warm launch | ... s | `startup:` lines |
| Update launch | ... s | prepare/first-window, old runtime removed |
| Low-disk unpack | pass/fail | message |
| Fixed file: v3.2.3 GPU | ... s | wall, gate, diarization |
| Fixed file: v3.2.3 CPU | ... s | wall, gate, diarization |
| Fixed file: current GPU | ... s | wall, gate, diarization |
| Fixed file: current CPU | ... s | wall, gate, diarization |
| Diarization 4/8 threads | ... / ... s | identical turn count |
| A–F matrix | table/attachment | seconds, repeats, GPU |
| Repetition repair | pass/fail | repeat count, repaired span |
| Same-speech resampler | pass/fail | word differences |
| Slow fallback | pass/fail | visible reason, log line |
| Clean-PC checklist | pass/fail | section list |
```

Record only timings, counts, versions, device names and pass/fail. Never paste
transcript text, the confidential file name, its duration, or its location
into the repository.

Commit only the documentation you changed:

```powershell
git add REWRITE_STATUS.md docs/rewrite/WINDOWS_AUDIT_HANDOFF.md
git commit -m "docs: record Windows audit results"
git push origin rune-rewrite
```

If a check fails, record the failure and the exact log line (without
transcript text) and stop before promotion. Do not "fix" a timing by changing
the measurement method after the fact.

## 12. Promotion gate

Do not publish the draft release or move `Latest` until:

- [ ] build/selftest passes on the packed exe;
- [ ] cold/warm/update and low-disk unpack pass;
- [ ] current build is within 10% of `v3.2.3` on CPU and GPU for the fixed
      file, with matching transcripts;
- [ ] diarization thread counts and turn output match;
- [ ] the repetition A–F matrix is understood and the app-level repair works
      (or the failure is documented and accepted by the owner);
- [ ] same-speech resampler comparison passes;
- [ ] the slow-fallback UI test passes;
- [ ] the clean-PC checklist passes;
- [ ] the owner approves the version bump/promotion.

Until then: `v3.2.3` remains `Latest`; `v4.0.1` and `v4.0.2` (and any `v4.0.2-rc.*` draft)
remains a pre-release.
