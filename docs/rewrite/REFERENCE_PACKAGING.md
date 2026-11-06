# EasyScribe → Rune Rewrite — Reference: Packaging, Installation & Distribution

**Oracle:** EasyScribe **v3.2.3** (`APP_VERSION = "3.2.3"`), branch `main` at commit `8323516`
**Target:** Rust + Rune rewrite in `/home/user/EasyScribe-rune` (branch `rune-rewrite`)
**Status:** reference specification. Sections 2–8 and 10 describe current v3.2.3 behaviour/obligations; sections 9 and 11–13 define the port/test/CI target. Where the current installer machinery conflicts with the target, §6.7 wins for packaging only; app behaviour is always parity.
**Source documents:** `EASYSCRIBE_AUDIT.md` §7.10, `EASYSCRIBE_ACTION_PLAN.md` §8.3, `REWRITE_PLAN.md`, `CLAUDE.md` Rules 5, 14, 16–18, 20.

Normative words: **MUST**, **SHOULD**, **MAY**. “Current” means the Python v3.2.3 implementation; “target” means the Rust/Rune rewrite.

**Contents**

1. [Executive summary](#1-executive-summary)
2. [Current artifact and install layout](#2-current-artifact-and-install-layout-v323)
3. [Install marker `.easyscribe-install.json`](#3-install-marker-easyscribe-installjson)
4. [Launcher behaviour](#4-launcher-behaviour-normative-current-behaviour)
5. [Update bundle and hash validation](#5-update-bundle-and-hash-validation)
6. [Single-.exe history and the rewrite goal](#6-single-exe-history-and-the-explicit-rewrite-goal)
7. [Asset and model inventory](#7-asset-and-model-inventory)
8. [Versioning and naming conventions](#8-versioning-and-naming-conventions)
9. [Tests that encode behaviour](#9-tests-that-encode-behaviour)
10. [Licensing and notices](#10-licensing-and-notices-requirements)
11. [CI and release requirements](#11-ci-and-release-requirements)
12. [Exact constants quick reference](#12-exact-current-constants-and-quick-reference)
13. [Open questions and risks](#13-open-questions--risks-for-the-rust-packaging-work)
14. [Source references](#14-source-references-oracle)

---

## 1. Executive summary

The current release is called “one .exe”, but it is a **self-extracting installer**: a PyInstaller ONEFILE launcher with a ~1.5 GB `app.bundle` embedded. On every double-click the PyInstaller bootloader unpacks the whole bundle to `%TEMP%` *before any Python code runs*; the Python launcher then either installs to `<exe_dir>\EasyScribe\` or launches the already-installed `EasyScribe.exe`. The installer UI, version-aware in-place update, backup/rollback, marker file and shortcuts all exist to make that model usable.

The rewrite’s explicit goal is different: **one fully self-contained Windows `.exe` with no installer**, which extracts its embedded payload **once** to a private cache and opens the GUI directly on every later run. This is the real fix for audit §7.10; the action plan’s §8.3 option (a) (“installer semantics”) was a pragmatic workaround for the Python/PyInstaller architecture, not the desired end state.

| Concern | Current v3.2.3 | Rune target |
|---|---|---|
| User-visible artifact | `EasyScribe-v3.2.3.exe` (launcher, ~1.4–1.6 GB) + installed `EasyScribe\` folder | `EasyScribe-v3.2.3.exe` only |
| Install step | GUI folder picker, “Install”, progress, launch | none; first run extracts payload with a “Getting ready” progress window |
| Repeat launch | PyInstaller re-extracts bundle to `%TEMP%`, then launches installed exe | cache hit; no bulk copy |
| Install marker | `.easyscribe-install.json` | none; payload footer + cache `.complete` manifest |
| Version comparison | `_version_key`, never downgrade, in-place update | not needed; each exe carries its own payload |
| Rollback | `.easyscribe-old\` backup + `updating` marker | old exe remains runnable; cache is immutable and version-keyed |
| Shortcuts | Desktop + Start Menu, once per install | none automatically (optional user action) |
| User data | `<install>\recordings`, `<install>\logs` | stable per-user data root, outside versioned runtime |
| Models | real files under `<install>\models` and `<install>\whispercpp` | real files under the extracted runtime |
| Update download | full exe or “small” update exe with hash-checked kept files | replace the exe; new payload hash → new runtime; old runtime GC’d |
| Offline | enforced by `offline_guard` + no network code | no network dependencies in the shipped binary; all bytes embedded |
| Licenses | `LICENSE.txt`, notices, `licenses\` in install folder | same files in payload/runtime + in-app About/Privacy entry |

**What must survive the rewrite:** every user-visible app behaviour in v3.2.3 (file transcription, live recording, diarization, rename, recovery, settings, transcript formats, logs, privacy, offline guarantees, icon/version), the models and engines, and the licensing obligations. **What is intentionally removed:** the installer window, install-location picker, marker file, automatic shortcuts, update `.exe`, version comparison, backup/rollback, and `app.bundle`.

---

## 1.1 Implemented layout in `rune-rewrite` (2026-11-06)

This is the implemented packaging contract, not a future target.

**Artifact.** One MSVC `.exe` contains the normal Rust/Rune program, an appended
ZIP payload, and a 128-byte footer. Windows ignores the overlay. The payload
contains `models/`, `ffmpeg/`, `whispercpp/ggml-large-v3-turbo-q5_0.bin`,
`assets/`, `licenses/` and `LICENSE`.

**Footer.** `ESCFTR01` + version/flags + payload offset/size + manifest
offset/size (relative to the payload) + SHA-256 + app version + build id +
CRC32 + `ESCEOF01`. `easyscribe-pack` writes it streaming, so memory stays flat
for the ~1.7 GB output. `read_footer` scans the last 1 MiB so an Authenticode
certificate appended after the footer does not hide it.

**Cache.** First launch verifies the payload SHA-256 while copying it into
`%LOCALAPPDATA%\EasyScribe\runtime\v1-<hash>.tmp-<uuid>`, extracts, writes
`.complete`, and renames the staging folder to `v1-<hash>`. A `fs2` lock on
`runtime\.lock` serializes two first launches. Cached launches read
`.complete`, check required files are non-empty, and start directly. A new
payload hash creates a new runtime; older `v1-*` runtimes are removed after a
successful launch. User data (`settings.json`, logs, `temp/`, `recordings/`)
lives in `%LOCALAPPDATA%\EasyScribe`, not in the versioned runtime. Portable
mode uses `EasyScribe-data` beside the exe.

**Splash.** A first launch shows `Preparing EasyScribe` with verification,
unpacking and finishing stages; cached launches never show it. `--selftest`
uses the same extraction path but no window, then loads the Silero VAD, sherpa
Whisper ONNX recognizer, whisper.cpp GGUF and diarization models and decodes a
short local WAV.

**Native libraries.** sherpa-onnx **1.12.9** Win x64 static libraries
(SHA-256 `4f3ada021afc97497f7570db622370a43c9b44d47d3dfb649aa0a7133a7afe58`)
match the C API generated by `sherpa-rs` 0.6.8. They use the static MSVC
runtime, so the build sets `RUSTFLAGS=-C target-feature=+crt-static
-C link-arg=advapi32.lib`, `CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded`,
`CMAKE_POLICY_DEFAULT_CMP0091=NEW` and `GGML_NATIVE=OFF`. The explicit
`advapi32` link resolves onnxruntime's ETW telemetry imports and espeak-ng's
registry lookup, which the static archives do not always declare as default
libraries. Linux uses the dynamic sherpa build for development.

**Preview limitation.** `release-static` does **not** enable `vulkan`.
`whisper-rs` links `vulkan-1.dll` as a load-time import; on a machine without
the loader/driver the process fails before `main`, so the CPU fallback can never
run. The preview is CPU-only. A future GPU path needs explicit delay-loading or
a separate worker process.

**CI.** `.github/workflows/rune-release.yml` builds the app and packer together,
checks `dumpbin` for `sherpa|onnxruntime|VCRUNTIME|MSVCP|VULKAN`, requires the
Windows GUI subsystem, runs `--selftest` on the packed exe, verifies the
extracted file list, verifies a cached launch does not rewrite `.complete`,
writes a `.sha256`, and publishes a draft pre-release. `workflow_dispatch`
requires the workflow file to exist on the default branch; the workflow itself
checks out `rune-rewrite`.

---

## 2. Current artifact and install layout (v3.2.3)

### 2.1 Release artifacts

Built by `.github/workflows/build-release.yml` (manual `workflow_dispatch`, input `tag`, default `v3.2.3`):

| Artifact | Built from | Contents | Purpose |
|---|---|---|---|
| `EasyScribe-v3.2.3.exe` | `launcher/launcher.spec` → `dist\launcher.exe`, renamed | PyInstaller ONEFILE launcher + embedded `app.bundle` (~1.5 GB) + `assets/EasyScribe.ico` + `assets/splash.png` | First install; repeat launch fallback |
| `EasyScribe-update-v3.2.3.exe` | same spec rebuilt with an update `app.bundle` | launcher + smaller bundle; `.easyscribe-update.json` inside bundle | Update an existing 3.1.0+ install without re-downloading models/ffmpeg/GGUF |

The release is created as a **draft** (`draft: true`) and marked pre-release when the tag contains `-` (e.g. `v3.0.0-beta2`). Assets are attached by `softprops/action-gh-release@v2`.

### 2.2 How the launcher gets the bundle

- `launcher/launcher.spec` adds `("../app.bundle", ".")` to PyInstaller `datas`, so the bootloader extracts it into `sys._MEIPASS`.
- `_find_bundle()` returns `Path(sys._MEIPASS) / "app.bundle"` when frozen, else `<repo>/app.bundle` in development.
- `app.bundle` is a ZIP made by `Compress-Archive -Path "dist\EasyScribe\*"`; it contains the **contents** of `dist\EasyScribe`, not the `EasyScribe` folder itself.
- `launcher.spec` is `console=False`, `upx=False` (bundle already compressed), with `Splash("../assets/splash.png", ...)` and `version=_version_info("EasyScribe installer", "launcher.exe")`.
- `EasyScribe_whisper.spec` builds the app as ONEDIR: `dist\EasyScribe\EasyScribe.exe` + `_internal\`; models/ffmpeg/whispercpp are copied in later by CI, deliberately **not** packed into `_internal\`.

### 2.3 Installed tree

Default install directory is `<directory containing the launcher exe>\EasyScribe\` (`_exe_dir / "EasyScribe"`). A Browse selection is redirected to `<chosen>\EasyScribe` unless `<chosen>` is already a marked install or already contains `EasyScribe.exe` (pre-marker v2.0.0 install).

```
<launcher_dir>\
  EasyScribe-v3.2.3.exe          the downloaded one-file launcher (kept by the user)
  EasyScribe\                    created by the installer
    EasyScribe.exe               PyInstaller ONEDIR main executable
    .easyscribe-install.json     ownership + version + shortcut/update state
    _internal\                   PyInstaller runtime
      assets\EasyScribe.ico
      assets\EasyScribe.png
      ... Python DLLs, .pyd files, package data ...
    models\
      whisper\
        turbo-encoder.int8.onnx
        turbo-decoder.int8.onnx
        turbo-tokens.txt
      diarization\
        segmentation.onnx
        embedding.onnx
      silero_vad.onnx
    whispercpp\
      whisper-cli.exe
      *.dll
      ggml-large-v3-turbo-q5_0.bin
    ffmpeg\
      ffmpeg.exe
      ffprobe.exe
    LICENSE.txt
    THIRD-PARTY-NOTICES.txt
    licenses\
      ... third-party texts and python-packages\<name>\...
    recordings\                  in-progress PCM + JSON sidecars + settings.json (USER DATA)
    logs\                        EasyScribe_YYYYMMDD_HHMMSS.log (USER DATA)
    temp\                        scratch WAV/JSON; cleaned at start and exit (DERIVED)
    .easyscribe-old\             transient backup during an in-place update; never user data
```

Notes:

- `recordings\` is **not** where finished recordings go by default. `DEFAULT_RECORDINGS_DIR` is `Documents\EasyScribe` (resolved with `SHGetKnownFolderPath` so OneDrive-redirected Documents works). `recordings\` is the crash-safe work directory and holds `settings.json`.
- `logs\` keeps the last `MAX_LOG_FILES = 10` files, named `EasyScribe_<YYYYMMDD_HHMMSS>.log`; file names/progress only, never transcript text.
- `temp\` is created by `config.py` at import and cleaned of `*.wav` and `*.json` at startup and exit.
- `models\` must live next to `EasyScribe.exe` (`BASE_DIR`), not in `_internal\`; `validate_build.py --assembled` fails if `_internal\models` exists.
- `.easyscribe-update.json` exists only **inside** an update bundle; it is never installed.
- The launcher itself is not copied into the install folder.

### 2.4 Ownership and replacement rules

| Path | Owner | Replaced on update? | Preserved by cleanup? |
|---|---|---|---|
| `EasyScribe.exe` | app | yes | no |
| `_internal\` | app | yes | no |
| `models\` | app | yes (full bundle) / kept (update bundle) | no |
| `whispercpp\` | app | yes (full bundle); `.bin` kept by update bundle | no |
| `ffmpeg\` | app | yes (full bundle) / kept (update bundle) | no |
| `LICENSE.txt`, notices, `licenses\` | app | yes | no |
| `.easyscribe-install.json` | installer | preserved/updated | no (cleanup deletes it, then recreates) |
| `recordings\` | **user** | never moved/deleted | **yes** |
| `logs\` | **user** | never moved/deleted | **yes** |
| `temp\` | app/derived | not an app item; left alone by update | no (incomplete-install cleanup deletes it) |
| `.easyscribe-old\` | installer transient | created/deleted during update | deleted by recovery |

The two protected top-level names are exactly `USER_DATA_DIRS = frozenset({"recordings", "logs"})`.

---

## 3. Install marker `.easyscribe-install.json`

**Location:** `<install_dir>\.easyscribe-install.json`
**Purpose:** (a) proof that the installer created/owns the directory, (b) installed-version record for update decisions, (c) one-time shortcut flag, (d) crash-recovery journal flag during an update.
**Encoding:** UTF-8, default `json.dumps` output (single line, default `", "`/`": "` separators, no indentation, no trailing newline). Writes are **not** atomic and not fsynced; a torn write is treated as “unknown” by readers.

### 3.1 Fields

| Field | Type | Written by | Required | Semantics |
|---|---|---|---|---|
| `app` | string | `_write_marker`, `_set_marker_version`, `_mark_shortcuts_created`, `_set_updating_flag` | effectively yes | Always `"EasyScribe"`; identifies the marker |
| `version` | string | same | effectively yes | Version of the launcher that last wrote it; compared by `_version_key` |
| `shortcuts_created` | boolean | `_mark_shortcuts_created` | no | `true` once shortcut creation was attempted; never cleared |
| `updating` | boolean | `_set_updating_flag` | no | `true` only between “old files moved to backup” and “new version recorded”; clears on success/rollback |

No schema version field exists. Unknown fields are preserved by `_set_marker_version`, `_mark_shortcuts_created` and `_set_updating_flag`, because each reader loads the JSON dict and only adds/removes its own key.

### 3.2 Exact write behaviours

```python
# fresh install; does NOT overwrite an existing marker
_write_marker(install_dir):
    if not marker.is_file():
        write {"app": "EasyScribe", "version": VERSION}      # single-line default JSON

# after a successful update; preserves shortcuts_created and unknown fields
_set_marker_version(install_dir):
    data = read_json_or_{}
    data["app"] = "EasyScribe"
    data["version"] = VERSION
    data.pop("updating", None)
    write data

# once, after _create_shortcuts() (even if individual shortcut creation failed)
_mark_shortcuts_created(install_dir):
    data = read_json_or_{"app": "EasyScribe", "version": VERSION}
    data["shortcuts_created"] = True
    write data

# update journal
_set_updating_flag(install_dir, True/False):
    data = read_json_or_{"app": "EasyScribe", "version": "0"}
    set/remove data["updating"]
    write data
```

### 3.3 Example states

```json
{"app": "EasyScribe", "version": "3.2.3"}
```
Fresh install, before first launch.

```json
{"app": "EasyScribe", "version": "3.2.3", "shortcuts_created": true}
```
After the first `_start_app()` / install completion.

```json
{"app": "EasyScribe", "version": "3.1.0", "shortcuts_created": true, "updating": true}
```
Mid-update (old version still recorded until `_set_marker_version`).

```json
{"app": "EasyScribe", "version": "0", "updating": true}
```
Only if `_set_updating_flag` had to start from an unreadable/corrupt marker. After a rollback this can remain as `version: "0"`; the next run treats it as unknown/older and offers an update again.

### 3.4 Reader semantics

| Function | Behaviour |
|---|---|
| `_installed_version` | Reads JSON; returns `data["version"]` only if it is a non-empty string; otherwise `None` (covers missing file, bad JSON, non-dict, wrong type, empty string) |
| `_needs_update` | `None` ⇒ `True`; else `_version_key(installed) < _version_key(VERSION)` |
| `_shortcuts_created` | Missing/bad marker ⇒ `False`; else `bool(data.get("shortcuts_created"))` |
| `_update_in_progress` | Missing/bad marker ⇒ `False`; else `bool(data.get("updating"))` |
| `_safe_to_clean` | Missing dir ⇒ `True`; complete install (`EasyScribe.exe` exists) ⇒ `False`; marker exists ⇒ `True`; otherwise `False` |

**Safety invariant:** the marker is the *only* permission to delete a directory as an incomplete install. An unmarked, non-empty directory MUST NEVER be cleaned. A marked directory with `EasyScribe.exe` is treated as a complete install and is never wiped. The marker is not a security boundary (a user can create it); it prevents accidental data loss, not a determined attacker.

---

## 4. Launcher behaviour (normative current behaviour)

### 4.1 Entry point decision tree

`main()` in `launcher/launcher.py`:

```
default_dir = <dir containing launcher exe> / "EasyScribe"
bundle      = _find_bundle()

if not (default_dir / "EasyScribe.exe").is_file() and bundle is not None:
    if bundle is an update bundle (has .easyscribe-update.json):
        default_dir = _install_dir_from_shortcut() or default_dir

_recover_interrupted_update(default_dir)

if (default_dir / "EasyScribe.exe").is_file() and not _needs_update(default_dir):
    # current or newer install: no installer window at all
    _wait_until_started(_start_app(default_dir))   # 15 s WaitForInputIdle
    _close_splash()
    return

# first run, or an older install: show InstallerApp and wait for a click
InstallerApp(install_dir=default_dir).mainloop()
```

Key consequences:

- There is **no automatic update**. An older install always shows the installer window and waits for **Update and open** (`CLAUDE.md` Rule 18).
- A **newer** install is launched silently; it is never downgraded.
- A pre-marker v2.0.0 install has no marker ⇒ `_needs_update` is true ⇒ installer is shown, and the user can update in place.
- An update `.exe` normally lives in `Downloads`, so it consults the Start Menu/Desktop shortcut to find the installed folder. If none is found, the default `<exe_dir>\EasyScribe` is used and, because an update bundle cannot do a new install, the Install button is disabled until the user browses to a real install.
- `_recover_interrupted_update` runs before the decision, so a crashed update is rolled back before any launch/update.

### 4.2 Path resolution and data-loss guards

| Function | Rule |
|---|---|
| `_resolve_install_dir(chosen)` | If `chosen` contains `.easyscribe-install.json` **or** `EasyScribe.exe`, use `chosen`; otherwise use `chosen / "EasyScribe"` |
| `_safe_to_clean(install_dir)` | `not exists` ⇒ `True`; `EasyScribe.exe` exists ⇒ `False`; marker exists ⇒ `True`; else `False` |
| `_clean_incomplete_install(install_dir)` | Delete every top-level item except `recordings` and `logs`; keep those two. Recreates marker afterwards via `_write_marker` |
| `_main_exe(install_dir)` | `<install_dir>/EasyScribe.exe` |

The Browse dialog lets the user pick any existing folder. The redirect-to-subdirectory plus the marker rule are the fix for audit §3.3: selecting `Documents` must install into `Documents\EasyScribe`, and a failed install must not delete `Documents`.

### 4.3 New install (`_extract_thread`)

1. If `install_dir` exists and has no `EasyScribe.exe`:
   - If `_safe_to_clean(install_dir)`: status “Cleaning incomplete previous install…”, then `_clean_incomplete_install` (keeps `recordings`, `logs`).
   - Else if the directory is non-empty: error
     `'<dir>' is not empty and was not created by EasyScribe. Please choose an empty or new folder.`
     and stop **without deleting anything**.
2. `install_dir.mkdir(parents=True, exist_ok=True)`.
3. `_write_marker(install_dir)` (does not overwrite an existing marker).
4. Open the bundle ZIP; iterate **all** members in archive order; `zf.extract(member, install_dir)`; report progress every 200 members and at the end (`int(100 * i / total)`).
5. On success, queue `done`; UI shows “Done. Opening EasyScribe… Next time, open it from the Desktop or Start menu shortcut.”; after 1500 ms `_do_launch`.
6. On exception: if `_safe_to_clean(install_dir)` and the dir exists, `_clean_incomplete_install`; then show the error. Because the marker was written in step 3, a failed extraction is cleanable on the next attempt. An unmarked pre-existing directory is never cleaned.

The CI pre-creates an empty `recordings/` in the assembled folder so the bundle normally carries an empty `recordings/` entry; it contains no user data.

### 4.4 Launch path (`_start_app` / `_do_launch`)

```
_start_app(install_dir):
    _write_marker(install_dir)          # backfill a pre-marker v2.0.0 install
    _ensure_shortcuts(install_dir)      # once per marker; best effort
    Popen([install_dir / "EasyScribe.exe"], cwd=install_dir)
```

- `_do_launch` disables the installer inputs, sets “Opening EasyScribe…” (unless it is the post-install launch, which keeps “Done” and the full bar), starts the app, and waits on a background thread for `WaitForInputIdle(proc, 15000)`.
- The installer window destroys itself once the wait finishes (or times out); the splash is closed by `_close_splash()`.
- The silent `main()` path does not create a window; it waits up to 15 s for the app's input-idle state, then closes the splash. If the app never starts, the user sees only the splash closing — there is no installer error in this path.
- `_start_app` does not check the process exit code after launch.

### 4.5 Version comparison (`_version_key`)

Regex:

```python
r"^v?(\d+(?:\.\d+)*)(?:-?([a-zA-Z]+)\.?(\d*))?$"
```

Algorithm:

1. Strip whitespace.
2. Optional leading `v`.
3. Numeric part: one or more dot-separated integers; any number of components.
4. Optional pre-release label: optional `-`, letters, optional `.`, optional digits. Examples accepted: `3.0.0-beta2`, `3.0.0beta2`, `3.0.0-beta.2`, `3.0.0-rc1`, `v2.1`.
5. If no match: key `((0,), (0, 0, 0))` — sorts first.
6. Numeric tuple: pad to at least 3 components with `0`; longer tuples are kept and compare lexicographically.
7. No label: stage tuple `(9, 0, 0)` — a final release.
8. Label: `alpha`/`a` ⇒ 1, `beta`/`b` ⇒ 2, `rc` ⇒ 3, anything else ⇒ 0; second element is the integer suffix (0 if absent); third is 0.
9. Compare `(numeric_tuple, stage_tuple)` lexicographically.

Examples (ascending):

| Version | Key |
|---|---|
| `junk` | `((0,), (0, 0, 0))` |
| `2.0.0` | `((2, 0, 0), (9, 0, 0))` |
| `v2.1` | `((2, 1, 0), (9, 0, 0))` |
| `3.0.0-alpha1` | `((3, 0, 0), (1, 1, 0))` |
| `3.0.0-beta1` | `((3, 0, 0), (2, 1, 0))` |
| `3.0.0-beta2` | `((3, 0, 0), (2, 2, 0))` |
| `3.0.0-rc1` | `((3, 0, 0), (3, 1, 0))` |
| `3.0.0` | `((3, 0, 0), (9, 0, 0))` |
| `3.0.1` | `((3, 0, 1), (9, 0, 0))` |
| `3.1.0` | `((3, 1, 0), (9, 0, 0))` |
| `10.0.0` | `((10, 0, 0), (9, 0, 0))` |

`3.0` equals `3.0.0`; `3.0.0.0` is greater than `3.0.0`; `3.0.0-dev1` sorts before `alpha1` because unknown labels use stage 0. A version that cannot be parsed sorts before everything, including `0.0.0`.

`_needs_update(install_dir)` is `installed is None or _version_key(installed) < _version_key(VERSION)`. Equal or newer ⇒ no update. `VERSION` must exactly match `APP_VERSION`; `tests/test_version.py` enforces that.

### 4.6 In-place update (`_update_install`)

Preconditions: `install_dir\EasyScribe.exe` exists and `_needs_update` is true; the bundle is either a full bundle (`_bundle_keeps` returns `None`) or an update bundle with a valid `.easyscribe-update.json`.

1. `keep = _bundle_keeps(bundle) or {}`
2. `_check_kept_files(install_dir, keep)`:
   - For every `rel -> sha256`, `install_dir/rel` must be a file and its SHA-256 must equal the recorded value.
   - First missing/changed file ⇒ `NeedsFullInstallError(rel)`. This happens **before any file is moved**, so nothing changes.
3. If `.easyscribe-old\` exists, delete it (it only ever holds app files; a leftover from a finished update is safe to remove).
4. Open the ZIP; compute `app_items` = top-level names in the bundle excluding `recordings`, `logs`, `.easyscribe-install.json`, `.easyscribe-update.json`. This is the set of app items to replace.
5. Create `.easyscribe-old\`; set marker `"updating": true`.
6. For each `name` in **sorted** `app_items`: if `install_dir/name` exists, `os.replace` it into `.easyscribe-old/name`.
   - On `OSError`: `_restore_backup`, clear `updating`, raise `UpdateBlockedError` (typically the app is still running; Windows refuses to rename a folder containing a running exe/DLL).
7. Extract every ZIP member whose top-level is in `app_items` to `install_dir`; report progress every 200 members and at the end.
8. For every kept file whose top-level is in `app_items` (e.g. `whispercpp/ggml-...bin` when `whispercpp/` was replaced): `_carry_over(backup/rel, install_dir/rel)` — `os.link` (instant, same volume), falling back to `shutil.copy2`.
   - Kept files in folders **not** in `app_items` (e.g. `models/`, `ffmpeg/` in an update bundle) were never moved; they stay in place.
9. Ensure `install_dir/recordings` and `install_dir/logs` exist (`mkdir(exist_ok=True)`); they are never overwritten from the bundle.
10. Any `BaseException` during extraction/carry-over: `_restore_backup`, clear `updating`, re-raise (previous version kept).
11. Success: `_set_marker_version` (writes current `VERSION`, removes `updating`, preserves `shortcuts_created` and unknown fields), delete `.easyscribe-old\`, call `_refresh_shell_icons()` (`SHChangeNotify(SHCNE_ASSOCCHANGED, ...)`).

**Important current limitation:** only top-level names present in the **new** bundle are moved aside. An old app file/dir that no longer exists in the new bundle is left in place (stale file). This is harmless for the shipped bundle but means the install is not a byte-exact mirror of the bundle. The target payload model removes this class of problem by extracting a fresh versioned runtime.

### 4.7 Rollback and interrupted-update recovery

`_restore_backup(install_dir)`:

```
if backup dir does not exist: return
for each item in backup.iterdir():
    target = install_dir / item.name
    if target is a dir:  rmtree(target, ignore_errors=True)
    elif target exists:  unlink()
    os.replace(item, target)
rmtree(backup, ignore_errors=True)
```

`_recover_interrupted_update(install_dir)`:

| Marker state | Backup dir | Action |
|---|---|---|
| `updating: true` | exists | `_restore_backup`, then clear `updating` |
| `updating: true` | missing | clear `updating` (nothing to restore) |
| no `updating` | exists | update completed but backup removal was interrupted: delete backup |
| no `updating` | missing | nothing |

OSError during recovery is swallowed (`pass`); the next run tries again. The function runs in `main()` for the default directory and in `_on_action()` for the user-selected directory.

Failure semantics:

| Failure | Result |
|---|---|
| App still open / file locked during step 6 | `UpdateBlockedError`; all moved items restored; marker `updating` cleared; old version fully usable |
| Bad ZIP / disk full / any exception during extraction | `_restore_backup`; old files back; `updating` cleared |
| Crash/power loss after old files moved | Next start sees `updating: true`; restores old files |
| Crash after new version recorded but before backup delete | Next start sees no `updating`; deletes leftover backup |
| Kept file missing/changed | `NeedsFullInstallError` before any move; nothing changed |
| Corrupt marker during update | `_set_updating_flag` writes `{"app":"EasyScribe","version":"0","updating":true}`; after success version is corrected; after rollback version may stay `"0"` and trigger another update offer |

### 4.8 Shortcuts

Created only on Windows (`sys.platform == "win32"`), once per install, best effort.

| Shortcut | Path | Target | Working dir | Icon |
|---|---|---|---|---|
| Desktop | `%USERPROFILE%\Desktop\EasyScribe.lnk` | `<install>\EasyScribe.exe` | `<install>` | `<target>,0` |
| Start Menu | `%APPDATA%\Microsoft\Windows\Start Menu\Programs\EasyScribe.lnk` | same | same | same |

Implementation: `subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], check=True, capture_output=True, timeout=15)` with `WScript.Shell.CreateShortcut`; single quotes in paths are doubled via `_ps_quote`. `_create_shortcut` catches **all** exceptions and returns silently. `_create_shortcuts` creates parent directories first.

`_ensure_shortcuts`:

```
if _shortcuts_created(install_dir): return
_create_shortcuts(install_dir)
_mark_shortcuts_created(install_dir)
```

Because the flag is set even when PowerShell failed, shortcuts are never retried automatically. `_ensure_shortcuts` is called on every `_start_app`, so a pre-existing install gets shortcuts on its next launch.

`_install_dir_from_shortcut()` (used only by update `.exe` when the default install is absent):

1. Check Start Menu then Desktop `.lnk`.
2. Read `TargetPath` with the same PowerShell/WScript.Shell call (`creationflags=0x08000000`, no window).
3. If the target's parent contains `EasyScribe.exe`, return that parent.
4. First match wins; `None` if no usable shortcut.

Limitations: Desktop is taken from `%USERPROFILE%\Desktop`, not the shell known folder, so a OneDrive-redirected Desktop may be missed; `%APPDATA%`/`%USERPROFILE%` are assumed present; a moved install leaves stale shortcuts; no shortcut repair.

`_refresh_shell_icons()` calls `SHChangeNotify(0x08000000, 0, None, None)` after an update so Explorer does not keep the old cached icon.

### 4.9 Splash

`launcher.spec` includes PyInstaller `Splash("../assets/splash.png", binaries=..., datas=..., always_on_top=False)` with no `text_pos` (so the bootloader does not print every extracted file name). The image is 440×150, generated by `assets/make_splash.py`, with the EasyScribe mark, “EasyScribe”, “Getting ready…”, and “The first start can take a minute. Please wait.”

`_close_splash()` imports `pyi_splash` (only present in the built exe) and calls `close()`, swallowing all exceptions. It is called:

- 200 ms after the installer window is constructed (`InstallerApp.__init__`), and
- after the silent repeat-launch wait in `main()`.

This exists because the PyInstaller bootloader is silent while it unpacks ~1.5 GB before Python runs (`CLAUDE.md` Rule 17). In the Rust target there is no PyInstaller bootloader; the first-run loading window replaces it.

### 4.10 Installer UI state machine

`InstallerApp(ctk.CTk)`, fixed size, light theme, title `EasyScribe <VERSION>`.

On every `_refresh_state()`:

| Condition | Status text | Button |
|---|---|---|
| update bundle + no `EasyScribe.exe` in chosen dir | “This file updates an installed EasyScribe to 3.2.3. It cannot install it new. Select Browse and choose your EasyScribe folder, or download the full EasyScribe-v3.2.3.exe.” | `Install`, disabled |
| `EasyScribe.exe` exists + `_needs_update` | “EasyScribe <installed> is installed in this folder. Select Update and open to update it to 3.2.3. Your recordings are kept. To install in a different folder, select Browse.” | `Update and open` |
| `EasyScribe.exe` exists + current/newer | “EasyScribe is already installed in this folder.” | `Open EasyScribe` |
| otherwise | “EasyScribe will be installed in this folder.” | `Install` |

`_on_action()` re-runs `_recover_interrupted_update`, then:

```
if main exe exists and _needs_update:  _do_install(update=True)
elif main exe exists:                   _do_launch()
elif not update-only:                   _do_install(update=False)
```

`_set_inputs(False)` disables the button, Browse and directory entry, sets `_busy = True`, shows the progress bar. `_busy` makes `_on_action` and `_refresh_state` no-ops, so a double click/Enter cannot start a second install. `Return` is bound to `_on_action`.

Background threads communicate only through `queue.Queue` messages:

| Message | UI effect |
|---|---|
| `status` | Set status text |
| `progress` | `bar.set(pct/100)`, “Copying the app files… <pct>%” |
| `done` | Full bar, “Done. Opening EasyScribe… Next time, open it from the Desktop or Start menu shortcut.”, after 1500 ms `_do_launch` |
| `error` | “Could not finish: <text>”, re-enable inputs |

`_poll_queue` runs every 80 ms. The `done` branch calls `_do_launch` with `after_install = self._busy` true, so the “Done” text and full bar are kept while the app starts.

### 4.11 Error messages (exact strings)

| Situation | Message |
|---|---|
| Bundle missing in launcher | `The app files are missing from this installer. Download it again.` |
| Update-only, no install | `This file updates an installed EasyScribe to {VERSION}. It cannot install it new.\nSelect Browse and choose your EasyScribe folder, or download the full EasyScribe-v{VERSION}.exe.` |
| Update start | `Checking the installed files, then updating to {VERSION}…` |
| Missing/changed kept file | `this update needs a file that is missing or different in your install ({rel}).\nDownload the full EasyScribe-v{VERSION}.exe instead. Nothing was changed.` |
| App open / locked file | `EasyScribe is still open, or a file in its folder is in use.\nClose EasyScribe, then select Update and open again.` |
| Other update failure | `{exc}\nThe previous version was kept.` |
| UI wrapper for any worker error | `Could not finish: {value}` |
| Unmarked non-empty folder | `'{install_dir}' is not empty and was not created by EasyScribe. Please choose an empty or new folder.` |
| Cleaning status | `Cleaning incomplete previous install…` |
| Install progress | `Copying the app files… {pct}%` |
| Install done | `Done. Opening EasyScribe…\nNext time, open it from the Desktop or Start menu shortcut.` |
| Opening | `Opening EasyScribe…` |

### 4.12 Invariants the rewrite must not lose

1. **Never delete a directory the app did not create.** Marker/ownership check first; otherwise refuse.
2. **Never delete or move `recordings/` or `logs/`** during install, update or incomplete-install cleanup.
3. **Never downgrade automatically.** A current or newer install is launched, not replaced.
4. **Never change an install without a click.** No background/automatic update; the user chooses Update and open.
5. **Check before touching.** Kept-file hashes are validated before any move.
6. **Back up before replacing; restore on any failure.** Move old app items to `.easyscribe-old`, extract, and roll back on error or power loss.
7. **Recover on next start.** `updating: true` plus a backup means “put the old files back”.
8. **A locked file stops the update before data is replaced.**
9. **User data survives interrupted installs and updates.**
10. **No network access in the launcher or app** (Rule 13).

### 4.13 Known current limitations / edge cases (do not accidentally “fix” in the target without deciding)

| Area | Current behaviour / limitation |
|---|---|
| Damaged update metadata | If `.easyscribe-update.json` exists but `keep` is missing/empty/not a dict, `_bundle_keeps` raises `ValueError`. `InstallerApp.__init__` catches it and sets `_is_update_only = False`, so the UI may offer Install and `_extract_thread` can extract the partial bundle as if it were a full install — leaving models/ffmpeg missing until the app's dependency check fails. The target payload verifier eliminates this class. |
| Free disk space | The launcher does **not** check free space before extraction. A disk-full error is caught and rolled back (update) or cleaned (install), but the message is generic. `MIN_FREE_DISK_BYTES` is only used by the GUI's output-folder check. The target should check before extracting. |
| Bundle authenticity | No signature/hash of the full bundle or launcher exe; only update kept-file hashes are checked. Trust is the download source. The target adds a payload SHA-256 (integrity) and recommends Authenticode (authenticity). |
| Silent launch failure | In the silent repeat-launch path, if `EasyScribe.exe` fails to start, `_wait_until_started` times out after 15 s, the splash closes, and the user sees no error. The installer path does not catch `_start_app` exceptions either. |
| Shortcut retry | `_mark_shortcuts_created` runs even if PowerShell failed, so shortcuts are never retried. |
| Incomplete-install cleanup | `_clean_incomplete_install` deletes **everything** except `recordings` and `logs` inside a marked install with no `EasyScribe.exe`, including unknown files the user may have put there. This is safe only because the marker asserts installer ownership. |
| Stale app files | Update moves aside only top-level app items present in the **new** bundle; old app files/dirs not in the new bundle remain. |
| ZIP extraction | `zipfile.extract` sanitizes absolute paths and `..` components, but the bundle is unauthenticated: an attacker who can replace the exe can replace app files. The target extractor must still reject `..`/absolute paths explicitly and never follow payload symlinks. |
| Version parser | Accepts non-semver forms (`v2.1`, `3.0.0.1`, `3.0.0beta2`) and maps unknown labels to stage 0; Windows numeric versions collide for all pre-releases of the same numeric version. |
| Recovery errors | `_recover_interrupted_update` swallows `OSError`; if restore fails, the next start retries, but the install may be unusable in the meantime. |
| Upgrade path | Pre-marker v2.0.0 installs are detected by the presence of `EasyScribe.exe`; `_write_marker` backfills `version = VERSION`, which could mislabel an old app as current if called outside the guarded `main()` path. |

### 4.14 Launcher function inventory and target disposition

| Function | Current role | Target disposition |
|---|---|---|
| `_set_window_icon` | Set installer `.ico` | Drop; PE icon |
| `_find_bundle` | Locate embedded/dev `app.bundle` | Drop; footer payload |
| `_main_exe` | `<install>/EasyScribe.exe` | Drop; one exe |
| `_resolve_install_dir` | Redirect Browse choice to `<chosen>/EasyScribe` | Drop; no install dir |
| `_safe_to_clean` | Marker-based ownership check | Replace with cache/staging safety; never delete unowned data |
| `_version_key` | Version ordering | Optional legacy-marker reader; otherwise drop |
| `_installed_version` | Read marker version | Drop (legacy import only) |
| `_needs_update` | Compare installed vs launcher | Drop; each exe carries its payload |
| `_write_marker` | Create ownership/version marker | Drop |
| `_set_marker_version` | Record successful update | Drop; `.complete` manifest |
| `_shortcuts_created` | Read one-time flag | Drop |
| `_mark_shortcuts_created` | Set one-time flag | Drop |
| `_ps_quote` / `_create_shortcut` / `_create_shortcuts` | PowerShell `.lnk` creation | Drop automatic creation; optional user-invoked shortcut |
| `_install_dir_from_shortcut` | Find install for update exe | Drop; no update exe |
| `_ensure_shortcuts` | Create once per install | Drop |
| `_refresh_shell_icons` | `SHChangeNotify` after update | Drop; no in-place update |
| `UpdateBlockedError` | Locked app files during update | Concept: surface locked cache/file errors; no backup rollback |
| `_bundle_app_items` | Top-level app members | Concept: payload file list |
| `NeedsFullInstallError` | Kept file missing/changed | Drop; payload verification |
| `_bundle_keeps` | Read `.easyscribe-update.json` | Drop |
| `_file_sha256` | Hash a file | Keep; payload/entry hashing |
| `_check_kept_files` | Verify kept files | Concept: verify every extracted entry |
| `_carry_over` | Hard-link/copy kept file from backup | Drop; fresh versioned runtime |
| `_restore_backup` | Roll back old app files | Concept: delete staging on failure; old runtime untouched |
| `_set_updating_flag` / `_update_in_progress` | Crash journal | Drop; `.complete` + staging dirs |
| `_recover_interrupted_update` | Restore or clean after crash | Concept: clean stale staging at startup |
| `_update_install` | In-place replace with backup | Replace with extract-new-runtime + activate + GC |
| `_clean_incomplete_install` | Wipe app files, keep recordings/logs | Concept: delete staging only; never user data |
| `InstallerApp` | Installer GUI/state machine | Replace with first-run progress window; no choices |
| `_wait_until_started` / `_start_app` / `_close_splash` | Launch installed app/wait/splash | Replace with direct GUI start; first-run loading window |
| `main` | Decide launch/install/update | Replace with footer read → cache check/extract → GUI |

---

## 5. Update bundle and hash validation

### 5.1 Builder (`assets/make_update_bundle.py`)

Usage: `python assets/make_update_bundle.py <assembled app folder> <output zip>`

Constants:

```python
UPDATE_INFO_FILENAME = ".easyscribe-update.json"   # same as launcher.py
KEEP = ("models/*", "ffmpeg/*", "whispercpp/*.bin")
SKIP_TOP = ("recordings",)
```

Algorithm:

1. Require `<app_dir>/EasyScribe.exe` to exist; otherwise `SystemExit`.
2. Walk every file under `app_dir` recursively, sorted.
3. Skip any file whose first path component is `recordings`.
4. If the POSIX relative path matches any `KEEP` pattern (`fnmatch`), compute its SHA-256 and record `rel -> hex` in `keep`; do **not** write it to the ZIP.
5. Otherwise write it to the ZIP with `ZIP_DEFLATED, compresslevel=6`.
6. If `keep` is empty, `SystemExit("No files to keep: is this the assembled app folder?")`.
7. Write `.easyscribe-update.json` as `json.dumps({"keep": keep}, indent=1)`.
8. Print kept file count/MB and output size.

What is kept and why:

| Pattern | Files | Reason |
|---|---|---|
| `models/*` | `models/whisper/*` (encoder/decoder/tokens), `models/diarization/*`, `models/silero_vad.onnx` | Downloaded from versioned, byte-stable URLs; ~1.07 GB |
| `ffmpeg/*` | `ffmpeg.exe`, `ffprobe.exe` | Pinned GyanD 7.1.1 release; ~150 MB |
| `whispercpp/*.bin` | `ggml-large-v3-turbo-q5_0.bin` | Pinned HuggingFace file; 574,041,195 bytes |

What is always shipped in an update bundle: `EasyScribe.exe`, `_internal/` (Python runtime and package DLLs), `whispercpp/whisper-cli.exe`, `whispercpp/*.dll`, `licenses/`, `LICENSE.txt`, `THIRD-PARTY-NOTICES.txt` (whatever is in the assembled folder). `whisper-cli.exe` and its DLLs are **not** kept because CI rebuilds them when the cache expires, so their bytes change between releases.

### 5.2 Launcher validation

`_bundle_keeps(bundle)`:

```
open zip
if UPDATE_INFO_FILENAME not in namelist: return None          # full bundle
keep = json.loads(zf.read(UPDATE_INFO_FILENAME)).get("keep")
if keep is not a dict or empty: raise ValueError("The update information in this installer is damaged.")
return keep
```

`_check_kept_files(install_dir, keep)`:

```
for rel, expected_sha in keep.items():
    path = install_dir / rel
    if not path.is_file() or _file_sha256(path) != expected_sha:
        raise NeedsFullInstallError(rel)
```

`_file_sha256` reads 1 MiB chunks. There is no size check beyond reading; symlinks are followed by `is_file()`/`open` (an edge case, not a supported layout).

`_carry_over(src, dst)`:

```
dst.parent.mkdir(parents=True, exist_ok=True)
try: os.link(src, dst)          # same volume: instant, no extra space
except OSError: shutil.copy2(src, dst)
```

### 5.3 Update-bundle state machine

| Bundle | Installed files | Result |
|---|---|---|
| Full bundle (no `.easyscribe-update.json`) | anything | All top-level app items replaced; `keep = {}` |
| Update bundle | every kept file present and SHA-256 matches | App items replaced; kept files in replaced folders hard-linked/copied from backup; kept folders left untouched |
| Update bundle | any kept file missing/changed | `NeedsFullInstallError`; **nothing changed**; user must download the full exe |
| Update bundle | target has no `EasyScribe.exe` | Cannot install new; Install disabled; user must browse to an install or use the full exe |
| Any bundle | app running / locked file | `UpdateBlockedError`; old files restored; close app and retry |

### 5.4 Security and integrity properties

- The launcher verifies **kept-file hashes** against the update manifest. It does **not** verify a signature or hash of the bundle/exe itself; trust is the download source and, eventually, code signing.
- The manifest is not signed. An attacker who can replace the update exe can replace any app file. This is the same trust model as the full exe.
- If a kept file's download URL or pinned version changes, existing installs will fail the hash check and the release notes must tell users to use the full `.exe`. This is why the workflow comment says “If a download URL or version above changes, users need the full .exe for that release (say so in the release notes).”
- `whisper-cli.exe`/DLLs are always shipped because their build is not byte-stable across CI cache expiry.

---

## 6. Single-.exe history and the explicit rewrite goal

### 6.1 Timeline

| Version / commit | Distribution model | What changed / why |
|---|---|---|
| v1.x (`v1.0` tag) | **Two ZIPs** per release: `EasyScribe-<tag>-app.zip` (~1.1 GB) and `EasyScribe-<tag>-model.zip` (~1.5 GB) | First install extracts both into one folder; updates re-extract the app ZIP. No single exe. |
| v2.0.0 (`f17ac7f`) | **7-Zip SFX single exe**: SFX → extracts `launcher.exe` + `app.bundle` to `%TEMP%`; launcher → extracts app to `%LOCALAPPDATA%` | First “single .exe” attempt. Root cause of its failure: the SFX runs `launcher.exe` from `%TEMP%`, so `sys.executable` points into the temp directory and the launcher cannot know where the user placed the original `.exe`. Portability broken. |
| `ce8ecc8` (v2.0.0 line) | **PyInstaller ONEFILE launcher** with embedded `app.bundle`; extracts to `<exe_dir>\EasyScribe\` | Eliminated 7-Zip SFX and AppData. `sys.executable` is now the user's original path, so install-next-to-exe works on any drive/USB/shared folder. This is the architecture still used in v3.2.3. |
| `63235e1` | GUI installer + silent repeat-run | Replaced console print loop with a CustomTkinter installer; on repeat runs (installed exe present) skip GUI and launch silently. |
| Phase 8.3 (`cfeafa3`, pre-v3.0.0) | “Installer semantics” | Audit §7.10: repeat launches re-extract the whole bundle. Action plan §8.3 option (a): keep ONEFILE, create Desktop/Start-Menu shortcuts to the installed exe, tell the user to launch from there. Option (b) sidecar bundle explicitly deferred. |
| v3.0.0-beta1 (`367d798`; shipped in beta2 `9d6cbd4`) | Version-aware in-place updates | Marker version comparison, `.easyscribe-old` backup, rollback, `updating` journal, “close EasyScribe” handling, AppUserModelID/shell icon refresh. |
| v3.0.0-beta3 (`7d09063`) | Splash during one-file unpack | PyInstaller `Splash` shows “Getting ready” because the bootloader is silent for up to a minute (`CLAUDE.md` Rule 17). |
| v3.2.0 (`01f5daa`) | Small update `.exe` | `make_update_bundle.py` omits models/ffmpeg/GGUF, lists SHA-256, launcher validates and keeps installed copies. |
| v3.2.3 (`8323516`) | Current | Installer GUI, version-aware update, full + update exe, shortcuts, splash. |

### 6.2 Audit §7.10 (verbatim substance)

> Restructure the single-exe so repeat launches don't re-extract the embedded ~1 GB bundle (PyInstaller ONEFILE semantics, §2). Options: ship `app.bundle` as a sidecar appended after the exe (read by offset), or a true SFX (NSIS/7z) where repeat-run is a no-op launch.

The audit’s §2 scorecard is blunt: “One exe that self-extracts next to itself, no admin/AppData. But the launcher is PyInstaller ONEFILE with the ~700 MB–1 GB `app.bundle` embedded — the bootloader unpacks **the whole thing to `%TEMP%` on every launch**, including repeat runs that just relaunch the installed copy. Repeat runs should take ~1 s; instead each double-click costs a multi-hundred-MB temp extraction.”

Costs of the current model:

- First run: ~1.5 GB written to `%TEMP%`, then ~1.5 GB more written to the install folder.
- Every later run: ~1.5 GB written to a new `%TEMP%\_MEIxxxx` directory, then deleted; disk churn, antivirus scanning, slow start, `%TEMP%` bloat.
- The installed folder is required for normal fast use; the downloaded exe is a one-time installer. This is why §8.3 added shortcuts.
- `app.bundle` and `app.bundle`-inside-exe duplicate storage (the user keeps the big exe plus the installed folder).
- The model is still “installer”, just without an install wizard; deleting the installed folder forces a reinstall.

### 6.3 Action plan §8.3 (verbatim substance)

> **(a) Installer semantics (ADOPT THIS — recommended, simplest):** keep ONEFILE, but on first run create Desktop/Start-Menu shortcuts to the *installed* `EasyScribe.exe` and tell the user to launch from there; the big exe is a one-time installer. Honest, ~20 lines ... Extend `tests/test_launcher_safety.py` for any new path logic.
>
> **(b) Sidecar bundle:** more moving parts; do not implement — note as the revisit option if the user rejects (a)’s UX after smoke testing.

Phase 8.3 was implemented in `cfeafa3` exactly as option (a): shortcuts, marker `shortcuts_created`, README installer flow. It is a **UX workaround**, not a true single-exe application.

### 6.4 The explicit rewrite goal

The rewrite requirement is stronger than §8.3(a):

> **One fully self-contained Windows `.exe`. Double-click it; the app opens. No installer, no install-location picker, no shortcut requirement, no version compare, no `app.bundle`, no `%TEMP%` re-extraction. The exe can be moved or copied; it still works. All functionality of v3.2.3 remains, fully offline.**

“Self-contained” means the exe alone is sufficient to run; the extracted runtime is a **derived cache**, not an installation. It can be deleted and recreated, and it never owns user data. This is the correct end state for the product: the user downloads one file and runs it. The payload is an implementation detail, not an installation.

### 6.5 Why models and tools must be real files on disk

The payload cannot be mounted as a virtual filesystem for the engines used by EasyScribe:

| Component | Interface | Why a real path is required |
|---|---|---|
| sherpa-onnx (Whisper ONNX, Silero VAD, diarization) | C API configs take `const char *` model paths; ONNX Runtime opens/maps the files | No memory-buffer constructor in the supported C API surface; `sherpa_onnx` Python wrapper and C API both take filenames |
| whisper.cpp / `whisper-rs` | `whisper_init_from_file_with_params()` takes a GGUF path; ggml mmaps it | No in-memory model API in the stable interface used here |
| `ffmpeg.exe` / `ffprobe.exe` | Separate processes | Windows `CreateProcess` needs a real executable path; ffmpeg reads local files only (`-protocol_whitelist file`) |
| Native DLLs (sherpa-onnx, ONNX Runtime, whisper.cpp) | Windows loader | `LoadLibrary`/import resolution needs files on disk (or a loaded module path), not a ZIP member |
| Licenses | User-readable files | License compliance expects the texts to be shipped as files; also easier to open from the app |

Therefore **extract-once is mandatory**. The only question is where and how safely.

### 6.6 Design alternatives

| Option | Repeat launch | First run | Verdict |
|---|---|---|---|
| PyInstaller ONEFILE (current) | re-extract ~1.5 GB to `%TEMP%` | two extractions | **Reject**: audit §7.10 |
| 7-Zip/NSIS SFX (v2.0.0) | temp extraction / installed copy; SFX path problems | temp extraction + AppData | **Reject**: third-party stub, AV, path/portability issues |
| Sidecar `app.bundle` next to exe (audit option) | read by offset, extract once | one extraction | Viable, but violates “one file”; users can separate/delete the sidecar |
| Installer + shortcuts (plan §8.3(a)) | fast via shortcut | GUI install | **Reject for target**: still an installer; user goal is no installer |
| Embedded PE resource | extract once | one extraction | Awkward for ~2 GB; resource tooling/limits; not needed |
| `include_bytes!` payload in `.rdata` | extract once | one extraction | Build memory/link time huge; binary bloat; no streaming; **reject** |
| **Appended payload + extract-once cache** | cache hit, <1 s | one extraction | **Adopt** |
| Download models on first run | fast | network required | **Reject**: violates offline guarantee |

### 6.7 Recommended Rust design (normative for the rewrite)

#### 6.7.1 Artifact format

`EasyScribe.exe` is a normal Windows PE x86-64 executable with an **appended payload container** and a fixed **footer** at end-of-file. Windows ignores the overlay; the app reads its own file. The PE itself contains the Rune UI, Rust code, icon, VERSIONINFO and manifest.

```
+--------------------------------------------------+ 0
| PE image (headers, .text, .rdata, resources)      |
+--------------------------------------------------+ payload_offset
| ESPAK container: header, entries, index           |
+--------------------------------------------------+ payload_offset + payload_size
| 128-byte footer                                   |
+--------------------------------------------------+ EOF
```

Build order: `cargo build --release` → copy binary → append container → append footer → Authenticode sign. Never sign before packing; the signature must cover the final bytes.

**Why appended, not a PE resource:** no 2 GB resource/tooling constraints, no linker memory blow-up, easy to verify/truncate-detect, works with any linker, standard self-extracting practice. Windows does not require the PE checksum to match the overlay; `signtool` may update it.

#### 6.7.2 Footer layout (128 bytes, little-endian)

| Offset | Size | Field | Value / meaning |
|---|---|---|---|
| `0x00` | 8 | magic | ASCII `ESCFTR01` |
| `0x08` | 2 | footer version | `1` |
| `0x0A` | 2 | flags | bit 0: payload uses per-entry zstd; bit 1: signed build (informational); others reserved |
| `0x0C` | 4 | footer size | `128` |
| `0x10` | 8 | payload offset | absolute file offset of container |
| `0x18` | 8 | payload size | container length, excluding footer |
| `0x20` | 8 | manifest offset | relative to payload start |
| `0x28` | 8 | manifest size | uncompressed JSON length |
| `0x30` | 32 | payload SHA-256 | hash of exactly `[payload_offset, payload_offset+payload_size)` |
| `0x50` | 16 | app version | UTF-8, NUL-padded, e.g. `3.2.3` |
| `0x60` | 16 | build id | hex/UUID; diagnostics only |
| `0x70` | 4 | footer CRC32 | CRC-32 over bytes `0x00..0x6F` |
| `0x74` | 4 | reserved | zero |
| `0x78` | 8 | trailer magic | ASCII `ESCEOF01` |

Startup validation:

1. File length ≥ 128; read last 128 bytes.
2. Check `ESCFTR01` at `0x00` and `ESCEOF01` at `0x78`.
3. Check footer version, size, CRC32.
4. Check `payload_offset + payload_size + 128 == file_len`.
5. Check `payload_offset` ≥ PE headers; do not require PE parsing.
6. If any check fails: user-facing “This EasyScribe file is damaged or incomplete. Download it again.”; exit without extracting.

The SHA-256 is verified when extracting; on later cache hits we trust the cache’s `.complete` record rather than rehashing ~1.5 GB every launch (that would defeat the <1 s goal). `--verify` / `--repair` performs a full hash on demand.

#### 6.7.3 Container layout

`ESPAK001` container:

| Offset | Size | Field |
|---|---|---|
| `0x00` | 8 | magic ASCII `ESPAK001` |
| `0x08` | 2 | container format version = `1` |
| `0x0A` | 2 | flags (reserved) |
| `0x0C` | 4 | entry count |
| `0x10` | 8 | index offset (relative to container start) |
| `0x18` | 8 | index size |
| `0x20` | 8 | total uncompressed bytes |
| `0x28` | 8 | total stored bytes (after compression) |
| `0x30` | 32 | reserved (zero) |
| `0x50` | ... | entry data, in index order |

Index entries (sequential at `index_offset`):

| Field | Type | Meaning |
|---|---|---|
| `path_len` | u16 | UTF-8 path length, ≤ 32 KiB |
| `path` | bytes | normalized relative path, `/` separators, no drive/leading `/`/`..` |
| `method` | u8 | `0` store, `1` zstd |
| `flags` | u8 | reserved |
| `offset` | u64 | relative to container start |
| `compressed_size` | u64 | bytes on disk |
| `size` | u64 | uncompressed bytes |
| `sha256` | 32 | uncompressed content hash |

The manifest is entry 0 (`manifest.json`), JSON:

```json
{
  "schema": 1,
  "app": "EasyScribe",
  "app_version": "3.2.3",
  "payload_format": 1,
  "target": "windows-x86_64",
  "built_utc": "2026-11-06T00:00:00Z",
  "build_id": "…",
  "total_uncompressed": 2000000000,
  "files": [
    {"path": "models/whisper/turbo-encoder.int8.onnx", "size": 674716297, "method": "store", "sha256": "…"}
  ]
}
```

Container rules:

- Paths are relative, `/`-separated, case-insensitively unique on Windows; reject absolute paths, drive letters, `..`, NUL, reserved device names (`CON`, `NUL`, `COM1`…), trailing dots/spaces.
- Create parent directories as needed; no directory entries.
- No symlinks/hardlinks in the payload; the builder dereferences or rejects them.
- Per-entry zstd (level 3–6) for text/small binaries; `store` for models/GGUF when compression ratio is poor (measure at build; do not waste first-run CPU on incompressible weights). The builder may choose automatically with `--store-if-ratio-above 0.95`.
- The whole payload hash in the footer covers the container bytes; per-entry hashes catch corruption and allow targeted repair.
- ZIP64 is not required; the container has 64-bit offsets. Keep the final exe under GitHub’s 2 GB per-asset limit.

#### 6.7.4 Build tool (`easyscribe-pack`)

Proposed CLI:

```
easyscribe-pack pack \
  --binary target/release/easyscribe.exe \
  --payload-dir build/payload \
  --out dist/EasyScribe-v3.2.3.exe \
  --app-version 3.2.3 \
  [--store PATTERN]... [--zstd-level 3] [--build-id HEX]

easyscribe-pack verify --exe dist/EasyScribe-v3.2.3.exe [--full]
easyscribe-pack list   --exe dist/EasyScribe-v3.2.3.exe
easyscribe-pack extract --exe dist/EasyScribe-v3.2.3.exe --dest <dir> [--force]
```

Builder responsibilities:

1. Walk `payload-dir`; reject symlinks, hidden build junk, absolute/unsafe paths; sort by path; normalize case collisions.
2. Compute size + SHA-256 per entry; choose store/zstd; compress; write container; write manifest.
3. Copy the binary; append container; append footer; write output with `0o755`-equivalent (Windows ignores).
4. `verify` re-reads footer, checks CRC/size/offset/trailer, hashes the payload, parses the index, optionally hashes every entry, and checks required files (`manifest.json`, models, ffmpeg, whisper/sherpa libs, licenses).
5. `extract` is the same code path as runtime extraction; used by tests and support.
6. Deterministic mode: honor `SOURCE_DATE_EPOCH`, fixed `built_utc` fallback, stable entry order, fixed zstd level, no absolute build paths in the manifest. The payload hash then identifies the build.
7. `--print-hash`, `--print-size` for CI gates.

Tests for the tool (see §9.9): footer round-trip, truncated file, corrupted byte, wrong magic, path traversal, case collision, reserved names, store/zstd round-trip, extraction atomicity, concurrent extraction, cache reuse, disk-full simulation.

#### 6.7.5 Runtime bootstrap

```
fn bootstrap() -> Result<Paths>:
    exe = current_exe()
    footer = read_and_validate_footer(exe)        # last 128 bytes; no bulk read
    payload_id = "v1-" + hex(footer.payload_sha256)[0..32]
    runtime_root = data_root()/ "runtime"
    runtime = runtime_root / payload_id
    if cache_complete(runtime, footer):            # fast path: < 1 s
        return paths(runtime)
    lock = acquire_exclusive_lock(runtime_root/".lock")
    if cache_complete(runtime, footer):            # lost race
        return paths(runtime)
    container = read_container_header(exe, footer)
    check_free_space(data_root(), container.total_uncompressed + 256 MiB)
    staging = runtime_root / (payload_id + ".tmp-" + pid + "-" + random)
    cleanup_stale_staging(runtime_root)
    extract(exe, footer, staging, progress_callback)  # verify every entry
    write_complete(staging, footer)                   # atomic file, fsync
    activate(staging, runtime)                         # rename; if target exists, drop staging
    gc_runtimes(runtime_root, keep=runtime)
    return paths(runtime)
```

`cache_complete(runtime, footer)` requires:

- `runtime\.complete` exists and is valid JSON with `payload_sha256`, `app_version`, `payload_format`.
- Values match the footer.
- Critical files exist and are non-empty: `manifest.json`, the sherpa/ONNX model files, `ffmpeg\ffmpeg.exe`, `ffmpeg\ffprobe.exe`, the whisper GGUF (or whatever the chosen engine needs), the sherpa DLLs / whisper library.
- It does **not** rehash 1.5 GB. `--verify`/`--repair` or a corrupted-file error triggers a full re-extract.

`write_complete` writes to a temp file in the staging dir, `fsync`s, renames to `.complete`. Only then is the staging dir renamed to the final path. On Windows, `MoveFileEx` with replace/rename semantics; if the final exists, another process won; delete staging.

Concurrency:

- Use a named mutex (`Global\EasyScribe.Payload.<payload-id>` or per-user `Local\...`) or an exclusive `LockFileEx` on `runtime\.lock`. A second process waits with a “EasyScribe is getting ready…” window rather than extracting twice.
- Stale staging dirs (`*.tmp-*`) are deleted at startup while holding the lock; no other process can be using them.
- Never delete a runtime dir that another process has open; `gc` is best effort and skips the current dir and any dir with an active lock/marker.

Failure handling:

| Failure | User-visible behaviour |
|---|---|
| Footer invalid / truncated exe | “This EasyScribe file is damaged or incomplete. Download it again.” |
| Payload hash/entry hash mismatch | Remove staging; “The app files could not be verified. Download EasyScribe again.” |
| Disk full | “EasyScribe needs about N MB free to start. Free some space and try again.” Show path; do not leave a partial final runtime |
| Cache dir not writable | “EasyScribe cannot write to <path>. Set EASYSCRIBE_DATA_DIR to a writable folder or run from a user account.” |
| Extraction interrupted/crash | Staging dir remains; next launch cleans it and retries; existing runtime untouched |
| Model file missing in cache | `cache_complete` fails → re-extract; if still missing, app error dialog naming the file |

#### 6.7.6 Data and cache layout

Recommended default (consistent with `REWRITE_PLAN.md`):

```
%LOCALAPPDATA%\EasyScribe\                 (data root; stable across versions)
  runtime\
    v1-<payload-id>\                        extracted payload for this exe
      manifest.json
      models\...
      ffmpeg\...
      whispercpp\...                        or whisper libs/GGUF, depending on engine choice
      sherpa\...                            C API + ONNX Runtime DLLs, if dynamic
      licenses\...
      LICENSE.txt
      THIRD-PARTY-NOTICES.txt
      .complete                             activation marker
    .lock
  recordings\                              crash-safe PCM + JSON, settings.json
  logs\                                    EasyScribe_YYYYMMDD_HHMMSS.log, keep 10
  temp\                                    scratch WAV/JSON; cleaned at start/exit
```

- Use `SHGetKnownFolderPath(FOLDERID_LocalAppData)` (not the `%LOCALAPPDATA%` env var alone) so redirected profiles work.
- `DEFAULT_RECORDINGS_DIR` remains `Documents\EasyScribe` via `FOLDERID_Documents`.
- `recordings\settings.json` keys remain `recordings_dir` and `speaker_speed_factor`; unknown keys are ignored, missing/broken file means defaults.
- `temp` cleanup patterns remain `*.wav` and `*.json`.
- Log rotation remains 10 files, name `EasyScribe_<timestamp>.log`, no transcript text.
- Environment overrides for development/tests/portable use:
  - `EASYSCRIBE_DATA_DIR` — data root (runtime + recordings + logs + temp).
  - `EASYSCRIBE_RUNTIME_DIR` — runtime root only.
  - `EASYSCRIBE_PAYLOAD_DIR` — dev mode: use an already-extracted payload directory instead of the exe footer (matches `REWRITE_PLAN.md`).
  - Optional `EasyScribe.portable` marker next to the exe → use `<exe_dir>\EasyScribe-data` as data root. This restores USB-style portability without an installer. Do not write next to the exe by default (Program Files/USB read-only).
- Why `%LOCALAPPDATA%` and not `<exe_dir>`: the cache survives moving/copying the exe; no admin rights; no writes to Downloads/Program Files; one runtime per payload hash. The exe remains self-contained: delete the cache and it recreates.
- Parity note: v3.2.3 had `recordings/` and `logs/` inside the install folder. The target moves them to the stable data root. Behaviour is unchanged (crash recovery, settings, logs), but the Privacy panel and README paths must be updated. Finished recordings still default to `Documents\EasyScribe`; transcripts still go next to each source file or the chosen output folder. If exact old paths are required, use the portable marker or `EASYSCRIBE_DATA_DIR`.

#### 6.7.7 First-run UX

- Before extraction, show a lightweight Rune window (or native splash) with the EasyScribe mark and:
  - “Getting ready…”
  - “The first start can take a minute. Please wait.”
  - A determinate progress bar based on bytes written / total uncompressed bytes, plus optional “Extracting 1.2 GB of 1.9 GB”.
- No folder picker, no Install button, no EULA checkbox, no “Create shortcut” checkbox. This is not an installer.
- On later runs, no window if the cache is valid (or a <250 ms splash only if cache validation is slow). The app window opens directly.
- On first-run failure, show a modal error with Retry / Open data folder / Exit; never leave a half-extracted runtime active.
- Keep the app's existing `SetCurrentProcessExplicitAppUserModelID("EasyScribe.App")` equivalent early in startup so taskbar grouping/icon are correct (`CLAUDE.md` Rule 15). The icon is embedded in the PE, not a loose file.

#### 6.7.8 Update and migration semantics

- **Update = replace the exe.** The new exe has a different payload SHA-256, so it extracts to a new `runtime\v1-<id>`; the old runtime is GC’d after a successful start. No in-place mutation of a running install, no backup/rollback, no “close EasyScribe” rename dance, no update bundle, no marker. The old exe is the rollback artifact; keep it if you want an instant rollback. The app does not maintain backup copies.
- **Downgrade = run an older exe.** There is no persistent installed version to compare. The older exe extracts/uses its own runtime. `settings.json` must be forward/backward tolerant (ignore unknown keys, defaults for missing).
- **User data is stable** across all payload versions because it lives outside `runtime\`.
- **Old v3.2.3 install:** the new exe ignores the old `EasyScribe\` app files. Recommended one-time migration: if `<exe_dir>\EasyScribe\recordings` exists and the new data root has no `settings.json`, copy `settings.json`; optionally scan the old `recordings` for orphan `.pcm`/`.json` and offer recovery. Do not copy old app binaries; tell the user the old folder can be deleted after verifying recordings/transcripts. Document this in release notes.
- **No shortcuts:** the old Desktop/Start Menu shortcuts point at the old installed exe; users should replace them with a shortcut to the new single exe if they want one. The app MAY offer a user-invoked “Create desktop shortcut” menu item, but MUST NOT create shortcuts automatically as part of startup.
- **No `app.bundle`, no `.easyscribe-update.json`, no `.easyscribe-old`, no `shortcuts_created`, no `updating`** in the target. Those are Python-installer artifacts only.

#### 6.7.9 Payload contents (target)

| Payload path | Required | Notes |
|---|---|---|
| `manifest.json` | yes | container entry 0; schema/version/hashes |
| `models/whisper/turbo-{encoder,decoder}.int8.onnx`, `turbo-tokens.txt` | yes | live mode + sherpa fallback |
| `models/silero_vad.onnx` | yes | VAD file/live |
| `models/diarization/{segmentation,embedding}.onnx` | yes | speaker identification |
| `whispercpp/ggml-large-v3-turbo-q5_0.bin` | yes | file engine |
| `whispercpp/whisper-cli.exe` + `*.dll` | if subprocess engine | OR whisper.cpp static/dynamic libs linked via `whisper-rs`; either way GGUF is a real file |
| `ffmpeg/ffmpeg.exe`, `ffmpeg/ffprobe.exe` | yes | subprocess decode; same arguments |
| `sherpa/` C API DLL + ONNX Runtime DLLs | if dynamic | exact set depends on TECH_RESEARCH; pin version and hash |
| `licenses/`, `LICENSE.txt`, `THIRD-PARTY-NOTICES.txt` | yes | compliance |
| `assets/` icon/png | optional | icon is in PE resources; PNG may be used by Rune |
| `splash` image | optional | Rune can draw the splash natively; no need for a PNG |

Do **not** ship: Python runtime, `_internal`, pip packages, Tk/Tcl, PyInstaller bootloader, test audio, ASR A/B data, `models/hf_cache`, dev symlinks, `.pdb` files.

#### 6.7.10 Windows PE requirements

- x86-64 only (Windows 10/11).
- Subsystem `WINDOWS` (GUI), not console. `--version`/`--verify` may attach/allocate a console only when run with a flag; or write to stderr.
- Manifest: `requestedExecutionLevel level="asInvoker"`, `dpiAware`/`dpiAwareness` PerMonitorV2, `longPathAware true`, supportedOS Windows 10/11.
- VERSIONINFO:
  - `CompanyName` = `EasyScribe`
  - `FileDescription` = `EasyScribe`
  - `FileVersion` = `3.2.3` (numeric `3.2.3.0`)
  - `InternalName` = `EasyScribe`
  - `OriginalFilename` = `EasyScribe.exe`
  - `ProductName` = `EasyScribe`
  - `ProductVersion` = `3.2.3`
  - Translation 1033/1200, matching `assets/version_resource.py`.
- Icon: `assets/EasyScribe.ico` embedded as the first icon group; window/taskbar icon from the same resource.
- Static CRT (`+crt-static`) to avoid a VC++ redistributable; if not possible, include `vcruntime140.dll` in the payload and load from the runtime dir. Prefer static.
- If Rune/wgpu/sherpa use non-system DLLs, either link statically or ship them in the runtime and set the DLL search directory before loading. Do not rely on the current working directory.

#### 6.7.11 Code signing, SmartScreen, antivirus

- A 1.4–1.6 GB unsigned exe with an appended payload will trigger SmartScreen and possibly AV heuristics. Code signing is strongly recommended; it is not required for functionality but is required for a good first-run experience.
- Sign **after** packing, with SHA-256 and an RFC 3161 timestamp; verify with `signtool verify /pa`.
- Publish a `.sha256` next to the exe and on the release page.
- Do not use UPX or other packers; the payload is already compressed and packers worsen AV false positives.
- Test on a clean Windows VM with Defender enabled; submit false positives to Microsoft if needed.
- Mark-of-the-Web/“Unblock” guidance in README remains relevant for unsigned builds; signing reduces it.

#### 6.7.12 Offline guarantee in the single-exe model

EasyScribe’s product promise is “nothing leaves this computer”; the rewrite must preserve it without relying on a Python monkeypatch:

- **No network code/dependencies in the shipped binary.** Forbid `reqwest`, `hyper`, `ureq`, `curl`, `isahc`, `surf`, `tokio` with net features, and any direct `std::net`/WinSock use. CI runs `cargo deny`/`cargo tree` plus a source scan (the Rust analogue of `tests/test_offline_guard.py`).
- **No runtime downloads.** The payload is embedded; first run reads only the exe and writes the cache. No model download, no update check, no telemetry, no crash upload, no `webbrowser`/URL opening.
- **ffmpeg/ffprobe stay local-file-only.** Keep `-protocol_whitelist file` on every invocation and never pass a URL. The app’s file pickers only produce local paths.
- **Build-time network is isolated.** CI downloads models/tools with pinned URLs and hashes; the shipped app has no such code.
- **Optional defense in depth.** If exact Python-style runtime blocking is required, a Windows `WinSock` `connect`/`getaddrinfo` shim could be added, but it is not necessary if the dependency/source audit is enforced; document the choice. Do not add a network stack just to block it.
- **Tests:** assert no network crates/imports, assert ffmpeg args include the whitelist, and run a first-run/second-run test with the network disabled.

---

## 7. Asset and model inventory

### 7.1 Exact files and sizes (v3.2.3)

Sizes are bytes from the oracle checkout; MB = 10^6, MiB = 2^20.

| Component | Path in install | Size | Source (CI) | Cache key | License | Used by |
|---|---|---|---|---|---|---|
| Whisper turbo encoder (int8 ONNX) | `models/whisper/turbo-encoder.int8.onnx` | 674,716,297 B / 674.7 MB / 643.5 MiB | `https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-turbo.tar.bz2` | `sherpa-onnx-whisper-turbo-v1` | MIT (OpenAI Whisper weights) | live mode; sherpa fallback |
| Whisper turbo decoder (int8 ONNX) | `models/whisper/turbo-decoder.int8.onnx` | 361,080,764 B / 361.1 MB / 344.4 MiB | same tarball | same | MIT | same |
| Whisper turbo tokens | `models/whisper/turbo-tokens.txt` | 816,730 B / 0.82 MB / 797.6 KiB | same tarball | same | MIT | same |
| Silero VAD | `models/silero_vad.onnx` | 643,854 B / 0.64 MB / 628.8 KiB | `https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx` | `sherpa-onnx-silero-vad-v1` | MIT | VAD file + live |
| pyannote segmentation-3.0 | `models/diarization/segmentation.onnx` | 5,992,913 B / 6.0 MB / 5.7 MiB | `https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2` (`model.onnx` renamed) | `sherpa-onnx-diarization-v1` | MIT | diarization |
| WeSpeaker ResNet34-LM | `models/diarization/embedding.onnx` | 26,530,550 B / 26.5 MB / 25.3 MiB | `https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx` | same | CC BY 4.0 | diarization |
| whisper.cpp GGUF | `whispercpp/ggml-large-v3-turbo-q5_0.bin` | 574,041,195 B / 574.0 MB / 547.4 MiB | `https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin` | `ggml-large-v3-turbo-q5_0-v1` | MIT | file transcription |
| whisper.cpp Vulkan CLI + DLLs | `whispercpp/whisper-cli.exe`, `whispercpp/*.dll` | ~50 MB cache entry (a CPU-only build is <1 MB; Vulkan build + DLLs is the shipped size) | CI builds from `https://github.com/ggml-org/whisper.cpp` tag `v1.8.6` with `-DGGML_VULKAN=ON`; Vulkan SDK 1.4.313.2 build-time only | `whispercpp-vulkan-v1.8.6-vksdk1.4.313.2-v1` | MIT | file transcription (subprocess path) |
| ffmpeg + ffprobe | `ffmpeg/ffmpeg.exe`, `ffmpeg/ffprobe.exe` | ~150 MB download/cache; only these two exes are copied | `https://github.com/GyanD/codexffmpeg/releases/download/7.1.1/ffmpeg-7.1.1-essentials_build.zip` | `ffmpeg-win64-gyan-7.1.1-essentials-v1` | GPL v3 (essentials build) | all audio/video decode |
| sherpa-onnx + ONNX Runtime native libs | `_internal/sherpa_onnx/...` in current bundle | part of the Python runtime; exact size depends on wheel. Rust target: pin `sherpa-onnx==1.13.8` C API libs + ORT DLLs | `pip install sherpa-onnx==1.13.8` wheel, or official C API release | pin exact version; add a cache key in Rust CI | Apache 2.0 (sherpa-onnx); MIT (ONNX Runtime) | VAD/live/diarization/fallback |
| App icon | `_internal/assets/EasyScribe.ico` | 25,883 B | repo `assets/EasyScribe.ico`; generated by `assets/make_icon.py` (Pillow, dev only) | n/a | EasyScribe MIT | windows, taskbar, shortcuts |
| App PNG | `_internal/assets/EasyScribe.png` | 10,320 B | repo `assets/EasyScribe.png` | n/a | EasyScribe MIT | README/dialogs |
| Splash | launcher `splash.png` | 17,474 B | repo `assets/splash.png`; generated by `assets/make_splash.py` (Pillow + customtkinter, dev only) | n/a | EasyScribe MIT | PyInstaller bootloader |
| License texts | `LICENSE.txt`, `THIRD-PARTY-NOTICES.txt`, `licenses/*` | repo `licenses/` is 11 files; plus Python/package licenses generated | `assets/collect_licenses.py` at assemble time | n/a | see §10 | compliance |

**Model totals:** ONNX Whisper 1,036,613,791 B (1.04 GB / 988.6 MiB); diarization 32,523,463 B (32.5 MB / 31.0 MiB); Silero 643,854 B; all ONNX models 1,069,781,108 B (1.07 GB / 1,020.2 MiB). Add GGUF 574,041,195 B ⇒ 1,643,822,303 B (1.64 GB / 1.53 GiB) of models alone.

**Planning sizes:** `app.bundle` is described by README as “about 1.5 GB”; the v2.1.0 validation build was 1.39 GB; current v3.2.3 is in the same 1.4–1.6 GB range. Uncompressed payload including ffmpeg/runtime/licenses is roughly 1.9–2.1 GB. Keep the release asset under GitHub’s 2 GB per-file limit; verify in CI.

### 7.2 SHA-256 of the oracle model/assets

These were computed from the local v3.2.3 checkout and SHOULD be used as CI download verification / payload manifest values where the same bytes are used:

```
20accd02388482eb3a46bd615631adfdc85e1eb2c7db9ea3f02a40ffe6b81547  models/whisper/turbo-decoder.int8.onnx
b02dcdf54f348741e93fe732b67d933c8dcb6735655f710640143081db38878b  models/whisper/turbo-encoder.int8.onnx
b34b360dbb493e781e479794586d661700670d65564001f23024971d1f2fa126  models/whisper/turbo-tokens.txt
e9848563da86f263117134dfd7ad63c92355b37de492b55e325400c9d9c39012  models/diarization/embedding.onnx
220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079  models/diarization/segmentation.onnx
9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6  models/silero_vad.onnx
394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2  ggml-large-v3-turbo-q5_0.bin
20476067e474343b49c6536d81835c1653385989432944f0e0344c3eb0cfcf20  assets/EasyScribe.ico
f0568b6b7a654a27b8c03e8473f0621b57f142036577e856f6fbca496be3067a  assets/EasyScribe.png
0614d6ea504470c5247ec479801e65bba2a22a753d43ee66c00895058cbcd6ba  assets/splash.png
```

The CI currently does not verify these hashes; the Rust pipeline SHOULD (`Get-FileHash` / `sha256sum` in the cache-miss download step, and `easyscribe-pack verify --full` after packing).

### 7.3 Where CI gets each asset

| Asset | CI step | Download / build | Notes |
|---|---|---|---|
| ffmpeg | step 3 | GyanD `codexffmpeg` release 7.1.1, `essentials_build.zip`; find `ffmpeg.exe`/`ffprobe.exe` in the extracted tree; copy only those two | Do **not** use BtbN `autobuild-*` (mutable/deleted tags); pinned GyanD releases are permanent |
| Whisper ONNX | step 5 | `sherpa-onnx-whisper-turbo.tar.bz2`; extract `turbo-encoder.int8.onnx`, `turbo-decoder.int8.onnx`, `turbo-tokens.txt` | Cache `models\whisper` |
| Silero VAD | step 6 | direct `silero_vad.onnx` | Cache the file |
| Diarization | step 7 | pyannote segmentation tarball (`model.onnx` → `segmentation.onnx`); WeSpeaker ONNX direct | Note upstream typo `speaker-recongition-models` is correct; do not “fix” it |
| whisper.cpp CLI | step 7b | `git clone --branch v1.8.6 --depth 1`; install Vulkan SDK 1.4.313.2 silently; `cmake -S ... -B ... -A x64 -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_VULKAN=ON -DGGML_NATIVE=OFF -DWHISPER_SDL2=OFF`; build target `whisper-cli` | Vulkan SDK is build-time only; runtime needs the user's GPU driver (`vulkan-1.dll`), not SDK DLLs. Verify `whisper-cli.exe --help` runs |
| GGUF | step 7c | HuggingFace `ggerganov/whisper.cpp` `ggml-large-v3-turbo-q5_0.bin` | 574,041,195 B |
| sherpa-onnx + ONNX Runtime native libs | current | pip `sherpa-onnx==1.13.8` via `requirements.txt`; PyInstaller `collect_all` bundles `.pyd`/`.libs` | Rust target: pin the C API/DLL version and hash |
| Python packages | step 4 | `pip install -r requirements.txt` (pinned) | `sherpa-onnx==1.13.8`, `numpy==2.4.6`, `sounddevice==0.5.6`, `customtkinter==6.0.0`, `tkinterdnd2==0.6.3`, `pyinstaller==6.22.3`, `pyinstaller-hooks-contrib==2026.8` |

### 7.4 Asset generation (dev-only)

| Script | Purpose | Dependencies | Output |
|---|---|---|---|
| `assets/make_icon.py` | Teal rounded tile + five white sound-wave bars; supersampled; 16–256 px frames | Pillow | `assets/EasyScribe.ico`, `assets/EasyScribe.png` |
| `assets/make_splash.py` | 440×150 “Getting ready…” splash; Roboto from customtkinter; same palette | Pillow + customtkinter | `assets/splash.png` |
| `assets/version_resource.py` | Builds `VSVersionInfo` from `APP_VERSION`; both PyInstaller specs import it | PyInstaller | version resource |
| `assets/collect_licenses.py` | Copies licenses into assembled app | stdlib | `LICENSE.txt`, notices, `licenses/` |
| `assets/make_update_bundle.py` | Builds partial update bundle | stdlib | `update.bundle` |

These scripts are build-time only and MUST NOT be shipped in the payload.

---

## 8. Versioning and naming conventions

### 8.1 Exact version

`APP_VERSION = "3.2.3"` in `src/config.py`. There is no leading `v` in the constant. `launcher/launcher.py` has `VERSION = "3.2.3"`. The workflow default tag is `v3.2.3`. The current release name is `EasyScribe v3.2.3`.

`tests/test_version.py` enforces:

- launcher `VERSION` == `APP_VERSION`;
- workflow default tag == `v` + `APP_VERSION`;
- `version_resource.py` reads `"src" / "config.py"`;
- both specs contain `EasyScribe.ico` and `version=_version_info(`;
- `assets/EasyScribe.ico` and `.png` exist;
- the ICO header is `00 00 01 00`.

### 8.2 Version syntax and ordering

The project uses a pragmatic semver-like scheme:

```
MAJOR.MINOR.PATCH[-(alpha|beta|rc)N]
```

Examples: `3.2.3`, `3.0.0-beta1`, `3.0.0-rc2`, `v2.1`.
The launcher’s parser also accepts `3.0`, `3`, `3.0.0.1`, `3.0.0beta2`, `3.0.0-beta.2`, and leading `v`. It does **not** accept `+build` metadata, `-beta-2`, or arbitrary text; those sort first (unknown).

Ordering rules (see §4.5): numeric components first, then stage (`alpha`/`a` = 1, `beta`/`b` = 2, `rc` = 3, unknown = 0, final = 9), then numeric suffix. Pre-release < release. A newer version is never downgraded by the launcher.

### 8.3 Windows file version

`assets/version_resource.py`:

- Takes `APP_VERSION`, splits at `-`, extracts up to three integer groups, pads to four with zeros.
- `3.2.3` → `(3, 2, 3, 0)`.
- `3.0.0-beta1` → `(3, 0, 0, 0)` (the pre-release suffix is ignored; all betas of the same numeric version share a Windows file version).
- `3.0.0.1` → `(3, 0, 0, 0)` (fourth numeric group is truncated by `[:3]`).
- `FileVersion`/`ProductVersion` strings keep the full `APP_VERSION` (including `-beta1`).
- `CompanyName`/`ProductName` = `EasyScribe`; `FileDescription` = `EasyScribe` or `EasyScribe installer`; `InternalName` = filename without extension; `OriginalFilename` = filename; translation 1033/1200.

The Rust build should generate the same VERSIONINFO from `Cargo.toml`/build script and keep the same numeric mapping. If the project later uses SemVer build metadata, define a deterministic mapping (e.g. numeric prerelease number) rather than silently colliding.

### 8.4 Names (exact)

| Thing | Name |
|---|---|
| Application | `EasyScribe` |
| App version | `3.2.3` |
| Release tag | `v3.2.3` |
| Release title | `EasyScribe v3.2.3` |
| Full release artifact | `EasyScribe-v3.2.3.exe` |
| Update artifact | `EasyScribe-update-v3.2.3.exe` |
| Main executable | `EasyScribe.exe` |
| PyInstaller runtime dir | `_internal\` |
| Bundle | `app.bundle` |
| Install marker | `.easyscribe-install.json` |
| Backup dir | `.easyscribe-old\` |
| Update info (inside bundle) | `.easyscribe-update.json` |
| Install folder | `EasyScribe\` |
| Models dir | `models\` |
| Whisper ONNX dir | `models\whisper\` |
| Diarization dir | `models\diarization\` |
| VAD file | `models\silero_vad.onnx` |
| whisper.cpp dir | `whispercpp\` |
| whisper.cpp binary | `whispercpp\whisper-cli.exe` |
| whisper.cpp GGUF | `whispercpp\ggml-large-v3-turbo-q5_0.bin` |
| ffmpeg dir | `ffmpeg\` |
| Log file | `logs\EasyScribe_YYYYMMDD_HHMMSS.log` |
| Recording work file | `recordings\recording_YYYYMMDD_HHMMSS.pcm` + `.json` |
| Finished recording | `<chosen folder>\recording_YYYYMMDD_HHMMSS.wav` |
| Transcript | `<stem>.txt` |
| Best-quality transcript | `<stem> (best quality).txt` |
| Speaker transcript | `<stem> (speakers).txt` |
| Best-quality speaker transcript | `<stem> (best quality) (speakers).txt` |
| Settings | `recordings\settings.json` |
| Default recordings folder | `Documents\EasyScribe` |
| AppUserModelID | `EasyScribe.App` |
| Proposed Rust payload magic | `ESPAK001` / footer `ESCFTR01` / trailer `ESCEOF01` |
| Proposed Rust artifact | same `EasyScribe-v3.2.3.exe` (do not add `-windows-x64` unless the release policy changes) |

### 8.5 Rust version source of truth

Recommended:

- `[package] version = "3.2.3"` in `crates/easyscribe/Cargo.toml`.
- `build.rs` emits `ES_APP_VERSION` (or use `env!("CARGO_PKG_VERSION")`).
- UI title `EasyScribe {ES_APP_VERSION}`; log header `EasyScribe v{ES_APP_VERSION} starting`; `--version` prints `EasyScribe 3.2.3`.
- Windows resource generated from the same value; payload footer `app_version` from the same value.
- CI gate parses `Cargo.toml`, compares to the workflow input tag (`v` + version), the artifact filename, `--version` output, and the footer/manifest.
- `PAYLOAD_FORMAT_VERSION` is independent (`1`); never conflate it with the app version.

---

## 9. Tests that encode behaviour

### 9.1 Harness conventions

All Python tests are standalone scripts, not pytest:

- Run as `python tests/test_<name>.py` from the project root.
- Most define a small `OK`/`FAIL` helper (`_check(name, fn)` or `check(name, ok)`) that collects failures and `sys.exit(1)` if any; `validate_build.py` accumulates `ok`/`errors` lists and prints its own report.
- `tests/run_all.py` runs every `tests/test_*.py` in sorted order **except** `test_diarization.py` and `test_diarizer_process.py` (they need real models and `tests/data/test_speech.wav`, which are absent before CI’s download steps). It runs with `cwd=ROOT.parent`, prints a summary, exits 1 if any file fails.
- CI runs `tests/test_sherpa_api.py` first, then `tests/run_all.py`, before model downloads (fail-fast). `tests/validate_build.py` runs after PyInstaller and again after assembly.

### 9.2 `tests/test_launcher_safety.py` — 27 checks

This is the packaging oracle. It imports `launcher` and `make_update_bundle`; it needs tkinter/customtkinter installed but no display and does not start `mainloop`.

| # | Check name | What it asserts |
|---|---|---|
| 1 | `_resolve_install_dir(picked dir) -> <dir>/EasyScribe` | An arbitrary existing folder (e.g. Documents) resolves to `<chosen>/EasyScribe`, never the folder itself |
| 2 | `_resolve_install_dir(marked install) -> unchanged` | A folder with `.easyscribe-install.json` resolves to itself |
| 3 | `_resolve_install_dir(pre-marker v2.0.0 install) -> unchanged` | A folder with `EasyScribe.exe` but no marker resolves to itself |
| 4 | `_safe_to_clean(nonexistent dir) -> True` | A non-existent install dir is trivially cleanable |
| 5 | `_safe_to_clean(marked, no exe) -> True (cleanable)` | Marker + no `EasyScribe.exe` ⇒ cleanable |
| 6 | `_safe_to_clean(complete install) -> False (never wiped)` | Marker + `EasyScribe.exe` ⇒ never wiped |
| 7 | `_safe_to_clean(unmarked non-empty folder) -> False` | No marker, no exe, user file present ⇒ not cleanable (Documents scenario) |
| 8 | `failure path with no marker leaves directory untouched` | `_safe_to_clean` is false, so the `_extract_thread` error path does not rmtree; `user_data.txt` survives |
| 9 | `_write_marker creates marker with app+version` | File exists; `app == "EasyScribe"`; `version == launcher.VERSION` |
| 10 | `_write_marker does not overwrite existing marker` | Existing `version: "1.0.0"` stays `1.0.0`, not bumped |
| 11 | `_shortcuts_created(no marker) -> False` | Missing marker ⇒ false |
| 12 | `_shortcuts_created(marker, not yet marked) -> False` | Marker without flag ⇒ false |
| 13 | `_mark_shortcuts_created sets flag, keeps app+version` | `shortcuts_created` true; `app`/`version` preserved |
| 14 | `_ensure_shortcuts creates shortcuts only once` | Monkeypatches `_create_shortcuts`; two calls ⇒ exactly one call; flag true |
| 15 | `_version_key: pre-release < release, numeric order` | `["junk","2.0.0","v2.1","3.0.0-alpha1","3.0.0-beta1","3.0.0-beta2","3.0.0-rc1","3.0.0","3.0.1","3.1.0","10.0.0"]` is sorted; `3.0 == 3.0.0` |
| 16 | `_needs_update: older/pre-marker yes, same/newer no` | `2.1.0` ⇒ true; no marker ⇒ true; current ⇒ false; `99.0.0` ⇒ false (no downgrade) |
| 17 | `_update_install replaces app files, keeps recordings+logs` | New exe/new `_internal`/new model; old dll/model gone; `whispercpp/whisper-cli.exe` present; `recordings/meeting.wav` + `.txt` and `logs/easyscribe.log` unchanged; no backup; marker version current, `shortcuts_created` true, no `updating`; progress ends at 100 |
| 18 | `_update_install works on a pre-marker v2.0.0 install` | Marker written with current version; recording survives |
| 19 | `_update_install while app open -> blocked, old files back` | Monkeypatches `_move` to raise `PermissionError` for `_internal`; raises `UpdateBlockedError`; old exe/dll/model/recording intact; no backup; version still `2.1.0`; `updating` false |
| 20 | `_update_install bad bundle -> old files back` | Bundle bytes are not a ZIP; raises `BadZipFile`; old exe and recording intact |
| 21 | `_recover_interrupted_update puts old files back` | Simulates `updating: true`, old exe/`_internal` moved to backup, half-extracted `_internal/half.dll`; recovery restores old exe/dll, removes half file and backup, clears `updating`, keeps recording, `_needs_update` true |
| 22 | `_recover_interrupted_update removes a finished backup` | Current version + leftover backup ⇒ backup deleted, main exe remains |
| 23 | `_clean_incomplete_install keeps recordings+logs` | Removes `_internal`; keeps `recordings/meeting.wav` and `logs/easyscribe.log` |
| 24 | `update bundle leaves out models, ffmpeg, GGUF` | Constants agree; update ZIP has `EasyScribe.exe` + `whispercpp/whisper-cli.exe`; no `models/`, `ffmpeg/`, `recordings`; no GGUF; `keep` exactly `models/whisper/encoder.onnx`, `ffmpeg/ffmpeg.exe`, `whispercpp/ggml-model.bin`; full bundle returns `None` |
| 25 | `partial update keeps the big files, replaces the rest` | New exe/new whisper-cli; GGUF/model/ffmpeg old bytes kept; recording kept; no update info/backup; version current |
| 26 | `partial update refuses a changed install, changes nothing` | Modify `models/whisper/encoder.onnx`; raises `NeedsFullInstallError` mentioning `encoder.onnx`; old exe intact; no backup; version `3.1.0`; `updating` false |
| 27 | `partial update failure -> old files and GGUF back` | Monkeypatches `_carry_over` to raise `OSError("disk full")`; old exe, GGUF, old whisper-cli restored; no backup; version `3.1.0` |

### 9.3 `tests/test_version.py` — 8 checks

| Check | Assertion |
|---|---|
| launcher VERSION matches | `launcher.py` `VERSION == APP_VERSION` |
| workflow default tag is `v{app_version}` | First `default: '...'` in `build-release.yml` equals `v3.2.3` |
| version_resource reads APP_VERSION | `version_resource.py` contains literal `"src" / "config.py"` |
| `EasyScribe_whisper.spec` sets icon + version | Contains `EasyScribe.ico` and `version=_version_info(` |
| `launcher/launcher.spec` sets icon + version | Same |
| `assets/EasyScribe.ico` exists | File present |
| `assets/EasyScribe.png` exists | File present |
| ICO header valid | First 4 bytes `00 00 01 00` |

### 9.4 `tests/validate_build.py` — up to ~35 checks

Usage: `python tests/validate_build.py <dist_dir> [--assembled]`. Exits 1 on any failure; prints `OK`/`FAIL` lines.

Base checks (after PyInstaller, before `app.bundle`):

| Check | Assertion |
|---|---|
| `EasyScribe.exe` | Exists at `<dist>/EasyScribe.exe` |
| `_internal/` | Directory exists; without it, stop after report |
| `numpy` | `_internal/numpy/` exists with ≥1 `.pyd` |
| `sherpa_onnx` | `_internal/sherpa_onnx/` exists with ≥1 `.pyd` |
| `customtkinter` | `_internal/customtkinter/` directory exists |
| `tkinterdnd2` | `_internal/tkinterdnd2/` directory exists |
| `sounddevice` | At least one of: `_internal/_sounddevice_data/` dir, `**/_sounddevice*.pyd`, `**/portaudio*.dll` |
| window icon | `_internal/assets/EasyScribe.ico` exists |

`--assembled` checks (after CI copies models/ffmpeg/whispercpp/licenses; before `app.bundle`):

| Check | Assertion |
|---|---|
| `ffmpeg/ffmpeg.exe`, `ffmpeg/ffprobe.exe` | Each exists and size > 0 |
| `models/silero_vad.onnx` | Exists, size > 0 |
| `models/whisper/turbo-encoder.int8.onnx` | Exists, size > 0 |
| `models/whisper/turbo-decoder.int8.onnx` | Exists, size > 0 |
| `models/whisper/turbo-tokens.txt` | Exists, size > 0 |
| `models/diarization/segmentation.onnx` | Exists, size > 0 |
| `models/diarization/embedding.onnx` | Exists, size > 0 |
| `whispercpp/whisper-cli.exe` | Exists, size > 0 |
| `whispercpp/ggml-large-v3-turbo-q5_0.bin` | Exists, size > 0 |
| `whispercpp/*.dll` | At least one DLL (otherwise `whisper-cli.exe` cannot start) |
| `licenses/python-packages/` | At least one file |
| No ASIO PortAudio DLLs | No `_internal/_sounddevice_data/portaudio-binaries/*-asio.dll` (Steinberg SDK license; sounddevice only loads them with `SD_ENABLE_ASIO`) |
| No duplicate models | `_internal/models` must **not** exist |
| `LICENSE.txt` | Exists, size > 0 |
| `THIRD-PARTY-NOTICES.txt` | Exists, size > 0 |
| `licenses/Python-LICENSE.txt` | Exists, size > 0 |
| Each repo `licenses/*` except notices | Copied into `dist/licenses/`, exists, size > 0 (currently 10 files) |

CI additionally runs `dist\EasyScribe\whispercpp\whisper-cli.exe --help` and fails if it does not start. That is a shell check, not part of `validate_build.py`.

### 9.5 `tests/run_all.py`

- Globs `tests/test_*.py`, sorted.
- Excludes exactly `test_diarization.py` and `test_diarizer_process.py` (real models + test audio; these are run manually/with models).
- Runs each with `sys.executable`, `cwd=ROOT.parent`.
- Prints `=== <file> ===`; collects failures; exits 1 with `FAILED: n/m test file(s) failed: ...`.
- CI calls it after `test_sherpa_api.py`, before model downloads. Any new `tests/test_*.py` automatically joins the suite; keep it fast/offline.

### 9.6 Other tests that encode app behaviour

| File | Checks | What it asserts |
|---|---|---|
| `tests/test_offline_guard.py` | 14 | `offline_guard.install()` idempotent; blocks `socket.create_connection`, `connect`, `connect_ex`, UDP `sendto`, `getaddrinfo`, `gethostbyname`, connect-by-hostname; blocks `urllib` (as `OSError`); loopback + `localhost` + `socketpair` still work; `ffmpeg_wrapper._LOCAL_FILES_ONLY == ("-protocol_whitelist", "file")`; used by ffprobe and ffmpeg exactly twice; applied before `-i`; no network library imports in `src/` or `launcher/` (`requests`, `urllib`, `urllib3`, `http`, `httpx`, `aiohttp`, `ftplib`, `smtplib`, `webbrowser`, `huggingface_hub`, `socketserver`, `xmlrpc`) |
| `tests/test_recovery.py` | 3 | `recover_pcm_to_wav` truncates to `samples_written` (1000 of 1200), writes 16 kHz mono 16-bit WAV; `_handle_orphan(recover=False)` keeps `.pcm` + `.json`; `recover=True` writes `.wav`, removes both sidecars, frame count correct |
| `tests/test_recordings_folder.py` | 4 | `settings` round-trip (`recordings_dir`), default on missing file, default on missing folder, default on broken JSON; `MicRecorder.stop(dest_dir=...)` writes WAV to chosen folder and removes raw; unusable dest falls back next to raw; `_handle_orphan(dest_dir=...)` recovers to chosen folder |
| `tests/test_rename.py` | 6 | `check_name` accepts “Team meeting 21 Oct” and rejects empty, `/`, `:`, quotes, `?`, reserved `CON`/`lpt1`, trailing dot, and a 200-character input (the real limit is `MAX_NAME = 120`); `recording_set` includes `.wav`, `.txt`, `(best quality).txt`, `(best quality) (speakers).txt`, not `_2.wav`/other timestamp/`.json`; `rename_all` renames all, leaves others; taken name raises `FileExistsError` and changes nothing; a failure on the 3rd rename rolls all back; transcript-only rename leaves `.wav` |
| `tests/test_vad_bounds.py` | 8 | `VAD_PAD_SEC < VAD_MIN_SILENCE_SEC/2`; `pad_and_clamp` single segment exact padding; adjacent at minimum gap no overlap; far apart; start clamp to 0; end clamp to total; 10 adjacent; empty input ⇒ `[]`; output sorted, within `[0,total]`, count preserved |
| `tests/test_transcript_builders.py` | 10 | `_build_plain_transcript`: empty ⇒ `""`; no timestamps joins with space; timestamps single block; gap ≥ 2 s splits with blank line. `_build_diarized_transcript`: empty; same speaker groups; speaker change + gap splits; unmapped label falls back. `assign_speakers`: max overlap wins; no overlap ⇒ `SPEAKER_00` |
| `tests/test_ffmpeg_wrapper.py` | 4 | `_iter_lines` splits on `\r` and yields the final line; works when pipe reads split a line (sizes 1,3,7,64); empty stream ⇒ `[]`; `_parse_duration("  Duration: 01:02:03.50, start") == 3723.5`; no match ⇒ `0.0` |
| `tests/test_whispercpp_wrapper.py` | 11 | `_parse_output_json` converts ms→s, strips text, drops blank entries; empty list; `_describe_devices` Vulkan GPU name/`Other Vulkan device`/CPU line, CPU-only “no GPU found”, unnamed GPU “name not reported”; `_pick_gpu` prefers dedicated (uma 0), single integrated ⇒ 0, no GPU ⇒ None, tie ⇒ Vulkan order; `_parse_progress_line` 100/42/unrelated None; `is_available` false when binary/model missing + error names both; true with local `whispercpp/` (skips if absent); pre-cancelled `transcribe_file` raises `CancelledError` immediately |
| `tests/test_live_transcriber.py` | 6 | `pad_live_segment`: no padding when history is after segment start; applies `PAD_SAMPLES` lead-in; clamps to previous segment end; clamps to history start. Decode queue drops oldest under sustained backpressure and never exceeds max. Stop flushes speech in progress + queued audio (4000 samples) |
| `tests/test_transitions.py` | 7 | Veil order `setup, place, hold, change`, fade-out alphas ascending to 1.0, fade-in descending to 0.0, hide at end; joining during fade-out only runs; joining while lifting re-covers; animations off ⇒ `setup, hold, change, draw, hide`; not front ⇒ `setup, change`; failing Windows call still runs change; `create()` outside Windows returns None |
| `tests/test_sherpa_api.py` | 1 | `transcriber._build_recognizer_config()` returns `sherpa_onnx.OfflineRecognizerConfig`; constructing the binding `_OfflineRecognizer(cfg)` with missing models raises `RuntimeError` (bad path), not `TypeError` (bad constructor signature) |
| `tests/test_diarization.py` | 3 steps | `DiarizationEngine().is_available()` true (models present); `TranscriptionEngine.transcribe(diarize=True)` on `tests/data/test_speech.wav` produces non-empty transcript; transcript contains `[Speaker` |
| `tests/test_diarizer_process.py` | 2 | Diarization in a child process reports monotonic progress ending at 1.0 with turns; cancel during a ~8-minute synthetic file ends with `CancelledError` in <60 s |
| `tests/compare_whisper_models.py` | utility | Drives the real `TranscriptionEngine` per model directory (VAD pipeline, not a naive whole-file path), prints transcript + timing; not in `run_all` |
| `tests/diagnose_sounddevice.py` | utility | Prints sounddevice version/path/contents so CI logs show the PortAudio layout; not in `run_all` |
| `tests/asr_ab/` | developer tool | Network-using A/B harness (`fetch.py`, `ab_models.py`); 2026-10-01 results rank `current_files` (whisper.cpp GGUF q5_0, 574 MB) best mean WER 24.3, `current_live` (sherpa ONNX int8, 1037 MB) 29.8; not shipped, not in CI |

### 9.7 Porting matrix for the Rust rewrite

| Python test | Rust target | Action |
|---|---|---|
| `test_launcher_safety.py` | `easyscribe-pack` / payload tests + core data tests | **Do not port installer/marker/update cases 1–6, 9–22, 24–27 literally**; they test machinery that no longer exists. Port the safety invariants: never delete unmarked/user data (7, 8, 23), version ordering (15, 16) if a legacy-marker reader remains, and replace update/rollback/crash recovery (17–22, 24–27) with payload atomicity/GC tests |
| `test_version.py` | `crates/easyscribe/tests/version_consistency.rs` or `xtask version-check` | Port; read `Cargo.toml`, workflow default tag, VERSIONINFO resource, artifact name, `--version`, footer/manifest |
| `validate_build.py` | `easyscribe-pack verify --full` + `crates/easyscribe-pack/tests/validate_payload.rs` | Port required-file list, license list, no duplicate models, no ASIO (if PortAudio gone, drop ASIO check), plus payload footer/container checks |
| `run_all.py` | `cargo test --workspace` + `xtask test-all` | Replace; keep fast/offline ordering before heavy downloads |
| `test_offline_guard.py` | `crates/easyscribe-core/tests/offline.rs` + CI `cargo deny`/source scan | Port blocked-network semantics to a dependency/source audit; keep ffmpeg `-protocol_whitelist file` and no-network-imports checks. Runtime socket hook is optional; if omitted, document that the binary has no network code and CI enforces it |
| `test_recovery.py` | `easyscribe-core::recovery` unit tests | Port exactly (truncation, decline keeps files, accept cleans up) |
| `test_recordings_folder.py` | `easyscribe-core::settings` + `mic` tests | Port (settings fallbacks, destination fallback, recovery destination) |
| `test_rename.py` | `easyscribe-core::rename` tests | Port exactly (name validation, set membership, atomic undo, transcript-only) |
| `test_vad_bounds.py` | `easyscribe-core::vad` tests | Port exactly (invariant + property cases) |
| `test_transcript_builders.py` | `easyscribe-core::transcript` + `diarizer::assign_speakers` | Port exactly (block/gap rules, labels, overlap) |
| `test_ffmpeg_wrapper.py` | `easyscribe-core::ffmpeg` tests | Port (CR splitting, chunked reads, duration parse) |
| `test_whispercpp_wrapper.py` | `easyscribe-core::whisper` tests | Port JSON/device/progress parsing, availability, cancellation; adapt to `whisper-rs` if the subprocess is removed |
| `test_live_transcriber.py` | `easyscribe-core::live` tests | Port padding/queue/flush |
| `test_transitions.py` | Rune UI tests / remove | Rune replaces the Tk transition machinery; port only user-visible “no half-drawn state” expectations if applicable |
| `test_sherpa_api.py` | `easyscribe-core::sherpa` smoke test | Port config/ABI smoke; no models needed |
| `test_diarization.py` / `test_diarizer_process.py` | ignored/real-model integration tests | Keep as `#[ignore]`/feature-gated; run on a machine with models + audio |
| `compare_whisper_models.py` | dev tool | Optional; keep out of shipped crate |
| `tests/asr_ab/` | dev tool | Keep in `tests/` only; uses network, never in `src/` |

### 9.8 New tests required for the payload design

Add these to `crates/easyscribe-pack` / `easyscribe-core`:

1. **Footer round-trip** — pack a fixture, read footer, assert magic/version/CRC/offset/size/hash/version string.
2. **Truncation** — cut 1 byte/128 bytes/1 KB; `read_footer` fails with a clear error.
3. **Corruption** — flip a byte in payload/entry; full verification fails; extraction removes staging, leaves existing runtime.
4. **Path traversal** — entries `../evil`, `/abs`, `C:\abs`, `..\..\`, NUL, reserved names; packer rejects or extractor refuses.
5. **Case collision** — `Models/A` vs `models/a` on Windows; packer rejects.
6. **Store/zstd round-trip** — every method produces byte-identical files and correct SHA-256.
7. **Atomic activation** — kill/fail after partial extraction; no `.complete`; next run re-extracts; existing runtime untouched.
8. **Concurrent first run** — two processes/threads race; exactly one extraction; both use the same runtime.
9. **Cache hit** — second run does not read the payload body; only footer + `.complete` (instrument the reader).
10. **GC** — old runtime dirs removed; current kept; active/locked dirs skipped.
11. **Disk-space error** — simulate insufficient space; no final runtime, actionable error.
12. **Data preservation** — update/GC never deletes `recordings`, `logs`, `settings.json`, or user output.
13. **License completeness** — required texts present in payload/runtime.
14. **`--verify`** — full hash catches corruption; `--repair` re-extracts.
15. **Version consistency** — Cargo/workflow/VERSIONINFO/artifact/footer all agree.
16. **Offline** — no network crates/imports; ffmpeg args include `-protocol_whitelist file`; no update check.

### 9.9 CI test order

1. Fast pure tests (transcript, rename, VAD, ffmpeg, recovery, settings, payload fixture) before any downloads.
2. `cargo clippy` / `cargo fmt --check`.
3. ABI smoke (`sherpa` config; ffmpeg presence).
4. Download/cache models/tools.
5. Windows integration: build/load models, `whisper` smoke, pack/verify.
6. `easyscribe-pack verify --full` after packing.
7. Release smoke: `EasyScribe.exe --version`, `--payload-verify`, optional headless transcription.

---

## 10. Licensing and notices requirements

### 10.1 App license

EasyScribe is MIT: `LICENSE` (repo root) / `LICENSE.txt` (install/payload). Copyright (c) 2026 Tom. The MIT notice must be included in all copies or substantial portions; the payload includes `LICENSE.txt`.

### 10.2 Current bundled components and obligations

| Component | Version / file | License | Notice text in repo | Obligation |
|---|---|---|---|---|
| FFmpeg | 7.1.1, `ffmpeg.exe`, `ffprobe.exe` (GyanD essentials) | GPL v3 (build) | `licenses/FFmpeg-GPL-3.0.txt`, `licenses/FFmpeg-LICENSE.md` | Include GPL text and source/written offer; state EasyScribe runs it as a separate program, not a derivative |
| whisper.cpp + ggml | v1.8.6, `whisper-cli.exe`, DLLs, GGUF conversion | MIT | `licenses/whisper.cpp-MIT.txt` | Include copyright/license |
| sherpa-onnx | 1.13.8 (requirements), libs | Apache 2.0 | `licenses/sherpa-onnx-Apache-2.0.txt` | Include license; include upstream NOTICE if present; state changes if any |
| ONNX Runtime | bundled in sherpa-onnx | MIT | `licenses/onnxruntime-MIT.txt` | Include notice |
| PortAudio | via sounddevice 0.5.6 (Windows DLL) | MIT | `licenses/PortAudio-MIT.txt` | Include notice; remove ASIO builds to avoid Steinberg SDK |
| Python 3.11 / Tcl/Tk | PyInstaller runtime | PSF and others | `licenses/Python-LICENSE.txt` | Include Python license; Python package licenses |
| Python packages | numpy, sounddevice, cffi, pycparser, customtkinter, darkdetect, packaging, tkinterdnd2 | per package | `licenses/python-packages/<name>/...` | Copy dist-info license files; `collect_licenses.py` fails if required package missing/no license |
| PyInstaller bootloader | 6.22.3 | GPL v2+ with bootloader exception | in `THIRD-PARTY-NOTICES.txt` | Notice only; exception permits any-license output |
| Whisper large-v3-turbo | ONNX export + GGUF | MIT (OpenAI) | `licenses/OpenAI-Whisper-model-MIT.txt` | Include notice; identify both forms |
| Silero VAD | `silero_vad.onnx` | MIT | `licenses/Silero-VAD-MIT.txt` | Include notice |
| pyannote segmentation-3.0 | `segmentation.onnx` | MIT | `licenses/pyannote-segmentation-MIT.txt` | Include notice |
| WeSpeaker ResNet34-LM | `embedding.onnx`, VoxCeleb-trained | CC BY 4.0 | `licenses/WeSpeaker-model-CC-BY-4.0.txt` | Attribution: WeSpeaker project, VoxCeleb1/VoxCeleb2 datasets, converted by sherpa-onnx; indicate if changed (not changed) |
| Rune (target) | pinned git rev | MIT (confirmed from `/home/user/rune-ref`) | add to notices | Include MIT text |
| Rust standard library / crates (target) | locked versions | MIT/Apache-2.0/BSD/etc. | generate `licenses/rust-crates/` | Include all crate licenses/notices; use `cargo-about` or `cargo-deny` |

The canonical notice is `licenses/THIRD-PARTY-NOTICES.txt`; it lists versions, source URLs, and the FFmpeg written offer. `README.md` also has a License section. Keep both in sync when versions change.

### 10.3 `assets/collect_licenses.py` (current behaviour)

Run with the build venv active after assembly:

```
python assets/collect_licenses.py dist/EasyScribe
```

It:

1. Requires exactly one argument: the app folder; errors if not a directory.
2. Copies `LICENSE` → `<dest>/LICENSE.txt`.
3. Copies `licenses/THIRD-PARTY-NOTICES.txt` → `<dest>/THIRD-PARTY-NOTICES.txt`.
4. Copies every other file in `licenses/` → `<dest>/licenses/`.
5. Finds Python's `LICENSE.txt` under `sys.base_prefix` or `lib/pythonX.Y/`; copies to `licenses/Python-LICENSE.txt`; error if missing.
6. For `REQUIRED_PACKAGES` (`sherpa-onnx`, `numpy`, `sounddevice`, `cffi`, `pycparser`, `customtkinter`, `darkdetect`, `tkinterdnd2`) and `OPTIONAL_PACKAGES` (`packaging`):
   - Uses `importlib.metadata` to find files in the `.dist-info` whose names contain `license`, `licence`, `copying`, `notice`, or `authors`;
   - Copies them to `licenses/python-packages/<package>/<relative dist-info path>`;
   - Errors if a required package is missing or has no license file.
7. Prints counts; exits 1 if any error.

`tests/validate_build.py --assembled` enforces the result: `LICENSE.txt`, notices, Python license, every repo license file, non-empty `licenses/python-packages/`, and no ASIO DLLs.

### 10.4 ASIO removal

`sounddevice` ships PortAudio builds with ASIO (Steinberg SDK) under `_internal/_sounddevice_data/portaudio-binaries/*-asio.dll`. EasyScribe never sets `SD_ENABLE_ASIO`, so it loads `libportaudio64bit.dll` and the ASIO builds are dead weight with extra licensing. CI removes them with `Remove-Item ... -ErrorAction Stop`; `validate_build.py` fails if any remain.

If the Rust rewrite uses `cpal`/WASAPI instead of PortAudio, the PortAudio notice and the ASIO removal step become obsolete; update `THIRD-PARTY-NOTICES.txt` and `validate_build` accordingly. Do not silently drop the notice while PortAudio is still shipped.

### 10.5 Rust/Rune changes

- Remove Python/Tk/PyInstaller notices if the shipped payload contains no Python runtime, Tcl/Tk, or PyInstaller bootloader.
- Keep FFmpeg (GPL v3), whisper.cpp/ggml (MIT), sherpa-onnx (Apache 2.0), ONNX Runtime (MIT), Whisper model (MIT), Silero (MIT), pyannote (MIT), WeSpeaker (CC BY 4.0).
- Add Rune (MIT), Rust std, and every crate actually linked. Because Rust statically links most crates, their notices must be reproduced in `THIRD-PARTY-NOTICES.txt`/`licenses/rust-crates/`. Generate with `cargo about generate` (or `cargo deny` + `cargo-license`); commit the generated texts; check in CI.
- Add an About/Licenses entry in the app (Privacy panel or About) that opens the runtime `licenses` folder or shows the notices. The old app had the files in the install folder; the target should make them discoverable even though the exe is self-contained.
- Keep the FFmpeg written offer: “for three years after this release, we will give a copy of the complete corresponding source code … Open an issue at https://github.com/Tom1tk/EasyScribe/issues.” Update the version/source URL when ffmpeg is bumped. Keep the source available for the offered period.
- CC BY 4.0 attribution must remain visible: WeSpeaker project, VoxCeleb1/VoxCeleb2, sherpa-onnx conversion, “not changed by EasyScribe”.
- Apache 2.0: include the license; if upstream ships a `NOTICE`, include it. State modifications if the C API build differs.
- MIT: include copyright notices for OpenAI Whisper, ggml authors, Microsoft ONNX Runtime, PortAudio, Silero, pyannote/CNRS, Rune/Anomaly.
- Release page should repeat the license summary and link to `THIRD-PARTY-NOTICES.txt` (the current workflow body does).

### 10.6 CI enforcement (target)

`easyscribe-pack verify --licenses` (or a dedicated `xtask licenses`) MUST fail if:

- `LICENSE.txt`, `THIRD-PARTY-NOTICES.txt`, or any required `licenses/*` text is missing/empty;
- a shipped model/tool has no corresponding notice entry;
- `cargo-about` output is stale relative to `Cargo.lock` (run `cargo about generate --check` in CI);
- FFmpeg GPL text/written offer is missing;
- CC BY attribution text is missing.

---

## 11. CI and release requirements

### 11.1 Current workflow (`.github/workflows/build-release.yml`)

Trigger: `workflow_dispatch` only. Input `tag`, description `'Release tag: "v" + APP_VERSION in src/config.py (e.g. v3.0.0)'`, default `v3.2.3`. Permissions: `contents: write`. Runner: `windows-latest`.

| Step | What it does | Cache |
|---|---|---|
| 1 | Checkout | — |
| 2 | Python 3.11 | — |
| 3 | Download ffmpeg 7.1.1 essentials; copy `ffmpeg.exe`/`ffprobe.exe` to `tools\ffmpeg` | `ffmpeg-win64-gyan-7.1.1-essentials-v1` |
| 4 | Create `venv`; `pip install -r requirements.txt` | `venv-win64-py3.11-v4` |
| 4b | `python tests/test_sherpa_api.py` | — |
| 4c | `python tests/run_all.py` | — |
| 5 | Download/extract Whisper turbo ONNX to `models\whisper` | `sherpa-onnx-whisper-turbo-v1` |
| 6 | Download `silero_vad.onnx` | `sherpa-onnx-silero-vad-v1` |
| 7 | Download diarization segmentation + embedding | `sherpa-onnx-diarization-v1` |
| 7b | Install Vulkan SDK 1.4.313.2 (cache miss); build whisper.cpp v1.8.6 `whisper-cli` with `-DGGML_VULKAN=ON`; verify `--help` | `whispercpp-vulkan-v1.8.6-vksdk1.4.313.2-v1` |
| 7c | Download GGUF model | `ggml-large-v3-turbo-q5_0-v1` |
| 8 | `tests/diagnose_sounddevice.py` (log layout) | — |
| 9 | `pyinstaller EasyScribe_whisper.spec --noconfirm` | — |
| 10 | `python tests/validate_build.py dist\EasyScribe` | — |
| 11 | Assemble: copy ffmpeg, models, whispercpp; remove ASIO DLLs; `assets/collect_licenses.py`; create empty `recordings` | — |
| 11b | `validate_build.py dist\EasyScribe --assembled`; run `whisper-cli.exe --help` from `dist` | — |
| 12 | `Compress-Archive -Path "dist\EasyScribe\*" -DestinationPath app.bundle` | — |
| 13 | `pyinstaller launcher/launcher.spec` | — |
| 13b | `make_update_bundle.py dist\EasyScribe update.bundle`; swap `app.bundle`; build launcher to `dist-update` | — |
| 14 | Rename to `EasyScribe-v3.2.3.exe` / `EasyScribe-update-v3.2.3.exe` | — |
| 15 | Create draft GitHub release with both assets + long release notes | — |

The release body includes “What's new” sections for 3.2.3, 3.2.2, 3.2.1, 3.2.0, 3.1.0, 3.0.0, installation instructions, SmartScreen warning, update instructions, and a Licenses paragraph. `prerelease: contains(tag, '-')`.

### 11.2 Current requirements/gaps

- Fail-fast ordering is good: API smoke + unit tests before model downloads, validation before the expensive compress.
- The **launcher** is built but not smoke-tested; only the app is validated.
- The **launcher exe** has no hash/signature verification; it trusts the downloaded file.
- The **full/update bundle** is not checked for required files beyond the app folder validation.
- `tests/test_version.py` runs via `run_all.py`; it is a text-level check, not a build artifact check.
- The workflow does not check the final artifact size against GitHub’s 2 GB limit.
- Caches can silently go stale; bump keys when versions change (Rule 5).
- No code signing; SmartScreen warning documented.
- Model downloads are not hash-verified (only cache presence).
- `AppData`/SFX historical issues are documented in `CLAUDE.md` Rules 1–6, 17–18.

### 11.3 Required properties for the Rust/Rune pipeline

The rewrite MUST preserve these properties:

1. **Pinned, reproducible inputs.** `Cargo.lock`, pinned Rust toolchain, pinned Rune rev, pinned sherpa-onnx/whisper.cpp/ffmpeg/model versions, pinned GitHub Action versions.
2. **Fail-fast tests before heavy downloads.** Pure core tests first; model/tool downloads after.
3. **Validate before packing.** Models/ffmpeg/whisper/sherpa/libs/licenses present; `whisper-cli --help` (if subprocess); `ffmpeg -version`; `sherpa` load smoke.
4. **Pack then verify then sign.** `easyscribe-pack pack` → `verify --full` → `signtool sign` → `signtool verify` → hash.
5. **Artifact only from CI.** No local unverified builds published.
6. **Draft release; manual publish.** Keep `draft: true`; pre-release when tag has `-`.
7. **No network in shipped app.** Build-time downloads only; app has no updater/telemetry.
8. **Licenses complete.** `cargo-about --check`, notices, FFmpeg GPL, CC BY.
9. **Version consistency.** Tag/`Cargo.toml`/VERSIONINFO/`--version`/artifact/footer all equal.
10. **Smoke test the actual exe** (footer, payload, model load, optional transcription).

### 11.4 Proposed Rust workflow

```yaml
name: Build and Publish Release
on:
  workflow_dispatch:
    inputs:
      tag:
        description: 'Release tag: "v" + Cargo package version (e.g. v3.2.3)'
        required: true
        default: 'v3.2.3'
permissions:
  contents: write
jobs:
  build:
    runs-on: windows-latest
    steps:
      - actions/checkout@v4
      - dtolnay/rust-toolchain@stable   # pin exact 1.96.0 via rust-toolchain.toml
      - Swatinem/rust-cache@v2          # key: Cargo.lock + rust-toolchain + Rune rev + features
      - install cmake/ninja/LLVM as needed for whisper-rs / sherpa
      - cache + download ffmpeg 7.1.1 (hash-verify)
      - cache + download Whisper ONNX turbo, Silero, diarization (hash-verify)
      - cache + build whisper.cpp v1.8.6 Vulkan (or prebuilt libs) (hash-verify)
      - cache + download GGUF (hash-verify)
      - cache + fetch/build sherpa-onnx C API 1.13.8 (hash-verify)
      - cargo fmt --check
      - cargo clippy --workspace --all-targets -- -D warnings
      - cargo test --workspace --locked
      - cargo build --release --locked --features vulkan
      - stage payload: models, ffmpeg, whisper/sherpa libs, licenses (cargo-about), assets
      - cargo run -p easyscribe-pack -- pack --binary target/release/easyscribe.exe \
          --payload-dir build/payload --out dist/EasyScribe-v3.2.3.exe --app-version 3.2.3
      - cargo run -p easyscribe-pack -- verify --exe dist/EasyScribe-v3.2.3.exe --full
      - dist/EasyScribe-v3.2.3.exe --version        # must print 3.2.3
      - dist/EasyScribe-v3.2.3.exe --payload-verify # no full extraction, or temp extraction
      - signtool sign /fd SHA256 /tr <timestamp> /td SHA256 ...
      - signtool verify /pa dist/EasyScribe-v3.2.3.exe
      - Get-FileHash dist/EasyScribe-v3.2.3.exe -Algorithm SHA256 > dist/EasyScribe-v3.2.3.exe.sha256
      - softprops/action-gh-release@v2 with draft: true, prerelease: contains(tag, '-'),
        files: EasyScribe-v3.2.3.exe + .sha256
```

Notes:

- **Payload staging** must use the same layout the runtime expects; `easyscribe-pack` should generate `manifest.json`, not hand-write it.
- **whisper engine choice** (subprocess `whisper-cli.exe` vs `whisper-rs`) affects staging and smoke tests. If `whisper-rs`, no `whisper-cli.exe`; keep the GGUF and ensure Vulkan/CPU logging is preserved via the crate’s callbacks. If subprocess, keep `whisper-cli.exe` + all DLLs and run `--help`.
- **sherpa C API** choice (prebuilt official DLLs vs build from source) must be pinned and hashed. Do not link against the Python wheel’s private `.pyd`; use the C API shared library/headers matching the pinned version.
- **Rune** is pinned to the rev recorded in `REWRITE_STATUS.md` (`781e2b6d87cd5f32d6f4365956fad4baaf24ee50`); local development may use `[patch]` to `/home/user/rune-ref`, but CI uses the git rev.
- **Static CRT** (`+crt-static`) preferred; otherwise add `vcruntime140.dll` to payload and test on a clean Windows VM.
- **Cache keys** should be versioned like the current ones, e.g. `sherpa-onnx-win64-1.13.8-v1`, `whispercpp-vulkan-v1.8.6-vksdk1.4.313.2-v1`, `ffmpeg-win64-gyan-7.1.1-essentials-v1`, `sherpa-onnx-whisper-turbo-v1`, `sherpa-onnx-silero-vad-v1`, `sherpa-onnx-diarization-v1`, `ggml-large-v3-turbo-q5_0-v1`. Bump on any version/feature change (Rule 5 analog).
- **Build time/disk**: whisper.cpp Vulkan + Rust LTO can be 20–40 min cold; cache aggressively. Ensure runner disk has room for payload staging (~2 GB) + target (~several GB) + models (~1.7 GB).
- **Artifact size gate**: fail if `EasyScribe-v3.2.3.exe` > 2,000,000,000 bytes (GitHub limit); if close, improve compression or split (splitting violates one-file goal; prefer dropping redundant data).
- **No update exe**: remove `assets/make_update_bundle.py`, the update-layout build steps, `.easyscribe-update.json`, and the update asset from the release. Keep only the single exe + `.sha256`.
- **No `app.bundle`**: remove `Compress-Archive` and the launcher build. `easyscribe-pack` appends the payload.
- **Release notes** must be rewritten: first-run “Getting ready”, no installer, where data lives, how to move/delete, SmartScreen/signing, licenses. Remove “choose install folder/Update and open/shortcut” instructions.
- **Version gate** before build: parse `Cargo.toml` (`^version = "3.2.3"`), assert `"v" + version == inputs.tag`, assert artifacts names; after build assert `--version` and footer.
- **Draft release** remains; do not auto-publish.

### 11.5 What to delete/rename from the Python packaging

| Current file/step | Rust target |
|---|---|
| `launcher/launcher.py`, `launcher/launcher.spec` | delete; replace with `crates/easyscribe-pack` + runtime bootstrap |
| `EasyScribe_whisper.spec` | delete; Cargo release build + PE resources |
| `assets/version_resource.py` | replace with `build.rs`/`winres` |
| `assets/make_update_bundle.py` | delete |
| `assets/collect_licenses.py` | replace with `cargo-about` + staging script |
| `assets/make_icon.py` / `make_splash.py` | keep as asset generators if Rune uses the same images; otherwise port to Rust/build script |
| `tests/test_launcher_safety.py` | replace with payload safety tests; keep data-invariant cases |
| `tests/validate_build.py` | replace with `easyscribe-pack verify`/Rust tests |
| `tests/run_all.py` | replace with `cargo test --workspace`/xtask |
| `requirements.txt` | delete; `Cargo.lock` is the dependency pin |
| `.github/workflows/build-release.yml` | rewrite for Rust/Rune |
| `CLAUDE.md` PyInstaller rules | keep as historical lessons; add a “Rust/Rune packaging” section; do not blindly port PyInstaller-specific rules |
| `README.md` distribution section | rewrite for single exe/extract-once; update paths and sizes |

### 11.6 Target acceptance checklist

- [ ] Release is exactly one `EasyScribe-v3.2.3.exe` (plus `.sha256`); no `.zip`, no `app.bundle`, no update exe, no launcher/sidecar.
- [ ] The exe starts on a clean Windows 10/11 x64 VM with no Python, no VC++ redistributable (static CRT or bundled runtime), and no network.
- [ ] First run shows “Getting ready…” with progress; no folder picker/Install button/EULA/shortcut checkbox.
- [ ] First run extracts once; second run opens in ~1 s and does not copy/read the payload body.
- [ ] Moving/copying the exe to another folder/user still works; cache is recreated if deleted.
- [ ] All models, ffmpeg/ffprobe, whisper GGUF/CLI (or linked libs), sherpa/ONNX DLLs and licenses are real files under the runtime dir.
- [ ] `recordings/`, `logs/`, `settings.json` and user output are never deleted by extraction, GC, repair or version changes.
- [ ] Footer/container verification catches truncation/corruption; extraction is atomic; concurrent first runs do not corrupt the cache.
- [ ] No network code/dependencies in the shipped binary; ffmpeg/ffprobe use `-protocol_whitelist file`; no update check/telemetry.
- [ ] All v3.2.3 app behaviour is present (file/live/record/diarize/rename/recovery/settings/privacy/logs/transcript formats).
- [ ] Licenses complete (`LICENSE.txt`, `THIRD-PARTY-NOTICES.txt`, `licenses/`, FFmpeg GPL/written offer, CC BY attribution, Rust/Rune crate notices).
- [ ] Version `3.2.3` consistent across Cargo, Windows VERSIONINFO, `--version`, workflow tag, artifact name, footer/manifest.
- [ ] Authenticode signed and verified; SHA-256 published; SmartScreen/AV guidance in README.
- [ ] `cargo fmt`, `clippy -D warnings`, unit/integration/payload tests green; CI draft release created with one asset.

---

## 12. Exact current constants and quick reference

```python
# launcher/launcher.py
VERSION = "3.2.3"
MARKER_FILENAME = ".easyscribe-install.json"
BACKUP_DIRNAME = ".easyscribe-old"
USER_DATA_DIRS = frozenset({"recordings", "logs"})
UPDATE_INFO_FILENAME = ".easyscribe-update.json"
_exe_dir = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent.parent
```

```python
# src/config.py
APP_NAME = "EasyScribe"
APP_VERSION = "3.2.3"
BASE_DIR = get_base_dir()  # frozen: Path(sys.executable).parent.resolve(); dev: project root
RECORDING_WORK_DIR = BASE_DIR / "recordings"
SETTINGS_FILE = RECORDING_WORK_DIR / "settings.json"
DEFAULT_RECORDINGS_DIR = _documents_dir() / APP_NAME
WHISPER_ENCODER = BASE_DIR/"models"/"whisper"/"turbo-encoder.int8.onnx"
WHISPER_DECODER = BASE_DIR/"models"/"whisper"/"turbo-decoder.int8.onnx"
WHISPER_TOKENS  = BASE_DIR/"models"/"whisper"/"turbo-tokens.txt"
VAD_MODEL_PATH  = BASE_DIR/"models"/"silero_vad.onnx"
DIARIZATION_SEGMENTATION_MODEL = BASE_DIR/"models"/"diarization"/"segmentation.onnx"
DIARIZATION_EMBEDDING_MODEL    = BASE_DIR/"models"/"diarization"/"embedding.onnx"
FFMPEG_BIN  = BASE_DIR/"ffmpeg"/"ffmpeg.exe"
FFPROBE_BIN = BASE_DIR/"ffmpeg"/"ffprobe.exe"
WHISPERCPP_BIN   = BASE_DIR/"whispercpp"/("whisper-cli.exe" if win32 else "whisper-cli")
WHISPERCPP_MODEL = BASE_DIR/"whispercpp"/"ggml-large-v3-turbo-q5_0.bin"
WHISPERCPP_BEAM_SIZE = 5
LOGS_DIR = BASE_DIR/"logs"
TEMP_DIR = BASE_DIR/"temp"
MIN_FREE_DISK_BYTES = 512 * 1024 * 1024
MAX_LOG_FILES = 10
```

```python
# assets/make_update_bundle.py
UPDATE_INFO_FILENAME = ".easyscribe-update.json"
KEEP = ("models/*", "ffmpeg/*", "whispercpp/*.bin")
SKIP_TOP = ("recordings",)
```

```text
APP_VERSION 3.2.3
Tag v3.2.3
Artifact EasyScribe-v3.2.3.exe
Update artifact EasyScribe-update-v3.2.3.exe
```

---

## 13. Open questions / risks for the Rust packaging work

1. **Data root parity.** Moving `recordings/`/`logs/` from `<install>\` to `%LOCALAPPDATA%\EasyScribe\` changes user-visible paths in the Privacy panel/README. Decide and document: default per-user root + optional portable marker, or preserve `<exe_dir>\EasyScribe` when writable. The plan currently says `%LOCALAPPDATA%`.
2. **Legacy migration.** Do we import `settings.json`/orphan PCMs from an old v3.2.3 install? Recommended: one-time, non-destructive copy of settings + recovery scan; never copy old app binaries.
3. **Rune runtime dependencies.** Confirm whether Rune/wgpu requires non-system DLLs; if so, static-link or ship in payload and set DLL search path. Verify with `dumpbin /dependents` on a clean Windows runner.
4. **sherpa-onnx distribution.** Official Windows C API prebuilt vs build from source; exact DLL set, size, license/NOTICE, hash. Pin 1.13.8 (requirements) or the version chosen in TECH_RESEARCH.
5. **whisper.cpp integration.** `whisper-rs` in-process vs `whisper-cli.exe` subprocess. The latter preserves current device stderr parsing/cancellation exactly; the former reduces payload and process overhead but needs Vulkan feature build and logging parity.
6. **Payload size.** Uncompressed ~1.9–2.1 GB; compressed exe target <2 GB. Measure `store` vs zstd for ONNX/GGUF; if too large, improve compression or revisit the live-mode model choice. The sherpa Whisper ONNX is needed for live mode as well as the file fallback, so “drop the fallback” saves code but **not** the 1.04 GB of ONNX files. Do not compromise 100% functionality.
7. **First-run extraction time.** 1.5–2 GB write + decompression on a slow HDD/USB can take minutes. Show progress; consider `store` for models to reduce CPU at the cost of size. Test on HDD/USB.
8. **Antivirus/SmartScreen.** Large appended payload/self-extraction is a heuristic target. Code signing, clean-VM testing, no UPX, and a stable publisher name are required for a good UX.
9. **GitHub 2 GB asset limit.** Gate in CI; if exceeded, the one-file requirement conflicts with the limit and needs a documented fallback (e.g. split model pack) — avoid unless unavoidable.
10. **Cache corruption/repair.** Define whether `--repair` is automatic on model-load failure or only user-invoked. Do not silently re-download (offline).
11. **Settings schema compatibility.** Running older/newer exes against the same data root must not corrupt `settings.json`; use tolerant JSON, atomic writes, unknown-key preservation.
12. **Code signing certificate.** Decide OV/EV, timestamp server, and CI secret handling; unsigned builds must still work with documented SmartScreen steps.
13. **Long paths / non-ASCII usernames.** Use `\\?\`/longPathAware and Unicode APIs; test a path with spaces/non-ASCII.
14. **Permissions.** `%LOCALAPPDATA%` is normally writable; locked-down enterprise profiles may redirect/deny. Provide a clear error and `EASYSCRIBE_DATA_DIR` escape hatch.
15. **Uninstall.** Document: delete the exe, `%LOCALAPPDATA%\EasyScribe`, and the chosen recordings folder; no registry entries. If the app creates shortcuts, remove them on request.

---

## 14. Source references (oracle)

| Topic | File / section |
|---|---|
| Launcher constants, install/update/version logic | `launcher/launcher.py` lines 39–462, 845–866 |
| Installer UI/state/error strings | `launcher/launcher.py` lines 553–843 |
| Launcher PyInstaller spec/splash | `launcher/launcher.spec` |
| App PyInstaller spec/collected packages | `EasyScribe_whisper.spec` |
| Update bundle builder | `assets/make_update_bundle.py` |
| Windows version resource | `assets/version_resource.py` |
| License collection | `assets/collect_licenses.py` |
| Icon/splash generation | `assets/make_icon.py`, `assets/make_splash.py` |
| CI/release | `.github/workflows/build-release.yml` |
| Launcher safety tests | `tests/test_launcher_safety.py` (27 checks) |
| Version consistency tests | `tests/test_version.py` |
| Build/assembly validation | `tests/validate_build.py` |
| Test aggregation/exclusions | `tests/run_all.py` |
| Offline guarantee | `src/offline_guard.py`, `tests/test_offline_guard.py`, README “Offline guarantee” |
| Layout/distribution prose | `README.md` lines 196–233, 271–305 |
| Packaging lessons | `CLAUDE.md` Rules 5, 14, 16–18, 20 |
| Audit finding | `EASYSCRIBE_AUDIT.md` §7.10 (and §2 scorecard) |
| Action plan decision | `EASYSCRIBE_ACTION_PLAN.md` §8.3 |
| Rewrite architecture/decisions | `REWRITE_PLAN.md`, `REWRITE_STATUS.md` |
| Licenses | `licenses/THIRD-PARTY-NOTICES.txt` and `licenses/*` |
| Config paths/constants | `src/config.py` |
| Logging/rotation | `src/logger.py` |
| Settings persistence | `src/settings.py` |
| App startup/dependency checks/taskbar ID | `src/main.py` |
