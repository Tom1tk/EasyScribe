# EasyScribe → Rune Rewrite — Technical Research & Implementation Reference

**Status:** engineering reference for the Rust/Rune rewrite. This is not user-facing documentation.
**Oracle:** EasyScribe **v3.2.3**, branch `main` at commit `8323516`.
**Target:** `/home/user/EasyScribe-rune`, branch `rune-rewrite`.
**Rune pin:** `anomalyco/rune` at `781e2b6d87cd5f32d6f4365956fad4baaf24ee50` (crate version `0.0.1`; `rune` and `rune-kit` are workspace crates in that repository).
**Rust:** workspace `rust-version = "1.88"`, edition 2024.

> **Implemented decision (2026-11-06).** The shipped preview uses in-process
> `whisper-rs` 0.16 (whisper.cpp 1.8.3, CPU-only), sherpa-onnx **1.12.9**
> static libraries, and the static MSVC runtime. Section 8.1's
> `whisper-cli.exe` Vulkan recommendation was **not** adopted for this release:
> a load-time `vulkan-1.dll` import breaks CPU fallback on machines without a
> Vulkan driver, and subprocess isolation would add a second engine path. The
> in-process engine preserves beam size 5, transcript output and cooperative
> Stop; whisper decode runs on a fresh thread to avoid ONNX Runtime
> thread-local state. See `REWRITE_STATUS.md` and
> `REFERENCE_PACKAGING.md` §1.1 for the implemented contract. The rest of this
> document remains the research rationale, including the failed/risky paths.

This document records what was verified against source, what is a recommendation, and what still needs a Windows spike. It deliberately distinguishes:

- **Rune built-in** — already in the pinned Rune/rune-kit source; no extra UI framework.
- **Add** — a Rust crate the rewrite must add because Rune has no Windows implementation.
- **Spike** — plausible but unverified on Windows in this environment; do not ship without a CI/VM check.

Normative words: **MUST**, **SHOULD**, **MAY**. “Current” means the Python v3.2.3 implementation; “scaffold” means the Rust workspace as it exists today; “Rune” means the pinned revision above. Do not confuse the two when a section compares behavior.

**Verification sources**

| Area | Source checked |
|---|---|
| Rune app/runtime/El/Tx | `crates/rune/src/tooling.rs`, `app.rs`, `el.rs`, `el/tx.rs`, `runtime.rs`, `runtime/*`, `file_dialog.rs`, `clipboard/*`, `window/handle.rs`, `windows.rs` |
| Rune behavior/API docs | `crates/rune/README.md` at the pin (sections “What exists”, “App lifetime”, “Text fields”, “File dialogs”, “Dropping files in”, “Multiple windows and overlays”, “Agent tooling”) |
| rune-kit | `crates/rune-kit/README.md`, `src/kit.rs`, `src/actions.rs`, `src/text_input.rs`, `src/dialog.rs`, `src/segmented.rs`, `src/feedback.rs`, `src/lib.rs` |
| Windows/native crates | crate sources in `~/.cargo/registry/src/...`: `cpal 0.18.2`, `hound 3.5.1`, `rodio 0.23.0`, `rfd 0.17.2`, `tracing 0.1.44`, `tracing-appender 0.2.4`, `sherpa-onnx 1.13.8`, `sherpa-onnx-sys 1.13.8`, `whisper-rs 0.16.0`, `whisper-rs-sys 0.15.0`, `rubato 5.0.1`, `symphonia 0.6.1`, `embed-resource 3.0.12`, `windows-sys 0.61.2` |
| Current app | `src/*.py`, `crates/easyscribe-core/src/*.rs`, `crates/easyscribe/src/*.rs`, `Cargo.toml`, `Cargo.lock`, `CLAUDE.md` |
| Packaging/UI/backend parity | `docs/rewrite/REFERENCE_PACKAGING.md`, `REFERENCE_BACKEND.md`, `REFERENCE_UI.md` |

**Contents**

1. [Executive summary and decisions](#1-executive-summary-and-decisions)
2. [Target architecture](#2-target-architecture)
3. [Rune application model](#3-rune-application-model)
4. [rune-kit, theme and components](#4-rune-kit-theme-and-components)
5. [Input, files and Windows integration](#5-input-files-and-windows-integration)
6. [Headless tests and screenshots](#6-headless-tests-and-screenshots)
7. [Native crates: what Rune has, what to add](#7-native-crates-what-rune-has-what-to-add)
8. [Speech inference](#8-speech-inference)
9. [FFmpeg versus Rust decoding](#9-ffmpeg-versus-rust-decoding)
10. [Single-file Windows `.exe`](#10-single-file-windows-exe)
11. [Implementation order and acceptance gates](#11-implementation-order-and-acceptance-gates)
12. [Risk register](#12-risk-register)
13. [Source map](#13-source-map)

---

## 1. Executive summary and decisions

### 1.1 The shape of the rewrite

EasyScribe is a **single-window, fixed-light-theme, fully offline Windows desktop app**. The rewrite should keep that shape:

- **UI:** one Rune window, with in-window overlays for the speaker-naming panel, the Privacy panel, the activity/result card, and the details log. A second native window is optional for Privacy if exact modeless/Toplevel parity is required; it is not needed to ship.
- **State:** one `AppState` owned by the Rune runtime, plus `Arc` handles to background jobs and immutable path/config data.
- **Work:** all long operations (payload extraction, model load, ffmpeg, whisper, diarization, file IO) run off the UI thread; results land in state through `poll` or `AppHandle::update`.
- **Native services:** Rune for drag/drop, clipboard, file dialogs, shortcuts, windows and modals; `cpal`/`hound`/`rodio` for Windows microphone/WAV/playback; `rfd` only for multi-select and pre-GUI message boxes; Win32 for icon/AUMID/opening files.
- **Speech:** official `sherpa-onnx` 1.13.8 for the live VAD + Whisper-ONNX path and diarization; bundled `whisper-cli.exe` 1.8.6 (Vulkan) for file transcription, preserving current quality and process isolation. `whisper-rs` 0.16 CPU is an optional in-process fallback, not the release engine.
- **Audio decode:** keep bundled `ffmpeg.exe`/`ffprobe.exe`. Do not replace them with Symphonia/`ffmpeg-next` in this rewrite.
- **Packaging:** append a compressed payload and a 128-byte footer to the GUI exe; extract once to `%LOCALAPPDATA%\EasyScribe\runtime\v1-<hash>`; no installer, no `%TEMP%` re-extraction, no sidecar bundle.

### 1.2 Decision table

| Decision | Choice | Why | Status |
|---|---|---|---|
| UI framework | Rune only | Handles windows, text, drag/drop, dialogs, clipboard, modals, shortcuts, headless tests; one renderer | Adopt |
| State ownership | `AppState` on UI thread; `Arc` services | Rune state is `'static` but not `Send`; workers must not touch it | Adopt |
| Worker → UI | `mpsc` + `Waker` + `poll`, or `AppHandle::update` | Rune has no app async executor; both are documented patterns | Adopt |
| File dialogs | `Tx::open_file/save_file/open_folder` | Native, headless-scriptable, no blocking | Adopt |
| Multi-file picker | `rfd::FileDialog::pick_files()` on a worker | Rune’s dialog API is single-select | Add (scoped) |
| Pre-GUI errors | `rfd::MessageDialog` | No Rune window exists yet; matches current native dialogs | Add (scoped) |
| Drag/drop | `El::on_drop_files` + `accepts_extensions` | Built-in, Explorer-compatible, headless-testable | Adopt |
| Clipboard | `Tx::copy` / `Tx::read_clipboard` | Built-in; no `arboard` | Adopt |
| Window icon | Win32 `WM_SETICON` + `embed-resource` | `rune::set_app_icon` is macOS-only | Add |
| Open file/folder | Win32 `ShellExecuteW("open", …)` | `Tx::open_url` accepts only `http`, `https`, `mailto` | Add |
| Shortcuts | `AppSpec::on_key` | Current app has no global hotkeys; `App::global_shortcut` is OS-wide | Adopt |
| Microphone | `cpal` 0.18 | Rune `Recorder` is macOS-only | Add |
| WAV | `hound` 3.5 | Rune has no WAV writer; current core already uses hound | Add |
| Speaker clip playback | `rodio` 0.23 + `wav` feature | Rune `Player` is macOS-only | Add |
| File watching | none | Not needed for parity; `notify` only if a real feature needs it | Avoid |
| Logging | `tracing` + `tracing-appender` (or keep current `log` writer) | Structured jobs, non-blocking file writes, rotation | Add |
| JSON | `serde_json` | Manifest, settings, sidecars | Add |
| File ASR | bundled `whisper-cli` 1.8.6 Vulkan | Current quality/behavior; process isolation; beam 5 | Adopt |
| In-process ASR | `whisper-rs` 0.16 CPU, feature-gated | Useful fallback when sidecar is missing; Vulkan breaks single-exe static CRT | Optional |
| Live ASR | official `sherpa-onnx` 1.13.8 | Current live path; CPU; no Vulkan provider | Adopt |
| Diarization | official `sherpa-onnx` 1.13.8, in a child worker | Progress needs C callback; cancellation needs kill | Adopt |
| `sherpa-rs` 0.6.8 | remove | Deprecated; pins 1.12.9; no progress callback in the same shape | Migrate |
| Audio decode | bundled ffmpeg/ffprobe | Covers Opus, AVI, WMA, video containers; parity | Adopt |
| Rust decode | not for shipping | Symphonia lacks Opus/AVI/WMA; `ffmpeg-next` adds FFmpeg dev/DLL complexity | Avoid |
| Payload container | keep ZIP for v1; ESPAK001 is the better long-term format | Existing scaffold works; footer/PE fixes are mandatory; container rewrite is not | Conditional |
| Payload location | appended overlay + footer | No 2 GB PE resource limit; streaming; easy verify | Adopt |
| Extraction | once to per-user cache, atomic staging | Fast repeat launch; no installer; read-only exe-safe | Adopt |
| Code signing | Authenticode after packing | SmartScreen/AV; must sign the final bytes | Adopt |
| Static CRT | `+crt-static` for MSVC | sherpa static archives are `/MT`; avoids VC redist | Spike |
| whisper-rs Vulkan | not in release | Links `vulkan-1` at load; `/MD` C++ objects conflict with `/MT` | Avoid |
| Installer/shortcuts | none | User goal is one exe; no install step | Adopt |

### 1.3 What is still unproven

These are the items that need a Windows build/VM before they can be called done:

1. **Rune/wgpu on Windows** in this exact revision: window creation, DPI, drag/drop, native dialog, text input/IME. Build a “hello, window” smoke binary first.
2. **`+crt-static` with Rune + wgpu + sherpa-onnx static archives.** If link fails, fall back to shared sherpa + VC runtime, or keep `/MD` and ship the VC runtime in the payload. Do not guess.
3. **`whisper-cli.exe` Vulkan load-time dependencies.** Run `dumpbin /dependents` on the CI artifact. If it imports `vulkan-1.dll`, decide between a CPU fallback binary and the existing sherpa fallback.
4. **`sherpa-onnx` static link on Windows**: ONNX Runtime may pull Windows system libraries through `#pragma comment(lib, ...)`; `sherpa-onnx-sys` emits no Windows system libs itself. Linker errors are the signal.
5. **rfd multi-select on a worker thread** with COM apartment/owner behavior. Test on Windows 10 and 11.
6. **Authenticode + appended footer**: `signtool` adds a certificate table after the footer. The parser in §10.3 is designed for it but needs a signed-artifact test.

---

## 2. Target architecture

### 2.1 Process and data flow

```
EasyScribe.exe
  │
  ├─ startup (no Rune window yet)
  │    ├─ SetCurrentProcessExplicitAppUserModelID("EasyScribe.App")
  │    ├─ read own PE + payload footer
  │    ├─ cache hit?  ── yes ──► Rune main window
  │    └─ cache miss ──► Rune "Getting ready" scene
  │                         └─ extraction worker → staging → .complete → rename
  │
  ├─ Rune UI thread (AppState, view, Cx, El, Tx)
  │    ├─ file job request ──► worker thread / child processes
  │    ├─ record request  ──► cpal callback → 16 kHz queue → live inference thread
  │    └─ poll() drains mpsc events and applies them
  │
  ├─ file job worker
  │    ├─ ffmpeg.exe  (16 kHz mono PCM WAV)
  │    ├─ whisper-cli.exe (Vulkan, beam 5)
  │    └─ optional: EasyScribe.exe --worker diarize (sherpa-onnx)
  │
  └─ live worker
       ├─ Silero VAD (sherpa-onnx)
       └─ Whisper ONNX recognizer (sherpa-onnx)
```

### 2.2 Why a child process for diarization (and optionally file ASR)

The current Python app runs diarization in a **child process** (`multiprocessing`) and sends progress over a pipe. That is not incidental:

- ONNX Runtime/sherpa-onnx can abort the process on some concurrent or malformed calls; Python serializes all ONNX calls with a lock for the same reason.
- The official Rust `OfflineSpeakerDiarization::process` has **no progress callback** and no cancellation. The C API’s progress callback exists, but in 1.13.8 the callback’s return value is documented as ignored, so it cannot abort.
- A child process can be killed for **Stop**, gives clean memory release after a 1–2 GB diarization load, and keeps a native crash from taking down the UI.

Use the same executable with an internal `--worker` mode. This avoids shipping a second binary and keeps the payload small.

```rust
// main.rs: internal worker dispatch before any Rune startup.
if let Some(code) = easyscribe::diarize_worker::run_if_requested() {
    std::process::exit(code); // no Rune, no window, no GPU
}
```

Worker protocol (stdout, one JSON object per line):

```json
{"kind":"progress","value":42}
{"kind":"done","turns":[{"start":0.0,"end":2.4,"speaker":"SPEAKER_00"}]}
{"kind":"error","message":"..."}
```

Parent requirements:

- `CREATE_NO_WINDOW` (`0x0800_0000`) on Windows.
- `stdout` piped and read; `stderr` null (or read on a dedicated thread if native logs are wanted). Never leave a piped stream unread.
- Keep `Child` in `Arc<Mutex<Option<Child>>>` or a job handle so `on_close`/Stop can `kill()`.
- The child reads the WAV and writes no files; there is no worker output to delete. The parent deletes its own temp WAV after the job.
- Do not pass user text on the command line; pass file paths only. Validate paths.

### 2.3 Data and cache layout

```
%LOCALAPPDATA%\EasyScribe\
  runtime\
    v1-<payload-hash-prefix>\
      manifest.json
      models\
        whisper\turbo-{encoder,decoder}.int8.onnx
        whisper\turbo-tokens.txt
        silero_vad.onnx
        diarization\segmentation.onnx
        diarization\embedding.onnx
      whispercpp\
        whisper-cli.exe
        *.dll
        ggml-large-v3-turbo-q5_0.bin
      ffmpeg\
        ffmpeg.exe
        ffprobe.exe
      licenses\
      LICENSE.txt
      THIRD-PARTY-NOTICES.txt
      .complete
    .lock
  recordings\           crash-safe PCM + JSON, settings.json
  logs\                 EasyScribe_YYYYMMDD_HHMMSS.log, keep 10
  temp\                 scratch WAV/JSON, cleaned at start/close
```

- `%LOCALAPPDATA%` is per-user and writable without admin. Use `dirs::data_local_dir()` (it resolves the known folder on Windows) or `SHGetKnownFolderPath(FOLDERID_LocalAppData)`.
- User data is **outside** the versioned runtime. Updating the exe must never move or delete `recordings\` or `logs\`.
- `EASYSCRIBE_DATA_DIR`, `EASYSCRIBE_RUNTIME_DIR`, `EASYSCRIBE_PAYLOAD_DIR` remain development/test overrides.
- An `EasyScribe.portable` marker next to the exe MAY redirect the data root to `<exe_dir>\EasyScribe-data`; this is optional and must not be the default (Program Files/read-only media).

### 2.4 Startup phases

The scaffold already has most of the right shape: `main` calls `payload::needs_prepare()` (a cheap footer + cache check), starts Rune with a provisional `Paths::at(data_root)`, and `AppState::begin_prepare()` spawns `payload::prepare_with_progress` on a worker. The remaining cleanup is to collapse `needs_prepare`, `Paths::at`, and `Paths::discover` into one explicit `Boot` inspection so footer damage is not swallowed by `unwrap_or(false)` and headless/dev paths are chosen deliberately:

The concrete `Boot` type is defined in §10.6.1; in outline:

```rust
pub enum Boot {
    Dev(Paths),
    Ready(Paths),
    NeedsExtraction {
        exe: PathBuf,
        footer: Footer,
        data_root: PathBuf,
    },
}
```

`AppState` starts in `Phase::Preparing { progress, ... }`; the view shows the EasyScribe mark, “Getting ready…”, a determinate bar from `bytes_written / total_uncompressed`, and “The first start can take a minute. Please wait.” A worker thread extracts; `poll` drains progress; on success it builds `Paths` and replaces the phase. On failure the same window shows a modal with **Retry**, **Open data folder**, **Exit**.

Pre-GUI fatal errors (bad footer, truncated exe, no writable data dir) use `rfd::MessageDialog` and exit. Do not create a Rune window just to report that the payload is unreadable; the Rune binary may still work, but the message is clearer before GPU startup.

---

## 3. Rune application model

### 3.1 Exact `AppSpec` contract

From `crates/rune/src/tooling.rs`:

```rust
pub struct AppSpec<S: 'static> {
    pub title: &'static str,
    pub size: (f32, f32),
    pub min_size: Option<(f32, f32)>,
    pub init: fn() -> S,
    pub view: fn(&S, &mut Cx<S>) -> El,
    pub scenes: &'static [(&'static str, fn(&mut S))],
    pub fonts: &'static [&'static [u8]],
    pub families: (&'static str, &'static str),
    pub setup: Option<fn(&mut S, Waker)>,
    pub poll: Option<fn(&mut S, f64, &mut Tx) -> bool>,
    pub on_key: Option<fn(&mut S, &KeyEvent, &mut Tx) -> bool>,
}

impl<S: 'static> AppSpec<S> {
    pub fn new(title: &'static str, size: (f32, f32), init: fn() -> S, view: fn(&S, &mut Cx<S>) -> El) -> Self;
}
```

Key consequences:

- `init`, `view`, `setup`, `poll`, `on_key` are **function pointers**, not closures. They cannot capture a channel, a handle, or a runtime value. Put those in `AppState` or use `App::init`.
- `App::init(move || state)` accepts an `FnOnce() -> S + 'static` closure and replaces `AppSpec::init`. This is the correct place to inject an `AppHandle`, a `Receiver`, or a pre-built `Paths`.
- `AppSpec::setup` receives `Waker = Arc<dyn Fn() + Send + Sync>`. It is called once after state construction, before the first view.
- `AppSpec::poll` runs at the start of every frame, including timer wakes and hidden-window wakes. It must be cheap and return `true` only when visible state changed.
- `AppSpec::on_key` is the window-level shortcut hook. It is called only for keys the focused control and focus navigation did not consume.

### 3.2 `App`, `AppHandle`, `WindowHandle`

Verified signatures:

```rust
impl<S: 'static> App<S> {
    pub fn new(spec: AppSpec<S>) -> Self;
    pub fn init(self, f: impl FnOnce() -> S + 'static) -> Self;
    pub fn on_close(self, f: fn(&mut S, &mut Tx)) -> Self;
    pub fn on_window(self, f: impl FnMut(&WindowHandle) + 'static) -> Self;
    pub fn on_ready(self, f: impl FnMut(&WindowHandle) + 'static) -> Self;
    pub fn handle(&self) -> AppHandle<S>;
    pub fn window<T: 'static>(&mut self, spec: WindowSpec<T>) -> (AppHandle<T>, WindowHandle);
    pub fn run(self);
}

impl<S: 'static> AppHandle<S> {
    pub fn update(&self, f: impl FnOnce(&mut S, &mut Tx) + Send + 'static);
    pub fn update_if(&self, f: impl FnOnce(&mut S, &mut Tx) -> bool + Send + 'static);
    pub fn wake(&self);
    pub fn open(&self);
    pub fn show(&self);
    pub fn hide(&self);
    pub fn close(&self);
    pub fn quit(&self);
}
```

`AppHandle<S>` is `Clone + Send + Sync` even when `S` is not `Send`. `update` runs the closure on the UI thread, in call order, and commits on the next frame. The closure itself must be `Send`; capture owned values (`PathBuf`, `String`, a job result), never references into state.

`WindowHandle::with_raw_handle`:

```rust
pub fn with_raw_handle<R>(
    &self,
    f: impl for<'a> FnOnce(winit::raw_window_handle::WindowHandle<'a>) -> R,
) -> Result<R, String>;
```

Do not retain the raw pointer after the callback. `on_window` runs after every native creation (including reopen); `on_ready` runs once per native creation after the first frame and is **not called by headless tooling**.

### 3.3 State, view, `Cx`, `El`

`view` is a pure function of `&S` and `&mut Cx<S>`; it returns an `El` tree. Handlers are queued and run between frames with `&mut S`, then the view re-runs and the diff retargets properties.

- **State:** `AppState` holds all UI state. It is not `Send`; do not move it to a thread.
- **`Cx`:** host environment (`system_dark`, `scale` via `cx.env().scale`, `is_reduced_motion`, `titlebar_inset`, `accent`, `scroll_bars`). EasyScribe is fixed light, so `system_dark` is ignored for the palette; use it only for diagnostics.
- **`El`:** layout/style/handlers. Keys are one namespace; prefix app keys (`files/row/0`, `privacy/close`) to avoid collisions with kit keys (`label`, `value`, `input`, …).
- **Handlers:** `.on_press(|s| ...)`, `.on_press_tx(|s, tx| ...)`, `.on_drag`, `.on_drop_files`, `.on_key` on text inputs. The handler runs on the UI thread.

### 3.4 Minimal app skeleton

```rust
use std::path::PathBuf;
use std::sync::mpsc::Receiver;

use rune::{App, AppSpec, Cx, El, Key, KeyEvent, TextEdit, Tx, Waker};
use rune_kit::{Dialog, Kit, TextInput};

fn placeholder() -> AppState {
    unreachable!("App::init supplies the real state")
}

fn main() {
    // Must run before the first window and before any shell dialog.
    platform::set_process_identity();

    let startup = match startup::open() {
        Ok(startup) => startup,
        Err(error) => {
            // No Rune window exists yet; this is the one acceptable blocking rfd use.
            rfd::MessageDialog::new()
                .set_title("EasyScribe cannot start")
                .set_description(&error.to_string())
                .set_buttons(rfd::MessageButtons::Ok)
                .show();
            std::process::exit(1);
        }
    };

    let mut spec = AppSpec::new(
        "EasyScribe 3.2.3",
        (900.0, initial_height()),
        placeholder,
        view,
    );
    spec.min_size = Some((760.0, 600.0));
    spec.scenes = SCENES;
    spec.fonts = &[];
    // Rune discovers no generic system fonts; this names the bundled Inter
    // and the system mono slot. Do not use ("Segoe UI", "Consolas").
    spec.families = ("Inter", "ui-monospace");
    spec.setup = Some(setup);
    spec.poll = Some(poll);
    spec.on_key = Some(on_key);

    let app = App::new(spec);
    let handle = app.handle();
    let state_handle = handle.clone();

    app.init(move || AppState::new(startup, state_handle))
        .on_window(platform::set_window_icon)
        .on_ready(move |_window| {
            // Desktop only; not called headless. Start optional status work here.
            let handle = handle.clone();
            std::thread::spawn(move || {
                // ... slow, optional work ...
                handle.update(|s, _tx| s.status_ready = true);
            });
        })
        .on_close(AppState::on_close)
        .run();
}

fn view(s: &AppState, cx: &mut Cx<AppState>) -> El {
    let k = s.kit();
    let mut layer: Vec<El> = Vec::new();

    let body = match &s.phase {
        Phase::Preparing(p) => preparing_view(s, p, &k),
        Phase::Ready(paths) => main_view(s, paths, &k, &mut layer),
    };

    rune_kit::root(body, layer)
}

fn setup(s: &mut AppState, waker: Waker) {
    s.set_waker(waker);
}

fn poll(s: &mut AppState, now: f64, tx: &mut Tx) -> bool {
    s.poll(now, tx)
}

fn on_key(s: &mut AppState, ev: &KeyEvent, tx: &mut Tx) -> bool {
    if ev.key == Key::Escape && s.close_top_overlay() {
        return true;
    }
    if ev.is_command('o') && s.phase.is_ready() {
        s.choose_files(tx);
        return true;
    }
    false
}
```

Notes:

- `spec.fonts = &[]` keeps Rune’s bundled Inter. Supplying a Segoe UI TTF would work only if the file is embedded; `families: ("Segoe UI", …)` alone does **not**, because Rune creates its font collection with `system_fonts: false` and only registers symbol/CJK/mono/math system faces.
- `on_ready` is intentionally not used for anything the headless tests or first view need. Use `App::init`/`setup` for state that must exist in tests.
- `on_close` is a function pointer; if it needs a captured job handle, put that handle in `AppState` and use `AppState::on_close`.

### 3.5 Threading, background work, async results, timers

Rune has **no public async executor** for app work. `Tx::open_file`/`read_clipboard` are callback-based and must not be awaited with `pollster::block_on` on the UI thread. The supported patterns are:

#### Pattern A: `mpsc` + `Waker` + `poll` (preferred for jobs)

```rust
pub enum JobEvent {
    Progress(f32),
    Log(String),
    Done(JobResult),
    Failed(String),
}

pub struct Jobs {
    tx: Sender<JobEvent>,
    rx: Receiver<JobEvent>,
    cancel: Arc<AtomicBool>,
    waker: Option<Waker>,
    child: Arc<Mutex<Option<std::process::Child>>>,
}

impl AppState {
    pub fn set_waker(&mut self, waker: Waker) {
        self.jobs.waker = Some(waker);
    }

    pub fn start_job(&mut self) {
        let tx = self.jobs.tx.clone();
        let cancel = self.jobs.cancel.clone();
        let child = self.jobs.child.clone();
        let waker = self.jobs.waker.clone();

        std::thread::Builder::new()
            .name("EasyScribeJob".into())
            .spawn(move || {
                let send = |event| {
                    let _ = tx.send(event);
                    if let Some(waker) = &waker {
                        waker(); // wakes poll; no state access from this thread
                    }
                };
                match workers::run(&cancel, &child, &send) {
                    Ok(result) => send(JobEvent::Done(result)),
                    Err(error) => send(JobEvent::Failed(error.to_string())),
                }
            })
            .expect("spawn job");
    }

    pub fn poll(&mut self, _now: f64, _tx: &mut Tx) -> bool {
        let mut changed = false;
        while let Ok(event) = self.jobs.rx.try_recv() {
            changed = true;
            self.apply_job_event(event);
        }
        changed
    }
}
```

`Waker` is `Arc<dyn Fn() + Send + Sync>`. `AppHandle::wake()` is the same wake; if you already have a handle, use `Arc::new(move || handle.wake())` and skip `setup`.

#### Pattern B: `AppHandle::update` (preferred for one-shot results)

```rust
let handle = app.handle();
std::thread::spawn(move || {
    let result = expensive_owned_work();       // returns Send data
    handle.update(move |s, _tx| s.finish(result));
});
```

`update` runs on the UI thread; the closure must be `Send + 'static`, but `S` need not be `Send`. Do not call `handle.update` from the audio callback or from a hot loop for every 512-sample chunk; use a queue and wake at a lower rate.

#### Pattern C: `rune::Latest<T>` for “newest answer wins”

Use `Latest` for search/path completion, not for transcription. From the Rune README:

```rust
results: Latest::default().debounce(Duration::from_millis(60)),
// setup:
s.results.set_waker(waker);
// handler:
s.results.request(query.clone(), |q| index.search(&q));
// poll:
let Some(hits) = s.results.take() else { return false };
s.hits = hits;
true
```

`Latest` keeps up to four shared workers, drops stale answers, and wakes once per answer. It is `Send`-friendly but `take()` must happen in `poll`.

#### Timers

`Tx::wake_after(Duration)` schedules `poll`; it does **not** create a timer thread. Deadlines due by the same frame coalesce into one `poll`.

```rust
pub struct RecordTimer {
    pub next_tick: f64,   // seconds on the poll clock (virtual in headless)
    pub seconds: f64,
}

fn poll(s: &mut AppState, now: f64, tx: &mut Tx) -> bool {
    let mut changed = s.poll_events();

    if s.recording {
        if now >= s.timer.next_tick {
            s.timer.seconds += 0.1;
            s.timer.next_tick = now + 0.1;
            changed = true;
        }
        // Keep a 10 Hz timer alive while recording. Coalesces with other wakes.
        tx.wake_after(Duration::from_millis(100));
    }

    changed
}
```

Use the `now` argument, not `Instant::now()`, so headless tests advance deterministically. `wake_after` is unscaled by Rune’s animation time scale. For debounce, keep the latest deadline in state and reschedule the remaining duration. If a frame arrives late, add the elapsed time since the last tick (`now - last_tick`), not a fixed 0.1, or the timer undercounts.

#### Threading rules (hard requirements)

| Rule | Reason |
|---|---|
| Never touch `AppState`, `El`, `Cx`, `Tx`, `TextEdit` from a worker | State is UI-thread-owned and not `Send` |
| Never block the UI thread on a native dialog, ffmpeg, model load, or `join()` of a long job | Freezes the window; Rune warns on handlers/poll over 8 ms in debug |
| Capture only owned, `Send` data in `AppHandle::update` | The closure crosses threads |
| Keep `TextEdit` in state; do not reconstruct it per frame | Reconstruction resets caret, selection and undo history |
| Do not `pollster::block_on` a Rune file-dialog future on the UI thread | Deadlocks native dialog presentation |
| Use one inference thread or one mutex for ONNX calls | Concurrent sherpa calls can abort the process |
| Make `poll` return `true` only when visible state changed | An idle window should draw nothing |
| Never assume `on_ready` ran in tests | Headless tooling does not call it |
| Do not rely on `on_close` to cancel work after the window is gone | Rune cannot veto a close; `on_close` runs, then the process ends |

### 3.6 Closing behavior

Verified: Rune’s `WindowEvent::CloseRequested` closes the window; `App::on_close` runs before it goes, but there is **no cancel API**. The current Python app has no `WM_DELETE_WINDOW` handler and no close confirmation, so parity is “close immediately.”

Recommended `AppState::on_close`:

```rust
fn on_close(&mut self, _tx: &mut Tx) {
    // Stop new work.
    self.jobs.cancel.store(true, Ordering::SeqCst);

    // Kill any child process so it cannot outlive the app.
    if let Some(mut child) = self.jobs.child.lock().unwrap().take() {
        let _ = child.kill();
        let _ = child.wait();
    }

    // Stop capture and join the PCM writer. The writer fsyncs every ~50 chunks,
    // so this is bounded; do not wait for a full decode here.
    if self.mic.is_recording() {
        let _ = self.mic.stop(false, None); // keep raw PCM for recovery
    }

    // Do not delete an unfinished PCM/JSON pair. Recovery offers it next launch.
    logger::clean_temp_files(&self.paths.temp_dir);
}
```

If a confirm-on-close is ever required, it needs a Win32 `WM_CLOSE` subclass through `with_raw_handle`; Rune cannot cancel a close. That is a product change, not parity, and is out of scope.

---

## 4. rune-kit, theme and components

### 4.1 What the kit is good for here

`rune-kit` supplies semantic components that read a `Kit` of tokens. EasyScribe’s visual language is specific (teal/coral/amber feature colours, 38 px segmented switch, exact card metrics). Use the kit where its behavior/accessibility is valuable and its look can be themed:

| UI element | Use kit? | Recommendation |
|---|---|---|
| Text input / text area | Yes | `TextInput`, `TextArea`; bind `TextEdit` in state |
| Dialog / modal | Yes | `Dialog` for Privacy/confirm; `.on_close` required |
| Buttons | Partly | `Button` for neutral/secondary/ghost; feature CTAs need a local `feature_button` for exact teal/coral/amber and 44 px height |
| Progress | Partly | `Progress` for determinate/indeterminate; colour is `border_active`, so a local bar is needed for green/coral/amber states |
| Segmented mode switch | No | Local element: selected colour changes teal↔coral and kit uses one accent |
| Cards, rows, step badges | No | Local layout/theme; kit `Card` is a different visual |
| Toast/error banner | No | Current banner is inline and persistent; a toast would change behavior |
| Tooltip | Optional | `Tooltip` for icon-only buttons if needed |
| Checkbox/radio tiles | Partly | Kit `Checkbox` for standard boxes; option tiles are custom 2 px borders |

Do not rewrite the existing hand-built views just to use kit components. The kit is a tool, not a requirement.

### 4.2 Theme mapping

Rune has `rune::Theme` (a named token map whose colour values are read by key) and `rune_kit::Kit` (semantic component tokens). EasyScribe is fixed light, so the app already owns a plain palette struct in `crate::theme::Theme`; map that struct onto `Kit` at startup. Do not confuse it with `rune::Theme`, which has no `teal`/`coral` fields. The simplest reliable approach is:

```rust
use crate::theme::Theme;
use rune::Color;
use rune_kit::Kit;

pub fn easy_kit(t: &Theme) -> Kit {
    let mut k = Kit::neutral(false); // false = light

    // Page / surfaces
    k.c.page = t.bg;
    k.c.bg_base = t.surface;
    k.c.bg_deep = t.bg;
    k.c.layer_01 = t.surface_alt;
    k.c.layer_02 = t.secondary;
    k.c.layer_03 = t.secondary_hover;
    k.c.layer_04 = t.border;

    // Text
    k.c.text_base = t.ink;
    k.c.text_muted = t.muted;
    k.c.text_faint = t.faint;
    k.c.text_accent = t.teal_ink;
    k.c.icon_base = t.ink;
    k.c.icon_muted = t.muted;
    k.c.icon_faint = t.faint;

    // Borders / interaction
    k.c.border_muted = t.border;
    k.c.border_base = t.border;
    k.c.border_strong = t.border_strong;
    k.c.border_focus = t.teal;
    k.c.hover = t.secondary_hover;
    k.c.pressed = t.border_strong;
    k.c.bg_button_neutral = t.secondary;
    k.c.bg_accent = t.teal;

    // Feature states used by kit components.
    k.c.success = rune_kit::StateColors {
        bg: t.green_soft,
        fg: t.green_ink,
        border: t.green,
        icon: t.green_ink,
    };
    k.c.warning = rune_kit::StateColors {
        bg: t.amber_soft,
        fg: t.amber_ink,
        border: t.amber,
        icon: t.amber_ink,
    };
    k.c.danger = rune_kit::StateColors {
        bg: t.crimson_soft,
        fg: t.crimson,
        border: t.crimson,
        icon: t.crimson,
    };
    k.c.info = rune_kit::StateColors {
        bg: t.sky_soft,
        fg: t.sky_ink,
        border: t.sky,
        icon: t.sky_ink,
    };

    // Metrics from REFERENCE_UI.md.
    k.radius.sm = 6.0;
    k.radius.md = 10.0;
    k.radius.lg = 14.0;
    k.ty.ui = rune::tok::TextStyle::new(13.0, 400.0).lh(19.5);
    k.ty.ui_medium = rune::tok::TextStyle::new(13.0, 700.0).lh(19.5);
    k.ty.ui_compact = rune::tok::TextStyle::new(13.0, 400.0).lh(16.0);
    k.ty.small = rune::tok::TextStyle::new(12.0, 400.0).lh(17.0);
    k.ty.small_medium = rune::tok::TextStyle::new(12.0, 700.0).lh(17.0);
    k.ty.mono = rune::tok::TextStyle::new(11.0, 400.0).lh(16.5)
        .family(rune::tok::Family::SystemMono);

    k
}
```

`StateColors` fields are `bg`, `fg`, `border`, `icon`; the assignment above matches the public struct. If the kit changes a field, the compiler will say so; that is preferable to copying color literals into components.

**Fonts.** Rune’s `TextSystem` registers bundled Inter and creates the collection with `system_fonts: false`. Only symbol/CJK/mono/math system faces are loaded for fallback. Therefore:

- `families: ("Segoe UI", "Consolas")` silently falls back to Inter for sans; it does not load Segoe UI.
- Use `families: ("Inter", "ui-monospace")` and `Family::SystemMono` for the details log/timer.
- If exact Segoe UI metrics are required, embed `segoeui.ttf`/`segoeuib.ttf` and pass their bytes in `spec.fonts`, then name the family. That adds Microsoft font licensing and payload size; not recommended.
- Text will be slightly wider/narrower than Tk’s Segoe UI. Preserve the layout metrics, not glyph-for-glyph metrics; screenshot comparisons should be structural.

### 4.3 Components with exact imports and signatures

```rust
use rune_kit::{
    Button, ButtonVariant, ControlSize, Dialog, Kit, Progress, SegmentedControl,
    TextArea, TextInput,
};
```

`Button` (from `src/actions.rs`):

```rust
Button::new("Clear all")
    .variant(ButtonVariant::GhostMuted)
    .control_size(ControlSize::Small)
    .disabled(s.busy)
    .on_press(|s: &mut AppState| s.clear_files())
    .build(&k);
```

`ButtonVariant` values: `Neutral`, `Danger`, `Warning`, `Outline`, `Contrast`, `Submit`, `Ghost`, `GhostMuted`, `GhostFaint`, `Loading`. `Button::on_press` is `Fn(&mut S)`; `on_press_tx` is `Fn(&mut S, &mut Tx)`.

`TextInput` (from `src/text_input.rs`):

```rust
TextInput::new(&s.rename_edit)
    .label("Transcript name")
    .placeholder("Name")
    .width(None)          // fill parent
    .height(36.0)
    .on_edit(|s: &mut AppState| &mut s.rename_edit)
    .on_submit(|s: &mut AppState, _tx: &mut Tx| s.commit_rename())
    .build(&k);
```

- `TextInput::new` takes `&TextEdit` and clones it; the editor itself stays in state.
- `.on_edit` is `impl for<'a> Fn(&'a mut S) -> &'a mut TextEdit + 'static`. Without it, typing has nowhere to land.
- `.on_submit` exists on `TextInput` and `TextArea`; `TextInput` is single-line.
- `.width(None)` fills the parent; default width is 280.
- `.height` includes padding; the field will not go below the style’s line height.
- `.invalid(bool)` changes text/placeholder/caret colour when `k.invalid_text` is set.

`TextArea`:

```rust
TextArea::new(&s.live_edit)
    .placeholder(LISTENING)
    .rows(6)
    .max_height(220.0)
    .on_edit(|s: &mut AppState| &mut s.live_edit)
    .build(&k);
```

`Dialog` (from `src/dialog.rs`):

```rust
Dialog::new("privacy", "Your privacy")
    .description("EasyScribe works fully offline.")
    .width(520.0)
    .fit()                       // size to body instead of 368 pt
    .click_outside(false)        // Privacy closes only via Close/Escape
    .open(s.privacy_open)
    .on_close(|s: &mut AppState| s.privacy_open = false)
    .child(privacy_body(s, &k))
    .build(&k, &mut layer);
```

**Critical:** `Dialog::build` returns nothing unless `.on_close(...)` was called. It sets both the close button and the modal dismiss policy. A dialog without `on_close` is silently not built.

`Progress`:

```rust
Progress::new(Some(s.progress))   // 0..=1; None = indeterminate
    .label("Transcribing")
    .show_value()
    .build(&k);
```

`SegmentedControl`:

```rust
SegmentedControl::new(&["Transcribe files", "Record"], Some(s.mode.index()))
    .label("Mode")
    .fit()
    .on_pick(|s: &mut AppState, index: usize, _tx: &mut Tx| s.switch_mode(index))
    .build(&k);
```

It uses one accent (`k.c.bg_accent`), so it cannot show teal for Files and coral for Record without a custom element. The existing local segmented switch is the correct parity choice.

### 4.4 Text input rules that prevent bugs

1. **Store `TextEdit` in `AppState`.** Do not call `TextEdit::new` inside `view`.
2. **Bind with `.on_edit`.** Read `.text` for the value. Use `.replace_text(...)` for programmatic changes (rename suggestion, clearing after submit); direct assignment to `.text` bypasses secure wiping and mark invariants.
3. **Focus once.** Set `edit.focused = true` when opening the rename/naming panel; the runtime clears it when focus moves. Do not force it every frame.
4. **Submit/cancel.** `Enter` triggers `.on_submit`; `Escape` should be handled by the overlay/`on_key` and restore the original value.
5. **Undo history.** `TextEdit::clear()` clears text and history while keeping focus; use after a commit if the field stays open.
6. **Secure fields.** Not used by EasyScribe; if added, `TextInput::secure()` and `tx.copy_secret` are available.

### 4.5 Speaker naming panel and Privacy: modal vs window

Current behavior:

- **Speaker naming** is an in-window panel/card, not an OS dialog. It blocks completion of the job while names are entered. Implement it as an in-window overlay (`El` absolute layer) or a `Dialog` with a custom body. If it must block background interaction, call `.modal(true, …)` on the custom overlay `El`; `Dialog` applies its own modal policy through `.on_close(...)`. Otherwise keep the current modeless panel look.
- **Privacy** is a separate non-modal Toplevel, transient to the main window, with its own title bar, icon and Escape close. It does not change app state.

Rune options:

| Option | Behavior | Recommendation |
|---|---|---|
| `rune_kit::Dialog` | In-window modal; traps Tab, handles Escape, background inert | Simplest, accessible, headless-testable; changes “non-modal” behavior |
| `App::window(WindowSpec)` | Real second native window; independent state; closes by hiding | Closest to Toplevel; no parent/transient API; needs a second `AppSpec`/state and icon |
| Custom absolute overlay | Non-modal in-window panel; Escape handled by `on_key` | Closest visual parity without a second window; must implement focus trapping yourself if desired |

For v1, use `Dialog` for Privacy and speaker naming. It is the lowest-risk, testable path. If the product owner insists on a separate Toplevel, use `App::window`; do not use a second `rune::run`.

Secondary window sketch:

```rust
let privacy_spec = AppSpec::new(
    "Your privacy",
    (520.0, 560.0),
    PrivacyState::new,
    privacy_view,
);
let hidden = rune::WindowOptions { visible: false, resizable: false, ..Default::default() };
let (privacy, _window) = app.window(
    rune::WindowSpec::new("privacy", privacy_spec).options(hidden),
);
// Later, from a button:
privacy.open();
```

`App::window` must be called before `app.run()`. Store the returned `AppHandle<PrivacyState>` in `AppState`; `open()` shows and focuses, `close()` hides. There is no “transient to parent” or “centre over parent” API; compute a position or accept the OS default.

---

## 5. Input, files and Windows integration

Rune covers the interaction surface the Python app built from Tk/TkDND/CTk: pointer input, text fields and IME, file drops, native file dialogs, clipboard, modal layers, window creation and keyboard routing. Three things are not portable and need native code: the **window icon** (`rune::set_app_icon` is macOS-only), **opening a file or folder in Explorer** (`Tx::open_url` deliberately accepts only `http`, `https` and `mailto`), and **multi-file selection** (Rune's dialog API is single-select). Those are small, isolated Win32/rfd additions; everything else stays in Rune.

### 5.1 Drag and drop

`El::on_drop_files` receives every accepted path in one call. It is the direct replacement for the TkDND `<<Drop>>` handler, including the “drop anywhere on the card” behaviour:

```rust
use rune::{el, Cx, El, Tx};

fn files_view(s: &AppState, _cx: &mut Cx<AppState>) -> El {
    el()
        .key("files.drop")
        .w_full()
        .h_full()
        .on_drop_files(|s: &mut AppState, files, _tx| {
            if s.is_idle() {
                s.add_paths(files);
            }
        })
        .accepts_extensions(&[
            "mp4", "mkv", "mov", "avi", "webm",
            "mp3", "wav", "m4a", "flac", "ogg", "opus", "aac",
        ])
        .on_drop_rejected(|s: &mut AppState, files, _tx| {
            s.drop_note = Some(s.skipped_note(files));
        })
        .on_drop_hover(|s: &mut AppState, over, _tx| {
            s.drop_hover = over && s.is_idle();
        })
}
```

Facts that matter for parity:

- `accepts_extensions` strips a leading `.` and compares extensions case-insensitively (`JPG` and `.jpg` are the same). `accepts_files` takes a predicate when the rule is not a suffix.
- The target is the **innermost** `on_drop_files` element under the pointer that accepts any of the dragged files. It receives the files it accepts; files it rejects go to `on_drop_rejected` if that is set. Put a target on the root as well if a drop outside the card should still work.
- A drop target takes no pointer events, so an overlay target does not steal clicks from the elements below it.
- While a modal is open, its root stands in for the window root; drops outside the modal reach nothing. This is the correct behaviour for the speaker-naming and Privacy panels.
- `on_drop_hover(true/false)` drives the teal border/background highlight. With `on_drop_move`, `tx.drop_point()` gives the position in the element's own points for a drop caret; not needed for a whole-card target.
- Windows Explorer supplies directories as paths too. Reject them with `path.is_file()`; do not recurse. This matches the current Python handler, which reports skipped entries by name.
- Headless tests use `Runtime::drag_files` / `drop_files` (§6); the same callbacks run.

Deduplicate on Windows by the path's case-insensitive string form, not by `canonicalize` (which produces `\\?\` paths that look wrong in the list). Keep the user's original path for display and for `ffmpeg`, and use the original path as the batch key.

### 5.2 Native file dialogs

Rune's dialog API is asynchronous and headless-scriptable. `FileDialogOptions` is a plain struct; the builder only appends filters:

```rust
use rune::{FileDialogOptions, Tx};

const MEDIA: &[&str] = &[
    "mp4", "mkv", "mov", "avi", "webm",
    "mp3", "wav", "m4a", "flac", "ogg", "opus", "aac",
];

fn choose_one(tx: &mut Tx) {
    let options = FileDialogOptions {
        title: "Add audio or video".into(),
        directory: None,
        file_name: None,
        filters: Vec::new(),
    }
    .filter("Audio and video", MEDIA)
    .filter("All files", &["*"]);

    tx.open_file(options, |s: &mut AppState, result, _tx| match result {
        Ok(Some(path)) => s.add_paths(&[path]),
        Ok(None) => {} // cancel
        Err(error) => s.error = Some(error.to_string()),
    });
}
```

- `Tx::save_file` uses the same options and returns a path that may not exist yet; set `file_name` to the suggested transcript name, e.g. `"meeting (speakers).txt"`.
- `Tx::open_folder` picks an existing folder; `filters` and `file_name` do not apply. Use it for the recordings/output folder.
- The dialog is shown after the handler's frame finishes. The callback runs as an action on a later frame; it is not called inline. Do not hold a borrow of state across it.
- One dialog at a time. A second request while one is open is answered with `FileDialogError("a file dialog is already open")`; double-clicks must be guarded in the UI as well.
- Native dialogs need the `native-file-dialog` feature on the `rune` dependency. It is already enabled in the workspace. Headless runs keep the request open as a `fileDialog` object in `--tree` and answer it with a script's `file-dialog` / `file-dialog-cancel` event.
- Never call `pollster::block_on` on the UI thread to await a Rune dialog: the macOS implementation presents a sheet from the main thread, and blocking it deadlocks; the Windows implementation is owned by the window. Use the callback API.

#### Multi-file selection

Rune has no multi-select open dialog (`FileDialogOptions` has no `multiple` field; `FileDialogs::open` returns one `PathBuf`). The current app must add files in batches, so use `rfd::FileDialog::pick_files()` on a short-lived worker thread:

```rust
use std::path::PathBuf;
use std::sync::mpsc;
use rune::Waker;

pub enum DialogMsg {
    Picked(Vec<PathBuf>),
    Cancelled,
}

fn pick_many(tx: mpsc::Sender<DialogMsg>, wake: Waker) {
    std::thread::spawn(move || {
        let picked = rfd::FileDialog::new()
            .set_title("Add audio or video")
            .add_filter("Audio and video", MEDIA)
            .add_filter("All files", &["*"])
            .pick_files();

        let message = match picked {
            Some(paths) => DialogMsg::Picked(paths),
            None => DialogMsg::Cancelled,
        };
        let _ = tx.send(message);
        wake();
    });
}
```

- `pick_files()` is blocking and returns `None` for both cancellation and a dialog that could not open; rfd 0.17 does not expose the error separately. Treat `None` as cancel and leave the existing list unchanged.
- Create the dialog **inside** the worker. `rfd` calls `CoInitializeEx(COINIT_APARTMENTTHREADED)` per pick on Windows; a long-lived thread that already initialized COM as MTA would make every later call fail with `RPC_E_CHANGED_MODE`. A fresh short-lived thread avoids that.
- Do not pass a Rune `WindowHandle` to `set_parent`: Rune's handle wrappers do not implement `HasWindowHandle`/`HasDisplayHandle`. Omitting the parent gives an unowned dialog, not an OS-modal one; guard with a `dialog_open: bool` in state so the main window cannot start a second dialog. If true ownership is required later, wrap the raw `HWND` in a tiny type implementing both traits and pass it to `set_parent`.
- On Windows, `pick_files` uses `IFileOpenDialog` with `FOS_ALLOWMULTISELECT`; the returned paths are ordinary `PathBuf`s. Test on Windows 10 and 11 because owner/COM behaviour is the part that cannot be verified from the Linux checkout.
- Feed the result into the same `add_paths` path as a drop, so extension filtering, deduplication and the skipped-file note are identical.

### 5.3 Clipboard

Rune owns the clipboard worker and the platform formats. Use `Tx`, never a direct `arboard` dependency:

```rust
// Copy the finished transcript.
button("Copy").on_press_tx(|s: &mut AppState, tx: &mut Tx| {
    tx.copy(s.transcript.clone());
});

// Read without pasting (for an “Import from clipboard” action).
tx.read_clipboard(|s: &mut AppState, content, _tx| {
    s.import_text(content.text());
});
```

- `Tx::copy(text)` requests a plain-text write after the handler. `Tx::copy_rich(...)` can add HTML/RTF; not needed for a transcript.
- `Tx::read_clipboard` reads asynchronously on the clipboard worker and calls the closure as an update on a later frame. It always answers, with empty content when the clipboard holds nothing readable.
- `Tx::request_paste()` and root `.on_paste(...)` handle Ctrl+V when no field is focused. A focused `TextEdit` takes precedence.
- Headless runs never write the system clipboard. For tests, install `rune::clipboard::memory()` with `Runtime::use_clipboard(&clipboard)` and inspect `clipboard.text()` / `content()`.
- Do not copy transcript text into the log or into a diagnostics string; the clipboard is the only place it should go.

### 5.4 Window icon, AppUserModelID and manifest

`rune::set_app_icon` is a macOS-only API: on Windows it compiles to a no-op. The executable must carry the icon as a PE resource, and the window must receive it after native creation.

**Resource file** (`assets/easyscribe.rc`):

```rc
#define RT_MANIFEST 24

1 ICON "EasyScribe.ico"
1 VERSIONINFO
FILEVERSION 3,2,3,0
PRODUCTVERSION 3,2,3,0
FILEOS 0x40004L
FILETYPE 0x1L
BEGIN
  BLOCK "StringFileInfo"
  BEGIN
    BLOCK "040904B0"
    BEGIN
      VALUE "CompanyName", "EasyScribe"
      VALUE "FileDescription", "EasyScribe"
      VALUE "FileVersion", "3.2.3"
      VALUE "InternalName", "EasyScribe"
      VALUE "OriginalFilename", "EasyScribe.exe"
      VALUE "ProductName", "EasyScribe"
      VALUE "ProductVersion", "3.2.3"
    END
  END
  BLOCK "VarFileInfo"
  BEGIN
    VALUE "Translation", 0x0409, 0x04B0
  END
END

1 RT_MANIFEST "easyscribe.manifest"
```

`build.rs`:

```rust
fn main() {
    #[cfg(windows)]
    embed_resource::compile("assets/easyscribe.rc", embed_resource::NONE)
        .manifest_required()
        .expect("compile easyscribe.rc");
}
```

`Cargo.toml`:

```toml
[target.'cfg(windows)'.build-dependencies]
embed-resource = "3.0.12"
```

**Icon at runtime.** `App::on_window` runs on the UI thread after every native creation (including recreation after a GPU release), and `WindowHandle::with_raw_handle` borrows the native handle for the callback:

```rust
#[cfg(windows)]
fn set_window_icon(window: &rune::WindowHandle) {
    use windows_sys::Win32::Foundation::{HWND, LPARAM, WPARAM};
    use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        GetSystemMetrics, LoadImageW, SendMessageW, ICON_BIG, ICON_SMALL,
        IMAGE_ICON, LR_DEFAULTCOLOR, SM_CXICON, SM_CXSMICON, SM_CYICON,
        SM_CYSMICON, WM_SETICON,
    };

    let _ = window.with_raw_handle(|handle| {
        let raw = handle.as_raw();
        let rune::raw_window_handle::RawWindowHandle::Win32(win32) = raw else {
            return;
        };
        let hwnd = win32.hwnd.get() as HWND;
        unsafe {
            let module = GetModuleHandleW(std::ptr::null());
            let big = LoadImageW(
                module,
                1usize as *const u16, // resource id 1
                IMAGE_ICON,
                GetSystemMetrics(SM_CXICON),
                GetSystemMetrics(SM_CYICON),
                LR_DEFAULTCOLOR,
            );
            let small = LoadImageW(
                module,
                1usize as *const u16,
                IMAGE_ICON,
                GetSystemMetrics(SM_CXSMICON),
                GetSystemMetrics(SM_CYSMICON),
                LR_DEFAULTCOLOR,
            );
            if !big.is_null() {
                SendMessageW(hwnd, WM_SETICON, ICON_BIG as WPARAM, big as isize as LPARAM);
            }
            if !small.is_null() {
                SendMessageW(hwnd, WM_SETICON, ICON_SMALL as WPARAM, small as isize as LPARAM);
            }
        }
    });
}
```

Wire it with:

```rust
let app = rune::App::new(spec)
    .on_window(|window| {
        #[cfg(windows)]
        set_window_icon(window);
    });
```

Do not destroy the `HICON`s while the window lives; the process-lifetime resource is fine. Do not store the borrowed `HWND` in app state: `with_raw_handle` is callback-scoped and the window can be recreated.

**AppUserModelID.** Set it once at the top of `main`, before Rune creates a window:

```rust
#[cfg(windows)]
fn set_app_user_model_id() {
    use windows_sys::Win32::UI::Shell::SetCurrentProcessExplicitAppUserModelID;
    let id: Vec<u16> = "EasyScribe.App".encode_utf16().chain(Some(0)).collect();
    unsafe {
        let _ = SetCurrentProcessExplicitAppUserModelID(id.as_ptr());
    }
}
```

The string must match the shortcut's `System.AppUserModel.ID` if one is ever created; for v1 no shortcut is installed, so the constant only groups taskbar windows and pins. It must be set before the first `HWND` is created.

**Manifest** (`assets/easyscribe.manifest`):

```xml
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
  <assemblyIdentity version="3.2.3.0" processorArchitecture="amd64"
                    name="EasyScribe" type="win32"/>
  <description>EasyScribe</description>
  <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">
    <security>
      <requestedPrivileges>
        <requestedExecutionLevel level="asInvoker" uiAccess="false"/>
      </requestedPrivileges>
    </security>
  </trustInfo>
  <application xmlns="urn:schemas-microsoft-com:asm.v3">
    <windowsSettings>
      <dpiAware xmlns="http://schemas.microsoft.com/SMI/2005/WindowsSettings">true/pm</dpiAware>
      <dpiAwareness xmlns="http://schemas.microsoft.com/SMI/2016/WindowsSettings">PerMonitorV2</dpiAwareness>
      <longPathAware xmlns="http://schemas.microsoft.com/SMI/2016/WindowsSettings">true</longPathAware>
    </windowsSettings>
  </application>
  <compatibility xmlns="urn:schemas-microsoft-com:compatibility.v1">
    <application>
      <supportedOS Id="{8e0f7a12-bfb3-4fe8-b9a5-48fd50a15a9a}"/>
    </application>
  </compatibility>
</assembly>
```

`asInvoker` is required: the app writes only to `%LOCALAPPDATA%` and per-user documents, never Program Files.

### 5.5 Opening files and folders in Explorer

`Tx::open_url` intentionally rejects `file:` and custom schemes (`crates/rune/src/open_url.rs`). Use `ShellExecuteW` for “Open recording”, “Open transcript”, “Open logs folder” and “Show in folder”:

```rust
#[cfg(windows)]
fn shell_open(path: &std::path::Path) -> Result<(), String> {
    use windows_sys::Win32::UI::Shell::ShellExecuteW;
    use windows_sys::Win32::UI::WindowsAndMessaging::SW_SHOWNORMAL;

    fn wide(text: &str) -> Vec<u16> {
        text.encode_utf16().chain(Some(0)).collect()
    }

    let operation = wide("open");
    let file = wide(&path.to_string_lossy());
    let result = unsafe {
        ShellExecuteW(
            std::ptr::null_mut(),
            operation.as_ptr(),
            file.as_ptr(),
            std::ptr::null(),
            std::ptr::null(),
            SW_SHOWNORMAL,
        )
    };
    if result as isize <= 32 {
        Err(format!("Windows could not open {} (ShellExecuteW returned {})",
                    path.display(), result as isize))
    } else {
        Ok(())
    }
}

#[cfg(windows)]
fn reveal_in_explorer(path: &std::path::Path) -> Result<(), String> {
    use windows_sys::Win32::UI::Shell::ShellExecuteW;
    use windows_sys::Win32::UI::WindowsAndMessaging::SW_SHOWNORMAL;

    fn wide(text: &str) -> Vec<u16> {
        text.encode_utf16().chain(Some(0)).collect()
    }
    let operation = wide("open");
    let explorer = wide("explorer.exe");
    let arguments = wide(&format!("/select,\"{}\"", path.display()));
    let result = unsafe {
        ShellExecuteW(
            std::ptr::null_mut(),
            operation.as_ptr(),
            explorer.as_ptr(),
            arguments.as_ptr(),
            std::ptr::null(),
            SW_SHOWNORMAL,
        )
    };
    if result as isize <= 32 {
        Err(format!("Windows could not reveal {} (ShellExecuteW returned {})",
                    path.display(), result as isize))
    } else {
        Ok(())
    }
}
```

- `ShellExecuteW` returns a value greater than 32 on success; 0/`SE_ERR_*` values are failures. Do not interpret a null `HINSTANCE` as success.
- Opening a folder with `ShellExecuteW("open", folder, …)` opens Explorer at that folder; `explorer.exe /select,<path>` selects a file.
- `explorer.exe` parses `/select,` itself; wrap the path in double quotes. Windows paths cannot contain `"`, so no escaping bug is possible. If the path is a directory, use the plain open form instead.
- Do not build `cmd.exe /c start ...`: quoting and `&`/`^` injection are avoidable, and a console window flashes.
- These calls are fast enough for the UI thread, but file/folder opening on a corrupt shell extension can block; run them on a worker and report failure through the same channel if measured slow.
- This is also the correct implementation for the “Open logs folder” action in Privacy; do not use `Tx::open_url("file:///...")`, which returns `UnsupportedScheme`.

### 5.6 Shortcuts

In-app shortcuts use `AppSpec::on_key` (or `El::on_key` for a focused subtree). The signature is a plain `fn` pointer, so it cannot capture state; dispatch through methods on `AppState`:

```rust
use rune::{Key, KeyEvent, Tx};

fn on_key(s: &mut AppState, event: &KeyEvent, tx: &mut Tx) -> bool {
    if event.is_command('o') {
        s.choose_files(tx);
        return true;
    }
    if event.is_command('f') && event.mods.shift {
        s.focus_search(tx);
        return true;
    }
    match &event.key {
        Key::Escape => {
            s.escape(tx); // close modal, cancel rename, or clear selection
            true
        }
        Key::Enter if !tx.key_repeat() => {
            s.primary_action(tx);
            true
        }
        _ => false,
    }
}

let spec = rune::AppSpec {
    on_key: Some(on_key),
    // `AppSpec::init` is a fn pointer; use `App::init` when the state needs
    // captured values such as `Paths` or an `AppHandle`.
    ..rune::AppSpec::new("EasyScribe", (1100.0, 760.0), placeholder_state, view)
};
```

- `KeyEvent::is_command(c)` is the portable accelerator check: it matches `Key::Command(c)` (Cmd on macOS, Ctrl on Windows/Linux) and the special Cmd-Space `Key::Text(" ")` form. Check `event.mods.shift` separately for Ctrl+Shift chords.
- `tx.physical_key()` gives the layout-independent key (`PhysicalKeyCode::Digit7`), for shifted punctuation shortcuts that move between layouts. `tx.key_base()` gives the unmodified character on macOS/Windows/Linux.
- `tx.key_repeat()` is true for OS auto-repeat. Use it to ignore repeats for one-shot actions (open, submit, start recording), but keep repeats for arrows and navigation.
- Key routing sends keys to the focused control first; `on_key` only sees what the control and focus traversal leave. A focused `TextEdit` therefore keeps typing, while Ctrl+O still reaches the window handler because the field does not claim it.
- `Dialog` and other rune-kit modal components already handle Escape and Tab focus trapping. Do not add a second Escape handler inside the dialog body.
- `App::global_shortcut` registers an **OS-wide** hotkey that works while EasyScribe is in the background. The current app has none; do not use it for Ctrl+O. If a future “start recording” global hotkey is added, register it with `App::global_shortcut("Ctrl+Alt+R", handler)` and show the OS conflict error; it never reaches `on_key`.
- The mirror of `on_key` is `on_key_up`; not needed here. A modifier-only hold-to-talk feature would require `App::modifier_keys()`.

### 5.7 Close behaviour and process lifetime

Rune cannot cancel a window close. `on_close` runs before the window closes, then the process exits; there is no “return false to keep the window open” and no second callback. This is different from Tk's `WM_DELETE_WINDOW` protocol handler. Design for it:

```rust
let app = rune::App::new(spec)
    .on_close(|s: &mut AppState, _tx: &mut rune::Tx| {
        s.shutdown(); // signal workers; kill children; flush WAV/logs
    });
```

- `shutdown()` must be idempotent. It sets the global `CancelToken`, kills `whisper-cli`, `ffmpeg`, and the diarization child, stops the cpal stream, finalizes the WAV, flushes the log writer, and removes temp files.
- Do not show a confirmation dialog from `on_close`: the window is already going away, and the callback has no way to veto. If a “record in progress” confirmation is a product requirement, it needs a Win32 `WM_CLOSE` subclass on the raw `HWND`; that is out of scope for v1 and risks fighting Rune's lifecycle. Document the behaviour change instead.
- `Tx::quit()` skips `on_close` and exits; `Tx::close_window()` runs it. Use `close_window()` for the title-bar-equivalent action; never call `std::process::exit` from a handler, because it skips destructors and leaves children.
- On Windows, assign every child process to a job object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` (§10.7.6). Then even a crash or a `TerminateProcess` of the GUI kills ffmpeg/whisper/diarization; `on_close` is a graceful path, not the only safety net.
- `App::keep_running()` is not needed. A single-window app that exits on close is the current product. If a tray or background recording mode is added later, use `AppHandle` and `App::windowless`; the state and GPU can outlive the window.

### 5.8 Windows file-system details

- **DPI:** the manifest's PerMonitorV2 setting plus Rune's `Env::scale` handling covers mixed-DPI monitors. Do not set process DPI awareness with `SetProcessDpiAwarenessContext` as well; the manifest is enough and avoids a race with winit.
- **Long paths:** `longPathAware` in the manifest lets the Unicode APIs accept paths over `MAX_PATH` when the machine policy allows it. The cache path is short (`%LOCALAPPDATA%\EasyScribe\runtime\v1-<32 hex>`), but user media paths can be long; pass them as `Path`/`OsStr`, never through a fixed `MAX_PATH` buffer.
- **Reserved names:** payload entries must reject `CON`, `NUL`, `COM1`…`COM9`, `LPT1`…`LPT9`, trailing dots/spaces, `:`, and case-insensitive duplicates. The same validation applies to transcript filenames derived from a recording name; Windows silently strips trailing dots and refuses reserved stems.
- **Read-only exe:** never write beside the executable. The payload cache, logs, temp audio and settings live under `%LOCALAPPDATA%\EasyScribe` (or the portable data directory selected by `EasyScribe.portable`).
- **Antivirus:** a 1.5 GB unsigned exe with an appended archive is a heuristic magnet. Code signing and a stable, versioned payload hash are part of the design, not an afterthought (§10.7.10).

---

## 6. Headless tests and screenshots

Rune's headless mode is the same renderer, layout engine and input router with a virtual clock and no window. It is the only practical way to test this UI on Linux CI and the right way to catch layout regressions before a Windows VM run.

### 6.1 CLI surface

The binary's normal Rune flags are enough; EasyScribe does not need a custom test runner:

```sh
S=target/release/easyscribe

# One deterministic frame.
$S --snap out/files.png --scene files --size 1100x760 --scale 1

# A named time after a scripted interaction.
$S --script tests/ui/drop.json --snap out/after-drop.png --at mark:after-drop

# The element tree at the same time.
$S --tree out/files.json --scene files --at 0.0

# A strip of frames for a motion review.
$S --script tests/ui/record.json --frames 90 --every 6 --cols 3 \
   --strip out/record-strip.png

# A video, if ffmpeg is on PATH.
$S --script tests/ui/demo.json --video out/demo.mp4 --fps 60

# List scenes and windows.
$S --list-scenes
$S --list-windows
```

- `--snap` renders offscreen at `--at` seconds (default: end of the script/frames). It needs a GPU adapter; on Linux CI use Mesa's lavapipe (`VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json`) or `LIBGL_ALWAYS_SOFTWARE=1` for the GL path. No display or XKB library is needed for `--snap`; windowed runs do need `libxkbcommon-x11-0`.
- `--tree` writes the element tree at `--at`: key, path, role, label, text, presented rect, target rect, opacity, fill/text colour, hover/press, moving, focus, modal and dialog state. A `fileDialog` object appears while a dialog is open; a `popupMenu`, `tooltip`, `video`, `accessibility` and `windows` object can also appear.
- `--script` replays timestamped `move`, `down`, `up`, `tap`, `double-click`, `drag`, `type`, `key`, `key-up`, `paste`, `file-dialog`, `file-dialog-cancel`, `drag_files`, `drop_files`, `menu-pick`, `menu-cancel`, `resize`, `wheel`, `magnify`, `rotate`, `smart-magnify`, `shortcut` and `mark` events. A `mark` names a time for `--at mark:name` / `--start mark:name-0.2`.
- `--frames N` alone is headless; `--native` with `--trace` is the only windowed tracing mode. `--out DIR` writes PNG frames; `--video` needs `ffmpeg` on PATH, so CI must either set PATH to the bundled ffmpeg or install one.
- `--scene NAME` applies one of `AppSpec::scenes` before the first frame. Keep scenes cheap and deterministic: set state, do not start threads, do not read the clock or the filesystem.
- Headless runs never call `on_ready`; start work in `AppSpec::setup`, `poll`, or a `Waker`-driven channel instead. This is also true for the first-run extraction: a headless smoke test must use `EASYSCRIBE_PAYLOAD_DIR` and must never extract 1.8 GB.

### 6.2 Driving a `Runtime` directly

For unit tests, build the same `Runtime` the window would use and feed events. `Runtime::new` takes a `fn` view, not a closure; `Env::new(system_dark, scale)`:

```rust
use rune::runtime::{Env, Runtime};
use rune::{Key, KeyEvent, Mods, TextStyle};

fn harness(state: AppState) -> Runtime<AppState> {
    let mut runtime = Runtime::new(
        state,
        view,
        (1100.0, 760.0),
        Env::new(false, 1.0), // light theme, 1x
    );
    runtime.frame(0.0); // first commit
    runtime
}

fn click(runtime: &mut Runtime<AppState>, key: &str) {
    let (x, y) = runtime.find(key).unwrap_or_else(|| panic!("no element {key}"));
    runtime.pointer_move(x, y);
    runtime.pointer_down(x, y);
    runtime.pointer_up(x, y);
    runtime.frame(1.0 / 60.0);
}

fn type_text(runtime: &mut Runtime<AppState>, text: &str) {
    for character in text.chars() {
        runtime.key(KeyEvent {
            key: Key::Text(character.to_string()),
            mods: Mods::default(),
        });
        runtime.frame(1.0 / 60.0);
    }
}
```

The public input surface used by tests:

| Call | Effect |
|---|---|
| `runtime.frame(dt)` | Advance animation time, run queued handlers, rebuild/lay out; returns whether state changed |
| `runtime.update(\|s, tx\| …)` | Change state on the UI thread; commits next frame. Unlike `AppHandle::update`, this call has no `Send` bound because it runs inline |
| `runtime.find("key")` | Centre of the first element with that key (or path suffix); `None` if absent |
| `runtime.presented("key")` | `([x, y, w, h], opacity)` as last drawn |
| `runtime.elements()` | `Vec<ElementInfo>` for the whole tree; use for assertions on many rows |
| `runtime.pointer_move/down/up(x, y)` | Primary pointer; `pointer` reads the current position |
| `runtime.key(KeyEvent { … })` / `runtime.paste(text)` | Keyboard/text input; queued until the next frame |
| `runtime.drag_files(paths, Some((x, y)))` | Start an external file drag over the window |
| `runtime.drop_files(paths, Some((x, y)))` | Drop the files at a point |
| `runtime.file_dialog()` | `Option<(FileDialogKind, &FileDialogOptions)>` while a dialog is open |
| `runtime.resolve_file_dialog(Some(path))` | Answer the open dialog as the user; callback runs next frame |
| `runtime.use_clipboard(&memory)` | Route clipboard to an in-memory board for tests |

A complete dialog test:

```rust
#[test]
fn add_files_dialog_adds_a_path() {
    let mut runtime = harness(AppState::idle());
    click(&mut runtime, "files.add");

    let (kind, options) = runtime.file_dialog().expect("dialog");
    assert_eq!(kind, rune::FileDialogKind::Open);
    assert_eq!(options.title, "Add audio or video");

    runtime
        .resolve_file_dialog(Some(std::path::PathBuf::from("/media/talk.mp3")))
        .unwrap();
    runtime.frame(0.0);

    assert_eq!(runtime.state.files.len(), 1);
    assert_eq!(runtime.state.files[0].path, std::path::PathBuf::from("/media/talk.mp3"));
}
```

A drop test:

```rust
#[test]
fn dropping_an_mp3_adds_it_and_rejects_a_txt() {
    let mut runtime = harness(AppState::idle());
    let mp3 = std::path::PathBuf::from("/media/talk.mp3");
    let txt = std::path::PathBuf::from("/media/notes.txt");

    runtime.drag_files(vec![mp3.clone(), txt.clone()], Some((400.0, 300.0)));
    runtime.frame(0.0);
    assert!(runtime.state.drop_hover);

    runtime.drop_files(vec![mp3, txt], Some((400.0, 300.0)));
    runtime.frame(0.0);
    assert_eq!(runtime.state.files.len(), 1);
    assert!(runtime.state.drop_note.as_deref().unwrap().contains("notes.txt"));
}
```

Text entry has one important rule: keep the `TextEdit` in app state, bind it with `.on_edit(|s| &mut s.field)`, and set `focused = true` once to request focus. Do not construct a new `TextEdit` on each view rebuild:

```rust
fn rename_view(s: &AppState, _cx: &mut Cx<AppState>) -> El {
    el().child(
        rune::text_input(&s.rename, TextStyle::new(14.0, 400.0))
            .key("rename.input")
            .on_edit(|s: &mut AppState| &mut s.rename)
            .on_submit(|s: &mut AppState, _tx| s.commit_rename()),
    )
}
```

To focus it in a test, set `runtime.state.rename.focused = true` before the first frame; the runtime maintains the flag from then on.

### 6.3 Scenes to freeze

Make each scene a small `fn(&mut AppState)` and register it in `AppSpec::scenes`:

| Scene | State | What the PNG catches |
|---|---|---|
| `files` | three queued files, one failed, timestamps on | Card spacing, list rows, progress bars, disabled states |
| `record` | idle record tab, default mic, record-only | Mode switch, timer at 00:00, level meter at rest |
| `live` | recording, two live lines, timer 00:42, level 0.6 | Live transcript growth, timer, meter, Stop button |
| `result` | finished transcript, copy/save/name controls | Result card, filename field, action row |
| `speakers` | naming panel, two speakers, clip buttons, shares | Modal scrim, clip playback row, Skip/confirm |
| `privacy` | Privacy dialog open | Text wrapping, paths, close button |
| `first-run` | extracting at 37% | Getting-ready progress, no main controls |
| `error` | failed file with long path | Error text truncation, retry/remove buttons |

Screenshots are compared with a small per-pixel tolerance because GPU rasterisation differs by driver. Store the golden PNGs at `tests/ui/golden/<scene>.png`, render at `--scale 1` for stable text metrics, and compare only the content area (crop known chrome if the platform adds any). A structural `--tree` assertion is usually more valuable than a pixel diff: assert the keys, labels and row count, not exact coordinates.

### 6.4 What headless cannot prove

- Native dialog presentation, IME composition windows, drag images, taskbar icon/AUMID grouping, DPI transitions, clipboard formats outside Rune's memory backend, audio device capture, GPU Vulkan availability, and Authenticode verification.
- `on_ready`, frame presentation timing, and window creation/recreation. `on_ready` is desktop-only by design.
- `--snap` uses the same renderer but not the same swapchain; it does not catch a Windows-only wgpu/DX12 failure. Keep the Windows smoke build (§11) as a separate gate.

---

## 7. Native crates: what Rune has, what to add

Rune is a UI framework, not an audio or Windows-shell abstraction. The rewrite should add the smallest set of mature crates for the gaps, and keep every Rune-owned capability in Rune. The table is the dependency decision; the subsections give the exact API surface and the traps.

### 7.1 Dependency decision table

| Capability | Rune built-in? | Decision | Crate / version | Notes |
|---|---|---|---|---|
| Window, view, layout, animation | Yes | Use Rune | `rune` @ `781e2b6` | One renderer, one input router |
| Text input, IME, selection, undo | Yes | Use Rune | `rune` / `rune-kit` | `TextEdit` in state; `text_input` |
| Drag/drop from Explorer | Yes (Windows) | Use Rune | `rune::El::on_drop_files` | No `drag` crate |
| Single-file open/save/folder dialog | Yes (Windows/macOS) | Use Rune | `rune::Tx::open_file/save_file/open_folder` | Requires `native-file-dialog` |
| Multi-file picker | No | Add, scoped | `rfd` 0.17.2 | Worker thread; `pick_files` |
| Pre-GUI fatal error | No | Add, scoped | `rfd` 0.17.2 | `MessageDialog` before Rune starts |
| Clipboard | Yes | Use Rune | `Tx::copy`, `Tx::read_clipboard` | No `arboard` |
| In-app shortcuts | Yes | Use Rune | `AppSpec::on_key` | No global hotkey |
| Global hotkey (future) | Yes | Use Rune | `App::global_shortcut` | OS-wide; not needed |
| Window icon on Windows | No (macOS only) | Add Win32 | `windows-sys` 0.61.2 + `embed-resource` 3.0.12 | `WM_SETICON`; PE resource |
| AUMID / taskbar grouping | No | Add Win32 | `windows-sys` 0.61.2 | Before first window |
| Open file/folder in Explorer | No (`Tx::open_url` refuses `file:`) | Add Win32 | `windows-sys` 0.61.2 | `ShellExecuteW` |
| Microphone capture | No (macOS only) | Add | `cpal` 0.18.2 | WASAPI on Windows |
| WAV read/write | No | Add | `hound` 3.5.1 | Already used by core |
| Speaker clip playback | No (macOS only) | Add | `rodio` 0.23.0 (`playback`, `wav`) | No `symphonia` directly |
| File watching | No | Avoid for v1 | `notify` 8.2.1 (optional) | Not needed for parity |
| Structured logging | No | Add | `tracing` 0.1.44, `tracing-appender` 0.2.4, `tracing-subscriber` 0.3.22 | Keep `log` bridge |
| JSON | No | Add | `serde_json` 1.0.149+ | Manifest, settings, sidecars |
| Bounded live queues | No | Add | `crossbeam-channel` 0.5.15 | Decode queue must evict the oldest; `std::sync::mpsc::SyncSender` cannot receive |
| Job objects for child kill | No | Add Win32 | `windows-sys` 0.61.2 | `KILL_ON_JOB_CLOSE` |
| Disk-space check | No | Keep existing | `fs2` 0.4.3 | `available_space` (GetDiskFreeSpaceExW on Windows) |

Keep the existing `zip`, `sha2`, `crc32fast`, `fs2`, `dirs`, `uuid`, `chrono`, `regex`, `thiserror` and `serde` crates. `fs2` already gives a cross-platform advisory file lock, so the payload lock does not need hand-written `LockFileEx` code.

Several rows below are already present in the scaffold (`cpal`, `hound`, `rodio`, `rfd`, `serde_json`, `fs2`); “Add” means the rewrite relies on the crate for a missing capability, not necessarily that Cargo gains a new line. The genuinely new dependencies are `crossbeam-channel`, `tracing`/`tracing-appender`/`tracing-subscriber`, `windows-sys`, and `embed-resource`; `sherpa-onnx` replaces `sherpa-rs`.

### 7.2 `rfd` 0.17.2 — multi-select and pre-GUI dialogs

`rfd` is already a transitive dependency of Rune's native file dialogs at the same 0.17.2 version; adding it directly does not introduce a second GTK/X11 stack on Windows.

```toml
rfd = { version = "0.17.2", default-features = false }
```

The Windows backend is compiled unconditionally; the default features only pull the Linux XDG/Wayland backends. Disabling them keeps the Linux CI check from needing GTK/Wayland development packages. `common-controls-v6` is optional and only affects dialog styling; do not add it unless the visual difference matters.

**Multi-select** (the one capability Rune lacks):

```rust
let picked: Option<Vec<std::path::PathBuf>> = rfd::FileDialog::new()
    .set_title("Add audio or video")
    .add_filter("Audio and video", &[
        "mp4", "mkv", "mov", "avi", "webm",
        "mp3", "wav", "m4a", "flac", "ogg", "opus", "aac",
    ])
    .set_directory(last_folder)
    .pick_files();
```

- `pick_files()` blocks until the user answers. Call it only on a worker thread; never on Rune's UI thread. The UI stays responsive because Rune keeps drawing while the worker is in the dialog.
- `set_directory` accepts a path; `set_file_name` is for Save/pick-one, not multi-select.
- On Windows, rfd's backend calls `CoInitializeEx(COINIT_APARTMENTTHREADED)` around each pick and `CoUninitialize` after. A fresh short-lived worker is the safest COM apartment; a worker that has already initialized MTA will get `RPC_E_CHANGED_MODE` and the dialog will not show.
- `set_parent` requires `HasWindowHandle + HasDisplayHandle`. Rune's `WindowHandle` does not expose those traits, so either omit the parent (and guard against a second dialog) or wrap the raw `HWND` in a local type implementing both. Do not add `winit` just for the parent; `rune::raw_window_handle` re-exports the raw-handle types.
- `pick_files` returns `None` for cancel and for a failed dialog; rfd 0.17 does not separate the two. The UI should treat `None` as cancel. If the dialog cannot open at all, the user still has the drop target and the single-file Rune dialog.

**Pre-GUI fatal errors** (invalid footer, cache directory inaccessible, unsupported CPU):

```rust
use rfd::{MessageButtons, MessageDialog, MessageLevel};

fn fatal(title: &str, detail: &str) {
    let _ = MessageDialog::new()
        .set_level(MessageLevel::Error)
        .set_title(title)
        .set_description(detail)
        .set_buttons(MessageButtons::Ok)
        .show();
}
```

This is the only place a blocking native dialog is appropriate: Rune has not started, so there is no UI thread to block. After the Rune window exists, use a Rune modal, not `MessageDialog`, so the dialog matches the app and is testable.

### 7.3 `cpal` 0.18.2 — Windows microphone capture

```toml
cpal = "0.18.2"
crossbeam-channel = "0.5.15"   # bounded mic/decode queues; see §8.5
```

cpal's Windows backend is WASAPI. The default host is `cpal::default_host()`; the default input device may be `None` when no microphone is connected. The default config is the device mix format, which is often 48 kHz stereo even though the app wants 16 kHz mono. **Do not assume WASAPI will convert for you**; capture at the device's default rate/channels and resample on the worker.

The exact 0.18 API (the fourth `build_input_stream` argument is the initialization timeout, a 0.18 change):

```rust
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Sample, SampleFormat, SizedSample, StreamConfig};

fn open_input(
    device: &cpal::Device,
    tx: crossbeam_channel::Sender<Vec<f32>>,
) -> Result<cpal::Stream, String> {
    let supported = device.default_input_config().map_err(|e| e.to_string())?;
    let config: StreamConfig = supported.config();
    let format = supported.sample_format();
    let channels = config.channels as usize;

    let error = |err| eprintln!("microphone stream error: {err}");
    let stream = match format {
        SampleFormat::F32 => {
            let tx = tx.clone();
            device.build_input_stream::<f32, _, _>(
                config,
                move |data, _| send_mono(data, channels, &tx),
                error,
                None,
            )
        }
        SampleFormat::I16 => {
            let tx = tx.clone();
            device.build_input_stream::<i16, _, _>(
                config,
                move |data, _| send_mono(data, channels, &tx),
                error,
                None,
            )
        }
        SampleFormat::U16 => {
            let tx = tx.clone();
            device.build_input_stream::<u16, _, _>(
                config,
                move |data, _| send_mono(data, channels, &tx),
                error,
                None,
            )
        }
        other => return Err(format!("unsupported microphone sample format: {other}")),
    }
    .map_err(|e| e.to_string())?;

    stream.play().map_err(|e| e.to_string())?;
    Ok(stream)
}

fn send_mono<T: SizedSample + Sample>(
    data: &[T],
    channels: usize,
    tx: &crossbeam_channel::Sender<Vec<f32>>,
) {
    // `data` is interleaved. Average the channels; the queue is bounded and
    // `try_send` drops rather than blocking the audio thread.
    let mono: Vec<f32> = data
        .chunks(channels)
        .map(|frame| {
            frame.iter().map(|sample| sample.to_sample::<f32>()).sum::<f32>()
                / channels as f32
        })
        .collect();
    let _ = tx.try_send(mono);
}
```

Important details:

- `config.channels` is `u16`; `config.sample_rate` is `SampleRate(u32)`.
- `SampleFormat` is non-exhaustive in practice; handle `I8`, `I32`, `U8`, `U32`, `F64` or return a clear unsupported-format error. WASAPI usually supplies `F32` or `I16`, but a virtual cable can supply others.
- `Stream` is not guaranteed `Send` across platforms. Create it and keep it alive on a dedicated capture thread; do not store it in `AppState` or move it to another thread. The thread owns `Option<Stream>` and drops it on stop.
- `build_input_stream` runs the data callback on the audio thread. Do not lock, log, resample, or write files there. `send_mono` allocates one `Vec` per callback; that is a pragmatic trade-off, not real-time-safe. For a production recorder, use a preallocated buffer pool (or push into a lock-free ring) and `try_send` a recycled buffer. If the queue is full, count a dropped chunk and let the UI show a warning; never block.
- The `tx` in this snippet is the capture-to-resampler queue, not the VAD queue. The resampler output is accumulated and re-blocked into exactly 512-sample chunks before it enters the `MIC_QUEUE_CHUNKS` queue (§8.5.6). Do not feed arbitrary callback lengths to the detector.
- The capture thread sends raw mono frames plus the device sample rate to a resampling worker. Use `sherpa_onnx::LinearResampler::create(device_rate, 16_000)`; it is a filtered resampler, already linked through sherpa-onnx, and avoids a second DSP crate. `resample(&mono, false)` returns arbitrary-length output; accumulate and drain 512-sample windows for Silero. At stop, call `resample(&[], true)` and flush the residual.
- Device enumeration for the UI: `host.input_devices()` then `device.description()?.name()` (0.18 replaced `name()` with `description()`; `Display` on `Device` also yields the name). De-duplicate by name; Windows exposes the same microphone through several driver APIs. “Default microphone” maps to `host.default_input_device()`.
- The capture thread should stop on a `recv_timeout`/`AtomicBool` and drop the stream before joining. Do not call `Stream::pause` from the UI thread; cpal streams are not required to be thread-safe for control.
- `on_close` must stop capture and finalize the WAV. The job object (§10.7.6) is for child processes, not the audio thread.

### 7.4 `hound` 3.5.1 — WAV

```toml
hound = "3.5.1"
```

Final WAV writing:

```rust
use hound::{SampleFormat as HoundSampleFormat, WavSpec, WavWriter};

let spec = WavSpec {
    channels: 1,
    sample_rate: 16_000,
    bits_per_sample: 16,
    sample_format: HoundSampleFormat::Int,
};
let mut writer = WavWriter::create(&wav_path, spec).map_err(|e| e.to_string())?;
for &sample in &samples {
    let value = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
    writer.write_sample(value).map_err(|e| e.to_string())?;
}
writer.finalize().map_err(|e| e.to_string())?;
```

- `WavWriter::create` writes the header immediately; `finalize` patches the RIFF/data sizes. If the process crashes before `finalize`, the WAV is unusable. That is why the recording path should mirror the current Python design: write raw `.pcm` plus a JSON sidecar with `sample_rate`, `channels`, `bit_depth`, `samples_written`, and convert with hound on a clean stop or on recovery. `WavWriter::flush` only flushes the `BufWriter`; it does not fsync or patch the header.
- `write_sample` is generic over `hound::Sample`; `i16`, `i32`, `f32` are supported. Match `bits_per_sample` and `sample_format` consistently.
- `WavReader` is useful for reading a clip for playback and for diarization; `WavReader::spec()` and `samples::<i16>()` are the stable calls. Do not use it to decode compressed media; that is ffmpeg's job.
- `hound` has no resampler and no channel mixer. Resample and downmix before writing.

### 7.5 `rodio` 0.23.0 — speaker clip playback

```toml
rodio = { version = "0.23.0", default-features = false, features = ["playback", "wav"] }
```

The scaffold already enables `playback` + `wav`; keep both. `wav` pulls Symphonia's WAV/PCM decoder; it does not pull MP3/FLAC/Vorbis.

Rodio 0.23 changed the output API: build a device sink, then connect a `Player` to its mixer.

```rust
use rodio::{Decoder, DeviceSinkBuilder, Player};

let sink = DeviceSinkBuilder::open_default_sink().map_err(|e| e.to_string())?;
let player = Player::connect_new(sink.mixer());
let file = std::fs::File::open(&clip_path).map_err(|e| e.to_string())?;
let source = Decoder::try_from(file).map_err(|e| e.to_string())?;

player.append(source);
player.sleep_until_end(); // blocks the calling thread, not the UI
```

- Keep `sink` alive for as long as `player` should be audible. Dropping the sink/mixer stops output; the `Player` alone is a queue, not a device.
- `sleep_until_end()` blocks. Run the whole preview on a playback worker thread, or poll `player.empty()` with a short sleep while checking a cancel flag (so a second clip can stop the first). `Player::stop()` and `Player::clear()` are available.
- One `DeviceSinkBuilder::open_default_sink()` per playback session is fine; opening a device per clip adds latency and can fail when the default device changes. A single playback worker that owns the sink and receives `Play(path)`/`Stop` commands is the cleanest design.
- `Decoder::try_from(File)` sets the byte length and seekability from metadata; prefer it over `Decoder::new(BufReader)` for local files.
- No network streams, no `reqwest` feature. `default-features = false` avoids the MP3/FLAC/MP4 decoders the app does not need; if a future feature plays MP3 clips, add `mp3` explicitly.
- Headless tests do not play audio; the `Player` is behind a trait so the UI can assert “preview requested” without a device.

### 7.6 `notify` 8.2.1 — optional, not needed for parity

`notify` 8.2.1 is the last 8.x release and supports Rust 1.85. `notify` 9.0 is currently an RC and raises the MSRV, so do not use it.

```rust
use notify::{RecursiveMode, Watcher};

let mut watcher = notify::recommended_watcher(move |event: notify::Result<notify::Event>| {
    if let Ok(event) = event {
        let _ = UI_TX.send(UiMsg::FilesChanged(event.paths));
        WAKE();
    }
})
.map_err(|e| e.to_string())?;
watcher
    .watch(&watch_dir, RecursiveMode::NonRecursive)
    .map_err(|e| e.to_string())?;
```

- The callback runs on notify's worker thread. Do not touch `AppState`; send a message and call the Rune `Waker`.
- Editors and virus scanners produce bursts of create/modify/rename events. Debounce in app state with `tx.wake_after(Duration::from_millis(250))`; do not wake per event.
- Watch the transcript output folder only, not the runtime/models folder (hundreds of megabytes, no useful events), and never watch the file being written by the app itself without filtering its own writes.
- v3.2.3 has no live file watching. Adding it changes product behaviour; keep it out of v1 unless a concrete user story requires it.

### 7.7 `tracing` 0.1.44 + `tracing-appender` 0.2.4 + `tracing-subscriber` 0.3.22

`tracing` replaces the current Python logger and the scaffold's `log` calls, while `tracing-log` keeps records from wgpu/winit/Rune visible.

```toml
tracing = "0.1.44"
tracing-appender = "0.2.4"
tracing-subscriber = { version = "0.3.22", features = ["fmt", "env-filter", "tracing-log"] }
```

```rust
use tracing_subscriber::EnvFilter;

fn init_logging(log_dir: &std::path::Path) -> tracing_appender::non_blocking::WorkerGuard {
    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
    let appender = tracing_appender::rolling::never(
        log_dir,
        format!("EasyScribe_{timestamp}.log"),
    );
    let (writer, guard) = tracing_appender::non_blocking(appender);

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("info,easyscribe=debug"))
        .with_writer(writer)
        .with_ansi(false)
        .with_target(false)
        .init(); // the `tracing-log` feature installs the `log` bridge here

    guard
}
```

- Keep the returned `WorkerGuard` alive until process exit. Dropping it flushes and stops the writer thread; if it is dropped at the end of `init_logging`, the last lines are lost.
- `rolling::never(dir, "EasyScribe_<timestamp>.log")` gives one file per run, matching the current naming. `RollingFileAppender::builder().max_log_files(10)` only rotates when a rotation period is set; for one-file-per-run, delete the oldest `EasyScribe_*.log` at startup after the new file is created. Do not delete the file currently open.
- `tracing_subscriber::fmt` with `with_ansi(false)` avoids escape codes in the log file. Set `RUST_LOG` support only if a developer build needs it; the shipped app should use a fixed filter.
- `#[instrument(skip_all, fields(job = %id))]` on job functions gives spans without capturing transcript text. Never log `segment.text`, `transcript`, clipboard content, or model output. Log file names, durations, counts, engine/device, and errors only.
- The `log` crate can remain for low-level core code during migration; the `tracing-log` feature bridges it when `.init()` runs. Long term, use `tracing` in core too, but do not mix two subscribers. Do not call `tracing_log::LogTracer::init()` directly unless you also add `tracing-log` as a direct dependency; the feature already does it.
- The logger must be initialized **after** the data root is known (so the log path is correct) and before payload extraction; first-run extraction failures must be logged, not only shown in a dialog. The scaffold currently defers `app::startup` until after extraction, so split logger setup out of `startup` and call it immediately after `data_root` is resolved.

### 7.8 `serde_json` 1.0

`serde_json` is already in the workspace. Use it for the payload manifest, settings, recovery sidecars, whisper JSON and worker IPC.

```rust
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
struct Settings {
    recordings_dir: Option<std::path::PathBuf>,
    speaker_speed_factor: Option<f64>,
    #[serde(flatten)]
    extra: serde_json::Map<String, serde_json::Value>,
}

impl Default for Settings { /* defaults */ }
```

- `#[serde(default)]` makes missing fields use defaults; `#[serde(flatten)] extra` preserves unknown keys when the file is rewritten. Do not use `deny_unknown_fields`: older/newer builds must tolerate each other.
- Read with `serde_json::from_reader(BufReader::new(File::open(path)?))`; write with `to_vec_pretty` to a temp file then rename, so a crash cannot leave a truncated `settings.json`.
- The whisper JSON is not strict: `offsets`, `timestamps` and `text` may be missing in a failed/partial output. Use `#[serde(default)]` on each field and drop empty text.
- Worker IPC should be one JSON object per line (JSONL), not a single document; the parent can read partial output while the child is still running. `serde_json::Deserializer::from_reader(...).into_iter::<Message>()` handles streaming.
- `serde_json::Value` is convenient for the manifest but typed structs catch schema drift earlier. Use typed structs for footer/manifest/settings and `Value` only for diagnostics.

### 7.9 `windows-sys` 0.61.2 and `embed-resource` 3.0.12

```toml
[target.'cfg(windows)'.dependencies]
windows-sys = { version = "0.61.2", features = [
    "Win32_Foundation",
    "Win32_UI_Shell",
    "Win32_UI_WindowsAndMessaging",
    "Win32_System_LibraryLoader",
    "Win32_System_JobObjects",
    "Win32_System_Threading",
    "Win32_System_Console",
    "Win32_Storage_FileSystem",
] }

[target.'cfg(windows)'.build-dependencies]
embed-resource = "3.0.12"
```

- `windows-sys` is raw FFI: every call is `unsafe`, handles are `*mut c_void`/`isize` newtypes, and there are no RAII wrappers. Keep the unsafe blocks tiny and wrap each in a safe function with a `Result`.
- `Win32_UI_Shell` covers `ShellExecuteW`, `SetCurrentProcessExplicitAppUserModelID` and `SHGetKnownFolderPath` if the `dirs` crate is not enough.
- `Win32_UI_WindowsAndMessaging` covers `WM_SETICON`, `LoadImageW`, `SendMessageW`, `GetSystemMetrics`.
- `Win32_System_JobObjects` covers `CreateJobObjectW`, `SetInformationJobObject`, `AssignProcessToJobObject`; `Win32_System_Threading` covers `GetCurrentProcess`; `Win32_System_Console` covers `AttachConsole` for `--version`/`--payload-verify` output from a GUI-subsystem binary.
- `Win32_Storage_FileSystem` covers `GetDiskFreeSpaceExW` and `LockFileEx` if `fs2` is ever replaced.
- `embed-resource` compiles the `.rc` at build time on Windows. It is a build dependency only; it adds nothing to the shipped binary. Call `manifest_required()`: it returns `Ok` for `NotWindows`, so a non-Windows cross-check still builds, while a Windows build missing `rc.exe` fails instead of silently shipping without a manifest.
- Do not add `winres` 0.1: it is unmaintained, has no clean manifest support, and its version-info handling is weaker than an `.rc` file.

### 7.10 What not to add

| Crate | Why not |
|---|---|
| `arboard` | Rune already owns the Windows clipboard (text, CF_HTML, PNG) |
| `enigo` / `rdev` | No global input synthesis or hooks in the product |
| `winit` direct | Rune owns the event loop; use `WindowHandle::with_raw_handle` for the rare native call |
| `tokio` | No network, no async runtime; `std::thread` + `mpsc` + Rune `Waker` is enough |
| `reqwest` / `ureq` | Violates the offline promise; CI forbids network crates |
| `symphonia` direct | Rodio already pulls the WAV decoder; ffmpeg handles everything else |
| `ffmpeg-next` | Requires FFmpeg dev libraries/DLLs and a C toolchain; see §9 |
| `notify` by default | No parity requirement; adds a watcher thread and self-trigger complexity |
| `sherpa-rs` | Deprecated shim pinning 1.12.9; replace with official `sherpa-onnx` (§8) |
| `whisper-rs` in release | Vulkan/CRT conflicts and no process isolation; optional CPU fallback only (§8) |
| `winres` | Unmaintained; use `embed-resource` |
| `anyhow` in the app core | The workspace already has a typed `Error`; keep error boundaries explicit |

---

## 8. Speech inference

The app has three different speech paths, and they must not share an ONNX Runtime session:

```
file:    input ──ffmpeg──▶ 16 kHz mono WAV ──whisper-cli──▶ [(start, end, text)]
                                             └─(missing CLI)─▶ sherpa VAD + Whisper ONNX

live:    mic ──cpal──▶ 16 kHz mono ──VAD(512)──▶ segments ──Whisper ONNX──▶ text lines

diarize: WAV ──child process──▶ pyannote segmentation + WeSpeaker embedding ──▶ turns
```

The separation is deliberate. `whisper-cli` is a subprocess, so a Vulkan driver crash cannot take down the UI. The live path needs the Whisper ONNX recognizer loaded in-process for latency, but it is serialized behind one mutex. Diarization is the only path with no cooperative cancellation, so it runs in a killable child.

### 8.1 Engine decisions

| Path | Engine | Version / feature | Why |
|---|---|---|---|
| File transcription | bundled `whisper-cli.exe` | whisper.cpp **1.8.6**, Vulkan build, `-bs 5` | Exact current quality and segmentation; process isolation; no in-process Vulkan |
| File fallback | `sherpa-onnx` Whisper ONNX | **1.13.8**, CPU, static | Works when the CLI or GGUF is absent; same models as live |
| Live / VAD chunks | `sherpa-onnx` Whisper ONNX | **1.13.8**, CPU, static | Current live path; no streaming model; VAD segments are decoded offline |
| VAD | `sherpa-onnx` Silero | **1.13.8**, `silero_vad.onnx` | Same model and tuning as Python |
| Diarization | `sherpa-onnx` | **1.13.8**, CPU, child process | Official API; progress needs the C callback; cancellation needs kill |
| In-process ASR fallback | `whisper-rs` | **0.16.0**, CPU only, feature-gated | Optional emergency path; never the release engine |

This **supersedes** the `REWRITE_PLAN.md` entry that proposed `whisper-rs` with Vulkan as the file engine. The subprocess design is now normative because it preserves the reference binary's exact version (1.8.6 vs 1.8.3 inside `whisper-rs` 0.16), avoids the static-CRT conflict, and isolates GPU driver failures.

#### 8.1.1 Why not `whisper-rs` + Vulkan in the release build

Verified in `~/.cargo/registry/src/.../whisper-rs-0.16.0` and `whisper-rs-sys-0.15.0`:

- `whisper-rs` default features are empty; `vulkan = ["whisper-rs-sys/vulkan", "_gpu", "dep:libc"]` enables `GGML_VULKAN=ON` in the bundled CMake build.
- `whisper-rs-sys` 0.15.0 bundles **whisper.cpp 1.8.3** (`whisper.cpp/CMakeLists.txt: project("whisper.cpp" VERSION 1.8.3)`). The reference ships **1.8.6**. Timestamps and segmentation can differ.
- `whisper-rs-sys/build.rs` never sets `CMAKE_MSVC_RUNTIME_LIBRARY` and never passes `/MT`. CMake's MSVC default is `/MD` (`MultiThreadedDLL`) for Release. A Rust binary built with `-C target-feature=+crt-static` uses `/MT` (`MultiThreaded`), and the link fails with:
  ```
  LNK2038: mismatch detected for 'RuntimeLibrary': value 'MD_DynamicRelease'
  doesn't match value 'MT_StaticRelease'
  ```
- The Vulkan feature links `vulkan-1.lib`. That becomes a load-time import in the exe. On a machine with no Vulkan loader/driver, the process can fail before `main`.
- The CMake build requires a C++ toolchain and adds build time for a path that is not the primary engine.

If a fallback is ever shipped, build it as a separate **CPU-only** feature with the same MSVC runtime as the rest of the binary, and prove transcript parity against `whisper-cli` on the fixture corpus. Do not enable `vulkan` in the release profile. The release `Cargo.toml` should not depend on `whisper-rs` at all.

#### 8.1.2 Version pins

```toml
# crates/easyscribe-core/Cargo.toml
sherpa-onnx = { version = "=1.13.8", default-features = false, features = ["static"] }
sherpa-onnx-sys = "=1.13.8"   # only if the diarization callback shim is used directly
```

- Pin with `=`: the C API and model config layout are not semver-stable across minor releases.
- `sherpa-onnx` 2.0.1 exists on crates.io, but it is a major API bump. Do not take it during this rewrite without a full model/config audit.
- `sherpa-rs` 0.6.8 is removed. It pins `sherpa-rs-sys` 0.6.8 → sherpa-onnx **1.12.9**, hardcodes `greedy_search`, and its `DiarizeConfig` defaults (`min_duration_on: 0.0`, `min_duration_off: 0.0`, `num_clusters: 4`) do **not** match the Python reference (`-1`, `0.3`, `0.5`). Those are silent parity bugs in the current scaffold.
- `sherpa-onnx-sys` defaults to `static`; do not enable `shared` unless the deployment deliberately ships DLLs and sets a DLL search directory.

#### 8.1.3 Windows static-link requirements

`sherpa-onnx-sys` resolves the prebuilt archive in this order:

1. `SHERPA_ONNX_LIB_DIR` — a directory that already contains the extracted libraries.
2. `SHERPA_ONNX_ARCHIVE_DIR` — a directory containing the exact archive named
   `sherpa-onnx-v1.13.8-win-x64-static-MT-Release-lib.tar.bz2`.
3. Download from GitHub at build time.

CI must pre-populate one of the first two; the build must not depend on a network fetch. The Windows static archive is built with `/MT`, which is why `+crt-static` is the preferred target configuration. The archive includes ONNX Runtime; no `onnxruntime.dll` is shipped.

The build script emits no Windows system libraries. If the MSVC linker asks for `advapi32`, `bcrypt`, `userenv`, `ws2_32`, `ole32`, or `shell32`, add them to the `easyscribe` binary's `build.rs` with `cargo:rustc-link-lib=dylib=...`; do not guess preemptively. This is a Gate 0 spike (§11).

### 8.2 File transcription with `whisper-cli`

#### 8.2.1 Availability and engine selection

```rust
pub fn whisper_cli_available(paths: &Paths) -> bool {
    paths.whispercpp_bin.is_file() && paths.whispercpp_model.is_file()
}

pub enum FileEngine {
    WhisperCli,
    Sherpa,
}

pub fn choose_file_engine(paths: &Paths) -> FileEngine {
    if whisper_cli_available(paths) && cli_started(paths) {
        FileEngine::WhisperCli
    } else {
        FileEngine::Sherpa
    }
}

/// Cached result of the first `whisper-cli` spawn. A missing Vulkan/VC DLL
/// fails here; a successful spawn with no GPU still returns true.
pub fn cli_started(paths: &Paths) -> bool {
    static STARTED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *STARTED.get_or_init(|| probe_gpu(paths).is_ok())
}
// In production, cache one `CliProbe { started: bool, gpu: Option<i32> }`
// instead of two OnceLocks; the two snippets above show the predicates, not
// the final storage layout.
```

- The CLI path is `paths.whispercpp_bin` (`whispercpp/whisper-cli.exe`); the model is `paths.whispercpp_model` (`whispercpp/ggml-large-v3-turbo-q5_0.bin`).
- If both exist, the CLI is used. If either is missing, the code falls back to the sherpa VAD path. This matches `whispercpp_wrapper.is_available()` and `TranscriptionEngine.transcribe`.
- A CLI **failure after it starts** (nonzero exit, missing JSON) is an error. Do not silently retry with sherpa; that would hide a GPU/driver problem and change output.
- A **load-time** failure is different: the Vulkan build imports `vulkan-1.dll`, and on a VM with no Vulkan loader `CreateProcess` fails before `main`. `is_file()` cannot see this. Gate 0 must test a clean VM. If the Vulkan CLI cannot start, ship a CPU-only `whisper-cli-cpu.exe` (a few MB) in `whispercpp/` and select it when the Vulkan probe fails, or route file transcription to sherpa. Do not assume the Vulkan build falls back to CPU when the loader itself is missing.

#### 8.2.2 Exact argv

```rust
pub fn whisper_command(
    paths: &Paths,
    audio: &Path,
    out_stem: &Path,
    gpu: Option<i32>,
) -> std::process::Command {
    let mut cmd = std::process::Command::new(&paths.whispercpp_bin);
    cmd.arg("-m").arg(&paths.whispercpp_model)
        .arg("-f").arg(audio)
        .arg("-bs").arg("5")
        .arg("-t").arg(num_threads().to_string());
    if let Some(index) = gpu {
        cmd.arg("-dev").arg(index.to_string());
    }
    cmd.arg("-pp")
        .arg("--output-json")
        .arg("--output-file").arg(out_stem)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}
```

- The order is exact: model, file, beam size, threads, optional device, `-pp`, JSON, output stem.
- `-bs 5` is beam search, not greedy. Do not "optimize" it to `-bs 1`.
- `out_stem` is `paths.temp_dir.join(format!("whispercpp_{}", uuid::Uuid::new_v4().simple()))`; the JSON path is `out_stem.with_extension("json")`.
- `stdout` is `null`, because whisper-cli writes transcript text to stdout and an unread pipe would fill. The JSON file is authoritative.
- `stderr` is piped and read on a dedicated thread.
- `-dev` is omitted entirely when the probe returns `None`. Passing `-dev 0` when no Vulkan device exists changes behavior; do not do it.

`num_threads()` is the reference formula:

```rust
pub fn num_threads() -> usize {
    let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    (cpus / 2).clamp(4, 8)
}
```

Examples: 1 CPU → 4, 8 → 4, 10 → 5, 16 → 8, 32 → 8.

#### 8.2.3 GPU probe

The reference runs the CLI once with an empty model to make it print the Vulkan device list, then picks a device. Port it with `OnceLock<Option<i32>>`:

```rust
static BEST_GPU: std::sync::OnceLock<Option<i32>> = std::sync::OnceLock::new();

pub fn best_gpu(paths: &Paths) -> Option<i32> {
    *BEST_GPU.get_or_init(|| probe_gpu(paths).unwrap_or_else(|e| {
        log::warn!("GPU probe failed, whisper.cpp picks the device: {e}");
        None
    }))
}
```

`probe_gpu`:

1. Create `paths.temp_dir.join(format!("probe_{}.bin", uuid::Uuid::new_v4().simple()))` and write zero bytes.
2. Run `whisper-cli -m <empty> -f <empty>` with stderr piped, stdout null, `CREATE_NO_WINDOW`.
3. Enforce a 30 s deadline with a `try_wait` loop; on timeout, `kill()` and `wait()`.
4. Read stderr as UTF-8 lossy, split lines, call `pick_gpu(&lines)`.
5. Delete the empty file in all paths.
6. Any error returns `Err`; `best_gpu` logs and caches `None`.

Exact regexes (same as `src/whispercpp_wrapper.py:46-52`):

```rust
static VULKAN_DEV: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"ggml_vulkan: (\d+) = (.+?) \| (.*)").unwrap()
});
static SYSTEM_INFO: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"system_info: n_threads = (\d+) / (\d+) \|(.*)").unwrap()
});
static USING: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"whisper_backend_init_gpu: using (\S+) backend").unwrap()
});
static PROGRESS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"whisper_print_progress_callback: progress = (\d+)%").unwrap()
});
```

`pick_gpu` ranking, exactly:

```rust
fn pick_gpu(lines: &[String]) -> Option<i32> {
    let mut devices: Vec<(i32, String)> = Vec::new();
    for line in lines {
        if let Some(c) = VULKAN_DEV.captures(line) {
            if let Ok(index) = c[1].parse::<i32>() {
                devices.push((index, c[3].to_string()));
            }
        }
    }
    devices.into_iter().max_by_key(|(index, details)| {
        (
            !details.contains("uma: 1"),                 // dedicated first
            !details.contains("matrix cores: none"),     // matrix cores (or unknown) first
            -*index,                                     // lower index wins ties
        )
    }).map(|(index, _)| index)
}
```

Consequences, verified from the Python tests:

- dedicated (`uma: 1` absent) beats integrated;
- if both are dedicated, `matrix cores: none` loses to a device without that string;
- a device line with no `matrix cores` at all is treated as having matrix cores;
- ties go to the lowest Vulkan index.

The GPU probe is cached for the process lifetime. Changing GPUs or drivers requires restarting the app, as in Python.

The probe doubles as a load check. If `Command::spawn` fails with `ERROR_DLL_NOT_FOUND` (for example `vulkan-1.dll` missing), cache `cli_started = false` alongside the GPU result, and make `choose_file_engine` return `Sherpa` for the rest of the session. Do not retry a known-unloadable CLI for every file. If `cli_started` is true but the GPU list is empty, the CLI can still run on CPU; keep it.

#### 8.2.4 Parsing the JSON output

```rust
#[derive(serde::Deserialize)]
struct WhisperOutput {
    #[serde(default)]
    transcription: Vec<WhisperEntry>,
}

#[derive(serde::Deserialize)]
struct WhisperEntry {
    #[serde(default)]
    text: String,
    #[serde(default)]
    offsets: WhisperOffsets,
}

#[derive(serde::Deserialize, Default)]
struct WhisperOffsets {
    #[serde(default)]
    from: i64,
    #[serde(default)]
    to: i64,
}

pub fn parse_whisper_json(path: &Path) -> std::io::Result<Vec<(f64, f64, String)>> {
    let bytes = std::fs::read(path)?;
    let output: WhisperOutput = serde_json::from_slice(&bytes)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    Ok(output.transcription.into_iter().filter_map(|entry| {
        let text = entry.text.trim().to_owned();
        if text.is_empty() {
            return None;
        }
        Some((
            entry.offsets.from as f64 / 1000.0,
            entry.offsets.to as f64 / 1000.0,
            text,
        ))
    }).collect())
}
```

- Offsets are **milliseconds**, divided by 1000.0.
- Missing offsets default to 0.0; missing `transcription` yields an empty list.
- Whitespace-only text is dropped. Interior whitespace and punctuation are preserved.
- The `timestamps` field in whisper.cpp JSON is ignored.

#### 8.2.5 Progress, device logs, cancellation

Reader thread:

```rust
let stderr = child.stderr.take().expect("piped stderr");
let (line_tx, line_rx) = std::sync::mpsc::channel::<String>();
let log_lines = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));

let log_lines_reader = std::sync::Arc::clone(&log_lines);
let reader = std::thread::Builder::new()
    .name("whisper-stderr".into())
    .spawn(move || {
        use std::io::BufRead;
        for line in std::io::BufReader::new(stderr).lines() {
            let Ok(line) = line else { break };
            let line = line.trim_end().to_owned();
            if let Ok(mut lines) = log_lines_reader.lock() {
                lines.push(line.clone());
            }
            let _ = line_tx.send(line);
        }
    })?;
```

`BufRead::lines()` splits on `\n` and strips a trailing `\r`; whisper.cpp progress lines end in `\n`, so this is sufficient. It is **not** sufficient for ffmpeg (§9).

Main loop:

```rust
let mut progress = 0.0_f32;
loop {
    if let Some(status) = child.try_wait()? {
        break status;
    }
    if cancel.is_cancelled() {
        let _ = child.kill();
        let _ = child.wait();
        let _ = std::fs::remove_file(&json_path);
        return Err(Error::Cancelled);
    }
    while let Ok(line) = line_rx.try_recv() {
        if let Some(c) = PROGRESS.captures(&line) {
            if let Ok(pct) = c[1].parse::<u32>() {
                progress = (pct as f32 / 100.0).clamp(0.0, 1.0);
                sink.progress(progress);
            }
        }
        if SYSTEM_INFO.is_match(&line) {
            let lines = log_lines.lock().unwrap();
            for text in describe_devices(&lines[..]) {
                sink.log(&format!("[whisper.cpp] {text}"));
            }
        }
    }
    std::thread::sleep(std::time::Duration::from_millis(100));
}
```

- Cancellation latency is up to 100 ms plus process teardown.
- On Windows, `Child::kill()` is `TerminateProcess`; there is no graceful `SIGTERM`. The reference calls `terminate()` then `wait(5)`, then `kill()`. Rust's immediate kill is the forced path. The JSON file is deleted in all cancellation/error paths.
- After normal exit, join the reader (it ends at pipe EOF), drain `line_rx`, and deliver any remaining progress. Then call `sink.progress(1.0)`.
- `describe_devices` is only a log helper. Port the exact `_describe_devices` algorithm from `REFERENCE_BACKEND.md` §10.5 if log parity is required; otherwise emit `Engine:`/`GPU:`/`CPU:` lines defensively and never panic on malformed feature text. The Python implementation can raise `IndexError` on a malformed `= 1` feature; do not copy that defect.

On nonzero exit:

```rust
let stderr_text = log_lines.lock().unwrap().join("\n");
let snippet = tail_chars(&stderr_text, 800).trim().to_owned();
return Err(Error::Transcription(format!(
    "whisper-cli exited with code {}.\n\n{}",
    status.code().unwrap_or(-1),
    snippet
)));
```

If the JSON does not exist:

```rust
return Err(Error::Transcription(format!(
    "whisper-cli completed but did not produce {}",
    json_path.file_name().unwrap_or_default().to_string_lossy()
)));
```

`tail_chars` must count **characters**, not bytes, to match Python's `stderr_text[-800:]`:

```rust
fn tail_chars(text: &str, n: usize) -> &str {
    match text.char_indices().rev().nth(n.saturating_sub(1)) {
        Some((index, _)) => &text[index..],
        None => text,
    }
}
```

#### 8.2.6 Cleanup

`--output-file <stem> --output-json` normally writes only `<stem>.json`. Delete both `<stem>.json` and `<stem>` in a guard, because whisper-cli has created other extensions across versions. The reference only deletes the JSON and can leak an extensionless file (`REFERENCE_BACKEND.md` §7.7); the rewrite should not reproduce that leak.

### 8.3 Official `sherpa-onnx`: recognizer setup

#### 8.3.1 Configuration

The Python reference constructs this (`src/transcriber.py:289-304`):

```python
OfflineModelConfig(
    whisper=OfflineWhisperModelConfig(
        encoder=..., decoder=..., language="en", task="transcribe"
    ),
    tokens=...,
    provider="cpu",
    num_threads=NUM_THREADS,
)
OfflineRecognizerConfig(model_config=model_cfg)
```

The Rust equivalent must set the fields whose defaults differ:

```rust
use sherpa_onnx::{
    OfflineRecognizer, OfflineRecognizerConfig, OfflineWhisperModelConfig,
};

pub fn build_recognizer(paths: &Paths) -> Result<OfflineRecognizer> {
    let mut config = OfflineRecognizerConfig::default();

    config.model_config.whisper = OfflineWhisperModelConfig {
        encoder: Some(paths.whisper_encoder.to_string_lossy().into_owned()),
        decoder: Some(paths.whisper_decoder.to_string_lossy().into_owned()),
        language: Some("en".into()),
        task: Some("transcribe".into()),
        tail_paddings: -1,          // Python default; Rust's Default is 0
        enable_token_timestamps: false,
        enable_segment_timestamps: false,
    };
    config.model_config.tokens = Some(paths.whisper_tokens.to_string_lossy().into_owned());
    config.model_config.provider = Some("cpu".into());
    config.model_config.num_threads = num_threads() as i32;
    config.decoding_method = Some("greedy_search".into());

    OfflineRecognizer::create(&config).ok_or_else(|| {
        Error::ModelNotFound("Could not load model: sherpa-onnx returned null".into())
    })
}
```

Critical details:

- `tail_paddings` **must be `-1`**. The Rust struct's derived `Default` is `0`; the Python default is `-1`. The current scaffold passes `tail_paddings: None`, which the old `sherpa-rs` wrapper turns into `0`, so it already differs from Python.
- `decoding_method` is `greedy_search`. `sherpa-rs` hardcoded this; the official crate defaults to `None` (library default), so set it explicitly.
- `provider` is `"cpu"`. There is no CUDA/DirectML path in this product.
- `num_threads` is the same 4–8 formula as whisper-cli.
- `enable_*_timestamps` are false because file timestamps come from whisper.cpp or the VAD segment bounds; live mode has no timestamps.
- `OfflineRecognizer::create` returns `Option`, not `Result`. `None` means the native constructor failed (missing/corrupt model, OOM, unsupported operator). Map it to `ModelNotFoundError`, matching Python's `RuntimeError` catch.

#### 8.3.2 Decoding one chunk

```rust
pub fn transcribe_samples(
    recognizer: &OfflineRecognizer,
    samples: &[f32],
) -> Result<String> {
    let stream = recognizer.create_stream();
    stream.accept_waveform(16_000, samples);
    recognizer.decode(&stream);          // NOT decode_stream
    let text = stream
        .get_result()
        .map(|result| result.text.trim().to_owned())
        .unwrap_or_default();
    Ok(text)
}
```

Verified API names in `sherpa-onnx` 1.13.8: `OfflineRecognizer::create_stream`, `OfflineStream::accept_waveform`, `OfflineRecognizer::decode`, `OfflineStream::get_result`, `OfflineRecognizerResult { text, tokens, timestamps, durations }`. There is no `decode_stream` method; the Python name is `decode_stream`, the Rust name is `decode`.

The `OfflineRecognizer` wrapper is `Send + Sync` by `unsafe impl`, but the underlying C library is not safe for concurrent calls on the same recognizer. Keep one `Mutex<OfflineRecognizer>` or one dedicated inference thread. In live mode, the same mutex must cover the VAD and the recognizer (§8.5).

#### 8.3.3 Model validation

`validate_model_directory()` checks, in order, `silero_vad.onnx`, `whisper_encoder`, `whisper_decoder`, `whisper_tokens`; it does **not** check diarization or whisper.cpp. Startup aborts if this list is non-empty, even when whisper-cli could handle files, because live mode and the sherpa fallback need them.

```rust
pub fn validate_sherpa_models(paths: &Paths) -> Vec<String> {
    [&paths.vad_model, &paths.whisper_encoder, &paths.whisper_decoder, &paths.whisper_tokens]
        .into_iter()
        .filter(|path| !path.is_file())
        .map(|path| format!("Missing: {}", path.display()))
        .collect()
}
```

Error text:

```text
Model validation failed:
Missing: <path>
Missing: <path>
```

and on constructor failure:

```text
Could not load model: <native error>
```

### 8.4 File-mode VAD

#### 8.4.1 Configuration

```rust
use sherpa_onnx::{SileroVadModelConfig, VadModelConfig, VoiceActivityDetector};

pub fn build_file_vad(paths: &Paths) -> Result<VoiceActivityDetector> {
    let mut config = VadModelConfig::default();
    config.silero_vad = SileroVadModelConfig {
        model: Some(paths.vad_model.to_string_lossy().into_owned()),
        threshold: 0.5,
        min_silence_duration: 0.5,
        min_speech_duration: 0.25,
        window_size: 512,
        max_speech_duration: 20.0,
    };
    config.sample_rate = 16_000;
    config.num_threads = 1;
    config.provider = Some("cpu".into());
    config.debug = false;

    // Python's VoiceActivityDetector default is 60 seconds; Rust requires it explicitly.
    VoiceActivityDetector::create(&config, 60.0).ok_or_else(|| {
        Error::Transcription("Could not create VAD: sherpa-onnx returned null".into())
    })
}
```

The three values that are easy to get wrong:

| Field | Python default | Rust `Default` | Must set |
|---|---:|---:|---|
| `window_size` | 512 | 0 | `512` |
| `max_speech_duration` | 20.0 | 0.0 | `20.0` |
| `buffer_size_in_seconds` | 60 | no default, argument | `60.0` (file), `30.0` (live) |

`threshold`, `min_silence_duration`, and `min_speech_duration` match the Python constants (0.5, 0.5, 0.25). Keep `num_threads = 1`; the VAD is cheap and serialized with the recognizer.

#### 8.4.2 Feed loop and segment extraction

```rust
pub const VAD_WINDOW: usize = 512;
pub const PAD_SAMPLES: i64 = 3200; // exact; do not recompute from f32 arithmetic

pub fn detect_bounds(
    vad: &VoiceActivityDetector,
    samples: &[f32],
    cancel: &CancelToken,
) -> Result<Vec<(i64, i64)>> {
    if samples.len() > i32::MAX as usize {
        return Err(Error::Transcription(
            "Audio is too long for the VAD (more than 37 hours at 16 kHz).".into(),
        ));
    }
    let mut bounds = Vec::new();
    let mut pos = 0usize;

    while pos < samples.len() {
        cancel.check()?;
        let end = (pos + VAD_WINDOW).min(samples.len());
        let mut chunk = [0.0_f32; VAD_WINDOW];
        chunk[..end - pos].copy_from_slice(&samples[pos..end]);
        vad.accept_waveform(&chunk);       // final partial window is zero-padded
        pos = end;
        drain_vad(vad, &mut bounds);
    }

    vad.flush();
    drain_vad(vad, &mut bounds);
    Ok(bounds)
}

fn drain_vad(vad: &VoiceActivityDetector, out: &mut Vec<(i64, i64)>) {
    while !vad.is_empty() {
        let (start, n) = {
            let segment = vad.front().expect("non-empty");
            (i64::from(segment.start()), i64::from(segment.n()))
        }; // segment Drop frees the copy returned by SherpaOnnxVoiceActivityDetectorFront
        out.push((start, start + n));
        vad.pop();
    }
}
```

Verified C API contract (`c-api.h`, `SherpaOnnxVoiceActivityDetectorFront`): the returned segment is a **copy owned by the caller**, freed with `SherpaOnnxDestroySpeechSegment`. The Rust wrapper's `SpeechSegment` implements `Drop` and does exactly that. `SpeechSegment::samples()` borrows the wrapper, so copy or measure before the wrapper drops and before `vad.pop()`. Do not hold the wrapper across `pop()`.

Cancellation semantics, for parity:

- Checked once per 512-sample window (32 ms), **before** feeding.
- A cancel that arrives after the last window but before `flush()` is not observed. The flush is bounded and then the caller checks again.
- The final partial window is zero-padded to exactly 512; the padded zeros are part of the VAD timeline.
- Segments are emitted in detector order, with no merge/sort/filter. Preserve that order.

At 16 kHz, `i32` sample indices overflow at `2_147_483_647 / 16_000 ≈ 134_217 s ≈ 37.3 hours`. Reject longer input with a clear error instead of wrapping. If such files are a requirement, process in 30-hour windows and offset the segment indices manually; do not silently truncate.

#### 8.4.3 Padding and clamping

```rust
pub fn pad_and_clamp(bounds: &[(i64, i64)], total_samples: i64) -> Vec<(i64, i64)> {
    bounds.iter().enumerate().map(|(i, &(start, end))| {
        let prev_end = if i == 0 { 0 } else { bounds[i - 1].1 };
        let next_start = bounds.get(i + 1).map(|b| b.0).unwrap_or(total_samples);
        (
            (start - PAD_SAMPLES).max(prev_end),
            (end + PAD_SAMPLES).min(next_start),
        )
    }).collect()
}
```

- Pad both sides by exactly 3200 samples (0.2 s).
- Clamp against the **original** previous end and next start, not against already-padded values.
- First lower bound is 0; last upper bound is `total_samples`.
- The invariant `PAD_SAMPLES < 0.5 * 0.5 * 16_000 = 4000` guarantees adjacent padded segments cannot overlap when the detector's silence gap is at least 0.5 s.
- Keep `pub const PAD_SAMPLES: i64 = 3200;` and add a const assertion that `PAD_SAMPLES as f64 / 16_000.0 == 0.2`. The current core computes it from `f32`; pin it.

Per-segment decode:

```rust
for (index, &(start, end)) in padded.iter().enumerate() {
    cancel.check()?;
    let text = transcribe_samples(&recognizer, &samples[start as usize..end as usize])?;
    if !text.is_empty() {
        segments.push((start as f64 / 16_000.0, end as f64 / 16_000.0, text));
    }
    sink.progress((index + 1) as f32 / padded.len().max(1) as f32);
}
```

If `padded` is empty, do not divide by zero and do not call `progress`; the final write sets 1.0.

### 8.5 Live VAD chunking and decode

The live path is not streaming ASR. It is VAD segmentation followed by offline Whisper decoding of each segment. That is why latency is one segment behind speech and why the transcript has no timestamps.

#### 8.5.1 Constants and queues

| Constant | Value | Behavior |
|---|---:|---|
| `LIVE_VAD_BUFFER_SECONDS` | 30.0 | Detector circular buffer |
| `MIC_QUEUE_CHUNKS` | 1875 | 512-sample chunks waiting for VAD; **drop newest** when full |
| `DECODE_QUEUE_SEGMENTS` | 8 | VAD segments waiting for the recognizer; **drop oldest** when full |
| `HISTORY_SAMPLES` | 11_392 | 3200 pad + 16 × 512 history |
| `STOP_TIMEOUT` | 60 s | Total join budget |
| `VAD_WINDOW` | 512 | Feed size |

`HISTORY_SAMPLES` covers the 0.2 s leading pad plus 16 chunks (0.512 s) of VAD detection lag. Do not shrink it without re-deriving the lag.

#### 8.5.2 Thread topology

```
cpal callback ──try_send──▶ mic queue (1875)
                                │
                         VAD thread (owns detector)
                                │  padded segments
                         decode queue (8)
                                │
                         decode thread (owns recognizer)
                                │  text
                         event channel ──▶ Rune poll
```

- The VAD thread and decode thread share `Arc<Mutex<()>>` (`onnx_lock`). Every `accept_waveform`/`flush` and every `create_stream`/`decode` is inside the lock. This prevents concurrent ONNX Runtime calls, which can abort the process in native code.
- Start the decode thread **before** the VAD thread, matching the reference. The queue would tolerate the reverse, but the order removes a startup race.
- Create the decode queue as `let (decode_tx, decode_rx) = crossbeam_channel::bounded(8);` and give the VAD thread `decode_rx.clone()` for eviction. The decoder owns the other clone.
- Both threads are named (`LiveVAD`, `LiveDecoder`) and daemon-like: the UI must not wait forever on them.
- The cpal stream must stay on the thread that created it. Keep the `cpal::Stream` in a `Recorder` actor thread; do not move it to the VAD thread. The recorder converts to mono 16 kHz and sends `Vec<f32>` chunks. See §7.3 for the cpal/WASAPI details.

#### 8.5.3 VAD loop

```rust
use crossbeam_channel::RecvTimeoutError;

pub fn run_vad(
    mut detector: VoiceActivityDetector,
    mic_rx: crossbeam_channel::Receiver<Vec<f32>>,
    decode_tx: crossbeam_channel::Sender<Vec<f32>>,
    decode_evict: crossbeam_channel::Receiver<Vec<f32>>,
    cancel: Arc<AtomicBool>,
    vad_done: Arc<AtomicBool>,
    onnx_lock: Arc<Mutex<()>>,
    sink: EventSink,
) {
    let mut history: VecDeque<f32> = VecDeque::with_capacity(HISTORY_SAMPLES);
    let mut history_start: i64 = 0;
    let mut prev_end: i64 = 0;
    let mut overflow_notified = false;

    loop {
        // Check every iteration, not only on timeout: a continuously fed
        // microphone queue would otherwise never let Stop through.
        if cancel.load(Ordering::SeqCst) { break; }
        match mic_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(chunk) => feed(
                &mut detector, &mut history, &mut history_start, &mut prev_end,
                &chunk, &decode_tx, &decode_evict, &onnx_lock, &mut overflow_notified, &sink,
            ),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }

    // Drain everything already queued, then flush. This is the reference's
    // "do not lose audio captured before Stop" rule.
    while let Ok(chunk) = mic_rx.try_recv() {
        feed(
            &mut detector, &mut history, &mut history_start, &mut prev_end,
            &chunk, &decode_tx, &decode_evict, &onnx_lock, &mut overflow_notified, &sink,
        );
    }
    {
        let _guard = onnx_lock.lock().expect("onnx mutex poisoned");
        detector.flush();
    }
    drain_live(&detector, &mut history, history_start, &mut prev_end,
               &decode_tx, &decode_evict, &mut overflow_notified, &sink);
    vad_done.store(true, Ordering::SeqCst);
}
```

`feed`:

```rust
fn feed(
    detector: &VoiceActivityDetector,
    history: &mut VecDeque<f32>,
    history_start: &mut i64,
    prev_end: &mut i64,
    chunk: &[f32],
    decode_tx: &crossbeam_channel::Sender<Vec<f32>>,
    decode_evict: &crossbeam_channel::Receiver<Vec<f32>>,
    onnx_lock: &Mutex<()>,
    overflow_notified: &mut bool,
    sink: &EventSink,
) {
    history.extend(chunk.iter().copied());
    if history.len() > HISTORY_SAMPLES {
        let trim = history.len() - HISTORY_SAMPLES;
        history.drain(..trim);
        *history_start += trim as i64;
    }
    {
        let _guard = onnx_lock.lock().expect("onnx mutex poisoned");
        detector.accept_waveform(chunk);
    }
    drain_live(detector, history, *history_start, prev_end, decode_tx, decode_evict, overflow_notified, sink);
}
```

`drain_live`:

```rust
fn drain_live(
    detector: &VoiceActivityDetector,
    history: &VecDeque<f32>,
    history_start: i64,
    prev_end: &mut i64,
    decode_tx: &crossbeam_channel::Sender<Vec<f32>>,
    decode_evict: &crossbeam_channel::Receiver<Vec<f32>>,
    overflow_notified: &mut bool,
    sink: &EventSink,
) {
    while !detector.is_empty() {
        let (seg_start, seg_end, samples) = {
            let segment = detector.front().expect("non-empty");
            let start = i64::from(segment.start());
            let samples = segment.samples().to_vec();
            (start, start + samples.len() as i64, samples)
        }; // wrapper drops, freeing the native copy
        let padded = pad_live_segment(history, history_start, *prev_end, seg_start, &samples);
        *prev_end = seg_end; // original end, not padded
        detector.pop();

        match put_dropping_oldest(decode_tx, decode_evict, padded) {
            SendOutcome::Overflow if !*overflow_notified => {
                *overflow_notified = true;
                log::warn!("Live decode queue full — dropped oldest segment");
                sink.overflow();
            }
            SendOutcome::Closed => {
                // The decoder has exited; keep draining the VAD so it can
                // reach `flush()` without blocking, but send nothing else.
            }
            _ => {}
        }
    }
}
```

`pad_live_segment` pads only the **leading** edge:

```rust
pub fn pad_live_segment(
    history: &VecDeque<f32>,
    history_start: i64,
    prev_end: i64,
    seg_start: i64,
    seg: &[f32],
) -> Vec<f32> {
    let pad_start = (seg_start - PAD_SAMPLES).max(prev_end).max(history_start);
    if pad_start >= seg_start {
        return seg.to_vec();
    }
    let from = (pad_start - history_start) as usize;
    let to = ((seg_start - history_start) as usize).min(history.len());
    if from >= to {
        return seg.to_vec();
    }
    let mut out = Vec::with_capacity((to - from) + seg.len());
    out.extend(history.iter().skip(from).take(to - from));
    out.extend_from_slice(seg);
    out
}
```

- `prev_end` is the original end of the previous segment; it prevents leading padding from reaching into the previous segment.
- `history_start` prevents reading older samples than the ring buffer holds.
- No trailing padding in live mode.
- If no padding is available, return a copy of the segment. The Python version returns the same array object; callers only read, so a copy is behaviorally equivalent.

`put_dropping_oldest`:

```rust
enum SendOutcome {
    Sent,
    Overflow,
    Closed,
}

fn put_dropping_oldest<T>(
    tx: &crossbeam_channel::Sender<T>,
    evict: &crossbeam_channel::Receiver<T>,
    item: T,
) -> SendOutcome {
    use crossbeam_channel::TrySendError;
    match tx.try_send(item) {
        Ok(()) => SendOutcome::Sent,
        Err(TrySendError::Full(item)) => {
            let _ = evict.try_recv();       // evict the oldest
            let _ = tx.try_send(item);      // best effort; a race can still drop
            SendOutcome::Overflow
        }
        Err(TrySendError::Disconnected(_)) => SendOutcome::Closed,
    }
}
```

The first send failure is what drives the one-time overflow warning, exactly as in Python. The second send may also fail only under an unexpected multi-producer race; the VAD thread is the sole producer. If the decoder has already exited, `try_send` returns `Disconnected` and `put_dropping_oldest` returns `Closed`: treat that as a stopped session, not an overflow, and stop feeding. Do not log the overflow warning in that case.

#### 8.5.4 Decode loop

```rust
fn decode_one(
    recognizer: &Mutex<Option<OfflineRecognizer>>,
    onnx_lock: &Mutex<()>,
    sink: &EventSink,
    samples: Vec<f32>,
) -> bool {
    // Drop both locks before sending; the event sink must never run under them.
    let result = {
        let _guard = onnx_lock.lock().expect("onnx mutex poisoned");
        let guard = recognizer.lock().expect("recognizer mutex poisoned");
        match guard.as_ref() {
            Some(recognizer) => transcribe_samples(recognizer, &samples),
            None => Err(Error::ModelNotFound("Speech model is not loaded.".into())),
        }
    };
    match result {
        Ok(text) if !text.is_empty() => {
            sink.live_segment(text);
            true
        }
        Ok(_) => true,
        Err(error) => {
            sink.live_error(&error.to_string());
            false
        }
    }
}

pub fn run_decoder(
    recognizer: std::sync::Arc<Mutex<Option<OfflineRecognizer>>>,
    decode_rx: crossbeam_channel::Receiver<Vec<f32>>,
    vad_done: Arc<AtomicBool>,
    onnx_lock: Arc<Mutex<()>>,
    sink: EventSink,
) {
    use crossbeam_channel::RecvTimeoutError;
    loop {
        match decode_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(samples) => {
                if !decode_one(recognizer.as_ref(), &onnx_lock, &sink, samples) {
                    break;
                }
            }
            Err(RecvTimeoutError::Timeout) => {
                if vad_done.load(Ordering::SeqCst) {
                    // The VAD thread sets vad_done after it stops producing and
                    // drops decode_tx, so recv() drains and then disconnects.
                    while let Ok(samples) = decode_rx.recv() {
                        if !decode_one(recognizer.as_ref(), &onnx_lock, &sink, samples) {
                            break;
                        }
                    }
                    break;
                }
            }
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
}
```

- The final blocking drain happens only after `vad_done` is set, so the last segments are not lost; the decoder still processes segments as they arrive before that.
- `on_segment` is called **outside** the ONNX lock and outside the recognizer lock. The event sink only sends and wakes; it does not touch Rune state.
- Empty/whitespace-only results are dropped.
- A recognizer error is surfaced as a live error event and stops the decoder. The Python implementation silently kills the thread (`src/live_transcriber.py:179-202`); that is a defect, not parity to preserve.
- Live output has no timestamps; each non-empty segment becomes one line in the live transcript.

#### 8.5.5 Stop and flush ordering

The UI's Stop handler must:

1. Set the cancel flag (`Arc<AtomicBool>`), not a Rune-only flag.
2. Let the VAD thread observe it within 100 ms, drain `mic_rx`, flush the detector, and push the final segments.
3. Wait for the decoder to finish the queue.
4. Stop the cpal stream and convert the PCM to WAV.

The reference waits up to 60 s total. Rust:

```rust
pub fn stop_live(
    cancel: &Arc<AtomicBool>,
    vad: std::thread::JoinHandle<()>,
    decoder: std::thread::JoinHandle<()>,
    timeout: Duration,
) {
    cancel.store(true, Ordering::SeqCst);
    let deadline = Instant::now() + timeout;
    while !(vad.is_finished() && decoder.is_finished()) {
        if Instant::now() >= deadline {
            log::warn!("Live transcription did not finish within {} s", timeout.as_secs());
            return; // dropping JoinHandle detaches; do not block the UI
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let _ = vad.join();
    let _ = decoder.join();
}
```

- If the deadline expires, the threads are detached. The UI must stop accepting their events (drop the receiver or ignore events with the old session id) so a late `on_segment` cannot mutate the next recording. The Python code has exactly this edge (`stop()` drops the handle and the daemon may still call `on_segment`); Rust should close it with a session id.
- The full recording is still saved even if the live transcript lags. The live transcript is a convenience, not the source of truth.
- On overflow, log exactly:
  ```text
  [Warning] transcription falling behind. Some audio was skipped in the live view; the full recording is still saved
  ```
  and call the UI's `on_overflow` once per session.

#### 8.5.6 cpal integration boundary

The recorder actor owns the `cpal::Stream`. It captures at the device's default config, converts all sample formats to `f32` in `[-1, 1]`, downmixes to mono, resamples to 16 kHz, and sends chunks. It must **re-block** the resampled stream into exactly 512-sample chunks before enqueueing: `MIC_QUEUE_CHUNKS` is a count of 512-sample chunks (60 s), not a count of arbitrary cpal callback buffers. Do not request 16 kHz from WASAPI and assume conversion; shared-mode WASAPI does not auto-convert in cpal. The VAD and recognizer must only ever see 16 kHz mono `f32`.

`crossbeam_channel::bounded(1875)` with `try_send` gives the reference's drop-newest behavior at the microphone boundary. 1875 chunks × 512 samples / 16 kHz = 60 s of buffering, matching the Python queue. The decode queue is `crossbeam_channel::bounded(8)`; keep a clone of its `Receiver` in the VAD thread solely to evict the oldest segment. `crossbeam_channel::Receiver` is `Clone`; `std::sync::mpsc::Receiver` is not, which is why the standard library is not sufficient here.

### 8.6 Diarization in a child process

#### 8.6.1 Why a child

The official Rust `OfflineSpeakerDiarization` API exposes:

```rust
pub fn create(config: &OfflineSpeakerDiarizationConfig) -> Option<Self>;
pub fn process(&self, samples: &[f32]) -> Option<OfflineSpeakerDiarizationResult>;
```

There is **no** progress callback and **no** cancellation token. The C API does have:

```c
typedef int32_t (*SherpaOnnxOfflineSpeakerDiarizationProgressCallback)(
    int32_t num_processed_chunks, int32_t num_total_chunks, void *arg);

const SherpaOnnxOfflineSpeakerDiarizationResult *
SherpaOnnxOfflineSpeakerDiarizationProcessWithCallback(
    const SherpaOnnxOfflineSpeakerDiarization *sd, const float *samples,
    int32_t n, SherpaOnnxOfflineSpeakerDiarizationProgressCallback callback,
    void *arg);
```

But `c-api.h` states: “The current implementation reports progress but ignores the callback's return value.” Returning nonzero does **not** cancel. Killing the process is the only reliable cancellation. That is exactly why the Python app uses a `spawn` child (`src/diarizer.py:1-8`, `REFERENCE_BACKEND.md` §12.9) and why the rewrite should too.

A child also isolates the large diarization models and any native abort from the UI process. The parent does not load the audio; the child does, matching the current pipeline.

#### 8.6.2 Worker entry point

`main` must check for the worker argument **before** any Rune/COM/window initialization:

```rust
fn main() {
    if let Some(code) = easyscribe::diarize_worker::run_if_requested() {
        std::process::exit(code);
    }
    // ... platform::set_app_identity(), Rune, etc.
}
```

Command line:

```text
EasyScribe.exe --worker diarize --wav <path> --threads <n>
```

- `run_if_requested` parses `args_os`, requires exactly `--worker diarize`, reads `--wav` and `--threads`, and returns `None` for every other invocation.
- The child must not call `set_app_identity`, must not create a window, and must not start the payload bootstrap. It only needs the already-linked sherpa library and the WAV path.
- The parent spawns `std::env::current_exe()`, not a second binary in the payload. The child is therefore the same signed exe.
- On Windows, use `CREATE_NO_WINDOW` so a console window never flashes.
- Redirect `stdin` to null, `stdout` to a pipe, `stderr` to null (or a log file). Native sherpa logs go to stderr; if you pipe it and never read it, the child can block on a full pipe. Null is the safe default; if logs are wanted, use a dedicated reader thread.

#### 8.6.3 Exact configuration

```rust
use sherpa_onnx::{
    FastClusteringConfig, OfflineSpeakerDiarization, OfflineSpeakerDiarizationConfig,
    OfflineSpeakerSegmentationModelConfig, OfflineSpeakerSegmentationPyannoteModelConfig,
    SpeakerEmbeddingExtractorConfig,
};

pub fn build_diarizer(paths: &Paths, threads: i32) -> Option<OfflineSpeakerDiarization> {
    let config = OfflineSpeakerDiarizationConfig {
        segmentation: OfflineSpeakerSegmentationModelConfig {
            pyannote: OfflineSpeakerSegmentationPyannoteModelConfig {
                model: Some(paths.separator_model.to_string_lossy().into_owned()),
                window_shift_ratio: 0.1,
            },
            num_threads: threads,
            debug: false,
            provider: Some("cpu".into()),
        },
        embedding: SpeakerEmbeddingExtractorConfig {
            model: Some(paths.embedding_model.to_string_lossy().into_owned()),
            num_threads: threads,
            debug: false,
            provider: Some("cpu".into()),
        },
        clustering: FastClusteringConfig {
            num_clusters: -1,       // automatic speaker count
            threshold: 0.5,
            compute_confidence: false,
        },
        min_duration_on: 0.3,       // Python default; do not use 0.0
        min_duration_off: 0.5,      // Python default; do not use 0.0
    };
    OfflineSpeakerDiarization::create(&config)
}
```

The current `sherpa-rs` code sets `num_clusters: -1` but `min_duration_on: 0.0` and `min_duration_off: 0.0`. That is a silent behavioral difference from Python's `OfflineSpeakerDiarizationConfig` defaults (`0.3`, `0.5`). Fix it during migration.

`threads` is `NUM_THREADS` (4–8). The reference passes the same value to both the segmentation and embedding models.

#### 8.6.4 Progress callback shim

The safe crate does not expose `ProcessWithCallback`. Two clean options:

1. **Vendor `sherpa-onnx` 1.13.8** under `vendor/sherpa-onnx` and add one method (recommended: keeps all config code safe).
2. Use `sherpa-onnx-sys` directly in the child and build the config structs by hand (works, but duplicates the config mapping and is easier to get wrong).

Vendored patch (`vendor/sherpa-onnx/src/offline_speaker_diarization.rs`):

```rust
use std::ffi::c_void;

type ProgressCallback = unsafe extern "C" fn(i32, i32, *mut c_void) -> i32;

unsafe extern "C" {
    fn SherpaOnnxOfflineSpeakerDiarizationProcessWithCallback(
        sd: *const sys::OfflineSpeakerDiarization,
        samples: *const f32,
        n: i32,
        callback: Option<ProgressCallback>,
        arg: *mut c_void,
    ) -> *const sys::OfflineSpeakerDiarizationResult;
}

impl OfflineSpeakerDiarization {
    pub fn process_with_callback<F>(&self, samples: &[f32], mut progress: F)
        -> Option<OfflineSpeakerDiarizationResult>
    where
        F: FnMut(i32, i32),
    {
        unsafe extern "C" fn trampoline<F: FnMut(i32, i32)>(
            done: i32,
            total: i32,
            arg: *mut c_void,
        ) -> i32 {
            // The pointer is valid for the duration of the C call.
            let callback = unsafe { &mut *(arg as *mut F) };
            callback(done, total);
            0 // ignored by sherpa-onnx; progress only
        }

        if samples.len() > i32::MAX as usize {
            return None;
        }
        let mut callback = progress;
        let ptr = unsafe {
            SherpaOnnxOfflineSpeakerDiarizationProcessWithCallback(
                self.ptr,
                samples.as_ptr(),
                samples.len() as i32,
                Some(trampoline::<F>),
                &mut callback as *mut F as *mut c_void,
            )
        };
        if ptr.is_null() {
            None
        } else {
            Some(OfflineSpeakerDiarizationResult { ptr })
        }
    }
}
```

`Cargo.toml`:

```toml
[patch.crates-io]
sherpa-onnx = { path = "vendor/sherpa-onnx" }
```

- The callback runs on the thread that calls `process_with_callback` (the child's main thread); it must only send a message, never touch Rune state.
- The return value is ignored by the C library. Do not build cancellation on it.
- If upstream adds this method, drop the patch. Track the upstream issue/PR in the risk register.

#### 8.6.5 Child pipeline

The child writes one JSON object per line to stdout:

```rust
fn send(message: &WorkerMsg) {
    use std::io::Write;
    let mut out = std::io::stdout().lock();
    if serde_json::to_writer(&mut out, message).is_ok() {
        let _ = out.write_all(b"\n");
        let _ = out.flush();
    }
}
```

```rust
let (samples, sample_rate) = load_wav_16k_mono(&wav_path)?;
if sample_rate != 16_000 {
    return fail("Diarization requires 16 kHz mono audio.");
}
let diarizer = build_diarizer(&paths, threads).ok_or("Could not create the diarizer.")?;
let mut last_pct = -1i32;
let result = diarizer.process_with_callback(&samples, |done, total| {
    let pct = if total > 0 { (100 * done / total).clamp(0, 100) } else { 0 };
    if pct != last_pct {
        last_pct = pct;
        send(&WorkerMsg::Progress { value: pct });
    }
}).ok_or("Diarization failed.")?;

let turns: Vec<Turn> = result.sort_by_start_time().into_iter().map(|segment| Turn {
    start: f64::from(segment.start),
    end: f64::from(segment.end),
    speaker: format!("SPEAKER_{:02}", segment.speaker),
}).collect();
send(&WorkerMsg::Done { turns });
```

- `sort_by_start_time()` returns `Vec<OfflineSpeakerDiarizationSegment>` by value; use it directly.
- `speaker` is an `i32`; format with `{:02}` so 7 → `SPEAKER_07`, 100 → `SPEAKER_100`.
- `start`/`end` are seconds (`f32`), not samples.
- The child loads the WAV once. A 3-hour 16 kHz mono `f32` buffer is ~691 MB (`3 * 3600 * 16000 * 4`); 1 hour is ~230 MB. Account for the diarization models and ONNX Runtime workspace on top. Do not run diarization concurrently with a live decode in the same process; the child makes that impossible.
- The child checks `OfflineSpeakerDiarization::sample_rate()` if the model reports a different rate. Production WAVs are 16 kHz; reject anything else with a clear error rather than resampling silently.

#### 8.6.6 Parent/child protocol

Use newline-delimited JSON on stdout:

```json
{"kind":"progress","value":42}
{"kind":"done","turns":[{"start":0.0,"end":1.2,"speaker":"SPEAKER_00"}]}
{"kind":"error","message":"RuntimeError: ..."}
```

Rust types:

```rust
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WorkerMsg {
    Progress { value: i32 },
    Done { turns: Vec<Turn> },
    Error { message: String },
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct Turn {
    pub start: f64,
    pub end: f64,
    pub speaker: String,
}
```

Parent:

```rust
let mut command = Command::new(std::env::current_exe()?);
command
    .arg("--worker").arg("diarize")
    .arg("--wav").arg(&wav_path)
    .arg("--threads").arg(num_threads().to_string())
    .stdin(Stdio::null())
    .stdout(Stdio::piped())
    .stderr(Stdio::null());
#[cfg(windows)]
{
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    command.creation_flags(CREATE_NO_WINDOW);
}
let mut child = command.spawn()?;

let stdout = child.stdout.take().expect("piped stdout");
let (msg_tx, msg_rx) = mpsc::channel::<Result<WorkerMsg, String>>();
let reader = std::thread::spawn(move || {
    use std::io::BufRead;
    for line in std::io::BufReader::new(stdout).lines() {
        match line {
            Ok(line) if line.trim().is_empty() => {}
            Ok(line) => {
                let parsed = serde_json::from_str::<WorkerMsg>(&line)
                    .map_err(|e| format!("bad worker message: {e}: {line}"));
                if msg_tx.send(parsed).is_err() { break; }
            }
            Err(e) => { let _ = msg_tx.send(Err(e.to_string())); break; }
        }
    }
});
```

Main loop:

```rust
let mut last_logged = -1;
loop {
    if cancel.is_cancelled() {
        let _ = child.kill();
        let _ = child.wait();
        let _ = reader.join();
        return Err(Error::Cancelled);
    }
    match msg_rx.recv_timeout(Duration::from_millis(200)) {
        Ok(Ok(WorkerMsg::Progress { value })) => {
            sink.progress(value as f32 / 100.0);
            if value / 10 > last_logged {
                last_logged = value / 10;
                sink.log(&format!("[Diarize] Progress: {value}%"));
            }
        }
        Ok(Ok(WorkerMsg::Done { turns })) => {
            let _ = child.kill();
            let _ = child.wait();
            let _ = reader.join();
            return Ok(turns);
        }
        Ok(Ok(WorkerMsg::Error { message })) => {
            let _ = child.kill();
            let _ = child.wait();
            let _ = reader.join();
            return Err(Error::Diarization(format!("Speaker diarization failed: {message}")));
        }
        Ok(Err(message)) => {
            let _ = child.kill();
            let _ = child.wait();
            let _ = reader.join();
            return Err(Error::Diarization(message));
        }
        Err(RecvTimeoutError::Timeout) => {
            if let Some(status) = child.try_wait()? {
                let code = status.code().unwrap_or(-1);
                return Err(Error::Diarization(format!(
                    "Speaker identification stopped unexpectedly (exit code {code})."
                )));
            }
        }
        Err(RecvTimeoutError::Disconnected) => {
            let _ = reader.join();
            let status = child.wait()?;
            let code = status.code().unwrap_or(-1);
            return Err(Error::Diarization(format!(
                "Speaker identification stopped unexpectedly (exit code {code})."
            )));
        }
    }
}
```

- `recv_timeout(200 ms)` is what makes cancellation responsive even during the speech-finding phase, when no progress arrives.
- On `Done`, kill the child even though it should exit on its own. The reference calls `_stop_process` in `finally` after receiving `"done"`; this reaps a child that is slow to exit.
- If the child exits without a `done`/`error` message, the error includes the exit code.
- If the child sends `{"kind":"error","message":"RuntimeError: ..."}`, the parent wraps it exactly as `Speaker diarization failed: RuntimeError: ...`.
- Unknown `kind` values are ignored by `serde` only if the enum is `#[serde(other)]`-tolerant; the reference ignores unknown kinds. Add an `Unknown` variant or parse `Value` first if forward compatibility matters. For v1, an unknown kind is a protocol error, which is acceptable and easier to debug.

#### 8.6.7 Cancellation and process lifetime

- Cancellation is `TerminateProcess` on Windows (Rust `Child::kill`). There is no graceful stop. The child reads only the WAV and writes no files, so there is nothing to clean up.
- The parent must kill the child in `on_close`, in the job's error path, and after `done`.
- Put child processes in a Windows Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`. If the parent crashes, the OS kills the child. `on_close` handles the normal path; the job handles the crash path.
- Do not use TCP or named pipes for the protocol. A TCP loopback is still a network API and weakens the offline guarantee. Anonymous pipes are sufficient.
- Diarization is the slowest stage. The reference estimates `3.0 + 0.045 * audio_sec + 0.165 * speech_sec` seconds, adjusted by the learned `speaker_speed_factor` (clamped 0.2–5.0). Keep that estimate and the “less than a minute” formatting; it is user-visible before the run.

#### 8.6.8 Error strings

| Condition | Exact text |
|---|---|
| Models missing | `The speaker models are missing.` |
| WAV load failure | `Could not load audio for diarization: {error}` |
| Child exits unexpectedly | `Speaker identification stopped unexpectedly (exit code {code}).` |
| Child sends error | `Speaker diarization failed: {message}` |
| User cancels | `Speaker identification cancelled by user` |
| Non-diarization failure in the file pipeline | `Diarization failed, falling back to plain: {error}` (log only) |
| UI fallback notice | `[Diarize] Warning: {error}. Writing the transcript without speaker names.` |

The file pipeline catches every diarization error except cancellation and falls back to a plain transcript. The “Find speakers” action does **not** fall back; it surfaces the error.

### 8.7 Migrating off `sherpa-rs` 0.6.8

| Current (`sherpa-rs`) | Official (`sherpa-onnx` 1.13.8) | Notes |
|---|---|---|
| `SileroVad::new(SileroVadConfig, buffer)` | `VoiceActivityDetector::create(&VadModelConfig, buffer)` | Move Silero fields into `config.silero_vad`; set `sample_rate`, `num_threads`, `provider` on the outer config |
| `SileroVadConfig { model, min_silence_duration, min_speech_duration, max_speech_duration, threshold, sample_rate, window_size, provider, num_threads, debug }` | `VadModelConfig { silero_vad: SileroVadModelConfig { model, threshold, min_silence_duration, min_speech_duration, window_size, max_speech_duration }, sample_rate, num_threads, provider, debug, .. }` | `window_size = 512`, `max_speech_duration = 20.0` |
| `SileroVad::front() -> SpeechSegment { start, samples: Vec<f32> }` | `VoiceActivityDetector::front() -> Option<SpeechSegment>` with `start()`, `n()`, `samples()` | Copy samples before `pop`; the wrapper owns a native copy and frees it on `Drop` |
| `SileroVad::accept_waveform(Vec<f32>)` | `accept_waveform(&[f32])` | No ownership transfer |
| `SileroVad::is_empty(&mut self)` | `is_empty(&self)` | No mutable borrow needed |
| `WhisperConfig { encoder, decoder, tokens, language, bpe_vocab, tail_paddings, provider, num_threads, debug }` | `OfflineRecognizerConfig { model_config: OfflineModelConfig { whisper: OfflineWhisperModelConfig { .. }, tokens, provider, num_threads, .. }, decoding_method, .. }` | Set `tail_paddings: -1`, `decoding_method: "greedy_search"` |
| `WhisperRecognizer::new(config)` | `OfflineRecognizer::create(&config)` | Returns `Option`, not `Result` |
| `recognizer.transcribe(sr, samples) -> result` | `create_stream` + `accept_waveform(sr, samples)` + `decode(&stream)` + `get_result()` | Rust method is `decode`, not `decode_stream` |
| `Diarize::new(seg, emb, DiarizeConfig { num_clusters, threshold, min_duration_on, min_duration_off, provider, debug })` | `OfflineSpeakerDiarization::create(&OfflineSpeakerDiarizationConfig { segmentation, embedding, clustering, min_duration_on, min_duration_off })` | Use `-1`, `0.5`, `0.3`, `0.5`; current 0.0/0.0 is wrong |
| `Diarize::compute(samples, Some(callback))` | `process(&samples)` or the vendored `process_with_callback` | Official safe API has no callback |
| `sherpa_rs::diarize::Segment { start, end, speaker }` | `OfflineSpeakerDiarizationSegment { start, end, speaker, confidence }` | `start`/`end` are seconds |
| `sherpa-rs` downloads/builds its own native lib | `sherpa-onnx-sys` prebuilt archive with `SHERPA_ONNX_ARCHIVE_DIR` | Pin 1.13.8 and the archive name |

The migration is not just a rename. Two current defaults are wrong relative to Python:

1. `WhisperConfig.tail_paddings: None` → `0`, but Python uses `-1`.
2. `DiarizeConfig.min_duration_on/off: 0.0/0.0`, but Python uses the library defaults `0.3/0.5`.

Fix both and add parity tests that compare against the Python output on the same WAV.

### 8.8 Model inventory and size budget

| Model | Path in runtime | Bytes | Used by |
|---|---|---:|---|
| Whisper turbo encoder INT8 | `models/whisper/turbo-encoder.int8.onnx` | 674,716,297 | Live, sherpa fallback |
| Whisper turbo decoder INT8 | `models/whisper/turbo-decoder.int8.onnx` | 361,080,764 | Live, sherpa fallback |
| Whisper turbo tokens | `models/whisper/turbo-tokens.txt` | 816,730 | Live, sherpa fallback |
| Silero VAD | `models/silero_vad.onnx` | 643,854 | File/live VAD |
| pyannote segmentation-3.0 | `models/diarization/segmentation.onnx` | 5,992,913 | Diarization |
| WeSpeaker ResNet34-LM | `models/diarization/embedding.onnx` | 26,530,550 | Diarization |
| whisper.cpp GGUF q5_0 | `whispercpp/ggml-large-v3-turbo-q5_0.bin` | 574,041,195 | File transcription |
| whisper-cli + DLLs | `whispercpp/whisper-cli.exe`, `*.dll` | ~50 MB | File transcription |
| ffmpeg + ffprobe | `ffmpeg/*.exe` | ~150 MB | Decode |

Totals:

- ONNX Whisper + VAD + diarization: **1,069,781,108 B** (1.07 GB).
- Add GGUF: **1,643,822,303 B** (1.64 GB).
- Add ffmpeg/whisper-cli/licenses: roughly **1.85–2.0 GB** uncompressed.
- The packed exe must stay under GitHub's 2 GiB per-asset limit (2,147,483,648 B). Measure the final signed artifact; do not estimate from the uncompressed total.

`whisper-cli` and its DLLs are only needed for the file engine. The sherpa ONNX models are needed for live mode and are the fallback. Diarization models are optional for startup but required for the “Find speakers” checkbox.

### 8.9 Engine events and Rune integration

All three paths emit the same event type; the Rune UI never sees a native handle:

```rust
pub enum EngineEvent {
    Status(Status),
    Progress(f32),
    Log(String),
    LiveSegment(String),
    LiveError(String),
    Overflow,
    Finished,
    Failed(String),
}

#[derive(Clone)]
pub struct EventSink {
    tx: std::sync::mpsc::Sender<EngineEvent>,
    waker: rune::Waker,
}

impl EventSink {
    pub fn send(&self, event: EngineEvent) {
        let _ = self.tx.send(event);
        (self.waker)();
    }

    pub fn status(&self, status: Status) { self.send(EngineEvent::Status(status)); }
    pub fn progress(&self, value: f32) { self.send(EngineEvent::Progress(value.clamp(0.0, 1.0))); }
    pub fn log(&self, line: &str) { self.send(EngineEvent::Log(line.to_owned())); }
    pub fn live_segment(&self, text: String) { self.send(EngineEvent::LiveSegment(text)); }
    pub fn live_error(&self, message: &str) { self.send(EngineEvent::LiveError(message.to_owned())); }
    pub fn overflow(&self) { self.send(EngineEvent::Overflow); }
}
```

- `Waker` is `Arc<dyn Fn() + Send + Sync>`. Calling it wakes Rune's `poll`; it does not run UI code on the worker thread.
- The UI drains the receiver in `poll` and returns `true` only when visible state changed.
- `LiveError` is non-terminal: the decoder stops, but the recording and WAV continue. `Failed` is reserved for a file/live job that cannot produce its result.
- Throttle `Progress` at the source or in `poll`; do not call `AppHandle::update` for every 512-sample chunk or every whisper progress line.
- The worker owns the `CancelToken`; the UI sets it. The worker checks it at the documented points and between segments.
- Extraction starts from `setup`/`begin_prepare` (or a guarded `on_ready`), not inference. Inference starts from a user action, after the runtime is ready. This avoids a multi-second model load on the first frame.

Suggested job entry points:

```rust
pub fn run_file_job(
    request: FileJob,
    cancel: CancelToken,
    sink: EventSink,
) -> Result<()>;

pub fn run_live_session(
    request: LiveJob,
    cancel: CancelToken,
    sink: EventSink,
) -> Result<LiveSummary>;

pub fn run_diarize_job(
    wav: PathBuf,
    cancel: CancelToken,
    sink: EventSink,
) -> Result<Vec<Turn>>;
```

`EngineEvent::Finished` is sent exactly once, after the transcript or WAV is written and the final progress is 1.0. `Failed` is sent exactly once on error. Cancellation sends neither; the UI's cancel path sets the final `Cancelled` status.

### 8.10 Parity checklist

- [ ] `-bs 5`, `-t` 4–8, `-pp`, `--output-json`, `--output-file`, exact arg order.
- [ ] GPU probe is cached once; `-dev` omitted when no device; no `-dev 0` default.
- [ ] JSON offsets are milliseconds → seconds; blank text skipped; text trimmed.
- [ ] `tail_paddings = -1` in the sherpa recognizer.
- [ ] VAD: threshold 0.5, min silence 0.5, min speech 0.25, window 512, max speech 20.0, file buffer 60, live buffer 30.
- [ ] Final VAD window zero-padded to 512; `flush()` drained.
- [ ] `PAD_SAMPLES = 3200`; file pads both sides; live pads leading only; `prev_end` uses the original end.
- [ ] Live history 11,392 samples; mic queue 1875 drop-newest; decode queue 8 drop-oldest.
- [ ] VAD and recognizer never call ONNX Runtime concurrently.
- [ ] Diarization `num_clusters = -1`, `threshold = 0.5`, `min_duration_on = 0.3`, `min_duration_off = 0.5`.
- [ ] Diarization child is killable; progress is 0–100; cancellation is `Speaker identification cancelled by user`.
- [ ] File diarization failure falls back to plain; “Find speakers” failure surfaces.
- [ ] No transcript text is written to the log file; only the UI log gets raw segment text.
- [ ] `Finished`/`Failed` exactly once; late live events cannot mutate the next session.

---

## 9. FFmpeg versus Rust decoding

### 9.1 What must be decoded

The file picker accepts exactly these extensions (`src/config.py:127-132`, `REFERENCE_UI.md`):

```text
.mp4 .mkv .mov .avi .webm .mp3 .wav .m4a .flac .ogg .opus .aac
```

Typical audio codecs inside them: AAC-LC/HE-AAC, MP3, Opus, Vorbis, FLAC, ALAC, PCM, and occasionally WMA/AC-3 in old files. Video containers add H.264/H.265/VP8/VP9/AV1 video tracks that must be ignored, not decoded. The extractor does not enforce the extension list; ffmpeg decides. A file with a `.mp4` extension but an unusual codec must still work if ffmpeg supports it.

The output contract is narrow: **mono, 16 kHz, signed 16-bit little-endian PCM WAV**. Everything downstream (VAD, Whisper ONNX, diarization, whisper-cli) consumes that. This makes the decoder choice a pure coverage/robustness decision, not an audio-quality one.

### 9.2 Candidate comparison

| Criterion | bundled `ffmpeg.exe` / `ffprobe.exe` | `ffmpeg-next` 9.0.0 in-process | `symphonia` 0.6.1 pure Rust | Windows Media Foundation |
|---|---|---|---|---|
| Container coverage | All 12 extensions; AVI, MKV/WebM, MP4/MOV, OGG, raw AAC/ADTS | Same as the linked FFmpeg build | MP4/MKV/OGG/WAV/AIFF/CAF; **no AVI**, no raw ADTS | Depends on installed codecs; no OGG/Opus/FLAC guarantee |
| Codec coverage | AAC, MP3, Opus, Vorbis, FLAC, ALAC, PCM, WMA, AC-3, … | Same as FFmpeg | AAC, MP3, Vorbis, FLAC, ALAC, PCM; **no Opus**, no WMA | AAC, MP3, WMA, some; no Opus/Vorbis/FLAC by default |
| Build requirements | none; prebuilt exe is in the payload | FFmpeg **dev** libraries/import libs at build; FFmpeg DLLs at runtime, or a full static FFmpeg build | pure Rust | Windows SDK |
| Runtime dependencies | `ffmpeg.exe`, `ffprobe.exe` (~150 MB) | `avcodec-*.dll`, `avformat-*.dll`, `avutil-*.dll`, `swresample-*.dll` (~60–100 MB) + CRT | none | system |
| Crash isolation | process boundary; a malformed file kills ffmpeg, not the app | none; a decoder bug is an access violation in the UI process | none | none |
| Cancellation | kill the child; immediate | must unwind/abort the decode loop; some codecs block in native code | cooperative loop | COM call, no clean cancel mid-file |
| Progress | parse `-stats` `time=` from stderr | must instrument packet timestamps | must instrument | must instrument |
| Parity with v3.2.3 | exact same binary/args | close, but a different FFmpeg build/version can change timestamps | different decoder, different PCM | different decoder |
| License | GPLv3 for the Gyan essentials build | depends on the linked FFmpeg build | MPL-2.0 | proprietary system component |
| Unsafe surface | none in-process | large C API surface | small, safe Rust | large COM surface |

### 9.3 Why Symphonia is not enough

Verified from `symphonia-0.6.1/Cargo.toml` and its README:

- Default features are `mkv`, `ogg`, `wav`, `pcm`, `vorbis`, `flac`, `adpcm`, `all-meta`, `opt-simd`. `mp3`, `aac`, `isomp4`, and `alac` are opt-in; `all` enables them.
- There is **no Opus decoder** in 0.6.1. The README lists `symphonia-adapter-libopus`, which requires a native `libopus` build. `.opus` and Opus-in-`.webm`/`.mkv`/`.ogg` would fail without that native dependency.
- There is **no AVI demuxer**. `.avi` is one of the twelve accepted extensions.
- There is **no WMA decoder**.
- There is **no raw ADTS demuxer** for `.aac`; the AAC decoder handles AAC-LC inside MP4, not a bare `.aac` stream.
- `.mov` is not explicitly listed; the ISO/MP4 demuxer may handle some QuickTime files, but it is not a guaranteed parity path.
- `all` still does not add Opus or AVI. You would need an Opus adapter, an AVI demuxer, and WMA support to match ffmpeg.

A pure-Rust path could be built from Symphonia + `libopus` + a custom AVI demuxer + WMA, but that is a research project with a new native dependency, not a rewrite task. It also changes the decoded PCM, which can change VAD boundaries and therefore transcript segments.

### 9.4 Why `ffmpeg-next` is not enough

`ffmpeg-next` 9.0.0 is a maintenance-mode wrapper for FFmpeg 3.4–8.0. It is not a decoder; it is a binding to the same FFmpeg libraries. Choosing it means:

- Install FFmpeg development headers/import libraries in CI, or enable the `build` feature and compile FFmpeg from source on every platform.
- Ship the FFmpeg DLL set in the payload and make the loader find it before the first call (or link statically, which brings LGPL/GPL configuration questions and a large build).
- Move decoding into the UI process, where a malformed file or a decoder bug is an access violation, not a child-process exit.
- Write unsafe `AVFormatContext`/`AVCodecContext`/`SwrContext` setup and teardown, or rely on the wrapper's safe layer and still own the error/cancel path.
- Reimplement progress and cancellation around `av_read_frame`, including codecs that block.

The only thing it saves is spawning a process. That is not worth the build and crash surface when `ffmpeg.exe` is already required for parity and is only ~150 MB in a ~1.9 GB payload.

### 9.5 Decision

**Keep the bundled `ffmpeg.exe` and `ffprobe.exe`.** Rationale:

1. **Parity:** the current app's behavior is defined by these binaries and args. A different decoder can change VAD boundaries, timestamps, and transcripts.
2. **Coverage:** all twelve extensions, including `.avi` and `.opus`.
3. **Isolation:** a malformed media file can crash ffmpeg without taking down the Rune UI.
4. **Cancellation:** `Child::kill()` is immediate and reliable.
5. **Build simplicity:** no FFmpeg dev libraries, no C toolchain, no DLL search-path work.
6. **Size:** ~150 MB is acceptable relative to 1.64 GB of models; it is not the dominant cost.

Do not add `symphonia` as a direct dependency. Rodio already pulls the WAV decoder for speaker-clip playback; that is the only Rust decoding needed.

### 9.6 Exact wrapper behavior

#### 9.6.1 Validation

```rust
pub fn validate_ffmpeg(paths: &Paths) -> Result<()> {
    let mut missing = Vec::new();
    if !paths.ffmpeg_bin.exists() {
        missing.push(format!("ffmpeg.exe not found at: {}", paths.ffmpeg_bin.display()));
    }
    if !paths.ffprobe_bin.exists() {
        missing.push(format!("ffprobe.exe not found at: {}", paths.ffprobe_bin.display()));
    }
    if missing.is_empty() {
        Ok(())
    } else {
        Err(Error::FfmpegNotFound(missing.join("\n")))
    }
}
```

The reference uses `exists()`, not `is_file()`; a directory with the right name passes validation and then fails at spawn. Preserve that only if exact behavior matters; `is_file()` is safer and produces a clearer error. Document the deviation if changed.

#### 9.6.2 Fast path for already-suitable WAV

```rust
pub fn is_already_suitable_wav(paths: &Paths, input: &Path) -> bool {
    if !input.extension().is_some_and(|ext| ext.eq_ignore_ascii_case("wav")) {
        return false;
    }
    let Some(info) = probe_audio(paths, input) else { return false };
    info.codec_name.as_deref() == Some("pcm_s16le")
        && info.sample_rate == Some(16_000)
        && info.channels == Some(1)
}
```

- Case-insensitive `.wav`; `.wave` does not match.
- Requires the exact ffprobe codec string `pcm_s16le`.
- Missing fields fail.
- On success, copy the file directly to a UUID-named temp WAV and return it. Log:
  ```text
  [ffmpeg] Already mono 16 kHz PCM WAV — copying directly: <name>
  ```
- The copy is not a decode; it preserves the bytes and is much faster for the common “already prepared WAV” case.
- Wrap copy errors as `Error::FfmpegExtraction`, not a raw `io::Error`. The Python fast path lets `shutil.copy2` errors escape raw; that is a defect.

#### 9.6.3 Probe

```rust
#[derive(Debug, Clone, Default)]
pub struct StreamInfo {
    pub codec_name: Option<String>,
    pub sample_rate: Option<u32>,
    pub channels: Option<u32>,
    pub duration: Option<String>,
}

pub fn probe_audio(paths: &Paths, input: &Path) -> Option<StreamInfo> {
    let mut cmd = std::process::Command::new(&paths.ffprobe_bin);
    cmd.args([
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_name,sample_rate,channels,duration",
            "-of", "json",
            "-protocol_whitelist", "file",
        ])
        .arg(input)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    let output = cmd.output().ok()?;

    let value: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
    let stream = value.get("streams")?.as_array()?.first()?;
    Some(StreamInfo {
        codec_name: stream.get("codec_name").and_then(|v| v.as_str()).map(str::to_owned),
        sample_rate: stream.get("sample_rate").and_then(|v| {
            v.as_u64().or_else(|| v.as_str().and_then(|s| s.parse().ok()))
        }).map(|v| v as u32),
        channels: stream.get("channels").and_then(|v| {
            v.as_u64().or_else(|| v.as_str().and_then(|s| s.parse().ok()))
        }).map(|v| v as u32),
        duration: stream.get("duration").and_then(|v| {
            v.as_str().map(str::to_owned).or_else(|| v.as_f64().map(|f| f.to_string()))
        }),
    })
}
```

- `-protocol_whitelist file` is after `-of json` and before the positional input, exactly as the reference.
- `ffprobe` missing, timeout, bad JSON, no audio stream, or permission errors all return `None`. Log at debug:
  ```text
  ffprobe failed for <name>: <error>
  ```
- `Command::output()` has no timeout. The reference uses 30 s. If exact parity matters, run the child on a helper thread and kill it after 30 s; do not let a corrupt network-path file block the UI. The probe is always on a worker thread, so this is a responsiveness issue, not a UI freeze.
- `duration` may be a JSON string or number; accept both.

#### 9.6.4 Exact argv

```rust
pub fn ffmpeg_command(paths: &Paths, input: &Path, output: &Path) -> std::process::Command {
    let mut cmd = std::process::Command::new(&paths.ffmpeg_bin);
    cmd.arg("-y")
        .arg("-protocol_whitelist").arg("file")
        .arg("-i").arg(input)
        .arg("-ac").arg("1")
        .arg("-ar").arg("16000")
        .arg("-vn")
        .arg("-acodec").arg("pcm_s16le")
        .arg("-stats")
        .arg(output)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}
```

- Each token is a separate `arg`; never build a shell string. Paths with spaces, quotes, `&`, or Unicode must pass through unchanged.
- `-y` overwrites the UUID temp file; collisions are effectively impossible.
- `-protocol_whitelist file` must appear before `-i`. It is the offline guard: ffmpeg may open only local files, not `http`, `tcp`, `udp`, `rtmp`, etc.
- `-vn` drops video; `-acodec pcm_s16le` is placed after `-vn`.
- `-stats` writes progress to **stderr** using `\r`. This is why the reader must split on CR and LF (§9.6.5).
- `stdout` is `DEVNULL`; ffmpeg writes the WAV to the file, not stdout.

#### 9.6.5 The CR/LF reader

`BufRead::lines()` only splits on `\n`. ffmpeg's `-stats` uses `\r`, so `lines()` would merge all progress updates into one giant line. Implement the reference's `_iter_lines` semantics:

```rust
fn next_ffmpeg_line<R: std::io::BufRead>(
    reader: &mut R,
    buf: &mut Vec<u8>,
) -> std::io::Result<Option<String>> {
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            if buf.is_empty() {
                return Ok(None);
            }
            let line = String::from_utf8_lossy(buf).trim_end().to_owned();
            buf.clear();
            return Ok(Some(line));
        }
        if let Some(pos) = available.iter().position(|&b| b == b'\r' || b == b'\n') {
            buf.extend_from_slice(&available[..pos]);
            let was_cr = available[pos] == b'\r';
            reader.consume(pos + 1);

            // Consume the LF of a CRLF pair so it does not yield an empty line.
            if was_cr {
                let next = reader.fill_buf()?;
                if next.first() == Some(&b'\n') {
                    reader.consume(1);
                }
            }

            let line = String::from_utf8_lossy(buf).trim_end().to_owned();
            buf.clear();
            return Ok(Some(line));
        }
        let len = available.len();
        buf.extend_from_slice(available);
        reader.consume(len);
    }
}
```

- CRLF is one terminator, not two.
- A lone `\r` or `\n` terminates a line.
- At EOF, the remaining buffer is returned even without a terminator.
- UTF-8 is decoded lossily, like Python's `errors="replace"`.
- `.trim_end()` matches Python's `.rstrip()` for ASCII; it also removes trailing Unicode whitespace, which is harmless.

#### 9.6.6 Progress and logs

Regexes:

```rust
static DURATION: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"Duration:\s*(\d+):(\d+):([\d.]+)").unwrap());
static TIME: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"time=(\d+):(\d+):([\d.]+)").unwrap());
```

- The first positive duration is recorded; later ones are ignored.
- `time=` returns seconds, or `-1.0` if absent.
- A malformed numeric field must return `0.0`/`-1.0` and log a warning. The Python code calls `float()` and can kill the reader thread; do not copy that defect.
- Progress logs are throttled to one per second:
  ```text
  [ffmpeg] Extracting… 00:01:23 / 00:12:00  (12%)
  ```
  Two spaces before `(`, as in the reference. The percentage is `min(pos / duration * 100, 100)`.
- Lines containing `error`, `warn`, `invalid`, `no such`, or `stream #` (case-insensitive) are logged as `[ffmpeg] <line>`, except lines starting with `ffmpeg version` or containing `encoder` (case-insensitive).
- Numeric progress is not sent to the UI's progress bar in the reference; the status stays “Extracting Audio”. The log callback gets the text. Preserve that, or add a determinate bar as a deliberate UX change.

#### 9.6.7 Cancellation and completion

```rust
loop {
    if let Some(status) = child.try_wait()? {
        break status;
    }
    if cancel.is_cancelled() {
        let _ = child.kill();
        let _ = child.wait();
        let _ = std::fs::remove_file(&temp_wav);
        return Err(Error::Cancelled);
    }
    drain_ffmpeg_lines(&line_rx, &mut state, &sink);
    std::thread::sleep(std::time::Duration::from_millis(100));
}
```

- Cancellation latency is up to 100 ms plus process teardown.
- On cancel, delete the temp WAV. The Python code calls `terminate()`, waits 5 s, then `kill()`; Rust's `kill()` is the forced path. ffmpeg does not need graceful shutdown.
- On success, join the reader, drain remaining lines, verify the output exists and is non-empty, and log:
  ```text
  [ffmpeg] Extraction complete — {size_kb} KB written to {name}
  ```
  where `{size_kb}` is formatted with comma thousands separators by a small helper. Rust's `format!` has no Python-style `:,` grouping; do not write `{size_kb:,}` in Rust.
- On nonzero exit, keep the last 800 characters of stderr:
  ```text
  ffmpeg exited with code {code}.\n\n{snippet}
  ```
- Any unexpected error:
  ```text
  Unexpected error during audio extraction: {error}
  ```
- The caller owns the returned temp WAV and deletes it in `finally` after the file's transcription/diarization.

#### 9.6.8 Temp file lifecycle

- The temp name is `paths.temp_dir.join(format!("{}.wav", Uuid::new_v4().simple()))`.
- Startup cleanup removes top-level `temp/*.wav` and `temp/*.json`; it does not recurse and does not delete subdirectories. Keep that contract.
- A crash can leave a temp WAV; the next startup cleanup removes it.
- The fast-path copy is also a temp WAV and is removed by the same cleanup if the process crashes.

### 9.7 Offline and security properties

- `-protocol_whitelist file` on both `ffmpeg` and `ffprobe`. CI should grep the built argv for exactly two uses, matching `tests/test_offline_guard.py`.
- Never accept a URL from the file picker. `PathBuf` only; reject `http://`, `https://`, `file://` strings if a path is ever parsed from text.
- No shell. `Command::arg` passes each token separately; shell metacharacters in filenames are inert.
- Keep ffmpeg's working directory at the app data dir, not the user's current directory, so a relative `-i` cannot resolve unexpectedly.
- Do not add `-i` before `-protocol_whitelist`; option order matters.
- The temp directory is per-user under `%LOCALAPPDATA%`; no world-writable `%TEMP%` path.
- ffmpeg writes only the temp WAV. It does not write next to the source file.

### 9.8 Licensing

The reference uses the Gyan `essentials_build` of FFmpeg 7.1.1, which is **GPLv3**. Shipping it in the payload means:

- Include the FFmpeg license text and the exact source/build information in `THIRD-PARTY-NOTICES.txt` and `licenses/`.
- Provide a written offer or a durable link to the corresponding source for the shipped build, as GPLv3 requires. The release page can host the source archive or a link to the exact upstream tag plus the build configuration.
- Do not statically link FFmpeg into the app or otherwise combine it into a single work without legal review. The current design keeps it as a separate executable, which is the simplest posture.
- If legal requires LGPL only, replace the ffmpeg build with an LGPL-configured build and re-test all twelve extensions. Codec coverage may shrink (notably some encoders, not decoders); do not assume the GPL build's coverage carries over.
- `ffprobe` is part of the same build and the same license.

This is a product/legal decision, not a technical one. Record the chosen build and its license in the release checklist.

### 9.9 Tests

1. **CR/LF parser:** feed `b"frame=1\rframe=2\r\nframe=3\n"` and assert three lines; split the input at every byte boundary and assert identical output.
2. **Duration/time parser:** valid, missing, malformed (`00:00:1.2.3`) → `0.0`/`-1.0`, no panic.
3. **Fast path:** a 16 kHz mono `pcm_s16le` WAV is copied byte-for-byte; a 44.1 kHz stereo WAV is not.
4. **Probe:** fake ffprobe JSON with string and numeric fields; missing streams → `None`.
5. **Argv:** exact order and exactly one `-protocol_whitelist file`; no shell string.
6. **Cancel:** fake ffmpeg that sleeps; cancel within 250 ms; temp file deleted.
7. **Nonzero exit:** fake ffmpeg writes an error and exits 1; message contains the code and last 800 chars.
8. **Paths:** spaces, Unicode, leading `-`, and a 260+ character path all pass as single argv entries.
9. **End-to-end:** the same input file produces the same WAV format (mono/16k/s16le) as the Python wrapper, and the same duration within one sample.

---

## 10. Single-file Windows `.exe`

### 10.1 Artifact shape

```
+--------------------------------------------------+  0
| PE image: headers, .text, .rdata, resources      |
+--------------------------------------------------+  payload_offset
| ZIP container: manifest.json, models, tools      |
+--------------------------------------------------+  payload_offset + payload_size
| 128-byte footer                                  |
+--------------------------------------------------+  unsigned EOF
| 0..7 bytes alignment padding                     |
+--------------------------------------------------+  cert_offset
| WIN_CERTIFICATE table (Authenticode)             |
+--------------------------------------------------+  EOF
```

The executable is a normal Windows PE x86-64 GUI binary. Windows ignores the overlay. The app opens its own file, reads the footer, verifies the payload, and extracts once into a per-user cache. The cache is derived data; deleting it recreates it. The exe can be moved or copied; the cache key follows the payload hash.

Build order is mandatory:

```text
cargo build --release
  → copy easyscribe.exe
  → append ZIP container
  → append 128-byte footer
  → Authenticode sign the final file
```

Never sign before packing. The signature must cover the footer and payload. Never modify the exe after signing.

### 10.2 Footer layout

128 bytes, little-endian, at `logical_end - 128 - pad`:

| Offset | Size | Field | Notes |
|---|---:|---|---|
| `0x00` | 8 | magic | ASCII `ESCFTR01` |
| `0x08` | 2 | footer version | `1` |
| `0x0A` | 2 | flags | bit 0: per-entry zstd (future); bit 1: signed build (informational); rest reserved |
| `0x0C` | 4 | footer size | `128` |
| `0x10` | 8 | payload offset | absolute file offset of the container |
| `0x18` | 8 | payload size | container length, excluding footer/signature |
| `0x20` | 8 | manifest offset | **relative to payload start** |
| `0x28` | 8 | manifest size | uncompressed JSON length |
| `0x30` | 32 | payload SHA-256 | hash of `[payload_offset, payload_offset + payload_size)` |
| `0x50` | 16 | app version | UTF-8, NUL-padded (`3.2.3`) |
| `0x60` | 16 | build id | UTF-8, NUL-padded; at most 16 bytes |
| `0x70` | 4 | CRC32 | CRC-32 over bytes `0x00..0x6F` (112 bytes) |
| `0x74` | 4 | reserved | zero |
| `0x78` | 8 | trailer magic | ASCII `ESCEOF01` |

Two fields need discipline:

- **`manifest_offset` is relative to the payload start**, not the file. Both `pack` (`crates/easyscribe-core/src/payload.rs:710`) and `pack_streaming` (`:762`) currently store the relative offset returned by their container inspector; keep that invariant and lock it with a test:
  ```rust
  assert!(footer.manifest_offset < footer.payload_size);
  ```
- **`build_id` is 16 bytes.** The CLI's default `"{version}-{unix_seconds}"` is exactly 16 bytes for `3.2.3-1762345678`, but a prerelease version overflows and is silently truncated by `copy_padded`. Define the field as 16 bytes and either truncate deliberately (documented) or use a 16-character git SHA. Do not pass a 32-character UUID hex string.

`flags` bit 1 (“signed build”) must be set **before** signing, because changing the footer afterward invalidates the signature. Have the packer accept `--signed` and set the bit, then sign; if the build is unsigned, leave it 0. Treat the bit as informational, not as proof of a valid signature.

`Footer::from_bytes` must also reject:

- `version != 1`;
- `footer_size != 128`;
- `payload_size == 0`;
- `manifest_size == 0`;
- `manifest_offset >= payload_size`;
- CRC mismatch;
- a `payload_offset` that is not beyond the PE headers (checked by the caller).

### 10.3 The Authenticode problem

`REFERENCE_PACKAGING.md` §6.7.2 says:

```text
payload_offset + payload_size + 128 == file_len
```

That is true only **before signing**. Authenticode appends an Attribute Certificate Table (`WIN_CERTIFICATE`) after the payload/footer. The PE optional header's Security directory entry (data directory index 4) points at it. The table is outside the mapped image and is not included in the image's section table. After signing:

```text
payload_offset + payload_size + 128 + pad == cert_offset
cert_offset + cert_size == file_len
```

`pad` is 0–7 bytes so the certificate table starts on an 8-byte boundary. The footer is no longer at `file_len - 128`. The scaffold no longer assumes that: `read_footer` scans the final 1 MiB backward for `ESCFTR01` and accepts a footer when `payload_offset + payload_size + 128` equals the footer's end (`payload.rs:138-178`), so a normal Authenticode certificate after 0–7 padding bytes is found. That heuristic is still not robust:
- a certificate table larger than 1 MiB hides the footer entirely;
- the scanner does not read the PE Security directory or validate `SizeOfHeaders`, so it cannot distinguish “certificate table” from arbitrary overlay bytes;
- a random `ESCFTR01` inside a large certificate can be tried before the real footer (CRC/end checks reject it, but the search is still heuristic).

Parse the Security directory, compute `logical_end`, and scan only the 0–7 alignment bytes. That is the spec-correct reader; keep the 1 MiB tail scan only as a diagnostic fallback for nonstandard overlays.

#### 10.3.1 PE layout parsing

```rust
const PE_SECURITY_DIR_INDEX: usize = 4;

struct PeLayout {
    headers_end: u64,   // SizeOfHeaders
    cert_offset: u64,   // file offset, not RVA
    cert_size: u32,
}

fn pe_layout(file: &mut File, len: u64) -> Result<Option<PeLayout>> {
    if len < 0x40 {
        return Ok(None);
    }
    let mut dos = [0u8; 0x40];
    file.seek(SeekFrom::Start(0))?;
    file.read_exact(&mut dos)?;
    if &dos[0..2] != b"MZ" {
        return Ok(None);
    }
    let pe = u32::from_le_bytes(dos[0x3C..0x40].try_into().unwrap()) as u64;
    if pe.checked_add(24).is_none_or(|end| end > len) {
        return Err(damaged());
    }

    let mut sig = [0u8; 4];
    file.seek(SeekFrom::Start(pe))?;
    file.read_exact(&mut sig)?;
    if &sig != b"PE\0\0" {
        return Ok(None);
    }

    let mut coff = [0u8; 20];
    file.seek(SeekFrom::Start(pe + 4))?;
    file.read_exact(&mut coff)?;
    let opt_size = u16::from_le_bytes(coff[16..18].try_into().unwrap()) as u64;
    let opt = pe + 24;
    if opt.checked_add(opt_size).is_none_or(|end| end > len) {
        return Err(damaged());
    }

    let mut magic = [0u8; 2];
    file.seek(SeekFrom::Start(opt))?;
    file.read_exact(&mut magic)?;
    let (num_dirs_off, dirs_off) = match u16::from_le_bytes(magic) {
        0x10B => (92u64, 96u64),   // PE32
        0x20B => (108u64, 112u64), // PE32+
        _ => return Ok(None),
    };
    if opt_size < dirs_off {
        return Err(damaged());
    }

    let headers_end = u64::from(read_u32_at(file, opt + 60)?);

    let num_dirs = read_u32_at(file, opt + num_dirs_off)?;
    if num_dirs <= PE_SECURITY_DIR_INDEX as u32 {
        return Ok(Some(PeLayout { headers_end, cert_offset: 0, cert_size: 0 }));
    }
    if opt_size < dirs_off + (PE_SECURITY_DIR_INDEX as u64 + 1) * 8 {
        return Err(damaged());
    }

    let dir = opt + dirs_off + (PE_SECURITY_DIR_INDEX as u64) * 8;
    let cert_offset = u64::from(read_u32_at(file, dir)?);
    let cert_size = read_u32_at(file, dir + 4)?;
    Ok(Some(PeLayout { headers_end, cert_offset, cert_size }))
}

fn read_u32_at(file: &mut File, offset: u64) -> std::io::Result<u32> {
    let mut bytes = [0u8; 4];
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}
```

Verified offsets:

- `e_lfanew` is at DOS offset `0x3C`.
- `SizeOfOptionalHeader` is at COFF offset 16.
- Optional header `SizeOfHeaders` is at offset 60 for both PE32 and PE32+.
- `NumberOfRvaAndSizes` is at optional offset 92 (PE32) / 108 (PE32+).
- Data directories start at optional offset 96 (PE32) / 112 (PE32+).
- The Security entry is index 4, 8 bytes: `VirtualAddress` (a **file offset** for this directory), `Size`.

Check `cert_offset + cert_size == file_len` when signed. If not, the file has trailing junk or a truncated certificate table; treat it as damaged.

#### 10.3.2 Footer lookup

```rust
pub fn read_footer(exe: &Path) -> Result<Option<Footer>> {
    let mut file = File::open(exe)?;
    let len = file.metadata()?.len();
    if len < FOOTER_SIZE as u64 {
        return Ok(None);
    }
    let Some(pe) = pe_layout(&mut file, len)? else {
        return Ok(None); // not a PE; dev checkout or non-Windows test binary
    };

    let signed = pe.cert_size > 0;
    let logical_end = if signed {
        if pe.cert_offset.checked_add(u64::from(pe.cert_size)) != Some(len) {
            return Err(damaged());
        }
        pe.cert_offset
    } else {
        len
    };

    let pads: &[u64] = if signed { &[0, 1, 2, 3, 4, 5, 6, 7] } else { &[0] };
    for &pad in pads {
        let Some(start) = logical_end
            .checked_sub(pad)
            .and_then(|end| end.checked_sub(FOOTER_SIZE as u64))
        else {
            continue;
        };
        let mut bytes = [0u8; FOOTER_SIZE];
        file.seek(SeekFrom::Start(start))?;
        file.read_exact(&mut bytes)?;
        if &bytes[0..8] != FOOTER_MAGIC || &bytes[120..128] != TRAILER_MAGIC {
            continue;
        }

        // Magic matched, so a CRC/version failure is damage, not "no footer".
        let footer = Footer::from_bytes(&bytes)?;
        if footer.payload_offset < pe.headers_end {
            continue;
        }
        if footer.payload_offset
            .checked_add(footer.payload_size)
            .and_then(|end| end.checked_add(FOOTER_SIZE as u64))
            .and_then(|end| end.checked_add(pad))
            != Some(logical_end)
        {
            continue;
        }
        return Ok(Some(footer));
    }

    if signed {
        Err(damaged())
    } else {
        Ok(None)
    }
}
```

- For an unsigned dev build, only `pad = 0` is accepted; a footer must end exactly at EOF.
- For a signed build, scan 0–7 bytes backward. The arithmetic check prevents accepting a random `ESCEOF01` elsewhere.
- If magic is found but `Footer::from_bytes` fails CRC/version, return the “damaged” error immediately. Do not keep scanning.
- `payload_offset >= pe.headers_end` rejects a footer that claims the PE headers are payload.
- `checked_add` prevents integer overflow from a malicious footer.

#### 10.3.3 Signing the final bytes

```bat
signtool sign /fd SHA256 /td SHA256 /tr http://timestamp.digicert.com /a ^
  dist\EasyScribe-v3.2.3.exe

signtool verify /pa /v dist\EasyScribe-v3.2.3.exe
```

- `/fd SHA256` selects the file digest.
- `/tr` + `/td SHA256` request an RFC 3161 timestamp with SHA-256. Without a timestamp, the signature becomes invalid when the certificate expires.
- Sign **after** packing. The certificate table goes after the footer; the parser above handles it.
- `signtool` may update the PE checksum; that field is outside the footer, so the payload hash is unaffected.
- The signature covers the PE image, payload, and footer. The certificate table itself is excluded from the digest, as required by Authenticode.
- If the certificate is an EV certificate, the SmartScreen reputation path is faster, but normal OV signing plus reputation works over time.
- Signing a >1.5 GB file takes time; give CI a generous timeout.

Optional runtime verification: call `WinVerifyTrust` before extraction and show a warning if the signature is missing/invalid. The app’s primary integrity check is the payload SHA-256; signature verification protects against a tampered exe being re-signed by a different publisher only if the app pins the expected publisher. That is a product decision, not required for parity.

### 10.4 Container: ZIP v1 versus ESPAK001

The rewrite keeps the **ZIP container for v1**. This is a deliberate deviation from `REFERENCE_PACKAGING.md` §6.7.3, which specifies ESPAK001 as the long-term format.

| Property | ZIP v1 (current) | ESPAK001 (target) |
|---|---|---|
| Magic | `PK\x03\x04` local header / `PK\x05\x06` EOCD | `ESPAK001` |
| Offset width | 32-bit, ZIP64 extensions | 64-bit native |
| Index location | central directory at end | index near the front |
| Per-entry hash | CRC-32 | SHA-256 |
| Compression | deflate/store | zstd/store |
| Startup cost | read central directory, seek per entry | read index once, stream entries |
| Path safety | `zip` crate + our sanitizer | custom sanitizer |
| Current code | implemented and tested | not implemented |

Why keep ZIP for v1:

1. The footer/PE/signing fixes are mandatory and testable now; a container rewrite is orthogonal and adds risk.
2. `zip` 2.x supports ZIP64 and large files; the payload is under 2 GiB anyway.
3. The manifest already carries per-file SHA-256; the ZIP CRC catches decompression corruption. A full `--verify` can hash every entry.
4. The existing packer/tests can be fixed rather than replaced.

When to switch to ESPAK001:

- the final exe approaches the 2 GiB GitHub limit and ZIP64 overhead/scanning becomes material;
- startup measurement shows central-directory parsing is a bottleneck (unlikely at ~1.8 GB);
- per-entry SHA-256 becomes a requirement for partial repair;
- zstd compression materially reduces the artifact.

For v1 ZIP rules:

- `manifest.json` is entry 0, stored (method 0), not deflated.
- All other entries use deflate level 3 for small/text files and store for files >8 MiB (the current `build_container` heuristic). Models and GGUF are incompressible; do not burn first-run CPU.
- `large_file(true)` on every entry, so ZIP64 is available if a future model exceeds 4 GiB.
- No directory entries; create parents on extraction.
- No symlinks. `zip` exposes the Unix mode; reject entries whose file type is symlink (`mode & 0o170000 == 0o120000`).
- Reject absolute paths, drive letters, `..`, NUL, Windows reserved device names (`CON`, `PRN`, `AUX`, `NUL`, `COM1`–`COM9`, `LPT1`–`LPT9`, with or without an extension), trailing dots/spaces, and alternate data stream separators (`:`).
- Reject case-insensitive duplicate paths (`Models/A.onnx` vs `models/a.onnx`) because Windows collapses them.
- Reject entries whose uncompressed size exceeds the manifest total or a per-entry sanity limit. A ZIP bomb must not fill the disk before the free-space check can react.
- Normalize `\` to `/` in the manifest, but accept both when reading; Windows treats both as separators.

`zip`'s `enclosed_name()` rejects `..` and absolute paths, but it does **not** reject `CON`, trailing dots, ADS, or case collisions. The additional sanitizer is required. The current `sanitize_relative` only rejects non-normal components; it does not implement these Windows rules.

### 10.5 Extraction and cache

#### 10.5.1 Layout

```text
%LOCALAPPDATA%\EasyScribe\                 data root
  runtime\
    v1-<first 32 hex of payload SHA-256>\
      manifest.json
      models\...
      ffmpeg\...
      whispercpp\...
      licenses\...
      LICENSE.txt
      THIRD-PARTY-NOTICES.txt
      .complete
    .lock
    v1-<old>.tmp-<pid>-<random>\           stale staging, cleaned under lock
  recordings\                              crash-safe PCM + JSON, settings.json
  logs\                                    EasyScribe_YYYYMMDD_HHMMSS.log, keep 10
  temp\                                    scratch WAV/JSON, cleaned at start/exit
```

- `%LOCALAPPDATA%` comes from `SHGetKnownFolderPath(FOLDERID_LocalAppData)`, not the environment variable alone. Redirected profiles must work.
- `DEFAULT_RECORDINGS_DIR` is `Documents\EasyScribe` via `FOLDERID_Documents`.
- `recordings\settings.json` keeps the two keys `recordings_dir` and `speaker_speed_factor`; unknown keys are ignored.
- The cache survives moving/copying the exe. Deleting it forces a re-extract; user data is untouched.
- `EASYSCRIBE_DATA_DIR` overrides the data root; `EASYSCRIBE_RUNTIME_DIR` overrides only the runtime root; `EASYSCRIBE_PAYLOAD_DIR` bypasses the footer for development; `EasyScribe.portable` next to the exe selects `<exe_dir>\EasyScribe-data`.

#### 10.5.2 Cache hit

`cache_complete(runtime, footer)` must be cheap and strict:

```rust
const PAYLOAD_FORMAT_ZIP: u16 = 1; // `Manifest::payload_format` / CONTAINER_VERSION

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Marker {
    payload_sha256: String,
    app_version: String,
    payload_format: u16,
    build_id: String,
}

fn cache_complete(runtime: &Path, footer: &Footer) -> bool {
    let Ok(text) = std::fs::read_to_string(runtime.join(".complete")) else { return false };
    let Ok(marker) = serde_json::from_str::<Marker>(&text) else { return false };
    if marker.payload_sha256 != hex(&footer.payload_sha256)
        || marker.app_version != footer.app_version
        || marker.payload_format != PAYLOAD_FORMAT_ZIP
        || marker.build_id != footer.build_id
    {
        return false;
    }

    let Ok(bytes) = std::fs::read(runtime.join("manifest.json")) else { return false };
    if bytes.len() as u64 != footer.manifest_size {
        return false;
    }
    let Ok(manifest) = serde_json::from_slice::<Manifest>(&bytes) else { return false };
    if manifest.total_uncompressed == 0 || manifest.files.is_empty() {
        return false;
    }
    let sum = manifest.files.iter().try_fold(0_u64, |total, entry| {
        total.checked_add(entry.size)
    });
    if sum != Some(manifest.total_uncompressed) {
        return false;
    }

    // Files the engines cannot start without, even if an older manifest omitted them.
    if !CRITICAL.iter().all(|relative| runtime.join(relative).is_file()) {
        return false;
    }

    // Every manifest entry must exist with the exact recorded size. This is
    // metadata-only; it does not hash 1.8 GB on every launch.
    manifest.files.iter().all(|entry| {
        let Ok(relative) = sanitize_relative(Path::new(&entry.path)) else { return false };
        std::fs::metadata(runtime.join(relative))
            .map(|m| m.is_file() && m.len() == entry.size)
            .unwrap_or(false)
    })
}
```

Add a critical-file list for engines that a malformed/older manifest could omit:

```rust
const CRITICAL: &[&str] = &[
    "manifest.json",
    "models/silero_vad.onnx",
    "models/whisper/turbo-encoder.int8.onnx",
    "models/whisper/turbo-decoder.int8.onnx",
    "models/whisper/turbo-tokens.txt",
    "models/diarization/segmentation.onnx",
    "models/diarization/embedding.onnx",
    "ffmpeg/ffmpeg.exe",
    "ffmpeg/ffprobe.exe",
    "whispercpp/whisper-cli.exe",
    "whispercpp/ggml-large-v3-turbo-q5_0.bin",
];
```

Write the marker with the same `hex` and `PAYLOAD_FORMAT_ZIP` used above; otherwise a strict reader rejects every fresh extraction. The current extractor writes `footer.cache_key().trim_start_matches("v1-")` (32 hex chars) into `.complete`; the strict reader above expects the full 64-character hash, so change both together or every launch will re-extract. The marker file is `.complete`, and it is written only after every entry has been extracted and size-checked.

A cache hit is `< 1 s`: read `.complete`, read a small manifest, stat ~100 files. Do not read the payload, hash models, or open ZIP central directories on the fast path.

A same-size corrupted file passes the size check. That is the deliberate trade-off for a fast launch. `easyscribe --payload-verify --full` / `--repair` rehashes every entry against the manifest and re-extracts on mismatch. A runtime error from a model loader should offer “Repair app files”, which deletes the runtime and re-extracts.

#### 10.5.3 Locking

The scaffold already uses an OS file lock (`payload.rs:504-535`), which is the right primitive. Two gaps remain: it only treats `ErrorKind::WouldBlock` as contention (Windows can report `ERROR_LOCK_VIOLATION` as raw OS error 33), and its 300 s deadline is too short for a slow first extraction. Keep the lock file forever; never delete it.

```rust
use fs2::FileExt;

struct CacheLock {
    file: File,
}

impl CacheLock {
    fn acquire(path: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true).read(true).write(true).open(path)?;
        let deadline = Instant::now() + Duration::from_secs(900);
        loop {
            match file.try_lock_exclusive() {
                Ok(()) => return Ok(Self { file }),
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.raw_os_error() == Some(33) => // ERROR_LOCK_VIOLATION
                {
                    if Instant::now() >= deadline {
                        return Err(Error::Payload(
                            "EasyScribe is already getting ready in another window.".into(),
                        ));
                    }
                    std::thread::sleep(Duration::from_millis(200));
                }
                Err(e) => return Err(Error::Io(e)),
            }
        }
    }
}

impl Drop for CacheLock {
    fn drop(&mut self) {
        let _ = fs2::FileExt::unlock(&self.file);
    }
}
```

- Do not delete the `.lock` file. Deleting a lock file while another process holds it creates a race where two processes lock different inodes.
- The reference uses 300 s. Use 900 s: on a slow HDD, first extraction of 1.8 GB can exceed 5 minutes. Show “EasyScribe is getting ready in another window…” rather than an error. If the deadline is reached, offer Retry/Exit.
- If the data dir is on a network share where byte-range locks are unreliable, fall back to a named mutex (`Local\EasyScribe.Payload.<hash>`) or the old create-new lock with a PID liveness check. Do not silently proceed unlocked.

#### 10.5.4 Free space

```rust
let manifest = read_manifest_from_zip(exe, footer)?;
let needed = manifest.total_uncompressed.saturating_add(256 * 1024 * 1024);
let available = fs2::available_space(runtime_root)?;
if available < needed {
    return Err(Error::Payload(format!(
        "EasyScribe needs about {} MB free to start. Free some space and try again.",
        needed / 1_000_000
    )));
}
```

- `fs2::available_space` uses `GetDiskFreeSpaceExW` on Windows.
- 256 MiB margin covers logs, temp WAVs, and filesystem slack.
- `total_uncompressed` comes from the manifest, which is read from the ZIP central directory before extraction. Verify it against the sum of entry sizes; a malicious manifest must not understate the requirement.
- If the runtime already exists and is incomplete, its files still occupy space; `available_space` already accounts for that. Do not subtract it again.
- If extraction fails with `ERROR_DISK_FULL`, delete staging and show the same message with the actual path.

#### 10.5.5 Extraction sequence

```text
inspect footer (fast)
  → compute runtime = runtime_root / cache_key
  → if cache_complete: return Ready
  → acquire lock
  → if cache_complete (lost race): return Ready
  → clean every *.tmp-* (the lock guarantees no live writer)
  → read manifest, check free space
  → staging = runtime_root / "<cache_key>.tmp-<pid>-<random>"
  → verify payload SHA-256 (0.00–0.45)
  → extract ZIP entries into staging (0.45–0.95), verifying CRC and sizes
  → write .complete.tmp, fsync, rename to .complete (0.98)
  → activate staging → runtime (1.00)
  → gc old runtimes
```

`extract` must seek back to `payload_offset` after hashing. The current code does this (`payload.rs:370-372`), but the hash loop and the extraction loop should share one checked range:

```rust
fn payload_reader(exe: &Path, footer: &Footer) -> Result<impl Read + Seek> {
    let mut file = File::open(exe)?;
    file.seek(SeekFrom::Start(footer.payload_offset))?;
    Ok(file.take(footer.payload_size))
}
```

`ZipArchive::new` needs `Read + Seek`; `Take<File>` implements both and stops exactly at the payload end, so the certificate table after the footer is invisible.

`activate` must handle an existing **incomplete** runtime, not just an existing complete one:

```rust
fn activate(staging: &Path, runtime: &Path, footer: &Footer) -> Result<()> {
    if !runtime.exists() {
        return std::fs::rename(staging, runtime).map_err(Error::Io);
    }
    if cache_complete(runtime, footer) {
        let _ = std::fs::remove_dir_all(staging);
        return Ok(());
    }

    // A previous extraction crashed before .complete, or files were deleted.
    let bad = runtime.with_file_name(format!(
        "{}.bad-{}",
        runtime.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default(),
        std::process::id()
    ));
    std::fs::rename(runtime, &bad)?;
    match std::fs::rename(staging, runtime) {
        Ok(()) => {
            let _ = std::fs::remove_dir_all(&bad);
            Ok(())
        }
        Err(error) => {
            // Put the old directory back so the next launch can try again.
            let _ = std::fs::rename(&bad, runtime);
            Err(Error::Io(error))
        }
    }
}
```

The scaffold already handles the original discard bug: it removes an incomplete `runtime` and renames staging into place (`payload.rs:390-402`). That still has a failure window: if `remove_dir_all` succeeds and the rename fails, there is no runtime to restore. Keep the `activate` above so the old directory is renamed aside and restored on failure.

#### 10.5.6 Failure behavior

| Failure | Behavior |
|---|---|
| Footer magic/CRC/size invalid | “This EasyScribe file is damaged or incomplete. Download it again.” Exit before creating a window (or show a Rune error scene if already running). |
| Signature present but footer not found | Same damaged message. Do not fall back to dev mode. |
| Payload SHA-256 mismatch | Delete staging; “The app files could not be verified. Download EasyScribe again.” |
| ZIP CRC/size mismatch | Delete staging; same verification message. |
| Disk full | Delete staging; “EasyScribe needs about N MB free to start. Free some space and try again.” |
| Data dir not writable | “EasyScribe cannot write to <path>. Set EASYSCRIBE_DATA_DIR to a writable folder or run from a user account.” |
| Interrupted extraction | Staging remains; next launch cleans it under the lock and retries. Existing complete runtime untouched. |
| Another process extracting | Wait; show “EasyScribe is getting ready in another window…” |
| Critical file missing after extraction | `cache_complete` fails; re-extract; if still missing, name the file in an error dialog. |
| Old runtime in use during GC | Deletion fails harmlessly; retry next launch. |

Never leave a partial runtime activated. `.complete` is the only activation marker, and it is written last.

#### 10.5.7 Fix list for the current `payload.rs`

| # | Current behavior | Fix |
|---:|---|---|
| 1 | `read_footer` scans the last 1 MiB and uses an exact end equation; it does not parse the PE Security directory or check `SizeOfHeaders` | Parse PE32/PE32+ directories; compute `logical_end`; scan 0–7 padding bytes; keep the tail scan only as a diagnostic fallback |
| 2 | `.complete.tmp` is written and renamed without `sync_all` | `sync_all` the marker before rename, close/flush every extracted file, and test a crash between extraction and activation |
| 3 | `CacheLock` already uses `fs2`, but only maps `ErrorKind::WouldBlock` and waits 300 s | Also map Windows `ERROR_LOCK_VIOLATION` (raw 33); use a 900 s deadline; never delete the lock file |
| 4 | `cache_complete` checks a JSON marker and `metadata.len() > 0`, but not the manifest, exact sizes, or build id | Typed `Marker`, manifest size/bytes, total sum, exact per-entry sizes, `CRITICAL` list |
| 5 | An incomplete `runtime` is removed before staging is renamed; a failed rename loses the old runtime | Rename the old runtime to `.bad-<pid>`, activate staging, restore on failure |
| 6 | `main` uses `needs_prepare().unwrap_or(false)` plus separate `Paths::at`/`Paths::discover` calls; footer errors are masked and inspection runs twice | One `Boot::inspect` before Rune; `NeedsExtraction` carries the footer into the worker |
| 7 | `main.rs` headless detection misses `--tree`/`--frames` | `rune::tooling::is_headless()` |
| 8 | `read_footer` accepts a `payload_offset` inside the PE headers; only the proposed parser checks `SizeOfHeaders` | Reject `payload_offset < headers_end` |
| 9 | No free-space check | `fs2::available_space` + manifest total + 256 MiB |
| 10 | `prepare_with_progress` hashes the entire payload and then reads it again to extract it | Hash while extracting through one bounded `Take`, or document the double read and keep the progress split |
| 11 | ZIP sanitizer misses reserved names/ADS/trailing dots/case collisions | Add Windows path rules |
| 12 | ZIP symlink entries are extracted as files | Reject symlink Unix mode |
| 13 | `pack` builds the whole container in memory | Use `pack_streaming` for production; keep `build_container` test-only |
| 14 | Manifest time is `Utc::now()` | Honor `SOURCE_DATE_EPOCH`; deterministic entry order |
| 15 | Build id is a free-form string silently truncated to 16 bytes | Validate/truncate deliberately; use a 16-char git SHA |
| 16 | `read_footer` returns `None` when no magic is found; release errors surface later, while debug falls back to dev paths | Make absence in release an explicit `Boot::Damaged`; only debug/`EASYSCRIBE_PAYLOAD_DIR` may use dev mode |

### 10.6 Startup and the first-run scene

The scaffold already starts Rune before extraction: `main` uses `needs_prepare()` for a cheap cache probe, creates provisional paths, and `AppState::begin_prepare()` spawns `payload::prepare_with_progress` on a worker (`crates/easyscribe/src/app.rs:462-495`). The first frame therefore shows the “Getting ready…” scene, not a blank process. The remaining problem is structural: `needs_prepare()` and `Paths::discover()` each inspect the footer/cache, `needs_prepare().unwrap_or(false)` hides footer damage, and the worker repeats discovery inside `prepare_with_progress`. Collapse this into one `Boot` inspection and pass its result to both the splash and the worker.

#### 10.6.1 `Boot` state

```rust
pub enum Boot {
    Dev(Paths),
    Ready(Paths),
    NeedsExtraction {
        exe: PathBuf,
        footer: Footer,
        data_root: PathBuf,
    },
}

pub fn inspect() -> Result<Boot> {
    if let Some(dir) = std::env::var_os("EASYSCRIBE_PAYLOAD_DIR") {
        let data = payload::data_root();
        return Ok(Boot::Dev(Paths::for_runtime(PathBuf::from(dir), data)?));
    }

    let exe = std::env::current_exe()?;
    let Some(footer) = payload::read_footer(&exe)? else {
        if cfg!(debug_assertions) {
            return Ok(Boot::Dev(Paths::at(config::dev_base_dir())?));
        }
        return Err(damaged());
    };

    let data_root = payload::data_root();
    let runtime = payload::runtime_root(&data_root).join(footer.cache_key());
    if payload::cache_complete(&runtime, &footer) {
        Ok(Boot::Ready(Paths::for_runtime(runtime, data_root)?))
    } else {
        Ok(Boot::NeedsExtraction { exe, footer, data_root })
    }
}
```

`inspect` only reads the footer/PE headers and stats the cache. It must not hash or extract. `bootstrap::extract` is the worker half: it takes the already-validated `exe`/`footer`/`data_root`, opens the bounded `Take<File>` payload, extracts into staging with the progress callback, activates the runtime, and returns `Paths::for_runtime`.

#### 10.6.2 Rune startup

`AppSpec::init` is a function pointer with no captures. Keep a `OnceLock<Boot>` set before `rune::run`, or use `App::init(move || state)` with the captured `Boot`. The latter is cleaner:

```rust
let boot = bootstrap::inspect()?;
let app = rune::App::new(spec).init(move || AppState::new(boot));
```

For `NeedsExtraction`, `AppState` starts in `Phase::Preparing { progress: 0.0 }`. Start the worker from `AppSpec::setup`, which runs once when the runtime is created and is also called by headless tooling:

```rust
fn setup(state: &mut AppState, waker: Waker) {
    state.set_waker(waker);
    if matches!(&state.boot, Boot::NeedsExtraction { .. }) {
        state.begin_prepare();
    }
}
```

`begin_prepare` (already in the scaffold) does the actual work:

- spawns one `std::thread`;
- calls `bootstrap::extract(&exe, &footer, &data_root, ...)` with the fields carried by `Boot::NeedsExtraction` (the scaffold currently calls `payload::prepare_with_progress`, which repeats the footer check);
- sends `Preparing { stage, progress }`, `Prepared(Paths)`, or `PrepareFailed(String)` through the same channel as the engine events;
- calls the stored `Waker` after each send;
- `poll` drains the channel, updates the splash, and swaps `self.paths` on success.

Do **not** spawn from `view`; it runs every frame. `on_ready` is an acceptable alternative only if extraction must wait for the native window. It is not called by headless tooling and runs once per native creation (including reopen), so guard it with a `boot_started` flag/`Once` in state. Do not start a second extraction when the window is recreated after a hide/show.

- If `on_ready` is used, `handle.update` runs on the UI thread and its closure must be `Send`. `Paths` and the error string are `Send`.
- Throttle progress to ~10 Hz. Do not send an event per ZIP entry.
- The first frame can show a static “Getting ready…” with an indeterminate bar; once the manifest is read, switch to determinate `done / total_uncompressed`.
- Text: “Getting ready…”, “The first start can take a minute. Please wait.”, optional “Extracting 1.2 GB of 1.9 GB”.
- No folder picker, no Install button, no EULA checkbox, no shortcut checkbox. This is not an installer.
- On failure, show Retry / Open data folder / Exit. Never leave a half-extracted runtime active.
- On later runs, `Boot::Ready` is returned before `rune::run`; the window opens directly, no splash.

#### 10.6.3 Headless detection

Use the tooling helper, not a hand-maintained argument list:

```rust
let headless = rune::tooling::is_headless();
```

The current `main.rs` matches `--snap`, `--scene`, `--list-scenes`, `--script`, `--desktop`, and `--selftest`, plus `EASYSCRIBE_HEADLESS`; it misses `--tree` and `--frames`, so those modes run the GUI startup path (dialogs, recovery, dependency checks) and can block. `rune::tooling::is_headless()` covers `--tree`/`--frames` and the other tooling modes.

For headless:

- Prefer `EASYSCRIBE_PAYLOAD_DIR` so no extraction is needed.
- If a payload exe is tested headlessly, extract synchronously before `rune::run` (there is no Rune window to show progress), or test `bootstrap::extract` directly and then launch with `EASYSCRIBE_PAYLOAD_DIR`.
- Fatal errors go to `eprintln!`, not a native dialog.

#### 10.6.4 Pre-GUI fatal errors

Only footer/PE damage and a non-writable data root happen before Rune exists. Use `rfd::MessageDialog` on the main thread:

```rust
rfd::MessageDialog::new()
    .set_level(rfd::MessageLevel::Error)
    .set_title("EasyScribe cannot start")
    .set_description(&text)
    .show();
```

`rfd` uses the native Windows task dialog; it does not require Rune. Do not use it for recoverable first-run extraction errors; those belong in the Rune scene with Retry.

### 10.7 Windows build requirements

#### 10.7.1 Target and CRT

```toml
# .cargo/config.toml
[target.x86_64-pc-windows-msvc]
rustflags = ["-C", "target-feature=+crt-static"]
```

```toml
# Cargo.toml [profile.release]
[profile.release]
codegen-units = 16
lto = "thin"
strip = true
# panic = "abort" only after testing Rune's panic behavior
```

- x86-64 Windows 10/11 only. Do not ship ARM64 in v1; the prebuilt sherpa archive exists, but the model/toolchain matrix is unproven.
- `+crt-static` matches the sherpa `/MT` static archive and avoids a VC++ redistributable.
- If the link fails, fall back in this order:
  1. Build sherpa-onnx from source with the same `/MT` runtime.
  2. Use the shared sherpa build, ship its DLLs plus `onnxruntime.dll` in the runtime, and set the DLL search directory before the first load.
  3. Use `/MD` and ship `vcruntime140.dll`, `vcruntime140_1.dll`, `msvcp140.dll` in the runtime. This is the least desirable: it adds files and a DLL search-path dependency.
- `whisper-rs` is not in the release dependency graph, so its `/MD` objects cannot conflict.
- `whisper-cli.exe` is a separate process. Check its imports with `dumpbin /dependents`; if it needs `vcruntime140.dll`/`msvcp140.dll`, place those DLLs next to it in `whispercpp/`. Windows searches the exe's directory first. Do not assume the VC redist is installed.
- `ffmpeg.exe` from Gyan is usually self-contained; verify with `dumpbin`. If it imports VC DLLs, ship them next to it.

#### 10.7.2 Resource file

`assets/easyscribe.rc`:

```rc
1 ICON "EasyScribe.ico"
1 24 "easyscribe.manifest"

1 VERSIONINFO
FILEVERSION 3,2,3,0
PRODUCTVERSION 3,2,3,0
FILEOS 0x40004
FILETYPE 0x1
BEGIN
  BLOCK "StringFileInfo"
  BEGIN
    BLOCK "040904b0"
    BEGIN
      VALUE "CompanyName", "EasyScribe"
      VALUE "FileDescription", "EasyScribe"
      VALUE "FileVersion", "3.2.3"
      VALUE "InternalName", "EasyScribe"
      VALUE "OriginalFilename", "EasyScribe.exe"
      VALUE "ProductName", "EasyScribe"
      VALUE "ProductVersion", "3.2.3"
    END
  END
  BLOCK "VarFileInfo"
  BEGIN
    VALUE "Translation", 0x409, 1200
  END
END
```

`build.rs`:

```rust
fn main() {
    #[cfg(windows)]
    embed_resource::compile("assets/easyscribe.rc", embed_resource::NONE)
        .manifest_required()
        .expect("compile easyscribe.rc");
}
```

- `1 ICON` is the first icon group; the window/taskbar icon loads from the same resource.
- `1 24` is `RT_MANIFEST` with resource ID 1.
- `manifest_required()` fails the build if the resource compiler is missing; `manifest_optional()` would silently ship without a manifest. The manifest is required for DPI and `asInvoker`; use `required`.
- Keep `FILEVERSION` numeric (4 parts) and `FileVersion` string in sync with `CARGO_PKG_VERSION`. Add a build test or generate the `.rc` from `build.rs` so they cannot drift.
- `FILEOS 0x40004` is `VOS_NT_WINDOWS32`; `FILETYPE 0x1` is `VFT_APP`. These match the current Python version resource.

#### 10.7.3 Manifest

`assets/easyscribe.manifest`:

```xml
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
  <assemblyIdentity version="3.2.3.0" processorArchitecture="amd64"
                    name="EasyScribe" type="win32"/>
  <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">
    <security>
      <requestedPrivileges>
        <requestedExecutionLevel level="asInvoker" uiAccess="false"/>
      </requestedPrivileges>
    </security>
  </trustInfo>
  <application xmlns="urn:schemas-microsoft-com:asm.v3">
    <windowsSettings>
      <dpiAware xmlns="http://schemas.microsoft.com/SMI/2005/WindowsSettings">true/pm</dpiAware>
      <dpiAwareness xmlns="http://schemas.microsoft.com/SMI/2016/WindowsSettings">PerMonitorV2</dpiAwareness>
      <longPathAware xmlns="http://schemas.microsoft.com/SMI/2016/WindowsSettings">true</longPathAware>
    </windowsSettings>
  </application>
  <compatibility xmlns="urn:schemas-microsoft-com:compatibility.v1">
    <application>
      <supportedOS Id="{8e0f7a12-bfb3-4fe8-b9a5-48fd50a15a9a}"/>
    </application>
  </compatibility>
</assembly>
```

- `asInvoker`: no admin prompt. The app writes only to `%LOCALAPPDATA%` and `Documents`.
- `PerMonitorV2`: Rune/wgpu handles DPI changes; the manifest opts into per-monitor behavior.
- `longPathAware`: enables paths >260 characters if the OS policy allows it. Still use the `\\?\` prefix for the runtime path when it is long; the manifest alone is not a guarantee.
- `supportedOS` GUID is Windows 10/11. Do not add Windows 7/8 GUIDs.
- The manifest is embedded by `embed-resource`; do not also ship a loose `.manifest` next to the exe.

#### 10.7.4 Subsystem and console

```rust
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]
```

- Release builds are GUI subsystem: no console window on double-click.
- Debug builds keep the console for `cargo run` and logs.
- `--version`, `--payload-verify`, `--tree` may need stdout/stderr. In a GUI-subsystem release binary, `AttachConsole(ATTACH_PARENT_PROCESS)` (or `AllocConsole` if launched from Explorer) before writing. `rune::tooling` writes files for `--snap`/`--frames`; `--tree` prints, so attach.
- Do not switch the whole app to the console subsystem to make one CLI flag work; it would flash a console on every double-click.

#### 10.7.5 Icon and AUMID

- The icon is in the PE resources; `WM_SETICON` with `ICON_SMALL`/`ICON_BIG` sets the window/taskbar icon after creation. Use `LoadImageW(GetModuleHandleW(null), MAKEINTRESOURCEW(1), IMAGE_ICON, 0, 0, LR_DEFAULTCOLOR)`.
- Call `SetCurrentProcessExplicitAppUserModelID(L"EasyScribe.App")` before creating the window. This groups taskbar buttons and makes the icon/name correct.
- Rune's `set_app_icon` is macOS-only; do not use it.
- The `.ico` must contain 16/20/24/32/40/48/64/256 px frames for crisp DPI scaling. The current `assets/EasyScribe.ico` does; verify with `identify` or a resource viewer.

#### 10.7.6 Job object

Create a job object at startup with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` and assign the current process:

```text
CreateJobObjectW(null, null)
SetInformationJobObject(job, JobObjectExtendedLimitInformation,
    { LimitFlags: JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, ... })
AssignProcessToJobObject(job, GetCurrentProcess())
```

- Keep the job handle open for the process lifetime. When the app exits or crashes, the OS terminates all child processes (whisper-cli, diarize worker).
- If `AssignProcessToJobObject` fails because the app is already in a job (CI, some launchers), fall back to assigning each child at spawn. Do not fail startup.
- `on_close` still kills children explicitly for a clean shutdown; the job is the crash safety net.
- Do not use `CREATE_BREAKAWAY_FROM_JOB` unless the app must escape a parent job; it can fail with access denied.

#### 10.7.7 Long paths

- `longPathAware` in the manifest is necessary but not sufficient. Win32 APIs still need `\\?\` for paths over `MAX_PATH` unless the process policy is set.
- Implement `to_extended_path(path: &Path) -> PathBuf`: make it absolute, normalize `.`/`..`, then prefix `\\?\` (or `\\?\UNC\` for UNC). Use it for `File::open`, `fs::rename`, `remove_dir_all`, and `current_exe` reads.
- `std::fs::canonicalize` returns a `\\?\` path on Windows; `dunce::canonicalize` strips it. Do not add `dunce`; keep the prefix.
- The runtime path under `%LOCALAPPDATA%` can exceed 260 characters with a long username and a long payload hash. Test with a 300-character data root.
- ZIP entry paths are relative and short; the destination prefix is the long part.

#### 10.7.8 Deterministic builds

- `SOURCE_DATE_EPOCH` controls `built_utc` and the build id timestamp. If unset, use a fixed epoch (e.g. `0`) or fail the release build.
- Sort payload entries by normalized path; no filesystem iteration order.
- Fix the deflate level and the store/deflate threshold.
- Do not embed absolute build paths in the manifest.
- `build_id` is the first 16 hex characters of the git SHA or a deterministic UUID; never `SystemTime::now()`.
- The payload SHA-256 then identifies the build; two builds from the same inputs should produce the same container bytes (the PE itself may differ due to timestamps/PDB paths; strip and use `--remap-path-prefix` if reproducible PE is required).

#### 10.7.9 Size limit and release asset

- GitHub's per-asset limit is **2 GiB** (2,147,483,648 bytes). The final signed exe must be below it.
- Current planning estimate: 1.85–2.0 GB uncompressed payload; the ZIP may not shrink ONNX/GGUF much. Measure `dist/EasyScribe-v3.2.3.exe` before publishing.
- If over: reduce the GGUF quant (q5_0 → q4_0), drop the sherpa Whisper ONNX fallback (keep live-only? live needs it), or switch to ESPAK001/zstd. Do not add a runtime download; it violates the offline promise.
- Publish a `.sha256` next to the exe and on the release page.
- Keep the previous version's exe as the rollback artifact; there is no installer/backup.

#### 10.7.10 Code signing and SmartScreen

- Sign the final exe with SHA-256 and an RFC 3161 timestamp.
- Test on a clean Windows 10 and 11 VM with Defender enabled and no prior reputation.
- SmartScreen may still warn for a new certificate; reputation builds over downloads. EV certificates shorten the path but do not eliminate it.
- Do not use UPX or other packers. The payload is already compressed; packers increase AV false positives and can break the PE overlay.
- Submit false positives to Microsoft Defender and other vendors with the signed binary and the payload design explanation.
- Consider signing `whisper-cli.exe` and `ffmpeg.exe` before packing if the certificate allows it. They are extracted and executed; an unsigned child can trigger AV even when the outer exe is signed.
- Do not instruct users to disable Defender or add exclusions as the primary fix. Document the “Unblock” flow for unsigned test builds only.

### 10.8 Packaging pitfalls checklist

| # | Pitfall | Mitigation |
|---:|---|---|
| 1 | Footer no longer at EOF after signing | PE Security directory parser; scan 0–7 padding bytes |
| 2 | Signing after packing changes the hash | Hash only `[payload_offset, payload_offset + payload_size)`; sign last |
| 3 | PE checksum mismatch | Windows ignores it; `signtool` may update it; never include it in the payload hash |
| 4 | `payload_offset` points into headers | Require `payload_offset >= SizeOfHeaders` |
| 5 | ZIP path traversal | `enclosed_name` + Windows reserved/trailing/ADS/case checks |
| 6 | ZIP symlink escapes the runtime | Reject symlink Unix mode |
| 7 | ZIP bomb fills the disk | Manifest total + per-entry limits before extraction; free-space check |
| 8 | Concurrent first run extracts twice | `fs2` exclusive lock on `runtime\.lock` |
| 9 | Stale lock after a crash | OS file lock releases on process death; never delete the lock file |
| 10 | Interrupted extraction leaves a partial runtime | Staging + `.complete` last; atomic activation |
| 11 | Existing incomplete runtime is kept forever | Rename aside, activate staging, restore on failure |
| 12 | Cache hit rehashes 1.8 GB | Marker + manifest sizes + critical-file stat; full hash only on `--verify` |
| 13 | Cache hit trusts truncated files | Exact size check; `--repair` for corruption |
| 14 | Old runtime GC deletes a running app's files | Skip current; deletion failures are harmless; retry next launch |
| 15 | Stale `*.tmp-*` accumulates | Clean every `.tmp-*` under the extraction lock; no live writer can exist while it is held |
| 16 | Data dir is read-only | Friendly error with `EASYSCRIBE_DATA_DIR` guidance; portable marker fallback |
| 17 | Long paths fail | `longPathAware` + `\\?\` prefix; test 300-char data root |
| 18 | Case-insensitive path collision | Reject duplicate normalized lowercase paths in the builder |
| 19 | Antivirus flags the self-extracting exe | Sign, timestamp, avoid packers, submit false positives, test clean VM |
| 20 | Unsigned child process triggers AV | Sign `whisper-cli.exe`/`ffmpeg.exe` before packing if possible |
| 21 | VC runtime missing on a clean machine | Static CRT; otherwise ship `vcruntime140*.dll`/`msvcp140.dll` next to the exe |
| 22 | `whisper-cli` Vulkan import fails on old GPUs | A Vulkan-capable build may reject an old device but still run on CPU; a missing loader fails before `main`. Test both; keep the sherpa fallback. Do not assume Vulkan was used |
| 23 | First run takes minutes with no UI | Rune “Getting ready…” scene; `setup`/`begin_prepare` worker; determinate progress |
| 24 | Headless tooling triggers GUI startup | `rune::tooling::is_headless()`; `EASYSCRIBE_PAYLOAD_DIR` |
| 25 | Update/downgrade breaks settings | Data lives outside `runtime\`; settings ignores unknown keys |
| 26 | Old v3.2.3 install is ignored | One-time copy of `settings.json`; optional orphan recovery; document old folder removal |
| 27 | Final exe exceeds 2 GiB | Measure; reduce quant/model set or container overhead; never download at runtime |
| 28 | Build is non-deterministic | `SOURCE_DATE_EPOCH`, sorted entries, fixed levels, no absolute paths |
| 29 | Signing certificate expires | RFC 3161 timestamp; verify `/pa` |
| 30 | App is moved/copied | Cache key is payload hash; data root is per-user; no install path dependency |

### 10.9 Release acceptance for the payload

1. Pack a ~50 MB fixture, sign it, and run `read_footer`; the signed artifact must validate, and an unsigned artifact with a valid footer must also validate (only a truncated or CRC-damaged footer is rejected).
2. Pack the full payload, sign it, and verify:
   - `easyscribe-pack verify --full` passes;
   - `signtool verify /pa` passes;
   - the app starts from a different directory and from a read-only directory;
   - the cache hit is under 1 s on the target VM;
   - deleting the runtime re-extracts and preserves settings/recordings.
3. Run two instances simultaneously on a cold cache; only one extracts, the other waits and then opens.
4. Kill the process during extraction; the next launch cleans staging and succeeds.
5. Corrupt one byte of a model in the cache; `cache_complete` still passes (same size); `--repair` detects and re-extracts; the model loader error offers repair.
6. Fill the disk (or use a small volume) and confirm the free-space message and no partial runtime.
7. Test a 300-character `%LOCALAPPDATA%` path and a Unicode username.
8. Test on a clean Windows 10 and 11 VM with no VC++ redist and no Vulkan driver.
9. Compare the packed exe's `dumpbin /dependents` with the expected list; no `vcruntime140.dll` if static, no `vulkan-1.dll` in the main exe.

---

## 11. Implementation order and acceptance gates

The order is dependency-driven: payload paths first, then core engines, then UI, then live/diarization, then release. Do not port screens before the payload/bootstrap is testable; every screen needs `Paths`, and every long job needs a cancellation/event path that the UI can drive.

### 11.1 Gate 0 — Windows feasibility spike

This is not a demo. It is the set of unknowns that can invalidate the architecture. Do it on a clean Windows 10 and Windows 11 VM, not only on the development machine.

| # | Spike | Pass condition |
|---:|---|---|
| 0.1 | Rune hello-window at the pinned revision | Window opens, resizes, DPI changes, text input/IME works, drag/drop delivers paths, native file dialog opens, clipboard copy/paste works, modal overlay mounts |
| 0.2 | `+crt-static` + `sherpa-onnx` static link | A binary that creates a VAD and an `OfflineRecognizer` links and runs; `dumpbin /dependents` shows no `vcruntime140.dll`/`msvcp140.dll` |
| 0.3 | `whisper-cli` 1.8.6 Vulkan build | `whisper-cli --help` runs; `dumpbin /dependents` lists `vulkan-1.dll` only if expected; a machine without Vulkan falls back to CPU or fails cleanly |
| 0.4 | ffmpeg 7.1.1 essentials | `ffmpeg -version`/`ffprobe -version` run; decode a `.mp4`, `.mkv`, `.avi`, `.opus`, `.aac` fixture; `dumpbin /dependents` checked |
| 0.5 | cpal microphone | Enumerate devices, capture 5 s at the device rate, convert to 16 kHz mono, write WAV, play it with rodio |
| 0.6 | rfd multi-select | `pick_files()` from a worker thread returns multiple paths; no COM deadlock; owner window behavior acceptable |
| 0.7 | Pack + sign fixture | Append a 10 MB ZIP + footer, sign, `read_footer` accepts the signed file; `signtool verify /pa` passes |
| 0.8 | Job object | Child processes die when the parent is force-killed; normal `on_close` also kills them |

Exit criteria: every item passes, or a fallback is chosen and recorded in §1.2. The most likely failures are 0.2 (static CRT/system libs) and 0.3 (Vulkan/VC runtime). Do not start the UI port until 0.2/0.3 are resolved.

### 11.2 Gate 1 — Payload, cache and bootstrap

Work:

1. Fix items 1–16 in §10.5.7 (most are in `payload.rs`; items 6–7 are in `main.rs`).
2. Add `bootstrap.rs` with `inspect` (fast, returns `Boot`) and `extract` (worker, progress, cancel).
3. Change `Paths::discover` to a development/test helper; production `main` uses `bootstrap::inspect`.
4. Keep the existing “Getting ready…” scene and `begin_prepare` worker; wire them to `Boot` instead of re-inspecting.
5. Harden `CacheLock` (map raw 33, 900 s) and `cache_complete` (typed marker, manifest, exact sizes, build id).
6. Add the free-space check and the rename-aside activation with rollback.
7. Add Windows path sanitization (reserved names, trailing dots/spaces, ADS, case collisions, symlinks).
8. Make the manifest deterministic (`SOURCE_DATE_EPOCH`, sorted entries, fixed compression).
9. Add `easyscribe-pack verify/list/extract` and `--print-hash/--print-size`.
10. Add tests for signed/unsigned/truncated/corrupt footers, concurrent extraction, cache hit, disk full, path traversal, and case collision.

Exit criteria:

- A signed fixture and a signed full artifact both parse; the unsigned artifact is rejected only when damaged.
- Cold extraction writes `.complete` last; killing the process mid-extraction leaves no active runtime and the next launch succeeds.
- Warm launch does not read the payload or hash models; it stats the manifest entries and opens in under 1 s on the target VM.
- Two simultaneous cold launches extract once; the second waits and then opens.
- Deleting the runtime preserves settings and recordings.
- `easyscribe-pack verify --full` catches a one-byte model corruption.

### 11.3 Gate 2 — Core audio and file transcription

Work:

1. Port the ffmpeg wrapper (§9): fast path, probe, CR/LF reader, progress, cancel, errors.
2. Port WAV duration/load and transcript formatting; keep the reference's exact block/timestamp rules.
3. Implement `whisper-cli` command, GPU probe, JSON parse, stderr reader, cancel, cleanup.
4. Replace `sherpa-rs` with official `sherpa-onnx` for the VAD and recognizer; fix `tail_paddings = -1`.
5. Implement file-engine selection and the sherpa fallback VAD path.
6. Implement the job orchestration and `EngineEvent` sink.
7. Build a parity corpus: a short WAV, a 10-minute meeting, a video with a video track, a stereo 44.1 kHz file, an `.opus`, an `.avi`, and a file with a Unicode name.

Exit criteria:

- The same WAV through the Python oracle and the Rust CLI produces the same segment texts and timestamps (allowing only documented float formatting differences).
- The fast path copies a 16 kHz mono PCM WAV byte-for-byte.
- A malformed file produces the ffmpeg error snippet, not a panic.
- Cancel during extraction/transcription returns within 250 ms and leaves no temp file.
- The sherpa fallback produces the same text as the Python sherpa path on the same WAV.

### 11.4 Gate 3 — Rune UI shell

Work:

1. Port the screens and state machine from `REFERENCE_UI.md`: file list, record, running, done, error, naming, find speakers, privacy, details.
2. Wire drag/drop, file dialogs, clipboard, shortcuts, close behavior, icon/AUMID/manifest.
3. Port settings, recordings-folder selection, speaker speed, log rotation, and recovery prompts.
4. Implement the speaker-naming modal with `TextEdit` stored in state.
5. Add headless scenes for every state and snapshot them at the pinned Rune revision.
6. Replace `families: ("Segoe UI", "Consolas")` with bundled Inter + `("Inter", "ui-monospace")` or `Family::SystemMono`.
7. Use `rune::tooling::is_headless()`.

Exit criteria:

- All scenes render without missing fonts/icons; snapshot diffs are reviewed and accepted.
- Drag/drop accepts the 12 extensions; invalid files produce the exact skipped note.
- Native dialogs are non-blocking; the multi-select worker returns paths without freezing the UI.
- The naming modal preserves caret/selection/undo across frames.
- Closing the window kills children and stops capture; no orphan process.
- Keyboard shortcuts work; no OS-wide hotkey is registered.

### 11.5 Gate 4 — Live recording

Work:

1. cpal recorder actor: device enumeration, default config, sample-format conversion, mono downmix, resample to 16 kHz.
2. VAD thread, decode thread, histories, queues, overflow handling.
3. Stop/flush ordering and the 60 s join budget with session IDs.
4. PCM/JSON crash-safe writer and recovery scan.
5. rodio speaker-clip playback.
6. Live transcript UI and “Finishing the last words” state.

Exit criteria:

- A 30-minute recording produces a complete WAV and a live transcript; no ONNX abort.
- Stop drains queued audio, flushes the VAD, and decodes the final segments.
- A forced kill during recording leaves a `.pcm` + `.json` pair that recovery offers on next launch.
- Overflow logs once and does not corrupt the full recording.
- Late events from a timed-out session cannot mutate the next session.
- Playback of a 5 s speaker clip works on a clean VM.

### 11.6 Gate 5 — Diarization

Work:

1. `--worker diarize` entry point before Rune.
2. Official `sherpa-onnx` config with `-1`, `0.5`, `0.3`, `0.5`.
3. Vendored `process_with_callback` patch or raw C API shim.
4. Parent reader thread, 200 ms poll, progress/log mapping, kill on cancel, exit-code errors.
5. Speaker mapping, clip extraction, naming panel, plain fallback.
6. Job object coverage for the child.

Exit criteria:

- A 2-hour file produces the same speaker count and turn boundaries as Python within one frame/sample tolerance.
- Cancel kills the child within 500 ms; no orphan; the UI returns to Ready.
- A missing model disables the checkbox with the exact message.
- A child crash yields `Speaker identification stopped unexpectedly (exit code N).`
- A file-pipeline diarization failure writes a plain transcript and logs the warning; “Find speakers” surfaces the error.
- Memory stays within the VM budget for a 3-hour file, or the app shows a clear error rather than OOM-killing.

### 11.7 Gate 6 — Release artifact

Work:

1. Assemble the full payload: models, GGUF, whisper-cli/DLLs, ffmpeg/ffprobe, licenses, notices.
2. Build with `+crt-static`, embed icon/manifest/version, GUI subsystem.
3. Pack, sign, timestamp, verify.
4. Add CI: asset download with pinned hashes, build, pack, sign, `verify --full`, size gate, clean-VM smoke.
5. Add license collection and the FFmpeg source offer.
6. Test first run, second run, move, copy, read-only location, delete-cache, update, downgrade, and offline.
7. Measure final size and startup time.

Exit criteria:

- A clean Windows 10 and 11 VM with no VC++ redist, no Vulkan driver, and no network runs the app from a USB stick.
- The second launch is under 1 s to first frame.
- No network connection is attempted (process monitor / firewall log).
- `signtool verify /pa` passes; SmartScreen behavior is recorded.
- The final exe is under 2 GiB; the `.sha256` matches.
- Deleting the extracted runtime and relaunching recreates it without touching user data.
- The old v3.2.3 folder can be deleted after settings migration; no app files are written next to the exe by default.

### 11.8 Definition of done

- [ ] Gate 0 spikes pass on Windows 10 and 11.
- [ ] Signed footer parsing works; `payload_offset + payload_size + 128 + pad == cert_offset`.
- [ ] Cold extraction is atomic; warm launch does not rehash.
- [ ] `whisper-cli` command and JSON parsing match the Python oracle.
- [ ] `tail_paddings = -1`; VAD window/max speech/buffer sizes match.
- [ ] Live stop/flush loses no queued audio.
- [ ] Diarization cancel kills the child; no orphan.
- [ ] All headless scenes snapshot cleanly.
- [ ] All 12 extensions decode; `.avi` and `.opus` are in the smoke set.
- [ ] `-protocol_whitelist file` appears exactly twice in the built argv tests.
- [ ] The final exe is signed, timestamped, under 2 GiB, and runs offline.
- [ ] Licenses/notices include FFmpeg, whisper.cpp, sherpa-onnx, ONNX Runtime, WeSpeaker, and the models.

---

## 12. Risk register

Likelihood/impact: H/M/L. “Trigger” is the observable that says the risk is real, not theoretical.

| # | Risk | L | I | Trigger | Mitigation |
|---:|---|---|---|---|---|
| 1 | Rune/wgpu Windows immaturity (DPI, IME, drag/drop, dialogs) | M | H | Gate 0.1 fails or needs workarounds | Spike first; keep a Win32 fallback for the specific feature; pin Rune; do not fork unless unavoidable |
| 2 | `+crt-static` + sherpa static link fails (`LNK2038`, missing system libs) | M | H | Gate 0.2 link errors | Add required system libs; build sherpa `/MT` from source; fall back to shared sherpa + DLLs/VC runtime |
| 3 | `whisper-cli` Vulkan build fails on clean VMs | M | H | Gate 0.3 or 6 clean-VM run | Ship a CPU fallback binary or use the sherpa fallback; verify `dumpbin`; test no-Vulkan machine |
| 4 | Authenticode breaks footer parsing | H if unfixed | H | Signed exe says “damaged” | PE Security directory parser; signed fixture test; scan 0–7 padding |
| 5 | First extraction is slow/disk-heavy | M | M | Cold start > 2 min on HDD, disk-full reports | Rune “Getting ready” scene; progress; free-space check; 256 MiB margin; document ~2× transient space |
| 6 | SmartScreen/AV blocks the unsigned-looking 2 GB exe | H | H | Clean-VM SmartScreen warning/Defender quarantine | Sign + timestamp; submit false positives; avoid packers; sign child binaries; test clean VM early |
| 7 | Final exe exceeds GitHub 2 GiB | M | H | `dist/*.exe` size gate | Measure early; reduce GGUF quant; drop/compress payload; never add runtime download |
| 8 | FFmpeg GPLv3 compliance | M | H | Legal review | Ship license/source offer; keep separate process; consider LGPL build and re-test coverage |
| 9 | cpal WASAPI format/resample bugs | M | H | Gate 0.5/4: wrong pitch, dropped audio, device open failure | Capture at device mix format; implement/test resampler; keep stream on its thread; test 44.1/48/96 kHz and mono/stereo |
| 10 | Concurrent ONNX calls abort the process | M | H | Native abort during live recording | One mutex/one inference thread; never call VAD and recognizer concurrently; test under load |
| 11 | Diarization memory blow-up on long files | M | H | 3-hour file OOM | Child isolation; estimate memory; reject/segment very long files; show a clear error |
| 12 | Diarization cancel leaves an orphan | L | H | Process Explorer shows `--worker diarize` after cancel | Kill + wait; Job Object kill-on-close; test forced kill |
| 13 | Model licensing/attribution missing | M | H | Release audit | Ship CC BY 4.0 attribution for WeSpeaker, MIT for Whisper/pyannote/Silero, Apache/MIT for sherpa/ORT |
| 14 | Rune API drift at a newer pin | M | M | Build breaks after update | Pin the exact commit; vendor only if necessary; re-run headless scenes on upgrade |
| 15 | Text input/IME regressions in the naming modal | M | M | Gate 3 manual test | Keep `TextEdit` in state; test CJK IME, paste, undo, focus loss; no per-frame rebuild |
| 16 | Long paths / Unicode usernames | M | M | Gate 6 clean-VM test | `longPathAware`; `\\?\` helper; 300-char data root test; Unicode fixture names |
| 17 | Multi-instance extraction race | M | H | Two cold launches | `fs2` lock; second process waits; test simultaneous start |
| 18 | Cache corruption passes size check | M | M | Model loader error with same-size file | `--repair` full hash; error dialog offers repair; manifest per-entry SHA-256 for verify |
| 19 | Old v3.2.3 settings/recordings migration loses data | L | H | User reports missing recordings | Copy only `settings.json` when absent; never delete old folder; optional orphan recovery; document |
| 20 | Live transcript lags/overflows on slow CPUs | M | M | Overflow warning in normal use | Drop-oldest queue; one-time warning; full WAV is source of truth; lower thread priority; document |
| 21 | `sherpa-onnx` 2.x temptation breaks models/config | L | H | Dependency bump PR | Pin `=1.13.8`; require a model/API audit before any major bump |
| 22 | `whisper-rs` fallback is accidentally enabled | L | H | `vulkan` feature in release lockfile | Keep it out of the release dependency graph; CI `cargo tree`/`cargo deny` check |
| 23 | Font fallback silently uses the wrong face | M | L | UI snapshots show wrong mono/sans | Bundle Inter; use `("Inter", "ui-monospace")`/`Family::SystemMono`; snapshot text metrics |
| 24 | Child process survives a UI crash | L | H | Orphan after force-kill | Job Object with `KILL_ON_JOB_CLOSE`; `on_close` kill; test |
| 25 | Signing certificate expires / timestamp missing | L | H | `signtool verify` fails after expiry | RFC 3161 timestamp; verify `/pa`; keep the timestamped artifact |
| 26 | Deterministic build breaks due to time/order | M | L | Payload hash changes between identical builds | `SOURCE_DATE_EPOCH`, sorted entries, fixed compression, no absolute paths |
| 27 | Cache marker hash format drifts between writer and reader | M | M | Every launch re-extracts; warm start slow | One `hex` helper; 64-char marker; test writer/reader round-trip |
| 28 | `payload_offset`/footer arithmetic overflow on malicious input | L | M | Fuzz test panic | `checked_add` everywhere; reject out-of-range; fuzz the footer parser |
| 29 | Headless tooling starts GUI recovery/dialogs | M | M | CI hangs on `--tree`/`--frames` | `rune::tooling::is_headless()`; `EASYSCRIBE_PAYLOAD_DIR`; no dialogs in headless |
| 30 | App writes next to the exe in Program Files/read-only media | L | M | Startup permission error | Default to `%LOCALAPPDATA%`; portable marker only when writable; clear error |

### 12.1 Stop-ship risks

The release must not ship if any of these is unresolved:

1. Signed footer lookup (§10.3) — test the real Authenticode certificate size; the 1 MiB heuristic must not reject the signed release.
2. Static CRT/sherpa link (§10.7.1) — the app would not start on a clean VM.
3. `whisper-cli` Vulkan no-driver behavior (§8.1.1) — file transcription could fail on common laptops.
4. Diarization orphan/kill (§8.6.7) — a cancel could leave a CPU-heavy process behind.
5. FFmpeg `-protocol_whitelist file` missing (§9.7) — the offline guarantee is broken.
6. 2 GiB asset limit (§10.7.9) — the release cannot be uploaded.

### 12.2 “Unknown unknowns” to probe early

- Windows Defender's behavior when a signed 1.9 GB exe writes ~1.9 GB of executables/models into `%LOCALAPPDATA%` on first run.
- Rune's text input with a Windows IME (Japanese/Chinese/Korean) and with a compose key.
- `rfd` multi-select ownership when the Rune window is the foreground window; some COM dialog implementations need an owner HWND.
- `cpal` behavior when the default device changes mid-recording (headset unplugged).
- `whisper-cli` on a hybrid-GPU laptop where the “best” Vulkan device is not the one Windows would choose.
- Diarization on a file with music/noise; the segmentation model may produce many short turns.

---

## 13. Source map

### 13.1 Rune and rune-kit

| Topic | Source checked |
|---|---|
| `AppSpec`, headless dispatch, `is_headless` | `crates/rune/src/tooling.rs` |
| `App`, `AppHandle`, `on_ready`, `on_close`, `update` | `crates/rune/src/app.rs` |
| `WindowHandle::with_raw_handle`, raw handle access | `crates/rune/src/window/handle.rs` |
| `El`, handlers, drag/drop, text input | `crates/rune/src/el.rs`, `crates/rune/src/el/drop.rs`, `crates/rune/src/el/text.rs` |
| `Tx`, timers, clipboard, file dialogs, `physical_key` | `crates/rune/src/el/tx.rs`, `file_dialog.rs`, `clipboard/tx.rs` |
| Runtime/headless input, `--snap`/`--tree`/`--script`/`--frames` | `crates/rune/src/runtime.rs`, `crates/rune/src/runtime/*` |
| Windows window placement and raw handle; icon/AUMID are app Win32 code | `crates/rune/src/windows.rs`, `window/handle.rs`; `set_app_icon` is macOS-only (`platform/app_icon.rs`) |
| Rune behavior/API docs | `crates/rune/README.md` at `781e2b6d87cd5f32d6f4365956fad4baaf24ee50` |
| rune-kit components/theme | `crates/rune-kit/README.md`, `src/kit.rs`, `src/actions.rs`, `src/text_input.rs`, `src/dialog.rs`, `src/segmented.rs`, `src/feedback.rs`, `src/lib.rs` |

### 13.2 Native crates

| Topic | Source checked |
|---|---|
| rfd 0.17.2 multi-select/message dialog | `~/.cargo/registry/src/.../rfd-0.17.2/src/*` |
| cpal 0.18.2 device/config/stream | `~/.cargo/registry/src/.../cpal-0.18.2/src/*` |
| hound 3.5.1 WAV reader/writer | `~/.cargo/registry/src/.../hound-3.5.1/src/*` |
| rodio 0.23.0 sink/decoder/`wav` feature | `~/.cargo/registry/src/.../rodio-0.23.0/src/*` |
| notify 8.2.1 (optional) | `~/.cargo/registry/src/.../notify-8.2.1/src/*` |
| tracing 0.1.44 / tracing-appender 0.2.4 | `~/.cargo/registry/src/.../tracing-*` |
| serde_json 1.0 | `~/.cargo/registry/src/.../serde_json-1.0.*` |
| fs2 0.4.3 file locks/free space | `~/.cargo/registry/src/.../fs2-0.4.3/src/*` |
| windows-sys 0.61.2 | `~/.cargo/registry/src/.../windows-sys-0.61.2/src/Windows/Win32/*` |
| embed-resource 3.0.12 | `~/.cargo/registry/src/.../embed-resource-3.0.12/src/lib.rs` |

### 13.3 Speech and audio

| Topic | Source checked |
|---|---|
| `sherpa-onnx` 1.13.8 safe API | `~/.cargo/registry/src/.../sherpa-onnx-1.13.8/src/{vad.rs,offline_asr.rs,offline_speaker_diarization.rs,speaker_embedding.rs,wave.rs}` |
| `sherpa-onnx-sys` 1.13.8 configs/build | `~/.cargo/registry/src/.../sherpa-onnx-sys-1.13.8/src/*`, `build.rs` |
| C API diarization callback/progress | `/home/user/EasyScribe/.venv/lib/python3.13/site-packages/sherpa_onnx/include/sherpa-onnx/c-api/c-api.h` (`SherpaOnnxOfflineSpeakerDiarizationProcessWithCallback`, `...ProgressCallback`) |
| `whisper-rs` 0.16.0 API | `~/.cargo/registry/src/.../whisper-rs-0.16.0/src/{whisper_ctx.rs,whisper_ctx_wrapper.rs,whisper_state/*,whisper_params.rs}` |
| `whisper-rs-sys` 0.15.0 bundled whisper.cpp | `~/.cargo/registry/src/.../whisper-rs-sys-0.15.0/whisper.cpp/CMakeLists.txt` (`VERSION 1.8.3`), `build.rs` |
| whisper.cpp 1.8.6 CLI behavior | `REFERENCE_BACKEND.md` §10; current `src/whispercpp_wrapper.py` |
| ffmpeg wrapper behavior | `REFERENCE_BACKEND.md` §8; current `src/ffmpeg_wrapper.py` |
| Symphonia 0.6.1 coverage | `~/.cargo/registry/src/.../symphonia-0.6.1/Cargo.toml`, `README.md` |
| ffmpeg-next 9.0.0 requirements | `~/.cargo/registry/src/.../ffmpeg-next-9.0.0/Cargo.toml`, `README.md` |

### 13.4 Current EasyScribe code

| Topic | Source |
|---|---|
| Payload/footer/ZIP/pack | `crates/easyscribe-core/src/payload.rs` (notably `read_footer`, `prepare_with_progress`, `extract_to_with_progress`, `CacheLock`, `pack_streaming`) |
| Paths/payload bootstrap | `crates/easyscribe-core/src/config.rs` (`Paths::discover`, `for_runtime`, `data_root`) |
| Current startup/dialogs/headless detection | `crates/easyscribe/src/main.rs` |
| Current state/worker/poll | `crates/easyscribe/src/app.rs`, `workers.rs` |
| Current sherpa usage | `crates/easyscribe-core/src/sherpa_engine.rs`, `live.rs`, `diarizer.rs` |
| Current ffmpeg/WAV/transcript | `crates/easyscribe-core/src/ffmpeg.rs`, `audio.rs`, `transcript.rs` |
| Current VAD constants | `crates/easyscribe-core/src/vad.rs` |
| Current dialogs (blocking rfd) | `crates/easyscribe/src/dialogs.rs` |
| Current theme/fonts | `crates/easyscribe/src/theme.rs` |

### 13.5 Reference documents

| Topic | Document/section |
|---|---|
| Exact backend behavior, VAD, whisper.cpp, diarization, live | `docs/rewrite/REFERENCE_BACKEND.md` §8–§14 |
| Payload/footer/container/model inventory/Windows PE | `docs/rewrite/REFERENCE_PACKAGING.md` §6–§7, §11 |
| Screens, tokens, dialogs, focus/close behavior | `docs/rewrite/REFERENCE_UI.md` |
| Python source line ranges | `REFERENCE_BACKEND.md` citations (`src/*.py` line numbers) |
| Current tests encoding behavior | `REFERENCE_PACKAGING.md` §9; Python `tests/test_*.py` |
| Superseded in-process whisper plan | `REWRITE_PLAN.md` (superseded by §8.1 for the file engine) |

### 13.6 External specifications

| Topic | Specification |
|---|---|
| PE32/PE32+ headers, optional header, data directories | Microsoft PE Format specification; `winnt.h` (`IMAGE_OPTIONAL_HEADER32/64`, `IMAGE_DIRECTORY_ENTRY_SECURITY`) |
| Authenticode certificate table | Microsoft PE Format, “Attribute Certificate Table”; `WIN_CERTIFICATE`; RFC 3161 timestamping |
| WASAPI shared mode / device formats | Microsoft Windows Core Audio documentation |
| Job objects / kill-on-close | Microsoft `JOBOBJECT_EXTENDED_LIMIT_INFORMATION`, `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` |
| ZIP format / ZIP64 | PKWARE APPNOTE; `zip` crate 2.x |
| Silero VAD / Whisper / pyannote / WeSpeaker model cards | Upstream model repositories; licenses in §8.8 and `REFERENCE_PACKAGING.md` §10 |

### 13.7 How to re-verify a claim

1. For Rune APIs, read the pinned source, not the changelog or docs website. `crates/rune/README.md` is part of the pin.
2. For crate APIs, read the exact version in `~/.cargo/registry/src/...`; do not trust `docs.rs` for a different version.
3. For C APIs, read the header that ships with the pinned native library (`c-api.h`), not the Python docstring.
4. For parity behavior, run the Python oracle on the same fixture and diff the output; do not infer from comments.
5. For Windows behavior, run on a clean VM. A Linux `cargo check` or a cross-compile is not evidence that the MSVC link, COM dialog, WASAPI device, or Authenticode path works.

---

*End of technical research. The normative product behavior remains `REFERENCE_BACKEND.md`, `REFERENCE_PACKAGING.md`, and `REFERENCE_UI.md`; this document records the implementation choices and the evidence behind them.*
