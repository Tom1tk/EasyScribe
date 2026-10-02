# EasyScribe — Development Rules & Lessons

This file is automatically loaded by Claude Code. Rules here are derived from real mistakes
made during development. Read before making any build or packaging changes.

---

## PyInstaller Packaging Rules

### Rule 1: `excludes` overrides `collect_all()` — never exclude ML transitive deps

**The mistake we kept making (pre-v2.0.0, when diarization ran on `pyannote.audio`):**
Adding a package to `excludes` in `EasyScribe.spec` to reduce bundle size, not realising
it was a transitive runtime dependency of `pyannote.audio`. This caused a whack-a-mole
series of `No module named '<X>'` errors on live builds:
- scipy → removed from excludes
- torchaudio → removed from excludes
- pandas → removed from excludes
- sklearn, matplotlib → removed from excludes

**The rule:** Before adding any package to `excludes`, verify it does NOT appear in the
transitive dependency tree of `sherpa_onnx`, `faster_whisper`, or `ctranslate2`. If in
doubt, leave it out of excludes.

**How PyInstaller processes excludes:** The `excludes` list is applied *after* analysis,
including after `collect_all()`. So even if `collect_all('sherpa_onnx')` discovers a
package, an explicit `excludes` entry will strip it from the final bundle. There is no
warning — the package just silently disappears and crashes at runtime.

---

### Rule 2: PyInstaller hooks only trigger on *Python-analyzed* imports

**The mistake:** We wrote `hooks/hook-nvidia.py` expecting it to collect CUDA DLLs.
It never ran because PyInstaller only triggers hooks when it statically analyzes a Python
`import` statement for the hook's package. `ctranslate2` loads CUDA DLLs via the C++
`LoadLibraryA()` call in `cublas_stub.cc` — PyInstaller never sees an `import nvidia.*`.

**The rule:** For DLLs loaded via C++ (not Python imports), collect them directly in the
spec file using a function that runs unconditionally at build time (like `_collect_nvidia_dlls()`
in `EasyScribe.spec`). Do not rely on hooks for packages that aren't Python-imported.

---

### Rule 3: `os.add_dll_directory()` does NOT work for C++ `LoadLibraryA()`

**The mistake:** We tried calling `os.add_dll_directory()` (which calls Windows
`AddDllDirectory()`) to help ctranslate2 find CUDA DLLs. This only affects calls to
`LoadLibraryExW` with the `LOAD_LIBRARY_SEARCH_USER_DIRS` flag — ctranslate2's C++ code
uses plain `LoadLibraryA()`, which only searches `PATH` and the standard DLL search order.

**The rule:** When a C++ library loads DLLs via `LoadLibraryA()`, the only way to add
search paths is to **prepend to `os.environ["PATH"]`** before the library is imported.
See `src/cuda_setup.py` for the working implementation.

---

### Rule 4: Packages with C extensions need `collect_all()`, not just `hiddenimports`

**The rule:** If a package has compiled Cython or C extensions (`.pyd` on Windows, `.so`
on Linux), adding it to `hiddenimports` imports the top-level package but does NOT copy
the compiled binaries. Use `collect_all('<package>')` instead. Key package this applies to:
- `sherpa_onnx` — ships compiled `.so`/`.pyd` extensions plus a `.libs` directory;
  see the `collect_all('sherpa_onnx')` call in `EasyScribe.spec`

Packages with PyInstaller built-in hooks (`pandas`, `matplotlib`) can use `hiddenimports`
since the hook handles their binaries automatically.

---

### Rule 5: Bump the venv cache key after any dependency changes

**The rule:** The GitHub Actions venv cache (`venv-win64-py3.11-vN`) is keyed to a static
string. Any change to pip dependencies (new packages, removed excludes, etc.) requires
bumping the version suffix (`v3` → `v4` → ...) to force a clean rebuild. Failing to do
this means the old cached venv is used and changes don't take effect.

---

### Rule 6: A bundled native library's *runtime* DLL deps aren't always obvious from its package name

**The mistake:** We bundled `nvidia-cublas-cu12`, `nvidia-cuda-runtime-cu12`,
`nvidia-cudnn-cu12`, `nvidia-cuda-nvrtc-cu12` for ctranslate2 and assumed that covered
"CUDA 12 DLLs" generally. When sherpa-onnx's bundled `onnxruntime_providers_cuda.dll`
tried to load on a real GPU machine, it failed with
`OrtSessionOptionsAppendExecutionProvider_Cuda: Failed to load shared library` — silently
falling back to CPU (the code's graceful-degradation path masked the real error; only the
rotating file log's `logger.warning(...)` line had the actual exception text).

The actual cause: `onnxruntime_providers_cuda.dll` depends on `cufft64_11.dll`
(cuFFT — NVIDIA kept cuFFT's SONAME at `11` even inside CUDA 12.x toolkits, so the
"11" in the filename does NOT mean "needs CUDA 11"). No `nvidia-cufft-cu12` package was
installed, so `LoadLibraryA` failed with "module not found" on that dependency.

**The rule:** When bundling a precompiled native library that talks to CUDA
(`onnxruntime_providers_cuda.dll`, ctranslate2's CUDA stubs, etc.), don't assume the
nvidia-*-cu12 package set you already have covers it — **PyInstaller's build-time
warnings tell you exactly which DLLs it couldn't resolve**:
```
WARNING: Library not found: could not resolve 'cufft64_11.dll', dependency of
'...\sherpa_onnx\lib\onnxruntime_providers_cuda.dll'.
```
Grep the PyInstaller build log for `Library not found` after adding any new
CUDA-touching package, map each missing DLL name to its `nvidia-<x>-cu12` pip package
(`cufft64_11.dll` → `nvidia-cufft-cu12`, `cusparse64_12.dll` → `nvidia-cusparse-cu12`,
etc. — NVIDIA's library SONAMEs are stable across toolkit versions and don't match the
CUDA major version), and add it to the install list. Ignore warnings for optional
providers you never request (e.g. `onnxruntime_providers_tensorrt.dll` wanting
`nvinfer_10.dll` — TensorRT is opt-in and irrelevant if you only use `provider="cuda"`).

---

## v2.0 Packaging Rules

### Rule 7 (corrected 2026-06-11): sherpa-onnx has NO Vulkan provider

v2.0.0 shipped believing `provider="vulkan"` gave GPU acceleration. It never did.

Valid provider strings in sherpa-onnx 1.13.2: `cpu`, `cuda`, `directml`, `coreml`,
`xnnpack`, `nnapi`. An unrecognized provider string does **not** raise — sherpa-onnx
logs `Unsupported string: vulkan. Fallback to cpu` and silently constructs the
recognizer on CPU. This means a `try/except RuntimeError` fallback pattern around
`OfflineRecognizer(cfg)` can never detect a bad provider string — construction
always "succeeds", just silently on CPU.

v2.1 removes `provider="vulkan"`, the GPU device dropdown, `preferred_gpu_index`,
`list_gpus()`, and `src/vulkan_probe.py` entirely — EasyScribe is CPU-only and says
so. `vulkan_probe.py` lives on in git history; it's good code (note its LLP64
ctypes fix) and gets resurrected if/when a real GPU backend (e.g. whisper.cpp
Vulkan, see Phase 8 strategic track) is evaluated.

**The lesson:** verify provider claims empirically (grep the bundled native libs,
or check the ONNX Runtime execution-provider list) before building features on them.

> **Phase 8 update (2026-06-11):** "EasyScribe is CPU-only" above describes
> **sherpa-onnx** specifically — that part of Rule 7 still holds. sherpa-onnx
> 1.13.2 has no Vulkan provider, and VAD, diarization, and live mode (all
> sherpa-onnx) remain CPU-only. File transcription, however, now has a second,
> optional engine: `whisper-cli` (whisper.cpp), built in CI with
> `-DGGML_VULKAN=ON` (see `.github/workflows/build-release.yml` and
> `src/whispercpp_wrapper.py`) — a different binary/codebase from sherpa-onnx,
> so this doesn't contradict the finding above. `whisper-cli` logs which device
> it actually used (`whisper_backend_init_gpu: device N: <name>` or
> `no GPU found`), and `whispercpp_wrapper.py` surfaces that line to the UI log
> — the same "verify empirically, never assume from build flags" discipline
> this rule established. `vulkan_probe.py` was deliberately *not* resurrected:
> a separate ctypes GPU probe would be a second, possibly-disagreeing source of
> "is there a GPU" info — `whisper-cli`'s own stderr device line is the actual
> ground truth and is already surfaced.

### Rule 8: Silero VAD model must be downloaded and bundled explicitly

`sherpa_onnx.get_default_vad_model()` does NOT exist in sherpa-onnx 1.13.2.
Download `silero_vad.onnx` from the sherpa-onnx GitHub releases and reference it via
`config.VAD_MODEL_PATH`. Bundle as `("models/silero_vad.onnx", "models")` in both specs.

### Rule 9: Parakeet TDT uses OfflineTransducerModelConfig, not a CTC config

`OfflineNemoCtcModelConfig` does not exist as of sherpa-onnx 1.13.2.
Parakeet TDT 0.6B v3 int8 is a transducer model — use `OfflineTransducerModelConfig`
with `encoder_filename`, `decoder_filename`, `joiner_filename`.
The real NeMo CTC class is `OfflineNemoEncDecCtcModelConfig` but it is not used here.

> **v2.1: parakeet variant removed.** Kept for historical reference only —
> the transducer-config lesson may resurface if another transducer model is
> evaluated in the future.

### Rule 10: venv cache key has no variant suffix — both matrix jobs share it

CI builds two variants (whisper, parakeet) in parallel. They install identical pip deps.
Cache key `venv-win64-py3.11-v2` has no variant suffix so the second job always hits cache.
Bump v2 → v3 etc. to force a clean rebuild after adding/removing packages.

> **v2.1: parakeet variant removed.** CI is now a single job; the
> shared-cache-across-matrix-jobs concern no longer applies, but the
> "bump the version suffix after dependency changes" lesson still does.

### Rule 11: Always list numpy explicitly in the CI pip install command

`numpy` is not a declared Python dep of `sherpa-onnx` on all platforms. If it is absent from
the venv, `collect_all('numpy')` silently catches `ImportError` and bundles nothing, then
`hiddenimports=["numpy"]` resolves to an empty module — the build succeeds but the exe crashes
with `ModuleNotFoundError: No module named 'numpy'` at runtime.

Always list `numpy` explicitly. Since v3.0.0-beta2 all dependencies are pinned in
`requirements.txt` (numpy included) and CI runs `pip install -r requirements.txt` —
keep numpy in that file, and bump the venv cache key when you change it. Run
`tests/validate_build.py dist\EasyScribe` immediately after `pyinstaller` (before creating
`app.bundle`) to catch this class of problem before the slow compress step.

### Rule 12: sherpa_onnx's public `OfflineRecognizer` class has no `__init__` — use the binding class

`sherpa_onnx.OfflineRecognizer` (the Python wrapper in `offline_recognizer.py`) only exposes
`from_transducer`/`from_whisper`/etc. classmethods that build `self.recognizer` internally.
Calling `sherpa_onnx.OfflineRecognizer(cfg)` directly raises
`TypeError: OfflineRecognizer() takes no arguments` — it inherits `object.__init__` and was
never given a config-based constructor. The actual config-based constructor is the binding
class: `from sherpa_onnx.lib._sherpa_onnx import OfflineRecognizer as _OfflineRecognizer`,
then `_OfflineRecognizer(cfg)` where `cfg` is an `OfflineRecognizerConfig`. Also note
`OfflineRecognizerConfig(...)` itself takes `model_config=`, not `model=`.

`tests/test_sherpa_api.py` builds the config and constructs `_OfflineRecognizer(cfg)`
against nonexistent model paths, asserting `RuntimeError` (bad path) rather than
`TypeError` (bad constructor signature). Run it locally
(`python tests/test_sherpa_api.py`, no model files or GPU needed) before triggering a
build — it now also runs as an early CI step, right after `pip install`, before any
model downloads.

---

### Rule 13: The app must stay fully offline (GDPR) — keep `offline_guard` first

EasyScribe is sold on "nothing leaves this computer". `src/main.py` calls
`offline_guard.install()` before any other import, so a library that tries to phone home
gets `NetworkBlockedError` instead of sending data. Never move that import down, never add
a network library to `src/` or `launcher/` (no update checks, telemetry, crash upload,
`webbrowser` links or model downloads at runtime), and keep `_LOCAL_FILES_ONLY`
(`-protocol_whitelist file`) on every ffmpeg/ffprobe call. `tests/test_offline_guard.py`
checks all three. Logs may hold file names and progress, never transcript text.

### Rule 14: The launcher must compare versions — never just "folder exists → launch"

**The mistake:** up to v3.0.0-beta1 the launcher opened any existing `EasyScribe\` folder.
A user with v2 installed double-clicked the v3 exe and got v2 again, with no message.

**The rule:** `launcher.py` reads the version from `.easyscribe-install.json` and compares
it with `VERSION` (`_version_key`: pre-release < release, never downgrade). An older
install is updated in place: app items move to `.easyscribe-old\`, the new bundle is
extracted, and the backup is restored on any error. `recordings\` and `logs\`
(`USER_DATA_DIRS`) are never moved or deleted — not by the update and not by
`_clean_incomplete_install`. A locked file (app still open) stops the update before any
file is replaced. An `"updating"` flag in the marker lets the next start recover from a
power loss. `tests/test_launcher_safety.py` covers each path; it needs tkinter.

### Rule 15: Set an AppUserModelID, or Windows shows a stale taskbar icon

`iconbitmap()` sets the window icon only. The taskbar groups by AppUserModelID and caches
icons per exe path, so after an update to the same path it can show the old icon.
`main.py` calls `SetCurrentProcessExplicitAppUserModelID("EasyScribe.App")` first, the
shortcuts set `IconLocation`, and the launcher calls `SHChangeNotify(SHCNE_ASSOCCHANGED)`
after an update.

### Rule 16: Validate the assembled folder, not only the PyInstaller output

`dist\EasyScribe` is changed after PyInstaller (models, whispercpp, ffmpeg are copied
in). Run `tests/validate_build.py dist\EasyScribe --assembled` after that step, and check
that the copied `whisper-cli.exe --help` starts, before `app.bundle` is made. Never use
`-ErrorAction SilentlyContinue` on a copy that the app needs.


### Rule 17: The one-file launcher needs a splash — the bootloader is silent for up to a minute

**The mistake:** beta2 users saw only a busy cursor for about 60 s on the first start.
The PyInstaller one-file bootloader extracts everything, including the 1.5 GB
`app.bundle`, to `%TEMP%` *before* any Python code runs, so nothing in `launcher.py`
can show a window in time.

**The rule:** keep the `Splash("../assets/splash.png", ...)` in `launcher/launcher.spec`
(the bootloader shows it first, then extracts the rest), and keep `_close_splash()` in
`launcher.py` on every path (installer window open, or the app started). Do not set
`text_pos`: the bootloader would then print each extracted file name. Regenerate the
image with `assets/make_splash.py` if the palette changes.

### Rule 18: The installer never changes an install without a click

**The mistake:** beta2 updated an older install at once when the exe was opened. The user
could not choose a different folder and was not asked.

**The rule:** `main()` opens the app directly only when the install is current. In every
other case it shows `InstallerApp` and waits for **Install** / **Update and open**.
`_set_inputs(False)` locks the window during the work so a second click cannot start a
second run.

### Rule 19: Tk on Windows never erases backgrounds — that is the "black blocks" flash

**Cause (Tk source, `win/tkWinX.c`, `tkWinWm.c`):** the `TkChild` window class has
`hbrBackground = NULL`, Tk answers `WM_ERASEBKGND` with 0, and `WM_PAINT` only queues an
Expose event that Tk redraws later from its idle loop. Until then the area is black.
CustomTkinter widgets are many child HWNDs, so a refocused window fills in as black blocks.

**The fix:** `src/win_paint.py` replaces the `TkChild` class window procedure with a ctypes
callback that fills the client rect with `C.BG` on `WM_ERASEBKGND` and passes every other
message to Tk's procedure. Call `win_paint.install()` before any widget is made (it only
affects windows created after it, plus the root's own child window). Keep the ctypes
callback object referenced for the life of the process (`_state`), or Windows calls freed
memory. `EASYSCRIBE_PAINT_FIX=0` switches it off. `WS_EX_COMPOSITED` is not a fix: Tk
draws outside `WM_PAINT`, so composited windows show stale content.

### Rule 20: Licenses ship in the bundle — update the notices when a component changes

**The mistake:** up to v3.0.0-beta3 the bundle had no license files. The gyan.dev ffmpeg
build is GPL v3, so each release must include the GPL text and the source (or a written
offer). CC BY 4.0 (WeSpeaker model), Apache-2.0 (sherpa-onnx) and MIT also require their
notices.

**The rule:** the texts live in `licenses/` in the repo. The CI assemble step runs
`python assets/collect_licenses.py $dist` with the build venv active. It copies
`LICENSE.txt`, `THIRD-PARTY-NOTICES.txt`, `licenses/*`, the Python `LICENSE.txt` and the
dist-info license files of each bundled package (`REQUIRED_PACKAGES`). It fails if one is
missing. `validate_build.py --assembled` checks the result. When you add, remove or
upgrade a bundled program, model or package, update `licenses/THIRD-PARTY-NOTICES.txt`
(version, source URL) and `REQUIRED_PACKAGES`. The ffmpeg notice has a 3-year written
offer for the source: keep the matching source available.
The ASIO PortAudio DLLs (`_sounddevice_data/portaudio-binaries/*-asio.dll`, Steinberg
SDK) are removed in CI: sounddevice loads them only when `SD_ENABLE_ASIO` is set.

---

## Version History

| Version | Key changes |
|---|---|
| v1.0.0 | Initial release |
| v1.0.1 | CUDA DLL collection moved from hook to spec; PATH prepend fix for LoadLibraryA |
| v1.0.2 | pyannote collect_all added; torchaudio removed from excludes; einops added |
| v1.0.3 | English-only mode; hallucination fix (condition_on_previous_text=False) |
| v1.0.4 | Remove pandas/sklearn/matplotlib from excludes; collect_all(sklearn) |
| v1.0.5–1.0.10 | Diarization offline fix iterations (hf_hub_download → config.yaml patching) |
| v1.0.11 | Fix CI verify step (explicit venv Python); bump pyannote cache key to v3 |
| v1.0.12 | Bump venv cache v4→v5 (omegaconf missing); add omegaconf explicitly to pip install |
| v1.0.13 | Fix: pass pytorch_model.bin path not snapshot dir to Model.from_pretrained |
| v1.0.14 | Fix: copy all snapshot files to tmp (params.yaml etc.); add traceback logging |
| v1.0.15 | Fix: disable PLDA (references unbundled pyannote/speaker-diarization-community-1) |
| v1.1.0 | Replace pyannote.audio diarization backend with sherpa-onnx (ONNX Runtime, GPU-capable via `provider="cuda"`, no PyTorch dependency); remove torch/torchaudio entirely |
| v1.1.0 (fix) | GPU diarization silently fell back to CPU on real hardware — `onnxruntime_providers_cuda.dll` needs `cufft64_11.dll`; add `nvidia-cufft-cu12` to bundled packages; bump venv cache v6→v7 |
| v1.1.0 (fix 2) | GPU diarization still fell back to CPU — `onnxruntime 1.26.0` (CPU) installed as faster-whisper dep; both it and sherpa_onnx bundle `onnxruntime.dll`; Windows caches by name so whichever loads first wins; fix by prepending `sherpa_onnx/lib/` to PATH in `cuda_setup.py` so the GPU version is cached first; bump venv cache v7→v8 |
| v2.0.0 | Full rewrite: sherpa-onnx for all inference (transcription + diarization + VAD); Vulkan GPU provider (no CUDA DLLs); single .exe via 7-zip SFX + AppData extract-once launcher; live microphone transcription (VAD-chunked, crash-safe PCM); two model variants (Whisper ONNX distil-large-v3, Parakeet TDT 0.6B v3 int8); removes faster-whisper, ctranslate2, nvidia-*-cu12 packages entirely |
| v2.1 | Switch the Whisper model from distil-large-v3 to large-v3-turbo (A/B accuracy winner); remove the Parakeet TDT variant and all `MODEL_VARIANT`/`variant.json` machinery — single model, single CI build; remove fictional Vulkan GPU support (`provider="vulkan"` always silently fell back to CPU) — delete `vulkan_probe.py`, GPU device dropdown, `NUM_THREADS` lifted to config.py |
| v2.1 (Phase 8) | Add whisper.cpp as a second, optional file-transcription engine: `whisper-cli` built in CI with `-DGGML_VULKAN=ON` (any GPU vendor, beam_size=5), bundled in `whispercpp/`. `TranscriptionEngine.transcribe()` uses it when bundled, else falls back unchanged to the sherpa-onnx VAD+greedy path. Unlike v2.0.0's fictional `provider="vulkan"`, this is real GPU acceleration — the device actually used is logged from whisper.cpp's own stderr, never assumed |
| v3.0.0-beta1 | New light, feature-coded UI (teal files, coral recording, amber speakers, sky timestamps, green done); offline guard blocks all non-loopback network access; ffmpeg restricted to local files; temp files cleaned at start and exit; in-app Privacy panel; logo as app icon (`assets/EasyScribe.ico`, generated by `assets/make_icon.py`) on both exes and all windows; Windows file-version resource from `APP_VERSION`; remove stale v1 `build_windows.bat` |
| v3.0.0-beta2 | Launcher compares versions and updates an older install in place (backup + rollback, user data kept); AppUserModelID + shell icon refresh for the taskbar icon; "Make full transcript" after a live recording; live VAD flush on stop; shared `CancelledError`; dependencies pinned in `requirements.txt` (venv cache v4); validation of the assembled app folder in CI |
| v3.0.0-beta3 | PyInstaller `Splash` ("Getting ready") during the one-file extraction; installer waits for the user before an update and uses the app's light CustomTkinter theme; `win_paint.py` stops the black-block flash on refocus (Windows); Show details / Privacy drawn as small outlined buttons |
| v3.0.0 | First full release. Third-party licenses and notices in the bundle (`licenses/`, `assets/collect_licenses.py`, checked by `validate_build.py`); ASIO PortAudio DLLs removed; exe renamed `EasyScribe-v3.0.0.exe` (no `-whisper`); release notes and README explain Mark of the Web (Unblock tip). Code signing deferred to 3.1 |
| v3.1.0 | Record tab (was "Record live") asks first: "Best quality transcript after recording" (default: record only, `MicRecorder.start(feed_vad=False)`, then the whisper.cpp file pipeline starts at stop with timestamps and the speakers switch) or "Show words as I speak" (live sherpa-onnx). "Make best quality transcript" after a live recording or a cancel stays in the Record tab and keeps the live transcript (`(best quality).txt`); mic level meter (`MicRecorder.get_level`) |
