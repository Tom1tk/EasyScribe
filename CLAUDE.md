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

### Rule 7: sherpa-onnx Vulkan provider — no DLL bundling needed

Unlike CUDA, Vulkan uses the system GPU driver. `provider="vulkan"` in `OfflineRecognizerConfig`
works on any machine with a GPU driver installed, with no additional DLLs required.
If Vulkan fails at runtime (e.g. headless CI), catch `RuntimeError` from `OfflineRecognizer(cfg)`
and retry with `provider="cpu"`.

### Rule 8: Silero VAD model must be downloaded and bundled explicitly

`sherpa_onnx.get_default_vad_model()` does NOT exist in sherpa-onnx 1.13.2.
Download `silero_vad.onnx` from the sherpa-onnx GitHub releases and reference it via
`config.VAD_MODEL_PATH`. Bundle as `("models/silero_vad.onnx", "models")` in both specs.

### Rule 9: Parakeet TDT uses OfflineTransducerModelConfig, not a CTC config

`OfflineNemoCtcModelConfig` does not exist as of sherpa-onnx 1.13.2.
Parakeet TDT 0.6B v3 int8 is a transducer model — use `OfflineTransducerModelConfig`
with `encoder_filename`, `decoder_filename`, `joiner_filename`.
The real NeMo CTC class is `OfflineNemoEncDecCtcModelConfig` but it is not used here.

### Rule 10: venv cache key has no variant suffix — both matrix jobs share it

CI builds two variants (whisper, parakeet) in parallel. They install identical pip deps.
Cache key `venv-win64-py3.11-v2` has no variant suffix so the second job always hits cache.
Bump v2 → v3 etc. to force a clean rebuild after adding/removing packages.

### Rule 11: Always list numpy explicitly in the CI pip install command

`numpy` is not a declared Python dep of `sherpa-onnx` on all platforms. If it is absent from
the venv, `collect_all('numpy')` silently catches `ImportError` and bundles nothing, then
`hiddenimports=["numpy"]` resolves to an empty module — the build succeeds but the exe crashes
with `ModuleNotFoundError: No module named 'numpy'` at runtime.

Always include `numpy` explicitly in the `pip install` line in the workflow. Run
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

`tests/test_sherpa_api.py` builds the config for both variants and constructs
`_OfflineRecognizer(cfg)` against nonexistent model paths, asserting `RuntimeError`
(bad path) rather than `TypeError` (bad constructor signature). Run it locally
(`python tests/test_sherpa_api.py`, no model files or GPU needed) before triggering a
build — it now also runs as an early CI step, right after `pip install`, before any
model downloads.

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
