# Transcriber7 — Development Rules & Lessons

This file is automatically loaded by Claude Code. Rules here are derived from real mistakes
made during development. Read before making any build or packaging changes.

---

## PyInstaller Packaging Rules

### Rule 1: `excludes` overrides `collect_all()` — never exclude ML transitive deps

**The mistake we kept making:** Adding a package to `excludes` in `Transcriber7.spec` to
reduce bundle size, not realising it was a transitive runtime dependency of `pyannote.audio`.
This caused a whack-a-mole series of `No module named '<X>'` errors on live builds:
- scipy → removed from excludes
- torchaudio → removed from excludes
- pandas → removed from excludes
- sklearn, matplotlib → removed from excludes

**The rule:** Before adding any package to `excludes`, verify it does NOT appear in the
transitive dependency tree of: `pyannote.audio`, `speechbrain`, `asteroid_filterbanks`,
`faster_whisper`, or `ctranslate2`. If in doubt, leave it out of excludes.

**How PyInstaller processes excludes:** The `excludes` list is applied *after* analysis,
including after `collect_all()`. So even if `collect_all('pyannote.audio')` discovers a
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
in `Transcriber7.spec`). Do not rely on hooks for packages that aren't Python-imported.

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
the compiled binaries. Use `collect_all('<package>')` instead. Key packages this applies to:
- `sklearn` (scikit-learn) — dozens of `.pyd` files across subpackages
- `speechbrain`, `asteroid_filterbanks` — already in `collect_all` loop

Packages with PyInstaller built-in hooks (`pandas`, `matplotlib`) can use `hiddenimports`
since the hook handles their binaries automatically.

---

### Rule 5: Bump the venv cache key after any dependency changes

**The rule:** The GitHub Actions venv cache (`venv-win64-py3.11-vN`) is keyed to a static
string. Any change to pip dependencies (new packages, removed excludes, etc.) requires
bumping the version suffix (`v3` → `v4` → ...) to force a clean rebuild. Failing to do
this means the old cached venv is used and changes don't take effect.

---

## Version History

| Version | Key changes |
|---|---|
| v1.0.0 | Initial release |
| v1.0.1 | CUDA DLL collection moved from hook to spec; PATH prepend fix for LoadLibraryA |
| v1.0.2 | pyannote collect_all added; torchaudio removed from excludes; einops added |
| v1.0.3 | English-only mode; hallucination fix (condition_on_previous_text=False) |
| v1.0.4 | Remove pandas/sklearn/matplotlib from excludes; collect_all(sklearn) |
