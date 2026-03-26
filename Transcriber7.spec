# -*- mode: python ; coding: utf-8 -*-
#
# Transcriber7.spec - PyInstaller build specification
#
# Produces a ONEDIR portable build in dist/PortableTranscriber/.
# The model files and ffmpeg binaries are NOT packed into the executable —
# they are copied alongside it by build_windows.bat after PyInstaller runs.
#
# To rebuild:
#   pyinstaller Transcriber7.spec --noconfirm

import sys
from pathlib import Path


def _collect_nvidia_dlls():
    """
    Collect nvidia CUDA package DLLs into the bundle unconditionally.

    hook-nvidia.py is never triggered because nothing imports nvidia.* —
    ctranslate2 loads CUDA DLLs via Windows LoadLibrary, not Python imports.
    Running the collection here (spec executes at build time regardless of
    imports) guarantees the DLLs land in _internal/nvidia/<pkg>/bin/.
    cuda_setup.py then calls os.add_dll_directory() on those dirs at startup.
    """
    result = []
    for sp in sys.path:
        nvidia_root = Path(sp) / "nvidia"
        if not nvidia_root.is_dir():
            continue
        for pkg_dir in nvidia_root.iterdir():
            if not pkg_dir.is_dir():
                continue
            for sub in ("bin", "lib"):
                dll_dir = pkg_dir / sub
                if not dll_dir.is_dir():
                    continue
                for f in dll_dir.iterdir():
                    if f.suffix.lower() in (".dll", ".so") and f.is_file():
                        dest = f"nvidia/{pkg_dir.name}/{sub}"
                        result.append((str(f), dest))
    print(f"[spec] Collected {len(result)} nvidia DLL(s) from site-packages")
    return result

block_cipher = None

# ── Locate CustomTkinter and tkinterdnd2 package directories ─────────────────
# These contain non-Python data files (themes, DLLs) that must be bundled.

def _site_pkg(name: str) -> Path:
    """Find the installed location of a package."""
    import importlib.util
    spec = importlib.util.find_spec(name)
    if spec and spec.submodule_search_locations:
        return Path(list(spec.submodule_search_locations)[0])
    raise FileNotFoundError(f"Cannot find package: {name}")

try:
    _ctk_dir = str(_site_pkg("customtkinter"))
    _dnd_dir = str(_site_pkg("tkinterdnd2"))
except FileNotFoundError as e:
    print(f"WARNING: {e}  — build may fail or features may be missing")
    _ctk_dir = None
    _dnd_dir = None

_datas = []
if _ctk_dir:
    _datas.append((_ctk_dir, "customtkinter"))
if _dnd_dir:
    _datas.append((_dnd_dir, "tkinterdnd2"))

# ── Collect pyannote.audio and dependencies via collect_all ──────────────────
# hiddenimports alone doesn't find data files or native sub-modules inside
# complex packages like pyannote.  collect_all() does the full sweep.
# scipy is also required by pyannote at runtime — do NOT exclude it below.
from PyInstaller.utils.hooks import collect_all as _collect_all

_pyannote_binaries: list = []
_pyannote_hidden: list = []
for _pkg in [
    "pyannote.audio",
    "pyannote.core",
    "pyannote.pipeline",
    "asteroid_filterbanks",
    "speechbrain",
    "sklearn",      # has Cython .pyd extensions — collect_all required, not just hiddenimports
]:
    try:
        _d, _b, _h = _collect_all(_pkg)
        _datas += _d
        _pyannote_binaries += _b
        _pyannote_hidden += _h
        print(f"[spec] collect_all({_pkg!r}): {len(_d)} datas, {len(_b)} bins, {len(_h)} hidden")
    except Exception as _e:
        print(f"[spec] Warning: could not collect {_pkg!r}: {_e}")

# ─────────────────────────────────────────────────────────────────────────────

a = Analysis(
    ["src/main.py"],
    pathex=["src"],          # so imports like `from config import ...` resolve
    binaries=_collect_nvidia_dlls() + _pyannote_binaries,
    datas=_datas,
    hiddenimports=[
        # ctranslate2 loads DLLs dynamically; PyInstaller misses them without this
        "ctranslate2",
        # faster-whisper submodules
        "faster_whisper",
        "faster_whisper.transcribe",
        "faster_whisper.audio",
        "faster_whisper.vad",
        "faster_whisper.tokenizer",
        "faster_whisper.utils",
        "faster_whisper.feature_extractor",
        # tokenizer backend
        "tokenizers",
        "tokenizers.models",
        # huggingface_hub is imported by faster-whisper and pyannote internally
        "huggingface_hub",
        "huggingface_hub.utils",
        # pyannote.audio speaker diarization (optional feature)
        "pyannote.audio",
        "pyannote.audio.pipelines",
        "pyannote.core",
        "asteroid_filterbanks",
        "speechbrain",
        # torch + torchaudio: CPU versions, required by pyannote.audio
        "torch",
        "torch.nn",
        "torchaudio",
        # pyannote.audio runtime deps that PyInstaller may miss (lazy imports)
        "einops",
        "omegaconf",
        "soundfile",
        # tkinterdnd2
        "tkinterdnd2",
        # pyannote.metrics core deps (pyinstaller built-in hooks handle their C extensions)
        "pandas",
        "matplotlib",
        "matplotlib.backends.backend_agg",
        # sklearn top-level (binaries collected via collect_all above)
        "sklearn",
        "sklearn.utils",
    ] + _pyannote_hidden,
    hookspath=["hooks"],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # torch_cuda.dll is ~1.5 GB — excluded to keep app.zip under 2 GB.
        # ctranslate2 GPU inference uses the CUDA runtime DLLs from the
        # separate nvidia-* packages collected by _collect_nvidia_dlls().
        # pyannote runs on CPU torch + CPU torchaudio (both kept).
        # NOTE: torchaudio is NOT excluded — pyannote.audio imports it at load time.
        # Reduce size — safe to exclude (not in pyannote's transitive dep tree)
        # WARNING: Do NOT add pandas, sklearn, matplotlib, scipy, or torchaudio here —
        # all are runtime deps of pyannote.audio and will cause "No module named" crashes.
        # See CLAUDE.md Rule 1 for full explanation.
        "numpy.distutils",
        "PIL",
        "notebook",
        "IPython",
        "cv2",
        "pytest",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# ── Strip torch_cuda.dll (~1.5 GB) — ctranslate2 uses CUDA via nvidia packages
import re as _re
_cuda_dll = _re.compile(r'torch_cuda.*\.dll', _re.IGNORECASE)
a.binaries = TOC([b for b in a.binaries if not _cuda_dll.search(b[0])])

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,   # ONEDIR: keep DLLs alongside exe, not packed inside
    name="PortableTranscriber",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,                # compress if UPX is available
    console=False,           # windowed app — no black console window
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon="assets/icon.ico",  # Uncomment and add an .ico file to use a custom icon
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="PortableTranscriber",   # dist/ subfolder name
)
