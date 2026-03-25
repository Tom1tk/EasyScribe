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

# ─────────────────────────────────────────────────────────────────────────────

a = Analysis(
    ["src/main.py"],
    pathex=["src"],          # so imports like `from config import ...` resolve
    binaries=[],
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
        # torch is needed by pyannote (CPU inference only — torch_cuda.dll
        # is excluded below to keep the bundle under 2 GB)
        "torch",
        "torch.nn",
        # tkinterdnd2
        "tkinterdnd2",
    ],
    hookspath=["hooks"],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # torch_cuda.dll is ~1.5 GB — excluded to keep app.zip under 2 GB.
        # ctranslate2 GPU inference uses the CUDA runtime DLLs from the
        # separate nvidia-* packages collected by hooks/hook-nvidia.py.
        # pyannote runs on CPU torch (torch_cpu.dll, ~300 MB) which is kept.
        "torchaudio",
        # Reduce size — things we never use
        "matplotlib",
        "numpy.distutils",
        "scipy",
        "PIL",
        "notebook",
        "IPython",
        "pandas",
        "sklearn",
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
