# -*- mode: python ; coding: utf-8 -*-
#
# EasyScribe_parakeet.spec - PyInstaller ONEDIR spec for the Parakeet TDT variant.
#
# Identical to EasyScribe_whisper.spec except models/parakeet/ replaces models/whisper/.
#
# Build: pyinstaller EasyScribe_parakeet.spec --noconfirm

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all as _collect_all

block_cipher = None


def _site_pkg(name: str) -> Path:
    import importlib.util
    spec = importlib.util.find_spec(name)
    if spec and spec.submodule_search_locations:
        return Path(list(spec.submodule_search_locations)[0])
    raise FileNotFoundError(f"Cannot find package: {name}")


# ── Static data dirs ──────────────────────────────────────────────────────────

_datas = []
for pkg in ("customtkinter", "tkinterdnd2"):
    try:
        _datas.append((str(_site_pkg(pkg)), pkg))
    except FileNotFoundError as e:
        print(f"WARNING: {e} — build may be incomplete")

# ── Bundled models ────────────────────────────────────────────────────────────

_datas += [
    ("models/parakeet/", "models/parakeet"),
    ("models/silero_vad.onnx", "models"),
    ("models/diarization/", "models/diarization"),
    ("models/variant.json", "models"),
]

# ── sherpa-onnx ───────────────────────────────────────────────────────────────

_sherpa_binaries: list = []
_sherpa_hidden: list = []
try:
    _d, _b, _h = _collect_all("sherpa_onnx")
    _datas += _d
    _sherpa_binaries += _b
    _sherpa_hidden += _h
    print(f"[spec] collect_all('sherpa_onnx'): {len(_d)} datas, {len(_b)} bins, {len(_h)} hidden")
except Exception as e:
    print(f"[spec] WARNING: could not collect sherpa_onnx: {e}")

# ── sounddevice ───────────────────────────────────────────────────────────────
# sounddevice 0.5.x is CFFI-based. Key files (all at site-packages root):
#   sounddevice.py       — main Python module
#   _sounddevice.py      — CFFI-generated wrapper
#   _sounddevice_data/   — directory containing the bundled PortAudio DLL
#   _cffi_backend.pyd    — the CFFI C extension (collected via cffi hiddenimport)
# We must add _sounddevice_data/ as a datas entry so PyInstaller bundles the DLL.

import importlib.util as _sd_ilu

_sd_binaries: list = []
_sd_hidden: list = []
try:
    _sd_spec = _sd_ilu.find_spec("sounddevice")
    if _sd_spec and _sd_spec.origin:
        _sd_origin = Path(_sd_spec.origin)
        _sd_parent = _sd_origin.parent          # site-packages (flat) or sounddevice/ (pkg)
        _sd_is_pkg = (_sd_origin.name == "__init__.py")
        _sd_dest = "sounddevice" if _sd_is_pkg else "."

        # Collect _sounddevice_data/ (contains PortAudio DLL)
        _sd_data_dir = _sd_parent / "_sounddevice_data"
        if _sd_data_dir.is_dir():
            _datas.append((str(_sd_data_dir), "_sounddevice_data"))
            _sd_dlls = [f.name for f in _sd_data_dir.iterdir()]
            print(f"[spec] sounddevice _sounddevice_data/: {_sd_dlls}")
        else:
            print(f"[spec] WARNING: _sounddevice_data/ not found at {_sd_data_dir}")

        _sd_hidden = ["_sounddevice", "cffi"]
    else:
        print("[spec] WARNING: sounddevice not found via find_spec")
except Exception as e:
    print(f"[spec] WARNING: sounddevice collection: {e}")

# ── numpy: explicit collection to ensure .pyd extensions are bundled ─────────

_np_binaries: list = []
_np_hidden: list = []
try:
    _d, _b, _h = _collect_all("numpy")
    _datas += _d
    _np_binaries += _b
    _np_hidden += _h
    print(f"[spec] collect_all('numpy'): {len(_d)} datas, {len(_b)} bins, {len(_h)} hidden")
except Exception as e:
    print(f"[spec] WARNING: could not collect numpy: {e}")

# ─────────────────────────────────────────────────────────────────────────────

a = Analysis(
    ["src/main.py"],
    pathex=["src"],
    binaries=_sherpa_binaries + _sd_binaries + _np_binaries,
    datas=_datas,
    hiddenimports=[
        "sherpa_onnx",
        "sounddevice",
        "tkinterdnd2",
        "numpy",
    ] + _sherpa_hidden + _sd_hidden + _np_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "PIL",
        "notebook",
        "IPython",
        "cv2",
        "pytest",
        "faster_whisper",
        "ctranslate2",
        "huggingface_hub",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="EasyScribe",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="EasyScribe",
)
