# -*- mode: python ; coding: utf-8 -*-
#
# EasyScribe_whisper.spec - PyInstaller ONEDIR spec for the Whisper ONNX variant.
#
# Produces dist/EasyScribe/ containing EasyScribe.exe + _internal/.
# Models and ffmpeg are bundled by the CI assembly step (not packed here).
#
# Build: pyinstaller EasyScribe_whisper.spec --noconfirm

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


# ── Static data dirs (themes, DLLs, Tcl extensions) ─────────────────────────

_datas = []
for pkg in ("customtkinter", "tkinterdnd2"):
    try:
        _datas.append((str(_site_pkg(pkg)), pkg))
    except FileNotFoundError as e:
        print(f"WARNING: {e} — build may be incomplete")

# ── Bundled models ────────────────────────────────────────────────────────────
# Models are placed here by the CI download step before PyInstaller runs.
# variant.json is written by CI to bake the variant into the bundle.

_datas += [
    ("models/whisper/", "models/whisper"),
    ("models/silero_vad.onnx", "models"),
    ("models/diarization/", "models/diarization"),
    ("models/variant.json", "models"),
]

# ── sherpa-onnx: compiled extensions + .libs dir ─────────────────────────────

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

# ── sounddevice: pyd extension + portaudio DLL ───────────────────────────────

_sd_binaries: list = []
_sd_hidden: list = []
try:
    _d, _b, _h = _collect_all("sounddevice")
    _datas += _d
    _sd_binaries += _b
    _sd_hidden += _h
    print(f"[spec] collect_all('sounddevice'): {len(_d)} datas, {len(_b)} bins, {len(_h)} hidden")
except Exception as e:
    print(f"[spec] WARNING: could not collect sounddevice: {e}")

# ─────────────────────────────────────────────────────────────────────────────

a = Analysis(
    ["src/main.py"],
    pathex=["src"],
    binaries=_sherpa_binaries + _sd_binaries,
    datas=_datas,
    hiddenimports=[
        "sherpa_onnx",
        "sounddevice",
        "sounddevice._sounddevice",
        "tkinterdnd2",
    ] + _sherpa_hidden + _sd_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "numpy.distutils",
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
