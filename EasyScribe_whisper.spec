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

# ── sounddevice ───────────────────────────────────────────────────────────────
# sounddevice 0.5+ is a single-file module (sounddevice.py) whose C extension
# _sounddevice.pyd lives at the site-packages root, not inside a package dir.
# collect_all() therefore skips binaries; we must add the .pyd explicitly.

_sd_binaries: list = []
_sd_hidden: list = []
try:
    import importlib.util as _sd_ilu
    _sd_ext = _sd_ilu.find_spec("_sounddevice")
    if _sd_ext and _sd_ext.origin:
        _sd_origin = Path(_sd_ext.origin)
        _sd_binaries.append((str(_sd_origin), "."))
        for _dll in _sd_origin.parent.glob("portaudio*.dll"):
            _sd_binaries.append((str(_dll), "."))
        _sd_hidden = ["_sounddevice"]
        print(f"[spec] sounddevice: {_sd_origin.name} + {len(_sd_binaries)-1} portaudio DLL(s)")
    else:
        print("[spec] WARNING: _sounddevice extension not found")
except Exception as e:
    print(f"[spec] WARNING: could not collect sounddevice: {e}")

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
