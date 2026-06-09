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
# sounddevice's C extension may be at the site-packages root (_sounddevice.pyd)
# OR inside the sounddevice package dir (sounddevice/_sounddevice.pyd).
# Glob both locations so this works across sounddevice 0.4.x and 0.5.x.

import site as _site

_sd_binaries: list = []
_sd_hidden: list = []
try:
    for _sp in _site.getsitepackages():
        _sp_path = Path(_sp)
        for _pyd in _sp_path.glob("_sounddevice*.pyd"):
            _sd_binaries.append((str(_pyd), "."))
            _sd_hidden = ["_sounddevice"]
            print(f"[spec] sounddevice (flat): {_pyd.name}")
        for _pyd in _sp_path.glob("sounddevice/_sounddevice*.pyd"):
            _sd_binaries.append((str(_pyd), "sounddevice"))
            if "_sounddevice" not in _sd_hidden:
                _sd_hidden.append("sounddevice._sounddevice")
            print(f"[spec] sounddevice (pkg): {_pyd.name}")
        for _dll in list(_sp_path.glob("portaudio*.dll")) + list(_sp_path.glob("sounddevice/portaudio*.dll")):
            dest = "sounddevice" if "sounddevice" in str(_dll.parent) else "."
            _sd_binaries.append((str(_dll), dest))
    if not _sd_binaries:
        print("[spec] WARNING: no _sounddevice*.pyd found in site-packages")
    else:
        print(f"[spec] sounddevice: {len(_sd_binaries)} file(s) collected")
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
