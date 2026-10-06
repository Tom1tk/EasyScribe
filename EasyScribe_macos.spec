# -*- mode: python ; coding: utf-8 -*-
#
# EasyScribe_macos.spec - PyInstaller spec for the macOS app (Apple Silicon).
#
# Produces dist/EasyScribe.app. The CI job "build-macos" then copies in:
#   Contents/MacOS/ffmpeg/        ffmpeg, ffprobe (LGPL build, see the workflow)
#   Contents/MacOS/whispercpp/    whisper-cli (Metal)
#   Contents/Resources/models/    the sherpa-onnx models
#   Contents/Resources/whispercpp/ the whisper.cpp model
# and signs the app again (ad hoc), because the copy breaks the signature.
# config.py finds these folders with IS_MAC_APP / RESOURCES_DIR.
#
# Build: pyinstaller EasyScribe_macos.spec --noconfirm

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all as _collect_all

sys.path.insert(0, str(Path(SPECPATH) / "src"))
from config import APP_VERSION  # noqa: E402  (config only makes folders in dev mode)

block_cipher = None


def _site_pkg(name: str) -> Path:
    import importlib.util
    spec = importlib.util.find_spec(name)
    if spec and spec.submodule_search_locations:
        return Path(list(spec.submodule_search_locations)[0])
    raise FileNotFoundError(f"Cannot find package: {name}")


# ── Static data dirs (themes, tkdnd) ──────────────────────────────────────────

_datas = []
for pkg in ("customtkinter", "tkinterdnd2"):
    try:
        _datas.append((str(_site_pkg(pkg)), pkg))
    except FileNotFoundError as e:
        print(f"WARNING: {e} — build may be incomplete")

_datas += [
    ("assets/EasyScribe.ico", "assets"),
    ("assets/EasyScribe.png", "assets"),  # window icon on macOS (config.APP_ICON_PNG)
]

# ── sherpa-onnx, sounddevice, numpy (CLAUDE.md Rules 4 and 11) ────────────────

_binaries: list = []
_hidden: list = ["sherpa_onnx", "sounddevice", "tkinterdnd2", "numpy", "_sounddevice", "cffi"]
for _pkg in ("sherpa_onnx", "numpy"):
    try:
        _d, _b, _h = _collect_all(_pkg)
        _datas += _d
        _binaries += _b
        _hidden += _h
        print(f"[spec] collect_all('{_pkg}'): {len(_d)} datas, {len(_b)} bins, {len(_h)} hidden")
    except Exception as e:
        print(f"[spec] WARNING: could not collect {_pkg}: {e}")

# sounddevice is CFFI-based; _sounddevice_data/ holds libportaudio.dylib
import importlib.util as _sd_ilu

_sd_spec = _sd_ilu.find_spec("sounddevice")
if _sd_spec and _sd_spec.origin:
    _sd_data_dir = Path(_sd_spec.origin).parent / "_sounddevice_data"
    if _sd_data_dir.is_dir():
        _datas.append((str(_sd_data_dir), "_sounddevice_data"))
        print(f"[spec] sounddevice _sounddevice_data/: {[f.name for f in _sd_data_dir.rglob('*')]}")
    else:
        print(f"[spec] WARNING: _sounddevice_data/ not found at {_sd_data_dir}")

# ─────────────────────────────────────────────────────────────────────────────

a = Analysis(
    ["src/main.py"],
    pathex=["src"],
    binaries=_binaries,
    datas=_datas,
    hiddenimports=_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["PIL", "notebook", "IPython", "cv2", "pytest"],
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
    upx=False,  # UPX breaks Mach-O code signatures
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch="arm64",
    codesign_identity=None,  # ad hoc; CI signs the assembled app again
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="EasyScribe",
)

app = BUNDLE(
    coll,
    name="EasyScribe.app",
    icon="assets/EasyScribe.icns",
    bundle_identifier="io.github.tom1tk.easyscribe",
    version=APP_VERSION,
    info_plist={
        "CFBundleName": "EasyScribe",
        "CFBundleDisplayName": "EasyScribe",
        "CFBundleShortVersionString": APP_VERSION,
        "CFBundleVersion": APP_VERSION,
        "LSMinimumSystemVersion": "14.0",
        "LSApplicationCategoryType": "public.app-category.productivity",
        "NSHighResolutionCapable": True,
        "NSHumanReadableCopyright": "MIT License. See THIRD-PARTY-NOTICES.txt.",
        # Without this text, macOS gives the app silence instead of the microphone
        "NSMicrophoneUsageDescription":
            "EasyScribe records from the microphone to make a transcript. "
            "The audio stays on this Mac.",
        "NSDocumentsFolderUsageDescription":
            "EasyScribe saves recordings and transcripts in Documents/EasyScribe.",
        "NSDesktopFolderUsageDescription":
            "EasyScribe reads the recordings that you choose and saves the transcripts next to them.",
        "NSDownloadsFolderUsageDescription":
            "EasyScribe reads the recordings that you choose and saves the transcripts next to them.",
        "NSRemovableVolumesUsageDescription":
            "EasyScribe reads the recordings that you choose and saves the transcripts next to them.",
    },
)
