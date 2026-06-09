# -*- mode: python ; coding: utf-8 -*-
#
# launcher.spec - PyInstaller ONEFILE spec for the EasyScribe launcher.
#
# Produces a tiny (~5 MB) single-file exe that:
#   1. Checks if EasyScribe is already installed in AppData
#   2. If not: extracts app.bundle (zip) to AppData
#   3. Launches EasyScribe.exe
#
# Build: pyinstaller launcher/launcher.spec --noconfirm

block_cipher = None

a = Analysis(
    ["launcher.py"],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude everything that isn't stdlib
        "customtkinter",
        "tkinterdnd2",
        "sherpa_onnx",
        "sounddevice",
        "numpy",
        "PIL",
        "ctranslate2",
        "faster_whisper",
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
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="launcher",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    console=True,   # show progress during extraction
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
