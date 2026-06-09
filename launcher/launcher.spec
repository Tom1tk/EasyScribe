# -*- mode: python ; coding: utf-8 -*-
#
# launcher.spec - PyInstaller ONEFILE spec for the EasyScribe portable launcher.
#
# app.bundle (zip of the full EasyScribe ONEDIR build) must be created
# BEFORE running this spec. CI creates it at the project root:
#   Compress-Archive -Path "dist\EasyScribe\*" -DestinationPath "app.bundle"
#
# The embedded app.bundle is extracted to <exe_dir>/EasyScribe/ on first run.
#
# Build: pyinstaller launcher/launcher.spec --noconfirm

block_cipher = None

a = Analysis(
    ["launcher.py"],
    pathex=[],
    binaries=[],
    datas=[
        # Embed the pre-built app bundle — path is relative to this spec file
        ("../app.bundle", "."),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
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
    upx=False,          # app.bundle is already compressed — UPX won't help
    upx_exclude=[],
    console=True,       # show extraction progress
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
