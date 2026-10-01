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
# Before Python starts, the one-file bootloader unpacks everything (about
# 1.5 GB) to %TEMP%. That takes up to a minute, so a Splash shows a
# "Getting ready" window first (assets/splash.png, made by
# assets/make_splash.py). launcher.py closes it when its own window opens.
#
# Build: pyinstaller launcher/launcher.spec --noconfirm

import sys
from pathlib import Path

sys.path.insert(0, str(Path(SPECPATH).parent / "assets"))
from version_resource import version_info as _version_info

block_cipher = None

a = Analysis(
    ["launcher.py"],
    pathex=[],
    binaries=[],
    datas=[
        # Embed the pre-built app bundle — path is relative to this spec file
        ("../app.bundle", "."),
        # Window icon for the installer window
        ("../assets/EasyScribe.ico", "."),
    ],
    # customtkinter: the installer uses the same look as the app. Its data
    # files (themes, fonts) come from the hooks-contrib hook.
    hiddenimports=["tkinter", "tkinter.filedialog", "customtkinter"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinterdnd2",
        "sherpa_onnx",
        "sounddevice",
        "numpy",
        "PIL",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Shown by the bootloader while it unpacks; Tcl/Tk is unpacked first for it.
# No text_pos: the bootloader would show each file name, and the only big
# file is app.bundle. The image itself says what is happening.
splash = Splash(
    "../assets/splash.png",
    binaries=a.binaries,
    datas=a.datas,
    always_on_top=False,
)

exe = EXE(
    pyz,
    a.scripts,
    splash,
    splash.binaries,
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
    console=False,      # GUI installer — no terminal window
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="../assets/EasyScribe.ico",
    version=_version_info("EasyScribe installer", "launcher.exe"),
)
