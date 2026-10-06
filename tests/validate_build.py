#!/usr/bin/env python3
"""
Post-build artifact validator. Run immediately after PyInstaller and before
creating app.bundle so broken bundles are caught before the 30-minute compress step.

Usage: python tests/validate_build.py <dist_dir> [--assembled]
       python tests/validate_build.py dist/EasyScribe
       python tests/validate_build.py dist/EasyScribe --assembled
       python tests/validate_build.py dist/EasyScribe.app --assembled   (macOS)

--assembled also checks the files that the CI assembly step copies next to the
exe (models, ffmpeg, whisper.cpp, license files). Run it after that step,
before app.bundle.

Exits 0 on success, 1 if any check fails.
"""
import sys
from pathlib import Path


def _find_pyds(directory: Path, pattern: str = "**/*.pyd") -> list[Path]:
    return list(directory.glob(pattern))


class _Layout:
    """Where each part of the build is, on Windows or in a macOS .app.

    exe_dir: the folder of the executable (config.BASE_DIR) - ffmpeg, whispercpp
    res_dir: config.RESOURCES_DIR - models, the whisper.cpp model, licenses
    internal: the PyInstaller folder with the Python packages
    """

    def __init__(self, dist: Path) -> None:
        self.mac = dist.suffix == ".app"
        if self.mac:
            self.exe_dir = dist / "Contents" / "MacOS"
            self.res_dir = dist / "Contents" / "Resources"
            self.internal = dist / "Contents" / "Frameworks"
            self.exe = self.exe_dir / "EasyScribe"
            self.ext = ""
            self.ext_glob = "**/*.so"
            self.icon = "assets/EasyScribe.png"
        else:
            self.exe_dir = self.res_dir = dist
            self.internal = dist / "_internal"
            self.exe = dist / "EasyScribe.exe"
            self.ext = ".exe"
            self.ext_glob = "**/*.pyd"
            self.icon = "assets/EasyScribe.ico"

    def assembled_files(self) -> list[tuple[Path, str]]:
        e = self.exe_dir
        r = self.res_dir
        return [
            (e, f"ffmpeg/ffmpeg{self.ext}"),
            (e, f"ffmpeg/ffprobe{self.ext}"),
            (r, "models/silero_vad.onnx"),
            (r, "models/whisper/turbo-encoder.int8.onnx"),
            (r, "models/whisper/turbo-decoder.int8.onnx"),
            (r, "models/whisper/turbo-tokens.txt"),
            (r, "models/diarization/segmentation.onnx"),
            (r, "models/diarization/embedding.onnx"),
            (e, f"whispercpp/whisper-cli{self.ext}"),
            (r, "whispercpp/ggml-large-v3-turbo-q5_0.bin"),
        ]


_REPO_LICENSES = Path(__file__).resolve().parent.parent / "licenses"

# License files from assets/collect_licenses.py. ffmpeg is GPL v3, so its
# license must ship with it; the other licenses require their notices too.
_LICENSE_FILES = ["LICENSE.txt", "THIRD-PARTY-NOTICES.txt", "licenses/Python-LICENSE.txt"] + [
    f"licenses/{f.name}" for f in sorted(_REPO_LICENSES.glob("*"))
    if f.is_file() and f.name != "THIRD-PARTY-NOTICES.txt"
]


def _check_assembled(lay: _Layout, ok: list[str], errors: list[str]) -> None:
    files = lay.assembled_files() + [(lay.res_dir, f) for f in _LICENSE_FILES]
    for base, rel in files:
        path = base / rel
        if path.is_file() and path.stat().st_size > 0:
            ok.append(f"{rel}  ({path.stat().st_size // 1024} KB)")
        else:
            errors.append(f"MISSING: {rel}  (expected {path})")

    if not lay.mac:
        dlls = list((lay.exe_dir / "whispercpp").glob("*.dll"))
        if dlls:
            ok.append(f"whispercpp/*.dll  ({len(dlls)} file(s))")
        else:
            errors.append("MISSING: whispercpp/*.dll  (whisper-cli.exe cannot start without them)")

    pkg_licenses = [p for p in (lay.res_dir / "licenses" / "python-packages").rglob("*") if p.is_file()]
    if pkg_licenses:
        ok.append(f"licenses/python-packages/  ({len(pkg_licenses)} file(s))")
    else:
        errors.append("MISSING: licenses/python-packages/  (run assets/collect_licenses.py)")

    # Only libportaudio64bit.dll is loaded; the ASIO builds are removed (see CI).
    asio = list(lay.internal.glob("_sounddevice_data/portaudio-binaries/*-asio.dll"))
    if asio:
        errors.append(f"UNEXPECTED: {asio[0].name}  (ASIO PortAudio DLLs must be removed)")

    # Models load from RESOURCES_DIR only; a copy in the PyInstaller folder doubles the size.
    stale = lay.internal / "models"
    if stale.exists() and stale.resolve() != (lay.res_dir / "models").resolve():
        errors.append(f"UNEXPECTED: {stale}  (models must not be packed by PyInstaller)")
    else:
        ok.append(f"{lay.internal.name}/models absent  (no duplicate models)")


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--assembled"]
    assembled = "--assembled" in sys.argv[1:]
    if len(args) != 1:
        print("Usage: validate_build.py <dist_dir> [--assembled]", file=sys.stderr)
        sys.exit(1)

    dist = Path(args[0])
    lay = _Layout(dist)
    internal = lay.internal

    errors: list[str] = []
    ok: list[str] = []

    # ── The executable ────────────────────────────────────────────────────────
    exe = lay.exe
    if exe.is_file():
        ok.append(f"{exe.name}  ({exe.stat().st_size // 1024} KB)")
    else:
        errors.append(f"MISSING: {exe}")

    # ── The PyInstaller folder must exist ─────────────────────────────────────
    if not internal.is_dir():
        errors.append(f"MISSING: {internal.name}/ directory under {dist}")
        # Can't check anything else without _internal
        _report(ok, errors)
        return

    # ── Per-package checks ────────────────────────────────────────────────────
    # Packages with their own subdirectory in _internal:
    dir_checks = [
        ("numpy",        "numpy",        1),   # numpy/*.pyd (*.so on macOS)
        ("sherpa_onnx",  "sherpa_onnx",  1),   # sherpa_onnx/*.pyd (*.so on macOS)
        ("customtkinter","customtkinter", 0),   # directory only
        ("tkinterdnd2",  "tkinterdnd2",  0),   # directory only
    ]

    for name, subdir, min_pyds in dir_checks:
        pkg_dir = internal / subdir
        if not pkg_dir.is_dir():
            errors.append(f"MISSING package dir: {name}  (expected {pkg_dir})")
            continue
        if min_pyds > 0:
            pyds = _find_pyds(pkg_dir, lay.ext_glob)
            if len(pyds) < min_pyds:
                errors.append(
                    f"MISSING binaries: {name}  (found {len(pyds)} {lay.ext_glob} files, need >= {min_pyds})"
                )
            else:
                ok.append(f"{name}  ({len(pyds)} extension file(s))")
        else:
            ok.append(f"{name}  (directory present)")

    # sounddevice 0.5.x (CFFI): ships _sounddevice_data/ with the PortAudio DLL
    # Accept either the CFFI data dir, a classic pyd, or a portaudio DLL
    sd_data_dir = internal / "_sounddevice_data"
    sd_pyds = list(internal.glob("**/_sounddevice*.pyd"))
    sd_dlls = list(internal.glob("**/portaudio*.dll"))
    if sd_data_dir.is_dir():
        contents = [f.name for f in sd_data_dir.iterdir()]
        ok.append(f"sounddevice  (_sounddevice_data/: {contents})")
    elif sd_pyds:
        ok.append(f"sounddevice  ({sd_pyds[0].name})")
    elif sd_dlls:
        ok.append(f"sounddevice  ({sd_dlls[0].name})")
    else:
        errors.append(
            f"MISSING: sounddevice audio data  "
            f"(no _sounddevice_data/, _sounddevice*.pyd, or portaudio*.dll under {internal})\n"
            f"    Required for microphone recording."
        )

    # App icon for the windows (config.APP_ICON, APP_ICON_PNG on macOS)
    icon = internal / lay.icon
    if icon.is_file():
        ok.append(f"{lay.icon}  (window icon)")
    else:
        errors.append(f"MISSING: window icon  (expected {icon})")

    if lay.mac:
        plist = dist / "Contents" / "Info.plist"
        text = plist.read_text(encoding="utf-8", errors="replace") if plist.is_file() else ""
        if "NSMicrophoneUsageDescription" in text:
            ok.append("Info.plist  (microphone usage text)")
        else:
            errors.append(f"MISSING: NSMicrophoneUsageDescription in {plist}  (no microphone without it)")

    if assembled:
        _check_assembled(lay, ok, errors)

    _report(ok, errors)


def _report(ok: list[str], errors: list[str]) -> None:
    print("\n-- Build validation -----------------------------------------------------")
    for line in ok:
        print(f"  OK   {line}")
    for line in errors:
        print(f"  FAIL {line}")
    print("-------------------------------------------------------------------------\n")

    if errors:
        print(f"BUILD VALIDATION FAILED — {len(errors)} issue(s) above.", file=sys.stderr)
        sys.exit(1)

    print(f"All {len(ok)} checks passed.")


if __name__ == "__main__":
    main()
