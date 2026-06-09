"""
launcher.py - EasyScribe portable installer / launcher.

Compiled as a PyInstaller ONEFILE with app.bundle embedded as a data file.
The final .exe IS the installer — no separate SFX or 7-zip required.

Behaviour:
  1. If EasyScribe.exe already exists in INSTALL_DIR -> launch it, exit.
  2. Otherwise: find app.bundle in sys._MEIPASS (embedded at build time).
  3. Extract app.bundle (zip) to INSTALL_DIR with a progress print.
  4. Launch MAIN_EXE via subprocess.Popen, exit.

INSTALL_DIR = <directory containing this exe>/EasyScribe/
This means the app installs next to wherever the user places the .exe —
on a USB stick, a shared folder, or a local drive — with no AppData or
admin access required.
"""

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

VERSION = "2.0.0"

# Install next to the launcher .exe, wherever the user placed it
_exe_dir = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent.parent
INSTALL_DIR = _exe_dir / "EasyScribe"
MAIN_EXE = INSTALL_DIR / "EasyScribe.exe"


def _find_bundle() -> Path | None:
    """Locate app.bundle — embedded in sys._MEIPASS when frozen, or at project root in dev."""
    if getattr(sys, "frozen", False):
        candidate = Path(sys._MEIPASS) / "app.bundle"
        if candidate.is_file():
            return candidate
    candidate = Path(__file__).parent.parent / "app.bundle"
    return candidate if candidate.is_file() else None


def _extract_bundle(bundle_path: Path) -> bool:
    """Extract app.bundle zip to INSTALL_DIR. Returns True on success."""
    print(f"Installing EasyScribe {VERSION} to {INSTALL_DIR}...")

    if INSTALL_DIR.exists() and not MAIN_EXE.exists():
        print("Cleaning incomplete previous install...")
        shutil.rmtree(INSTALL_DIR, ignore_errors=True)

    INSTALL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(bundle_path, "r") as zf:
            members = zf.namelist()
            total = len(members)
            for i, member in enumerate(members, start=1):
                zf.extract(member, INSTALL_DIR)
                if i % 500 == 0 or i == total:
                    pct = int(100 * i / total)
                    print(f"  Extracting... {pct}%", end="\r", flush=True)
        print("\nExtraction complete.            ")
        return True
    except Exception as exc:
        print(f"\nExtraction failed: {exc}")
        shutil.rmtree(INSTALL_DIR, ignore_errors=True)
        return False


def _launch() -> None:
    print(f"Launching {MAIN_EXE.name}...")
    subprocess.Popen([str(MAIN_EXE)], cwd=str(INSTALL_DIR))


def main() -> None:
    if MAIN_EXE.is_file():
        _launch()
        return

    bundle = _find_bundle()
    if bundle is None:
        print("ERROR: app.bundle not found. Re-download the installer.")
        input("Press Enter to exit.")
        sys.exit(1)

    if not _extract_bundle(bundle):
        print("ERROR: Installation failed. Check you have write access to this folder.")
        input("Press Enter to exit.")
        sys.exit(1)

    if not MAIN_EXE.is_file():
        print(f"ERROR: {MAIN_EXE} not found after extraction.")
        input("Press Enter to exit.")
        sys.exit(1)

    _launch()


if __name__ == "__main__":
    main()
