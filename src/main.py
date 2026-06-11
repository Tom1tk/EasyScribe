"""
main.py - EasyScribe v2.0 application entry point.

Import order:
  1. config  — establishes BASE_DIR and runtime dirs
  2. logger  — sets up rotating log file
  3. recovery — scans for orphaned PCM recordings (before GUI)
  4. gui     — builds the CustomTkinter window
"""

import config  # 1st: BASE_DIR + runtime dirs must be set before anything else

import atexit
import logging
import sys
import tkinter as tk
from tkinter import messagebox

from config import APP_NAME, DEFAULT_OUTPUT_DIR, FFMPEG_BIN, FFPROBE_BIN, TEMP_DIR
from logger import setup_logging
from transcriber import validate_model_directory


def _cleanup_temp_files() -> None:
    try:
        for wav in TEMP_DIR.glob("*.wav"):
            try:
                wav.unlink()
            except OSError:
                pass
    except Exception:
        pass


def _check_dependencies() -> list[str]:
    errors: list[str] = []

    if not FFMPEG_BIN.exists():
        errors.append(
            f"ffmpeg.exe not found.\nExpected at: {FFMPEG_BIN}\n\n"
            "The bundled ffmpeg folder may be missing."
        )

    if not FFPROBE_BIN.exists():
        errors.append(f"ffprobe.exe not found.\nExpected at: {FFPROBE_BIN}")

    model_errors = validate_model_directory()
    if model_errors:
        errors.append(
            "Model files missing or incomplete:\n"
            + "\n".join(f"  • {e}" for e in model_errors)
        )

    return errors


def main() -> None:
    setup_logging()
    log = logging.getLogger(APP_NAME)

    atexit.register(_cleanup_temp_files)

    # Create default output directory on first launch
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Offer to recover any recordings interrupted by a previous crash
    try:
        from recovery import prompt_and_recover
        prompt_and_recover(DEFAULT_OUTPUT_DIR)
    except Exception as exc:
        log.warning(f"Recovery check failed: {exc}")

    log.info("Checking dependencies…")
    errors = _check_dependencies()

    if errors:
        root = tk.Tk()
        root.withdraw()
        root.update()
        messagebox.showerror(
            f"{APP_NAME} — Missing Dependencies",
            "\n\n".join(errors),
            parent=root,
        )
        root.destroy()
        log.error(f"Startup aborted — missing dependencies: {errors}")
        sys.exit(1)

    log.info("Dependencies OK — launching GUI")

    from gui import TranscriberApp

    app = TranscriberApp()
    app.mainloop()

    log.info("Application exited normally")


if __name__ == "__main__":
    main()
