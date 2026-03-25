"""
main.py - Transcriber7 application entry point.

Import order matters:
  1. config  → sets offline env vars + creates runtime dirs
  2. logger  → sets up log file handlers
  3. gui     → builds CustomTkinter window

This order ensures HF_HUB_OFFLINE and friends are set before any
faster_whisper or huggingface_hub code is imported.
"""

# ── Import order is critical ──────────────────────────────────────────────────
import config      # 1st: sets HF_HOME + offline env vars before any HF import
import cuda_setup  # 2nd: registers CUDA DLL dirs via os.add_dll_directory()
                   #      must run before ctranslate2 / pyannote are imported

import atexit
import logging
import sys
import tkinter as tk
from tkinter import messagebox

from config import (
    APP_NAME,
    FFMPEG_BIN,
    FFPROBE_BIN,
    MODELS_DIR,
    TEMP_DIR,
)
from logger import setup_logging
from transcriber import validate_model_directory


def _cleanup_temp_files() -> None:
    """Remove any stray temp WAV files left over from a previous crash."""
    try:
        for wav in TEMP_DIR.glob("*.wav"):
            try:
                wav.unlink()
            except OSError:
                pass
    except Exception:
        pass


def _check_dependencies() -> list[str]:
    """Return a list of human-readable error strings for missing dependencies."""
    errors: list[str] = []

    if not FFMPEG_BIN.exists():
        errors.append(
            f"ffmpeg.exe not found.\n"
            f"Expected at: {FFMPEG_BIN}\n\n"
            "The bundled ffmpeg folder may be missing. Re-run build_windows.bat."
        )

    if not FFPROBE_BIN.exists():
        errors.append(
            f"ffprobe.exe not found.\n"
            f"Expected at: {FFPROBE_BIN}"
        )

    model_errors = validate_model_directory()
    if model_errors:
        errors.append(
            f"Model files missing or incomplete:\n"
            + "\n".join(f"  • {e}" for e in model_errors)
            + f"\n\nExpected model at: {MODELS_DIR}\n\n"
            "Copy the model snapshot files into that folder, or re-run build_windows.bat."
        )

    return errors


def main() -> None:
    setup_logging()
    log = logging.getLogger(APP_NAME)

    # Register temp-file cleanup on normal and abnormal exit
    atexit.register(_cleanup_temp_files)

    log.info("Checking dependencies…")
    errors = _check_dependencies()

    if errors:
        # Show errors before the main window opens so the user gets a clear
        # message rather than a cryptic crash inside the GUI.
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

    # Import GUI only after dependency check passes (avoids importing customtkinter
    # and tkinterdnd2 before we know they will be needed)
    from gui import TranscriberApp

    app = TranscriberApp()
    app.mainloop()

    log.info("Application exited normally")


if __name__ == "__main__":
    main()
