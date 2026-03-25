"""
ffmpeg_wrapper.py - Audio extraction and probing via bundled ffmpeg.

Uses the bundled ffmpeg.exe to convert any supported media file into a
temporary mono 16 kHz PCM WAV file suitable for Faster Whisper inference.

All subprocess calls use list form (never shell=True) so paths with spaces
work correctly on Windows.
"""

import json
import logging
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

from config import FFMPEG_BIN, FFPROBE_BIN, TEMP_DIR

logger = logging.getLogger(__name__)

# Windows flag to suppress console window flash when spawning ffmpeg
_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


# ─── Custom exceptions ────────────────────────────────────────────────────────


class FFmpegNotFoundError(RuntimeError):
    """Raised when the bundled ffmpeg.exe cannot be located."""


class FFmpegExtractionError(RuntimeError):
    """Raised when ffmpeg fails to extract or convert audio."""


class CancelledError(RuntimeError):
    """Raised when the operation is cancelled by the user."""


# ─── Public API ───────────────────────────────────────────────────────────────


def validate_ffmpeg() -> None:
    """
    Check that both ffmpeg.exe and ffprobe.exe are present.

    Raises FFmpegNotFoundError with a descriptive message if either is missing.
    """
    missing = []
    if not FFMPEG_BIN.exists():
        missing.append(f"ffmpeg.exe not found at: {FFMPEG_BIN}")
    if not FFPROBE_BIN.exists():
        missing.append(f"ffprobe.exe not found at: {FFPROBE_BIN}")
    if missing:
        raise FFmpegNotFoundError("\n".join(missing))


def probe_audio(input_path: Path) -> dict:
    """
    Use ffprobe to read audio stream metadata from a media file.

    Returns a dict with keys: codec_name, sample_rate, channels.
    Returns an empty dict if probing fails or no audio stream exists.
    """
    if not FFPROBE_BIN.exists():
        return {}

    cmd = [
        str(FFPROBE_BIN),
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_name,sample_rate,channels",
        "-of", "json",
        str(input_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=30,
            creationflags=_CREATE_NO_WINDOW,
        )
        data = json.loads(result.stdout.decode("utf-8", errors="replace"))
        streams = data.get("streams", [])
        if streams:
            return streams[0]
    except Exception as exc:
        logger.debug(f"ffprobe probe failed for {input_path.name}: {exc}")

    return {}


def _is_already_suitable_wav(input_path: Path) -> bool:
    """
    Return True if the file is already a mono 16 kHz PCM WAV.

    If so, we can skip ffmpeg extraction and use the file directly.
    """
    if input_path.suffix.lower() != ".wav":
        return False

    info = probe_audio(input_path)
    return (
        info.get("codec_name") == "pcm_s16le"
        and str(info.get("sample_rate", "0")) == "16000"
        and str(info.get("channels", "0")) == "1"
    )


def extract_audio(
    input_path: Path,
    cancel_event: threading.Event,
    log_callback: Callable[[str], None] | None = None,
) -> Path:
    """
    Extract and convert audio from *input_path* to a temporary mono 16 kHz WAV.

    Parameters
    ----------
    input_path:
        Path to any supported media file.
    cancel_event:
        Threading event; if set during extraction, the operation is terminated
        and CancelledError is raised.
    log_callback:
        Optional callable that receives informational strings for the GUI log.

    Returns
    -------
    Path
        Path to the temporary WAV file.  The **caller** is responsible for
        deleting this file (ideally in a ``finally`` block).

    Raises
    ------
    FFmpegNotFoundError
        If ffmpeg.exe is not present at the expected location.
    FFmpegExtractionError
        If ffmpeg exits with a non-zero return code.
    CancelledError
        If cancel_event is set before or during extraction.
    """
    if cancel_event.is_set():
        raise CancelledError("Cancelled before audio extraction started")

    validate_ffmpeg()

    # Fast path: file is already suitable
    if _is_already_suitable_wav(input_path):
        logger.info(f"Skipping extraction — file is already mono 16 kHz PCM WAV: {input_path.name}")
        if log_callback:
            log_callback(f"[Audio] Already suitable WAV, using directly: {input_path.name}")
        # Return the original path — caller must NOT delete an original file.
        # We signal this by returning a copy in temp so caller cleanup is safe.
        temp_wav = TEMP_DIR / f"{uuid.uuid4().hex}.wav"
        import shutil
        shutil.copy2(input_path, temp_wav)
        return temp_wav

    temp_wav = TEMP_DIR / f"{uuid.uuid4().hex}.wav"
    logger.info(f"Extracting audio: {input_path.name} → {temp_wav.name}")
    if log_callback:
        log_callback(f"[Audio] Extracting: {input_path.name}")

    cmd = [
        str(FFMPEG_BIN),
        "-y",                    # overwrite without asking
        "-i", str(input_path),   # input file
        "-ac", "1",              # mono
        "-ar", "16000",          # 16 kHz sample rate
        "-vn",                   # discard video stream
        "-acodec", "pcm_s16le",  # 16-bit PCM little-endian
        str(temp_wav),           # output
    ]

    logger.debug(f"ffmpeg cmd: {' '.join(cmd)}")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=_CREATE_NO_WINDOW,
        )

        # Poll for cancellation every 100 ms while ffmpeg runs
        while proc.poll() is None:
            if cancel_event.is_set():
                logger.info("Cancel requested — terminating ffmpeg")
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
                if temp_wav.exists():
                    temp_wav.unlink(missing_ok=True)
                raise CancelledError("Audio extraction cancelled by user")
            time.sleep(0.1)

        return_code = proc.returncode
        stderr_bytes = proc.stderr.read() if proc.stderr else b""
        stderr_text = stderr_bytes.decode("utf-8", errors="replace")

        if return_code != 0:
            # Show last 600 chars of stderr which usually contains the error
            snippet = stderr_text[-600:].strip()
            logger.error(f"ffmpeg failed (rc={return_code}): {snippet}")
            if temp_wav.exists():
                temp_wav.unlink(missing_ok=True)
            raise FFmpegExtractionError(
                f"ffmpeg failed with exit code {return_code}.\n\n{snippet}"
            )

        logger.info(f"Audio extraction complete: {temp_wav.name} ({temp_wav.stat().st_size // 1024} KB)")
        if log_callback:
            log_callback(f"[Audio] Extracted successfully ({temp_wav.stat().st_size // 1024} KB)")

        return temp_wav

    except (CancelledError, FFmpegExtractionError):
        raise
    except Exception as exc:
        if temp_wav.exists():
            temp_wav.unlink(missing_ok=True)
        raise FFmpegExtractionError(f"Unexpected error during audio extraction: {exc}") from exc
