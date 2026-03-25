"""
ffmpeg_wrapper.py - Audio extraction and probing via bundled ffmpeg.

Uses the bundled ffmpeg.exe to convert any supported media file into a
temporary mono 16 kHz PCM WAV file suitable for Faster Whisper inference.

ffmpeg's progress output is streamed live to the log callback so the user
can always see that something is happening (and how far along it is).
"""

import json
import logging
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

from config import FFMPEG_BIN, FFPROBE_BIN, TEMP_DIR

logger = logging.getLogger(__name__)

_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


# ─── Custom exceptions ────────────────────────────────────────────────────────

class FFmpegNotFoundError(RuntimeError):
    pass

class FFmpegExtractionError(RuntimeError):
    pass

class CancelledError(RuntimeError):
    pass


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _parse_duration(stderr_text: str) -> float:
    """Extract total duration in seconds from ffmpeg stderr output."""
    m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", stderr_text)
    if m:
        h, mn, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
        return h * 3600 + mn * 60 + s
    return 0.0


def _parse_time(line: str) -> float:
    """Extract the current encode position in seconds from a ffmpeg progress line."""
    m = re.search(r"time=(\d+):(\d+):([\d.]+)", line)
    if m:
        h, mn, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
        return h * 3600 + mn * 60 + s
    return -1.0


def _fmt_seconds(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    if h:
        return f"{h}h {m:02d}m {sec:02d}s"
    return f"{m}m {sec:02d}s"


# ─── Public API ───────────────────────────────────────────────────────────────

def validate_ffmpeg() -> None:
    missing = []
    if not FFMPEG_BIN.exists():
        missing.append(f"ffmpeg.exe not found at: {FFMPEG_BIN}")
    if not FFPROBE_BIN.exists():
        missing.append(f"ffprobe.exe not found at: {FFPROBE_BIN}")
    if missing:
        raise FFmpegNotFoundError("\n".join(missing))


def probe_audio(input_path: Path) -> dict:
    """Use ffprobe to read audio stream metadata."""
    if not FFPROBE_BIN.exists():
        return {}
    cmd = [
        str(FFPROBE_BIN),
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_name,sample_rate,channels,duration",
        "-of", "json",
        str(input_path),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, timeout=30,
            creationflags=_CREATE_NO_WINDOW,
        )
        data = json.loads(result.stdout.decode("utf-8", errors="replace"))
        streams = data.get("streams", [])
        if streams:
            return streams[0]
    except Exception as exc:
        logger.debug(f"ffprobe failed for {input_path.name}: {exc}")
    return {}


def _is_already_suitable_wav(input_path: Path) -> bool:
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

    Streams ffmpeg's progress output live to *log_callback* so the user always
    has visibility into what is happening (especially for long video files).
    """
    if cancel_event.is_set():
        raise CancelledError("Cancelled before audio extraction started")

    validate_ffmpeg()

    def log(msg: str) -> None:
        logger.info(msg)
        if log_callback:
            log_callback(msg)

    # ── Fast path: already a suitable WAV ────────────────────────────────────
    if _is_already_suitable_wav(input_path):
        log(f"[ffmpeg] Already mono 16 kHz PCM WAV — copying directly: {input_path.name}")
        temp_wav = TEMP_DIR / f"{uuid.uuid4().hex}.wav"
        import shutil
        shutil.copy2(input_path, temp_wav)
        return temp_wav

    # ── Probe: get duration and audio stream info for richer logging ─────────
    probe = probe_audio(input_path)
    if probe:
        codec   = probe.get("codec_name", "?")
        rate    = probe.get("sample_rate", "?")
        chans   = probe.get("channels", "?")
        dur_raw = probe.get("duration", "")
        dur_str = _fmt_seconds(float(dur_raw)) if dur_raw else "unknown duration"
        log(f"[ffmpeg] Input: codec={codec}, sample_rate={rate} Hz, channels={chans}, duration={dur_str}")
    else:
        log(f"[ffmpeg] Could not probe audio stream — proceeding anyway")

    temp_wav = TEMP_DIR / f"{uuid.uuid4().hex}.wav"
    log(f"[ffmpeg] Extracting audio → mono 16 kHz PCM WAV: {input_path.name}")
    log(f"[ffmpeg] Temp file: {temp_wav.name}")

    cmd = [
        str(FFMPEG_BIN),
        "-y",
        "-i", str(input_path),
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        "-acodec", "pcm_s16le",
        "-stats",          # print progress stats to stderr
        str(temp_wav),
    ]

    logger.debug(f"ffmpeg cmd: {' '.join(cmd)}")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=_CREATE_NO_WINDOW,
        )

        # ── Stream stderr live in a reader thread ─────────────────────────────
        # ffmpeg writes all useful output (duration, progress, errors) to stderr.
        # We collect it in a list so we can also inspect it after the process ends.
        stderr_lines: list[str] = []
        total_duration: list[float] = [0.0]   # mutable container for thread sharing
        last_progress_log: list[float] = [0.0]

        def _read_stderr() -> None:
            for raw in proc.stderr:  # type: ignore[union-attr]
                line = raw.decode("utf-8", errors="replace").rstrip()
                stderr_lines.append(line)

                # Extract total duration once from the header block
                if total_duration[0] == 0.0:
                    d = _parse_duration(line)
                    if d > 0:
                        total_duration[0] = d
                        log(f"[ffmpeg] Media duration: {_fmt_seconds(d)}")

                # Forward progress lines to the log at most once per second
                pos = _parse_time(line)
                if pos >= 0 and total_duration[0] > 0:
                    now = time.monotonic()
                    if now - last_progress_log[0] >= 1.0:
                        last_progress_log[0] = now
                        pct = min(pos / total_duration[0] * 100, 100)
                        log(
                            f"[ffmpeg] Extracting… "
                            f"{_fmt_seconds(pos)} / {_fmt_seconds(total_duration[0])}  "
                            f"({pct:.0f}%)"
                        )
                elif line and not line.startswith("ffmpeg version") and "encoder" not in line.lower():
                    # Forward other non-noise lines (errors, warnings, stream info)
                    if any(kw in line.lower() for kw in ("error", "warn", "invalid", "no such", "stream #")):
                        log(f"[ffmpeg] {line}")

        reader_thread = threading.Thread(target=_read_stderr, daemon=True)
        reader_thread.start()

        # ── Poll for cancellation ─────────────────────────────────────────────
        while proc.poll() is None:
            if cancel_event.is_set():
                logger.info("Cancel requested — terminating ffmpeg")
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
                reader_thread.join(timeout=2)
                if temp_wav.exists():
                    temp_wav.unlink(missing_ok=True)
                raise CancelledError("Audio extraction cancelled by user")
            time.sleep(0.1)

        reader_thread.join(timeout=5)
        return_code = proc.returncode

        if return_code != 0:
            stderr_text = "\n".join(stderr_lines)
            snippet = stderr_text[-800:].strip()
            logger.error(f"ffmpeg failed (rc={return_code}):\n{snippet}")
            if temp_wav.exists():
                temp_wav.unlink(missing_ok=True)
            raise FFmpegExtractionError(
                f"ffmpeg exited with code {return_code}.\n\n{snippet}"
            )

        size_kb = temp_wav.stat().st_size // 1024
        log(f"[ffmpeg] Extraction complete — {size_kb:,} KB written to {temp_wav.name}")
        return temp_wav

    except (CancelledError, FFmpegExtractionError):
        raise
    except Exception as exc:
        if temp_wav.exists():
            temp_wav.unlink(missing_ok=True)
        raise FFmpegExtractionError(f"Unexpected error during audio extraction: {exc}") from exc
