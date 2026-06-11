"""
whispercpp_wrapper.py - File transcription via bundled whisper.cpp (whisper-cli).

whisper.cpp does its own internal 30-second-window chunking and timestamping,
so this wrapper hands it the whole file in one subprocess call and parses
--output-json into the same (start_sec, end_sec, text) segment shape the
sherpa-onnx VAD-segment loop in transcriber.py produces — no changes needed
to the transcript-formatting helpers or diarization speaker assignment.

Modeled on ffmpeg_wrapper.py: subprocess + cancellation polling + stderr
streamed to the log callback.
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

import config
from config import TEMP_DIR, WHISPERCPP_BEAM_SIZE, WHISPERCPP_BIN, WHISPERCPP_MODEL

logger = logging.getLogger(__name__)

_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


# ─── Custom exceptions ────────────────────────────────────────────────────────

class WhisperCppNotFoundError(RuntimeError):
    pass

class WhisperCppError(RuntimeError):
    pass

class CancelledError(RuntimeError):
    pass


# ─── Helpers ──────────────────────────────────────────────────────────────────

_DEVICE_RE = re.compile(r"whisper_backend_init_gpu: device \d+: (.+)")
_NO_GPU_RE = re.compile(r"whisper_backend_init_gpu: no GPU found")
_PROGRESS_RE = re.compile(r"whisper_print_progress_callback: progress = (\d+)%")


def is_available() -> bool:
    """True if a bundled whisper-cli binary and GGUF model are present."""
    return WHISPERCPP_BIN.is_file() and WHISPERCPP_MODEL.is_file()


def validate_whispercpp() -> None:
    missing = []
    if not WHISPERCPP_BIN.is_file():
        missing.append(f"whisper-cli not found at: {WHISPERCPP_BIN}")
    if not WHISPERCPP_MODEL.is_file():
        missing.append(f"Whisper GGUF model not found at: {WHISPERCPP_MODEL}")
    if missing:
        raise WhisperCppNotFoundError("\n".join(missing))


def _parse_device_line(line: str) -> str | None:
    """
    Return a human-readable device description from a whisper.cpp stderr
    line, or None if the line isn't a device-init line.

    Even CPU-only builds attempt GPU init and log the outcome either way
    (see EASYSCRIBE_ACTION_PLAN.md Phase 8.1 findings) — never assume Vulkan
    was used just because the binary was built with -DGGML_VULKAN=ON.
    """
    m = _DEVICE_RE.search(line)
    if m:
        return m.group(1).strip()
    if _NO_GPU_RE.search(line):
        return "CPU (no GPU found)"
    return None


def _parse_progress_line(line: str) -> int | None:
    m = _PROGRESS_RE.search(line)
    return int(m.group(1)) if m else None


def _parse_output_json(json_path: Path) -> list[tuple[float, float, str]]:
    """Parse whisper-cli's --output-json file into (start_sec, end_sec, text) segments."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    segments: list[tuple[float, float, str]] = []
    for entry in data.get("transcription", []):
        text = entry.get("text", "").strip()
        if not text:
            continue
        offsets = entry.get("offsets", {})
        start = offsets.get("from", 0) / 1000.0
        end = offsets.get("to", 0) / 1000.0
        segments.append((start, end, text))
    return segments


# ─── Public API ───────────────────────────────────────────────────────────────

def transcribe_file(
    audio_path: Path,
    cancel_event: threading.Event,
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> list[tuple[float, float, str]]:
    """
    Transcribe *audio_path* (mono 16 kHz WAV) with whisper-cli using beam search.

    Runs one whisper-cli process for the whole file. Returns
    (start_sec, end_sec, text) segments.
    """
    if cancel_event.is_set():
        raise CancelledError("Cancelled before transcription started")

    validate_whispercpp()

    def log(msg: str) -> None:
        logger.info(msg)
        if log_callback:
            log_callback(msg)

    out_json = TEMP_DIR / f"whispercpp_{uuid.uuid4().hex}.json"
    out_stem = out_json.with_suffix("")

    cmd = [
        str(WHISPERCPP_BIN),
        "-m", str(WHISPERCPP_MODEL),
        "-f", str(audio_path),
        "-bs", str(WHISPERCPP_BEAM_SIZE),
        "-t", str(config.NUM_THREADS),
        "-pp",
        "--output-json",
        "--output-file", str(out_stem),
    ]

    logger.debug(f"whisper-cli cmd: {' '.join(cmd)}")
    log(f"[whisper.cpp] Transcribing with beam_size={WHISPERCPP_BEAM_SIZE}: {audio_path.name}")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=_CREATE_NO_WINDOW,
        )

        stderr_lines: list[str] = []
        device_logged = [False]

        def _read_stderr() -> None:
            for raw in proc.stderr:  # type: ignore[union-attr]
                line = raw.decode("utf-8", errors="replace").rstrip()
                stderr_lines.append(line)

                if not device_logged[0]:
                    device = _parse_device_line(line)
                    if device:
                        device_logged[0] = True
                        log(f"[whisper.cpp] Device: {device}")

                pct = _parse_progress_line(line)
                if pct is not None and progress_callback:
                    progress_callback(pct / 100.0)

        reader_thread = threading.Thread(target=_read_stderr, daemon=True)
        reader_thread.start()

        while proc.poll() is None:
            if cancel_event.is_set():
                logger.info("Cancel requested — terminating whisper-cli")
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
                reader_thread.join(timeout=2)
                raise CancelledError("Transcription cancelled by user")
            time.sleep(0.1)

        reader_thread.join(timeout=5)
        return_code = proc.returncode

        if return_code != 0:
            stderr_text = "\n".join(stderr_lines)
            snippet = stderr_text[-800:].strip()
            logger.error(f"whisper-cli failed (rc={return_code}):\n{snippet}")
            raise WhisperCppError(f"whisper-cli exited with code {return_code}.\n\n{snippet}")

        if not out_json.exists():
            raise WhisperCppError(f"whisper-cli completed but did not produce {out_json.name}")

        segments = _parse_output_json(out_json)
        log(f"[whisper.cpp] Decoded {len(segments)} segment(s)")
        if progress_callback:
            progress_callback(1.0)
        return segments

    except (CancelledError, WhisperCppError):
        raise
    except Exception as exc:
        raise WhisperCppError(f"Unexpected error during whisper.cpp transcription: {exc}") from exc
    finally:
        out_json.unlink(missing_ok=True)
