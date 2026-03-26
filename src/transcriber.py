"""
transcriber.py - Faster Whisper transcription engine wrapper.

Manages lazy model loading, automatic CUDA/CPU device selection, segment
iteration with cancellation support, and transcript file writing.

The model is loaded once on the first transcription call and then cached
for subsequent calls, avoiding long startup delays.
"""

import logging
import threading
import time
from pathlib import Path
from typing import Callable

from config import (
    MODELS_DIR,
    REQUIRED_MODEL_FILES,
    WHISPER_BEAM_SIZE,
    WHISPER_LANGUAGE,
    WHISPER_VAD_FILTER,
    WHISPER_VAD_MIN_SILENCE_MS,
)

logger = logging.getLogger(__name__)


# ─── Custom exceptions ────────────────────────────────────────────────────────


class ModelNotFoundError(RuntimeError):
    """Raised when the model directory or required files are missing."""


class TranscriptionError(RuntimeError):
    """Raised when transcription fails for reasons other than cancellation."""


class CancelledError(RuntimeError):
    """Raised when transcription is cancelled by the user."""


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm for timestamped output."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _fmt_seconds(s: float) -> str:
    """Human-readable duration, e.g. '1m 23s'."""
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    if h:
        return f"{h}h {m:02d}m {sec:02d}s"
    return f"{m}m {sec:02d}s"


def list_gpus() -> list[dict[str, str | int]]:
    """
    Return a list of available CUDA GPUs.

    Each entry is a dict with keys:
      - index (int): CUDA device index
      - name  (str): GPU display name

    Tries torch first (gives proper GPU names); falls back to ctranslate2
    so the app works without torch installed (as in the pre-built release).
    Returns an empty list if no CUDA GPUs are available.
    """
    gpus: list[dict[str, str | int]] = []

    # ── Primary: torch (provides GPU names) ──────────────────────────────────
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpus.append({"index": i, "name": torch.cuda.get_device_name(i)})
            return gpus
    except ImportError:
        pass  # torch not installed — use ctranslate2 fallback
    except Exception as exc:
        logger.debug(f"torch GPU enumeration failed: {exc}")

    # ── Fallback: ctranslate2 (always present, no GPU names) ─────────────────
    try:
        import ctranslate2  # type: ignore
        count = ctranslate2.get_cuda_device_count()
        for i in range(count):
            gpus.append({"index": i, "name": f"CUDA GPU {i}"})
    except Exception as exc:
        logger.debug(f"ctranslate2 GPU enumeration failed: {exc}")

    return gpus


def _detect_device(gpu_index: int | None = None) -> tuple[str, str, int | None]:
    """
    Detect the best available inference device.

    Parameters
    ----------
    gpu_index:
        Specific GPU device index to use (0-based).
        Pass None to auto-select the first available GPU,
        or -1 to force CPU.

    Returns (device, compute_type, resolved_device_index):
      - ("cuda", "float16", N)  if a CUDA-capable GPU is available
      - ("cpu", "int8", None)   otherwise

    Works with or without torch installed; ctranslate2 is the fallback.
    """
    if gpu_index == -1:
        logger.info("CPU explicitly selected by user")
        return "cpu", "int8", None

    gpus = list_gpus()

    if gpus:
        count = len(gpus)
        if gpu_index is not None and not (0 <= gpu_index < count):
            logger.warning(
                f"Requested GPU index {gpu_index} out of range "
                f"({count} GPU(s) found) — falling back to GPU 0"
            )
            gpu_index = 0
        elif gpu_index is None:
            gpu_index = 0

        name = gpus[gpu_index]["name"]
        logger.info(f"Using GPU {gpu_index}: {name} — float16")
        return "cuda", "float16", gpu_index

    logger.info("No CUDA GPUs found — using CPU int8")
    return "cpu", "int8", None


def validate_model_directory() -> list[str]:
    """
    Check that the model directory exists and contains all required files.

    Returns a list of missing file names (empty list means all good).
    """
    if not MODELS_DIR.exists():
        return [f"Model directory not found: {MODELS_DIR}"]

    missing = []
    for fname in REQUIRED_MODEL_FILES:
        if not (MODELS_DIR / fname).exists():
            missing.append(fname)
    return missing


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _build_plain_transcript(
    raw_segments: list[tuple[float, float, str]],
    add_timestamps: bool,
) -> str:
    """Build a plain (no speaker labels) transcript string from raw segments."""
    if add_timestamps:
        lines = [
            f"[{_format_timestamp(start)} --> {_format_timestamp(end)}] {text}"
            for start, end, text in raw_segments
        ]
        return "\n".join(lines)
    return " ".join(text for _, _, text in raw_segments)


# ─── Engine ───────────────────────────────────────────────────────────────────


class TranscriptionEngine:
    """
    Lazy-loading wrapper around faster_whisper.WhisperModel.

    Thread-safe: model loading is protected by a lock, but concurrent
    transcription calls are not supported. The GUI should only ever run
    one transcription thread at a time.
    """

    def __init__(self) -> None:
        self._model = None
        self._model_lock = threading.Lock()
        self._device: str | None = None
        self._compute_type: str | None = None
        self._device_index: int | None = None
        # -1 means force CPU; None means auto-detect; 0+ means specific GPU index
        self.preferred_gpu_index: int | None = None

    def _ensure_model_loaded(
        self,
        status_callback: Callable[[str], None],
    ) -> None:
        """Load the model if it has not been loaded yet."""
        with self._model_lock:
            if self._model is not None:
                return

            missing = validate_model_directory()
            if missing:
                detail = "\n".join(missing)
                raise ModelNotFoundError(
                    f"Model validation failed:\n{detail}\n\n"
                    f"Expected model at: {MODELS_DIR}"
                )

            status_callback("Loading Model")
            logger.info(f"Loading model from: {MODELS_DIR}")

            device, compute_type, resolved_index = _detect_device(self.preferred_gpu_index)
            self._device = device
            self._compute_type = compute_type
            self._device_index = resolved_index

            device_label = f"cuda:{resolved_index}" if device == "cuda" else "cpu"
            logger.info(
                f"Device: {device_label}, compute_type: {compute_type}"
            )

            # Validate model files and report sizes before loading
            model_files = list(MODELS_DIR.iterdir()) if MODELS_DIR.exists() else []
            for f in sorted(model_files):
                size_mb = f.stat().st_size / (1024 * 1024)
                logger.info(f"  Model file: {f.name}  ({size_mb:.1f} MB)")

            logger.info(
                f"[Model] Loading faster-whisper from {MODELS_DIR.name} "
                f"on {device_label} ({compute_type}) — this may take a moment…"
            )

            # Import here (not at module level) so offline env vars are already set
            from faster_whisper import WhisperModel  # type: ignore

            model_kwargs: dict = {
                "model_size_or_path": str(MODELS_DIR),
                "device": device,
                "compute_type": compute_type,
                "local_files_only": True,   # Explicit: never attempt any download
                "num_workers": 2,
                "download_root": None,      # Not used; belt-and-suspenders
            }
            if resolved_index is not None:
                model_kwargs["device_index"] = resolved_index

            load_start = time.monotonic()
            try:
                self._model = WhisperModel(**model_kwargs)
            except Exception as cuda_exc:
                # CUDA runtime DLLs missing (e.g. cublas64_12.dll not found) —
                # fall back to CPU automatically rather than crashing.
                if device == "cuda":
                    logger.warning(
                        f"[Model] CUDA load failed ({cuda_exc}); retrying on CPU int8…"
                    )
                    status_callback("Loading Model (CPU fallback)")
                    self._device = "cpu"
                    self._compute_type = "int8"
                    self._device_index = None
                    model_kwargs.update(
                        device="cpu", compute_type="int8"
                    )
                    model_kwargs.pop("device_index", None)
                    self._model = WhisperModel(**model_kwargs)
                else:
                    raise
            load_elapsed = time.monotonic() - load_start
            logger.info(f"[Model] Loaded successfully in {load_elapsed:.1f}s")

    def reload_model(self) -> None:
        """
        Unload and reload the model (needed when the user changes GPU selection).
        """
        with self._model_lock:
            if self._model is not None:
                del self._model
                self._model = None
                logger.info("Model unloaded for reload")

    def unload_model(self) -> None:
        """Release model memory. Useful if running on a memory-constrained system."""
        with self._model_lock:
            if self._model is not None:
                del self._model
                self._model = None
                logger.info("Model unloaded from memory")

    def transcribe(
        self,
        audio_path: Path,
        output_path: Path,
        add_timestamps: bool,
        cancel_event: threading.Event,
        status_callback: Callable[[str], None],
        progress_callback: Callable[[float], None],
        log_callback: Callable[[str], None],
        diarize: bool = False,
    ) -> None:
        """
        Transcribe *audio_path* and write the result to *output_path*.

        Parameters
        ----------
        audio_path:
            Path to the mono 16 kHz WAV file produced by ffmpeg_wrapper.
        output_path:
            Destination .txt file path.
        add_timestamps:
            If True, prefix each segment with [HH:MM:SS.mmm --> HH:MM:SS.mmm].
        cancel_event:
            Threading event; checked after each segment. Raises CancelledError.
        status_callback:
            Called with status strings for the GUI status label.
        progress_callback:
            Called with float 0.0–1.0 to update the progress bar.
        log_callback:
            Called with each transcribed segment text for the GUI log box.

        Raises
        ------
        ModelNotFoundError  – model files missing or corrupt
        CancelledError      – user cancelled
        TranscriptionError  – unexpected inference error
        """
        if cancel_event.is_set():
            raise CancelledError("Cancelled before transcription started")

        # Load model (no-op if already loaded)
        self._ensure_model_loaded(status_callback)

        status_callback("Transcribing")
        logger.info(f"Transcribing: {audio_path.name} → {output_path.name}")
        log_callback(f"[Whisper] Starting transcription of {audio_path.name}…")
        log_callback(f"[Whisper] beam_size={WHISPER_BEAM_SIZE}, vad_filter={WHISPER_VAD_FILTER}, language={'auto' if not WHISPER_LANGUAGE else WHISPER_LANGUAGE}")
        log_callback(f"[Whisper] Output options: timestamps={'on' if add_timestamps else 'off'}, speakers={'on' if diarize else 'off'}")
        log_callback(f"[Whisper] (Formatted transcript is written to the .txt output file, not the log)")

        try:
            log_callback("[Whisper] Initialising model pipeline (first-time setup may take a few seconds)…")
            infer_start = time.monotonic()
            segments_iter, info = self._model.transcribe(
                str(audio_path),
                beam_size=WHISPER_BEAM_SIZE,
                language=WHISPER_LANGUAGE,
                vad_filter=WHISPER_VAD_FILTER,
                vad_parameters={"min_silence_duration_ms": WHISPER_VAD_MIN_SILENCE_MS},
                word_timestamps=False,
                # Prevent repetition-loop hallucinations ("yeah yeah yeah..."):
                # conditioning on previous text creates feedback when audio is
                # ambiguous or quiet — disabling it stops the loop.
                condition_on_previous_text=False,
            )
        except Exception as exc:
            logger.exception(f"Model transcribe() call failed: {exc}")
            raise TranscriptionError(f"Transcription failed: {exc}") from exc

        duration: float = info.duration if info.duration else 0.0
        language_detected: str = info.language or "unknown"
        device_info = f"cuda:{self._device_index}" if self._device == "cuda" and self._device_index is not None else (self._device or "cpu")

        logger.info(f"Audio duration: {duration:.1f}s, detected language: {language_detected}")
        log_callback(f"[Whisper] Audio duration : {_fmt_seconds(duration)}")
        log_callback(f"[Whisper] Language       : {language_detected}")
        log_callback(f"[Whisper] Device         : {device_info} / {self._compute_type}")
        log_callback(f"[Whisper] Processing segments — output will appear below as it's decoded…")

        # (start_sec, end_sec, text) — always keep times for diarization mapping
        raw_segments: list[tuple[float, float, str]] = []
        segment_count = 0
        word_count = 0
        last_progress_log = time.monotonic()

        try:
            for segment in segments_iter:
                if cancel_event.is_set():
                    raise CancelledError("Transcription cancelled by user")

                # Update progress bar
                if duration > 0:
                    progress = min(segment.end / duration, 1.0)
                    progress_callback(progress)

                text = segment.text.strip()
                if not text:
                    continue

                segment_count += 1
                word_count += len(text.split())
                raw_segments.append((segment.start, segment.end, text))
                log_callback(text)

                # Periodic progress heartbeat — once per second in wall time
                now = time.monotonic()
                if now - last_progress_log >= 1.0:
                    last_progress_log = now
                    pos_str = _fmt_seconds(segment.end)
                    dur_str = _fmt_seconds(duration) if duration > 0 else "?"
                    pct = f"{min(segment.end / duration * 100, 100):.0f}%" if duration > 0 else "?%"
                    elapsed = now - infer_start
                    speed = segment.end / elapsed if elapsed > 0 else 0
                    log_callback(
                        f"[Progress] {pos_str} / {dur_str} ({pct}) — "
                        f"{segment_count} segments, {word_count} words — "
                        f"{speed:.1f}x realtime"
                    )

        except CancelledError:
            raise
        except Exception as exc:
            logger.exception(f"Error during segment iteration: {exc}")
            raise TranscriptionError(f"Error during transcription: {exc}") from exc

        total_elapsed = time.monotonic() - infer_start
        log_callback(
            f"[Whisper] Transcription complete — {segment_count} segments, "
            f"{word_count} words in {_fmt_seconds(total_elapsed)}"
        )

        # ── Optional speaker diarization ──────────────────────────────────────
        if diarize and raw_segments:
            status_callback("Identifying Speakers")
            try:
                from diarizer import DiarizationEngine, DiarizationError  # type: ignore
                _diarizer = DiarizationEngine()
                turns = _diarizer.diarize(audio_path, log_callback)
                assigned = _diarizer.assign_speakers(raw_segments, turns)
                # assigned: list of (speaker, text, start, end)
                # Build output lines grouped by consecutive speaker
                output_lines: list[str] = []
                prev_speaker: str | None = None
                for speaker, text, start, end in assigned:
                    # Normalise label: SPEAKER_00 → Speaker 0
                    label = speaker.replace("SPEAKER_", "Speaker ").replace("_", " ").title()
                    if add_timestamps:
                        line = f"[{label}] [{_format_timestamp(start)} --> {_format_timestamp(end)}] {text}"
                    elif label != prev_speaker:
                        output_lines.append(f"\n[{label}]")
                        line = text
                    else:
                        line = text
                    output_lines.append(line)
                    prev_speaker = label
                transcript_text = " ".join(output_lines).strip()
            except Exception as exc:
                logger.warning(f"Diarization failed, falling back to plain transcript: {exc}")
                log_callback(f"[Diarize] Warning: {exc} — writing plain transcript")
                transcript_text = _build_plain_transcript(raw_segments, add_timestamps)
        else:
            transcript_text = _build_plain_transcript(raw_segments, add_timestamps)

        # ── Write transcript ──────────────────────────────────────────────────
        status_callback("Writing Transcript")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        log_callback(f"[Output] Writing transcript to: {output_path}")
        logger.info(f"Writing transcript to: {output_path}")

        try:
            with open(output_path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(transcript_text)
                fh.write("\n")
        except PermissionError as exc:
            raise TranscriptionError(
                f"Cannot write to output file.\n"
                f"The file may be open in another program: {output_path}\n\n{exc}"
            ) from exc
        except OSError as exc:
            raise TranscriptionError(
                f"Failed to write transcript file: {output_path}\n\n{exc}"
            ) from exc

        size_kb = output_path.stat().st_size // 1024
        progress_callback(1.0)
        logger.info(f"Transcript written: {segment_count} segments, {output_path.stat().st_size} bytes")
        log_callback(f"[Done] Saved: {output_path}  ({size_kb} KB)")
