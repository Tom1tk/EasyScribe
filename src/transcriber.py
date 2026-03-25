"""
transcriber.py - Faster Whisper transcription engine wrapper.

Manages lazy model loading, automatic CUDA/CPU device selection, segment
iteration with cancellation support, and transcript file writing.

The model is loaded once on the first transcription call and then cached
for subsequent calls, avoiding long startup delays.
"""

import logging
import threading
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


def list_gpus() -> list[dict[str, str | int]]:
    """
    Return a list of available CUDA GPUs.

    Each entry is a dict with keys:
      - index (int): CUDA device index
      - name  (str): GPU display name

    Returns an empty list if CUDA is unavailable or torch is not installed.
    """
    gpus: list[dict[str, str | int]] = []
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpus.append({"index": i, "name": torch.cuda.get_device_name(i)})
    except Exception:
        pass
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
      - ("cuda", "float16", N)  if a CUDA-capable NVIDIA GPU is available
      - ("cpu", "int8", None)   otherwise
    """
    if gpu_index == -1:
        logger.info("CPU explicitly selected by user")
        return "cpu", "int8", None

    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            # If a specific index was requested, validate it
            if gpu_index is not None:
                if 0 <= gpu_index < count:
                    name = torch.cuda.get_device_name(gpu_index)
                    logger.info(f"Using GPU {gpu_index}: {name} — float16")
                    return "cuda", "float16", gpu_index
                else:
                    logger.warning(
                        f"Requested GPU index {gpu_index} is out of range "
                        f"(only {count} GPU(s) available) — falling back to GPU 0"
                    )
                    gpu_index = 0
            else:
                gpu_index = 0

            name = torch.cuda.get_device_name(gpu_index)
            logger.info(f"Auto-selected GPU {gpu_index}: {name} — float16")
            return "cuda", "float16", gpu_index
        else:
            logger.info("torch available but no CUDA GPU — falling back to CPU int8")
    except ImportError:
        logger.info("torch not importable — using CPU int8")
    except Exception as exc:
        logger.warning(f"CUDA detection error ({exc}) — using CPU int8")

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

            logger.info(
                f"Device: {device}, compute_type: {compute_type}, "
                f"device_index: {resolved_index}"
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

            self._model = WhisperModel(**model_kwargs)
            logger.info("Model loaded successfully")

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

        try:
            segments_iter, info = self._model.transcribe(
                str(audio_path),
                beam_size=WHISPER_BEAM_SIZE,
                language=WHISPER_LANGUAGE,
                vad_filter=WHISPER_VAD_FILTER,
                vad_parameters={"min_silence_duration_ms": WHISPER_VAD_MIN_SILENCE_MS},
                word_timestamps=False,
            )
        except Exception as exc:
            logger.exception(f"Model transcribe() call failed: {exc}")
            raise TranscriptionError(f"Transcription failed: {exc}") from exc

        duration: float = info.duration if info.duration else 0.0
        language_detected: str = info.language or "unknown"
        logger.info(
            f"Audio duration: {duration:.1f}s, detected language: {language_detected}"
        )
        device_info = self._device or "?"
        if self._device == "cuda" and self._device_index is not None:
            device_info = f"cuda:{self._device_index}"
        log_callback(
            f"[Info] Duration: {duration:.1f}s | Language: {language_detected} | "
            f"Device: {device_info}/{self._compute_type}"
        )

        collected_lines: list[str] = []

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

                if add_timestamps:
                    line = (
                        f"[{_format_timestamp(segment.start)} --> "
                        f"{_format_timestamp(segment.end)}] {text}"
                    )
                else:
                    line = text

                collected_lines.append(line)
                log_callback(line)

        except CancelledError:
            raise
        except Exception as exc:
            logger.exception(f"Error during segment iteration: {exc}")
            raise TranscriptionError(f"Error during transcription: {exc}") from exc

        # ── Write transcript ──────────────────────────────────────────────────
        status_callback("Writing Transcript")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing transcript to: {output_path}")

        try:
            separator = "\n" if add_timestamps else " "
            transcript_text = separator.join(collected_lines)
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

        progress_callback(1.0)
        logger.info(
            f"Transcript written: {len(collected_lines)} segments, "
            f"{output_path.stat().st_size} bytes"
        )
        log_callback(f"[Done] Saved: {output_path}")
