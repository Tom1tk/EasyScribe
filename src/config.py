"""
config.py - Application configuration and path resolution for EasyScribe v2.0.

sherpa-onnx replaces faster-whisper; no HuggingFace dependencies remain.
Single model: Whisper ONNX large-v3-turbo.
"""

import os
import sys
from pathlib import Path

# ─── App metadata ─────────────────────────────────────────────────────────────

APP_NAME = "EasyScribe"
APP_VERSION = "2.0.0"

# ─── Path resolution ──────────────────────────────────────────────────────────


def get_base_dir() -> Path:
    """
    Return the application base directory.

    - Frozen (PyInstaller): folder containing EasyScribe.exe (the EasyScribe/ install dir).
    - Development: project root (parent of src/).
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent.resolve()
    return Path(__file__).parent.parent.resolve()


BASE_DIR: Path = get_base_dir()

# ─── Output directory ─────────────────────────────────────────────────────────

DEFAULT_OUTPUT_DIR: Path = BASE_DIR / "recordings"

# ─── Whisper ONNX model paths (large-v3-turbo) ────────────────────────────────

_WHISPER_DIR: Path = BASE_DIR / "models" / "whisper"
WHISPER_ENCODER: Path = _WHISPER_DIR / "turbo-encoder.int8.onnx"
WHISPER_DECODER: Path = _WHISPER_DIR / "turbo-decoder.int8.onnx"
WHISPER_TOKENS: Path = _WHISPER_DIR / "turbo-tokens.txt"

# ─── VAD model (Silero) ────────────────────────────────────────────────────────

VAD_MODEL_PATH: Path = BASE_DIR / "models" / "silero_vad.onnx"

# ─── VAD constants ────────────────────────────────────────────────────────────

VAD_SAMPLE_RATE: int = 16000
VAD_CHUNK_SAMPLES: int = 512

# ─── Diarization model paths ──────────────────────────────────────────────────

DIARIZATION_DIR: Path = BASE_DIR / "models" / "diarization"
DIARIZATION_SEGMENTATION_MODEL: Path = DIARIZATION_DIR / "segmentation.onnx"
DIARIZATION_EMBEDDING_MODEL: Path = DIARIZATION_DIR / "embedding.onnx"

# ─── ffmpeg paths ─────────────────────────────────────────────────────────────

FFMPEG_DIR: Path = BASE_DIR / "ffmpeg"
FFMPEG_BIN: Path = FFMPEG_DIR / "ffmpeg.exe"
FFPROBE_BIN: Path = FFMPEG_DIR / "ffprobe.exe"

# ─── Runtime directories ──────────────────────────────────────────────────────

LOGS_DIR: Path = BASE_DIR / "logs"
TEMP_DIR: Path = BASE_DIR / "temp"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ─── Supported media extensions ───────────────────────────────────────────────

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".mp4", ".mkv", ".mov", ".avi", ".webm",
        ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac",
    }
)

# ─── Inference performance ─────────────────────────────────────────────────────
# sherpa-onnx is CPU-only here (no Vulkan provider — see CLAUDE.md Rule 7).

NUM_THREADS: int = min(4, os.cpu_count() or 4)

# ─── Miscellaneous ────────────────────────────────────────────────────────────

MIN_FREE_DISK_BYTES: int = 512 * 1024 * 1024  # 512 MB
MAX_LOG_FILES: int = 10
