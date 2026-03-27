"""
config.py - Application configuration, path resolution, and offline enforcement.

IMPORTANT: This module MUST be imported before any faster_whisper or huggingface
import anywhere in the process. It sets environment variables that prevent all
network access by the ML libraries.
"""

import os
import sys
from pathlib import Path

# ─── Offline enforcement ──────────────────────────────────────────────────────
# Set these BEFORE any huggingface/transformers/faster-whisper import.
# HF_HOME must be set before HF_HUB_OFFLINE so that pyannote.audio finds its
# bundled models in our local cache rather than the user's home directory.
#
# NOTE: get_base_dir() is defined below; we need a forward reference here.
# We resolve BASE_DIR inline rather than calling the function.
_frozen = getattr(sys, "frozen", False)
_base = Path(sys.executable).parent.resolve() if _frozen else Path(__file__).parent.parent.resolve()
os.environ["HF_HOME"] = str(_base / "models" / "hf_cache")
# Set HF_HUB_CACHE explicitly so huggingface_hub's module-level constant is correct
# even if the library is imported before HF_HOME takes effect.
os.environ["HF_HUB_CACHE"] = str(_base / "models" / "hf_cache" / "hub")

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HUGGINGFACE_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# Belt-and-suspenders: block proxy at OS level for this process
os.environ["NO_PROXY"] = "*"

# ─── App metadata ─────────────────────────────────────────────────────────────

APP_NAME = "Transcriber7"
APP_VERSION = "1.0.16"
MODEL_FOLDER_NAME = "faster-whisper-large-v3-turbo"

# ─── Path resolution ──────────────────────────────────────────────────────────


def get_base_dir() -> Path:
    """
    Return the application base directory.

    - When running as a PyInstaller frozen executable: the folder containing
      the .exe (dist/PortableTranscriber/).
    - When running from source in development: the project root (parent of src/).
    """
    if getattr(sys, "frozen", False):
        # sys.executable = ...PortableTranscriber/PortableTranscriber.exe
        return Path(sys.executable).parent.resolve()
    else:
        # __file__ = .../Transcriber7/src/config.py  →  parent.parent = project root
        return Path(__file__).parent.parent.resolve()


BASE_DIR: Path = get_base_dir()

# Core runtime directories
MODELS_DIR: Path = BASE_DIR / "models" / MODEL_FOLDER_NAME
# Hub cache directory for the pyannote speaker-diarization-3.1 model.
# snapshot_download() in CI writes to this path; at runtime HF_HUB_OFFLINE=1
# ensures pyannote loads from here without any network access.
DIARIZATION_MODELS_DIR: Path = (
    BASE_DIR / "models" / "hf_cache" / "hub" / "models--pyannote--speaker-diarization-3.1"
)
FFMPEG_DIR: Path = BASE_DIR / "ffmpeg"
FFMPEG_BIN: Path = FFMPEG_DIR / "ffmpeg.exe"
FFPROBE_BIN: Path = FFMPEG_DIR / "ffprobe.exe"
LOGS_DIR: Path = BASE_DIR / "logs"
TEMP_DIR: Path = BASE_DIR / "temp"

# Ensure runtime directories exist at startup (safe even if already present)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ─── Supported media extensions ───────────────────────────────────────────────

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {
        # Video
        ".mp4",
        ".mkv",
        ".mov",
        ".avi",
        ".webm",
        # Audio
        ".mp3",
        ".wav",
        ".m4a",
        ".flac",
        ".ogg",
        ".opus",
        ".aac",
    }
)

# ─── Required model files ─────────────────────────────────────────────────────

REQUIRED_MODEL_FILES: frozenset[str] = frozenset(
    {
        "config.json",
        "model.bin",
        "preprocessor_config.json",
        "tokenizer.json",
        "vocabulary.json",
    }
)

# ─── Transcription defaults ───────────────────────────────────────────────────

WHISPER_BEAM_SIZE: int = 5
WHISPER_VAD_FILTER: bool = True
WHISPER_VAD_MIN_SILENCE_MS: int = 500
WHISPER_LANGUAGE: str | None = "en"  # Force English; set to None for auto-detect

# Disk space safety threshold: warn if less than this many bytes free
MIN_FREE_DISK_BYTES: int = 512 * 1024 * 1024  # 512 MB

# Log file rotation: keep this many log files
MAX_LOG_FILES: int = 10
