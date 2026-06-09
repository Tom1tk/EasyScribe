"""
config.py - Application configuration and path resolution for EasyScribe v2.0.

sherpa-onnx replaces faster-whisper; no HuggingFace dependencies remain.
Model variant (whisper or parakeet) is baked in via models/variant.json at build time.
"""

import json
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

    - Frozen (PyInstaller): folder containing the .exe (AppData install dir).
    - Development: project root (parent of src/).
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent.resolve()
    return Path(__file__).parent.parent.resolve()


BASE_DIR: Path = get_base_dir()

# ─── Output directory ─────────────────────────────────────────────────────────

DEFAULT_OUTPUT_DIR: Path = Path.home() / "Documents" / "EasyScribe Recordings"

# ─── Model variant ────────────────────────────────────────────────────────────


def _load_model_variant() -> str:
    """Read variant from models/variant.json (frozen) or EASYSCRIBE_MODEL_VARIANT env (dev)."""
    variant_file = BASE_DIR / "models" / "variant.json"
    if variant_file.is_file():
        try:
            data = json.loads(variant_file.read_text(encoding="utf-8"))
            return str(data.get("variant", "whisper"))
        except Exception:
            pass
    return os.environ.get("EASYSCRIBE_MODEL_VARIANT", "whisper")


MODEL_VARIANT: str = _load_model_variant()

# ─── Whisper ONNX model paths (distil-large-v3) ───────────────────────────────

_WHISPER_DIR: Path = BASE_DIR / "models" / "whisper"
WHISPER_ENCODER: Path = _WHISPER_DIR / "distil-large-v3-encoder.int8.onnx"
WHISPER_DECODER: Path = _WHISPER_DIR / "distil-large-v3-decoder.int8.onnx"
WHISPER_TOKENS: Path = _WHISPER_DIR / "distil-large-v3-tokens.txt"

# ─── Parakeet TDT 0.6B v3 int8 model paths ───────────────────────────────────

_PARAKEET_DIR: Path = BASE_DIR / "models" / "parakeet"
PARAKEET_ENCODER: Path = _PARAKEET_DIR / "encoder.int8.onnx"
PARAKEET_DECODER: Path = _PARAKEET_DIR / "decoder.int8.onnx"
PARAKEET_JOINER: Path = _PARAKEET_DIR / "joiner.int8.onnx"
PARAKEET_TOKENS: Path = _PARAKEET_DIR / "tokens.txt"

# ─── VAD model (Silero — bundled in both variants) ───────────────────────────

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

# ─── Miscellaneous ────────────────────────────────────────────────────────────

MIN_FREE_DISK_BYTES: int = 512 * 1024 * 1024  # 512 MB
MAX_LOG_FILES: int = 10
