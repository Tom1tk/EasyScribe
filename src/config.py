"""
config.py - Application configuration and path resolution for EasyScribe.

sherpa-onnx replaces faster-whisper; no HuggingFace dependencies remain.
Single model: Whisper ONNX large-v3-turbo.
"""

import os
import sys
from pathlib import Path

# ─── App metadata ─────────────────────────────────────────────────────────────

APP_NAME = "EasyScribe"
APP_VERSION = "3.2.0"

# ─── Path resolution ──────────────────────────────────────────────────────────


def get_base_dir() -> Path:
    """
    Return the application base directory.

    - Frozen (PyInstaller): folder containing EasyScribe.exe (the EasyScribe/ install dir).
      On macOS: EasyScribe.app/Contents/MacOS (the folder of the executable).
    - Development: project root (parent of src/).
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent.resolve()
    return Path(__file__).parent.parent.resolve()


BASE_DIR: Path = get_base_dir()

# A frozen macOS build is an app bundle (EasyScribe.app). Its files are signed,
# so the app must not write inside it, and programs and data have separate
# folders: executables in Contents/MacOS (BASE_DIR), models in Contents/Resources.
IS_MAC_APP: bool = sys.platform == "darwin" and getattr(sys, "frozen", False)

RESOURCES_DIR: Path = BASE_DIR.parent / "Resources" if IS_MAC_APP else BASE_DIR


def user_data_dirs(mac_app: bool, base: Path, home: Path) -> tuple[Path, Path, Path]:
    """Return (recordings, logs, temp) folders.

    Windows and development: next to the app (the portable install folder).
    macOS app: the standard user folders, outside the signed app bundle.
    """
    if mac_app:
        return (
            home / "Documents" / "EasyScribe",
            home / "Library" / "Logs" / "EasyScribe",
            home / "Library" / "Caches" / "EasyScribe" / "temp",
        )
    return base / "recordings", base / "logs", base / "temp"


_RECORDINGS_DIR, _LOGS_DIR, _TEMP_DIR = user_data_dirs(IS_MAC_APP, BASE_DIR, Path.home())

# ─── Output directory ─────────────────────────────────────────────────────────

DEFAULT_OUTPUT_DIR: Path = _RECORDINGS_DIR

# ─── App icon ─────────────────────────────────────────────────────────────────
# Bundled by the spec into _internal/assets (sys._MEIPASS); assets/ in a dev checkout.

_ASSETS_DIR: Path = Path(getattr(sys, "_MEIPASS", BASE_DIR)) / "assets"
APP_ICON: Path = _ASSETS_DIR / "EasyScribe.ico"
APP_ICON_PNG: Path = _ASSETS_DIR / "EasyScribe.png"

# ─── Whisper ONNX model paths (large-v3-turbo) ────────────────────────────────

_WHISPER_DIR: Path = RESOURCES_DIR / "models" / "whisper"
WHISPER_ENCODER: Path = _WHISPER_DIR / "turbo-encoder.int8.onnx"
WHISPER_DECODER: Path = _WHISPER_DIR / "turbo-decoder.int8.onnx"
WHISPER_TOKENS: Path = _WHISPER_DIR / "turbo-tokens.txt"

# ─── VAD model (Silero) ────────────────────────────────────────────────────────

VAD_MODEL_PATH: Path = RESOURCES_DIR / "models" / "silero_vad.onnx"

# ─── VAD constants ────────────────────────────────────────────────────────────

VAD_SAMPLE_RATE: int = 16000
VAD_CHUNK_SAMPLES: int = 512

# ─── Diarization model paths ──────────────────────────────────────────────────

DIARIZATION_DIR: Path = RESOURCES_DIR / "models" / "diarization"
DIARIZATION_SEGMENTATION_MODEL: Path = DIARIZATION_DIR / "segmentation.onnx"
DIARIZATION_EMBEDDING_MODEL: Path = DIARIZATION_DIR / "embedding.onnx"

# ─── ffmpeg paths ─────────────────────────────────────────────────────────────

_EXE: str = ".exe" if sys.platform == "win32" else ""

FFMPEG_DIR: Path = BASE_DIR / "ffmpeg"
FFMPEG_BIN: Path = FFMPEG_DIR / f"ffmpeg{_EXE}"
FFPROBE_BIN: Path = FFMPEG_DIR / f"ffprobe{_EXE}"

# ─── whisper.cpp (file transcription, optional — beam search, Vulkan/Metal/CPU) ─
# If these aren't bundled (e.g. dev environments before CI bundles them — see
# CLAUDE.md Phase 8 notes), file transcription falls back to the sherpa-onnx
# VAD+greedy path below.

WHISPERCPP_DIR: Path = BASE_DIR / "whispercpp"
WHISPERCPP_BIN: Path = WHISPERCPP_DIR / f"whisper-cli{_EXE}"
WHISPERCPP_MODEL: Path = RESOURCES_DIR / "whispercpp" / "ggml-large-v3-turbo-q5_0.bin"
WHISPERCPP_BEAM_SIZE: int = 5

# ─── Runtime directories ──────────────────────────────────────────────────────

LOGS_DIR: Path = _LOGS_DIR
TEMP_DIR: Path = _TEMP_DIR

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
