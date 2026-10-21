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
APP_VERSION = "3.1.0"

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

# ─── Recordings ───────────────────────────────────────────────────────────────


def _documents_dir() -> Path:
    """The user's Documents folder (on Windows it may be moved, e.g. to OneDrive)."""
    if sys.platform == "win32":
        try:
            import ctypes
            import uuid
            from ctypes import wintypes

            class _GUID(ctypes.Structure):
                _fields_ = [("data", ctypes.c_byte * 16)]

            folder_id = _GUID()
            folder_id.data[:] = uuid.UUID("FDD39AD0-238F-46AF-ADB4-6C85480369C7").bytes_le  # Documents
            path_ptr = wintypes.LPWSTR()
            shell32 = ctypes.windll.shell32
            if shell32.SHGetKnownFolderPath(ctypes.byref(folder_id), 0, None, ctypes.byref(path_ptr)) == 0:
                try:
                    return Path(path_ptr.value)
                finally:
                    ctypes.windll.ole32.CoTaskMemFree(path_ptr)
        except Exception:
            pass
    documents = Path.home() / "Documents"
    return documents if documents.is_dir() else Path.home()


# In-progress microphone recordings (crash-safe PCM) and crash recovery. This
# folder is user data: the launcher keeps it on update (USER_DATA_DIRS).
RECORDING_WORK_DIR: Path = BASE_DIR / "recordings"
# Remembered choices (only the recordings folder for now). Lives in the folder
# above so an update keeps it.
SETTINGS_FILE: Path = RECORDING_WORK_DIR / "settings.json"
# Where finished recordings go until the user picks a folder.
DEFAULT_RECORDINGS_DIR: Path = _documents_dir() / APP_NAME

# ─── App icon ─────────────────────────────────────────────────────────────────
# Bundled by the spec into _internal/assets (sys._MEIPASS); assets/ in a dev checkout.

_ASSETS_DIR: Path = Path(getattr(sys, "_MEIPASS", BASE_DIR)) / "assets"
APP_ICON: Path = _ASSETS_DIR / "EasyScribe.ico"
APP_ICON_PNG: Path = _ASSETS_DIR / "EasyScribe.png"

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

# ─── whisper.cpp (file transcription, optional — beam search, Vulkan/CPU) ──────
# If these aren't bundled (e.g. dev environments before CI bundles them — see
# CLAUDE.md Phase 8 notes), file transcription falls back to the sherpa-onnx
# VAD+greedy path below.

WHISPERCPP_DIR: Path = BASE_DIR / "whispercpp"
WHISPERCPP_BIN: Path = WHISPERCPP_DIR / ("whisper-cli.exe" if sys.platform == "win32" else "whisper-cli")
WHISPERCPP_MODEL: Path = WHISPERCPP_DIR / "ggml-large-v3-turbo-q5_0.bin"
WHISPERCPP_BEAM_SIZE: int = 5

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

# About one thread per physical core (cpu_count counts hyper-threads), 4 to 8.
# Measured: whisper.cpp on CPU and diarization run ~2x faster at 8 threads than 4.
NUM_THREADS: int = min(8, max(4, (os.cpu_count() or 4) // 2))

# ─── Miscellaneous ────────────────────────────────────────────────────────────

MIN_FREE_DISK_BYTES: int = 512 * 1024 * 1024  # 512 MB
MAX_LOG_FILES: int = 10
