"""
settings.py - Choices EasyScribe remembers between sessions.

The recordings folder, and how fast this PC finds speakers (for the time
estimate). Stored as JSON in config.SETTINGS_FILE.
A missing or broken file just means "use the defaults".
"""

import json
import logging
from pathlib import Path

from config import SETTINGS_FILE

logger = logging.getLogger(__name__)

_RECORDINGS_DIR_KEY = "recordings_dir"
_SPEAKER_SPEED_KEY = "speaker_speed_factor"


def _load() -> dict:
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        logger.warning(f"Could not read settings, using defaults: {exc}")
        return {}
    return data if isinstance(data, dict) else {}


def load_recordings_dir() -> Path | None:
    """The folder the user chose for recordings, or None (use the default).

    A chosen folder that no longer exists (a removed USB drive, say) also
    gives None, so recordings never fail for that reason.
    """
    value = _load().get(_RECORDINGS_DIR_KEY)
    if not isinstance(value, str) or not value:
        return None
    folder = Path(value)
    if not folder.is_dir():
        logger.warning(f"Chosen recordings folder is missing, using the default: {folder}")
        return None
    return folder


def save_recordings_dir(folder: Path | None) -> None:
    """Remember *folder* for recordings; None goes back to the default."""
    _save(_RECORDINGS_DIR_KEY, str(folder) if folder is not None else None)


def load_speaker_speed() -> float | None:
    """How much slower (>1) or faster (<1) than expected this PC found speakers last time."""
    value = _load().get(_SPEAKER_SPEED_KEY)
    return float(value) if isinstance(value, (int, float)) and value > 0 else None


def save_speaker_speed(factor: float) -> None:
    _save(_SPEAKER_SPEED_KEY, round(factor, 3))


def _save(key: str, value: object) -> None:
    """Set *key* to *value* (None removes it) and write the file."""
    data = {k: v for k, v in _load().items() if k != key}
    if value is not None:
        data[key] = value
    try:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning(f"Could not save settings: {exc}")
