"""
settings.py - Choices EasyScribe remembers between sessions.

Only the recordings folder for now. Stored as JSON in config.SETTINGS_FILE.
A missing or broken file just means "use the defaults".
"""

import json
import logging
from pathlib import Path

from config import SETTINGS_FILE

logger = logging.getLogger(__name__)

_RECORDINGS_DIR_KEY = "recordings_dir"


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
    data = {k: v for k, v in _load().items() if k != _RECORDINGS_DIR_KEY}
    if folder is not None:
        data[_RECORDINGS_DIR_KEY] = str(folder)
    try:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning(f"Could not save settings: {exc}")
