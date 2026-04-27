"""
logger.py - Logging setup for EasyScribe.

Creates a timestamped log file in the logs/ directory next to the executable
and also streams to stdout. Rotates old log files, keeping only the most recent.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

# Import config first to ensure offline env vars and dirs are set
from config import APP_NAME, APP_VERSION, BASE_DIR, LOGS_DIR, MAX_LOG_FILES

import platform


def _rotate_logs() -> None:
    """Remove oldest log files if more than MAX_LOG_FILES exist."""
    log_files = sorted(LOGS_DIR.glob(f"{APP_NAME}_*.log"), key=lambda p: p.stat().st_mtime)
    while len(log_files) >= MAX_LOG_FILES:
        try:
            log_files.pop(0).unlink()
        except OSError:
            break


def setup_logging() -> logging.Logger:
    """
    Configure application-wide logging.

    Returns the root application logger. Call this once at startup in main.py.
    All other modules should use logging.getLogger(__name__).
    """
    _rotate_logs()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"{APP_NAME}_{timestamp}.log"

    log_format = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    # Only add stdout handler if not frozen (avoids PyInstaller windowed-mode errors)
    if not getattr(sys, "frozen", False):
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=logging.DEBUG,
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True,
    )

    logger = logging.getLogger(APP_NAME)
    logger.info("=" * 60)
    logger.info(f"{APP_NAME} v{APP_VERSION} starting")
    logger.info("=" * 60)
    logger.info(f"Log file    : {log_file}")
    logger.info(f"BASE_DIR    : {BASE_DIR}")
    logger.info(f"Python      : {sys.version}")
    logger.info(f"Platform    : {platform.platform()}")
    logger.info(f"Frozen      : {getattr(sys, 'frozen', False)}")
    return logger
