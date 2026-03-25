"""
cuda_setup.py - CUDA runtime DLL path registration for Windows.

Adapted from Buzz (https://github.com/chidiwilliams/buzz).

MUST be imported before any torch or CUDA-dependent library (ctranslate2,
pyannote) is imported.  Calling os.add_dll_directory() registers the nvidia
package lib directories in the Windows DLL search path so that calls like
LoadLibrary("cublas64_12.dll") succeed regardless of where the calling code
is located inside the bundle.

Works in two contexts:
  - Frozen (PyInstaller ONEDIR): nvidia DLLs are in
      sys._MEIPASS / "nvidia" / <pkg> / "bin"  (placed there by hook-nvidia.py)
  - Development: nvidia DLLs are in
      site-packages / "nvidia" / <pkg> / "lib" or "bin"
"""

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _nvidia_lib_dirs() -> list[Path]:
    """Return all nvidia package library/bin directories that exist."""
    candidates: list[Path] = []

    if getattr(sys, "frozen", False):
        # Frozen app: hook-nvidia.py copies DLLs to _MEIPASS/nvidia/<pkg>/bin/
        meipass = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        nvidia_root = meipass / "nvidia"
        if nvidia_root.is_dir():
            for pkg_dir in nvidia_root.iterdir():
                if pkg_dir.is_dir():
                    for sub in ("bin", "lib"):
                        d = pkg_dir / sub
                        if d.is_dir():
                            candidates.append(d)
    else:
        # Development: installed nvidia packages live in site-packages/nvidia/
        for sp in sys.path:
            nvidia_root = Path(sp) / "nvidia"
            if nvidia_root.is_dir():
                for pkg_dir in nvidia_root.iterdir():
                    if pkg_dir.is_dir():
                        for sub in ("bin", "lib"):
                            d = pkg_dir / sub
                            if d.is_dir():
                                candidates.append(d)

    return candidates


def setup_cuda_libraries() -> None:
    """
    Register nvidia CUDA library directories in the Windows DLL search path.

    Safe to call multiple times (os.add_dll_directory is idempotent for the
    same path within one process lifetime).
    """
    if sys.platform != "win32":
        return  # Windows only; Linux handled differently, macOS has no CUDA

    dirs = _nvidia_lib_dirs()
    if not dirs:
        logger.debug("cuda_setup: no nvidia package directories found")
        return

    for d in dirs:
        try:
            os.add_dll_directory(str(d))
            logger.debug(f"cuda_setup: registered DLL dir: {d}")
        except (OSError, AttributeError) as exc:
            logger.debug(f"cuda_setup: could not register {d}: {exc}")

    logger.info(f"cuda_setup: registered {len(dirs)} nvidia DLL director(ies)")


# Auto-run on import — this is the intended usage pattern
setup_cuda_libraries()
