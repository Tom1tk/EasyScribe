# PyInstaller hook — collect nvidia CUDA package DLLs
#
# When torch is installed with CUDA, pip also installs separate nvidia-*
# packages (nvidia-cublas-cu12, nvidia-cudnn-cu12, etc.).  Each package puts
# its DLLs under  site-packages/nvidia/<pkg>/bin/  (Windows).
#
# This hook copies those DLLs into  _internal/nvidia/<pkg>/bin/  inside the
# bundle, preserving the directory structure so that cuda_setup.py can locate
# them at runtime via os.add_dll_directory().

import os
import sys
from pathlib import Path

binaries: list[tuple[str, str]] = []

# Collect from every site-packages directory on the build machine
for sp in sys.path:
    nvidia_root = Path(sp) / "nvidia"
    if not nvidia_root.is_dir():
        continue
    for pkg_dir in nvidia_root.iterdir():
        if not pkg_dir.is_dir():
            continue
        for sub in ("bin", "lib"):
            dll_dir = pkg_dir / sub
            if not dll_dir.is_dir():
                continue
            for f in dll_dir.iterdir():
                if f.suffix.lower() in (".dll", ".so") and f.is_file():
                    dest = os.path.join("nvidia", pkg_dir.name, sub)
                    binaries.append((str(f), dest))
