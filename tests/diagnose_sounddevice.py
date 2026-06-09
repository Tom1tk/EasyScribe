#!/usr/bin/env python3
"""
Prints the sounddevice package layout so CI build logs show exactly
what files sounddevice ships on this platform/version.
"""
import importlib.util
from pathlib import Path

sd_spec = importlib.util.find_spec("sounddevice")
if not sd_spec or not sd_spec.origin:
    print("ERROR: sounddevice not importable")
    raise SystemExit(1)

sd_origin = Path(sd_spec.origin)
sd_dir = sd_origin.parent

import sounddevice
print(f"sounddevice version : {sounddevice.__version__}")
print(f"sounddevice __file__: {sd_origin}")
print(f"install directory   : {sd_dir}")
print("contents:")
for f in sorted(sd_dir.iterdir()):
    print(f"  {f.name}  ({f.stat().st_size} B)")
