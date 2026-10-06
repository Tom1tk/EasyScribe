#!/usr/bin/env python3
"""
Checks where EasyScribe keeps user data on Windows and in the macOS app.

Windows (and a dev checkout): recordings/, logs/, temp/ next to the exe.
macOS app: outside the signed bundle - ~/Documents/EasyScribe,
~/Library/Logs/EasyScribe, ~/Library/Caches/EasyScribe/temp.

No network, model files or GPU needed. Run from project root:
    python tests/test_platform_paths.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402

failures: list[str] = []


def check(name: str, ok: bool) -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        failures.append(name)


base = Path("/opt/EasyScribe")
home = Path("/Users/someone")

rec, logs, temp = config.user_data_dirs(False, base, home)
check("Windows: recordings next to the exe", rec == base / "recordings")
check("Windows: logs next to the exe", logs == base / "logs")
check("Windows: temp next to the exe", temp == base / "temp")

app = Path("/Applications/EasyScribe.app/Contents/MacOS")
rec, logs, temp = config.user_data_dirs(True, app, home)
check("macOS: recordings in ~/Documents/EasyScribe", rec == home / "Documents" / "EasyScribe")
check("macOS: logs in ~/Library/Logs/EasyScribe", logs == home / "Library" / "Logs" / "EasyScribe")
check("macOS: temp in ~/Library/Caches/EasyScribe/temp",
      temp == home / "Library" / "Caches" / "EasyScribe" / "temp")
check("macOS: no user data inside the app bundle",
      not any(".app" in str(p) for p in (rec, logs, temp)))

# In a dev checkout (not frozen) the layout is the same on every platform.
check("dev checkout: not a macOS app", config.IS_MAC_APP is False)
check("dev checkout: RESOURCES_DIR is BASE_DIR", config.RESOURCES_DIR == config.BASE_DIR)
check("dev checkout: ffmpeg name matches the platform",
      config.FFMPEG_BIN.name == ("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"))
check("dev checkout: whisper-cli name matches the platform",
      config.WHISPERCPP_BIN.name == ("whisper-cli.exe" if sys.platform == "win32" else "whisper-cli"))

if failures:
    print(f"\nFAILED: {len(failures)} check(s)", file=sys.stderr)
    sys.exit(1)
print("\nAll platform path checks passed.")
