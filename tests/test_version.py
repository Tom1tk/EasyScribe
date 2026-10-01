#!/usr/bin/env python3
"""
Checks that every version number matches APP_VERSION in src/config.py:
the launcher, the workflow's default release tag, and the version resource
helper used by both PyInstaller specs. Also checks the icon files exist.

No network, model files or GPU needed. Run from project root:
    python tests/test_version.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "launcher"))

failures: list[str] = []


def check(name: str, ok: bool) -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        failures.append(name)


config_src = (ROOT / "src" / "config.py").read_text(encoding="utf-8")
app_version = re.search(r'^APP_VERSION\s*=\s*"([^"]+)"', config_src, re.M).group(1)  # type: ignore[union-attr]
print(f"APP_VERSION = {app_version}")

launcher_src = (ROOT / "launcher" / "launcher.py").read_text(encoding="utf-8")
launcher_version = re.search(r'^VERSION\s*=\s*"([^"]+)"', launcher_src, re.M).group(1)  # type: ignore[union-attr]
check(f"launcher VERSION ({launcher_version}) matches", launcher_version == app_version)

workflow = (ROOT / ".github" / "workflows" / "build-release.yml").read_text(encoding="utf-8")
default_tag = re.search(r"default:\s*'([^']+)'", workflow).group(1)  # type: ignore[union-attr]
check(f"workflow default tag ({default_tag}) is v{app_version}", default_tag == f"v{app_version}")

helper = (ROOT / "assets" / "version_resource.py").read_text(encoding="utf-8")
check("version_resource.py reads APP_VERSION from config.py", '"src" / "config.py"' in helper)

for spec in ("EasyScribe_whisper.spec", "launcher/launcher.spec"):
    text = (ROOT / spec).read_text(encoding="utf-8")
    check(f"{spec} sets the icon and the version resource", "EasyScribe.ico" in text and "version=_version_info(" in text)

for name in ("EasyScribe.ico", "EasyScribe.png"):
    check(f"assets/{name} exists", (ROOT / "assets" / name).is_file())
check("ICO header is valid", (ROOT / "assets" / "EasyScribe.ico").read_bytes()[:4] == b"\x00\x00\x01\x00")

if failures:
    print(f"\nFAILED: {len(failures)} check(s)", file=sys.stderr)
    sys.exit(1)
print("\nAll version checks passed.")
