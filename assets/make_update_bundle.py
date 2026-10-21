"""
make_update_bundle.py - Build the small update bundle for EasyScribe-update-vX.exe.

Usage: python assets/make_update_bundle.py <assembled app folder> <output zip>

The bundle has every app file except the big ones that rarely change (KEEP:
models, ffmpeg, the whisper.cpp GGUF model). Those are listed with their
SHA-256 in launcher.UPDATE_INFO_FILENAME; the launcher checks the installed
copies before it changes anything and keeps them (see _update_install).
"""

import hashlib
import json
import sys
import zipfile
from fnmatch import fnmatch
from pathlib import Path

UPDATE_INFO_FILENAME = ".easyscribe-update.json"  # same as launcher.py
# Downloaded files, byte-identical between releases while their URLs and
# versions in build-release.yml stay the same. whisper-cli.exe and its DLLs
# are rebuilt when the CI cache expires, so they are always shipped.
KEEP = ("models/*", "ffmpeg/*", "whispercpp/*.bin")
SKIP_TOP = ("recordings",)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(app_dir: Path, out: Path) -> dict[str, str]:
    keep: dict[str, str] = {}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(p for p in app_dir.rglob("*") if p.is_file()):
            rel = path.relative_to(app_dir).as_posix()
            if rel.split("/", 1)[0] in SKIP_TOP:
                continue
            if any(fnmatch(rel, pattern) for pattern in KEEP):
                keep[rel] = _sha256(path)
            else:
                zf.write(path, rel)
        if not keep:
            raise SystemExit("No files to keep: is this the assembled app folder?")
        zf.writestr(UPDATE_INFO_FILENAME, json.dumps({"keep": keep}, indent=1))
    return keep


def main() -> None:
    app_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    if not (app_dir / "EasyScribe.exe").is_file():
        raise SystemExit(f"EasyScribe.exe not found in {app_dir}")
    keep = build(app_dir, out)
    kept_mb = sum((app_dir / rel).stat().st_size for rel in keep) / 1e6
    print(f"Kept from the install: {len(keep)} files, {kept_mb:.0f} MB")
    print(f"{out}: {out.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
