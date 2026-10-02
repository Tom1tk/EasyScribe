#!/usr/bin/env python3
"""
Copy the license files into the assembled app folder.

    python assets/collect_licenses.py dist/EasyScribe

Run it with the build venv active (CI step "Assemble app bundle contents"),
so the Python package licenses come from the exact versions in the bundle.

Result in the app folder:
    LICENSE.txt                 EasyScribe (MIT)
    THIRD-PARTY-NOTICES.txt     what is included, where its source is
    licenses/                   the texts from licenses/ in the repo,
                                Python-LICENSE.txt, python-packages/<name>/

tests/validate_build.py --assembled checks the result.
Exits 1 if a required license file is missing.
"""
import importlib.metadata as md
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Python packages that are in the bundle. Each must have a license file.
REQUIRED_PACKAGES = [
    "sherpa-onnx", "numpy", "sounddevice", "cffi", "pycparser",
    "customtkinter", "darkdetect", "tkinterdnd2",
]
# Copied when installed and when they have license files.
OPTIONAL_PACKAGES = ["packaging"]

_LICENSE_WORDS = ("license", "licence", "copying", "notice", "authors")


def _package_license_files(name: str) -> list[tuple[Path, Path]]:
    """(source, path relative to the dist-info folder) for each license file."""
    dist = md.distribution(name)
    out = []
    for f in dist.files or []:
        parts = f.parts
        if not parts or not parts[0].endswith(".dist-info"):
            continue
        if any(w in f.name.lower() for w in _LICENSE_WORDS):
            out.append((Path(dist.locate_file(f)), Path(*parts[1:])))
    return out


def _python_license() -> Path | None:
    base = Path(sys.base_prefix)
    for p in (base / "LICENSE.txt",
              base / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "LICENSE.txt"):
        if p.is_file():
            return p
    return None


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: collect_licenses.py <app_folder>", file=sys.stderr)
        return 1
    dest = Path(sys.argv[1])
    if not dest.is_dir():
        print(f"ERROR: {dest} is not a folder", file=sys.stderr)
        return 1
    errors: list[str] = []
    lic = dest / "licenses"
    lic.mkdir(exist_ok=True)

    shutil.copyfile(ROOT / "LICENSE", dest / "LICENSE.txt")
    shutil.copyfile(ROOT / "licenses" / "THIRD-PARTY-NOTICES.txt",
                    dest / "THIRD-PARTY-NOTICES.txt")
    for f in (ROOT / "licenses").iterdir():
        if f.is_file() and f.name != "THIRD-PARTY-NOTICES.txt":
            shutil.copyfile(f, lic / f.name)

    py = _python_license()
    if py:
        shutil.copyfile(py, lic / "Python-LICENSE.txt")
    else:
        errors.append(f"Python LICENSE.txt not found under {sys.base_prefix}")

    for name in REQUIRED_PACKAGES + OPTIONAL_PACKAGES:
        required = name in REQUIRED_PACKAGES
        try:
            files = _package_license_files(name)
        except md.PackageNotFoundError:
            if required:
                errors.append(f"package {name} is not installed")
            continue
        if not files:
            if required:
                errors.append(f"package {name} has no license file in its dist-info")
            continue
        for src, rel in files:
            target = lic / "python-packages" / name / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, target)
        print(f"  {name}: {len(files)} file(s)")

    count = sum(1 for p in lic.rglob("*") if p.is_file())
    print(f"License files in {lic}: {count}")
    for e in errors:
        print(f"ERROR: {e}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
