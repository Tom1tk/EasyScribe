#!/usr/bin/env python3
"""
Post-build artifact validator. Run immediately after PyInstaller and before
creating app.bundle so broken bundles are caught before the 30-minute compress step.

Usage: python tests/validate_build.py <dist_dir>
       python tests/validate_build.py dist/EasyScribe

Exits 0 on success, 1 if any check fails.
"""
import sys
from pathlib import Path


def _find_pyds(directory: Path, pattern: str = "**/*.pyd") -> list[Path]:
    return list(directory.glob(pattern))


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: validate_build.py <dist_dir>", file=sys.stderr)
        sys.exit(1)

    dist = Path(sys.argv[1])
    internal = dist / "_internal"

    errors: list[str] = []
    ok: list[str] = []

    # ── EasyScribe.exe ────────────────────────────────────────────────────────
    exe = dist / "EasyScribe.exe"
    if exe.is_file():
        ok.append(f"EasyScribe.exe  ({exe.stat().st_size // 1024} KB)")
    else:
        errors.append(f"MISSING: {exe}")

    # ── _internal must exist ──────────────────────────────────────────────────
    if not internal.is_dir():
        errors.append(f"MISSING: _internal/ directory under {dist}")
        # Can't check anything else without _internal
        _report(ok, errors)
        return

    # ── Per-package checks ────────────────────────────────────────────────────
    # Packages with their own subdirectory in _internal:
    dir_checks = [
        ("numpy",        "numpy",        1),   # numpy/*.pyd
        ("sherpa_onnx",  "sherpa_onnx",  1),   # sherpa_onnx/*.pyd
        ("customtkinter","customtkinter", 0),   # directory only
        ("tkinterdnd2",  "tkinterdnd2",  0),   # directory only
    ]

    for name, subdir, min_pyds in dir_checks:
        pkg_dir = internal / subdir
        if not pkg_dir.is_dir():
            errors.append(f"MISSING package dir: {name}  (expected {pkg_dir})")
            continue
        if min_pyds > 0:
            pyds = _find_pyds(pkg_dir)
            if len(pyds) < min_pyds:
                errors.append(
                    f"MISSING binaries: {name}  (found {len(pyds)} .pyd files, need >= {min_pyds})"
                )
            else:
                ok.append(f"{name}  ({len(pyds)} .pyd file(s))")
        else:
            ok.append(f"{name}  (directory present)")

    # sounddevice: C extension may be at _internal root (flat) or _internal/sounddevice/ (pkg)
    sd_pyds = list(internal.glob("**/_sounddevice*.pyd"))
    if not sd_pyds:
        errors.append(
            f"MISSING binaries: sounddevice  (no _sounddevice*.pyd anywhere under {internal})\n"
            f"    Required for microphone recording."
        )
    else:
        ok.append(f"sounddevice  ({sd_pyds[0].name} at {sd_pyds[0].parent.name}/)")

    _report(ok, errors)


def _report(ok: list[str], errors: list[str]) -> None:
    print("\n-- Build validation -----------------------------------------------------")
    for line in ok:
        print(f"  OK   {line}")
    for line in errors:
        print(f"  FAIL {line}")
    print("-------------------------------------------------------------------------\n")

    if errors:
        print(f"BUILD VALIDATION FAILED — {len(errors)} issue(s) above.", file=sys.stderr)
        sys.exit(1)

    print(f"All {len(ok)} checks passed.")


if __name__ == "__main__":
    main()
