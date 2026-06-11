#!/usr/bin/env python3
"""
Runs every fast, offline tests/test_*.py file and aggregates the results.

Excludes test_diarization.py: it's an end-to-end test that needs real model
files (models/whisper, models/diarization) and tests/data/test_speech.wav,
which aren't available before CI's model-download steps. Run it
manually/locally instead.

Run from project root: python tests/run_all.py
Exits 0 if every test file exits 0, 1 otherwise.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

_EXCLUDE = {"test_diarization.py"}


def main() -> None:
    test_files = sorted(p for p in ROOT.glob("test_*.py") if p.name not in _EXCLUDE)

    failed: list[str] = []
    for test_file in test_files:
        print(f"\n=== {test_file.name} ===")
        result = subprocess.run([sys.executable, str(test_file)], cwd=ROOT.parent)
        if result.returncode != 0:
            failed.append(test_file.name)

    print("\n=============================================================================")
    if failed:
        print(f"FAILED: {len(failed)}/{len(test_files)} test file(s) failed: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)
    print(f"All {len(test_files)} test files passed.")


if __name__ == "__main__":
    main()
