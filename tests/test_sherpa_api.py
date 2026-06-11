#!/usr/bin/env python3
"""
Local API-shape tests for the sherpa_onnx integration in src/transcriber.py.

These run fast, offline, and without real model files — they catch mismatches
between our code and the installed sherpa_onnx version's constructor signatures
(e.g. wrong kwargs, wrong class) before a slow CI build.

Run from project root: python tests/test_sherpa_api.py
Exits 0 on success, 1 on failure.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

failures: list[str] = []


def _check(name: str, fn) -> None:
    try:
        fn()
        print(f"  OK   {name}")
    except Exception as exc:
        print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
        failures.append(name)


def _check_recognizer_config() -> None:
    import transcriber
    from sherpa_onnx.lib._sherpa_onnx import OfflineRecognizer as _OfflineRecognizer

    cfg = transcriber._build_recognizer_config("cpu")

    import sherpa_onnx
    assert isinstance(cfg, sherpa_onnx.OfflineRecognizerConfig), type(cfg)

    # If model files aren't present (e.g. CI before the model download step),
    # constructing the recognizer must fail with RuntimeError (bad path), NOT
    # TypeError (bad constructor signature).
    try:
        _OfflineRecognizer(cfg)
    except RuntimeError:
        pass  # expected when model files are missing


def main() -> None:
    print("\n-- sherpa_onnx API tests --------------------------------------------------")
    _check("OfflineRecognizerConfig", _check_recognizer_config)
    print("---------------------------------------------------------------------------\n")

    if failures:
        print(f"FAILED: {len(failures)} check(s) failed.", file=sys.stderr)
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
