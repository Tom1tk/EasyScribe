#!/usr/bin/env python3
"""
Tests for src/whispercpp_wrapper.py's pure helpers (--output-json parsing,
stderr device/progress line parsing) and the cancellation/availability guards
— fast, offline, no whisper-cli binary or GGUF model needed.

Real-binary accuracy A/B, --output-json end-to-end, and mid-run cancellation
were exercised manually against a local whisper.cpp v1.8.6 build + the
ggml-large-v3-turbo-q5_0 model; results recorded in
EASYSCRIBE_ACTION_PLAN.md's Phase 8.1 checkpoint findings.

Run from project root: python tests/test_whispercpp_wrapper.py
Exits 0 on success, 1 on failure.
"""
import json
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import whispercpp_wrapper  # noqa: E402

failures: list[str] = []


def _check(name: str, fn) -> None:
    try:
        fn()
        print(f"  OK   {name}")
    except Exception as exc:
        print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
        failures.append(name)


# ─── _parse_output_json ────────────────────────────────────────────────────


def _check_parse_output_json() -> None:
    data = {
        "transcription": [
            {
                "timestamps": {"from": "00:00:00,000", "to": "00:00:05,420"},
                "offsets": {"from": 0, "to": 5420},
                "text": " Hello, my name is John.",
            },
            {
                "timestamps": {"from": "00:00:06,380", "to": "00:00:11,160"},
                "offsets": {"from": 6380, "to": 11160},
                "text": " Sure, I would be happy.",
            },
            {
                "timestamps": {"from": "00:00:11,500", "to": "00:00:11,500"},
                "offsets": {"from": 11500, "to": 11500},
                "text": "   ",
            },
        ]
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(data, fh)
        path = Path(fh.name)
    try:
        segments = whispercpp_wrapper._parse_output_json(path)
    finally:
        path.unlink()

    assert segments == [
        (0.0, 5.42, "Hello, my name is John."),
        (6.38, 11.16, "Sure, I would be happy."),
    ], segments


def _check_parse_output_json_empty() -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump({"transcription": []}, fh)
        path = Path(fh.name)
    try:
        assert whispercpp_wrapper._parse_output_json(path) == []
    finally:
        path.unlink()


# ─── _parse_device_line ────────────────────────────────────────────────────


def _check_parse_device_line_cpu() -> None:
    line = "whisper_backend_init_gpu: device 0: CPU (type: 0)"
    assert whispercpp_wrapper._parse_device_line(line) == "CPU (type: 0)"


def _check_parse_device_line_gpu() -> None:
    line = "whisper_backend_init_gpu: device 0: NVIDIA GeForce RTX 5070 (Vulkan)"
    assert whispercpp_wrapper._parse_device_line(line) == "NVIDIA GeForce RTX 5070 (Vulkan)"


def _check_parse_device_line_no_gpu() -> None:
    line = "whisper_backend_init_gpu: no GPU found"
    assert whispercpp_wrapper._parse_device_line(line) == "CPU (no GPU found)"


def _check_parse_device_line_unrelated() -> None:
    line = "whisper_model_load: n_vocab       = 51866"
    assert whispercpp_wrapper._parse_device_line(line) is None


# ─── _parse_progress_line ──────────────────────────────────────────────────


def _check_parse_progress_line() -> None:
    assert whispercpp_wrapper._parse_progress_line("whisper_print_progress_callback: progress = 100%") == 100
    assert whispercpp_wrapper._parse_progress_line("whisper_print_progress_callback: progress = 42%") == 42


def _check_parse_progress_line_unrelated() -> None:
    assert whispercpp_wrapper._parse_progress_line("main: processing 'foo.wav'") is None


# ─── availability / validation ─────────────────────────────────────────────


def _check_is_available_false_when_missing() -> None:
    orig_bin, orig_model = whispercpp_wrapper.WHISPERCPP_BIN, whispercpp_wrapper.WHISPERCPP_MODEL
    try:
        whispercpp_wrapper.WHISPERCPP_BIN = Path("/nonexistent/whisper-cli")
        whispercpp_wrapper.WHISPERCPP_MODEL = Path("/nonexistent/model.bin")
        assert whispercpp_wrapper.is_available() is False
        try:
            whispercpp_wrapper.validate_whispercpp()
            raise AssertionError("expected WhisperCppNotFoundError")
        except whispercpp_wrapper.WhisperCppNotFoundError as exc:
            assert "whisper-cli" in str(exc)
            assert "model.bin" in str(exc)
    finally:
        whispercpp_wrapper.WHISPERCPP_BIN, whispercpp_wrapper.WHISPERCPP_MODEL = orig_bin, orig_model


def _check_is_available_true_with_local_build() -> None:
    # Populated locally via whispercpp/ symlinks (gitignored) for Phase 8.1
    # validation; absent in CI before checkpoint (3) bundles whisper-cli.
    if not (ROOT / "whispercpp").is_dir():
        print("    (skip — whispercpp/ not set up locally)")
        return
    assert whispercpp_wrapper.is_available() is True


# ─── cancellation guard ─────────────────────────────────────────────────────


def _check_transcribe_file_pre_cancelled() -> None:
    cancel_event = threading.Event()
    cancel_event.set()
    try:
        whispercpp_wrapper.transcribe_file(Path("/nonexistent.wav"), cancel_event)
        raise AssertionError("expected CancelledError")
    except whispercpp_wrapper.CancelledError:
        pass


def main() -> None:
    print("\n-- whispercpp_wrapper tests --------------------------------------------------")
    _check("parse_output_json: skips blank text, converts ms to sec", _check_parse_output_json)
    _check("parse_output_json: empty transcription list", _check_parse_output_json_empty)
    _check("parse_device_line: CPU device", _check_parse_device_line_cpu)
    _check("parse_device_line: GPU device", _check_parse_device_line_gpu)
    _check("parse_device_line: no GPU found", _check_parse_device_line_no_gpu)
    _check("parse_device_line: unrelated line returns None", _check_parse_device_line_unrelated)
    _check("parse_progress_line: matches percentage", _check_parse_progress_line)
    _check("parse_progress_line: unrelated line returns None", _check_parse_progress_line_unrelated)
    _check("is_available: False when binary/model missing", _check_is_available_false_when_missing)
    _check("is_available: True with local whispercpp/ build", _check_is_available_true_with_local_build)
    _check("transcribe_file: pre-cancelled raises immediately", _check_transcribe_file_pre_cancelled)
    print("---------------------------------------------------------------------------\n")

    if failures:
        print(f"FAILED: {len(failures)} check(s) failed.", file=sys.stderr)
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
