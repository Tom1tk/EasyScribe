#!/usr/bin/env python3
"""
Tests for src/whispercpp_wrapper.py's pure helpers (--output-json parsing,
stderr device/progress line parsing) and the cancellation/availability guards
— fast, offline, no whisper-cli binary or GGUF model needed.

Real-binary accuracy A/B, --output-json end-to-end, and mid-run cancellation
were exercised manually against a local whisper.cpp v1.8.6 build + the
ggml-large-v3-turbo-q5_0 model during Phase 8 (see CLAUDE.md Rule 7).

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


# ─── _describe_devices ─────────────────────────────────────────────────────

_INFO = "system_info: n_threads = 8 / 16 | WHISPER : COREML = 0 | CPU : SSE3 = 1 | AVX2 = 1 | AVX512 = 0 |"


def _check_describe_devices_vulkan() -> None:
    lines = [
        "ggml_vulkan: Found 2 Vulkan devices:",
        "ggml_vulkan: 0 = NVIDIA GeForce RTX 5070 (NVIDIA) | uma: 0 | fp16: 1 | matrix cores: NV_coopmat2",
        "ggml_vulkan: 1 = AMD Radeon(TM) Graphics (AMD proprietary driver) | uma: 1 | fp16: 1",
        "whisper_backend_init_gpu: device 0: CPU (type: 0)",
        "whisper_backend_init_gpu: device 1: Vulkan0 (type: 1)",
        "whisper_backend_init_gpu: using Vulkan0 backend",
        _INFO,
    ]
    out = whispercpp_wrapper._describe_devices(lines)
    assert out[0] == "Engine: Vulkan (GPU Vulkan0)", out
    assert out[1] == "GPU: NVIDIA GeForce RTX 5070 (NVIDIA) | uma: 0 | fp16: 1 | matrix cores: NV_coopmat2", out
    assert out[2] == "Other Vulkan device 1: AMD Radeon(TM) Graphics (AMD proprietary driver) (not used)", out
    assert out[3].startswith("CPU: ") and out[3].endswith("| 8 of 16 threads | SSE3 AVX2"), out


def _check_describe_devices_cpu_only() -> None:
    lines = [
        "whisper_backend_init_gpu: device 0: CPU (type: 0)",
        "whisper_backend_init_gpu: no GPU found",
        _INFO,
    ]
    out = whispercpp_wrapper._describe_devices(lines)
    assert out[0] == "Engine: CPU (no GPU found)", out
    assert len(out) == 2 and out[1].startswith("CPU: "), out


def _check_describe_devices_unnamed_gpu() -> None:
    lines = ["whisper_backend_init_gpu: using Vulkan0 backend", _INFO]
    out = whispercpp_wrapper._describe_devices(lines)
    assert out[1] == "GPU: name not reported", out


def _check_pick_gpu_prefers_dedicated() -> None:
    lines = [
        "ggml_vulkan: 0 = AMD Radeon(TM) 610M (AMD proprietary driver) | uma: 1 | fp16: 1 | matrix cores: none",
        "ggml_vulkan: 1 = NVIDIA GeForce RTX 5070 Laptop GPU (NVIDIA) | uma: 0 | fp16: 1 | matrix cores: NV_coopmat2",
    ]
    assert whispercpp_wrapper._pick_gpu(lines) == 1
    assert whispercpp_wrapper._pick_gpu(lines[:1]) == 0  # only an integrated GPU
    assert whispercpp_wrapper._pick_gpu(["whisper_backend_init_gpu: no GPU found"]) is None
    two_dedicated = [
        "ggml_vulkan: 0 = GPU A | uma: 0 | matrix cores: none",
        "ggml_vulkan: 1 = GPU B | uma: 0 | matrix cores: none",
    ]
    assert whispercpp_wrapper._pick_gpu(two_dedicated) == 0  # tie: keep Vulkan's order


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
    _check("describe_devices: Vulkan GPU with real name", _check_describe_devices_vulkan)
    _check("describe_devices: CPU only", _check_describe_devices_cpu_only)
    _check("describe_devices: GPU without a name line", _check_describe_devices_unnamed_gpu)
    _check("pick_gpu: dedicated before integrated", _check_pick_gpu_prefers_dedicated)
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
