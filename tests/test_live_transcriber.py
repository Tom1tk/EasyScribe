#!/usr/bin/env python3
"""
Tests for src/vad.py's live-mode padding helper and src/live_transcriber.py's
bounded decode queue — fast, offline, no models or audio hardware needed.

Run from project root: python tests/test_live_transcriber.py
Exits 0 on success, 1 on failure.
"""
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import vad  # noqa: E402
import live_transcriber  # noqa: E402

failures: list[str] = []


def _check(name: str, fn) -> None:
    try:
        fn()
        print(f"  OK   {name}")
    except Exception as exc:
        print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
        failures.append(name)


def _check_pad_live_segment_no_padding_when_history_too_recent() -> None:
    # history only covers absolute samples [100, 200) — entirely after seg_start,
    # so no preceding audio is available and seg_samples must pass through unchanged.
    history = np.arange(100, dtype=np.float32)
    history_start = 100
    seg_samples = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    out = vad.pad_live_segment(history, history_start, 0, 50, seg_samples)
    assert np.array_equal(out, seg_samples), out


def _check_pad_live_segment_applies_padding() -> None:
    history = np.arange(4000, dtype=np.float32)
    history_start = 0
    seg_start = vad.PAD_SAMPLES + 50
    seg_samples = np.array([999.0], dtype=np.float32)

    out = vad.pad_live_segment(history, history_start, 0, seg_start, seg_samples)
    expected_lead_in = history[50:seg_start]  # length == PAD_SAMPLES
    assert len(out) == len(expected_lead_in) + 1, len(out)
    assert np.array_equal(out[:-1], expected_lead_in)
    assert out[-1] == 999.0


def _check_pad_live_segment_clamped_by_prev_segment_end() -> None:
    history = np.arange(4000, dtype=np.float32)
    history_start = 0
    prev_segment_end = vad.PAD_SAMPLES + 50  # closer to seg_start than PAD_SAMPLES allows
    seg_start = vad.PAD_SAMPLES + 200
    seg_samples = np.array([999.0], dtype=np.float32)

    out = vad.pad_live_segment(history, history_start, prev_segment_end, seg_start, seg_samples)
    expected_lead_in = history[prev_segment_end:seg_start]
    assert len(out) == len(expected_lead_in) + 1, len(out)
    assert np.array_equal(out[:-1], expected_lead_in)


def _check_pad_live_segment_clamped_by_history_start() -> None:
    # history covers absolute samples [3100, 3300); seg_start - PAD_SAMPLES == 50,
    # but history doesn't go back that far, so padding is clamped to history_start.
    history = np.arange(200, dtype=np.float32)
    history_start = 3100
    seg_start = 3250
    seg_samples = np.array([999.0], dtype=np.float32)

    out = vad.pad_live_segment(history, history_start, 0, seg_start, seg_samples)
    expected_lead_in = history[0:150]  # history[history_start - history_start : seg_start - history_start]
    assert len(out) == len(expected_lead_in) + 1, len(out)
    assert np.array_equal(out[:-1], expected_lead_in)


class _FakeResult:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeStream:
    def __init__(self) -> None:
        self.result = _FakeResult("x")

    def accept_waveform(self, sr: int, samples: np.ndarray) -> None:
        pass


class _SlowFakeRecognizer:
    """Fake recognizer whose decode_stream takes longer than the producer's pace."""

    def create_stream(self) -> _FakeStream:
        return _FakeStream()

    def decode_stream(self, stream: _FakeStream) -> None:
        time.sleep(0.05)


def _check_decode_queue_drops_oldest_under_backpressure() -> None:
    decode_queue: queue.Queue = queue.Queue(maxsize=live_transcriber._DECODE_QUEUE_MAXSIZE)
    cancel_event = threading.Event()
    onnx_lock = threading.Lock()
    received: list[str] = []

    decode_thread = threading.Thread(
        target=live_transcriber.LiveTranscriber._decode_loop,
        args=(_SlowFakeRecognizer(), decode_queue, cancel_event, received.append, onnx_lock),
        daemon=True,
    )
    decode_thread.start()

    dropped = False
    for _ in range(50):
        if not live_transcriber._put_dropping_oldest(decode_queue, np.zeros(4, dtype=np.float32)):
            dropped = True
        assert decode_queue.qsize() <= live_transcriber._DECODE_QUEUE_MAXSIZE, decode_queue.qsize()

    cancel_event.set()
    decode_thread.join(timeout=5)

    assert dropped, "expected the decode queue to drop oldest under sustained backpressure"


def main() -> None:
    print("\n-- live_transcriber tests --------------------------------------------------")
    _check("pad_live_segment: no padding when history too recent", _check_pad_live_segment_no_padding_when_history_too_recent)
    _check("pad_live_segment: applies padding", _check_pad_live_segment_applies_padding)
    _check("pad_live_segment: clamped by prev_segment_end", _check_pad_live_segment_clamped_by_prev_segment_end)
    _check("pad_live_segment: clamped by history_start", _check_pad_live_segment_clamped_by_history_start)
    _check("decode queue: drops oldest under backpressure", _check_decode_queue_drops_oldest_under_backpressure)
    print("---------------------------------------------------------------------------\n")

    if failures:
        print(f"FAILED: {len(failures)} check(s) failed.", file=sys.stderr)
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
