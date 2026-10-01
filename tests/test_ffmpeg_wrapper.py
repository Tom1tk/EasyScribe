#!/usr/bin/env python3
"""
Tests for src/ffmpeg_wrapper.py's pure helpers: stderr line splitting and
duration/progress parsing — fast, offline, no ffmpeg binary needed.

Run from project root: python tests/test_ffmpeg_wrapper.py
Exits 0 on success, 1 on failure.
"""
import io
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import ffmpeg_wrapper  # noqa: E402

failures: list[str] = []


def _check(name: str, fn) -> None:
    try:
        fn()
        print(f"  OK   {name}")
    except Exception as exc:
        print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
        failures.append(name)


class _ChunkedStream:
    """A byte stream that returns its data in fixed-size pieces, like a pipe."""

    def __init__(self, data: bytes, size: int) -> None:
        self._buf = io.BytesIO(data)
        self._size = size

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(self._size)


# ffmpeg -stats ends each progress update with \r, not \n.
_STDERR = (
    b"  Duration: 00:01:40.00, start: 0.000000, bitrate: 128 kb/s\n"
    b"size=  100kB time=00:00:10.00 bitrate=  1kbits/s\r"
    b"size=  200kB time=00:00:50.00 bitrate=  1kbits/s\r"
    b"size=  400kB time=00:01:40.00 bitrate=  1kbits/s\r\n"
    b"video:0kB audio:400kB"
)


def _check_splits_on_cr() -> None:
    lines = list(ffmpeg_wrapper._iter_lines(io.BytesIO(_STDERR)))
    times = [ffmpeg_wrapper._parse_time(l) for l in lines]
    assert [t for t in times if t >= 0] == [10.0, 50.0, 100.0], lines
    assert lines[-1] == "video:0kB audio:400kB", lines


def _check_small_chunks() -> None:
    # A line split across two pipe reads must still come out whole.
    whole = list(ffmpeg_wrapper._iter_lines(io.BytesIO(_STDERR)))
    for size in (1, 3, 7, 64):
        got = [l for l in ffmpeg_wrapper._iter_lines(_ChunkedStream(_STDERR, size)) if l]
        assert got == [l for l in whole if l], (size, got)


def _check_empty_stream() -> None:
    assert list(ffmpeg_wrapper._iter_lines(io.BytesIO(b""))) == []


def _check_parse_duration() -> None:
    assert ffmpeg_wrapper._parse_duration("  Duration: 01:02:03.50, start") == 3723.5
    assert ffmpeg_wrapper._parse_duration("no duration here") == 0.0


def main() -> None:
    print("\n-- ffmpeg_wrapper tests -----------------------------------------------------")
    _check("iter_lines: splits progress updates on \\r", _check_splits_on_cr)
    _check("iter_lines: lines split across pipe reads", _check_small_chunks)
    _check("iter_lines: empty stream", _check_empty_stream)
    _check("parse_duration: header line and no match", _check_parse_duration)
    print("---------------------------------------------------------------------------\n")

    if failures:
        print(f"FAILED: {len(failures)} check(s) failed.", file=sys.stderr)
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
