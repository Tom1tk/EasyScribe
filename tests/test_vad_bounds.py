#!/usr/bin/env python3
"""
Tests for src/vad.py's pad_and_clamp segment-padding helper and the
VAD_PAD_SEC < VAD_MIN_SILENCE_SEC/2 invariant it relies on — fast, offline,
no models or audio hardware needed.

Run from project root: python tests/test_vad_bounds.py
Exits 0 on success, 1 on failure.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import vad  # noqa: E402

failures: list[str] = []


def _check(name: str, fn) -> None:
    try:
        fn()
        print(f"  OK   {name}")
    except Exception as exc:
        print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
        failures.append(name)


# Real VAD output never produces a gap smaller than VAD_MIN_SILENCE_SEC
# between consecutive segments; that, combined with the invariant below, is
# what guarantees pad_and_clamp's padded segments never overlap.
_MIN_GAP_SAMPLES = int(vad.VAD_MIN_SILENCE_SEC * 16000)


def _assert_well_formed(padded: list[tuple[int, int]], bounds: list[tuple[int, int]], total_samples: int) -> None:
    assert len(padded) == len(bounds), (len(padded), len(bounds))
    for start, end in padded:
        assert 0 <= start <= end <= total_samples, (start, end, total_samples)
    for i in range(len(padded) - 1):
        assert padded[i][1] <= padded[i + 1][0], ("overlap", padded[i], padded[i + 1])
        assert padded[i][0] <= padded[i + 1][0], ("not sorted", padded[i], padded[i + 1])


def _check_invariant_pad_lt_half_min_silence() -> None:
    assert vad.VAD_PAD_SEC < vad.VAD_MIN_SILENCE_SEC / 2, (vad.VAD_PAD_SEC, vad.VAD_MIN_SILENCE_SEC)


def _check_single_segment() -> None:
    bounds = [(1000, 2000)]
    total = 5000
    padded = vad.pad_and_clamp(bounds, total)
    _assert_well_formed(padded, bounds, total)
    assert padded[0] == (max(0, 1000 - vad.PAD_SAMPLES), min(total, 2000 + vad.PAD_SAMPLES)), padded


def _check_adjacent_segments_at_minimal_gap() -> None:
    # Two segments separated by exactly the minimum realistic silence gap.
    bounds = [(0, 10000), (10000 + _MIN_GAP_SAMPLES, 30000)]
    total = 50000
    padded = vad.pad_and_clamp(bounds, total)
    _assert_well_formed(padded, bounds, total)


def _check_far_apart_segments() -> None:
    bounds = [(0, 1000), (100000, 101000), (500000, 501000)]
    total = 1000000
    padded = vad.pad_and_clamp(bounds, total)
    _assert_well_formed(padded, bounds, total)


def _check_segment_at_start_clamped_to_zero() -> None:
    bounds = [(0, 100)]
    total = 100000
    padded = vad.pad_and_clamp(bounds, total)
    _assert_well_formed(padded, bounds, total)
    assert padded[0][0] == 0, padded


def _check_segment_at_end_clamped_to_total() -> None:
    total = 100000
    bounds = [(total - 100, total)]
    padded = vad.pad_and_clamp(bounds, total)
    _assert_well_formed(padded, bounds, total)
    assert padded[0][1] == total, padded


def _check_many_adjacent_segments() -> None:
    # Several segments back-to-back, each separated by the minimum gap.
    bounds = []
    pos = 0
    for _ in range(10):
        bounds.append((pos, pos + 5000))
        pos += 5000 + _MIN_GAP_SAMPLES
    total = pos + 1000
    padded = vad.pad_and_clamp(bounds, total)
    _assert_well_formed(padded, bounds, total)


def _check_empty_bounds() -> None:
    assert vad.pad_and_clamp([], 1000) == []


def main() -> None:
    print("\n-- vad.py bounds tests -------------------------------------------------------")
    _check("VAD_PAD_SEC < VAD_MIN_SILENCE_SEC / 2 invariant", _check_invariant_pad_lt_half_min_silence)
    _check("pad_and_clamp: single segment", _check_single_segment)
    _check("pad_and_clamp: adjacent segments at minimal gap don't overlap", _check_adjacent_segments_at_minimal_gap)
    _check("pad_and_clamp: far-apart segments", _check_far_apart_segments)
    _check("pad_and_clamp: segment at start clamped to 0", _check_segment_at_start_clamped_to_zero)
    _check("pad_and_clamp: segment at end clamped to total_samples", _check_segment_at_end_clamped_to_total)
    _check("pad_and_clamp: many adjacent segments", _check_many_adjacent_segments)
    _check("pad_and_clamp: empty bounds", _check_empty_bounds)
    print("---------------------------------------------------------------------------\n")

    if failures:
        print(f"FAILED: {len(failures)} check(s) failed.", file=sys.stderr)
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
