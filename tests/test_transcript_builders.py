#!/usr/bin/env python3
"""
Tests for src/transcriber.py's transcript-formatting helpers
(_build_plain_transcript, _build_diarized_transcript) and
src/diarizer.py's DiarizationEngine.assign_speakers — pure functions over
synthetic segments, no models or audio needed.

Run from project root: python tests/test_transcript_builders.py
Exits 0 on success, 1 on failure.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from diarizer import DiarizationEngine  # noqa: E402
from transcriber import _build_diarized_transcript, _build_plain_transcript, _format_hms  # noqa: E402

failures: list[str] = []


def _check(name: str, fn) -> None:
    try:
        fn()
        print(f"  OK   {name}")
    except Exception as exc:
        print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
        failures.append(name)


# ─── _build_plain_transcript ──────────────────────────────────────────────


def _check_plain_empty() -> None:
    assert _build_plain_transcript([], add_timestamps=False) == ""
    assert _build_plain_transcript([], add_timestamps=True) == ""


def _check_plain_no_timestamps_joins_with_space() -> None:
    segs = [(0.0, 1.0, "Hello"), (1.0, 2.0, "world.")]
    assert _build_plain_transcript(segs, add_timestamps=False) == "Hello world."


def _check_plain_timestamps_single_block() -> None:
    segs = [(0.0, 1.0, "Hello"), (1.5, 2.5, "world.")]
    out = _build_plain_transcript(segs, add_timestamps=True)
    assert out == f"[{_format_hms(0.0)}]\nHello world.", out


def _check_plain_timestamps_splits_on_gap() -> None:
    segs = [(0.0, 1.0, "Hello."), (10.0, 11.0, "Goodbye.")]
    out = _build_plain_transcript(segs, add_timestamps=True)
    expected = f"[{_format_hms(0.0)}]\nHello." + "\n\n" + f"[{_format_hms(10.0)}]\nGoodbye."
    assert out == expected, out


# ─── _build_diarized_transcript ────────────────────────────────────────────


def _check_diarized_empty() -> None:
    assert _build_diarized_transcript([], {}, add_timestamps=False) == ""


def _check_diarized_no_timestamps_groups_consecutive_same_speaker() -> None:
    assigned = [
        ("SPEAKER_00", "Hello,", 0.0, 1.0),
        ("SPEAKER_00", "how are you?", 1.0, 2.0),
        ("SPEAKER_01", "Fine, thanks.", 2.0, 3.0),
    ]
    speaker_map = {"SPEAKER_00": "Speaker 1", "SPEAKER_01": "Speaker 2"}
    out = _build_diarized_transcript(assigned, speaker_map, add_timestamps=False)
    expected = "[Speaker 1]\nHello, how are you?" + "\n\n" + "[Speaker 2]\nFine, thanks."
    assert out == expected, out


def _check_diarized_timestamps_splits_on_speaker_change_and_gap() -> None:
    assigned = [
        ("SPEAKER_00", "Hello.", 0.0, 1.0),
        ("SPEAKER_00", "Still talking.", 1.2, 2.0),
        ("SPEAKER_01", "Reply.", 2.0, 3.0),
        ("SPEAKER_01", "More.", 10.0, 11.0),  # big gap, same speaker -> new block
    ]
    speaker_map = {"SPEAKER_00": "Speaker 1", "SPEAKER_01": "Speaker 2"}
    out = _build_diarized_transcript(assigned, speaker_map, add_timestamps=True)
    expected = "\n\n".join(
        [
            f"[{_format_hms(0.0)}] [Speaker 1]\nHello. Still talking.",
            f"[{_format_hms(2.0)}] [Speaker 2]\nReply.",
            f"[{_format_hms(10.0)}] [Speaker 2]\nMore.",
        ]
    )
    assert out == expected, out


def _check_diarized_unmapped_speaker_falls_back_to_raw_label() -> None:
    assigned = [("SPEAKER_05", "Hi.", 0.0, 1.0)]
    out = _build_diarized_transcript(assigned, {}, add_timestamps=False)
    assert out == "[SPEAKER_05]\nHi.", out


# ─── assign_speakers ────────────────────────────────────────────────────────


def _check_assign_speakers_picks_max_overlap() -> None:
    segments = [(0.0, 2.0, "Hello"), (5.0, 7.0, "world")]
    turns = [(0.0, 1.5, "SPEAKER_00"), (1.5, 4.0, "SPEAKER_01"), (4.5, 8.0, "SPEAKER_02")]
    assigned = DiarizationEngine.assign_speakers(segments, turns)
    assert assigned[0][0] == "SPEAKER_00", assigned[0]  # 1.5s overlap beats 0.5s
    assert assigned[0][1] == "Hello"
    assert assigned[0][2:] == (0.0, 2.0)
    assert assigned[1][0] == "SPEAKER_02", assigned[1]  # full overlap


def _check_assign_speakers_no_overlap_defaults_to_speaker_00() -> None:
    segments = [(100.0, 101.0, "lonely")]
    turns = [(0.0, 1.0, "SPEAKER_03")]
    assigned = DiarizationEngine.assign_speakers(segments, turns)
    assert assigned[0][0] == "SPEAKER_00", assigned[0]


def main() -> None:
    print("\n-- transcript builder tests --------------------------------------------------")
    _check("plain: empty input", _check_plain_empty)
    _check("plain: no timestamps joins with space", _check_plain_no_timestamps_joins_with_space)
    _check("plain: timestamps, single block", _check_plain_timestamps_single_block)
    _check("plain: timestamps, splits on gap", _check_plain_timestamps_splits_on_gap)
    _check("diarized: empty input", _check_diarized_empty)
    _check("diarized: no timestamps groups consecutive same speaker", _check_diarized_no_timestamps_groups_consecutive_same_speaker)
    _check("diarized: timestamps split on speaker change and gap", _check_diarized_timestamps_splits_on_speaker_change_and_gap)
    _check("diarized: unmapped speaker falls back to raw label", _check_diarized_unmapped_speaker_falls_back_to_raw_label)
    _check("assign_speakers: picks max overlap", _check_assign_speakers_picks_max_overlap)
    _check("assign_speakers: no overlap defaults to SPEAKER_00", _check_assign_speakers_no_overlap_defaults_to_speaker_00)
    print("---------------------------------------------------------------------------\n")

    if failures:
        print(f"FAILED: {len(failures)} check(s) failed.", file=sys.stderr)
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
