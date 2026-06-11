#!/usr/bin/env python3
"""
Local tests for src/recovery.py.

Covers:
- recover_pcm_to_wav: truncates to samples_written, produces a valid WAV.
- _handle_orphan: declining ("No") leaves the .pcm/.json pair untouched;
  accepting ("Yes") converts to .wav and removes the .pcm/.json pair.

Run from project root: python tests/test_recovery.py
Exits 0 on success, 1 on failure.
"""
import json
import sys
import tempfile
import wave
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from recovery import _handle_orphan, recover_pcm_to_wav

failures: list[str] = []


def _check(name: str, fn) -> None:
    try:
        fn()
        print(f"  OK   {name}")
    except Exception as exc:
        print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
        failures.append(name)


def _write_orphan(tmp_path: Path, samples_written: int, extra_samples: int = 0) -> tuple[Path, Path, dict]:
    pcm_path = tmp_path / "recording_20260101_000000.pcm"
    json_path = tmp_path / "recording_20260101_000000.json"

    total_samples = samples_written + extra_samples
    raw = (b"\x01\x02") * total_samples  # arbitrary non-zero int16 samples
    pcm_path.write_bytes(raw)

    metadata = {
        "sample_rate": 16000,
        "channels": 1,
        "bit_depth": 16,
        "samples_written": samples_written,
    }
    json_path.write_text(json.dumps(metadata), encoding="utf-8")
    return pcm_path, json_path, metadata


def _check_recover_pcm_to_wav_truncates() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pcm_path, _json_path, metadata = _write_orphan(tmp_path, samples_written=1000, extra_samples=200)
        wav_path = tmp_path / "recording_20260101_000000.wav"

        recover_pcm_to_wav(pcm_path, metadata, wav_path)

        with wave.open(str(wav_path), "rb") as wf:
            assert wf.getnchannels() == 1, wf.getnchannels()
            assert wf.getsampwidth() == 2, wf.getsampwidth()
            assert wf.getframerate() == 16000, wf.getframerate()
            assert wf.getnframes() == 1000, wf.getnframes()  # truncated, not 1200


def _check_handle_orphan_decline_keeps_files() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pcm_path, json_path, metadata = _write_orphan(tmp_path, samples_written=100)

        result = _handle_orphan(pcm_path, json_path, metadata, recover=False)

        assert result is None, result
        assert pcm_path.exists(), "declining must not delete the .pcm file"
        assert json_path.exists(), "declining must not delete the .json file"


def _check_handle_orphan_accept_recovers_and_cleans_up() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pcm_path, json_path, metadata = _write_orphan(tmp_path, samples_written=100)

        result = _handle_orphan(pcm_path, json_path, metadata, recover=True)

        assert result == pcm_path.with_suffix(".wav"), result
        assert result.exists()
        assert not pcm_path.exists()
        assert not json_path.exists()

        with wave.open(str(result), "rb") as wf:
            assert wf.getnframes() == 100, wf.getnframes()


def main() -> None:
    print("\n-- recovery.py tests -------------------------------------------------------")
    _check("recover_pcm_to_wav truncates to samples_written", _check_recover_pcm_to_wav_truncates)
    _check("_handle_orphan(recover=False) keeps files", _check_handle_orphan_decline_keeps_files)
    _check("_handle_orphan(recover=True) recovers + cleans up", _check_handle_orphan_accept_recovers_and_cleans_up)
    print("---------------------------------------------------------------------------\n")

    if failures:
        print(f"FAILED: {len(failures)} check(s) failed.", file=sys.stderr)
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
