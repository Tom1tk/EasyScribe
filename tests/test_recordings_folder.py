#!/usr/bin/env python3
"""
Local tests for the Record tab's recordings folder.

Covers:
- settings: the chosen folder is remembered; "Use default", a missing
  folder and a broken settings file all give None (the default).
- MicRecorder.stop(dest_dir=...): the WAV goes to the chosen folder, the raw
  files are removed; if that folder cannot be written the WAV stays next to
  the raw file instead of being lost.
- recovery._handle_orphan(dest_dir=...): a recovered recording goes to the
  chosen folder.

Run from project root: python tests/test_recordings_folder.py
Exits 0 on success, 1 on failure.
"""
import json
import sys
import tempfile
import wave
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import settings  # noqa: E402
from mic_recorder import MicRecorder  # noqa: E402
from recovery import _handle_orphan  # noqa: E402

failures: list[str] = []
_SAMPLES = 1600


def _check(name: str, fn) -> None:
    try:
        fn()
        print(f"  OK   {name}")
    except Exception as exc:
        print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
        failures.append(name)


def _raw_recording(folder: Path) -> tuple[Path, Path]:
    pcm = folder / "recording_20260101_000000.pcm"
    meta = folder / "recording_20260101_000000.json"
    pcm.write_bytes(b"\x01\x00" * _SAMPLES)
    meta.write_text(json.dumps({"sample_rate": 16000, "channels": 1, "bit_depth": 16,
                                "samples_written": _SAMPLES}))
    return pcm, meta


def _stopped_recorder(pcm: Path, meta: Path) -> MicRecorder:
    rec = MicRecorder()
    rec._pcm_path, rec._json_path, rec._samples_written = pcm, meta, _SAMPLES
    return rec


def _wav_frames(path: Path) -> int:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes()


def test_settings() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        settings.SETTINGS_FILE = tmp_path / "work" / "settings.json"
        assert settings.load_recordings_dir() is None, "no file should mean default"
        chosen = tmp_path / "chosen"
        chosen.mkdir()
        settings.save_recordings_dir(chosen)
        assert settings.load_recordings_dir() == chosen
        settings.save_recordings_dir(None)
        assert settings.load_recordings_dir() is None, "Use default was not saved"
        settings.save_recordings_dir(tmp_path / "gone")
        assert settings.load_recordings_dir() is None, "a missing folder must give the default"
        settings.SETTINGS_FILE.write_text("{not json")
        assert settings.load_recordings_dir() is None, "a broken file must give the default"


def test_stop_saves_to_chosen_folder() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work, dest = Path(tmp) / "work", Path(tmp) / "dest" / "new"
        work.mkdir()
        pcm, meta = _raw_recording(work)
        wav = _stopped_recorder(pcm, meta).stop(convert_to_wav=True, dest_dir=dest)
        assert wav == dest / "recording_20260101_000000.wav", wav
        assert _wav_frames(wav) == _SAMPLES
        assert not pcm.exists() and not meta.exists(), "raw files were not removed"


def test_stop_falls_back_when_folder_unusable() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "work"
        work.mkdir()
        blocker = Path(tmp) / "a_file"
        blocker.write_text("not a folder")
        pcm, meta = _raw_recording(work)
        wav = _stopped_recorder(pcm, meta).stop(convert_to_wav=True, dest_dir=blocker / "sub")
        assert wav == work / "recording_20260101_000000.wav", wav
        assert _wav_frames(wav) == _SAMPLES


def test_recovery_to_chosen_folder() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work, dest = Path(tmp) / "work", Path(tmp) / "dest"
        work.mkdir()
        pcm, meta = _raw_recording(work)
        wav = _handle_orphan(pcm, meta, json.loads(meta.read_text()), True, dest)
        assert wav == dest / "recording_20260101_000000.wav", wav
        assert _wav_frames(wav) == _SAMPLES
        assert not pcm.exists() and not meta.exists()


if __name__ == "__main__":
    print("Recordings folder tests:")
    _check("settings remember and fall back", test_settings)
    _check("stop() saves to the chosen folder", test_stop_saves_to_chosen_folder)
    _check("stop() falls back when the folder is unusable", test_stop_falls_back_when_folder_unusable)
    _check("recovery saves to the chosen folder", test_recovery_to_chosen_folder)
    if failures:
        print(f"\n{len(failures)} test(s) failed")
        sys.exit(1)
    print("\nAll recordings folder tests passed")
