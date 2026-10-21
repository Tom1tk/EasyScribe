#!/usr/bin/env python3
"""
Local tests for src/rename.py (the Name box after a result).

Covers: name checks, which files belong to a recording, renaming them
together, a taken name (nothing changes), and undo when one rename fails.

Run from project root: python tests/test_rename.py
Exits 0 on success, 1 on failure.
"""
import sys
import tempfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rename import check_name, recording_set, rename_all

failures: list[str] = []


def _check(name: str, fn) -> None:
    try:
        fn()
        print(f"  OK   {name}")
    except Exception as exc:
        failures.append(name)
        print(f"  FAIL {name}: {exc!r}")


STEM = "recording_20261021_143207"
SET = [f"{STEM}.wav", f"{STEM}.txt", f"{STEM} (best quality).txt",
       f"{STEM} (best quality) (speakers).txt"]
OTHERS = [f"{STEM}_2.wav", "recording_20261021_150000.txt", f"{STEM}.json"]


def _folder() -> Path:
    d = Path(tempfile.mkdtemp())
    for n in SET + OTHERS:
        (d / n).write_text(n)
    return d


def test_check_name() -> None:
    assert check_name("Team meeting 21 Oct") == ""
    for bad in ("", "a/b", "a:b", 'say "hi"', "what?", "CON", "lpt1", "name.", "x" * 200):
        assert check_name(bad), bad


def test_recording_set() -> None:
    d = _folder()
    assert [p.name for p in recording_set(d / f"{STEM}.wav")] == sorted(SET)


def test_rename_together() -> None:
    d = _folder()
    moved = rename_all(recording_set(d / f"{STEM}.wav"), STEM, "Team meeting")
    names = sorted(p.name for p in d.iterdir())
    assert "Team meeting (best quality) (speakers).txt" in names, names
    assert "Team meeting.wav" in names and not any(n in names for n in SET), names
    assert all(n in names for n in OTHERS), names
    assert moved[d / f"{STEM}.txt"] == d / "Team meeting.txt"
    assert (d / "Team meeting.txt").read_text() == f"{STEM}.txt"


def test_taken_name_changes_nothing() -> None:
    d = _folder()
    (d / "Team meeting.txt").write_text("other")
    try:
        rename_all(recording_set(d / f"{STEM}.wav"), STEM, "Team meeting")
        raise AssertionError("no FileExistsError")
    except FileExistsError:
        pass
    assert all((d / n).exists() for n in SET)
    assert (d / "Team meeting.txt").read_text() == "other"


def test_undo_on_failure() -> None:
    d = _folder()
    files = recording_set(d / f"{STEM}.wav")
    real = Path.rename
    calls = []

    def flaky(self, target):  # type: ignore[no-untyped-def]
        calls.append(self)
        if len(calls) == 3:
            raise PermissionError("in use")
        return real(self, target)

    with mock.patch.object(Path, "rename", flaky):
        try:
            rename_all(files, STEM, "Team meeting")
            raise AssertionError("no OSError")
        except PermissionError:
            pass
    assert sorted(p.name for p in d.iterdir()) == sorted(SET + OTHERS)


def test_transcript_only() -> None:
    d = _folder()
    t = d / f"{STEM}.txt"
    rename_all([t], STEM, "Notes")
    assert (d / "Notes.txt").exists() and (d / f"{STEM}.wav").exists()


print("test_rename.py")
for _name, _fn in list(globals().items()):
    if _name.startswith("test_") and callable(_fn):
        _check(_name, _fn)
if failures:
    print(f"{len(failures)} failure(s)")
    sys.exit(1)
print("All rename tests passed.")
