"""
rename.py - give a recording or a transcript a name of the user's choice.

A recording's files share its stem: "<stem>.wav", "<stem>.txt",
"<stem> (best quality).txt", "<stem> (speakers).txt" ... They are renamed
together, so they stay a set and "Find speakers" names its copy after the
new name. Either every file is renamed, or none is.
"""

import re
from pathlib import Path

_BAD_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_RESERVED = re.compile(r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])$", re.IGNORECASE)
MAX_NAME = 120


def check_name(name: str) -> str:
    """'' if *name* can be a Windows file name, else what is wrong with it."""
    if not name:
        return "Type a name first."
    if _BAD_CHARS.search(name):
        return 'A name cannot have these characters: \\ / : * ? " < > |'
    if _RESERVED.match(name) or name.endswith("."):
        return "Windows does not allow this name. Please choose another."
    if len(name) > MAX_NAME:
        return f"Please use a shorter name (at most {MAX_NAME} characters)."
    return ""


def recording_set(audio: Path) -> list[Path]:
    """The recording and the transcripts made from it, in its folder."""
    stem = audio.stem
    return sorted(
        p for p in audio.parent.iterdir()
        if p.is_file() and p.suffix.lower() in (".wav", ".txt")
        and (p.stem == stem or p.stem.startswith(stem + " ("))
    )


def rename_all(paths: list[Path], old_stem: str, new_stem: str) -> dict[Path, Path]:
    """Rename each path's leading *old_stem* to *new_stem*. Returns {old: new}.

    Raises FileExistsError (nothing changed) if a new name is taken, and
    OSError after undoing the renames already done if one fails.
    """
    plan = {p: p.with_name(new_stem + p.name[len(old_stem):]) for p in paths}
    taken = [new for old, new in plan.items() if new.exists() and new != old]
    if taken:
        raise FileExistsError(taken[0].name)
    done: dict[Path, Path] = {}
    try:
        for old, new in plan.items():
            if new != old:
                old.rename(new)
            done[old] = new
    except OSError:
        for old, new in done.items():
            if new != old:
                new.rename(old)
        raise
    return done
