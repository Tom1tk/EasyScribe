#!/usr/bin/env python3
"""
Local tests for the install-directory safety logic in launcher/launcher.py.

These cover the data-loss bug where Browse let the user pick *any* folder
(e.g. Documents) and an "incomplete previous install" cleanup step would
shutil.rmtree it. Run from project root: python tests/test_launcher_safety.py
Exits 0 on success, 1 on failure. No display / Tk mainloop required —
importing launcher.py only defines classes/functions, it doesn't run the GUI.
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "launcher"))

import launcher

failures: list[str] = []


def _check(name: str, fn) -> None:
    try:
        fn()
        print(f"  OK   {name}")
    except Exception as exc:
        print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
        failures.append(name)


def _check_resolve_picked_dir_redirects_to_subdir() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        chosen = Path(tmp)  # an arbitrary existing folder, e.g. Documents
        resolved = launcher._resolve_install_dir(chosen)
        assert resolved == chosen / "EasyScribe", resolved


def _check_resolve_marked_install_stays_in_place() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        chosen = Path(tmp)
        (chosen / launcher.MARKER_FILENAME).write_text("{}", encoding="utf-8")
        resolved = launcher._resolve_install_dir(chosen)
        assert resolved == chosen, resolved


def _check_resolve_premarker_install_stays_in_place() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        chosen = Path(tmp)
        launcher._main_exe(chosen).write_bytes(b"")
        resolved = launcher._resolve_install_dir(chosen)
        assert resolved == chosen, resolved


def _check_safe_to_clean_nonexistent_dir() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        install_dir = Path(tmp) / "EasyScribe"  # does not exist yet
        assert launcher._safe_to_clean(install_dir) is True


def _check_safe_to_clean_marked_incomplete_install() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        install_dir = Path(tmp)
        (install_dir / launcher.MARKER_FILENAME).write_text("{}", encoding="utf-8")
        assert launcher._safe_to_clean(install_dir) is True


def _check_safe_to_clean_complete_install_never_wiped() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        install_dir = Path(tmp)
        (install_dir / launcher.MARKER_FILENAME).write_text("{}", encoding="utf-8")
        launcher._main_exe(install_dir).write_bytes(b"")
        assert launcher._safe_to_clean(install_dir) is False


def _check_safe_to_clean_unmarked_nonempty_folder_never_wiped() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        install_dir = Path(tmp)
        (install_dir / "important_user_file.docx").write_text("don't delete me", encoding="utf-8")
        # No marker, no EasyScribe.exe — this is the Documents-folder scenario.
        assert launcher._safe_to_clean(install_dir) is False


def _check_failure_path_with_no_marker_leaves_dir_untouched() -> None:
    """
    Simulates the except-handler check in _extract_thread: if install_dir
    has no marker, _safe_to_clean is False, so the cleanup rmtree is skipped
    and the directory (and its contents) survive a failed install.
    """
    with tempfile.TemporaryDirectory() as tmp:
        install_dir = Path(tmp)
        existing_file = install_dir / "user_data.txt"
        existing_file.write_text("precious", encoding="utf-8")

        assert launcher._safe_to_clean(install_dir) is False
        # _extract_thread's except handler would skip shutil.rmtree here.
        assert existing_file.is_file()


def _check_write_marker_creates_file_with_app_and_version() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        install_dir = Path(tmp)
        launcher._write_marker(install_dir)

        marker_path = install_dir / launcher.MARKER_FILENAME
        assert marker_path.is_file()
        data = json.loads(marker_path.read_text(encoding="utf-8"))
        assert data["app"] == "EasyScribe", data
        assert data["version"] == launcher.VERSION, data


def _check_write_marker_does_not_overwrite_existing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        install_dir = Path(tmp)
        marker_path = install_dir / launcher.MARKER_FILENAME
        marker_path.write_text('{"app": "EasyScribe", "version": "1.0.0"}', encoding="utf-8")

        launcher._write_marker(install_dir)

        data = json.loads(marker_path.read_text(encoding="utf-8"))
        assert data["version"] == "1.0.0", data  # untouched, not bumped to current VERSION


def main() -> None:
    print("\n-- launcher install-safety tests -------------------------------------------")
    _check("_resolve_install_dir(picked dir) -> <dir>/EasyScribe", _check_resolve_picked_dir_redirects_to_subdir)
    _check("_resolve_install_dir(marked install) -> unchanged", _check_resolve_marked_install_stays_in_place)
    _check("_resolve_install_dir(pre-marker v2.0.0 install) -> unchanged", _check_resolve_premarker_install_stays_in_place)
    _check("_safe_to_clean(nonexistent dir) -> True", _check_safe_to_clean_nonexistent_dir)
    _check("_safe_to_clean(marked, no exe) -> True (cleanable)", _check_safe_to_clean_marked_incomplete_install)
    _check("_safe_to_clean(complete install) -> False (never wiped)", _check_safe_to_clean_complete_install_never_wiped)
    _check("_safe_to_clean(unmarked non-empty folder) -> False", _check_safe_to_clean_unmarked_nonempty_folder_never_wiped)
    _check("failure path with no marker leaves directory untouched", _check_failure_path_with_no_marker_leaves_dir_untouched)
    _check("_write_marker creates marker with app+version", _check_write_marker_creates_file_with_app_and_version)
    _check("_write_marker does not overwrite existing marker", _check_write_marker_does_not_overwrite_existing)
    print("---------------------------------------------------------------------------\n")

    if failures:
        print(f"FAILED: {len(failures)} check(s) failed.", file=sys.stderr)
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
