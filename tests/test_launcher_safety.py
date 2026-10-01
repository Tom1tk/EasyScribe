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
import zipfile
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


def _check_shortcuts_created_false_without_marker() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        install_dir = Path(tmp)
        assert launcher._shortcuts_created(install_dir) is False


def _check_shortcuts_created_false_before_marked() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        install_dir = Path(tmp)
        launcher._write_marker(install_dir)
        assert launcher._shortcuts_created(install_dir) is False


def _check_mark_shortcuts_created_sets_flag_and_keeps_other_fields() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        install_dir = Path(tmp)
        launcher._write_marker(install_dir)

        launcher._mark_shortcuts_created(install_dir)

        assert launcher._shortcuts_created(install_dir) is True
        data = json.loads((install_dir / launcher.MARKER_FILENAME).read_text(encoding="utf-8"))
        assert data["app"] == "EasyScribe", data
        assert data["version"] == launcher.VERSION, data


def _check_ensure_shortcuts_creates_once() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        install_dir = Path(tmp)
        launcher._write_marker(install_dir)

        calls: list[Path] = []
        original = launcher._create_shortcuts
        launcher._create_shortcuts = calls.append
        try:
            launcher._ensure_shortcuts(install_dir)
            launcher._ensure_shortcuts(install_dir)  # second call must be a no-op
        finally:
            launcher._create_shortcuts = original

        assert calls == [install_dir], calls
        assert launcher._shortcuts_created(install_dir) is True


# ── Version check and in-place update (B1) ────────────────────────────────────


def _check_version_order() -> None:
    k = launcher._version_key
    order = ["junk", "2.0.0", "v2.1", "3.0.0-alpha1", "3.0.0-beta1", "3.0.0-beta2",
             "3.0.0-rc1", "3.0.0", "3.0.1", "3.1.0", "10.0.0"]
    keys = [k(v) for v in order]
    assert keys == sorted(keys), list(zip(order, keys))
    assert k("3.0") == k("3.0.0")


def _make_install(root: Path, version: str | None) -> Path:
    """An older install: old app files, a recording, a log and a marker."""
    install_dir = root / "EasyScribe"
    (install_dir / "_internal").mkdir(parents=True)
    (install_dir / "_internal" / "old.dll").write_text("old", encoding="utf-8")
    (install_dir / "models").mkdir()
    (install_dir / "models" / "old_model.onnx").write_text("old", encoding="utf-8")
    launcher._main_exe(install_dir).write_text("old exe", encoding="utf-8")
    (install_dir / "recordings").mkdir()
    (install_dir / "recordings" / "meeting.wav").write_text("precious audio", encoding="utf-8")
    (install_dir / "recordings" / "meeting.txt").write_text("precious text", encoding="utf-8")
    (install_dir / "logs").mkdir()
    (install_dir / "logs" / "easyscribe.log").write_text("old log", encoding="utf-8")
    if version is not None:
        data = {"app": "EasyScribe", "version": version, "shortcuts_created": True}
        (install_dir / launcher.MARKER_FILENAME).write_text(json.dumps(data), encoding="utf-8")
    return install_dir


def _make_bundle(root: Path) -> Path:
    bundle = root / "app.bundle"
    with zipfile.ZipFile(bundle, "w") as zf:
        zf.writestr("EasyScribe.exe", "new exe")
        zf.writestr("_internal/new.dll", "new")
        zf.writestr("models/whisper/new_model.onnx", "new")
        zf.writestr("whispercpp/whisper-cli.exe", "new")
        zf.writestr("recordings/", "")
    return bundle


def _check_needs_update() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        older = _make_install(Path(tmp) / "a", "2.1.0")
        assert launcher._needs_update(older) is True
        premarker = _make_install(Path(tmp) / "b", None)
        assert launcher._needs_update(premarker) is True
        same = _make_install(Path(tmp) / "c", launcher.VERSION)
        assert launcher._needs_update(same) is False
        newer = _make_install(Path(tmp) / "d", "99.0.0")
        assert launcher._needs_update(newer) is False  # never downgrade


def _check_update_replaces_app_files_and_keeps_user_data() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        install_dir = _make_install(root, "2.1.0")
        progress: list[int] = []

        launcher._update_install(_make_bundle(root), install_dir, progress.append)

        assert launcher._main_exe(install_dir).read_text(encoding="utf-8") == "new exe"
        assert (install_dir / "_internal" / "new.dll").is_file()
        assert not (install_dir / "_internal" / "old.dll").exists()
        assert not (install_dir / "models" / "old_model.onnx").exists()
        assert (install_dir / "whispercpp" / "whisper-cli.exe").is_file()
        assert (install_dir / "recordings" / "meeting.wav").read_text(encoding="utf-8") == "precious audio"
        assert (install_dir / "recordings" / "meeting.txt").read_text(encoding="utf-8") == "precious text"
        assert (install_dir / "logs" / "easyscribe.log").read_text(encoding="utf-8") == "old log"
        assert not (install_dir / launcher.BACKUP_DIRNAME).exists()
        data = json.loads((install_dir / launcher.MARKER_FILENAME).read_text(encoding="utf-8"))
        assert data["version"] == launcher.VERSION, data
        assert data["shortcuts_created"] is True, data
        assert "updating" not in data, data
        assert progress and progress[-1] == 100, progress


def _check_update_premarker_install() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        install_dir = _make_install(root, None)
        launcher._update_install(_make_bundle(root), install_dir)
        assert launcher._installed_version(install_dir) == launcher.VERSION
        assert (install_dir / "recordings" / "meeting.wav").is_file()


def _check_update_blocked_when_app_open_restores_old_files() -> None:
    """Windows refuses to move a running app's files; simulate that on _internal."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        install_dir = _make_install(root, "2.1.0")
        original = launcher._move

        def locked_move(src, dst):
            if Path(src).name == "_internal" and Path(src).parent == install_dir:
                raise PermissionError("[WinError 5] Access is denied")
            return original(src, dst)

        launcher._move = locked_move
        try:
            launcher._update_install(_make_bundle(root), install_dir)
            raise AssertionError("expected UpdateBlockedError")
        except launcher.UpdateBlockedError:
            pass
        finally:
            launcher._move = original

        assert launcher._main_exe(install_dir).read_text(encoding="utf-8") == "old exe"
        assert (install_dir / "_internal" / "old.dll").is_file()
        assert (install_dir / "models" / "old_model.onnx").is_file()
        assert (install_dir / "recordings" / "meeting.wav").is_file()
        assert not (install_dir / launcher.BACKUP_DIRNAME).exists()
        assert launcher._installed_version(install_dir) == "2.1.0"
        assert launcher._update_in_progress(install_dir) is False


def _check_update_failure_during_extract_restores_old_files() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        install_dir = _make_install(root, "2.1.0")
        bundle = root / "app.bundle"
        bundle.write_bytes(b"not a zip")
        try:
            launcher._update_install(bundle, install_dir)
            raise AssertionError("expected an error for a bad bundle")
        except zipfile.BadZipFile:
            pass
        assert launcher._main_exe(install_dir).read_text(encoding="utf-8") == "old exe"
        assert (install_dir / "recordings" / "meeting.wav").is_file()


def _check_recover_interrupted_update() -> None:
    """Power loss after the old files were moved aside: put them back."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        install_dir = _make_install(root, "2.1.0")
        backup = install_dir / launcher.BACKUP_DIRNAME
        backup.mkdir()
        launcher._set_updating_flag(install_dir, True)
        launcher._move(launcher._main_exe(install_dir), backup / "EasyScribe.exe")
        launcher._move(install_dir / "_internal", backup / "_internal")
        (install_dir / "_internal").mkdir()  # partly extracted new files
        (install_dir / "_internal" / "half.dll").write_text("half", encoding="utf-8")

        launcher._recover_interrupted_update(install_dir)

        assert launcher._main_exe(install_dir).read_text(encoding="utf-8") == "old exe"
        assert (install_dir / "_internal" / "old.dll").is_file()
        assert not (install_dir / "_internal" / "half.dll").exists()
        assert not backup.exists()
        assert launcher._update_in_progress(install_dir) is False
        assert (install_dir / "recordings" / "meeting.wav").is_file()
        assert launcher._needs_update(install_dir) is True  # the next run updates again


def _check_recover_removes_leftover_backup_of_finished_update() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        install_dir = _make_install(Path(tmp), launcher.VERSION)
        backup = install_dir / launcher.BACKUP_DIRNAME
        (backup / "_internal").mkdir(parents=True)
        launcher._recover_interrupted_update(install_dir)
        assert not backup.exists()
        assert launcher._main_exe(install_dir).is_file()


def _check_clean_incomplete_install_keeps_user_data() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        install_dir = _make_install(Path(tmp), "2.1.0")
        launcher._main_exe(install_dir).unlink()  # incomplete
        launcher._clean_incomplete_install(install_dir)
        assert (install_dir / "recordings" / "meeting.wav").is_file()
        assert (install_dir / "logs" / "easyscribe.log").is_file()
        assert not (install_dir / "_internal").exists()


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
    _check("_shortcuts_created(no marker) -> False", _check_shortcuts_created_false_without_marker)
    _check("_shortcuts_created(marker, not yet marked) -> False", _check_shortcuts_created_false_before_marked)
    _check("_mark_shortcuts_created sets flag, keeps app+version", _check_mark_shortcuts_created_sets_flag_and_keeps_other_fields)
    _check("_ensure_shortcuts creates shortcuts only once", _check_ensure_shortcuts_creates_once)
    _check("_version_key: pre-release < release, numeric order", _check_version_order)
    _check("_needs_update: older/pre-marker yes, same/newer no", _check_needs_update)
    _check("_update_install replaces app files, keeps recordings+logs", _check_update_replaces_app_files_and_keeps_user_data)
    _check("_update_install works on a pre-marker v2.0.0 install", _check_update_premarker_install)
    _check("_update_install while app open -> blocked, old files back", _check_update_blocked_when_app_open_restores_old_files)
    _check("_update_install bad bundle -> old files back", _check_update_failure_during_extract_restores_old_files)
    _check("_recover_interrupted_update puts old files back", _check_recover_interrupted_update)
    _check("_recover_interrupted_update removes a finished backup", _check_recover_removes_leftover_backup_of_finished_update)
    _check("_clean_incomplete_install keeps recordings+logs", _check_clean_incomplete_install_keeps_user_data)
    print("---------------------------------------------------------------------------\n")

    if failures:
        print(f"FAILED: {len(failures)} check(s) failed.", file=sys.stderr)
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
