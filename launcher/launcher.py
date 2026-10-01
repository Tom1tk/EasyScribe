"""
launcher.py - EasyScribe portable GUI installer / launcher.

Compiled as a PyInstaller ONEFILE (console=False) with app.bundle embedded.
On first run: shows a GUI to pick install location, extracts app.bundle there.
On repeat runs: detects existing EasyScribe.exe and launches immediately.
If the installed version is older than this launcher, it first updates the
app files in place. User data (recordings/, logs/) is never moved or deleted.

INSTALL_DIR defaults to <directory containing this exe>/EasyScribe/
"""

import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, ttk
import zipfile
from pathlib import Path

VERSION = "3.0.0-beta2"
MARKER_FILENAME = ".easyscribe-install.json"
# Old app files are moved here during an update, then deleted. It only ever
# holds app files (never user data), so a leftover copy is safe to remove.
BACKUP_DIRNAME = ".easyscribe-old"
# Never deleted by the installer: the user's recordings and transcripts, and logs.
USER_DATA_DIRS = frozenset({"recordings", "logs"})
_exe_dir = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent.parent


def _set_window_icon(win: tk.Tk) -> None:
    """Use the EasyScribe logo for the installer window (bundled by launcher.spec)."""
    base = Path(sys._MEIPASS) if getattr(sys, "frozen", False) else Path(__file__).parent.parent / "assets"
    icon = base / "EasyScribe.ico"
    if sys.platform == "win32" and icon.exists():
        try:
            win.iconbitmap(default=str(icon))
        except tk.TclError:
            pass


def _find_bundle() -> Path | None:
    if getattr(sys, "frozen", False):
        candidate = Path(sys._MEIPASS) / "app.bundle"
        if candidate.is_file():
            return candidate
    candidate = Path(__file__).parent.parent / "app.bundle"
    return candidate if candidate.is_file() else None


def _main_exe(install_dir: Path) -> Path:
    return install_dir / "EasyScribe.exe"


def _resolve_install_dir(chosen: Path) -> Path:
    """
    Resolve a user-picked folder to the actual install directory.

    If `chosen` is already a marked EasyScribe install, or already contains
    EasyScribe.exe (a pre-marker v2.0.0 install), install in place.
    Otherwise, install into `chosen/EasyScribe` so the installer never
    creates or deletes files directly inside a folder it doesn't own.
    """
    if (chosen / MARKER_FILENAME).is_file() or _main_exe(chosen).is_file():
        return chosen
    return chosen / "EasyScribe"


def _safe_to_clean(install_dir: Path) -> bool:
    """
    True if install_dir may be wiped as an incomplete previous install.

    Only directories this installer created (marked with MARKER_FILENAME)
    and that do not contain a complete install (EasyScribe.exe) are safe
    to remove. A directory that doesn't exist yet is trivially safe — there
    is nothing to remove. Anything else (a pre-existing folder this
    installer never marked) must never be deleted.
    """
    if not install_dir.exists():
        return True
    if _main_exe(install_dir).is_file():
        return False
    return (install_dir / MARKER_FILENAME).is_file()


def _version_key(version: str) -> tuple:
    """Sort key for versions like "3.0.0", "3.0.0-beta2", "v2.1".

    A pre-release sorts before its release (3.0.0-beta2 < 3.0.0), and
    alpha < beta < rc. A version that cannot be read sorts first.
    """
    m = re.match(r"^v?(\d+(?:\.\d+)*)(?:-?([a-zA-Z]+)\.?(\d*))?$", str(version).strip())
    if not m:
        return ((0,), (0, 0, 0))
    nums = tuple(int(n) for n in m.group(1).split("."))
    nums = nums + (0,) * (3 - len(nums)) if len(nums) < 3 else nums
    label = (m.group(2) or "").lower()
    if not label:
        return (nums, (9, 0, 0))  # final release
    stage = {"alpha": 1, "a": 1, "beta": 2, "b": 2, "rc": 3}.get(label, 0)
    return (nums, (stage, int(m.group(3) or 0), 0))


def _installed_version(install_dir: Path) -> str | None:
    """The version recorded in the marker, or None if unknown (pre-marker v2.0.0)."""
    try:
        data = json.loads((install_dir / MARKER_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    version = data.get("version") if isinstance(data, dict) else None
    return version if isinstance(version, str) and version else None


def _needs_update(install_dir: Path) -> bool:
    """True if the install at install_dir is older than this launcher."""
    installed = _installed_version(install_dir)
    if installed is None:
        return True
    return _version_key(installed) < _version_key(VERSION)


def _write_marker(install_dir: Path) -> None:
    """Write the ownership marker, backfilling pre-marker v2.0.0 installs."""
    marker = install_dir / MARKER_FILENAME
    if not marker.is_file():
        marker.write_text(json.dumps({"app": "EasyScribe", "version": VERSION}), encoding="utf-8")


def _set_marker_version(install_dir: Path) -> None:
    """Record this launcher's version in the marker, keeping the other fields."""
    marker = install_dir / MARKER_FILENAME
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
    except (OSError, ValueError):
        data = {}
    data["app"] = "EasyScribe"
    data["version"] = VERSION
    data.pop("updating", None)
    marker.write_text(json.dumps(data), encoding="utf-8")


def _shortcuts_created(install_dir: Path) -> bool:
    """True if Desktop/Start Menu shortcuts were already created for this install."""
    marker = install_dir / MARKER_FILENAME
    if not marker.is_file():
        return False
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(data.get("shortcuts_created"))


def _mark_shortcuts_created(install_dir: Path) -> None:
    marker = install_dir / MARKER_FILENAME
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {"app": "EasyScribe", "version": VERSION}
    data["shortcuts_created"] = True
    marker.write_text(json.dumps(data), encoding="utf-8")


def _ps_quote(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _create_shortcut(link_path: Path, target: Path, working_dir: Path) -> None:
    script = (
        f"$s = (New-Object -ComObject WScript.Shell).CreateShortcut({_ps_quote(str(link_path))}); "
        f"$s.TargetPath = {_ps_quote(str(target))}; "
        f"$s.WorkingDirectory = {_ps_quote(str(working_dir))}; "
        f"$s.IconLocation = {_ps_quote(str(target) + ',0')}; "
        f"$s.Save()"
    )
    try:
        link_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            check=True, capture_output=True, timeout=15,
        )
    except Exception:
        pass  # shortcuts are a convenience; the installed exe remains usable directly


def _create_shortcuts(install_dir: Path) -> None:
    """Create Desktop and Start Menu shortcuts to the installed EasyScribe.exe."""
    if sys.platform != "win32":
        return
    target = _main_exe(install_dir)
    _create_shortcut(Path(os.environ["USERPROFILE"]) / "Desktop" / "EasyScribe.lnk", target, install_dir)
    _create_shortcut(
        Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "EasyScribe.lnk",
        target, install_dir,
    )


def _ensure_shortcuts(install_dir: Path) -> None:
    """Create shortcuts once per install, tracked via the marker file."""
    if _shortcuts_created(install_dir):
        return
    _create_shortcuts(install_dir)
    _mark_shortcuts_created(install_dir)


def _refresh_shell_icons() -> None:
    """Ask Explorer to reload icons, so an updated exe does not keep an old cached icon."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        SHCNE_ASSOCCHANGED, SHCNF_IDLIST = 0x08000000, 0x0000
        ctypes.windll.shell32.SHChangeNotify(SHCNE_ASSOCCHANGED, SHCNF_IDLIST, None, None)
    except Exception:
        pass  # cosmetic only


# ── In-place update ───────────────────────────────────────────────────────────

class UpdateBlockedError(RuntimeError):
    """The old app files are in use (EasyScribe is probably still open)."""


# Indirection so tests can simulate a locked file (Windows refuses to rename a
# folder while a program in it is running).
_move = os.replace


def _bundle_app_items(names: list[str]) -> set[str]:
    """Top-level names in the bundle that are app files (not user data)."""
    tops = {n.replace("\\", "/").split("/", 1)[0] for n in names}
    return {t for t in tops if t and t not in USER_DATA_DIRS and t != MARKER_FILENAME}


def _restore_backup(install_dir: Path) -> None:
    """Put the old app files back after a failed or interrupted update."""
    backup = install_dir / BACKUP_DIRNAME
    if not backup.is_dir():
        return
    for item in list(backup.iterdir()):
        target = install_dir / item.name
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        elif target.exists():
            target.unlink()
        _move(item, target)
    shutil.rmtree(backup, ignore_errors=True)


def _set_updating_flag(install_dir: Path, updating: bool) -> None:
    marker = install_dir / MARKER_FILENAME
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
    except (OSError, ValueError):
        data = {"app": "EasyScribe", "version": "0"}
    if updating:
        data["updating"] = True
    else:
        data.pop("updating", None)
    marker.write_text(json.dumps(data), encoding="utf-8")


def _update_in_progress(install_dir: Path) -> bool:
    try:
        data = json.loads((install_dir / MARKER_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and bool(data.get("updating"))


def _recover_interrupted_update(install_dir: Path) -> None:
    """Repair an install whose update stopped part way (power loss, crash).

    The marker carries "updating": true from before the old files are moved
    until the new version is recorded. If it is still set, put the old app
    files back. Otherwise a leftover backup is from a finished update.
    """
    backup = install_dir / BACKUP_DIRNAME
    try:
        if _update_in_progress(install_dir):
            _restore_backup(install_dir)
            _set_updating_flag(install_dir, False)
        elif backup.is_dir():
            shutil.rmtree(backup, ignore_errors=True)
    except OSError:
        pass  # the next update attempt tries again


def _update_install(bundle: Path, install_dir: Path, progress=None) -> None:
    """Replace the app files in install_dir with the files in bundle.

    Only the bundle's top-level app items (EasyScribe.exe, _internal, models,
    ...) are replaced. recordings/, logs/ and any other files stay where they
    are and are never touched. The old app files are first moved to
    BACKUP_DIRNAME; if anything fails they are moved back.

    Raises UpdateBlockedError if the old files cannot be moved because they
    are in use.
    """
    backup = install_dir / BACKUP_DIRNAME
    if backup.exists():
        shutil.rmtree(backup)  # only ever holds app files, see BACKUP_DIRNAME
    with zipfile.ZipFile(bundle, "r") as zf:
        members = zf.namelist()
        app_items = _bundle_app_items(members)

        # 1. Move the old app files aside. Windows refuses this while
        #    EasyScribe is running, so this is also the "is it open?" check.
        backup.mkdir()
        _set_updating_flag(install_dir, True)
        try:
            for name in sorted(app_items):
                old = install_dir / name
                if old.exists():
                    _move(old, backup / name)
        except OSError as exc:
            _restore_backup(install_dir)
            _set_updating_flag(install_dir, False)
            raise UpdateBlockedError(str(exc)) from exc

        # 2. Extract the new app files. User-data folders in the bundle (the
        #    empty recordings/ folder) are only created, never overwritten.
        try:
            app_members = [m for m in members if m.replace("\\", "/").split("/", 1)[0] in app_items]
            total = len(app_members)
            for i, member in enumerate(app_members, start=1):
                zf.extract(member, install_dir)
                if progress and (i % 200 == 0 or i == total):
                    progress(int(100 * i / total))
            for name in USER_DATA_DIRS:
                (install_dir / name).mkdir(exist_ok=True)
        except BaseException:
            _restore_backup(install_dir)
            _set_updating_flag(install_dir, False)
            raise

    # 3. Record the new version (this also clears "updating"), then remove
    #    the old files.
    _set_marker_version(install_dir)
    shutil.rmtree(backup, ignore_errors=True)
    _refresh_shell_icons()


def _clean_incomplete_install(install_dir: Path) -> None:
    """Remove a marked, incomplete install, but keep user data (recordings, logs)."""
    for item in install_dir.iterdir():
        if item.name in USER_DATA_DIRS:
            continue
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
        else:
            try:
                item.unlink()
            except OSError:
                pass


class InstallerApp(tk.Tk):
    def __init__(self, install_dir: Path | None = None, auto_update: bool = False):
        super().__init__()
        self.title(f"EasyScribe {VERSION}")
        _set_window_icon(self)
        self.resizable(False, False)
        self._install_dir = tk.StringVar(value=str(install_dir or _exe_dir / "EasyScribe"))
        self._status_text = tk.StringVar()
        self._q: queue.Queue = queue.Queue()
        self._build_ui()
        self._refresh_state()
        if auto_update:
            # Started by main() for an older install: update without a click
            self.after(300, self._on_action)
        # Centre on screen
        self.update_idletasks()
        w, h = self.winfo_width(), self.winfo_height()
        x = (self.winfo_screenwidth() - w) // 2
        y = (self.winfo_screenheight() - h) // 2
        self.geometry(f"+{x}+{y}")

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        outer = tk.Frame(self, padx=20, pady=14)
        outer.pack(fill="both", expand=True)

        tk.Label(outer, text=f"EasyScribe  {VERSION}", font=("Segoe UI", 13, "bold")).pack(anchor="w")
        tk.Label(outer, text="Portable offline speech-to-text", fg="#555555").pack(anchor="w", pady=(0, 14))

        # Install location
        tk.Label(outer, text="Install location:", anchor="w").pack(fill="x")
        row = tk.Frame(outer)
        row.pack(fill="x", pady=(2, 12))
        self._dir_entry = tk.Entry(row, textvariable=self._install_dir, width=44)
        self._dir_entry.pack(side="left", ipady=3)
        self._browse_btn = tk.Button(row, text="Browse…", command=self._browse)
        self._browse_btn.pack(side="left", padx=(6, 0))

        # Progress bar
        self._bar = ttk.Progressbar(outer, length=440, mode="determinate", maximum=100)
        self._bar.pack(fill="x", pady=(0, 6))

        # Status line
        tk.Label(outer, textvariable=self._status_text, anchor="w", fg="#333333").pack(fill="x", pady=(0, 12))

        # Action button
        self._btn = tk.Button(outer, text="Install", width=20, height=2,
                              font=("Segoe UI", 10), command=self._on_action)
        self._btn.pack()

        # Wire up dir entry changes
        self._install_dir.trace_add("write", lambda *_: self._refresh_state())

    # ── State management ──────────────────────────────────────────────────────

    def _refresh_state(self):
        install_dir = Path(self._install_dir.get().strip())
        if _main_exe(install_dir).is_file() and _needs_update(install_dir):
            installed = _installed_version(install_dir) or "an older version"
            self._status_text.set(
                f"EasyScribe {installed} is installed at:\n{install_dir}\n"
                f"It will be updated to {VERSION}. Your recordings are kept."
            )
            self._btn.config(text="Update and open", state="normal")
            self._bar["value"] = 0
        elif _main_exe(install_dir).is_file():
            self._status_text.set(f"Already installed at:\n{install_dir}")
            self._btn.config(text="Open EasyScribe", state="normal")
            self._bar["value"] = 100
        else:
            self._status_text.set("Ready to install.")
            self._btn.config(text="Install", state="normal")
            self._bar["value"] = 0

    def _browse(self):
        chosen = filedialog.askdirectory(
            initialdir=self._install_dir.get() or str(_exe_dir),
            title="Choose install location",
        )
        if chosen:
            self._install_dir.set(str(_resolve_install_dir(Path(chosen))))

    # ── Actions ───────────────────────────────────────────────────────────────

    def _on_action(self):
        install_dir = Path(self._install_dir.get().strip())
        _recover_interrupted_update(install_dir)
        if _main_exe(install_dir).is_file() and _needs_update(install_dir):
            self._do_install(install_dir, update=True)
        elif _main_exe(install_dir).is_file():
            self._do_launch(install_dir)
        else:
            self._do_install(install_dir)

    def _do_launch(self, install_dir: Path):
        _write_marker(install_dir)  # backfill marker for pre-marker v2.0.0 installs
        _ensure_shortcuts(install_dir)
        subprocess.Popen([str(_main_exe(install_dir))], cwd=str(install_dir))
        self.destroy()

    def _do_install(self, install_dir: Path, update: bool = False):
        bundle = _find_bundle()
        if bundle is None:
            self._status_text.set("ERROR: app.bundle not found. Re-download the installer.")
            return
        self._btn.config(state="disabled")
        self._dir_entry.config(state="disabled")
        self._browse_btn.config(state="disabled")
        target = self._update_thread if update else self._extract_thread
        threading.Thread(target=target, args=(bundle, install_dir), daemon=True).start()
        self.after(80, self._poll_queue)

    # ── Background extraction ─────────────────────────────────────────────────

    def _update_thread(self, bundle: Path, install_dir: Path):
        self._q.put(("status", f"Updating EasyScribe to {VERSION}… Your recordings are kept."))
        try:
            _update_install(bundle, install_dir, lambda pct: self._q.put(("progress", pct)))
            self._q.put(("done", install_dir))
        except UpdateBlockedError:
            self._q.put((
                "error",
                "EasyScribe is still open, or a file in its folder is in use.\n"
                "Close EasyScribe, then click Update and open again.",
            ))
        except Exception as exc:
            self._q.put(("error", f"{exc}\nThe previous version was kept."))

    def _extract_thread(self, bundle: Path, install_dir: Path):
        try:
            if install_dir.exists() and not _main_exe(install_dir).is_file():
                if _safe_to_clean(install_dir):
                    self._q.put(("status", "Cleaning incomplete previous install…"))
                    _clean_incomplete_install(install_dir)
                elif any(install_dir.iterdir()):
                    self._q.put((
                        "error",
                        f"'{install_dir}' is not empty and was not created by "
                        f"EasyScribe. Please choose an empty or new folder.",
                    ))
                    return
            install_dir.mkdir(parents=True, exist_ok=True)
            _write_marker(install_dir)
            with zipfile.ZipFile(bundle, "r") as zf:
                members = zf.namelist()
                total = len(members)
                for i, member in enumerate(members, start=1):
                    zf.extract(member, install_dir)
                    if i % 200 == 0 or i == total:
                        pct = int(100 * i / total)
                        self._q.put(("progress", pct))
            self._q.put(("done", install_dir))
        except Exception as exc:
            if _safe_to_clean(install_dir) and install_dir.exists():
                _clean_incomplete_install(install_dir)
            self._q.put(("error", str(exc)))

    def _poll_queue(self):
        try:
            while True:
                kind, value = self._q.get_nowait()
                if kind == "status":
                    self._status_text.set(value)
                elif kind == "progress":
                    self._bar["value"] = value
                    self._status_text.set(f"Extracting… {value}%")
                elif kind == "done":
                    self._bar["value"] = 100
                    self._status_text.set(
                        "Installation complete. Opening EasyScribe…\n"
                        "Next time, open it from the desktop or Start menu shortcut."
                    )
                    self.after(1500, lambda v=value: self._do_launch(v))
                    return
                elif kind == "error":
                    self._status_text.set(f"Could not finish: {value}")
                    self._btn.config(state="normal")
                    self._dir_entry.config(state="normal")
                    self._browse_btn.config(state="normal")
                    return
        except queue.Empty:
            pass
        self.after(80, self._poll_queue)


def main():
    default_dir = _exe_dir / "EasyScribe"
    _recover_interrupted_update(default_dir)
    if _main_exe(default_dir).is_file():
        if _needs_update(default_dir):
            # Older install (e.g. v2 next to a new v3 exe) — update it first.
            # A newer install is opened as it is; it is never downgraded.
            app = InstallerApp(install_dir=default_dir, auto_update=True)
            app.mainloop()
            return
        # Same version at the default location — launch silently, no GUI shown
        _write_marker(default_dir)
        _ensure_shortcuts(default_dir)  # backfill shortcuts for pre-8.3 installs
        subprocess.Popen([str(_main_exe(default_dir))], cwd=str(default_dir))
        return
    # First run (or non-default install) — show the installer GUI
    app = InstallerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
