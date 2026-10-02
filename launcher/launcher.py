"""
launcher.py - EasyScribe portable GUI installer / launcher.

Compiled as a PyInstaller ONEFILE (console=False) with app.bundle embedded.
On first run: shows a GUI to pick install location, extracts app.bundle there.
On repeat runs: detects existing EasyScribe.exe and launches immediately.
If the installed version is older than this launcher, it shows the installer
and updates the app files in place only when the user selects "Update and
open". User data (recordings/, logs/) is never moved or deleted.

The bootloader shows a splash (launcher.spec) while it unpacks; it is
closed when the installer window or the app window is open.

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
from tkinter import filedialog
import zipfile
from pathlib import Path

import customtkinter as ctk

VERSION = "3.1.0"
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


# ─── Look: the same colours, fonts and logo as the app (src/gui.py) ─────────

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")  # base theme; every widget below overrides its colours


class C:
    """Colour tokens. A copy of the ones in src/gui.py that the installer uses."""

    BG = "#F4F6FB"
    SURFACE = "#FFFFFF"
    SURFACE_ALT = "#EEF1F7"
    BORDER = "#DCE1EC"
    INK = "#131A2B"
    MUTED = "#55607A"
    FAINT = "#9AA3B8"
    SECONDARY = "#E8ECF4"
    SECONDARY_HOVER = "#DCE1EC"
    TEAL = "#0B7A75"
    TEAL_HOVER = "#08625E"
    TEAL_SOFT = "#DDF3F1"
    TEAL_INK = "#075E5A"
    GREEN_INK = "#16613A"
    CRIMSON = "#A61F35"


R_SM, R_MD, R_LG = 6, 10, 14


def _ui_family() -> str | None:
    return "Segoe UI" if sys.platform == "win32" else None


def _logo(parent, size: int = 40) -> tk.Canvas:  # type: ignore[no-untyped-def]
    """The EasyScribe mark: a teal tile with a white sound wave (as in src/gui.py)."""
    canvas = tk.Canvas(parent, width=size, height=size, bg=C.BG, highlightthickness=0, bd=0)
    r, s = size * 0.28, size - 1
    points = [
        r, 0, s - r, 0, s, 0, s, r, s, s - r, s, s,
        s - r, s, r, s, 0, s, 0, s - r, 0, r, 0, 0,
    ]
    canvas.create_polygon(points, smooth=True, fill=C.TEAL, outline=C.TEAL)
    bar_w = size * 0.09
    gap = size * 0.07
    heights = (0.22, 0.46, 0.62, 0.38, 0.18)
    x = (size - (len(heights) * bar_w + (len(heights) - 1) * gap)) / 2
    mid = size / 2
    for h in heights:
        half = size * h / 2
        canvas.create_line(
            x + bar_w / 2, mid - half, x + bar_w / 2, mid + half,
            fill=C.SURFACE, width=bar_w, capstyle="round",
        )
        x += bar_w + gap
    return canvas


def _close_splash() -> None:
    """Close the bootloader's "Getting ready" window (see launcher.spec)."""
    try:
        import pyi_splash  # type: ignore  # only exists in the built exe

        pyi_splash.close()
    except Exception:
        pass


def _wait_until_started(proc: subprocess.Popen, timeout_ms: int = 15000) -> None:
    """Wait until EasyScribe.exe is ready for input (its window is open), or the timeout.

    So the splash or the installer stays on screen until the app shows,
    and the user never sees nothing at all.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.user32.WaitForInputIdle(int(proc._handle), timeout_ms)  # type: ignore[attr-defined]
    except Exception:
        pass


def _start_app(install_dir: Path) -> subprocess.Popen:
    _write_marker(install_dir)  # backfill marker for pre-marker v2.0.0 installs
    _ensure_shortcuts(install_dir)
    return subprocess.Popen([str(_main_exe(install_dir))], cwd=str(install_dir))


class InstallerApp(ctk.CTk):
    """Install, update or open EasyScribe. Nothing starts until the user clicks."""

    def __init__(self, install_dir: Path | None = None):
        super().__init__()
        self.title(f"EasyScribe {VERSION}")
        _set_window_icon(self)
        self.resizable(False, False)
        self.configure(fg_color=C.BG)
        family = _ui_family()
        self._f_app = ctk.CTkFont(family=family, size=22, weight="bold")
        self._f_body = ctk.CTkFont(family=family, size=13)
        self._f_strong = ctk.CTkFont(family=family, size=13, weight="bold")
        self._f_small = ctk.CTkFont(family=family, size=12)
        self._f_primary = ctk.CTkFont(family=family, size=14, weight="bold")

        self._install_dir = ctk.StringVar(value=str(install_dir or _exe_dir / "EasyScribe"))
        self._q: queue.Queue = queue.Queue()
        self._busy = False
        self._build_ui()
        self._refresh_state()
        # Centre on screen
        self.update_idletasks()
        w, h = self.winfo_width(), self.winfo_height()
        x = (self.winfo_screenwidth() - w) // 2
        y = (self.winfo_screenheight() - h) // 3
        self.geometry(f"+{x}+{y}")
        # Close the splash only when this window is on screen
        self.after(200, _close_splash)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, padx=24, pady=(20, 4), sticky="ew")
        _logo(header, 40).grid(row=0, column=0, rowspan=2, padx=(0, 12))
        title_row = ctk.CTkFrame(header, fg_color="transparent")
        title_row.grid(row=0, column=1, sticky="w")
        ctk.CTkLabel(title_row, text="EasyScribe", font=self._f_app, text_color=C.INK).pack(
            side="left"
        )
        ctk.CTkLabel(
            title_row, text=f" {VERSION} ", font=self._f_small, text_color=C.TEAL_INK,
            fg_color=C.TEAL_SOFT, corner_radius=R_SM, height=22,
        ).pack(side="left", padx=(10, 0))
        ctk.CTkLabel(
            header, text="Private speech-to-text. Everything stays on this computer.",
            font=self._f_body, text_color=C.MUTED,
        ).grid(row=1, column=1, sticky="w")

        card = ctk.CTkFrame(
            self, fg_color=C.SURFACE, corner_radius=R_LG, border_width=1, border_color=C.BORDER
        )
        card.grid(row=1, column=0, padx=24, pady=(14, 0), sticky="ew")
        card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(card, text="Install location", font=self._f_strong, text_color=C.INK).grid(
            row=0, column=0, columnspan=2, padx=18, pady=(16, 6), sticky="w"
        )
        self._dir_entry = ctk.CTkEntry(
            card, textvariable=self._install_dir, width=380, height=36, corner_radius=R_MD,
            font=self._f_body, fg_color=C.SURFACE, border_color=C.BORDER, text_color=C.INK,
        )
        self._dir_entry.grid(row=1, column=0, padx=(18, 8), sticky="ew")
        self._browse_btn = ctk.CTkButton(
            card, text="Browse…", width=96, height=36, corner_radius=R_MD, border_width=0,
            font=self._f_strong, fg_color=C.SECONDARY, hover_color=C.SECONDARY_HOVER,
            text_color=C.INK, text_color_disabled=C.FAINT, command=self._browse,
        )
        self._browse_btn.grid(row=1, column=1, padx=(0, 18))

        self._status = ctk.CTkLabel(
            card, text="", font=self._f_body, text_color=C.INK, anchor="w", justify="left",
            wraplength=470,
        )
        self._status.grid(row=2, column=0, columnspan=2, padx=18, pady=(14, 16), sticky="w")

        self._bar = ctk.CTkProgressBar(
            card, height=8, corner_radius=4, fg_color=C.SURFACE_ALT, progress_color=C.TEAL,
        )
        self._bar.set(0)
        # Shown only while files are copied (see _set_inputs)
        self._bar.grid(row=3, column=0, columnspan=2, padx=18, pady=(0, 18), sticky="ew")
        self._bar.grid_remove()

        foot = ctk.CTkFrame(self, fg_color="transparent")
        foot.grid(row=2, column=0, padx=24, pady=(14, 20), sticky="ew")
        foot.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            foot, text="No admin rights needed. Adds Desktop and Start menu shortcuts.",
            font=self._f_small, text_color=C.MUTED, anchor="w",
        ).grid(row=0, column=0, sticky="w")
        self._btn = ctk.CTkButton(
            foot, text="Install", width=170, height=40, corner_radius=R_MD, border_width=0,
            font=self._f_primary, fg_color=C.TEAL, hover_color=C.TEAL_HOVER,
            text_color=C.SURFACE, text_color_disabled=C.SURFACE, command=self._on_action,
        )
        self._btn.grid(row=0, column=1, padx=(12, 0))
        self.bind("<Return>", lambda _e: self._on_action())

        # Wire up dir entry changes
        self._install_dir.trace_add("write", lambda *_: self._refresh_state())

    def _set_status(self, text: str, tone: str = "normal") -> None:
        colour = {"normal": C.INK, "muted": C.MUTED, "done": C.GREEN_INK, "error": C.CRIMSON}[tone]
        self._status.configure(text=text, text_color=colour)

    def _set_inputs(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self._busy = not enabled
        self._btn.configure(state=state)
        self._browse_btn.configure(state=state)
        self._dir_entry.configure(state=state, text_color=C.INK if enabled else C.MUTED)
        self._btn.configure(fg_color=C.TEAL if enabled else C.FAINT)
        if enabled:
            self._bar.grid_remove()
        else:
            self._bar.grid()

    # ── State management ──────────────────────────────────────────────────────

    def _refresh_state(self):
        if self._busy:
            return
        install_dir = Path(self._install_dir.get().strip())
        if _main_exe(install_dir).is_file() and _needs_update(install_dir):
            installed = _installed_version(install_dir) or "an older version"
            self._set_status(
                f"EasyScribe {installed} is installed in this folder.\n"
                f"Select Update and open to update it to {VERSION}. "
                f"Your recordings are kept.\n"
                f"To install in a different folder, select Browse."
            )
            self._btn.configure(text="Update and open")
        elif _main_exe(install_dir).is_file():
            self._set_status("EasyScribe is already installed in this folder.", "muted")
            self._btn.configure(text="Open EasyScribe")
        else:
            self._set_status("EasyScribe will be installed in this folder.", "muted")
            self._btn.configure(text="Install")

    def _browse(self):
        chosen = filedialog.askdirectory(
            parent=self,
            initialdir=self._install_dir.get() or str(_exe_dir),
            title="Choose install location",
        )
        if chosen:
            self._install_dir.set(str(_resolve_install_dir(Path(chosen))))

    # ── Actions ───────────────────────────────────────────────────────────────

    def _on_action(self):
        if self._busy:
            return
        install_dir = Path(self._install_dir.get().strip())
        _recover_interrupted_update(install_dir)
        if _main_exe(install_dir).is_file() and _needs_update(install_dir):
            self._do_install(install_dir, update=True)
        elif _main_exe(install_dir).is_file():
            self._do_launch(install_dir)
        else:
            self._do_install(install_dir)

    def _do_launch(self, install_dir: Path):
        after_install = self._busy  # then keep its "Done" text and full bar
        self._set_inputs(False)
        if not after_install:
            self._set_status("Opening EasyScribe…", "done")
            self._bar.grid_remove()
        proc = _start_app(install_dir)
        started = threading.Event()

        def wait():
            _wait_until_started(proc)
            started.set()

        threading.Thread(target=wait, daemon=True).start()

        def close_when_started():
            if started.is_set():
                self.destroy()
            else:
                self.after(100, close_when_started)

        self.after(100, close_when_started)

    def _do_install(self, install_dir: Path, update: bool = False):
        bundle = _find_bundle()
        if bundle is None:
            self._set_status("The app files are missing from this installer. Download it again.", "error")
            return
        self._bar.set(0)
        self._set_inputs(False)
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
                "Close EasyScribe, then select Update and open again.",
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
                    self._set_status(value)
                elif kind == "progress":
                    self._bar.set(value / 100)
                    self._set_status(f"Copying the app files… {value}%")
                elif kind == "done":
                    self._bar.set(1)
                    self._set_status(
                        "Done. Opening EasyScribe…\n"
                        "Next time, open it from the Desktop or Start menu shortcut.",
                        "done",
                    )
                    self.after(1500, lambda v=value: self._do_launch(v))
                    return
                elif kind == "error":
                    self._set_status(f"Could not finish: {value}", "error")
                    self._set_inputs(True)
                    return
        except queue.Empty:
            pass
        self.after(80, self._poll_queue)


def main():
    default_dir = _exe_dir / "EasyScribe"
    _recover_interrupted_update(default_dir)
    if _main_exe(default_dir).is_file() and not _needs_update(default_dir):
        # Same or newer version at the default location: open it, no installer
        # window. A newer install is never downgraded. Keep the splash until
        # the app window opens.
        _wait_until_started(_start_app(default_dir))
        _close_splash()
        return
    # First run, or an older install (e.g. v2 next to a new v3 exe): show the
    # installer and wait for the user. They can keep the folder or choose
    # another one before anything is changed.
    app = InstallerApp(install_dir=default_dir)
    app.mainloop()


if __name__ == "__main__":
    main()
