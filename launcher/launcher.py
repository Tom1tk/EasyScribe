"""
launcher.py - EasyScribe portable GUI installer / launcher.

Compiled as a PyInstaller ONEFILE (console=False) with app.bundle embedded.
On first run: shows a GUI to pick install location, extracts app.bundle there.
On repeat runs: detects existing EasyScribe.exe and launches immediately.

INSTALL_DIR defaults to <directory containing this exe>/EasyScribe/
"""

import json
import queue
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, ttk
import zipfile
from pathlib import Path

VERSION = "2.0.0"
MARKER_FILENAME = ".easyscribe-install.json"
_exe_dir = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent.parent


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


def _write_marker(install_dir: Path) -> None:
    """Write the ownership marker, backfilling pre-marker v2.0.0 installs."""
    marker = install_dir / MARKER_FILENAME
    if not marker.is_file():
        marker.write_text(json.dumps({"app": "EasyScribe", "version": VERSION}), encoding="utf-8")


class InstallerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"EasyScribe {VERSION}")
        self.resizable(False, False)
        self._install_dir = tk.StringVar(value=str(_exe_dir / "EasyScribe"))
        self._status_text = tk.StringVar()
        self._q: queue.Queue = queue.Queue()
        self._build_ui()
        self._refresh_state()
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
        if _main_exe(install_dir).is_file():
            self._status_text.set(f"Already installed at:\n{install_dir}")
            self._btn.config(text="Launch EasyScribe", state="normal")
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
        if _main_exe(install_dir).is_file():
            self._do_launch(install_dir)
        else:
            self._do_install(install_dir)

    def _do_launch(self, install_dir: Path):
        _write_marker(install_dir)  # backfill marker for pre-marker v2.0.0 installs
        subprocess.Popen([str(_main_exe(install_dir))], cwd=str(install_dir))
        self.destroy()

    def _do_install(self, install_dir: Path):
        bundle = _find_bundle()
        if bundle is None:
            self._status_text.set("ERROR: app.bundle not found. Re-download the installer.")
            return
        self._btn.config(state="disabled")
        self._dir_entry.config(state="disabled")
        self._browse_btn.config(state="disabled")
        threading.Thread(target=self._extract_thread, args=(bundle, install_dir), daemon=True).start()
        self.after(80, self._poll_queue)

    # ── Background extraction ─────────────────────────────────────────────────

    def _extract_thread(self, bundle: Path, install_dir: Path):
        try:
            if install_dir.exists() and not _main_exe(install_dir).is_file():
                if _safe_to_clean(install_dir):
                    self._q.put(("status", "Cleaning incomplete previous install…"))
                    shutil.rmtree(install_dir, ignore_errors=True)
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
            if _safe_to_clean(install_dir):
                shutil.rmtree(install_dir, ignore_errors=True)
            self._q.put(("error", str(exc)))

    def _poll_queue(self):
        try:
            while True:
                kind, value = self._q.get_nowait()
                if kind == "progress":
                    self._bar["value"] = value
                    self._status_text.set(f"Extracting… {value}%")
                elif kind == "done":
                    self._bar["value"] = 100
                    self._status_text.set("Installation complete! Launching EasyScribe…")
                    self.after(900, lambda v=value: self._do_launch(v))
                    return
                elif kind == "error":
                    self._status_text.set(f"Installation failed: {value}")
                    self._btn.config(state="normal")
                    self._dir_entry.config(state="normal")
                    self._browse_btn.config(state="normal")
                    return
        except queue.Empty:
            pass
        self.after(80, self._poll_queue)


def main():
    # Already installed at the default location — launch silently, no GUI shown
    default_dir = _exe_dir / "EasyScribe"
    if _main_exe(default_dir).is_file():
        _write_marker(default_dir)  # backfill marker for pre-marker v2.0.0 installs
        subprocess.Popen([str(_main_exe(default_dir))], cwd=str(default_dir))
        return
    # First run (or non-default install) — show the installer GUI
    app = InstallerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
