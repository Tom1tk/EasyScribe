"""
gui.py - CustomTkinter main window for EasyScribe v2.

All transcription and recording work runs in daemon background threads.
The GUI communicates with worker threads exclusively via self.after(0, ...) callbacks.

Design notes (for whoever edits this next):
  - Audience: non-technical people transcribing private recordings on their
    own computer. Calm, plain language, one clear next step at a time.
  - Light theme only. Neutrals are one cool slate family; every feature owns
    one colour and that colour follows the feature everywhere (mode switch,
    step badges, option chips, progress bar, status text):
        Transcribe files -> teal     Record -> coral
        Speakers         -> amber    Timestamps  -> blue
        Offline / done   -> green    Errors      -> crimson
  - All text/background pairs below pass WCAG AA (4.5:1).
  - Visible text: sentence case, no em/en dashes, no exclamation marks.
"""

import functools
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Callable

import tkinter as tk

import customtkinter as ctk  # type: ignore

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore
    _DND_AVAILABLE = True
except Exception:
    DND_FILES = None  # type: ignore
    _DND_AVAILABLE = False

_AppBase = ctk.CTk  # type: ignore

from config import (
    APP_ICON,
    APP_ICON_PNG,
    APP_NAME,
    APP_VERSION,
    DEFAULT_RECORDINGS_DIR,
    LOGS_DIR,
    MAX_LOG_FILES,
    MIN_FREE_DISK_BYTES,
    RECORDING_WORK_DIR,
    SUPPORTED_EXTENSIONS,
)
import settings
from common import CancelledError, wav_duration
from ffmpeg_wrapper import (
    FFmpegExtractionError,
    FFmpegNotFoundError,
    extract_audio,
)
from diarizer import DiarizationEngine, estimate_seconds, fmt_estimate, speech_seconds  # type: ignore
from transcriber import (
    ModelNotFoundError,
    TranscriptionEngine,
    TranscriptionError,
    add_speakers_to_file,
)
from mic_recorder import MicRecorder
from live_transcriber import LiveTranscriber
import transitions
from rename import check_name, recording_set, rename_all
import win_paint

logger = logging.getLogger(__name__)

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")  # base theme; every widget below overrides its colours


# ─── Palette ──────────────────────────────────────────────────────────────────


_FILE_ROW_HEIGHT = 46  # two text lines plus row padding, in px


class C:
    """Colour tokens. Light theme only."""

    # Neutrals: one cool slate family
    BG = "#F4F6FB"
    SURFACE = "#FFFFFF"
    SURFACE_ALT = "#EEF1F7"
    BORDER = "#DCE1EC"
    BORDER_STRONG = "#C3CAD9"
    INK = "#131A2B"
    MUTED = "#55607A"
    FAINT = "#9AA3B8"
    SECONDARY = "#E8ECF4"
    SECONDARY_HOVER = "#DCE1EC"

    # Transcribe files
    TEAL = "#0B7A75"
    TEAL_HOVER = "#08625E"
    TEAL_SOFT = "#DDF3F1"
    TEAL_INK = "#075E5A"

    # Record
    CORAL = "#C8402A"
    CORAL_HOVER = "#A8341F"
    CORAL_SOFT = "#FDE7E1"
    CORAL_INK = "#A3341C"

    # Speakers
    AMBER = "#B87400"
    AMBER_SOFT = "#FFF2D9"
    AMBER_INK = "#8A5300"

    # Timestamps
    SKY = "#2F74C0"
    SKY_SOFT = "#E2EEFB"
    SKY_INK = "#1D5FA0"

    # Offline, privacy, success
    GREEN = "#1F8A53"
    GREEN_SOFT = "#E1F4E8"
    GREEN_INK = "#16613A"

    # Errors
    CRIMSON = "#A61F35"
    CRIMSON_SOFT = "#FCE8EB"


# Radius scale: small chips, controls, cards
R_SM, R_MD, R_LG = 6, 10, 14

# Internal status names (sent by transcriber.py and the workers) mapped to
# (plain-language title, text colour, progress-bar colour, progress mode).
# Progress mode: "bar" = real percentage, "busy" = moving bar, "pause" = still.
_STATUS: dict[str, tuple[str, str, str, str]] = {
    "Ready": ("Ready", C.MUTED, C.TEAL, "pause"),
    "Loading Model": ("Getting ready", C.INK, C.TEAL, "busy"),
    "Extracting Audio": ("Reading the audio", C.INK, C.TEAL, "busy"),
    "Transcribing": ("Transcribing", C.TEAL_INK, C.TEAL, "bar"),
    "Identifying Speakers": ("Finding the speakers", C.AMBER_INK, C.AMBER, "busy"),
    "Naming Speakers": ("Waiting for speaker names", C.AMBER_INK, C.AMBER, "pause"),
    "Writing Transcript": ("Saving the transcript", C.TEAL_INK, C.TEAL, "bar"),
    "Recording": ("Recording", C.CORAL_INK, C.CORAL, "busy"),
    "Finishing": ("Finishing the last words", C.CORAL_INK, C.CORAL, "busy"),
    "Saving Recording": ("Saving the recording", C.CORAL_INK, C.CORAL, "busy"),
    "Cancelling…": ("Stopping…", C.MUTED, C.FAINT, "busy"),
    "Done": ("Done", C.GREEN_INK, C.GREEN, "pause"),
    "Cancelled": ("Stopped", C.MUTED, C.FAINT, "pause"),
    "Failed": ("Something went wrong", C.CRIMSON, C.CRIMSON, "pause"),
}

_STATUS_HINTS: dict[str, str] = {
    "Loading Model": "Loading the speech model. This can take a minute.",
    "Identifying Speakers": "This step can take a while for long recordings.",
    "Naming Speakers": "Type each speaker's name below.",
}

_MODE_FILES = "Transcribe files"
_MODE_RECORD = "Record"

# Record tab: what to do with the recording
_REC_BEST = "best"
_REC_LIVE = "live"


def _ui_family() -> str | None:
    if sys.platform == "win32":
        return "Segoe UI"
    return None


def _mono_family() -> str:
    return "Consolas" if sys.platform == "win32" else "DejaVu Sans Mono"


def _shorten(text: str, limit: int) -> str:
    """Shorten *text* in the middle so the start and the extension stay visible."""
    if len(text) <= limit:
        return text
    keep = limit - 1
    head = keep * 2 // 3
    return text[:head] + "…" + text[-(keep - head):]


def _fmt_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("bytes", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "bytes" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _open_path(path: Path) -> None:
    """Open a file or folder with the default local program."""
    try:
        if sys.platform == "win32":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as exc:
        logger.warning(f"Could not open {path}: {exc}")
        messagebox.showerror("Could not open", f"Could not open:\n{path}\n\n{exc}")


# ─── Small widget factories ───────────────────────────────────────────────────


def _button(parent, text: str, kind: str = "secondary", **kw) -> ctk.CTkButton:  # type: ignore[no-untyped-def]
    """Create a button. kind: teal | coral | amber | secondary | subtle | link."""
    styles = {
        "teal": dict(fg_color=C.TEAL, hover_color=C.TEAL_HOVER, text_color=C.SURFACE),
        "amber": dict(fg_color=C.AMBER, hover_color="#946000", text_color=C.SURFACE),
        "coral": dict(fg_color=C.CORAL, hover_color=C.CORAL_HOVER, text_color=C.SURFACE),
        "secondary": dict(fg_color=C.SECONDARY, hover_color=C.SECONDARY_HOVER, text_color=C.INK),
        # Quiet but clearly a button: white with a thin outline
        "subtle": dict(
            fg_color=C.SURFACE, hover_color=C.SURFACE_ALT, text_color=C.MUTED,
            border_width=1, border_color=C.BORDER,
        ),
        "link": dict(fg_color="transparent", hover_color=C.SURFACE_ALT, text_color=C.TEAL_INK),
    }
    opts = dict(
        text=text,
        corner_radius=R_MD,
        height=36,
        border_width=0,
        text_color_disabled=C.FAINT,
    )
    opts.update(styles[kind])
    opts.update(kw)
    return ctk.CTkButton(parent, **opts)


def _fmt_minutes(seconds: float) -> str:
    """"18 min" for a duration; under a minute is "under 1 min"."""
    return "under 1 min" if seconds < 60 else f"{round(seconds / 60)} min"


def _fmt_took(seconds: float) -> str:
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes} min {secs} s" if minutes else f"{secs} s"


def _logo(parent, size: int = 40) -> tk.Canvas:  # type: ignore[no-untyped-def]
    """The EasyScribe mark: a teal tile with a white sound wave."""
    canvas = tk.Canvas(
        parent, width=size, height=size, bg=C.BG, highlightthickness=0, bd=0
    )
    r, s = size * 0.28, size - 1
    # Rounded square from a smoothed polygon (tk has no rounded rectangle).
    points = [
        r, 0, s - r, 0, s, 0, s, r, s, s - r, s, s,
        s - r, s, r, s, 0, s, 0, s - r, 0, r, 0, 0,
    ]
    canvas.create_polygon(points, smooth=True, fill=C.TEAL, outline=C.TEAL)
    bar_w = size * 0.09
    gap = size * 0.07
    heights = (0.22, 0.46, 0.62, 0.38, 0.18)
    total = len(heights) * bar_w + (len(heights) - 1) * gap
    x = (size - total) / 2
    mid = size / 2
    for h in heights:
        half = size * h / 2
        canvas.create_line(
            x + bar_w / 2, mid - half, x + bar_w / 2, mid + half,
            fill=C.SURFACE, width=bar_w, capstyle="round",
        )
        x += bar_w + gap
    return canvas


def _set_window_icon(win) -> None:  # type: ignore[no-untyped-def]
    """Show the EasyScribe logo in the title bar and taskbar.

    CTk and CTkToplevel replace the icon with the CustomTkinter one 200 ms
    after creation unless iconbitmap() was called, so every window calls this.
    """
    try:
        if sys.platform == "win32" and APP_ICON.exists():
            win.iconbitmap(str(APP_ICON))
        elif APP_ICON_PNG.exists():
            win._easyscribe_icon = tk.PhotoImage(file=str(APP_ICON_PNG))
            win.iconphoto(False, win._easyscribe_icon)
    except Exception as exc:
        logger.debug(f"Could not set window icon: {exc}")


def _card(parent) -> ctk.CTkFrame:  # type: ignore[no-untyped-def]
    return ctk.CTkFrame(
        parent, fg_color=C.SURFACE, corner_radius=R_LG, border_width=1, border_color=C.BORDER
    )


# ─── Speaker naming dialog ────────────────────────────────────────────────────


class SpeakerNamingPanel(ctk.CTkFrame):
    """Speaker names, typed in the main window once the speakers are found.

    Shown in the activity panel instead of a popup window: a popup could open
    behind the main window and block it, so the app looked frozen.
    """

    def __init__(
        self,
        parent,  # type: ignore[no-untyped-def]
        speaker_map: dict[str, str],
        clips_dict: dict[str, Path],
        shares: dict[str, float],
        on_done: Callable[[], None],
    ) -> None:
        super().__init__(
            parent, fg_color=C.SURFACE, corner_radius=R_MD, border_width=2, border_color=C.AMBER
        )
        self._speaker_map = speaker_map
        self._clips_dict = clips_dict
        self._shares = shares
        self._on_done = on_done
        self._entries: dict[str, ctk.CTkEntry] = {}
        family = _ui_family()
        self._f_title = ctk.CTkFont(family=family, size=18, weight="bold")
        self._f_body = ctk.CTkFont(family=family, size=13)
        self._f_strong = ctk.CTkFont(family=family, size=13, weight="bold")
        self._f_small = ctk.CTkFont(family=family, size=12)
        self._build()

        first = next(iter(self._entries.values()), None)
        if first is not None:
            self.after(100, first.focus_set)

    def _build(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        count = len(self._speaker_map)
        ctk.CTkLabel(
            self,
            text=f"We found {count} speaker{'s' if count != 1 else ''}",
            font=self._f_title, text_color=C.INK,
        ).grid(row=0, column=0, padx=20, pady=(18, 2), sticky="w")
        ctk.CTkLabel(
            self,
            text="Play a short sample of each voice, then type the person's name.\n"
                 "Leave a box empty to keep \"Speaker 1\", \"Speaker 2\" and so on.",
            font=self._f_body, text_color=C.MUTED, justify="left",
        ).grid(row=1, column=0, padx=20, pady=(0, 10), sticky="w")

        if count > 4:
            rows = ctk.CTkScrollableFrame(self, fg_color="transparent", height=4 * 62)
        else:
            rows = ctk.CTkFrame(self, fg_color="transparent")
        rows.grid(row=2, column=0, padx=12, sticky="ew")
        rows.grid_columnconfigure(0, weight=1)

        for i, (raw_label, default_name) in enumerate(self._speaker_map.items()):
            row = ctk.CTkFrame(
                rows, fg_color=C.SURFACE_ALT, corner_radius=R_MD,
                border_width=1, border_color=C.BORDER,
            )
            row.grid(row=i, column=0, padx=8, pady=4, sticky="ew")
            row.grid_columnconfigure(2, weight=1)

            ctk.CTkLabel(
                row, text=default_name, font=self._f_strong, text_color=C.AMBER_INK,
                fg_color=C.AMBER_SOFT, corner_radius=R_SM, width=92, height=28,
            ).grid(row=0, column=0, padx=(12, 8), pady=10)

            clip_path = self._clips_dict.get(raw_label)
            _button(
                row,
                "Play sample" if clip_path else "No sample",
                width=110, height=32,
                state="normal" if clip_path else "disabled",
                command=lambda p=clip_path: self._play_clip(p),
            ).grid(row=0, column=1, padx=(0, 8), pady=10)

            entry = ctk.CTkEntry(
                row, height=34, corner_radius=R_SM, font=self._f_body,
                placeholder_text="Type a name",
                fg_color=C.SURFACE, border_color=C.BORDER_STRONG, text_color=C.INK,
            )
            entry.grid(row=0, column=2, padx=(0, 12), pady=10, sticky="ew")
            entry.bind("<Return>", lambda _e: self._on_confirm())
            self._entries[raw_label] = entry

            share = self._shares.get(raw_label)
            if share is not None:
                ctk.CTkLabel(
                    row, text=f"{round(share * 100)}% of the talking",
                    font=self._f_small, text_color=C.MUTED, width=130, anchor="w",
                ).grid(row=0, column=3, padx=(0, 12), pady=10)

        actions = ctk.CTkFrame(self, fg_color="transparent")
        actions.grid(row=3, column=0, padx=20, pady=(12, 18), sticky="e")
        _button(actions, "Skip names", kind="subtle", width=110, command=self._on_skip).grid(
            row=0, column=0, padx=(0, 8)
        )
        _button(
            actions, "Save transcript with names", kind="amber", width=230,
            font=self._f_strong, command=self._on_confirm,
        ).grid(row=0, column=1)

    def _play_clip(self, clip_path: "Path | None") -> None:
        if clip_path is None:
            return
        try:
            if sys.platform == "win32":
                import winsound
                winsound.PlaySound(str(clip_path), winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as exc:
            logger.warning(f"Clip playback failed: {exc}")

    def _stop_playback(self) -> None:
        try:
            if sys.platform == "win32":
                import winsound
                winsound.PlaySound(None, 0)
        except Exception:
            pass

    def _on_confirm(self) -> None:
        for raw_label, entry in self._entries.items():
            name = entry.get().strip()
            if name:
                self._speaker_map[raw_label] = name
        self._finish()

    def _on_skip(self) -> None:
        self._finish()

    def _finish(self) -> None:
        self._stop_playback()
        self._on_done()


# ─── Privacy dialog ──────────────────────────────────────────────────────────


class PrivacyDialog(ctk.CTkToplevel):
    """Plain-language explanation of how EasyScribe handles data."""

    def __init__(self, parent: ctk.CTk) -> None:
        super().__init__(parent)
        family = _ui_family()
        f_title = ctk.CTkFont(family=family, size=18, weight="bold")
        f_head = ctk.CTkFont(family=family, size=13, weight="bold")
        f_body = ctk.CTkFont(family=family, size=13)

        self.title("Your privacy")
        _set_window_icon(self)
        self.configure(fg_color=C.BG)
        self.resizable(False, False)
        self.transient(parent)
        self.bind("<Escape>", lambda _e: self.destroy())
        self.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self, text="Your privacy", font=f_title, text_color=C.INK).grid(
            row=0, column=0, padx=24, pady=(22, 4), sticky="w"
        )
        ctk.CTkLabel(
            self,
            text="EasyScribe works fully offline. It never connects to the internet,\n"
                 "and it blocks any attempt to do so. Nothing is uploaded, shared\n"
                 "or tracked. There are no accounts, analytics or update checks.",
            font=f_body, text_color=C.GREEN_INK, fg_color=C.GREEN_SOFT,
            corner_radius=R_MD, justify="left", anchor="w",
        ).grid(row=1, column=0, padx=24, pady=(4, 14), ipadx=12, ipady=10, sticky="ew")

        ctk.CTkLabel(self, text="Where your data is saved", font=f_head, text_color=C.INK).grid(
            row=2, column=0, padx=24, pady=(0, 4), sticky="w"
        )
        items = [
            ("Transcripts", "Next to each recording, or in the folder you choose."),
            ("Recordings", f"{DEFAULT_RECORDINGS_DIR}, or the folder you choose."),
            ("Temporary audio", "Deleted after each file, and again when EasyScribe closes."),
            ("Technical logs", f"{LOGS_DIR}\nFile names and progress only, never the transcript "
                               f"text. Only the last {MAX_LOG_FILES} are kept."),
        ]
        box = _card(self)
        box.grid(row=3, column=0, padx=24, sticky="ew")
        box.grid_columnconfigure(1, weight=1)
        for i, (head, body) in enumerate(items):
            ctk.CTkLabel(box, text=head, font=f_head, text_color=C.INK, anchor="nw").grid(
                row=i, column=0, padx=(14, 12), pady=8, sticky="nw"
            )
            ctk.CTkLabel(
                box, text=body, font=f_body, text_color=C.MUTED, anchor="w",
                justify="left", wraplength=360,
            ).grid(row=i, column=1, padx=(0, 14), pady=8, sticky="w")

        ctk.CTkLabel(
            self,
            text="To remove everything, delete the transcripts you made and the\n"
                 "EasyScribe folder. Nothing is stored anywhere else.",
            font=f_body, text_color=C.MUTED, justify="left",
        ).grid(row=4, column=0, padx=24, pady=(12, 0), sticky="w")

        actions = ctk.CTkFrame(self, fg_color="transparent")
        actions.grid(row=5, column=0, padx=24, pady=(16, 22), sticky="e")
        _button(actions, "Open logs folder", width=150,
                command=lambda: _open_path(LOGS_DIR)).grid(row=0, column=0, padx=(0, 8))
        _button(actions, "Close", kind="teal", width=100, command=self.destroy).grid(row=0, column=1)

        self.update_idletasks()
        px = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        py = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{max(0, px)}+{max(0, py)}")
        self.after(50, self.focus_set)


# ─── Main window ─────────────────────────────────────────────────────────────


def _screen_change(method: Callable) -> Callable:
    """Run a method that changes the screen under the fade (transitions.py)."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        veil = getattr(self, "_veil", None)
        if veil is None:
            method(self, *args, **kwargs)
        else:
            veil.run(lambda: method(self, *args, **kwargs))
    return wrapper


class TranscriberApp(_AppBase):  # type: ignore
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()

        self.title(f"{APP_NAME} {APP_VERSION}")
        _set_window_icon(self)
        # Fit small laptop screens (1366x768) and use more room on larger ones.
        # CTk scales geometry by the Windows display scale, so convert the
        # screen height to unscaled units first.
        try:
            scale = float(self._get_window_scaling())
        except Exception:
            scale = 1.0
        available = int(self.winfo_screenheight() / max(scale, 0.5)) - 90
        height = max(600, min(820, available))
        self.geometry(f"900x{height}")
        self.minsize(760, 600)
        # Short screens show fewer file rows before the list scrolls.
        self._visible_rows = 3 if available >= 760 else 2
        self.resizable(True, True)
        self.configure(fg_color=C.BG)
        # Before any widget is made: paint backgrounds at once on Windows,
        # so the window does not show black blocks when it comes to the front.
        win_paint.install(self, C.BG)

        family = _ui_family()
        self._f_app = ctk.CTkFont(family=family, size=22, weight="bold")
        self._f_title = ctk.CTkFont(family=family, size=15, weight="bold")
        self._f_status = ctk.CTkFont(family=family, size=16, weight="bold")
        self._f_body = ctk.CTkFont(family=family, size=13)
        self._f_strong = ctk.CTkFont(family=family, size=13, weight="bold")
        self._f_small = ctk.CTkFont(family=family, size=12)
        self._f_primary = ctk.CTkFont(family=family, size=14, weight="bold")
        self._f_mono = ctk.CTkFont(family=_mono_family(), size=11)
        self._f_timer = ctk.CTkFont(family=_mono_family(), size=22, weight="bold")

        # ── State ────────────────────────────────────────────────────────────
        self._selected_files: list[Path] = []
        self._file_states: dict[Path, str] = {}
        self._output_folder: Path | None = None
        # Record tab: the folder the user chose (remembered), or None for the default.
        self._rec_folder: Path | None = settings.load_recordings_dir()
        self._cancel_event = threading.Event()
        self._stop_recording_event = threading.Event()
        self._engine = TranscriptionEngine()
        self._mic_recorder = MicRecorder()
        self._live_transcriber = LiveTranscriber()
        self._last_output_folder: Path | None = None
        self._last_transcript: Path | None = None
        # Audio of the last recording, for "Make best quality transcript"
        self._last_recording: Path | None = None
        # The recording that the running (or last) file batch transcribes
        self._batch_recording: Path | None = None
        # Timed lines of the last recording's transcript, for "Find speakers".
        self._last_segments: list[tuple[float, float, str]] | None = None
        self._naming_panel: SpeakerNamingPanel | None = None
        # Output path per input file, when it must not be "<stem>.txt"
        self._output_overrides: dict[Path, Path] = {}
        self._ui_state = "idle"
        self._mode = _MODE_FILES
        self._details_open = False
        self._live_has_text = False
        self._rec_started_at: float | None = None

        # ── tkdnd ─────────────────────────────────────────────────────────────
        self._dnd_enabled: bool = False
        if _DND_AVAILABLE:
            try:
                TkinterDnD._require(self)  # type: ignore
                self._dnd_enabled = True
            except Exception as exc:
                logger.warning(f"tkdnd extension failed, drag and drop disabled: {exc}")

        # ── Diarization availability ──────────────────────────────────────────
        self._diarization_available: bool = DiarizationEngine().is_available()

        # ── Mic selector state ────────────────────────────────────────────────
        self._mic_options: list[str] = []
        self._mic_index_map: dict[str, int | None] = {}

        self._build_ui()
        self._render_files()
        self._set_ui_state("idle")
        self._update_status("Ready")
        # Screen changes from here on fade out and in (Windows).
        self._veil = transitions.create(self, self._mode_switch, C.BG)

        logger.info("GUI initialised")

    # ─────────────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)

        self._build_header()

        self._mode_switch = ctk.CTkSegmentedButton(
            self,
            values=[_MODE_FILES, _MODE_RECORD],
            command=self._on_mode_changed,
            font=self._f_strong,
            height=38,
            corner_radius=R_MD,
            fg_color=C.SURFACE_ALT,
            selected_color=C.TEAL,
            selected_hover_color=C.TEAL_HOVER,
            unselected_color=C.SURFACE_ALT,
            unselected_hover_color=C.SECONDARY_HOVER,
            text_color=C.INK,
            text_color_disabled=C.FAINT,
        )
        self._mode_switch.set(_MODE_FILES)
        self._mode_switch.grid(row=1, column=0, padx=24, pady=(0, 12), sticky="w")

        self._files_view = ctk.CTkFrame(self, fg_color="transparent")
        self._files_view.grid_columnconfigure(0, weight=1)
        self._build_files_view(self._files_view)

        self._record_view = ctk.CTkFrame(self, fg_color="transparent")
        self._record_view.grid_columnconfigure(0, weight=1)
        self._record_view.grid_rowconfigure(1, weight=1)
        self._build_record_view(self._record_view)

        self._build_activity_panel()
        self._build_details()

        self._show_mode(_MODE_FILES)

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, padx=24, pady=(16, 12), sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(header, text=APP_NAME, font=self._f_app, text_color=C.INK).grid(
            row=0, column=0, sticky="w"
        )
        _logo(header, size=40).grid(row=0, column=1, sticky="e")

    # ── Transcribe files view ────────────────────────────────────────────────

    def _step_header(self, parent, number: str, title: str, colour: str, soft: str) -> ctk.CTkFrame:  # type: ignore[no-untyped-def]
        row = ctk.CTkFrame(parent, fg_color="transparent")
        ctk.CTkLabel(
            row, text=number, width=26, height=26, corner_radius=13,
            fg_color=soft, text_color=colour, font=self._f_strong,
        ).grid(row=0, column=0, padx=(0, 10))
        ctk.CTkLabel(row, text=title, font=self._f_title, text_color=C.INK).grid(
            row=0, column=1, sticky="w"
        )
        row.grid_columnconfigure(2, weight=1)
        return row

    def _build_files_view(self, view: ctk.CTkFrame) -> None:
        # ── Step 1: add recordings ────────────────────────────────────────────
        add_card = _card(view)
        add_card.grid(row=0, column=0, padx=24, pady=(0, 12), sticky="ew")
        add_card.grid_columnconfigure(0, weight=1)

        head = self._step_header(add_card, "1", "Add recordings", C.TEAL_INK, C.TEAL_SOFT)
        head.grid(row=0, column=0, padx=18, pady=(16, 10), sticky="ew")
        self._files_count_label = ctk.CTkLabel(
            head, text="", font=self._f_body, text_color=C.MUTED
        )
        self._files_count_label.grid(row=0, column=3, padx=(0, 4), sticky="e")
        self._clear_files_btn = _button(
            head, "Clear all", kind="link", width=80, height=28, command=self._on_clear_files
        )
        self._clear_files_btn.grid(row=0, column=4, sticky="e")

        self._drop_zone = ctk.CTkFrame(
            add_card, fg_color=C.SURFACE_ALT, corner_radius=R_MD,
            border_width=2, border_color=C.BORDER,
        )
        self._drop_zone.grid(row=1, column=0, padx=18, pady=(0, 10), sticky="ew")
        self._drop_zone.grid_columnconfigure(0, weight=1)

        self._drop_title = ctk.CTkLabel(
            self._drop_zone, text="", font=self._f_strong, text_color=C.INK, anchor="w"
        )
        self._drop_title.grid(row=0, column=0, padx=16, pady=(14, 0), sticky="w")
        self._drop_sub = ctk.CTkLabel(
            self._drop_zone, text="", font=self._f_small, text_color=C.MUTED, anchor="w"
        )
        self._drop_sub.grid(row=1, column=0, padx=16, pady=(0, 14), sticky="w")
        self._select_files_btn = _button(
            self._drop_zone, "Choose files", kind="teal", width=130,
            font=self._f_strong, command=self._on_select_files,
        )
        self._select_files_btn.grid(row=0, column=1, rowspan=2, padx=16, pady=14)

        if self._dnd_enabled and DND_FILES is not None:
            for widget in (self._drop_zone, self._drop_title, self._drop_sub, add_card):
                try:
                    widget.drop_target_register(DND_FILES)
                    widget.dnd_bind("<<Drop>>", self._on_drop)
                    widget.dnd_bind("<<DropEnter>>", self._on_drop_enter)
                    widget.dnd_bind("<<DropLeave>>", self._on_drop_leave)
                except Exception as exc:
                    logger.warning(f"Could not register drop target: {exc}")

        self._drop_note = ctk.CTkLabel(
            add_card, text="", font=self._f_small, text_color=C.CRIMSON,
            anchor="w", justify="left", wraplength=760,
        )
        self._drop_note.grid(row=2, column=0, padx=18, pady=(0, 8), sticky="ew")
        self._drop_note.grid_remove()

        self._file_list = ctk.CTkScrollableFrame(
            add_card, fg_color="transparent", height=40, corner_radius=0,
            scrollbar_button_color=C.BORDER_STRONG,
            scrollbar_button_hover_color=C.FAINT,
        )
        self._file_list.grid(row=3, column=0, padx=10, pady=(0, 12), sticky="ew")
        self._file_list.grid_columnconfigure(0, weight=1)
        # CTkScrollbar defaults to 200 px high, which stops the list from
        # shrinking to fit one or two files.
        try:
            self._file_list._scrollbar.configure(height=0)
        except Exception:
            pass

        # ── Step 2: options ───────────────────────────────────────────────────
        opt_card = _card(view)
        opt_card.grid(row=1, column=0, padx=24, pady=(0, 12), sticky="ew")
        opt_card.grid_columnconfigure(0, weight=1)

        opt_head = self._step_header(opt_card, "2", "Choose options", C.TEAL_INK, C.TEAL_SOFT)
        opt_head.grid(row=0, column=0, padx=18, pady=(16, 10), sticky="ew")
        self._change_options_btn = _button(
            opt_head, "Change", kind="link", width=80, height=28,
            command=self._on_change_options,
        )
        self._change_options_btn.grid(row=0, column=4, sticky="e")
        self._change_options_btn.grid_remove()

        # While a batch runs (or its result shows), the options fold into one
        # line so the progress and the result fit on small screens.
        self._options_summary = ctk.CTkLabel(
            opt_card, text="", font=self._f_body, text_color=C.MUTED,
            anchor="w", justify="left", wraplength=760,
        )
        self._options_summary.grid(row=1, column=0, padx=18, pady=(0, 14), sticky="ew")
        self._options_summary.grid_remove()

        self._opt_body = ctk.CTkFrame(opt_card, fg_color="transparent")
        # Inset by the border width so the frame does not paint over the
        # card's rounded border.
        self._opt_body.grid(row=2, column=0, padx=2, pady=(0, 2), sticky="ew")
        opt_card = self._opt_body
        opt_card.grid_columnconfigure((0, 1), weight=1, uniform="opt")

        ctk.CTkLabel(
            opt_card, text="Save transcripts to", font=self._f_strong, text_color=C.INK
        ).grid(row=1, column=0, columnspan=2, padx=18, sticky="w")
        out_row = ctk.CTkFrame(opt_card, fg_color="transparent")
        out_row.grid(row=2, column=0, columnspan=2, padx=18, pady=(2, 12), sticky="ew")
        out_row.grid_columnconfigure(0, weight=1)
        self._output_label = ctk.CTkLabel(
            out_row, text="", font=self._f_body, text_color=C.MUTED, anchor="w"
        )
        self._output_label.grid(row=0, column=0, sticky="ew")
        self._reset_output_btn = _button(
            out_row, "Use default", kind="link", width=100, height=32,
            command=self._on_reset_output_folder,
        )
        self._reset_output_btn.grid(row=0, column=1, padx=(8, 0))
        self._select_output_btn = _button(
            out_row, "Change folder", width=130, height=32, command=self._on_select_output_folder
        )
        self._select_output_btn.grid(row=0, column=2, padx=(8, 0))

        self._timestamps_var = ctk.BooleanVar(value=False)
        self._timestamps_cb = self._option_tile(
            opt_card, column=0,
            title="Add timestamps",
            body="Shows the time, like [00:01:23], at the start of each paragraph.",
            variable=self._timestamps_var,
            colour=C.SKY, colour_hover="#255F9E", soft=C.SKY_SOFT, ink=C.SKY_INK,
            enabled=True,
        )

        self._diarize_var = ctk.BooleanVar(value=False)
        self._diarize_cb = self._option_tile(
            opt_card, column=1,
            title="Name the speakers",
            body=(
                "Finds who is talking. You hear a short sample of each voice and "
                "type their name. Takes longer."
                if self._diarization_available
                else "Not available: the speaker files are missing from this copy of EasyScribe."
            ),
            variable=self._diarize_var,
            colour=C.AMBER, colour_hover="#946000", soft=C.AMBER_SOFT, ink=C.AMBER_INK,
            enabled=self._diarization_available,
        )

        # ── Start ─────────────────────────────────────────────────────────────
        action_row = ctk.CTkFrame(view, fg_color="transparent")
        action_row.grid(row=2, column=0, padx=24, pady=(0, 12), sticky="ew")
        action_row.grid_columnconfigure(2, weight=1)
        self._action_row = action_row

        self._transcribe_btn = _button(
            action_row, "Transcribe", kind="teal", width=200, height=44,
            font=self._f_primary, command=self._on_transcribe,
        )
        self._transcribe_btn.grid(row=0, column=0)
        self._start_hint = ctk.CTkLabel(
            action_row, text="", font=self._f_small, text_color=C.MUTED, anchor="w"
        )
        self._start_hint.grid(row=0, column=2, padx=14, sticky="w")

        self._refresh_output_label()

    def _option_tile(  # type: ignore[no-untyped-def]
        self, parent, column: int, title: str, body: str, variable,
        colour: str, colour_hover: str, soft: str, ink: str, enabled: bool,
    ) -> ctk.CTkCheckBox:
        tile = ctk.CTkFrame(
            parent, fg_color=soft if enabled else C.SURFACE_ALT, corner_radius=R_MD
        )
        pad_l = 18 if column == 0 else 6
        pad_r = 6 if column == 0 else 18
        tile.grid(row=3, column=column, padx=(pad_l, pad_r), pady=(0, 16), sticky="nsew")
        tile.grid_columnconfigure(0, weight=1)

        cb = ctk.CTkCheckBox(
            tile,
            text=title,
            variable=variable,
            font=self._f_strong,
            text_color=ink if enabled else C.MUTED,
            text_color_disabled=C.MUTED,
            fg_color=colour,
            hover_color=colour_hover,
            border_color=colour if enabled else C.BORDER_STRONG,
            checkmark_color=C.SURFACE,
            corner_radius=R_SM,
            checkbox_width=22,
            checkbox_height=22,
            state="normal" if enabled else "disabled",
        )
        cb.grid(row=0, column=0, padx=14, pady=(12, 2), sticky="w")
        ctk.CTkLabel(
            tile, text=body, font=self._f_small, text_color=C.MUTED,
            anchor="w", justify="left", wraplength=330,
        ).grid(row=1, column=0, padx=(46, 14), pady=(0, 12), sticky="w")
        return cb

    # ── Record view ──────────────────────────────────────────────────────────

    def _build_record_view(self, view: ctk.CTkFrame) -> None:
        rec_card = _card(view)
        rec_card.grid(row=0, column=0, padx=24, pady=(0, 12), sticky="ew")
        rec_card.grid_columnconfigure((0, 1), weight=1, uniform="rec")

        # ── Step 1: microphone ────────────────────────────────────────────────
        step1 = self._step_header(rec_card, "1", "Choose a microphone", C.CORAL_INK, C.CORAL_SOFT)
        step1.grid(row=0, column=0, columnspan=2, padx=18, pady=(16, 8), sticky="ew")
        self._mic_options, self._mic_index_map = self._build_mic_options()
        self._mic_var = ctk.StringVar(value=self._mic_options[0])
        self._mic_menu = ctk.CTkOptionMenu(
            rec_card,
            variable=self._mic_var,
            values=self._mic_options,
            width=340,
            height=36,
            corner_radius=R_MD,
            font=self._f_body,
            dropdown_font=self._f_body,
            fg_color=C.SECONDARY,
            button_color=C.SECONDARY_HOVER,
            button_hover_color=C.BORDER_STRONG,
            text_color=C.INK,
            text_color_disabled=C.FAINT,
            dropdown_fg_color=C.SURFACE,
            dropdown_hover_color=C.CORAL_SOFT,
            dropdown_text_color=C.INK,
            dynamic_resizing=False,
        )
        self._mic_menu.grid(row=1, column=0, columnspan=2, padx=18, pady=(0, 14), sticky="w")

        # ── Step 2: what to do with the recording ─────────────────────────────
        step2 = self._step_header(rec_card, "2", "What do you want?", C.CORAL_INK, C.CORAL_SOFT)
        step2.grid(row=2, column=0, columnspan=2, padx=18, pady=(0, 8), sticky="ew")
        self._rec_mode_var = ctk.StringVar(value=_REC_BEST)
        self._rec_tiles: dict[str, tuple[ctk.CTkFrame, ctk.CTkRadioButton]] = {}
        self._rec_choice_tile(
            rec_card, column=0, value=_REC_BEST,
            title="Best quality transcript after recording",
            body="Saves the recording.\nAdds times. Most accurate. Find speakers after.",
        )
        self._rec_choice_tile(
            rec_card, column=1, value=_REC_LIVE,
            title="Show words as I speak",
            body="Saves the recording.\nQuick text while you talk. Less accurate.",
        )

        # Steps 1 and 2 fold into one line during and after a live recording,
        # so the live transcript keeps its room on short screens.
        self._rec_setup = [step1, self._mic_menu, step2] + [t for t, _r in self._rec_tiles.values()]
        self._rec_summary = ctk.CTkFrame(rec_card, fg_color="transparent")
        self._rec_summary.grid(row=0, column=0, columnspan=2, padx=18, pady=(16, 10), sticky="ew")
        self._rec_summary.grid_columnconfigure(0, weight=1)
        self._rec_summary_label = ctk.CTkLabel(
            self._rec_summary, text="", font=self._f_small, text_color=C.MUTED, anchor="w"
        )
        self._rec_summary_label.grid(row=0, column=0, sticky="w")
        self._rec_change_btn = _button(
            self._rec_summary, "Change", kind="link", width=70, height=28,
            command=self._on_change_options,
        )
        self._rec_change_btn.grid(row=0, column=1)
        self._rec_summary.grid_remove()

        # Where the recordings go: the same design as "Save transcripts to".
        rec_where = ctk.CTkFrame(rec_card, fg_color="transparent")
        rec_where.grid(row=4, column=0, columnspan=2, padx=18, pady=(4, 12), sticky="ew")
        rec_where.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            rec_where, text="Save recordings to", font=self._f_strong, text_color=C.INK
        ).grid(row=0, column=0, sticky="w")
        rec_out_row = ctk.CTkFrame(rec_where, fg_color="transparent")
        rec_out_row.grid(row=1, column=0, pady=(2, 0), sticky="ew")
        rec_out_row.grid_columnconfigure(0, weight=1)
        self._rec_folder_label = ctk.CTkLabel(
            rec_out_row, text="", font=self._f_body, text_color=C.MUTED, anchor="w"
        )
        self._rec_folder_label.grid(row=0, column=0, sticky="ew")
        self._reset_rec_folder_btn = _button(
            rec_out_row, "Use default", kind="link", width=100, height=32,
            command=self._on_reset_rec_folder,
        )
        self._reset_rec_folder_btn.grid(row=0, column=1, padx=(8, 0))
        self._select_rec_folder_btn = _button(
            rec_out_row, "Change folder", width=130, height=32, command=self._on_select_rec_folder
        )
        self._select_rec_folder_btn.grid(row=0, column=2, padx=(8, 0))
        self._rec_setup.append(rec_where)
        self._refresh_rec_folder_label()

        # ── Step 3: record ────────────────────────────────────────────────────
        self._step_header(rec_card, "3", "Record", C.CORAL_INK, C.CORAL_SOFT).grid(
            row=6, column=0, columnspan=2, padx=18, pady=(0, 8), sticky="ew"
        )
        rec_row = ctk.CTkFrame(rec_card, fg_color="transparent")
        rec_row.grid(row=7, column=0, columnspan=2, padx=18, pady=(0, 16), sticky="ew")
        rec_row.grid_columnconfigure(3, weight=1)
        self._record_btn = _button(
            rec_row, "Start recording", kind="coral", width=200, height=44,
            font=self._f_primary, command=self._on_record,
        )
        self._record_btn.grid(row=0, column=0, padx=(0, 14))
        self._timer_label = ctk.CTkLabel(
            rec_row, text="00:00", font=self._f_timer, text_color=C.FAINT
        )
        self._timer_label.grid(row=0, column=1, padx=(0, 14))
        # Microphone level: shows that the microphone hears sound, which
        # matters most when no words appear during the recording.
        self._level_bar = ctk.CTkProgressBar(
            rec_row, width=120, height=8, corner_radius=4,
            # The track colour while idle: an empty bar would show a coral dot.
            fg_color=C.BORDER, progress_color=C.BORDER, mode="determinate",
        )
        self._level_bar.set(0)
        self._level_bar.grid(row=0, column=2)


        # ── Live transcript (only for "Show words as I speak") ────────────────
        live_card = _card(view)
        self._live_card = live_card
        live_card.grid(row=1, column=0, padx=24, pady=(0, 12), sticky="nsew")
        live_card.grid_columnconfigure(0, weight=1)
        live_card.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(live_card, text="Live transcript", font=self._f_title, text_color=C.INK).grid(
            row=0, column=0, padx=18, pady=(14, 6), sticky="w"
        )
        self._live_box = ctk.CTkTextbox(
            live_card, font=self._f_body, wrap="word", height=120,
            fg_color=C.SURFACE_ALT, text_color=C.INK, corner_radius=R_MD,
            border_width=0, state="disabled",
        )
        self._live_box.grid(row=1, column=0, padx=18, pady=(0, 16), sticky="nsew")
        self._reset_live_box()
        self._on_rec_mode_changed()

    def _rec_choice_tile(self, parent, column: int, value: str, title: str, body: str) -> None:  # type: ignore[no-untyped-def]
        tile = ctk.CTkFrame(
            parent, fg_color=C.SURFACE_ALT, corner_radius=R_MD,
            border_width=2, border_color=C.BORDER,
        )
        pad_l = 18 if column == 0 else 6
        pad_r = 6 if column == 0 else 18
        tile.grid(row=3, column=column, padx=(pad_l, pad_r), pady=(0, 10), sticky="nsew")
        tile.grid_columnconfigure(0, weight=1)
        radio = ctk.CTkRadioButton(
            tile, text=title, value=value, variable=self._rec_mode_var,
            command=self._on_rec_mode_changed,
            font=self._f_strong, text_color=C.INK, text_color_disabled=C.MUTED,
            fg_color=C.CORAL, hover_color=C.CORAL_HOVER, border_color=C.BORDER_STRONG,
            radiobutton_width=20, radiobutton_height=20,
        )
        radio.grid(row=0, column=0, padx=14, pady=(12, 2), sticky="w")
        note = ctk.CTkLabel(
            tile, text=body, font=self._f_small, text_color=C.MUTED,
            anchor="w", justify="left", wraplength=320,
        )
        note.grid(row=1, column=0, padx=(44, 14), pady=(0, 12), sticky="w")
        # The whole tile selects the choice, not only the small circle.
        for widget in (tile, note):
            widget.bind("<Button-1>", lambda _e, v=value: self._select_rec_mode(v))
        self._rec_tiles[value] = (tile, radio)

    def _select_rec_mode(self, value: str) -> None:
        if self._ui_state != "idle":
            return
        self._rec_mode_var.set(value)
        self._on_rec_mode_changed()

    def _on_rec_mode_changed(self) -> None:
        mode = self._rec_mode_var.get()
        for value, (tile, _radio) in self._rec_tiles.items():
            selected = value == mode
            tile.configure(
                fg_color=C.CORAL_SOFT if selected else C.SURFACE_ALT,
                border_color=C.CORAL if selected else C.BORDER,
            )
        if mode == _REC_BEST:
            self._live_card.grid_remove()
        else:
            self._live_card.grid()

    # ── Activity panel (status, progress, result) ────────────────────────────

    def _build_activity_panel(self) -> None:
        panel = _card(self)
        self._activity = panel
        panel.grid_columnconfigure(0, weight=1)

        top = ctk.CTkFrame(panel, fg_color="transparent")
        top.grid(row=0, column=0, padx=18, pady=(14, 0), sticky="ew")
        top.grid_columnconfigure(0, weight=1)
        self._status_label = ctk.CTkLabel(
            top, text="Ready", font=self._f_status, text_color=C.MUTED, anchor="w"
        )
        self._status_label.grid(row=0, column=0, sticky="w")
        self._percent_label = ctk.CTkLabel(
            top, text="", font=self._f_strong, text_color=C.MUTED, anchor="e"
        )
        self._percent_label.grid(row=0, column=1, sticky="e")
        # Stop sits next to the progress it stops (shown only while files run).
        self._cancel_btn = _button(
            top, "Stop", width=80, height=32, command=self._on_cancel
        )
        self._cancel_btn.grid(row=0, column=2, padx=(14, 0), sticky="e")

        self._batch_label = ctk.CTkLabel(
            panel, text="", font=self._f_body, text_color=C.MUTED, anchor="w", justify="left",
            wraplength=780,
        )
        self._batch_label.grid(row=1, column=0, padx=18, sticky="ew")

        self._progress_bar = ctk.CTkProgressBar(
            panel, height=8, corner_radius=4, fg_color=C.BORDER,
            progress_color=C.TEAL, mode="determinate", indeterminate_speed=0.6,
        )
        self._progress_bar.set(0)
        self._progress_bar.grid(row=2, column=0, padx=18, pady=(8, 14), sticky="ew")
        self._progress_running = False

        self._message_label = ctk.CTkLabel(
            panel, text="", font=self._f_body, anchor="w", justify="left",
            corner_radius=R_SM, wraplength=760,
        )
        self._message_label.grid(row=3, column=0, padx=18, pady=(0, 12), sticky="ew")
        self._message_label.grid_remove()

        self._result_row = ctk.CTkFrame(panel, fg_color="transparent")
        self._result_row.grid(row=4, column=0, padx=18, pady=(0, 14), sticky="w")
        self._open_transcript_btn = _button(
            self._result_row, "Open transcript", kind="teal", width=150,
            font=self._f_strong, command=self._on_open_transcript,
        )
        self._open_transcript_btn.grid(row=0, column=0, padx=(0, 8))
        self._open_output_btn = _button(
            self._result_row, "Open folder", width=120, command=self._on_open_output_folder
        )
        self._open_output_btn.grid(row=0, column=1, padx=(0, 8))
        self._retry_btn = _button(
            self._result_row, "Try again", width=110, command=self._on_change_options
        )
        self._retry_btn.grid(row=0, column=2)
        self._result_row.grid_remove()

        # After a result: an optional new name for the files just made.
        self._rename_row = ctk.CTkFrame(panel, fg_color=C.SURFACE_ALT, corner_radius=R_MD)
        self._rename_row.grid(row=5, column=0, padx=18, pady=(0, 16), sticky="ew")
        self._rename_row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            self._rename_row, text="Name", font=self._f_strong, text_color=C.INK
        ).grid(row=0, column=0, padx=(14, 10), pady=(12, 2), sticky="w")
        self._rename_entry = ctk.CTkEntry(
            self._rename_row, height=36, corner_radius=R_SM, font=self._f_body,
            placeholder_text="Type a new name, for example Team meeting 21 Oct",
            fg_color=C.SURFACE, border_color=C.BORDER_STRONG, text_color=C.INK,
        )
        self._rename_entry.grid(row=0, column=1, pady=(12, 2), sticky="ew")
        self._rename_entry.bind("<Return>", lambda _e: self._on_rename())
        _button(
            self._rename_row, "Rename", kind="teal", width=100,
            font=self._f_strong, command=self._on_rename,
        ).grid(row=0, column=2, padx=14, pady=(12, 2))
        self._rename_note = ctk.CTkLabel(
            self._rename_row, text="", font=self._f_small, text_color=C.MUTED,
            anchor="w", justify="left", wraplength=620,
        )
        self._rename_note.grid(row=1, column=1, columnspan=2, pady=(0, 10), sticky="w")
        self._rename_row.grid_remove()

        # After a recording: the same choices as before it, for the saved file.
        self._best_row = ctk.CTkFrame(panel, fg_color=C.SURFACE_ALT, corner_radius=R_MD)
        self._best_row.grid(row=6, column=0, padx=18, pady=(0, 16), sticky="ew")
        self._best_row.grid_columnconfigure(1, weight=1)
        self._full_transcript_btn = _button(
            self._best_row, "Make best quality transcript", kind="teal", width=240,
            font=self._f_strong, command=self._on_full_transcript,
        )
        self._full_transcript_btn.grid(row=0, column=0, padx=(14, 12), pady=12, sticky="w")
        self._best_note = ctk.CTkLabel(
            self._best_row, text="", font=self._f_small, text_color=C.MUTED,
            anchor="w", justify="left", wraplength=420,
        )
        self._best_note.grid(row=0, column=1, padx=(0, 14), pady=12, sticky="w")
        self._best_row.grid_remove()

        # After a recording's transcript: find the speakers, with a time estimate.
        self._speakers_card = ctk.CTkFrame(
            panel, fg_color=C.SURFACE, corner_radius=R_MD, border_width=2, border_color=C.AMBER
        )
        self._speakers_card.grid(row=7, column=0, padx=18, pady=(0, 16), sticky="ew")
        self._speakers_card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            self._speakers_card, text="Who said what?", font=self._f_title, text_color=C.INK
        ).grid(row=0, column=0, columnspan=2, padx=18, pady=(16, 2), sticky="w")
        ctk.CTkLabel(
            self._speakers_card,
            text="Find the different voices in this recording, then type each person's name. "
                 "The names are added to a copy of the transcript.",
            font=self._f_body, text_color=C.MUTED, anchor="w", justify="left", wraplength=640,
        ).grid(row=1, column=0, columnspan=2, padx=18, pady=(0, 10), sticky="w")
        estimate_row = ctk.CTkFrame(self._speakers_card, fg_color="transparent")
        estimate_row.grid(row=2, column=0, padx=18, sticky="w")
        self._speakers_estimate = ctk.CTkLabel(
            estimate_row, text="", font=self._f_strong, text_color=C.AMBER_INK
        )
        self._speakers_estimate.grid(row=0, column=0, padx=(0, 8))
        ctk.CTkLabel(
            estimate_row, text="rough estimate", font=self._f_small, text_color=C.AMBER_INK,
            fg_color=C.AMBER_SOFT, corner_radius=R_SM, height=22,
        ).grid(row=0, column=1, ipadx=6)
        self._speakers_basis = ctk.CTkLabel(
            self._speakers_card, text="", font=self._f_small, text_color=C.MUTED,
            anchor="w", justify="left", wraplength=520,
        )
        self._speakers_basis.grid(row=3, column=0, padx=18, pady=(2, 16), sticky="w")
        self._find_speakers_btn = _button(
            self._speakers_card, "Find speakers", kind="amber", width=170, height=40,
            font=self._f_strong, command=self._on_find_speakers,
        )
        self._find_speakers_btn.grid(row=2, column=1, rowspan=2, padx=18, pady=(0, 16), sticky="e")
        self._speakers_card.grid_remove()

        # Speaker names are typed here (SpeakerNamingPanel), not in a popup.
        self._naming_host = ctk.CTkFrame(panel, fg_color="transparent")
        self._naming_host.grid(row=8, column=0, padx=18, pady=(0, 16), sticky="ew")
        self._naming_host.grid_columnconfigure(0, weight=1)
        self._naming_host.grid_remove()

    # ── Details (technical log) ──────────────────────────────────────────────

    def _build_details(self) -> None:
        self._details_btn = _button(
            self, "Show details", kind="subtle", width=110, height=30,
            font=self._f_small, command=self._toggle_details,
        )
        self._details_btn.grid(row=4, column=0, padx=24, pady=(0, 10), sticky="w")
        self._privacy_btn = _button(
            self, "Privacy", kind="subtle", width=80, height=30,
            font=self._f_small, command=lambda: PrivacyDialog(self),
        )
        self._privacy_btn.grid(row=4, column=0, padx=24, pady=(0, 10), sticky="e")

        self._log_card = _card(self)
        self._log_card.grid_columnconfigure(0, weight=1)
        self._log_card.grid_rowconfigure(0, weight=1)
        self._log_box = ctk.CTkTextbox(
            self._log_card,
            state="disabled",
            font=self._f_mono,
            wrap="word",
            height=140,
            fg_color=C.SURFACE,
            text_color=C.MUTED,
            border_width=0,
        )
        self._log_box.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

    @_screen_change
    def _toggle_details(self) -> None:
        self._details_open = not self._details_open
        if self._details_open:
            self._log_card.grid(row=5, column=0, padx=24, pady=(0, 16), sticky="nsew")
            self.grid_rowconfigure(5, weight=1)
            self._details_btn.configure(text="Hide details")
        else:
            self._log_card.grid_remove()
            self.grid_rowconfigure(5, weight=0)
            self._details_btn.configure(text="Show details")

    # ── Modes ────────────────────────────────────────────────────────────────

    def _on_mode_changed(self, value: str) -> None:
        if self._ui_state != "idle":
            self._mode_switch.set(self._mode)
            return
        self._show_mode(value)

    @_screen_change
    def _show_mode(self, mode: str) -> None:
        self._mode = mode
        if self._mode_switch.get() != mode:
            self._mode_switch.set(mode)
        if mode == _MODE_FILES:
            self._record_view.grid_remove()
            self._files_view.grid(row=2, column=0, sticky="nsew")
            self.grid_rowconfigure(2, weight=0)
            self._mode_switch.configure(selected_color=C.TEAL, selected_hover_color=C.TEAL_HOVER)
        else:
            self._files_view.grid_remove()
            self._record_view.grid(row=2, column=0, sticky="nsew")
            self.grid_rowconfigure(2, weight=1)
            self._mode_switch.configure(selected_color=C.CORAL, selected_hover_color=C.CORAL_HOVER)
        # Segment text sits on both the coloured and the pale segment, so
        # only the selected one gets white text.
        for name, btn in getattr(self._mode_switch, "_buttons_dict", {}).items():
            try:
                btn.configure(text_color=C.SURFACE if name == mode else C.INK)
            except Exception:
                pass
        self._hide_activity()

    # ─────────────────────────────────────────────────────────────────────────
    # Drag & drop
    # ─────────────────────────────────────────────────────────────────────────

    def _on_drop_enter(self, event: object) -> object:
        if self._ui_state == "idle":
            self._drop_zone.configure(border_color=C.TEAL, fg_color=C.TEAL_SOFT)
        return getattr(event, "action", None)

    def _on_drop_leave(self, event: object) -> object:
        self._drop_zone.configure(border_color=C.BORDER, fg_color=C.SURFACE_ALT)
        return getattr(event, "action", None)

    def _on_drop(self, event: object) -> object:
        self._on_drop_leave(event)
        if self._ui_state != "idle":
            return getattr(event, "action", None)

        raw = getattr(event, "data", "")
        try:
            paths_raw: list[str] = self.tk.splitlist(raw)  # type: ignore[attr-defined]
        except Exception:
            paths_raw = raw.split()

        paths = [Path(p) for p in paths_raw]
        valid = [p for p in paths if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
        skipped = [p for p in paths if p not in valid]
        if valid:
            self._add_files(valid)
        self._show_drop_note(skipped)
        return getattr(event, "action", None)

    def _show_drop_note(self, skipped: list[Path]) -> None:
        if not skipped:
            self._drop_note.grid_remove()
            return
        names = ", ".join(_shorten(p.name, 40) for p in skipped[:3])
        more = f" and {len(skipped) - 3} more" if len(skipped) > 3 else ""
        self._drop_note.configure(
            text=f"Skipped {names}{more}. EasyScribe reads audio and video files: "
                 f"{', '.join(sorted(e.lstrip('.').upper() for e in SUPPORTED_EXTENSIONS))}."
        )
        self._drop_note.grid()

    # ─────────────────────────────────────────────────────────────────────────
    # Button handlers
    # ─────────────────────────────────────────────────────────────────────────

    def _on_select_files(self) -> None:
        ext_list = " ".join(f"*{e}" for e in sorted(SUPPORTED_EXTENSIONS))
        paths = filedialog.askopenfilenames(
            title="Choose recordings to transcribe",
            filetypes=[("Audio and video files", ext_list), ("All files", "*.*")],
        )
        if paths:
            chosen = [Path(p) for p in paths]
            self._add_files(chosen)
            self._show_drop_note(
                [p for p in chosen if p.suffix.lower() not in SUPPORTED_EXTENSIONS]
            )

    def _on_clear_files(self) -> None:
        if self._ui_state == "idle":
            self._hide_activity()
        self._selected_files.clear()
        self._file_states.clear()
        self._drop_note.grid_remove()
        self._render_files()

    def _on_remove_file(self, path: Path) -> None:
        if self._ui_state != "idle":
            return
        if path in self._selected_files:
            self._selected_files.remove(path)
        self._file_states.pop(path, None)
        self._render_files()

    def _on_select_output_folder(self) -> None:
        folder = filedialog.askdirectory(title="Choose where to save transcripts")
        if folder:
            self._output_folder = Path(folder)
            self._refresh_output_label()

    def _on_reset_output_folder(self) -> None:
        self._output_folder = None
        self._refresh_output_label()

    def _on_select_rec_folder(self) -> None:
        folder = filedialog.askdirectory(
            title="Choose where to save recordings", initialdir=str(self._recordings_folder())
        )
        if folder:
            self._rec_folder = Path(folder)
            settings.save_recordings_dir(self._rec_folder)
            self._refresh_rec_folder_label()

    def _on_reset_rec_folder(self) -> None:
        self._rec_folder = None
        settings.save_recordings_dir(None)
        self._refresh_rec_folder_label()

    def _recordings_folder(self) -> Path:
        return self._rec_folder or DEFAULT_RECORDINGS_DIR

    def _on_transcribe(self) -> None:
        if not self._selected_files:
            self._start_hint.configure(text="Add at least one recording first.", text_color=C.CRIMSON)
            return

        check_dir = self._output_folder or self._selected_files[0].parent
        try:
            usage = shutil.disk_usage(check_dir)
            if usage.free < MIN_FREE_DISK_BYTES:
                if not messagebox.askyesno(
                    "Low disk space",
                    f"Only {usage.free // (1024 * 1024)} MB of disk space is free.\n"
                    "Transcription may fail. Continue anyway?",
                ):
                    return
        except Exception:
            pass

        options = {
            "timestamps": bool(self._timestamps_var.get()),
            "diarize": bool(self._diarize_var.get()),
        }
        self._batch_recording = None
        self._file_states = {p: "Waiting" for p in self._selected_files}
        self._clear_log()
        self._start_batch(list(self._selected_files), options)

    def _start_batch(self, files: list[Path], options: dict) -> None:
        self._cancel_event.clear()
        self._last_transcript = None
        self._set_ui_state("running")
        self._show_activity()
        self._set_progress(0)

        threading.Thread(
            target=self._transcription_worker, args=(files, options),
            daemon=True, name="TranscriptionWorker",
        ).start()

    def _on_record(self) -> None:
        live = self._rec_mode_var.get() == _REC_LIVE
        self._stop_recording_event.clear()
        self._last_transcript = None
        self._last_segments = None
        self._set_ui_state("recording")
        self._clear_log()
        self._reset_live_box(listening=True)
        self._show_activity()
        self._safe_append_log(
            "[Record] Starting live recording…" if live
            else "[Record] Starting recording (best quality transcript after the stop)…"
        )
        self._record_btn.configure(text="Stop recording", command=self._on_stop_recording)

        mic_label = self._mic_var.get()
        device_index = self._mic_index_map.get(mic_label)
        threading.Thread(
            target=self._recording_worker, args=(device_index, live),
            daemon=True, name="RecordingWorker",
        ).start()

    def _on_stop_recording(self) -> None:
        self._stop_recording_event.set()
        self._record_btn.configure(state="disabled", text="Saving…")

    def _on_cancel(self) -> None:
        self._cancel_event.set()
        self._update_status("Cancelling…")
        self._cancel_btn.configure(state="disabled")

    def _on_open_output_folder(self) -> None:
        folder = self._last_output_folder or self._output_folder or self._recordings_folder()
        self._open_folder(folder)

    def _on_open_transcript(self) -> None:
        if self._last_transcript and self._last_transcript.exists():
            _open_path(self._last_transcript)
        else:
            self._on_open_output_folder()

    def _open_folder(self, folder: Path | None) -> None:
        if folder and folder.exists():
            _open_path(folder)
        else:
            messagebox.showinfo("No folder yet", "There is no folder to open yet.")

    # ─────────────────────────────────────────────────────────────────────────
    # Transcription worker thread
    # ─────────────────────────────────────────────────────────────────────────

    def _transcription_worker(self, files: list[Path], options: dict) -> None:
        total = len(files)
        saved: list[Path] = []
        failed: list[str] = []
        fatal: str | None = None

        for idx, input_file in enumerate(files, start=1):
            if self._cancel_event.is_set():
                break

            self._safe_set_batch_label(
                f"File {idx} of {total}: {_shorten(input_file.name, 70)}"
                if total > 1 else _shorten(input_file.name, 80)
            )
            self._safe_set_file_state(input_file, "Working")
            self._safe_append_log(f"\n{'─' * 50}")
            self._safe_append_log(f"[File {idx}/{total}] {input_file.name}")

            temp_wav: Path | None = None
            try:
                self._safe_update_status("Extracting Audio")
                self._safe_set_progress(0.0)

                temp_wav = extract_audio(input_file, self._cancel_event, self._safe_append_log)

                if self._cancel_event.is_set():
                    raise CancelledError("Cancelled")

                output_path = self._resolve_output_path(input_file)
                self._last_output_folder = output_path.parent

                segments = self._engine.transcribe(
                    audio_path=temp_wav,
                    output_path=output_path,
                    add_timestamps=options["timestamps"],
                    cancel_event=self._cancel_event,
                    status_callback=self._safe_update_status,
                    progress_callback=self._safe_set_progress,
                    log_callback=self._safe_append_log,
                    diarize=options["diarize"],
                    speaker_name_callback=(
                        self._make_speaker_naming_callback() if options["diarize"] else None
                    ),
                )
                saved.append(output_path)
                if self._batch_recording is not None:
                    self._last_segments = segments
                self._safe_set_file_state(input_file, "Done")

            except CancelledError:
                self._safe_append_log("[Cancelled] Operation stopped by user")
                self._safe_set_file_state(input_file, "Stopped")
                break

            except (FFmpegNotFoundError, ModelNotFoundError) as exc:
                logger.exception("Fatal dependency missing")
                self._safe_append_log(f"[Fatal] {exc}")
                self._safe_set_file_state(input_file, "Failed")
                fatal = str(exc)
                failed.append(input_file.name)
                break

            except (FFmpegExtractionError, TranscriptionError) as exc:
                logger.error(f"File {input_file.name} failed: {exc}")
                self._safe_append_log(f"[Error] {input_file.name}: {exc}")
                self._safe_set_file_state(input_file, "Failed")
                failed.append(input_file.name)

            except Exception as exc:
                logger.exception(f"Unexpected error on {input_file.name}")
                self._safe_append_log(f"[Unexpected Error] {input_file.name}: {exc}")
                self._safe_set_file_state(input_file, "Failed")
                failed.append(input_file.name)

            finally:
                if temp_wav and temp_wav.exists():
                    try:
                        temp_wav.unlink()
                    except OSError:
                        pass

        cancelled = self._cancel_event.is_set()
        if cancelled:
            final_status = "Cancelled"
        elif not failed:
            final_status = "Done"
        elif len(failed) == total or fatal:
            final_status = "Failed"
        else:
            final_status = "Done"
            self._safe_append_log(
                f"\n[Summary] Completed with {len(failed)} error(s) out of {total} file(s)"
            )

        self._safe_update_status(final_status)
        self.after(
            0,
            lambda: self._finish_batch(final_status, total, saved, failed, fatal),
        )

    def _finish_batch(
        self, status: str, total: int, saved: list[Path], failed: list[str], fatal: str | None
    ) -> None:
        self._set_ui_state("idle")
        self._last_transcript = saved[0] if len(saved) == 1 else None
        if saved:
            self._last_output_folder = saved[-1].parent

        message, tone = "", "neutral"
        if status == "Cancelled":
            title, colour = "Stopped", C.MUTED
            detail = (
                f"{_plural(len(saved), 'transcript')} saved before you stopped."
                if saved else "No transcript was saved."
            )
        elif fatal:
            title, colour = "EasyScribe could not continue", C.CRIMSON
            detail = "A program file is missing. Please download EasyScribe again."
            message, tone = fatal, "error"
        elif not failed:
            title = "Transcript ready" if len(saved) == 1 else f"{len(saved)} transcripts ready"
            colour = C.GREEN_INK
            detail = f"Saved in {_shorten(str(self._last_output_folder or ''), 90)}"
        else:
            colour = C.CRIMSON if not saved else C.AMBER_INK
            title = (
                "Transcription failed" if not saved
                else f"{len(saved)} of {total} transcripts ready"
            )
            detail = (
                f"Saved in {_shorten(str(self._last_output_folder or ''), 90)}" if saved else ""
            )
            names = ", ".join(_shorten(n, 40) for n in failed[:3])
            more = f" and {len(failed) - 3} more" if len(failed) > 3 else ""
            message = (
                f"Could not transcribe {names}{more}. The file may be damaged or have no "
                "sound. Select \"Show details\" to see why."
            )
            tone = "error"

        self._status_label.configure(text=title, text_color=colour)
        self._batch_label.configure(text=detail)
        self._percent_label.configure(text="")
        self._show_message(message, tone)
        self._stop_progress_motion()
        if saved:
            self._progress_bar.configure(progress_color=C.GREEN)
            self._progress_bar.set(1.0)
        else:
            self._progress_bar.set(0.0)
        retry = bool(failed) or status == "Cancelled"
        if self._batch_recording is not None:
            # A recording: "Try again" is the best quality button.
            self._show_results(
                transcript=len(saved) == 1, folder=True,
                full=retry and not fatal and self._batch_recording.exists(),
            )
            if status == "Done" and saved:
                self._offer_speakers()
        elif saved or retry:
            self._show_results(
                transcript=len(saved) == 1, folder=bool(saved), retry=retry and not fatal
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Recording worker thread
    # ─────────────────────────────────────────────────────────────────────────

    def _recording_worker(self, device_index: int | None, live: bool) -> None:
        transcript_lines: list[str] = []
        outcome: dict[str, object] = {"transcript": None, "audio": None, "error": None}

        def on_segment(text: str) -> None:
            transcript_lines.append(text)
            self._safe_append_log(text)
            self.after(0, lambda t=text: self._append_live(t))

        def on_overflow() -> None:
            self._safe_append_log(
                "[Warning] transcription falling behind. Some audio was "
                "skipped in the live view; the full recording is still saved"
            )

        try:
            # Record only: no speech model to load, so the recording starts at once.
            recognizer = self._engine.get_recognizer(self._safe_update_status) if live else None

            # The raw recording is written to the work folder (crash recovery
            # looks there); the finished WAV goes to the chosen folder at Stop.
            dest_dir = self._recordings_folder()
            RECORDING_WORK_DIR.mkdir(parents=True, exist_ok=True)
            self._mic_recorder.start(device_index, RECORDING_WORK_DIR, feed_vad=live)
            self._safe_update_status("Recording")
            self.after(0, self._start_timer)
            mic_queue = self._mic_recorder.get_queue()
            session_stem = self._mic_recorder.get_session_stem()

            if live:
                self._live_transcriber.start(
                    recognizer,
                    mic_queue,
                    self._stop_recording_event,
                    on_segment,
                    on_overflow,
                )

            self._safe_append_log("[Record] Listening. Select Stop recording when done")
            self._stop_recording_event.wait()

            if live:
                # Speech that was still in progress at Stop is decoded now.
                self._safe_update_status("Finishing")
                self._live_transcriber.stop()
            else:
                self._safe_update_status("Saving Recording")
            wav_path = self._mic_recorder.stop(convert_to_wav=True, dest_dir=dest_dir)
            if wav_path and wav_path.parent != dest_dir:
                self._safe_append_log(
                    f"[Warning] Could not save to {dest_dir}. The recording is in {wav_path.parent}"
                )
            saved_dir = wav_path.parent if wav_path else RECORDING_WORK_DIR

            # Transcript filename matches the recording's start timestamp
            # (session_stem), the same stem the .wav was saved under.
            if transcript_lines and session_stem:
                txt_path = saved_dir / f"{session_stem}.txt"
                try:
                    txt_path.write_text("\n".join(transcript_lines) + "\n", encoding="utf-8")
                    self._last_output_folder = saved_dir
                    outcome["transcript"] = txt_path
                    self._safe_append_log(f"[Done] Transcript saved: {txt_path}")
                except OSError as exc:
                    outcome["error"] = f"Could not save the transcript: {exc}"
                    self._safe_append_log(f"[Error] Could not write transcript: {exc}")

            if wav_path:
                outcome["audio"] = wav_path
                self._safe_append_log(f"[Done] Audio saved: {wav_path}")
                self._last_output_folder = saved_dir

            self._safe_update_status("Done")

        except ModelNotFoundError as exc:
            logger.exception("Model not found during recording")
            self._safe_append_log(f"[Fatal] {exc}")
            outcome["error"] = (
                f"A program file is missing. Please download EasyScribe again.\n{exc}"
            )
            self._safe_update_status("Failed")

        except Exception as exc:
            logger.exception("Recording worker error")
            self._safe_append_log(f"[Error] {exc}")
            outcome["error"] = (
                f"Could not record: {exc}\nCheck that a microphone is connected, and that "
                "Windows allows apps to use it (Settings, Privacy, Microphone)."
            )
            self._safe_update_status("Failed")

        finally:
            self.after(0, lambda: self._finish_recording(outcome, live))

    def _finish_recording(self, outcome: dict, live: bool) -> None:
        self._stop_timer()
        self._record_btn.configure(text="Start recording", command=self._on_record)
        self._set_ui_state("idle")
        self._stop_progress_motion()
        self._percent_label.configure(text="")

        transcript = outcome.get("transcript")
        audio = outcome.get("audio")
        error = outcome.get("error")
        self._last_transcript = transcript if isinstance(transcript, Path) else None
        self._last_recording = audio if isinstance(audio, Path) else None

        if error and not audio:
            self._status_label.configure(text="Could not record", text_color=C.CRIMSON)
            self._batch_label.configure(text="")
            self._show_message(str(error), "error")
            self._progress_bar.set(0.0)
            if not self._live_has_text:
                self._reset_live_box()
            return

        if not live and self._last_recording is not None:
            # "Best quality transcript after recording": start it at once.
            self._on_full_transcript()
            return

        self._status_label.configure(text="Recording saved", text_color=C.GREEN_INK)
        where = _shorten(str(self._last_output_folder or self._recordings_folder()), 60)
        if not live:
            self._batch_label.configure(text=f"The audio is in {where}.")
        elif transcript:
            self._batch_label.configure(text=f"The audio and the live transcript are in {where}.")
        else:
            self._batch_label.configure(
                text="No speech was heard live, so there is no live transcript. "
                     f"The audio is in {where}."
            )
        self._show_message(str(error) if error else "", "error")
        self._progress_bar.configure(progress_color=C.GREEN)
        self._progress_bar.set(1.0)
        self._show_results(transcript=bool(transcript), full=self._last_recording is not None)

    def _on_full_transcript(self) -> None:
        """Send the saved recording through the file pipeline.

        Used at once after a "best quality" recording, and by the "Make best
        quality transcript" button. It uses the more accurate engine, always
        adds timestamps; speakers are found afterwards with "Find speakers".
        It stays in the Record tab and does not touch the
        file list of the Files tab. A live transcript is kept: the new file
        gets its own name.
        """
        audio = self._last_recording
        if self._ui_state != "idle" or audio is None:
            return
        if not audio.exists():
            self._show_message("The recording was moved or deleted.", "error")
            return
        # Next to the recording: the Files tab's folder choice is for that tab.
        folder = audio.parent
        name = f"{audio.stem}.txt"
        if (folder / name).exists():
            name = f"{audio.stem} (best quality).txt"
        self._output_overrides[audio] = folder / name
        options = {"timestamps": True, "diarize": False}
        self._batch_recording = audio
        self._start_batch([audio], options)

    def _make_speaker_naming_callback(self) -> Callable:
        """Called on a worker thread: show the names panel and wait for it."""
        def callback(speaker_map: dict, clips_dict: dict, shares: dict) -> None:
            done_event = threading.Event()

            def show() -> None:
                try:
                    self._show_naming_panel(speaker_map, clips_dict, shares, done_event)
                except Exception:
                    # Never leave the worker waiting: keep "Speaker 1, 2…".
                    logger.exception("Could not show the speaker names panel")
                    done_event.set()

            self.after(0, show)
            while not done_event.wait(0.2):
                if self._cancel_event.is_set():
                    self.after(0, self._close_naming_panel)
                    raise CancelledError("Cancelled while naming speakers")

        return callback

    @_screen_change
    def _show_naming_panel(
        self, speaker_map: dict, clips_dict: dict, shares: dict, done_event: threading.Event
    ) -> None:
        self._close_naming_panel()

        def on_done() -> None:
            self._close_naming_panel()
            done_event.set()

        self._naming_panel = SpeakerNamingPanel(
            self._naming_host, speaker_map, clips_dict, shares, on_done
        )
        self._naming_panel.grid(row=0, column=0, sticky="ew")
        self._naming_host.grid()
        self._flash_taskbar()

    @_screen_change
    def _close_naming_panel(self) -> None:
        if self._naming_panel is not None:
            self._naming_panel.destroy()
            self._naming_panel = None
        self._naming_host.grid_remove()

    def _flash_taskbar(self) -> None:
        """Ask for attention without stealing focus (Windows taskbar flash)."""
        if sys.platform != "win32":
            return
        try:
            import ctypes
            ctypes.windll.user32.FlashWindow(int(self.wm_frame(), 16), True)
        except Exception:
            pass

    # ── Find speakers (after a recording) ────────────────────────────────────

    @_screen_change
    def _offer_speakers(self) -> None:
        audio, segments = self._last_recording, self._last_segments
        if not (self._diarization_available and audio and segments and self._last_transcript):
            return
        try:
            audio_sec = wav_duration(audio)
        except Exception:
            return
        speech_sec = min(speech_seconds(segments), audio_sec)
        estimate = fmt_estimate(estimate_seconds(audio_sec, speech_sec))
        self._speakers_estimate.configure(text=estimate[0].upper() + estimate[1:])
        self._speakers_basis.configure(
            text=f"Based on {_fmt_minutes(audio_sec)} of audio, with "
                 f"{_fmt_minutes(speech_sec)} of speech. You can keep using your computer "
                 "and stop at any time."
        )
        self._find_speakers_btn.configure(state="normal")
        self._speakers_card.grid()

    def _on_find_speakers(self) -> None:
        audio, segments, transcript = self._last_recording, self._last_segments, self._last_transcript
        if self._ui_state != "idle" or not (audio and segments and transcript):
            return
        if not audio.exists():
            self._show_message("The recording was moved or deleted.", "error")
            return
        output_path = transcript.with_name(f"{transcript.stem} (speakers).txt")
        self._cancel_event.clear()
        self._clear_log()
        self._set_ui_state("running")
        self._show_activity()
        self._batch_label.configure(text=_shorten(audio.name, 80))
        threading.Thread(
            target=self._speakers_worker, args=(audio, segments, transcript, output_path),
            daemon=True, name="SpeakersWorker",
        ).start()

    def _speakers_worker(
        self,
        audio: Path,
        segments: list[tuple[float, float, str]],
        transcript: Path,
        output_path: Path,
    ) -> None:
        started = time.monotonic()
        status, error = "Done", None
        try:
            add_speakers_to_file(
                audio, segments, output_path, self._cancel_event,
                self._safe_update_status, self._safe_set_progress, self._safe_append_log,
                self._make_speaker_naming_callback(),
            )
        except CancelledError:
            status = "Cancelled"
            self._safe_append_log("[Cancelled] Operation stopped by user")
        except Exception as exc:
            logger.exception("Finding speakers failed")
            self._safe_append_log(f"[Error] {exc}")
            status, error = "Failed", str(exc)
        took = time.monotonic() - started
        self._safe_update_status(status)
        self.after(
            0, lambda: self._finish_speakers(status, transcript, output_path, error, took)
        )

    def _finish_speakers(
        self, status: str, transcript: Path, output_path: Path, error: str | None, took: float
    ) -> None:
        self._set_ui_state("idle")
        self._close_naming_panel()
        self._stop_progress_motion()
        self._percent_label.configure(text="")
        if status == "Done":
            self._last_transcript = output_path
            self._last_output_folder = output_path.parent
            self._status_label.configure(text="Speakers added", text_color=C.GREEN_INK)
            self._batch_label.configure(
                text=f"{_shorten(output_path.name, 70)} · took {_fmt_took(took)}"
            )
            self._show_message("The first transcript without names is kept too.", "info")
            self._progress_bar.configure(progress_color=C.GREEN)
            self._progress_bar.set(1.0)
            self._show_results(transcript=True, folder=True)
            return
        self._last_transcript = transcript
        if status == "Cancelled":
            self._status_label.configure(text="Stopped", text_color=C.MUTED)
            self._batch_label.configure(text="No names were added. Your transcript is unchanged.")
            self._show_message("", "info")
        else:
            self._status_label.configure(text="Could not find the speakers", text_color=C.CRIMSON)
            self._batch_label.configure(text="Your transcript is unchanged.")
            self._show_message(
                f"{error}\nSelect \"Show details\" to see why.", "error"
            )
        self._progress_bar.set(0.0)
        self._show_results(transcript=True, folder=True)
        self._offer_speakers()

    # ─────────────────────────────────────────────────────────────────────────
    # Thread-safe GUI helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _safe_update_status(self, status: str) -> None:
        self.after(0, lambda s=status: self._update_status(s))

    def _safe_set_progress(self, value: float) -> None:
        self.after(0, lambda v=value: self._set_progress(v))

    def _safe_append_log(self, text: str) -> None:
        self.after(0, lambda t=text: self._append_log(t))

    def _safe_set_batch_label(self, text: str) -> None:
        self.after(0, lambda t=text: self._batch_label.configure(text=t))

    def _safe_set_file_state(self, path: Path, state: str) -> None:
        def apply() -> None:
            self._file_states[path] = state
            self._render_files()
        self.after(0, apply)

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers (main thread only)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_mic_options(self) -> tuple[list[str], dict[str, int | None]]:
        options: list[str] = ["Default microphone"]
        index_map: dict[str, int | None] = {"Default microphone": None}

        try:
            devices = MicRecorder.list_devices()
            for d in devices:
                # Windows lists one microphone once per audio driver API.
                # Show each name once (the first, most compatible, entry).
                label = _shorten(str(d["name"]).strip(), 48)
                if label not in index_map:
                    options.append(label)
                    index_map[label] = int(d["index"])
        except Exception as exc:
            logger.warning(f"Could not enumerate microphones: {exc}")

        return options, index_map

    def _add_files(self, paths: list[Path]) -> None:
        if self._ui_state == "idle" and self._mode == _MODE_FILES:
            # A new batch: the last result no longer applies.
            self._hide_activity()
            self._file_states = {
                p: s for p, s in self._file_states.items() if s == "Waiting"
            }
        existing = set(self._selected_files)
        for p in paths:
            if p not in existing and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                self._selected_files.append(p)
                existing.add(p)
        self._render_files()

    def _render_files(self) -> None:
        for child in self._file_list.winfo_children():
            child.destroy()

        count = len(self._selected_files)
        busy = self._ui_state != "idle"

        if self._dnd_enabled:
            title = "Drag audio or video files here" if count == 0 else "Drag more files here"
            sub = (
                "Or choose them from your computer. MP3, WAV, M4A, MP4 and more."
                if count == 0 else "Or add more from your computer."
            )
        else:
            title = "Choose the recordings to transcribe"
            sub = "MP3, WAV, M4A, MP4 and other audio and video files."
        self._drop_title.configure(text=title)
        self._drop_sub.configure(text=sub)
        # Once files are listed, the drop zone shrinks to one line.
        if count:
            self._drop_sub.grid_remove()
            self._drop_title.grid_configure(pady=14)
        else:
            self._drop_sub.grid()
            self._drop_title.grid_configure(pady=(14, 0))
        self._select_files_btn.configure(text="Choose files" if count == 0 else "Add files")

        self._files_count_label.configure(text=_plural(count, "file") if count else "")
        if count:
            self._clear_files_btn.grid()
        else:
            self._clear_files_btn.grid_remove()

        state_style = {
            "Waiting": (C.MUTED, C.SURFACE_ALT),
            "Working": (C.TEAL_INK, C.TEAL_SOFT),
            "Done": (C.GREEN_INK, C.GREEN_SOFT),
            "Failed": (C.CRIMSON, C.CRIMSON_SOFT),
            "Stopped": (C.MUTED, C.SURFACE_ALT),
        }

        for i, path in enumerate(self._selected_files):
            row = ctk.CTkFrame(self._file_list, fg_color="transparent")
            row.grid(row=i, column=0, padx=8, pady=3, sticky="ew")
            row.grid_columnconfigure(0, weight=1)

            try:
                size = _fmt_size(path.stat().st_size)
            except OSError:
                size = "Not found"
            ctk.CTkLabel(
                row, text=_shorten(path.name, 60), font=self._f_body,
                text_color=C.INK, anchor="w", height=20,
            ).grid(row=0, column=0, sticky="w")
            ctk.CTkLabel(
                row, text=f"{size}  ·  {_shorten(str(path.parent), 60)}",
                font=self._f_small, text_color=C.MUTED, anchor="w", height=18,
            ).grid(row=1, column=0, sticky="w")

            state = self._file_states.get(path)
            if state:
                fg, bg = state_style.get(state, (C.MUTED, C.SURFACE_ALT))
                ctk.CTkLabel(
                    row, text=state, font=self._f_small, text_color=fg, fg_color=bg,
                    corner_radius=R_SM, width=72, height=24,
                ).grid(row=0, column=1, rowspan=2, padx=(8, 4))

            _button(
                row, "Remove", kind="link", width=70, height=28,
                text_color=C.MUTED, state="disabled" if busy else "normal",
                command=lambda p=path: self._on_remove_file(p),
            ).grid(row=0, column=2, rowspan=2, padx=(4, 0))

        if count:
            self._file_list.configure(height=min(count, self._visible_rows) * _FILE_ROW_HEIGHT)
            self._file_list.grid()
        else:
            self._file_list.grid_remove()

        self._refresh_start_button()

    def _refresh_start_button(self) -> None:
        count = len(self._selected_files)
        if self._ui_state == "running":
            return
        if count == 0:
            self._transcribe_btn.configure(text="Transcribe")
            self._start_hint.configure(text="Add a recording to start.", text_color=C.MUTED)
        else:
            self._transcribe_btn.configure(text=f"Transcribe {_plural(count, 'file')}")
            self._start_hint.configure(text="", text_color=C.MUTED)
        self._set_primary_enabled(self._transcribe_btn, count > 0 and self._ui_state == "idle", C.TEAL)

    def _set_primary_enabled(self, btn: ctk.CTkButton, enabled: bool, colour: str) -> None:
        btn.configure(
            state="normal" if enabled else "disabled",
            fg_color=colour if enabled else C.SECONDARY,
        )

    def _refresh_output_label(self) -> None:
        if self._output_folder:
            self._output_label.configure(
                text=_shorten(str(self._output_folder), 70), text_color=C.INK
            )
            self._reset_output_btn.grid()
        else:
            self._output_label.configure(
                text="The same folder as each recording", text_color=C.MUTED
            )
            self._reset_output_btn.grid_remove()

    def _refresh_rec_folder_label(self) -> None:
        if self._rec_folder:
            self._rec_folder_label.configure(
                text=_shorten(str(self._rec_folder), 70), text_color=C.INK
            )
            self._reset_rec_folder_btn.grid()
        else:
            self._rec_folder_label.configure(
                text=_shorten(str(DEFAULT_RECORDINGS_DIR), 70), text_color=C.MUTED
            )
            self._reset_rec_folder_btn.grid_remove()

    def _resolve_output_path(self, input_file: Path) -> Path:
        if input_file in self._output_overrides:
            return self._output_overrides[input_file]
        folder = self._output_folder or input_file.parent
        return folder / (input_file.stem + ".txt")

    @_screen_change
    def _show_activity(self) -> None:
        self._activity.grid(row=3, column=0, padx=24, pady=(0, 10), sticky="ew")
        self._message_label.grid_remove()
        self._result_row.grid_remove()
        self._rename_row.grid_remove()
        self._best_row.grid_remove()
        self._speakers_card.grid_remove()
        self._batch_label.configure(text="")
        self._percent_label.configure(text="")

        self._fold_options(True)
        # The panel takes the start button's place, so the progress and the
        # result fit on short screens. "Change" or new files bring it back.
        self._action_row.grid_remove()

    @_screen_change
    def _hide_activity(self) -> None:
        self._activity.grid_remove()
        self._fold_options(False)
        self._action_row.grid()

    def _fold_options(self, folded: bool) -> None:
        # Both modes: the speakers card and names panel need the room.
        self._fold_rec_setup(folded)
        if not folded or self._mode != _MODE_FILES:
            self._options_summary.grid_remove()
            self._change_options_btn.grid_remove()
            self._opt_body.grid()
            return
        parts = [
            "Timestamps on" if self._timestamps_var.get() else "Timestamps off",
            "speaker names on" if self._diarize_var.get() else "speaker names off",
        ]
        where = (
            f"saved in {_shorten(str(self._output_folder), 50)}"
            if self._output_folder else "saved next to each recording"
        )
        self._options_summary.configure(text=f"{parts[0]}, {parts[1]}, {where}.")
        self._opt_body.grid_remove()
        self._options_summary.grid()
        if self._ui_state == "idle":
            self._change_options_btn.grid()
        else:
            self._change_options_btn.grid_remove()

    def _fold_rec_setup(self, folded: bool) -> None:
        for widget in self._rec_setup:
            widget.grid_remove() if folded else widget.grid()
        if not folded:
            self._rec_summary.grid_remove()
            return
        self._rec_summary_label.configure(
            text=f"{_shorten(self._mic_var.get(), 50)}. "
                 + ("Show words as I speak." if self._rec_mode_var.get() == _REC_LIVE
                    else "Best quality transcript after recording.")
        )
        self._rec_summary.grid()
        if self._ui_state == "idle":
            self._rec_change_btn.grid()
        else:
            self._rec_change_btn.grid_remove()

    def _on_change_options(self) -> None:
        if self._ui_state == "idle":
            self._hide_activity()

    def _show_message(self, text: str, tone: str) -> None:
        if not text:
            self._message_label.grid_remove()
            return
        fg, bg = (C.CRIMSON, C.CRIMSON_SOFT) if tone == "error" else (C.INK, C.SURFACE_ALT)
        self._message_label.configure(text=text, text_color=fg, fg_color=bg)
        self._message_label.grid(ipadx=10, ipady=8)

    @_screen_change
    def _show_results(
        self, transcript: bool, folder: bool = True, retry: bool = False, full: bool = False
    ) -> None:
        for btn, show in (
            (self._open_transcript_btn, transcript),
            (self._open_output_btn, folder),
            (self._retry_btn, retry),
        ):
            btn.grid() if show else btn.grid_remove()
        self._result_row.grid()
        if full:
            self._best_note.configure(
                text="Adds times. More accurate. Your live transcript is kept. "
                     "You can find the speakers after."
                if self._live_has_text
                else "Adds times. Most accurate. You can find the speakers after."
            )
            self._best_row.grid()
        else:
            self._best_row.grid_remove()
        self._offer_rename()

    # ── Rename (after a result) ──────────────────────────────────────────────

    def _rename_targets(self) -> tuple[list[Path], str] | None:
        """The files the Name box renames, and the name they share now."""
        audio = self._last_recording if self._mode == _MODE_RECORD else None
        if audio is not None and audio.exists():
            return recording_set(audio), audio.stem
        transcript = self._last_transcript
        if transcript is not None and transcript.exists():
            return [transcript], transcript.stem
        return None

    def _offer_rename(self) -> None:
        targets = self._rename_targets()
        if targets is None:
            self._rename_row.grid_remove()
            return
        files, stem = targets
        if self._mode != _MODE_RECORD:
            what = "Renames the transcript. Your original file is not changed."
        elif len(files) > 1:
            what = "Renames the recording and its transcripts together."
        else:
            what = "Renames the recording."
        # The current name, so a word can be added instead of typed again.
        self._rename_entry.delete(0, "end")
        self._rename_entry.insert(0, stem)
        self._rename_note.configure(text=what, text_color=C.MUTED)
        self._rename_row.grid()

    def _on_rename(self) -> None:
        targets = self._rename_targets()
        if self._ui_state != "idle" or targets is None:
            return
        files, stem = targets
        name = self._rename_entry.get().strip()
        problem = check_name(name)
        if not problem:
            try:
                moved = rename_all(files, stem, name)
            except FileExistsError as exc:
                problem = f'"{exc}" already exists in that folder. Please choose another name.'
            except OSError as exc:
                logger.warning(f"Rename failed: {exc}")
                problem = ("Could not rename. If the file is open in another program, "
                           "close it and try again.")
        if problem:
            self._rename_note.configure(text=problem, text_color=C.CRIMSON)
            return
        for attr in ("_last_recording", "_last_transcript", "_batch_recording"):
            path = getattr(self, attr)
            setattr(self, attr, moved.get(path, path) if path else path)
        self._rename_note.configure(text=f"\u2713 Renamed to {name}", text_color=C.GREEN_INK)
        self._append_log(f"[Rename] {stem} -> {name} ({len(moved)} files)")

    def _update_status(self, status: str) -> None:
        title, colour, bar_colour, mode = _STATUS.get(
            status, (status, C.INK, C.TEAL, "pause")
        )
        self._status_label.configure(text=title, text_color=colour)
        self._progress_bar.configure(progress_color=bar_colour)
        hint = _STATUS_HINTS.get(status)
        if hint and self._ui_state != "idle":
            self._show_message(hint, "info")
        elif self._ui_state != "idle":
            self._message_label.grid_remove()

        if mode == "busy":
            self._start_progress_motion()
            self._percent_label.configure(text="")
        else:
            self._stop_progress_motion()
            if mode == "bar":
                self._percent_label.configure(text="0%")

    def _set_progress(self, value: float) -> None:
        if self._progress_running:
            self._stop_progress_motion()
        value = max(0.0, min(1.0, float(value)))
        self._progress_bar.set(value)
        if self._ui_state == "running":
            self._percent_label.configure(text=f"{int(round(value * 100))}%")

    def _start_progress_motion(self) -> None:
        if not self._progress_running:
            self._progress_bar.configure(mode="indeterminate")
            self._progress_bar.start()
            self._progress_running = True

    def _stop_progress_motion(self) -> None:
        if self._progress_running:
            self._progress_bar.stop()
            self._progress_bar.configure(mode="determinate")
            self._progress_bar.set(0)
            self._progress_running = False

    def _start_timer(self) -> None:
        self._rec_started_at = time.monotonic()
        self._timer_label.configure(text_color=C.CORAL_INK)
        self._tick_timer()
        self._tick_level()

    def _tick_level(self) -> None:
        if self._rec_started_at is None:
            self._level_bar.set(0)
            self._level_bar.configure(progress_color=C.BORDER)
            return
        self._level_bar.configure(progress_color=C.CORAL)
        # Square root: quiet speech still moves the bar clearly.
        self._level_bar.set(min(1.0, self._mic_recorder.get_level() ** 0.5))
        self.after(100, self._tick_level)

    def _tick_timer(self) -> None:
        if self._rec_started_at is None:
            return
        secs = int(time.monotonic() - self._rec_started_at)
        h, rem = divmod(secs, 3600)
        m, s = divmod(rem, 60)
        self._timer_label.configure(text=f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}")
        self.after(500, self._tick_timer)

    def _stop_timer(self) -> None:
        self._rec_started_at = None
        self._timer_label.configure(text_color=C.FAINT)

    def _reset_live_box(self, listening: bool = False) -> None:
        self._live_has_text = False
        self._live_box.configure(state="normal", text_color=C.MUTED)
        self._live_box.delete("1.0", "end")
        self._live_box.insert(
            "end",
            "Listening. The words appear here as people speak." if listening
            else "Select Start recording. The words appear here as people speak.",
        )
        self._live_box.configure(state="disabled")

    def _append_live(self, text: str) -> None:
        self._live_box.configure(state="normal")
        if not self._live_has_text:
            self._live_box.delete("1.0", "end")
            self._live_box.configure(text_color=C.INK)
            self._live_has_text = True
        self._live_box.insert("end", text.strip() + "\n")
        self._live_box.see("end")
        self._live_box.configure(state="disabled")

    def _append_log(self, text: str) -> None:
        self._log_box.configure(state="normal")
        self._log_box.insert("end", text + "\n")
        self._log_box.see("end")
        self._log_box.configure(state="disabled")

    def _clear_log(self) -> None:
        self._log_box.configure(state="normal")
        self._log_box.delete("1.0", "end")
        self._log_box.configure(state="disabled")

    def _set_ui_state(self, state: str) -> None:
        """Toggle widgets between 'idle', 'running', and 'recording' states."""
        self._ui_state = state
        is_running = state == "running"
        is_recording = state == "recording"
        is_busy = is_running or is_recording
        normal_if_idle = "disabled" if is_busy else "normal"

        self._mode_switch.configure(state=normal_if_idle)
        self._set_primary_enabled(self._select_files_btn, not is_busy, C.TEAL)
        self._select_output_btn.configure(state=normal_if_idle)
        self._reset_output_btn.configure(state=normal_if_idle)
        self._clear_files_btn.configure(state=normal_if_idle)
        for cb, available in ((self._timestamps_cb, True), (self._diarize_cb, self._diarization_available)):
            cb.configure(state="normal" if (available and not is_busy) else "disabled")
        self._mic_menu.configure(state="disabled" if is_busy else "normal")
        self._select_rec_folder_btn.configure(state=normal_if_idle)
        self._reset_rec_folder_btn.configure(state=normal_if_idle)
        for _tile, radio in self._rec_tiles.values():
            radio.configure(state="disabled" if is_busy else "normal")

        # Record button: disabled while transcription runs; becomes Stop while recording
        self._record_btn.configure(state="disabled" if is_running else "normal")

        # Stop button: only shown during file transcription
        if is_running:
            self._cancel_btn.configure(state="normal")
            self._cancel_btn.grid()
            self._transcribe_btn.configure(text="Transcribing…")
            self._set_primary_enabled(self._transcribe_btn, False, C.TEAL)
            self._start_hint.configure(text="")
        else:
            self._cancel_btn.grid_remove()

        self._render_files()
        if self._activity.winfo_manager():
            self._fold_options(True)
        if not is_busy:
            self._stop_progress_motion()
