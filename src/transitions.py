"""
transitions.py - fade out, fade in when the EasyScribe window changes screen.

Why: when a screen changes (Files <-> Record, a job starts, a result shows),
every CustomTkinter widget draws itself on its own pass after it is placed.
Measured on a tab switch: 3-4 half-drawn frames (an empty card, grey boxes
without text, radio buttons without labels) before the finished screen.

How (Windows only):
1. Window drawing is held (WM_SETREDRAW off on Tk's client window), so the
   screen keeps showing the old content while Tk builds the new one.
2. A borderless "veil" window in the background colour, owned by the app
   window and placed over the area below the tabs, fades in (old content
   fades to the background, OUT_MS).
3. Drawing is switched back on and the whole window repaints at once,
   under the veil. After SETTLE_MS the veil fades away (IN_MS).

The change itself always runs at once, inside run(), so callers keep their
order. A change during a running transition joins it. With Windows
"Animation effects" off, the drawing is held for OUT_MS and released in
one go, without the veil. Not used while the app is not the front window.

If a step fails, changes are applied directly. EASYSCRIBE_TRANSITIONS=0
switches it off.
"""

import ctypes
import logging
import os
import sys
import time
from typing import Callable

logger = logging.getLogger(__name__)

OUT_MS = 90
SETTLE_MS = 40
IN_MS = 140
_TICK_MS = 15

_WM_SETREDRAW = 0x000B
_GWL_EXSTYLE = -20
_GWLP_HWNDPARENT = -8
_WS_EX_LAYERED = 0x00080000
_WS_EX_TOOLWINDOW = 0x00000080
_WS_EX_NOACTIVATE = 0x08000000
_LWA_ALPHA = 0x2
_SWP_NOACTIVATE = 0x0010
_SWP_SHOWWINDOW = 0x0040
_SW_HIDE = 0
_RDW_INVALIDATE, _RDW_ERASE, _RDW_ALLCHILDREN = 0x1, 0x4, 0x80
_RDW_UPDATENOW, _RDW_ERASENOW, _RDW_FRAME = 0x100, 0x200, 0x400
_SPI_GETCLIENTAREAANIMATION = 0x1042


def _animations_on() -> bool:
    """Windows' "Animation effects" setting (Accessibility > Visual effects)."""
    try:
        value = ctypes.c_int(1)
        ctypes.windll.user32.SystemParametersInfoW(
            _SPI_GETCLIENTAREAANIMATION, 0, ctypes.byref(value), 0
        )
        return bool(value.value)
    except Exception:
        return True


class Veil:
    """Covers the area below *top_widget* while the screen changes."""

    def __init__(self, root, top_widget, colour: str) -> None:  # type: ignore[no-untyped-def]
        import tkinter as tk

        self._root = root
        self._top_widget = top_widget
        self._win = tk.Toplevel(root, bg=colour)
        self._win.overrideredirect(True)
        self._win.geometry("1x1+-32000+-32000")
        self._hwnd = 0
        self._phase: str | None = None  # "out" | "settle" | "in"
        self._alpha = 0.0
        self._a0 = 0.0
        self._t0 = 0.0
        self._frozen = False
        self._animate = True
        self._ready = False

    # ── Windows calls (replaced in tests) ───────────────────────────────────

    def _setup(self) -> None:
        """First use: the veil and app windows exist now; set the veil's styles."""
        from ctypes import wintypes

        user32 = ctypes.WinDLL("user32")
        user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        user32.RedrawWindow.argtypes = [wintypes.HWND, ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT]
        user32.SetWindowPos.argtypes = [
            wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            wintypes.UINT,
        ]
        user32.SetLayeredWindowAttributes.argtypes = [
            wintypes.HWND, wintypes.COLORREF, wintypes.BYTE, wintypes.DWORD
        ]
        user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.GetForegroundWindow.restype = wintypes.HWND
        user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
        user32.SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
        self._user32 = user32
        self._root.update_idletasks()
        self._hwnd = int(self._win.wm_frame(), 16)
        self._root_hwnd = int(self._root.wm_frame(), 16)
        self._client_hwnd = self._root.winfo_id()
        ex = user32.GetWindowLongW(self._hwnd, _GWL_EXSTYLE)
        user32.SetWindowLongW(
            self._hwnd, _GWL_EXSTYLE, ex | _WS_EX_LAYERED | _WS_EX_TOOLWINDOW | _WS_EX_NOACTIVATE
        )
        # Owned by the app window: stays just above it, never above other apps.
        user32.SetWindowLongPtrW(self._hwnd, _GWLP_HWNDPARENT, self._root_hwnd)
        user32.ShowWindow(self._hwnd, _SW_HIDE)

    def _is_front(self) -> bool:
        return (self._user32.GetForegroundWindow() or 0) == self._root_hwnd

    def _hold_drawing(self, hold: bool) -> None:
        self._user32.SendMessageW(self._client_hwnd, _WM_SETREDRAW, 0 if hold else 1, 0)
        if not hold:
            self._user32.RedrawWindow(
                self._client_hwnd, None, None,
                _RDW_INVALIDATE | _RDW_ERASE | _RDW_ALLCHILDREN | _RDW_FRAME,
            )

    def _place_veil(self) -> None:
        """Show the veil, see-through, over the area below the top widget."""
        user32, top = self._user32, self._top_widget
        user32.SetLayeredWindowAttributes(self._hwnd, 0, 0, _LWA_ALPHA)
        x, y = self._root.winfo_rootx(), top.winfo_rooty() + top.winfo_height()
        w = self._root.winfo_width()
        h = self._root.winfo_rooty() + self._root.winfo_height() - y
        user32.SetWindowPos(self._hwnd, 0, x, y, w, h, _SWP_NOACTIVATE | _SWP_SHOWWINDOW)
        # Paint its background now (win_paint), not black on the first frame.
        user32.RedrawWindow(
            self._hwnd, None, None,
            _RDW_INVALIDATE | _RDW_ERASE | _RDW_ALLCHILDREN | _RDW_UPDATENOW | _RDW_ERASENOW,
        )

    def _show_veil(self, alpha: float) -> None:
        self._user32.SetLayeredWindowAttributes(self._hwnd, 0, int(round(alpha * 255)), _LWA_ALPHA)

    def _hide_veil(self) -> None:
        self._user32.ShowWindow(self._hwnd, _SW_HIDE)

    # ── Transition ──────────────────────────────────────────────────────────

    def run(self, change: Callable[[], None]) -> None:
        """Apply *change* now; hide the redraw behind a fade."""
        if self._phase is not None:
            self._join(change)
            return
        try:
            if not self._ready:
                self._setup()
                self._ready = True
            if not self._is_front() or self._root.state() not in ("normal", "zoomed"):
                change()
                return
            self._animate = _animations_on()
            if self._animate:
                self._place_veil()
            # Paint what is already pending (the tab that was just clicked),
            # then hold drawing while the screen changes.
            self._root.update_idletasks()
            self._hold_drawing(True)
            self._frozen = True
        except Exception:
            logger.exception("Screen transition not available")
            self._release()
            change()
            return
        self._start("out", 0.0)
        try:
            change()
        finally:
            self._tick()

    def _join(self, change: Callable[[], None]) -> None:
        """A change while a transition runs: cover again if the veil is lifting."""
        if self._phase in ("settle", "in"):
            try:
                self._hold_drawing(True)
                self._frozen = True
            except Exception:
                pass
            self._start("out", self._alpha)
        change()

    def _start(self, phase: str, alpha: float) -> None:
        self._phase, self._a0, self._alpha, self._t0 = phase, alpha, alpha, time.perf_counter()

    def _tick(self) -> None:
        try:
            self._step()
        except Exception:
            logger.exception("Screen transition failed")
            self._release()
            return
        if self._phase is not None:
            self._root.after(_TICK_MS, self._tick)

    def _step(self) -> None:
        elapsed = (time.perf_counter() - self._t0) * 1000
        if self._phase == "out":
            t = min(1.0, elapsed / OUT_MS)
            self._alpha = self._a0 + (1 - self._a0) * t * t  # ease in
            if self._animate:
                self._show_veil(self._alpha)
            if t >= 1.0:
                self._hold_drawing(False)
                self._frozen = False
                if self._animate:
                    self._start("settle", 1.0)
                else:
                    self._release()
        elif self._phase == "settle":
            if elapsed >= SETTLE_MS:
                self._start("in", 1.0)
        elif self._phase == "in":
            t = min(1.0, elapsed / IN_MS)
            self._alpha = (1 - t) ** 2  # ease out
            self._show_veil(self._alpha)
            if t >= 1.0:
                self._release()

    def _release(self) -> None:
        """End any transition: drawing on, veil hidden."""
        self._phase = None
        self._alpha = 0.0
        if self._frozen:
            self._frozen = False
            try:
                self._hold_drawing(False)
            except Exception:
                pass
        try:
            if self._hwnd:
                self._hide_veil()
        except Exception:
            pass


def create(root, top_widget, colour: str) -> "Veil | None":  # type: ignore[no-untyped-def]
    """A Veil on Windows, else None (screens then change directly)."""
    if sys.platform != "win32" or os.environ.get("EASYSCRIBE_TRANSITIONS") == "0":
        return None
    try:
        return Veil(root, top_widget, colour)
    except Exception:
        logger.exception("Screen transitions off")
        return None
