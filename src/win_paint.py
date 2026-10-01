"""
win_paint.py - stop black blocks when Windows repaints the EasyScribe window.

Why: Tk on Windows never paints a window's background when Windows asks.
Its child window class ("TkChild") has no background brush, and Tk answers
WM_ERASEBKGND with 0 ("not erased"). Tk draws the real content a moment
later, from its idle queue. Until then the area is black. The app window
is made of many CustomTkinter canvases (one child window each), so when
Windows must repaint it, for example when it comes back to the front, it
fills in as black blocks for up to a second.

Fix: install() replaces the window procedure of the "TkChild" class with
one that fills the area with the app background colour on WM_ERASEBKGND,
and passes every other message to Tk's own procedure. The real content
follows as before; the short gap is light, not black. Windows created
after install() use the new procedure, so call it before the widgets are
made.

Windows only. If a step fails, Tk stays as it is. Set the environment
variable EASYSCRIBE_PAINT_FIX=0 to switch it off.
"""

import ctypes
import logging
import os
import sys

logger = logging.getLogger(__name__)

_WM_ERASEBKGND = 0x0014
_GWLP_WNDPROC = -4
_GCLP_WNDPROC = -24

# Keeps the callback, brush and Tk's procedure alive for the whole process.
# Windows calls the callback until the process ends.
_state: dict = {}


def _colorref(colour: str) -> int:
    """'#RRGGBB' -> Windows COLORREF (0x00BBGGRR)."""
    r, g, b = int(colour[1:3], 16), int(colour[3:5], 16), int(colour[5:7], 16)
    return r | (g << 8) | (b << 16)


def install(root, background: str) -> bool:  # type: ignore[no-untyped-def]
    """Paint the background of Tk windows in *background* (#RRGGBB) at once.

    Returns True if the new procedure is in place.
    """
    if sys.platform != "win32" or os.environ.get("EASYSCRIBE_PAINT_FIX") == "0":
        return False
    if _state:
        return True
    try:
        from ctypes import wintypes

        lresult = ctypes.c_ssize_t
        wndproc_type = ctypes.WINFUNCTYPE(
            lresult, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
        )
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
        call_proc = user32.CallWindowProcW
        call_proc.restype = lresult
        call_proc.argtypes = [
            ctypes.c_void_p, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
        ]
        user32.SetClassLongPtrW.restype = ctypes.c_size_t
        user32.SetClassLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
        user32.GetWindowLongPtrW.restype = ctypes.c_size_t
        user32.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.SetWindowLongPtrW.restype = ctypes.c_size_t
        user32.SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
        user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        get_rect = user32.GetClientRect
        get_rect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        fill_rect = user32.FillRect
        fill_rect.argtypes = [wintypes.HDC, ctypes.POINTER(wintypes.RECT), wintypes.HBRUSH]
        gdi32.CreateSolidBrush.restype = wintypes.HBRUSH
        gdi32.CreateSolidBrush.argtypes = [wintypes.COLORREF]

        root.update_idletasks()
        hwnd = root.winfo_id()
        name = ctypes.create_unicode_buffer(64)
        user32.GetClassNameW(hwnd, name, 64)
        if name.value != "TkChild":
            logger.info(f"Paint fix not used: unexpected window class {name.value!r}")
            return False
        brush = gdi32.CreateSolidBrush(_colorref(background))
        if not brush:
            return False

        def proc(h, msg, wparam, lparam):  # type: ignore[no-untyped-def]
            if msg == _WM_ERASEBKGND:
                try:
                    rect = wintypes.RECT()
                    if get_rect(h, ctypes.byref(rect)):
                        fill_rect(wparam, ctypes.byref(rect), brush)
                        return 1  # erased
                except Exception:
                    pass
            return call_proc(_state["tk_proc"], h, msg, wparam, lparam)

        callback = wndproc_type(proc)
        address = ctypes.cast(callback, ctypes.c_void_p).value
        _state.update(callback=callback, brush=brush)

        own_proc = user32.GetWindowLongPtrW(hwnd, _GWLP_WNDPROC)
        tk_proc = user32.SetClassLongPtrW(hwnd, _GCLP_WNDPROC, address)
        if not tk_proc:
            _state.clear()
            logger.info("Paint fix not used: could not change the window class")
            return False
        _state["tk_proc"] = tk_proc
        # The main window's own child window already exists; give it the
        # new procedure too, but only if Tk has not replaced it itself.
        if own_proc == tk_proc:
            user32.SetWindowLongPtrW(hwnd, _GWLP_WNDPROC, address)
        logger.info("Paint fix active: window backgrounds are painted at once")
        return True
    except Exception as exc:
        logger.info(f"Paint fix not used: {exc}")
        return False
