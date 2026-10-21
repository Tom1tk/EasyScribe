#!/usr/bin/env python3
"""
Local tests for src/transitions.py (the fade between screens).

The Windows calls are replaced by a recorder and time is driven by hand, so
this runs anywhere without a display. Covers:
- drawing is held before the change, the change runs at once, drawing comes
  back after the fade-out, the veil is hidden at the end;
- a change during the fade-out only runs; one while the veil lifts covers again;
- animations off: no veil, drawing held then released;
- not the front window, or a failing Windows call: the change still runs.

Run from project root: python tests/test_transitions.py
Exits 0 on success, 1 on failure.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import transitions
from transitions import Veil

failures: list[str] = []


def _check(name: str, fn) -> None:
    try:
        fn()
        print(f"  OK   {name}")
    except Exception as exc:
        failures.append(name)
        print(f"  FAIL {name}: {exc!r}")


class _Root:
    def __init__(self) -> None:
        self.pending = []

    def update_idletasks(self) -> None:
        pass

    def state(self) -> str:
        return "normal"

    def after(self, _ms: int, fn) -> None:
        self.pending.append(fn)


class _Clock:
    now = 0.0


def _veil(front: bool = True, animate: bool = True) -> tuple[Veil, list]:
    log: list = []
    v = object.__new__(Veil)
    v._root = _Root()
    v._phase, v._alpha, v._a0, v._t0 = None, 0.0, 0.0, 0.0
    v._frozen, v._animate, v._ready, v._hwnd = False, True, False, 1
    v._setup = lambda: log.append("setup")
    v._is_front = lambda: front
    v._hold_drawing = lambda hold: log.append("hold" if hold else "draw")
    v._place_veil = lambda: log.append("place")
    v._show_veil = lambda a: log.append(round(a, 2))
    v._hide_veil = lambda: log.append("hide")
    transitions._animations_on = lambda: animate
    return v, log


def _run_out(v: Veil, ms: float) -> None:
    """Advance the clock by *ms* and run the scheduled ticks until idle or *ms* passes."""
    end = _Clock.now + ms / 1000
    while v._root.pending and _Clock.now < end:
        _Clock.now = min(end, _Clock.now + transitions._TICK_MS / 1000)
        v._root.pending.pop(0)()


transitions.time.perf_counter = lambda: _Clock.now


def test_order() -> None:
    v, log = _veil()
    v.run(lambda: log.append("change"))
    assert log[:4] == ["setup", "place", "hold", "change"], log
    assert v._phase == "out" and v._frozen
    _run_out(v, 1000)
    assert v._phase is None and not v._frozen
    assert log[-1] == "hide", log
    draw = log.index("draw")
    out_alphas = [a for a in log[:draw] if isinstance(a, float)]
    assert out_alphas[-1] == 1.0 and out_alphas == sorted(out_alphas), out_alphas
    in_alphas = [a for a in log[draw:] if isinstance(a, float)]
    assert in_alphas[-1] == 0.0 and in_alphas == sorted(in_alphas, reverse=True), in_alphas


def test_join_while_fading_out() -> None:
    v, log = _veil()
    v.run(lambda: log.append("first"))
    _run_out(v, 30)
    v.run(lambda: log.append("second"))
    assert log.count("hold") == 1 and "second" in log and v._phase == "out"
    _run_out(v, 1000)
    assert log[-1] == "hide" and log.count("draw") == 1, log


def test_join_while_lifting() -> None:
    v, log = _veil()
    v.run(lambda: None)
    _run_out(v, transitions.OUT_MS + transitions.SETTLE_MS + 60)
    assert v._phase == "in", v._phase
    lifted = v._alpha
    v.run(lambda: log.append("again"))
    assert log[-2:] == ["hold", "again"] and v._phase == "out" and v._a0 == lifted, log
    _run_out(v, 1000)
    assert log.count("draw") == 2 and log[-1] == "hide" and not v._frozen, log


def test_no_animation() -> None:
    v, log = _veil(animate=False)
    v.run(lambda: log.append("change"))
    _run_out(v, 1000)
    assert log == ["setup", "hold", "change", "draw", "hide"], log


def test_not_front() -> None:
    v, log = _veil(front=False)
    v.run(lambda: log.append("change"))
    assert log == ["setup", "change"] and v._phase is None and not v._root.pending, log


def test_failing_call() -> None:
    v, log = _veil()

    def broken(_hold: bool) -> None:
        raise OSError("no window")

    v._hold_drawing = broken
    v.run(lambda: log.append("change"))
    assert "change" in log and v._phase is None and not v._frozen, log


def test_off_outside_windows() -> None:
    if sys.platform != "win32":
        assert transitions.create(None, None, "#ffffff") is None


print("test_transitions.py")
for _name, _fn in list(globals().items()):
    if _name.startswith("test_") and callable(_fn):
        _check(_name, _fn)
if failures:
    print(f"{len(failures)} failure(s)")
    sys.exit(1)
print("All transition tests passed.")
