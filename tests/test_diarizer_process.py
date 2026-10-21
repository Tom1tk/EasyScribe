"""
Diarization runs in a child process: progress reaches the caller, and Cancel
ends the process quickly (sherpa-onnx itself ignores a request to stop).
Needs the diarization models in models/diarization/.
Run from project root: python tests/test_diarizer_process.py
"""
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
AUDIO = BASE_DIR / "tests" / "data" / "test_speech.wav"
sys.path.insert(0, str(BASE_DIR / "src"))

from common import CancelledError  # noqa: E402
from diarizer import DiarizationEngine  # noqa: E402


def _long_copy(dest: Path, repeats: int) -> None:
    """Write *AUDIO* repeated *repeats* times, so diarization takes a while."""
    with wave.open(str(AUDIO), "rb") as src:
        params = src.getparams()
        frames = src.readframes(src.getnframes())
    with wave.open(str(dest), "wb") as out:
        out.setparams(params)
        out.writeframes(frames * repeats)


def test_progress_reaches_caller() -> None:
    values: list[float] = []
    turns = DiarizationEngine().diarize(AUDIO, lambda _m: None, threading.Event(), values.append)
    assert turns, "no speaker turns"
    assert values, "no progress reported"
    assert values == sorted(values), "progress went backwards"
    assert values[-1] == 1.0, f"progress ended at {values[-1]}"
    print(f"progress: {len(values)} updates, {len(turns)} turns — OK")


def test_cancel_ends_process() -> None:
    cancel = threading.Event()
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "long.wav"
        _long_copy(wav, 20)  # ~8 min of audio, well over a minute to process

        def on_progress(_v: float) -> None:
            cancel.set()  # cancel as soon as the work is under way

        start = time.monotonic()
        try:
            DiarizationEngine().diarize(wav, lambda _m: None, cancel, on_progress)
        except CancelledError:
            pass
        else:
            raise AssertionError("diarize() finished instead of being cancelled")
        took = time.monotonic() - start
    # The first progress arrives after the speech-finding phase (~20% of the work).
    assert took < 60, f"cancel took {took:.0f}s"
    print(f"cancel: stopped after {took:.1f}s — OK")


def main() -> None:
    assert DiarizationEngine().is_available(), "Diarization models not found"
    test_progress_reaches_caller()
    test_cancel_ends_process()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
