"""
common.py - Small helpers that several modules share.

One cancel exception, one duration formatter and one WAV loader, so that the
copies in ffmpeg_wrapper, whispercpp_wrapper, vad, transcriber and diarizer
cannot drift apart.
"""

import wave
from pathlib import Path

import numpy as np


class CancelledError(RuntimeError):
    """Raised when the user stops a job. Every stage raises this same class."""


def fmt_duration(seconds: float) -> str:
    """Format *seconds* for the log: '4m 05s', or '1h 02m 03s'."""
    h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


def wav_duration(wav_path: Path) -> float:
    """Length of a WAV file in seconds, read from its header (no samples loaded)."""
    with wave.open(str(wav_path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def load_wav_float32(wav_path: Path) -> tuple[np.ndarray, int]:
    """Load a WAV file and return (mono float32 samples, sample_rate)."""
    with wave.open(str(wav_path), "rb") as wf:
        n_ch = wf.getnchannels()
        s_w = wf.getsampwidth()
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    dtype = np.int16 if s_w == 2 else (np.int32 if s_w == 4 else np.int8)
    scale = 32768.0 if s_w == 2 else (2147483648.0 if s_w == 4 else 128.0)
    samples = np.frombuffer(raw, dtype=dtype).astype(np.float32) / scale

    if n_ch > 1:
        samples = samples.reshape(-1, n_ch).mean(axis=1)

    return samples, sr
