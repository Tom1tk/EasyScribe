"""
vad.py - Shared Silero VAD configuration and segment-bounds helpers.

Used by transcriber.py (file transcription) and live_transcriber.py (live
microphone) so VAD tuning and padding logic can't drift between the two.

VAD-based segmentation replaces fixed-window chunking: sherpa-onnx's Whisper
decoder hard-rejects any chunk >= 30s, and a fixed-window chunker both hits
that limit and re-transcribes overlaps with no de-duplication, producing
garbled/duplicated text at chunk boundaries. VAD splits on natural
speech/silence boundaries instead, with no overlap to de-duplicate.
"""

import threading

import numpy as np

import config


class CancelledError(RuntimeError):
    """Raised when VAD segmentation is cancelled by the user."""


# ─── VAD tuning ────────────────────────────────────────────────────────────

VAD_THRESHOLD: float = 0.5
VAD_MIN_SILENCE_SEC: float = 0.5
VAD_MIN_SPEECH_SEC: float = 0.25

# Pad each segment by this much to avoid clipping soft/short onset words
# (e.g. "Of" in "Of course"): both sides in file mode (pad_and_clamp), the
# leading edge only in live mode (pad_live_segment). Must stay below half of
# VAD_MIN_SILENCE_SEC so padded neighbouring segments can never overlap.
VAD_PAD_SEC: float = 0.2

assert VAD_PAD_SEC < VAD_MIN_SILENCE_SEC / 2, (
    "VAD_PAD_SEC must stay below half of VAD_MIN_SILENCE_SEC so padding on "
    "adjacent segments can never overlap"
)

PAD_SAMPLES: int = int(VAD_PAD_SEC * config.VAD_SAMPLE_RATE)


def build_vad(buffer_size_in_seconds: int | None = None):
    """Construct a sherpa_onnx.VoiceActivityDetector with shared tuning."""
    import sherpa_onnx

    vad_config = sherpa_onnx.VadModelConfig(
        silero_vad=sherpa_onnx.SileroVadModelConfig(
            model=str(config.VAD_MODEL_PATH),
            threshold=VAD_THRESHOLD,
            min_silence_duration=VAD_MIN_SILENCE_SEC,
            min_speech_duration=VAD_MIN_SPEECH_SEC,
        ),
        sample_rate=config.VAD_SAMPLE_RATE,
    )
    if buffer_size_in_seconds is None:
        return sherpa_onnx.VoiceActivityDetector(vad_config)
    return sherpa_onnx.VoiceActivityDetector(
        vad_config, buffer_size_in_seconds=buffer_size_in_seconds
    )


def file_segments(
    vad_detector, samples: np.ndarray, cancel_event: threading.Event
) -> list[tuple[int, int]]:
    """
    Feed `samples` through `vad_detector` in VAD_CHUNK_SAMPLES windows.
    Returns (start_sample, end_sample) pairs for each detected speech segment.
    """
    window = config.VAD_CHUNK_SAMPLES
    bounds: list[tuple[int, int]] = []
    pos = 0
    while pos < len(samples):
        if cancel_event.is_set():
            raise CancelledError("VAD segmentation cancelled by user")
        end = min(pos + window, len(samples))
        chunk = samples[pos:end]
        if len(chunk) < window:
            chunk = np.pad(chunk, (0, window - len(chunk)))
        vad_detector.accept_waveform(chunk)
        pos = end
        while not vad_detector.empty():
            seg = vad_detector.front
            bounds.append((seg.start, seg.start + len(seg.samples)))
            vad_detector.pop()

    vad_detector.flush()
    while not vad_detector.empty():
        seg = vad_detector.front
        bounds.append((seg.start, seg.start + len(seg.samples)))
        vad_detector.pop()

    return bounds


def pad_and_clamp(bounds: list[tuple[int, int]], total_samples: int) -> list[tuple[int, int]]:
    """
    Expand each (start, end) segment by PAD_SAMPLES on both sides, clamped so
    padded segments never overlap each other or fall outside [0, total_samples].
    """
    padded: list[tuple[int, int]] = []
    for i, (b_start, b_end) in enumerate(bounds):
        prev_end = bounds[i - 1][1] if i > 0 else 0
        next_start = bounds[i + 1][0] if i + 1 < len(bounds) else total_samples
        c_start = max(prev_end, b_start - PAD_SAMPLES)
        c_end = min(next_start, b_end + PAD_SAMPLES)
        padded.append((c_start, c_end))
    return padded


def pad_live_segment(
    history: np.ndarray,
    history_start: int,
    prev_segment_end: int,
    seg_start: int,
    seg_samples: np.ndarray,
) -> np.ndarray:
    """
    Prepend up to PAD_SAMPLES of preceding audio from `history` (a rolling
    buffer of raw samples whose first element is absolute sample index
    `history_start`) to `seg_samples`, clamped so the padding never reaches
    back past `prev_segment_end` (the previous segment's end) or before
    `history_start` (the oldest buffered sample).
    """
    pad_start = max(prev_segment_end, seg_start - PAD_SAMPLES, history_start)
    if pad_start >= seg_start:
        return seg_samples
    lead_in = history[pad_start - history_start : seg_start - history_start]
    return np.concatenate([lead_in, seg_samples])
