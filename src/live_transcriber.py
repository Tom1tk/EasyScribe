"""
live_transcriber.py - VAD-chunked live microphone transcription.

Runs two daemon threads:
- VAD thread: feeds mic audio through Silero VAD, applies backward-only
  onset padding (vad.pad_live_segment), and pushes detected segments onto a
  bounded decode queue.
- Decode thread: pulls segments off the queue and runs the sherpa-onnx
  recognizer, so a slow decode never blocks VAD/capture.

When cancel_event is set, the VAD thread flushes the speech that is still
in progress, and the decode thread finishes the queue before it stops.
"""

import logging
import queue
import threading
import time
from typing import Callable

import numpy as np

import config
import vad

logger = logging.getLogger(__name__)

_DECODE_QUEUE_MAXSIZE = 8

# Upper limit for Stop to wait while the last queued segments are decoded.
_STOP_TIMEOUT_SEC = 60.0

# Rolling history buffer for backward-only onset padding: must cover at least
# vad.PAD_SAMPLES plus a margin for the VAD's own detection lag.
_HISTORY_SAMPLES = vad.PAD_SAMPLES + 16 * config.VAD_CHUNK_SAMPLES


class LiveTranscriber:
    """
    Runs a Silero VAD -> bounded decode queue -> OfflineRecognizer pipeline
    on two background threads.

    Usage:
        lt = LiveTranscriber()
        lt.start(recognizer, mic_queue, stop_event, on_segment)
        stop_event.set()   # signal from recording worker
        lt.stop()          # finish the last words, then join threads
    """

    def __init__(self) -> None:
        self._vad_thread: threading.Thread | None = None
        self._decode_thread: threading.Thread | None = None

    def start(
        self,
        recognizer,
        mic_queue: queue.Queue,
        cancel_event: threading.Event,
        on_segment: Callable[[str], None],
        on_overflow: Callable[[], None] | None = None,
    ) -> None:
        """Launch the VAD and decode loops on daemon threads."""
        decode_queue: queue.Queue = queue.Queue(maxsize=_DECODE_QUEUE_MAXSIZE)
        # Set by the VAD thread when it has pushed its last segment. The
        # decode thread runs until then and the queue is empty, so speech
        # that was still in progress at Stop gets into the transcript.
        vad_done = threading.Event()

        # sherpa-onnx's VoiceActivityDetector and OfflineRecognizer share the
        # same onnxruntime environment; concurrent Run() calls from
        # different threads can trigger a fatal (uncatchable) ScatterND/
        # GetElementType abort. Serialize all onnxruntime calls with this lock.
        onnx_lock = threading.Lock()

        self._decode_thread = threading.Thread(
            target=self._decode_loop,
            args=(recognizer, decode_queue, vad_done, on_segment, onnx_lock),
            daemon=True,
            name="LiveDecoder",
        )
        self._vad_thread = threading.Thread(
            target=self._vad_loop,
            args=(mic_queue, decode_queue, cancel_event, vad_done, on_overflow, onnx_lock),
            daemon=True,
            name="LiveVAD",
        )
        self._decode_thread.start()
        self._vad_thread.start()
        logger.info("LiveTranscriber started")

    def stop(self, timeout: float = _STOP_TIMEOUT_SEC) -> None:
        """
        Join the VAD and decode threads (cancel_event must already be set).

        This waits while the last queued segments are decoded, up to
        *timeout* seconds in total.
        """
        deadline = time.monotonic() + timeout
        for attr in ("_vad_thread", "_decode_thread"):
            thread = getattr(self, attr)
            if thread is not None:
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
                if thread.is_alive():
                    logger.warning(f"{thread.name} did not finish within {timeout:.0f} s")
                setattr(self, attr, None)
        logger.info("LiveTranscriber stopped")

    @staticmethod
    def _vad_loop(
        mic_queue: queue.Queue,
        decode_queue: queue.Queue,
        cancel_event: threading.Event,
        vad_done: threading.Event,
        on_overflow: Callable[[], None] | None,
        onnx_lock: threading.Lock,
    ) -> None:
        try:
            detector = vad.build_vad(buffer_size_in_seconds=30)
            logger.debug("VAD initialised")

            history = np.zeros(0, dtype=np.float32)
            history_start = 0
            prev_segment_end = 0
            overflow_notified = False

            def push_ready_segments() -> None:
                """Move each finished VAD segment onto the decode queue."""
                nonlocal prev_segment_end, overflow_notified
                while not detector.empty():
                    segment = detector.front
                    seg_start = segment.start
                    samples = vad.pad_live_segment(
                        history, history_start, prev_segment_end, seg_start, segment.samples
                    )
                    prev_segment_end = seg_start + len(segment.samples)

                    if not _put_dropping_oldest(decode_queue, samples) and not overflow_notified:
                        overflow_notified = True
                        logger.warning("Live decode queue full — dropped oldest segment")
                        if on_overflow is not None:
                            on_overflow()

                    detector.pop()

            def feed(chunk: np.ndarray) -> None:
                nonlocal history, history_start
                history = np.concatenate([history, chunk])
                if len(history) > _HISTORY_SAMPLES:
                    trim = len(history) - _HISTORY_SAMPLES
                    history = history[trim:]
                    history_start += trim

                with onnx_lock:
                    detector.accept_waveform(chunk)
                    push_ready_segments()

            while not cancel_event.is_set():
                try:
                    chunk: np.ndarray = mic_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                feed(chunk)

            # Stop was pressed. Use the audio that is already captured, then
            # flush the VAD so a sentence still in progress becomes a segment.
            while True:
                try:
                    feed(mic_queue.get_nowait())
                except queue.Empty:
                    break
            with onnx_lock:
                detector.flush()
                push_ready_segments()
        except Exception:
            logger.exception("Live VAD thread failed")
        finally:
            vad_done.set()

    @staticmethod
    def _decode_loop(
        recognizer,
        decode_queue: queue.Queue,
        vad_done: threading.Event,
        on_segment: Callable[[str], None],
        onnx_lock: threading.Lock,
    ) -> None:
        while True:
            try:
                samples = decode_queue.get(timeout=0.1)
            except queue.Empty:
                if vad_done.is_set():
                    return
                continue

            with onnx_lock:
                stream = recognizer.create_stream()
                stream.accept_waveform(config.VAD_SAMPLE_RATE, samples)
                recognizer.decode_stream(stream)
                text = stream.result.text.strip()

            if text:
                on_segment(text)


def _put_dropping_oldest(q: queue.Queue, item: np.ndarray) -> bool:
    """
    Put `item` on `q`. If full, drop the oldest item to make room.
    Returns False if an item had to be dropped, True otherwise.
    """
    try:
        q.put_nowait(item)
        return True
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass
        return False
