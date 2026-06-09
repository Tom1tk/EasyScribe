"""
live_transcriber.py - VAD-chunked live microphone transcription.

Uses Silero VAD to detect speech segments, then feeds each segment to a
sherpa-onnx OfflineRecognizer. Runs on a daemon thread; stops when
cancel_event is set.
"""

import logging
import queue
import threading
from typing import Callable

import numpy as np

import config

logger = logging.getLogger(__name__)


class LiveTranscriber:
    """
    Runs a Silero VAD → OfflineRecognizer pipeline on a background thread.

    Usage:
        lt = LiveTranscriber()
        lt.start(recognizer, mic_queue, stop_event, on_segment)
        stop_event.set()   # signal from recording worker
        lt.stop()          # join thread
    """

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None

    def start(
        self,
        recognizer,
        mic_queue: queue.Queue,
        cancel_event: threading.Event,
        on_segment: Callable[[str], None],
    ) -> None:
        """Launch the VAD loop on a daemon thread."""
        self._thread = threading.Thread(
            target=self._loop,
            args=(recognizer, mic_queue, cancel_event, on_segment),
            daemon=True,
            name="LiveTranscriber",
        )
        self._thread.start()
        logger.info("LiveTranscriber started")

    def stop(self) -> None:
        """Join the VAD loop thread (cancel_event must already be set)."""
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
            logger.info("LiveTranscriber stopped")

    def _loop(
        self,
        recognizer,
        mic_queue: queue.Queue,
        cancel_event: threading.Event,
        on_segment: Callable[[str], None],
    ) -> None:
        import sherpa_onnx

        # config.VAD_MODEL_PATH = BASE_DIR / "models" / "silero_vad.onnx"
        # sherpa_onnx.get_default_vad_model() does not exist in v1.13.2 — model bundled explicitly
        vad_config = sherpa_onnx.VadModelConfig(
            silero_vad=sherpa_onnx.SileroVadModelConfig(
                model=str(config.VAD_MODEL_PATH),
                threshold=0.5,
                min_silence_duration=0.5,
                min_speech_duration=0.25,
            ),
            sample_rate=config.VAD_SAMPLE_RATE,
        )
        vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=30)
        logger.debug("VAD initialised")

        while not cancel_event.is_set():
            try:
                chunk: np.ndarray = mic_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            vad.accept_waveform(chunk)

            while not vad.empty():
                segment = vad.front
                samples = segment.samples

                stream = recognizer.create_stream()
                stream.accept_waveform(config.VAD_SAMPLE_RATE, samples)
                recognizer.decode_stream(stream)

                text = stream.result.text.strip()
                if text:
                    on_segment(text)

                vad.pop()
