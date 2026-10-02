"""
mic_recorder.py - Microphone capture with crash-safe PCM writing.

MicRecorder opens a sounddevice InputStream at 16 kHz mono int16.
Audio chunks are placed on a float32 queue for the VAD loop, and
simultaneously written as raw PCM to disk with a JSON sidecar so
that a recording can be recovered if the process is killed.
"""

import json
import logging
import os
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from recovery import recover_pcm_to_wav

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000
_CHANNELS = 1
_CHUNK_SAMPLES = 512
_FSYNC_EVERY = 50  # flush and fsync after this many chunks

# Memory backstop for the VAD queue: ~60s of audio. The VAD loop is fast and
# should never approach this; if it does, drop oldest chunks rather than grow
# unbounded (the PCM writer path is unaffected — the saved recording is complete).
_VAD_QUEUE_MAXSIZE = 60 * _SAMPLE_RATE // _CHUNK_SAMPLES


class MicRecorder:
    """
    Captures microphone audio and writes crash-safe raw PCM to disk.

    get_queue() returns a Queue of float32 numpy arrays (16 kHz mono, 512 samples each).
    The PCM writer runs on a daemon thread and survives process kill — recovery.py
    handles the orphaned files on next launch.
    """

    def __init__(self) -> None:
        self._stream = None
        self._vad_queue: queue.Queue = queue.Queue(maxsize=_VAD_QUEUE_MAXSIZE)
        self._pcm_queue: queue.Queue = queue.Queue()
        self._writer_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pcm_path: Optional[Path] = None
        self._json_path: Optional[Path] = None
        self._samples_written: int = 0
        self._vad_drop_logged = False
        self._feed_vad = True
        # Loudest sample since the last get_level() call, 0.0 to 1.0
        self._peak: float = 0.0

    @staticmethod
    def list_devices() -> list[dict]:
        """Return available input devices as [{"index": int, "name": str}, ...]."""
        import sounddevice as sd
        result = []
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                result.append({"index": i, "name": d["name"]})
        return result

    def get_queue(self) -> queue.Queue:
        """Return the float32 audio queue consumed by the VAD loop."""
        return self._vad_queue

    def get_session_stem(self) -> Optional[str]:
        """Return this session's filename stem (e.g. "recording_20260611_120000"),
        or None if start() has not been called."""
        return self._pcm_path.stem if self._pcm_path is not None else None

    def get_level(self) -> float:
        """Return the loudest sample since the last call (0.0 to 1.0), then reset it."""
        peak, self._peak = self._peak, 0.0
        return peak

    def start(self, device_index: Optional[int], output_dir: Path, feed_vad: bool = True) -> None:
        """Start capture and crash-safe PCM writing.

        feed_vad=False records only: nothing reads the VAD queue, so no audio
        is put on it.
        """
        import sounddevice as sd

        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._pcm_path = output_dir / f"recording_{ts}.pcm"
        self._json_path = output_dir / f"recording_{ts}.json"
        self._samples_written = 0
        self._vad_drop_logged = False
        self._feed_vad = feed_vad
        self._peak = 0.0
        self._stop_event.clear()

        self._writer_thread = threading.Thread(
            target=self._pcm_writer_loop,
            daemon=True,
            name="PCMWriter",
        )
        self._writer_thread.start()

        self._stream = sd.InputStream(
            samplerate=_SAMPLE_RATE,
            channels=_CHANNELS,
            dtype="int16",
            blocksize=_CHUNK_SAMPLES,
            device=device_index,
            callback=self._audio_callback,
        )
        self._stream.start()
        logger.info(f"MicRecorder started: {self._pcm_path.name}")

    def stop(self, convert_to_wav: bool = True) -> Optional[Path]:
        """
        Stop capture. If convert_to_wav=True, convert PCM → WAV and delete raw files.
        Returns WAV path on success, None on failure or if convert_to_wav=False.
        """
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        self._stop_event.set()
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=5)
            self._writer_thread = None

        if not convert_to_wav or self._pcm_path is None:
            return None

        wav_path = self._pcm_path.with_suffix(".wav")
        metadata = {
            "sample_rate": _SAMPLE_RATE,
            "channels": _CHANNELS,
            "bit_depth": 16,
            "samples_written": self._samples_written,
        }
        try:
            recover_pcm_to_wav(self._pcm_path, metadata, wav_path)
            self._pcm_path.unlink(missing_ok=True)
            if self._json_path:
                self._json_path.unlink(missing_ok=True)
            logger.info(f"Recording saved: {wav_path.name}")
            return wav_path
        except Exception as exc:
            logger.error(f"PCM→WAV conversion failed: {exc}")
            return None

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            logger.warning(f"sounddevice status: {status}")
        chunk = indata[:, 0].copy()  # shape: (frames,), dtype int16
        if chunk.size:
            self._peak = max(self._peak, float(np.abs(chunk.astype(np.int32)).max()) / 32768.0)
        if self._feed_vad:
            try:
                self._vad_queue.put_nowait(chunk.astype(np.float32) / 32768.0)
            except queue.Full:
                # This runs for each ~30 ms chunk; log once, not hundreds of times.
                if not self._vad_drop_logged:
                    self._vad_drop_logged = True
                    logger.warning("VAD queue full — dropping audio from live transcription")
        self._pcm_queue.put(chunk.tobytes())

    def _pcm_writer_loop(self) -> None:
        if self._pcm_path is None:
            return
        chunk_count = 0
        try:
            with open(self._pcm_path, "ab") as pcm_file:
                while not self._stop_event.is_set() or not self._pcm_queue.empty():
                    try:
                        raw = self._pcm_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    pcm_file.write(raw)
                    self._samples_written += len(raw) // 2  # int16 = 2 bytes
                    chunk_count += 1
                    if chunk_count % _FSYNC_EVERY == 0:
                        pcm_file.flush()
                        os.fsync(pcm_file.fileno())
                        self._flush_json()

                pcm_file.flush()
                os.fsync(pcm_file.fileno())
            self._flush_json()
        except Exception as exc:
            logger.error(f"PCM writer error: {exc}")

    def _flush_json(self) -> None:
        if self._json_path is None:
            return
        data = {
            "sample_rate": _SAMPLE_RATE,
            "channels": _CHANNELS,
            "bit_depth": 16,
            "samples_written": self._samples_written,
        }
        try:
            self._json_path.write_text(json.dumps(data), encoding="utf-8")
        except Exception as exc:
            logger.warning(f"JSON sidecar write failed: {exc}")
