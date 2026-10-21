"""
diarizer.py - Optional speaker diarization using sherpa-onnx.

Diarization is run *after* transcription on the same mono 16 kHz WAV file,
in a separate process: sherpa-onnx ignores a request to stop, so Cancel ends
the process instead, and a crash or hang in the native code cannot freeze the
window.
Each transcript segment is assigned to the speaker whose sherpa-onnx turn has
the greatest time overlap with that segment.

Usage
-----
engine = DiarizationEngine()
if engine.is_available():
    turns = engine.diarize(audio_path, log_callback, cancel_event, progress_callback)
    segments = engine.assign_speakers(collected_segments, turns)
    # segments: list of (speaker_label, text, start, end)
"""

import logging
import multiprocessing
import threading
from pathlib import Path
from typing import Callable

import config
from common import CancelledError, fmt_duration, load_wav_float32, wav_duration

logger = logging.getLogger(__name__)

# Seconds between checks for Cancel and for the child process ending.
_POLL_SECONDS = 0.2

# Time model, measured on CPU with 4 threads (tests/data, 5 to 30 min): finding
# speech takes ~0.045 s per second of audio, telling voices apart ~0.165 s per
# second of speech, plus a few seconds to start. Each PC's own speed is learned
# from its last run (settings.save_speaker_speed).
_SEC_PER_AUDIO_SEC = 0.045
_SEC_PER_SPEECH_SEC = 0.165
_START_SECONDS = 3.0


def speech_seconds(segments: list[tuple[float, float, str]]) -> float:
    """Seconds covered by the transcript lines (overlaps counted once)."""
    total, reach = 0.0, 0.0
    for start, end, _text in sorted(segments):
        start = max(start, reach)
        if end > start:
            total += end - start
            reach = end
    return total


def _model_seconds(audio_sec: float, speech_sec: float | None) -> float:
    speech = audio_sec if speech_sec is None else min(speech_sec, audio_sec)
    return _START_SECONDS + _SEC_PER_AUDIO_SEC * audio_sec + _SEC_PER_SPEECH_SEC * speech


def estimate_seconds(audio_sec: float, speech_sec: float | None = None) -> float:
    """Rough time to find the speakers in *audio_sec* seconds of audio.

    *speech_sec* (from the transcript) makes it closer; without it all the
    audio is taken as speech.
    """
    from settings import load_speaker_speed

    return _model_seconds(audio_sec, speech_sec) * (load_speaker_speed() or 1.0)


def save_measured_speed(
    audio_path: Path, segments: list[tuple[float, float, str]], took_sec: float
) -> None:
    """Learn this PC's speed from a finished run, for the next estimate."""
    from settings import load_speaker_speed, save_speaker_speed

    try:
        expected = _model_seconds(wav_duration(audio_path), speech_seconds(segments))
    except Exception:
        return
    factor = min(5.0, max(0.2, took_sec / expected))
    previous = load_speaker_speed()
    save_speaker_speed(factor if previous is None else (previous + factor) / 2)


def fmt_estimate(seconds: float) -> str:
    """"About 4 minutes" style text, rounded the way a person would say it."""
    if seconds < 50:
        return "less than a minute"
    minutes = round(seconds / 60)
    if minutes <= 1:
        return "about 1 minute"
    if minutes < 60:
        return f"about {minutes} minutes"
    hours, rest = divmod(minutes, 60)
    rest = 5 * round(rest / 5)
    if rest == 60:
        hours, rest = hours + 1, 0
    hours_text = "1 hour" if hours == 1 else f"{hours} hours"
    return f"about {hours_text}" + (f" {rest} minutes" if rest else "")


class DiarizationError(RuntimeError):
    """Raised when diarization fails."""


class DiarizationEngine:
    """Runs the sherpa-onnx offline speaker diarization pipeline in a child process."""

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """
        Return True if the bundled sherpa-onnx diarization models are present
        on disk. Does NOT import sherpa_onnx — safe to call at startup.
        """
        from config import DIARIZATION_SEGMENTATION_MODEL, DIARIZATION_EMBEDDING_MODEL
        available = (
            DIARIZATION_SEGMENTATION_MODEL.is_file()
            and DIARIZATION_EMBEDDING_MODEL.is_file()
        )
        if not available:
            logger.debug(
                f"Diarization models not found at {DIARIZATION_SEGMENTATION_MODEL} / "
                f"{DIARIZATION_EMBEDDING_MODEL} (checkbox will be disabled)"
            )
        return available

    def diarize(
        self,
        audio_path: Path,
        log_callback: Callable[[str], None],
        cancel_event: threading.Event | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[tuple[float, float, str]]:
        """
        Run speaker diarization on *audio_path*.

        Returns a list of (start_sec, end_sec, speaker_label) tuples sorted
        by start time.  Speaker labels are strings like "SPEAKER_00".

        Raises CancelledError when *cancel_event* is set (the child process is
        ended), and DiarizationError on failure.
        """
        if not self.is_available():
            raise DiarizationError("The speaker models are missing.")

        try:
            audio_sec = wav_duration(audio_path)
        except Exception as exc:
            raise DiarizationError(f"Could not load audio for diarization: {exc}") from exc

        est_str = fmt_estimate(estimate_seconds(audio_sec))
        log_callback(
            f"[Diarize] Estimated wait: {est_str} for {fmt_duration(audio_sec)} of audio "
            "(rough estimate). Please wait…"
        )
        logger.info(f"Running diarization on: {audio_path.name} (estimate {est_str})")
        # The first ~20% of the time (finding speech) reports no progress.
        log_callback("[Diarize] Finding the parts with speech…")

        ctx = multiprocessing.get_context("spawn")
        recv_conn, send_conn = ctx.Pipe(duplex=False)
        proc = ctx.Process(
            target=_run_pipeline,
            args=(str(audio_path), config.NUM_THREADS, send_conn),
            daemon=True,
        )
        proc.start()
        send_conn.close()  # the child holds its own copy; EOF once it exits

        try:
            turns = _wait_for_turns(proc, recv_conn, cancel_event, log_callback, progress_callback)
        finally:
            _stop_process(proc)
            recv_conn.close()

        speaker_set = {t[2] for t in turns}
        log_callback(
            f"[Diarize] Found {len(speaker_set)} speaker(s) across {len(turns)} turn(s)"
        )
        logger.info(f"Diarization complete: {len(speaker_set)} speakers, {len(turns)} turns")
        return turns

    @staticmethod
    def assign_speakers(
        segments: list[tuple[float, float, str]],
        turns: list[tuple[float, float, str]],
    ) -> list[tuple[str, str, float, float]]:
        """
        Map each transcript segment to the speaker with the most overlap.

        Parameters
        ----------
        segments : list of (start_sec, end_sec, text)
        turns    : list of (start_sec, end_sec, speaker_label)

        Returns
        -------
        list of (speaker_label, text, start_sec, end_sec)
        """
        result: list[tuple[str, str, float, float]] = []

        for seg_start, seg_end, text in segments:
            best_speaker = "SPEAKER_00"
            best_overlap = 0.0

            for turn_start, turn_end, speaker in turns:
                overlap = min(seg_end, turn_end) - max(seg_start, turn_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = speaker

            result.append((best_speaker, text, seg_start, seg_end))

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Parent side: talk to the child process
# ─────────────────────────────────────────────────────────────────────────────


def _wait_for_turns(
    proc: multiprocessing.process.BaseProcess,
    conn,  # type: ignore[no-untyped-def]
    cancel_event: threading.Event | None,
    log_callback: Callable[[str], None],
    progress_callback: Callable[[float], None] | None,
) -> list[tuple[float, float, str]]:
    """Relay progress from the child until it sends the turns, fails or is cancelled."""
    logged_step = -1
    while True:
        if cancel_event is not None and cancel_event.is_set():
            logger.info("Cancel requested — ending the diarization process")
            raise CancelledError("Speaker identification cancelled by user")

        if not conn.poll(_POLL_SECONDS):
            if not proc.is_alive() and not conn.poll():
                raise DiarizationError(
                    f"Speaker identification stopped unexpectedly (exit code {proc.exitcode})."
                )
            continue

        try:
            kind, value = conn.recv()
        except EOFError:
            raise DiarizationError(
                f"Speaker identification stopped unexpectedly (exit code {proc.exitcode})."
            ) from None

        if kind == "progress":
            if progress_callback:
                progress_callback(value / 100.0)
            if value // 10 > logged_step:  # one log line per 10%
                logged_step = value // 10
                log_callback(f"[Diarize] Progress: {value}%")
        elif kind == "done":
            return value
        elif kind == "error":
            logger.error(f"Diarization failed in child process: {value}")
            raise DiarizationError(f"Speaker diarization failed: {value}")


def _stop_process(proc: multiprocessing.process.BaseProcess) -> None:
    """End the child process if it still runs, and reap it."""
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
    proc.join(timeout=5)


# ─────────────────────────────────────────────────────────────────────────────
# Child side: runs in the separate process
# ─────────────────────────────────────────────────────────────────────────────


def _build_pipeline(num_threads: int):  # type: ignore[no-untyped-def]
    from config import DIARIZATION_SEGMENTATION_MODEL, DIARIZATION_EMBEDDING_MODEL

    import sherpa_onnx

    # Diarization runs on CPU. sherpa-onnx has no Vulkan provider, and a
    # CUDA provider would need the separate CUDA DLL stack that v2.0
    # removed (CLAUDE.md Rules 6 and 7).
    seg_cfg = sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
        pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
            model=str(DIARIZATION_SEGMENTATION_MODEL)
        ),
        num_threads=num_threads,
        provider="cpu",
    )
    emb_cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(DIARIZATION_EMBEDDING_MODEL),
        num_threads=num_threads,
        provider="cpu",
    )
    clu_cfg = sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=0.5)
    diar_cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=seg_cfg, embedding=emb_cfg, clustering=clu_cfg
    )
    return sherpa_onnx.OfflineSpeakerDiarization(diar_cfg)


def _run_pipeline(audio_path: str, num_threads: int, conn) -> None:  # type: ignore[no-untyped-def]
    """
    Child process entry point. Sends ("progress", percent) messages, then
    ("done", turns) or ("error", message), over *conn*.
    """
    try:
        pipeline = _build_pipeline(num_threads)
        samples, _sr = load_wav_float32(Path(audio_path))
        last_pct = [-1]

        def _progress(num_done: int, num_total: int) -> int:
            pct = int(100 * num_done / num_total) if num_total > 0 else 0
            if pct != last_pct[0]:  # ~100 messages instead of one per chunk
                last_pct[0] = pct
                conn.send(("progress", pct))
            return 0  # sherpa-onnx ignores other values; the parent ends the process instead

        result = pipeline.process(samples, callback=_progress)
        turns = [
            (seg.start, seg.end, f"SPEAKER_{seg.speaker:02d}")
            for seg in result.sort_by_start_time()
        ]
        conn.send(("done", turns))
    except Exception as exc:
        conn.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()
