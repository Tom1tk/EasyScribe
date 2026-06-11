"""
transcriber.py - sherpa-onnx transcription engine for EasyScribe v2.0.

Replaces faster-whisper. Uses Whisper ONNX large-v3-turbo. Inference runs on
CPU only — sherpa-onnx 1.13.2 has no Vulkan provider (see CLAUDE.md Rule 7).
"""

import logging
import threading
import time
import wave
from pathlib import Path
from typing import Callable

import numpy as np

import config
import vad
import whispercpp_wrapper
from config import TEMP_DIR
from vad import CancelledError

logger = logging.getLogger(__name__)


# ─── Custom exceptions ────────────────────────────────────────────────────────


class ModelNotFoundError(RuntimeError):
    """Raised when model files are missing."""


class TranscriptionError(RuntimeError):
    """Raised when transcription fails for reasons other than cancellation."""


# ─── Transcript formatting helpers ───────────────────────────────────────────


def _format_hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


_BLOCK_GAP_SEC: float = 2.0


def _build_plain_transcript(
    raw_segments: list[tuple[float, float, str]],
    add_timestamps: bool,
) -> str:
    if not raw_segments:
        return ""

    if not add_timestamps:
        return " ".join(text for _, _, text in raw_segments)

    blocks: list[tuple[float, list[str]]] = []
    block_start = raw_segments[0][0]
    block_texts: list[str] = []
    prev_end = 0.0

    for start, end, text in raw_segments:
        if block_texts and (start - prev_end) >= _BLOCK_GAP_SEC:
            blocks.append((block_start, block_texts))
            block_start = start
            block_texts = []
        block_texts.append(text)
        prev_end = end

    if block_texts:
        blocks.append((block_start, block_texts))

    return "\n\n".join(f"[{_format_hms(ts)}]\n" + " ".join(texts) for ts, texts in blocks)


def _build_diarized_transcript(
    assigned: list[tuple[str, str, float, float]],
    speaker_map: dict[str, str],
    add_timestamps: bool,
) -> str:
    if not assigned:
        return ""

    parts: list[str] = []

    if not add_timestamps:
        block_label = speaker_map.get(assigned[0][0], assigned[0][0])
        block_texts: list[str] = [assigned[0][1]]

        for speaker_raw, text, _s, _e in assigned[1:]:
            label = speaker_map.get(speaker_raw, speaker_raw)
            if label == block_label:
                block_texts.append(text)
            else:
                parts.append(f"[{block_label}]\n" + " ".join(block_texts))
                block_label = label
                block_texts = [text]
        parts.append(f"[{block_label}]\n" + " ".join(block_texts))

    else:
        block_label = speaker_map.get(assigned[0][0], assigned[0][0])
        block_start = assigned[0][2]
        block_texts = [assigned[0][1]]
        prev_end = assigned[0][3]

        for speaker_raw, text, start, end in assigned[1:]:
            label = speaker_map.get(speaker_raw, speaker_raw)
            new_block = (label != block_label) or (start - prev_end) >= _BLOCK_GAP_SEC
            if new_block:
                parts.append(
                    f"[{_format_hms(block_start)}] [{block_label}]\n"
                    + " ".join(block_texts)
                )
                block_label = label
                block_start = start
                block_texts = [text]
            else:
                block_texts.append(text)
            prev_end = end

        parts.append(
            f"[{_format_hms(block_start)}] [{block_label}]\n" + " ".join(block_texts)
        )

    return "\n\n".join(parts)


def _extract_speaker_clip(
    audio_path: Path,
    turns: list[tuple[float, float, str]],
    speaker: str,
    out_path: Path,
    max_duration: float = 5.0,
) -> bool:
    best_start = best_end = None
    best_dur = 0.0
    for t_start, t_end, t_speaker in turns:
        if t_speaker == speaker:
            dur = t_end - t_start
            if dur > best_dur:
                best_dur = dur
                best_start, best_end = t_start, t_end

    if best_start is None:
        return False

    clip_end = min(best_end, best_start + max_duration)  # type: ignore[arg-type]
    try:
        with wave.open(str(audio_path), "rb") as wf:
            n_ch, s_w, sr = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
            wf.setpos(int(best_start * sr))
            raw = wf.readframes(int((clip_end - best_start) * sr))
        with wave.open(str(out_path), "wb") as wf_out:
            wf_out.setnchannels(n_ch)
            wf_out.setsampwidth(s_w)
            wf_out.setframerate(sr)
            wf_out.writeframes(raw)
        return True
    except Exception as exc:
        logger.warning(f"Could not extract speaker clip for {speaker}: {exc}")
        return False


def _fmt_seconds(s: float) -> str:
    h, m, sec = int(s // 3600), int((s % 3600) // 60), int(s % 60)
    return f"{h}h {m:02d}m {sec:02d}s" if h else f"{m}m {sec:02d}s"


# ─── Model / recognizer config ────────────────────────────────────────────────


def validate_model_directory() -> list[str]:
    """Check that required model files exist. Returns list of error strings."""
    errors: list[str] = []

    if not config.VAD_MODEL_PATH.is_file():
        errors.append(f"Missing: {config.VAD_MODEL_PATH}")

    for p in (config.WHISPER_ENCODER, config.WHISPER_DECODER, config.WHISPER_TOKENS):
        if not p.is_file():
            errors.append(f"Missing: {p}")

    return errors


def _build_recognizer_config(provider: str):
    import sherpa_onnx

    model_cfg = sherpa_onnx.OfflineModelConfig(
        whisper=sherpa_onnx.OfflineWhisperModelConfig(
            encoder=str(config.WHISPER_ENCODER),
            decoder=str(config.WHISPER_DECODER),
            language="en",
            task="transcribe",
        ),
        tokens=str(config.WHISPER_TOKENS),
        provider=provider,
        num_threads=config.NUM_THREADS,
    )

    return sherpa_onnx.OfflineRecognizerConfig(model_config=model_cfg)


# ─── Audio loading ────────────────────────────────────────────────────────────


def _load_wav_float32(wav_path: Path) -> tuple[np.ndarray, int]:
    """Load a WAV file and return (float32_samples, sample_rate)."""
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


# ─── Engine ───────────────────────────────────────────────────────────────────


class TranscriptionEngine:
    """Lazy-loading wrapper around sherpa_onnx.OfflineRecognizer."""

    def __init__(self) -> None:
        self._recognizer = None
        self._lock = threading.Lock()
        self._provider: str | None = None

    def _ensure_model_loaded(self, status_callback: Callable[[str], None]) -> None:
        with self._lock:
            if self._recognizer is not None:
                return

            missing = validate_model_directory()
            if missing:
                raise ModelNotFoundError(
                    "Model validation failed:\n" + "\n".join(missing)
                )

            status_callback("Loading Model")
            logger.info("Loading model, provider=cpu")

            from sherpa_onnx.lib._sherpa_onnx import OfflineRecognizer as _OfflineRecognizer

            try:
                cfg = _build_recognizer_config("cpu")
                self._recognizer = _OfflineRecognizer(cfg)
                self._provider = "cpu"
                logger.info("Model loaded: provider=cpu")
            except RuntimeError as exc:
                raise ModelNotFoundError(f"Could not load model: {exc}") from exc

    def unload_model(self) -> None:
        with self._lock:
            self._recognizer = None
            self._provider = None
            logger.info("Model unloaded")

    def get_recognizer(self, status_callback: Callable[[str], None]):
        """Public accessor for the loaded recognizer (loads on demand)."""
        self._ensure_model_loaded(status_callback)
        return self._recognizer

    def transcribe(
        self,
        audio_path: Path,
        output_path: Path,
        add_timestamps: bool,
        cancel_event: threading.Event,
        status_callback: Callable[[str], None],
        progress_callback: Callable[[float], None],
        log_callback: Callable[[str], None],
        diarize: bool = False,
        speaker_name_callback: Callable[[dict, dict], None] | None = None,
    ) -> None:
        if cancel_event.is_set():
            raise CancelledError("Cancelled before transcription started")

        status_callback("Transcribing")
        logger.info(f"Transcribing: {audio_path.name} → {output_path.name}")
        log_callback(f"[Transcribe] Starting: {audio_path.name}")

        try:
            samples, sr = _load_wav_float32(audio_path)
        except Exception as exc:
            raise TranscriptionError(f"Could not load audio: {exc}") from exc

        duration = len(samples) / sr
        log_callback(f"[Transcribe] Audio duration: {_fmt_seconds(duration)}")

        infer_start = time.monotonic()
        raw_segments: list[tuple[float, float, str]]

        if whispercpp_wrapper.is_available():
            log_callback(
                f"[Transcribe] engine: whisper.cpp (beam_size={config.WHISPERCPP_BEAM_SIZE}), "
                f"timestamps: {'on' if add_timestamps else 'off'}, "
                f"speakers: {'on' if diarize else 'off'}"
            )
            raw_segments = whispercpp_wrapper.transcribe_file(
                audio_path, cancel_event, log_callback, progress_callback
            )
            for _seg_start, _seg_end, text in raw_segments:
                log_callback(text)
        else:
            self._ensure_model_loaded(status_callback)
            log_callback(
                f"[Transcribe] engine: sherpa-onnx (provider={self._provider}), "
                f"timestamps: {'on' if add_timestamps else 'off'}, "
                f"speakers: {'on' if diarize else 'off'}"
            )

            log_callback("[Transcribe] Detecting speech segments (VAD)…")
            vad_detector = vad.build_vad()
            bounds = vad.file_segments(vad_detector, samples, cancel_event)
            padded_bounds = vad.pad_and_clamp(bounds, len(samples))
            num_chunks = len(padded_bounds)
            log_callback(f"[Transcribe] Processing {num_chunks} speech segment(s)…")

            raw_segments = []
            for i, (c_start, c_end) in enumerate(padded_bounds):
                if cancel_event.is_set():
                    raise CancelledError("Transcription cancelled by user")

                seg_start = c_start / sr
                seg_end = c_end / sr

                stream = self._recognizer.create_stream()
                stream.accept_waveform(sr, samples[c_start:c_end])
                self._recognizer.decode_stream(stream)

                text = stream.result.text.strip()
                if text:
                    raw_segments.append((seg_start, seg_end, text))
                    log_callback(text)

                progress_callback((i + 1) / num_chunks)
                elapsed = time.monotonic() - infer_start
                speed = seg_end / elapsed if elapsed > 0 else 0
                log_callback(
                    f"[Progress] segment {i+1}/{num_chunks} — "
                    f"{_fmt_seconds(seg_end)} / {_fmt_seconds(duration)} — "
                    f"{speed:.1f}x realtime"
                )

        segment_count = len(raw_segments)
        word_count = sum(len(text.split()) for _, _, text in raw_segments)
        total_elapsed = time.monotonic() - infer_start
        log_callback(
            f"[Transcribe] Complete — {segment_count} segments, "
            f"{word_count} words in {_fmt_seconds(total_elapsed)}"
        )

        # ── Optional speaker diarization ──────────────────────────────────────
        if diarize and raw_segments:
            status_callback("Identifying Speakers")
            try:
                from diarizer import DiarizationEngine
                _diarizer = DiarizationEngine()
                turns = _diarizer.diarize(audio_path, log_callback)
                assigned = _diarizer.assign_speakers(raw_segments, turns)
                speaker_map: dict[str, str] = {}
                for _ts, _te, spk in turns:
                    if spk not in speaker_map:
                        speaker_map[spk] = f"Speaker {len(speaker_map) + 1}"

                if speaker_name_callback is not None:
                    status_callback("Naming Speakers")
                    clips_dict: dict[str, Path] = {}
                    for raw_spk in speaker_map:
                        clip_path = TEMP_DIR / f"spk_clip_{raw_spk}.wav"
                        if _extract_speaker_clip(audio_path, turns, raw_spk, clip_path):
                            clips_dict[raw_spk] = clip_path
                    speaker_name_callback(speaker_map, clips_dict)
                    for p in clips_dict.values():
                        try:
                            p.unlink(missing_ok=True)
                        except OSError:
                            pass

                transcript_text = _build_diarized_transcript(assigned, speaker_map, add_timestamps)
            except Exception as exc:
                logger.warning(f"Diarization failed, falling back to plain: {exc}")
                log_callback(f"[Diarize] Warning: {exc} — writing plain transcript")
                transcript_text = _build_plain_transcript(raw_segments, add_timestamps)
        else:
            transcript_text = _build_plain_transcript(raw_segments, add_timestamps)

        # ── Write transcript ──────────────────────────────────────────────────
        status_callback("Writing Transcript")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        log_callback(f"[Output] Writing transcript to: {output_path}")
        logger.info(f"Writing transcript: {output_path}")

        try:
            with open(output_path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(transcript_text)
                fh.write("\n")
        except PermissionError as exc:
            raise TranscriptionError(
                f"Cannot write to output file.\n"
                f"The file may be open in another program: {output_path}\n\n{exc}"
            ) from exc
        except OSError as exc:
            raise TranscriptionError(
                f"Failed to write transcript: {output_path}\n\n{exc}"
            ) from exc

        size_kb = output_path.stat().st_size // 1024
        progress_callback(1.0)
        log_callback(f"[Done] Saved: {output_path}  ({size_kb} KB)")
