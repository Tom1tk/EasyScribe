"""
diarizer.py - Optional speaker diarization using sherpa-onnx.

Diarization is run *after* transcription on the same mono 16 kHz WAV file.
Each transcript segment is assigned to the speaker whose sherpa-onnx turn has
the greatest time overlap with that segment.

Usage
-----
engine = DiarizationEngine()
if engine.is_available():
    turns = engine.diarize(audio_path, log_callback)
    segments = engine.assign_speakers(collected_segments, turns)
    # segments: list of (speaker_label, text, start, end)
"""

import logging
import sys
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


class DiarizationError(RuntimeError):
    """Raised when diarization fails."""


class DiarizationEngine:
    """Lazy-loading wrapper around a sherpa-onnx offline speaker diarization pipeline."""

    def __init__(self) -> None:
        self._pipeline = None
        self._pipeline_on_gpu: bool = False

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
    ) -> list[tuple[float, float, str]]:
        """
        Run speaker diarization on *audio_path*.

        Returns a list of (start_sec, end_sec, speaker_label) tuples sorted
        by start time.  Speaker labels are strings like "SPEAKER_00".

        Raises DiarizationError on failure.
        """
        log_callback("[Diarize] Loading speaker diarization pipeline…")
        self._ensure_pipeline_loaded(log_callback)

        log_callback("[Diarize] Running speaker identification…")
        logger.info(f"Running diarization on: {audio_path.name}")

        try:
            import wave as _wave
            import numpy as _np
            with _wave.open(str(audio_path), "rb") as _wf:
                _nch = _wf.getnchannels()
                _sw = _wf.getsampwidth()
                _sr = _wf.getframerate()
                _raw = _wf.readframes(_wf.getnframes())
            _dtype = _np.int16 if _sw == 2 else (_np.int32 if _sw == 4 else _np.int8)
            _scale = 32768.0 if _sw == 2 else (2147483648.0 if _sw == 4 else 128.0)
            _samples = _np.frombuffer(_raw, dtype=_dtype).astype(_np.float32) / _scale
            if _nch > 1:
                _samples = _samples.reshape(-1, _nch).mean(axis=1)
            _audio_duration_sec = len(_samples) / _sr
            logger.info(f"Audio loaded: {_samples.shape}, sr={_sr}")
        except Exception as exc:
            logger.exception("Could not load audio for diarization")
            raise DiarizationError(f"Could not load audio for diarization: {exc}") from exc

        # Estimate runtime. Measured ~0.45× audio duration on CPU; GPU is
        # roughly 10-20x faster.
        if self._pipeline_on_gpu:
            est_sec = _audio_duration_sec * 0.05
        else:
            est_sec = _audio_duration_sec * 0.45
        if est_sec >= 60:
            est_str = f"~{int(est_sec / 60)} min"
        else:
            est_str = f"~{int(est_sec)} sec"
        if _audio_duration_sec >= 60:
            audio_dur_str = f"{int(_audio_duration_sec / 60)} min"
        else:
            audio_dur_str = f"{int(_audio_duration_sec)} sec"
        device_str = "GPU" if self._pipeline_on_gpu else "CPU"
        log_callback(
            f"[Diarize] Running on {device_str} — estimated wait: {est_str} "
            f"for {audio_dur_str} of audio. Please wait…"
        )
        logger.info(
            f"{device_str} diarization estimate: {est_str} for {_audio_duration_sec:.0f}s audio"
        )

        def _progress(num_done: int, num_total: int) -> int:
            if num_total > 0:
                pct = int(100 * num_done / num_total)
                log_callback(f"[Diarize] Progress: {pct}%")
            return 0  # 0 = continue processing

        try:
            result = self._pipeline.process(_samples, callback=_progress)  # type: ignore[union-attr]
        except Exception as exc:
            logger.exception("Diarization pipeline failed")
            raise DiarizationError(f"Speaker diarization failed: {exc}") from exc

        turns: list[tuple[float, float, str]] = [
            (seg.start, seg.end, f"SPEAKER_{seg.speaker:02d}")
            for seg in result.sort_by_start_time()
        ]

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

    # ─────────────────────────────────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────────────────────────────────

    def _ensure_pipeline_loaded(self, log_callback: Callable[[str], None]) -> None:
        if self._pipeline is not None:
            return

        from config import DIARIZATION_SEGMENTATION_MODEL, DIARIZATION_EMBEDDING_MODEL

        logger.info(
            f"Loading diarization pipeline: "
            f"segmentation={DIARIZATION_SEGMENTATION_MODEL}, "
            f"embedding={DIARIZATION_EMBEDDING_MODEL}"
        )

        try:
            import sherpa_onnx
        except ImportError as exc:
            raise DiarizationError(
                f"sherpa_onnx import failed: {exc}\n"
                "Re-build with diarization support enabled."
            ) from exc

        # GPU detection reuses transcriber's ctranslate2-based GPU enumeration —
        # works without torch (torch is no longer bundled).
        try:
            from transcriber import list_gpus
            _use_cuda = len(list_gpus()) > 0
        except Exception as _gpu_detect_exc:
            logger.warning(f"GPU detection failed, defaulting to CPU: {_gpu_detect_exc}")
            _use_cuda = False

        def _build_pipeline(provider: str):
            seg_cfg = sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(DIARIZATION_SEGMENTATION_MODEL)
                ),
                provider=provider,
            )
            emb_cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(DIARIZATION_EMBEDDING_MODEL),
                provider=provider,
            )
            clu_cfg = sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=0.5)
            config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
                segmentation=seg_cfg, embedding=emb_cfg, clustering=clu_cfg
            )
            return sherpa_onnx.OfflineSpeakerDiarization(config)

        pipeline = None
        if _use_cuda:
            # Diagnostic: test-load the CUDA provider DLL via ctypes first so that
            # if it fails we get the exact Windows error code (e.g. [WinError 126]
            # "module not found" = missing dep DLL, [WinError 127] = symbol mismatch).
            if sys.platform == "win32":
                import ctypes as _ctypes
                _lib_dir = Path(sherpa_onnx.__file__).parent / "lib"
                _cuda_ep = _lib_dir / "onnxruntime_providers_cuda.dll"
                if _cuda_ep.is_file():
                    try:
                        _ctypes.WinDLL(str(_cuda_ep))
                        logger.debug(f"ctypes pre-check: {_cuda_ep.name} loads OK")
                    except OSError as _cdl_exc:
                        logger.warning(f"ctypes pre-check: {_cuda_ep.name} FAILED: {_cdl_exc}")
                else:
                    logger.warning(f"ctypes pre-check: {_cuda_ep} not found")

            try:
                pipeline = _build_pipeline("cuda")
                self._pipeline_on_gpu = True
                logger.info("Diarization pipeline loaded on CUDA GPU")
                log_callback("[Diarize] Using GPU for speaker identification")
            except Exception as _gpu_exc:
                import traceback as _tb
                logger.warning(f"Could not load diarization pipeline on GPU: {_gpu_exc}")
                logger.debug(f"GPU pipeline traceback:\n{_tb.format_exc()}")
                pipeline = None

        if pipeline is None:
            try:
                pipeline = _build_pipeline("cpu")
                self._pipeline_on_gpu = False
                logger.info("Diarization pipeline loaded on CPU")
                log_callback(
                    "[Diarize] No GPU found — using CPU (may be slow for long files)"
                    if not _use_cuda
                    else "[Diarize] GPU unavailable for diarization — using CPU"
                )
            except Exception as exc:
                import traceback as _tb
                logger.error(f"Pipeline loading traceback:\n{_tb.format_exc()}")
                raise DiarizationError(
                    f"Could not load diarization pipeline from "
                    f"{DIARIZATION_SEGMENTATION_MODEL.parent}:\n{exc}"
                ) from exc

        self._pipeline = pipeline
        log_callback("[Diarize] Pipeline loaded")
        logger.info("Diarization pipeline loaded successfully")
