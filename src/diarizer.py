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
from pathlib import Path
from typing import Callable

from common import fmt_duration, load_wav_float32

logger = logging.getLogger(__name__)


class DiarizationError(RuntimeError):
    """Raised when diarization fails."""


class DiarizationEngine:
    """Lazy-loading wrapper around a sherpa-onnx offline speaker diarization pipeline."""

    def __init__(self) -> None:
        self._pipeline = None

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
            samples, sr = load_wav_float32(audio_path)
            logger.info(f"Audio loaded: {samples.shape}, sr={sr}")
        except Exception as exc:
            logger.exception("Could not load audio for diarization")
            raise DiarizationError(f"Could not load audio for diarization: {exc}") from exc

        # Measured ~0.45x the audio length on CPU.
        audio_sec = len(samples) / sr
        est_sec = audio_sec * 0.45
        est_str = f"~{int(est_sec / 60)} min" if est_sec >= 60 else f"~{int(est_sec)} sec"
        log_callback(
            f"[Diarize] Estimated wait: {est_str} for {fmt_duration(audio_sec)} of audio. "
            "Please wait…"
        )
        logger.info(f"Diarization estimate: {est_str} for {audio_sec:.0f}s audio")

        def _progress(num_done: int, num_total: int) -> int:
            if num_total > 0:
                pct = int(100 * num_done / num_total)
                log_callback(f"[Diarize] Progress: {pct}%")
            return 0  # 0 = continue processing

        try:
            result = self._pipeline.process(samples, callback=_progress)  # type: ignore[union-attr]
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

        # Diarization runs on CPU. sherpa-onnx has no Vulkan provider, and a
        # CUDA provider would need the separate CUDA DLL stack that v2.0
        # removed (CLAUDE.md Rules 6 and 7).
        seg_cfg = sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(DIARIZATION_SEGMENTATION_MODEL)
            ),
            provider="cpu",
        )
        emb_cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(DIARIZATION_EMBEDDING_MODEL),
            provider="cpu",
        )
        clu_cfg = sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=0.5)
        diar_cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=seg_cfg, embedding=emb_cfg, clustering=clu_cfg
        )
        try:
            pipeline = sherpa_onnx.OfflineSpeakerDiarization(diar_cfg)
        except Exception as exc:
            logger.exception("Could not load the diarization pipeline")
            raise DiarizationError(
                f"Could not load diarization pipeline from "
                f"{DIARIZATION_SEGMENTATION_MODEL.parent}:\n{exc}"
            ) from exc
        log_callback("[Diarize] Running on CPU")

        self._pipeline = pipeline
        log_callback("[Diarize] Pipeline loaded")
        logger.info("Diarization pipeline loaded successfully")
