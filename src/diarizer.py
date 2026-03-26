"""
diarizer.py - Optional speaker diarization using pyannote.audio.

Diarization is run *after* transcription on the same mono 16 kHz WAV file.
Each transcript segment is assigned to the speaker whose pyannote turn has
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
import shutil
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


class DiarizationError(RuntimeError):
    """Raised when diarization fails."""


class DiarizationEngine:
    """Lazy-loading wrapper around a pyannote.audio Speaker Diarization pipeline."""

    def __init__(self) -> None:
        self._pipeline = None

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """
        Return True if the pyannote diarization models are present on disk.
        Does NOT import pyannote — safe to call at startup.

        Checks that the HF hub cache directory for speaker-diarization-3.1
        exists and has at least one snapshot inside it.
        """
        from config import DIARIZATION_MODELS_DIR
        snapshots_dir = DIARIZATION_MODELS_DIR / "snapshots"
        try:
            available = snapshots_dir.is_dir() and any(snapshots_dir.iterdir())
        except OSError:
            available = False
        if not available:
            logger.debug(
                f"Diarization models not found at {DIARIZATION_MODELS_DIR} "
                "(checkbox will be disabled)"
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
            diarization = self._pipeline(str(audio_path))  # type: ignore[misc]
        except Exception as exc:
            logger.exception("Diarization pipeline failed")
            raise DiarizationError(f"Speaker diarization failed: {exc}") from exc

        turns: list[tuple[float, float, str]] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append((turn.start, turn.end, speaker))

        turns.sort(key=lambda t: t[0])
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

        from config import DIARIZATION_MODELS_DIR, BASE_DIR
        import tempfile

        logger.info(f"Loading diarization pipeline from: {DIARIZATION_MODELS_DIR}")

        try:
            from pyannote.audio import Pipeline  # type: ignore[import]
        except ImportError as exc:
            raise DiarizationError(
                f"pyannote.audio import failed: {exc}\n"
                "Re-build with diarization support enabled."
            ) from exc

        hub_dir = BASE_DIR / "models" / "hf_cache" / "hub"

        def _find_local_snapshot(repo_id: str) -> str | None:
            """Return path to the model checkpoint file for a bundled HF repo, or None.

            Returns the path to pytorch_model.bin or model.safetensors inside the
            snapshot directory.  Model.from_pretrained() checks Path(x).is_file() —
            passing a directory causes it to fall back to hf_hub_download() even when
            the directory exists.
            """
            dir_name = "models--" + repo_id.replace("/", "--")
            snaps = hub_dir / dir_name / "snapshots"
            if not snaps.is_dir():
                logger.warning(f"Snapshots dir missing for {repo_id}: {snaps}")
                return None
            snap_dirs = sorted(snaps.iterdir())
            if not snap_dirs:
                logger.warning(f"Snapshots dir is empty for {repo_id}: {snaps}")
                return None
            snap = snap_dirs[0]
            for fname in ("pytorch_model.bin", "model.safetensors"):
                fpath = snap / fname
                if fpath.is_file():
                    logger.info(f"  Found {repo_id} checkpoint: {fpath}")
                    return str(fpath)
            # No recognized checkpoint file — fall back to directory
            logger.warning(f"No pytorch_model.bin/model.safetensors in {snap}, using dir")
            return str(snap)

        # Locate the main pipeline snapshot
        snapshots_dir = DIARIZATION_MODELS_DIR / "snapshots"
        snapshot_dirs = sorted(snapshots_dir.iterdir())
        if not snapshot_dirs:
            raise DiarizationError("No snapshot directory found in diarization model cache")

        # Read and patch config.yaml before loading.
        # The config references sub-models by HF repo ID (e.g. "pyannote/segmentation-3.0"),
        # which causes Pipeline.from_pretrained to call hf_hub_download() — this fails
        # offline.  We replace repo IDs with paths to the actual pytorch_model.bin files
        # inside the local snapshot directories.  Model.from_pretrained() skips hub
        # download when Path(x).is_file() — a directory path is NOT sufficient.
        try:
            from omegaconf import OmegaConf
        except ImportError as exc:
            raise DiarizationError(f"omegaconf import failed: {exc}") from exc

        config_path = snapshot_dirs[0] / "config.yaml"
        if not config_path.is_file():
            raise DiarizationError(f"config.yaml not found at {config_path}")

        config = OmegaConf.load(config_path)

        for key in ("segmentation", "embedding"):
            repo_id = str(getattr(config.pipeline.params, key, ""))
            if not repo_id or Path(repo_id).exists():
                continue  # already a local path or empty
            local = _find_local_snapshot(repo_id)
            if not local:
                raise DiarizationError(
                    f"Sub-model '{repo_id}' not found in bundled cache.\n"
                    "Re-download model.zip to fix."
                )
            OmegaConf.update(config, f"pipeline.params.{key}", local)
            log_callback(f"[Diarize] {key}: {Path(local).name}")
            logger.info(f"Resolved {key}: {repo_id} -> {local}")

        # Write the patched config to a temp directory and load from there.
        # We copy ALL files from the main snapshot into tmp (params.yaml,
        # etc.) so Pipeline.from_pretrained() never needs to call hf_hub_download
        # for any additional file — only config.yaml is overwritten with our
        # patched version.
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                for f in snapshot_dirs[0].iterdir():
                    if f.is_file():
                        shutil.copy2(f, tmp_path / f.name)
                OmegaConf.save(config, tmp_path / "config.yaml")
                pipeline = Pipeline.from_pretrained(str(tmp_path))
        except DiarizationError:
            raise
        except Exception as exc:
            import traceback as _tb
            logger.error(f"Pipeline loading traceback:\n{_tb.format_exc()}")
            raise DiarizationError(
                f"Could not load diarization pipeline from {DIARIZATION_MODELS_DIR}:\n{exc}"
            ) from exc

        self._pipeline = pipeline
        log_callback("[Diarize] Pipeline loaded")
        logger.info("Diarization pipeline loaded successfully")
