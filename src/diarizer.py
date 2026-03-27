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
        self._pipeline_on_gpu: bool = False

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

        # pyannote 4.x requires torchcodec for file-path input (not bundled).
        # Pre-load the WAV using the stdlib `wave` module and pass as a
        # {'waveform': tensor, 'sample_rate': int} dict — bypasses torchcodec.
        try:
            import wave as _wave
            import numpy as _np
            import torch as _torch
            with _wave.open(str(audio_path), "rb") as _wf:
                _nch = _wf.getnchannels()
                _sw = _wf.getsampwidth()
                _sr = _wf.getframerate()
                _raw = _wf.readframes(_wf.getnframes())
            _dtype = _np.int16 if _sw == 2 else (_np.int32 if _sw == 4 else _np.int8)
            _scale = 32768.0 if _sw == 2 else (2147483648.0 if _sw == 4 else 128.0)
            _data = _np.frombuffer(_raw, dtype=_dtype).astype(_np.float32) / _scale
            if _nch > 1:
                _data = _data.reshape(-1, _nch).mean(axis=1)
            _waveform = _torch.tensor(_data).unsqueeze(0)  # (1, time)
            audio_input = {"waveform": _waveform, "sample_rate": _sr}
            _audio_duration_sec = _waveform.shape[1] / _sr
            logger.info(f"Audio pre-loaded: {_waveform.shape}, sr={_sr}")
        except Exception as exc:
            logger.warning(f"Could not pre-load audio, falling back to path: {exc}")
            audio_input = str(audio_path)  # type: ignore[assignment]
            _audio_duration_sec = 0.0

        # Warn about expected CPU runtime. Measured ~0.6× audio duration on CPU
        # (e.g. 14 min audio ≈ 8–9 min). GPU would be ~10–20× faster but requires
        # CUDA torch, which exceeds the 2 GB GitHub release file-size limit.
        if _audio_duration_sec > 0 and not self._pipeline_on_gpu:
            est_sec = _audio_duration_sec * 0.6
            if est_sec >= 60:
                est_str = f"~{int(est_sec / 60)} min"
            else:
                est_str = f"~{int(est_sec)} sec"
            if _audio_duration_sec >= 60:
                audio_dur_str = f"{int(_audio_duration_sec / 60)} min"
            else:
                audio_dur_str = f"{int(_audio_duration_sec)} sec"
            log_callback(
                f"[Diarize] Running on CPU — estimated wait: {est_str} "
                f"for {audio_dur_str} of audio. Please wait…"
            )
            logger.info(f"CPU diarization estimate: {est_str} for {_audio_duration_sec:.0f}s audio")

        try:
            diarization = self._pipeline(audio_input)  # type: ignore[misc]
        except Exception as exc:
            logger.exception("Diarization pipeline failed")
            raise DiarizationError(f"Speaker diarization failed: {exc}") from exc

        # pyannote 4.x returns DiarizeOutput(speaker_diarization=Annotation, ...)
        # pyannote 3.x returned an Annotation directly — handle both.
        annotation = getattr(diarization, "speaker_diarization", diarization)

        turns: list[tuple[float, float, str]] = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
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

        # Block pyannote.audio telemetry BEFORE importing pyannote.
        # pyannote 4.x imports opentelemetry at module load time and immediately
        # starts a background thread (PeriodicExportingMetricReader) that POSTs
        # usage metrics to otel.pyannote.ai:443 every 60 seconds.
        # Injecting a no-op fake module into sys.modules prevents the exporter
        # and its background thread from ever being created.
        try:
            import sys as _sys
            import types as _types

            _fake_metrics = _types.ModuleType("pyannote.audio.telemetry.metrics")
            for _fn in (
                "track_model_init",
                "track_pipeline_init",
                "track_pipeline_apply",
                "set_telemetry_metrics",
                "set_opentelemetry_log_level",
            ):
                setattr(_fake_metrics, _fn, lambda *a, **k: None)
            _fake_metrics.is_metrics_enabled = lambda: False  # type: ignore[attr-defined]

            _fake_telemetry = _types.ModuleType("pyannote.audio.telemetry")
            _fake_telemetry.__path__ = []  # type: ignore[attr-defined]  # marks it as a package
            _fake_telemetry.__package__ = "pyannote.audio.telemetry"  # type: ignore[attr-defined]
            for _fn in (
                "track_model_init",
                "track_pipeline_init",
                "track_pipeline_apply",
                "set_telemetry_metrics",
                "set_opentelemetry_log_level",
            ):
                setattr(_fake_telemetry, _fn, lambda *a, **k: None)

            _sys.modules["pyannote.audio.telemetry.metrics"] = _fake_metrics
            _sys.modules["pyannote.audio.telemetry"] = _fake_telemetry
            logger.info("Blocked pyannote telemetry (no-op module injected)")
        except Exception as _tel_exc:
            logger.warning(f"Could not block pyannote telemetry: {_tel_exc}")

        try:
            from pyannote.audio import Pipeline  # type: ignore[import]
        except ImportError as exc:
            raise DiarizationError(
                f"pyannote.audio import failed: {exc}\n"
                "Re-build with diarization support enabled."
            ) from exc

        # Patch get_plda in the speaker_diarization module globals to prevent a
        # hub download for pyannote/speaker-diarization-community-1 (unbundled PLDA
        # model).  SpeakerDiarization.__init__ calls get_plda(plda, ...) where
        # `plda` is a non-None DEFAULT PARAMETER referencing that hub model.
        # Python resolves get_plda at call time from SpeakerDiarization.__init__.__globals__
        # (= speaker_diarization module dict), so patching the module attribute here
        # intercepts the call even though it was already imported.
        try:
            import pyannote.audio.pipelines.speaker_diarization as _sd_module
            _sd_module.get_plda = lambda *args, **kwargs: None
            logger.info("Patched get_plda → None (PLDA model not bundled)")
        except Exception as _patch_exc:
            logger.warning(f"Could not patch get_plda: {_patch_exc}")

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
                # Copy all snapshot files EXCEPT params.yaml.
                # params.yaml contains calibrated params including a `plda` key
                # that references pyannote/speaker-diarization-community-1 — an
                # unbundled hub model.  Pipeline.from_pretrained merges params.yaml
                # into constructor kwargs before Klass(**params), triggering the
                # PLDA hub download.  Without params.yaml the pipeline uses default
                # uncalibrated params (cosine similarity clustering) — still accurate.
                for f in snapshot_dirs[0].iterdir():
                    if f.is_file() and f.name != "params.yaml":
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

        # Move pipeline to GPU if available — CPU diarization on long audio is
        # extremely slow (can take longer than the audio itself).
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                pipeline = pipeline.to(_torch.device("cuda"))
                self._pipeline_on_gpu = True
                logger.info("Diarization pipeline moved to CUDA GPU")
                log_callback("[Diarize] Using GPU for speaker identification")
            else:
                self._pipeline_on_gpu = False
                logger.info("No CUDA GPU — diarization will run on CPU (may be slow)")
                log_callback("[Diarize] No GPU found — using CPU (may be slow for long files)")
        except Exception as _gpu_exc:
            self._pipeline_on_gpu = False
            logger.warning(f"Could not move pipeline to GPU: {_gpu_exc}")

        self._pipeline = pipeline
        log_callback("[Diarize] Pipeline loaded")
        logger.info("Diarization pipeline loaded successfully")
