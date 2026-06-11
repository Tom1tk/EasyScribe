"""
recovery.py - Recover orphaned PCM recordings left by a crashed session.

Called from main.py before the GUI launches. For each recording_*.pcm + .json
pair found in the output directory, the user is offered a chance to recover it
as a .wav file.
"""

import json
import logging
import wave
from pathlib import Path

logger = logging.getLogger(__name__)


def scan_for_orphans(output_dir: Path) -> list[tuple[Path, Path, dict]]:
    """
    Find orphaned recording_*.pcm + recording_*.json pairs.
    Returns [(pcm_path, json_path, metadata), ...] sorted by filename.
    """
    if not output_dir.exists():
        return []

    orphans = []
    for pcm_path in sorted(output_dir.glob("recording_*.pcm")):
        json_path = pcm_path.with_suffix(".json")
        if not json_path.exists():
            continue
        try:
            metadata = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"Could not read {json_path.name}: {exc}")
            continue
        orphans.append((pcm_path, json_path, metadata))

    return orphans


def recover_pcm_to_wav(pcm_path: Path, metadata: dict, wav_path: Path) -> None:
    """Convert raw int16 PCM to WAV using metadata from the sidecar JSON."""
    sample_rate = int(metadata.get("sample_rate", 16000))
    channels = int(metadata.get("channels", 1))
    samples_written = int(metadata.get("samples_written", 0))
    byte_count = samples_written * 2  # int16 = 2 bytes per sample

    raw = pcm_path.read_bytes()[:byte_count]

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(raw)

    logger.info(f"Recovered {pcm_path.name} → {wav_path.name} ({len(raw):,} bytes)")


def _handle_orphan(pcm_path: Path, json_path: Path, metadata: dict, recover: bool) -> Path | None:
    """
    Apply the user's recovery choice for one orphaned recording.

    If recover is True, converts the PCM to a WAV file and deletes the
    .pcm/.json pair. If recover is False, leaves both files untouched so
    the prompt reappears next launch — declining must never delete the
    only copy of the recording.

    Returns the recovered WAV path, or None if recover was False.
    """
    if not recover:
        return None
    wav_path = pcm_path.with_suffix(".wav")
    recover_pcm_to_wav(pcm_path, metadata, wav_path)
    pcm_path.unlink(missing_ok=True)
    json_path.unlink(missing_ok=True)
    return wav_path


def prompt_and_recover(output_dir: Path) -> None:
    """
    Show a tkinter dialog for each orphaned recording and recover if confirmed.
    Called before the main GUI window is created.
    """
    import tkinter as tk
    from tkinter import messagebox

    orphans = scan_for_orphans(output_dir)
    if not orphans:
        return

    root = tk.Tk()
    root.withdraw()
    root.update()

    for pcm_path, json_path, metadata in orphans:
        samples = int(metadata.get("samples_written", 0))
        sr = int(metadata.get("sample_rate", 16000))
        duration_sec = samples / sr if sr > 0 else 0
        duration_str = f"{int(duration_sec // 60)}m {int(duration_sec % 60)}s"

        answer = messagebox.askyesno(
            "Unfinished Recording Found",
            f"An unfinished recording was found:\n\n"
            f"  File: {pcm_path.name}\n"
            f"  Duration: {duration_str}\n\n"
            f"Recover it as a .wav file now?\n"
            f"(Choosing No keeps the raw recording for next time.)",
            parent=root,
        )

        try:
            wav_path = _handle_orphan(pcm_path, json_path, metadata, answer)
        except Exception as exc:
            logger.error(f"Recovery failed for {pcm_path.name}: {exc}")
            messagebox.showerror(
                "Recovery Failed",
                f"Could not recover {pcm_path.name}:\n{exc}",
                parent=root,
            )
            continue

        if wav_path is not None:
            messagebox.showinfo(
                "Recovery Complete",
                f"Saved to:\n{wav_path}",
                parent=root,
            )

    root.destroy()
