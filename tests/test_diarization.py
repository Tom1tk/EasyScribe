"""
End-to-end local test for the production transcription + diarization pipeline.
Runs sherpa-onnx (VAD segmentation, Whisper recognizer, diarization) against
tests/data/test_speech.wav via TranscriptionEngine.transcribe(diarize=True).
Run from project root: python tests/test_diarization.py
"""
import sys
import tempfile
import threading
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
AUDIO = BASE_DIR / "tests" / "data" / "test_speech.wav"

sys.path.insert(0, str(BASE_DIR / "src"))

# ── Step 1: Confirm diarization models are bundled ───────────────────────
print("\n=== Step 1: Checking diarization models ===")
from diarizer import DiarizationEngine

assert DiarizationEngine().is_available(), "Diarization models not found at models/diarization/"
print("Diarization models found.")

# ── Step 2: Run the production pipeline (VAD + Whisper + diarization) ────
print(f"\n=== Step 2: Transcribing + diarizing {AUDIO.name} ===")
assert AUDIO.is_file(), f"Audio not found: {AUDIO}"

from transcriber import TranscriptionEngine

engine = TranscriptionEngine()
cancel_event = threading.Event()

with tempfile.TemporaryDirectory() as tmp:
    output_path = Path(tmp) / "transcript.txt"
    engine.transcribe(
        audio_path=AUDIO,
        output_path=output_path,
        add_timestamps=True,
        cancel_event=cancel_event,
        status_callback=lambda s: None,
        progress_callback=lambda f: None,
        log_callback=lambda s: print(f"  [transcribe] {s}"),
        diarize=True,
    )
    transcript = output_path.read_text(encoding="utf-8")

# ── Step 3: Show and verify the diarized transcript ──────────────────────
print("\n=== Final transcript ===")
print(transcript)

assert transcript.strip(), "Transcript is empty"
assert "[Speaker" in transcript, "Transcript has no speaker labels — diarization may have failed"

print("\n\nALL TESTS PASSED")
