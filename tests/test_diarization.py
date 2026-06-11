"""
End-to-end local test for diarization pipeline.
Uses tiny whisper + sherpa-onnx diarization against tests/data/test_speech.wav.
Run from project root: .venv/bin/python tests/test_diarization.py
"""
import os, sys, logging
from pathlib import Path

# ── Offline enforcement (same as config.py) ───────────────────────────────
BASE_DIR = Path(__file__).parent.parent
os.environ["HF_HOME"]              = str(BASE_DIR / "models" / "hf_cache")
os.environ["HF_HUB_CACHE"]        = str(BASE_DIR / "models" / "hf_cache" / "hub")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"
os.environ["HF_HUB_OFFLINE"]      = "1"
os.environ["HUGGINGFACE_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["NO_PROXY"] = "*"

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("test")

AUDIO = BASE_DIR / "tests" / "data" / "test_speech.wav"
HUB   = BASE_DIR / "models" / "hf_cache" / "hub"

sys.path.insert(0, str(BASE_DIR / "src"))

# ── Step 1: Load sherpa-onnx diarization engine ──────────────────────────
print("\n=== Step 1: Loading sherpa-onnx diarization engine ===")
from diarizer import DiarizationEngine

engine = DiarizationEngine()
assert engine.is_available(), "Diarization models not found at models/diarization/"
print("DiarizationEngine ready (models found).")

def _log(msg):
    print(f"  [diarize] {msg}")

# ── Step 2: Run diarization ───────────────────────────────────────────────
print(f"\n=== Step 2: Running diarization on {AUDIO.name} ===")
assert AUDIO.is_file(), f"Audio not found: {AUDIO}"

turns = engine.diarize(AUDIO, log_callback=_log)
turns.sort(key=lambda t: t[0])

speaker_set = {t[2] for t in turns}
print(f"Found {len(speaker_set)} speaker(s), {len(turns)} turn(s):")
for s, e, sp in turns:
    print(f"  [{s:.2f}s - {e:.2f}s] {sp}")

# ── Step 3: Transcribe with tiny whisper ─────────────────────────────────
print("\n=== Step 3: Transcribing with faster-whisper-tiny.en ===")
from faster_whisper import WhisperModel
tiny_model_path = str(next((HUB / "models--Systran--faster-whisper-tiny.en" / "snapshots").iterdir()))
model = WhisperModel(tiny_model_path, device="cpu", compute_type="int8")
segments_gen, info = model.transcribe(str(AUDIO), beam_size=5, vad_filter=True, language="en")
raw_segments = [(seg.start, seg.end, seg.text.strip()) for seg in segments_gen]
print(f"Transcribed {len(raw_segments)} segment(s):")
for s, e, t in raw_segments:
    print(f"  [{s:.2f}s - {e:.2f}s] {t}")

# ── Step 4: Assign speakers ──────────────────────────────────────────────
print("\n=== Step 4: Assigning speakers ===")
assigned = DiarizationEngine.assign_speakers(raw_segments, turns)

# Build speaker map (first appearance = Speaker 1)
speaker_map = {}
for speaker, _t, _s, _e in assigned:
    if speaker not in speaker_map:
        speaker_map[speaker] = f"Speaker {len(speaker_map) + 1}"
print(f"Speaker map: {speaker_map}")

print("\n=== Final transcript ===")
prev = None
for speaker, text, start, end in assigned:
    label = speaker_map.get(speaker, speaker)
    if label != prev:
        print(f"\n[{label}]")
    print(text)
    prev = label

print("\n\nALL TESTS PASSED")
