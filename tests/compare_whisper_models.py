#!/usr/bin/env python3
"""
Local A/B comparison of Whisper ONNX models for sherpa-onnx.

Runs the same audio file through the real production pipeline
(transcriber.TranscriptionEngine.transcribe — VAD segmentation, clamped
padding, etc.) once per Whisper ONNX model directory (each containing
<prefix>-encoder[.int8].onnx, <prefix>-decoder[.int8].onnx,
<prefix>-tokens.txt), and prints the resulting transcript + timing for each.

CPU only — for quality comparison, not speed benchmarking.

Usage:
    python tests/compare_whisper_models.py <audio.wav> <name=dir> [<name=dir> ...]

Example:
    python tests/compare_whisper_models.py tests/data/test_speech.wav \\
        distil-v3=models/whisper \\
        turbo=models/whisper_turbo
"""
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import config
from transcriber import TranscriptionEngine


def _find_model_files(model_dir: Path) -> tuple[Path, Path, Path]:
    """Find encoder/decoder/tokens files in a model directory.

    Prefers .int8.onnx if both int8 and fp32 are present.
    """
    encoders = sorted(model_dir.glob("*encoder*.onnx"))
    decoders = sorted(model_dir.glob("*decoder*.onnx"))
    tokens = list(model_dir.glob("*tokens.txt"))

    def _pick(paths: list[Path]) -> Path:
        int8 = [p for p in paths if "int8" in p.name]
        return int8[0] if int8 else paths[0]

    if not encoders or not decoders or not tokens:
        raise FileNotFoundError(f"Could not find encoder/decoder/tokens in {model_dir}")

    return _pick(encoders), _pick(decoders), tokens[0]


def run_one(name: str, model_dir: Path, audio_path: Path) -> None:
    print(f"\n=== {name} ({model_dir}) ===")
    try:
        encoder, decoder, tokens = _find_model_files(model_dir)
    except FileNotFoundError as exc:
        print(f"  SKIP: {exc}")
        return

    print(f"  encoder: {encoder.name}  ({encoder.stat().st_size / 1e6:.0f} MB)")
    print(f"  decoder: {decoder.name}  ({decoder.stat().st_size / 1e6:.0f} MB)")

    # Point the production config at this model directory's files, then drive
    # the real engine so the comparison reflects the actual VAD pipeline.
    config.WHISPER_ENCODER = encoder
    config.WHISPER_DECODER = decoder
    config.WHISPER_TOKENS = tokens

    engine = TranscriptionEngine()
    cancel_event = threading.Event()

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "transcript.txt"
        t0 = time.monotonic()
        engine.transcribe(
            audio_path=audio_path,
            output_path=out_path,
            add_timestamps=False,
            cancel_event=cancel_event,
            status_callback=lambda s: None,
            progress_callback=lambda f: None,
            log_callback=lambda s: None,
            diarize=False,
        )
        elapsed = time.monotonic() - t0
        text = out_path.read_text().strip()

    print(f"  total: {elapsed:.1f}s")
    print(f"  TEXT: {text}")


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    audio_path = Path(sys.argv[1])

    for arg in sys.argv[2:]:
        name, _, dir_str = arg.partition("=")
        run_one(name, Path(dir_str), audio_path)


if __name__ == "__main__":
    main()
