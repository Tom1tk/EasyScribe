"""
fetch.py - Download the A/B models and build the A/B test set.

Developer tool only. It uses the network, so it lives in tests/, never in
src/ or launcher/ (CLAUDE.md Rule 13).

    python tests/asr_ab/fetch.py models [--only cohere,phonon2] [--dest DIR]
    python tests/asr_ab/fetch.py data [--dest DIR]

Defaults: tests/asr_ab/models/ and tests/asr_ab/data/ (both git-ignored).
"data" needs ffmpeg on PATH and pyarrow, numpy, soundfile.

To test a new model, add an entry to MODELS here and an engine to
ab_models.py with the same name.
"""

import argparse
import collections
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
HF = "https://huggingface.co"
SR = 16000

# name -> (Hugging Face repo, files). The folder name is the engine name.
MODELS = {
    "ggml": ("ggerganov/whisper.cpp", ["ggml-large-v3-turbo-q5_0.bin"]),
    "whisper_turbo": ("csukuangfj/sherpa-onnx-whisper-turbo", [
        "turbo-encoder.int8.onnx", "turbo-decoder.int8.onnx", "turbo-tokens.txt"]),
    "phonon2": ("tiyuvta/Phonon-2-ONNX", [
        "config.json", "vocab.txt", "preprocessor-model.onnx",
        "encoder-model.exact4x2.onnx", "decoder_joint-model.exact4x2.onnx",
        "NOTICE", "README.md"]),
    "parakeet_v3": ("csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8", [
        "encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"]),
    "nemotron": ("csukuangfj2/sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25", [
        "encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt",
        "README.md"]),
    "qwen3_06b": ("csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25", [
        "conv_frontend.onnx", "encoder.int8.onnx", "decoder.int8.onnx",
        "tokenizer/merges.txt", "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json"]),
    "qwen3_17b": ("thieunv-asilla/sherpa-onnx-qwen3-asr-1.7B-int8", [
        "conv_frontend.onnx", "encoder.int8.onnx", "decoder.int8.onnx",
        "tokenizer/merges.txt", "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json", "README.md"]),
    "cohere": ("csukuangfj2/sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01", [
        "encoder.int8.onnx", "encoder.int8.onnx.data", "decoder.int8.onnx",
        "tokens.txt", "README.md"]),
}
SILERO_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
              "asr-models/silero_vad.onnx")

# Test-set sources (public datasets on Hugging Face).
TED_URL = (f"{HF}/datasets/distil-whisper/tedlium-long-form/resolve/main/data/"
           "test-00000-of-00001-7a1bb92f62e929b8.parquet")
TED_ROW = 6  # one complete 5.8 min talk
AMI_URL = f"{HF}/datasets/edinburghcstr/ami/resolve/main/sdm/test-00000-of-00004.parquet"
E22_URL = (f"{HF}/datasets/distil-whisper/earnings22/resolve/main/chunked/"
           "test-00010-of-00038-060854d670222f19.parquet")
CLIP_SEC = 150.0


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(5):
        try:
            print(f"  {url}", flush=True)
            with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.replace(dest)
            return
        except OSError as e:
            if attempt == 4:
                raise
            print(f"  retry after error: {e}", flush=True)
            time.sleep(2 ** (attempt + 1))


def fetch_models(dest: Path, only: list[str] | None) -> None:
    download(SILERO_URL, dest / "silero_vad.onnx")
    for name, (repo, files) in MODELS.items():
        if only and name not in only:
            continue
        print(name)
        for f in files:
            download(f"{HF}/{repo}/resolve/main/{f}", dest / name / f)


# ─── Test set ──────────────────────────────────────────────────────────────

def _decode(data: bytes):
    import numpy as np
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", "-", "-ac", "1",
                        "-ar", str(SR), "-f", "f32le", "-"],
                       input=data, capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.float32)


def _write(out: Path, name: str, audio, text: str) -> None:
    import numpy as np
    import soundfile as sf
    sf.write(out / f"{name}.wav", np.clip(audio, -1, 1), SR, subtype="PCM_16")
    (out / f"{name}.txt").write_text(text.strip() + "\n", "utf-8")
    print(f"  {name}: {len(audio) / SR / 60:.1f} min")


def _clip_from_segments(segs: list[dict]):
    """Place utterances at their real times in the CLIP_SEC window with the
    most speech. Overlapping speech is mixed, as in the real recording."""
    import numpy as np
    segs.sort(key=lambda s: s["start"])
    best = None
    for i, s in enumerate(segs):
        win = [x for x in segs[i:] if x["end"] <= s["start"] + CLIP_SEC]
        cov = sum(x["end"] - x["start"] for x in win)
        if best is None or cov > best[0]:
            best = (cov, s["start"], win)
    _, start, win = best
    end = max(x["end"] for x in win) - start
    buf = np.zeros(int((end + 0.5) * SR), np.float32)
    for x in win:
        a = _decode(x["bytes"])
        i = int((x["start"] - start) * SR)
        n = min(len(a), len(buf) - i)
        buf[i:i + n] += a[:n]
    return buf, " ".join(x["text"] for x in win)


def _by_recording(rows, key, start, end, text):
    by = collections.defaultdict(list)
    for r in rows:
        by[r[key]].append({"start": float(r[start]), "end": float(r[end]),
                           "text": r[text], "bytes": r["audio"]["bytes"]})
    # The recording with the most utterances in the shard.
    return by[max(by, key=lambda k: len(by[k]))]


def fetch_data(dest: Path) -> None:
    import pyarrow.parquet as pq
    dest.mkdir(parents=True, exist_ok=True)
    tmp = dest / "_download"
    try:
        p = tmp / "ted.parquet"
        download(TED_URL, p)
        t = pq.read_table(p)
        key = "transcription" if "transcription" in t.schema.names else "text"
        _write(dest, "ted_bigidea", _decode(t.column("audio")[TED_ROW].as_py()["bytes"]),
               t.column(key)[TED_ROW].as_py())

        p = tmp / "ami.parquet"
        download(AMI_URL, p)
        rows = pq.read_table(p).to_pylist()
        _write(dest, "ami_meeting", *_clip_from_segments(
            _by_recording(rows, "meeting_id", "begin_time", "end_time", "text")))

        p = tmp / "e22.parquet"
        download(E22_URL, p)
        rows = pq.read_table(p).to_pylist()
        _write(dest, "earnings_call", *_clip_from_segments(
            _by_recording(rows, "file_id", "start_ts", "end_ts", "transcription")))
    finally:
        for f in tmp.glob("*"):
            f.unlink()
        if tmp.exists():
            tmp.rmdir()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["models", "data"])
    p.add_argument("--dest", type=Path)
    p.add_argument("--only", help="comma-separated model names (models only)")
    a = p.parse_args()
    if a.what == "models":
        fetch_models(a.dest or HERE / "models",
                     a.only.split(",") if a.only else None)
    else:
        fetch_data(a.dest or HERE / "data")


if __name__ == "__main__":
    sys.exit(main())
