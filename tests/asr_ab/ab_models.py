"""
ab_models.py - A/B test speech-to-text models on short reference clips.

Developer tool only. It is not bundled and it does not run in CI.
compare_whisper_models.py compares Whisper ONNX folders by eye; this tool
compares different engines and gives a word error rate (WER) for each one.
See README.md in this folder for the steps and the last results.

Data folder: <name>.wav (16 kHz mono) + <name>.txt (reference text).
Models folder: one sub-folder per engine, as fetch.py makes it. Only the
engines you ask for must exist.

Extra packages (test only, not in requirements.txt):
    pip install jiwer whisper-normalizer soundfile pyarrow onnx-asr onnxruntime

Usage:
    python tests/asr_ab/ab_models.py --whisper-cli PATH
        [--engines current_files,cohere,...] [--data DIR] [--models DIR]
        [--out DIR]

The engines that need segments use the app's own VAD settings (src/vad.py),
so each engine gets the same speech segments as the live / fallback path.
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))

THREADS = 4
SR = 16000


def _dir_size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


# ─── Segmentation (same VAD settings as the app) ──────────────────────────

_vad_cache: dict[str, list[tuple[int, int]]] = {}


def vad_segments(name: str, samples: np.ndarray, vad_model: Path):
    if name not in _vad_cache:
        import config
        config.VAD_MODEL_PATH = vad_model
        import vad
        det = vad.build_vad(buffer_size_in_seconds=int(len(samples) / SR) + 10)
        bounds = vad.file_segments(det, samples, threading.Event())
        _vad_cache[name] = vad.pad_and_clamp(bounds, len(samples))
    return _vad_cache[name]


# ─── Engines ───────────────────────────────────────────────────────────────
# Each loader returns (size_path, fn). fn(name, wav_path, samples) -> text.

def eng_current_files(m: Path, a):
    model = m / "ggml" / "ggml-large-v3-turbo-q5_0.bin"

    def run(name, wav, samples):
        with tempfile.TemporaryDirectory() as td:
            stem = Path(td) / "out"
            cmd = [a.whisper_cli, "-m", str(model), "-f", str(wav), "-bs", "5",
                   "-t", str(THREADS), "-pp", "--output-json",
                   "--output-file", str(stem)]
            subprocess.run(cmd, check=True, capture_output=True)
            data = json.loads((stem.with_suffix(".json")).read_text("utf-8"))
            return " ".join(s["text"].strip() for s in data["transcription"])
    return model, run


def _per_segment(decode, vad_model):
    def run(name, wav, samples):
        out = []
        for s, e in vad_segments(name, samples, vad_model):
            out.append(decode(samples[s:e]).strip())
        return " ".join(t for t in out if t)
    return run


def _sherpa_decode(rec):
    def decode(x):
        st = rec.create_stream()
        st.accept_waveform(SR, x)
        rec.decode_stream(st)
        return st.result.text
    return decode


def eng_current_live(m: Path, a):
    import sherpa_onnx
    d = m / "whisper_turbo"
    rec = sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=str(d / "turbo-encoder.int8.onnx"),
        decoder=str(d / "turbo-decoder.int8.onnx"),
        tokens=str(d / "turbo-tokens.txt"),
        num_threads=THREADS, language="en", task="transcribe")
    return d, _per_segment(_sherpa_decode(rec), m / "silero_vad.onnx")


def eng_cohere(m: Path, a):
    import sherpa_onnx
    d = m / "cohere"
    rec = sherpa_onnx.OfflineRecognizer.from_cohere_transcribe(
        encoder=str(d / "encoder.int8.onnx"),
        decoder=str(d / "decoder.int8.onnx"),
        tokens=str(d / "tokens.txt"),
        num_threads=THREADS, language="en")
    return d, _per_segment(_sherpa_decode(rec), m / "silero_vad.onnx")


def eng_parakeet_v3(m: Path, a):
    import sherpa_onnx
    d = m / "parakeet_v3"
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(d / "encoder.int8.onnx"), decoder=str(d / "decoder.int8.onnx"),
        joiner=str(d / "joiner.int8.onnx"), tokens=str(d / "tokens.txt"),
        num_threads=THREADS, model_type="nemo_transducer")
    return d, _per_segment(_sherpa_decode(rec), m / "silero_vad.onnx")


def _qwen3(sub):
    def load(m: Path, a):
        import sherpa_onnx
        d = m / sub
        rec = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
            conv_frontend=str(d / "conv_frontend.onnx"),
            encoder=str(d / "encoder.int8.onnx"),
            decoder=str(d / "decoder.int8.onnx"),
            tokenizer=str(d / "tokenizer"),
            num_threads=THREADS, max_new_tokens=512, max_total_len=1024)
        return d, _per_segment(_sherpa_decode(rec), m / "silero_vad.onnx")
    return load


def eng_phonon2(m: Path, a):
    import onnx_asr
    d = m / "phonon2"
    model = onnx_asr.load_model("nemo-conformer-tdt", str(d),
                                quantization="exact4x2")
    return d, _per_segment(lambda x: model.recognize(x, sample_rate=SR),
                           m / "silero_vad.onnx")


def eng_nemotron(m: Path, a):
    """Streaming model: feed the whole file, as live mode would."""
    import sherpa_onnx
    d = m / "nemotron"
    rec = sherpa_onnx.OnlineRecognizer.from_transducer(
        encoder=str(d / "encoder.int8.onnx"), decoder=str(d / "decoder.int8.onnx"),
        joiner=str(d / "joiner.int8.onnx"), tokens=str(d / "tokens.txt"),
        num_threads=THREADS)

    def run(name, wav, samples):
        st = rec.create_stream()
        step = SR // 2
        for i in range(0, len(samples), step):
            st.accept_waveform(SR, samples[i:i + step])
            while rec.is_ready(st):
                rec.decode_stream(st)
        st.accept_waveform(SR, np.zeros(SR, dtype=np.float32))
        st.input_finished()
        while rec.is_ready(st):
            rec.decode_stream(st)
        return rec.get_result(st)
    return d, run


ENGINES = {
    "current_files": eng_current_files,
    "current_live": eng_current_live,
    "cohere": eng_cohere,
    "phonon2": eng_phonon2,
    "parakeet_v3": eng_parakeet_v3,
    "qwen3_06b": _qwen3("qwen3_06b"),
    "qwen3_17b": _qwen3("qwen3_17b"),
    "nemotron": eng_nemotron,
}


# ─── Scoring ───────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    from whisper_normalizer.english import EnglishTextNormalizer
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"\binaudible\b", " ", text, flags=re.I)
    return EnglishTextNormalizer()(text)


def wer(ref: str, hyp: str) -> float:
    import jiwer
    r, h = _norm(ref), _norm(hyp)
    return jiwer.wer(r, h) if h else 1.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=HERE / "data")
    p.add_argument("--models", type=Path, default=HERE / "models")
    p.add_argument("--whisper-cli", default="whisper-cli")
    p.add_argument("--engines", default=",".join(ENGINES))
    p.add_argument("--out", type=Path,
                   default=HERE / "results" / time.strftime("%Y-%m-%d"))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    clips = sorted(a.data.glob("*.wav"))
    audio = {}
    for w in clips:
        x, sr = sf.read(w, dtype="float32")
        assert sr == SR and x.ndim == 1, f"{w}: need 16 kHz mono"
        audio[w.stem] = x

    results = json.loads((a.out / "results.json").read_text()) \
        if (a.out / "results.json").exists() else {}
    for eng in a.engines.split(","):
        t0 = time.perf_counter()
        size_path, fn = ENGINES[eng](a.models, a)
        load_s = time.perf_counter() - t0
        row = {"size_mb": _dir_size(size_path) / 1e6, "load_s": load_s, "clips": {}}
        for w in clips:
            t0 = time.perf_counter()
            hyp = fn(w.stem, w, audio[w.stem])
            dt = time.perf_counter() - t0
            ref = w.with_suffix(".txt").read_text("utf-8")
            score = wer(ref, hyp)
            dur = len(audio[w.stem]) / SR
            row["clips"][w.stem] = {"wer": score, "sec": dt, "rtf": dt / dur}
            (a.out / f"{eng}__{w.stem}.txt").write_text(hyp, "utf-8")
            print(f"{eng:14s} {w.stem:15s} WER {score*100:5.1f}%  "
                  f"RTF {dt/dur:.3f}", flush=True)
        results[eng] = row
        (a.out / "results.json").write_text(json.dumps(results, indent=1))

    names = [w.stem for w in clips]
    print("\n| Engine | " + " | ".join(names) + " | Mean WER | RTF | Size MB |")
    print("|---" * (len(names) + 4) + "|")
    for eng, row in results.items():
        ws = [row["clips"][n]["wer"] for n in names]
        rtf = sum(row["clips"][n]["sec"] for n in names) / \
            sum(len(audio[n]) / SR for n in names)
        print(f"| {eng} | " + " | ".join(f"{w*100:.1f}" for w in ws) +
              f" | {np.mean(ws)*100:.1f} | {rtf:.3f} | {row['size_mb']:.0f} |")


if __name__ == "__main__":
    main()
