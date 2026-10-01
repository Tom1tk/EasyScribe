# ASR model A/B test

Use this tool to compare speech-to-text models with the engines that EasyScribe
uses now. It gives a word error rate (WER), the speed and the size on disk for
each engine. When a new model is released, add it and run the test again.

This is a developer tool only. It is not in the app bundle and it does not run
in CI. `fetch.py` uses the network, so it must stay in `tests/` (CLAUDE.md Rule 13).

## Steps

1. Install the test-only packages (they are not in `requirements.txt`):
   ```
   pip install -r requirements.txt
   pip install jiwer whisper-normalizer soundfile pyarrow onnx-asr onnxruntime
   ```
2. Download the models and build the test set. Both folders are git-ignored.
   You must have ffmpeg on PATH for the test set.
   ```
   python tests/asr_ab/fetch.py models              # about 10 GB, all engines
   python tests/asr_ab/fetch.py models --only ggml,whisper_turbo,phonon2
   python tests/asr_ab/fetch.py data                # about 20 MB, 10.8 min
   ```
3. Run the test. Give the path of a `whisper-cli` build for the
   `current_files` engine (for example `whispercpp\whisper-cli.exe` from a
   release build).
   ```
   python tests/asr_ab/ab_models.py --whisper-cli PATH
   python tests/asr_ab/ab_models.py --whisper-cli PATH --engines current_files,phonon2
   ```
   The results go to `results/<date>/`: `results.json` and one transcript per
   engine and clip. The script prints a markdown table at the end.

You can also test your own recordings. Put `<name>.wav` (16 kHz mono) and
`<name>.txt` (a correct transcript) in a folder, then use `--data <folder>`.

## Engines

| Engine | Model | Runtime | Method |
|---|---|---|---|
| `current_files` | Whisper large-v3-turbo q5_0 | whisper-cli, beam 5 (app args) | full file |
| `current_live` | Whisper large-v3-turbo int8 | sherpa-onnx, greedy | VAD segments |
| `phonon2` | Phonon-2 exact4x2 | onnx-asr | VAD segments |
| `nemotron` | Nemotron streaming 0.6B, 560 ms | sherpa-onnx online | streaming, full file |
| `qwen3_06b` / `qwen3_17b` | Qwen3-ASR int8 | sherpa-onnx | VAD segments |
| `cohere` | Cohere Transcribe int8 | sherpa-onnx | VAD segments |
| `parakeet_v3` | Parakeet TDT 0.6B v3 int8 | sherpa-onnx | VAD segments |

The VAD-segment engines use the app's own settings from `src/vad.py`.

To add a model:
1. Add its files to `MODELS` in `fetch.py`.
2. Add an engine with the same name to `ENGINES` in `ab_models.py`.
sherpa-onnx (pinned in `requirements.txt`) loads most new ONNX exports, so a
winner can often ship without a new runtime.

## Test set

| Clip | Length | Source | Notes |
|---|---|---|---|
| `ted_bigidea` | 5.8 min | TED-LIUM long-form, test row 6 | one speaker, clean |
| `earnings_call` | 2.5 min | Earnings-22, file 4468715 | telephone, accent; the reference keeps each "uh" and repeated word |
| `ami_meeting` | 2.5 min | AMI EN2002a, single distant mic | much overlapping speech, so all engines get a high WER |

`fetch.py` places each AMI and Earnings-22 utterance at its real time, so the
clip sounds like the real recording. Scoring uses the Whisper English
normalizer, and removes `<...>` tags and "inaudible" from the references.

## Results, 2026-10-01

Computer: 4 CPU cores, no GPU. sherpa-onnx 1.13.8, whisper.cpp 1.8.6 (CPU build).
WER in %. RTF = processing time / audio length (lower is faster).

| Engine | Meeting | Earnings | TED | Mean WER | RTF | Size MB |
|---|---|---|---|---|---|---|
| **current_files** | **47.3** | 23.0 | **2.6** | **24.3** | 1.07* | **574** |
| qwen3_06b | 57.4 | 24.0 | 2.7 | 28.1 | 0.27 | 987 |
| phonon2 | 62.4 | 20.9 | 3.1 | 28.8 | 0.10 | 736 |
| nemotron | 64.9 | **19.9** | 2.9 | 29.2 | 0.28 | 662 |
| **current_live** | 60.1 | 26.1 | 3.3 | 29.8 | 0.26 | 1037 |
| cohere | 65.9 | 22.7 | 2.8 | 30.5 | 0.14 | 2888 |
| qwen3_17b | 63.8 | 32.3 | 3.1 | 33.1 | 0.45 | 2404 |
| parakeet_v3 | 64.5 | 31.8 | 4.7 | 33.7 | **0.07** | 670 |

\* CPU only. In the app, whisper-cli uses the GPU through Vulkan and is much faster.

Decision: no change.
- The current file engine has the lowest mean WER and the smallest size.
- Part of its lead on the meeting clip comes from the method: it reads the full
  file with context, and the other engines get VAD segments.
- Cohere is not more accurate and is 5 times larger.
- Parakeet v3 lost again, as in the earlier test.
- Qwen3-ASR 1.7B is a community export, and it did worse than 0.6B.
- Phonon-2 and Nemotron are a little better than the current live model and
  faster, but the difference (about 1 point) is in the noise of a 10.8 min set.
  Test them on real recordings before you change live mode.
