# PyInstaller hook for faster_whisper
#
# faster_whisper ships ONNX model files in its assets/ directory
# (e.g. silero_vad_v5.onnx, silero_vad_v6.onnx) that are required at runtime
# for VAD (Voice Activity Detection). Without this hook PyInstaller misses them
# and transcription fails with: "Load model ... silero_vad_v6.onnx failed.
# File doesn't exist"

from PyInstaller.utils.hooks import collect_data_files

datas = collect_data_files("faster_whisper")
