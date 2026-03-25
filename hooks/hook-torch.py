# PyInstaller hook for torch
#
# torch ships CUDA runtime DLLs in torch/lib/ — cublas64_12.dll,
# cublasLt64_12.dll, cudart64_12.dll, cudnn*.dll, etc. — that ctranslate2
# needs at runtime for GPU (float16) inference.  Without this hook they are
# absent from the bundle and the model load fails with:
#   "Library cublas64_12.dll is not found or cannot be loaded"
#
# When torch is installed CPU-only these DLLs simply won't exist, so the
# hook collects nothing — the build still succeeds and falls back to CPU.

from PyInstaller.utils.hooks import collect_dynamic_libs

try:
    binaries = collect_dynamic_libs("torch")
except Exception:
    binaries = []
