# PyInstaller hook for torch
#
# torch ships CUDA runtime DLLs in torch/lib/ — cublas64_12.dll,
# cublasLt64_12.dll, cudart64_12.dll, cudnn*.dll, etc. — that ctranslate2
# needs at runtime for GPU (float16) inference.  Without this hook they are
# absent from the bundle and the model load fails with:
#   "Library cublas64_12.dll is not found or cannot be loaded"
#
# The app.zip (~1.1 GB) has enough headroom for these DLLs.
# The model is shipped separately in model.zip so the two ZIPs each stay
# under GitHub Releases' 2 GB asset limit.

from PyInstaller.utils.hooks import collect_dynamic_libs

try:
    binaries = collect_dynamic_libs("torch")
except Exception:
    binaries = []
