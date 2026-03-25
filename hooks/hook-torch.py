# PyInstaller hook for torch — CUDA runtime DLLs only
#
# ctranslate2 needs these CUDA runtime DLLs from torch/lib/ at runtime:
#   cublas64_*.dll      ~450 MB  (cuBLAS — matrix ops for float16 inference)
#   cublasLt64_*.dll    ~250 MB  (cuBLASLt — used by cuBLAS)
#   cudart64_*.dll      ~2 MB    (CUDA runtime)
#   cudnn_*.dll         ~100 MB  (cuDNN — neural net primitives)
#
# We intentionally do NOT use collect_dynamic_libs('torch') because that
# also pulls in torch_cuda.dll (~1.5 GB), torch_cpu.dll (~500 MB), and
# other PyTorch internals that ctranslate2 does not need — bloating the
# bundle past GitHub's 2 GB release asset limit.

import glob
import importlib.util
import os

def _cuda_runtime_dlls() -> list[tuple[str, str]]:
    try:
        spec = importlib.util.find_spec("torch")
        if not spec or not spec.origin:
            return []
        torch_lib = os.path.join(os.path.dirname(spec.origin), "lib")
        if not os.path.isdir(torch_lib):
            return []
        patterns = [
            "cublas64_*.dll",
            "cublasLt64_*.dll",
            "cudart64_*.dll",
            "cudnn_*.dll",
            "cudnn64_*.dll",
        ]
        dlls = []
        for pat in patterns:
            for f in glob.glob(os.path.join(torch_lib, pat)):
                dlls.append((f, "."))
        return dlls
    except Exception:
        return []

binaries = _cuda_runtime_dlls()
