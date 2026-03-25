# PyInstaller hook for ctranslate2
#
# ctranslate2 dynamically loads its CUDA / CPU kernels and data files.
# Without this hook, PyInstaller misses them and the bundled exe fails
# to load the model at runtime.

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

datas = collect_data_files("ctranslate2")
binaries = collect_dynamic_libs("ctranslate2")
