"""
version_resource.py - Windows file-version resource for the PyInstaller specs.

Shown in Explorer under Properties > Details. Both specs import this so the
app and the launcher always carry the same version as src/config.py.
"""
import re
from pathlib import Path

from PyInstaller.utils.win32.versioninfo import (
    FixedFileInfo,
    StringFileInfo,
    StringStruct,
    StringTable,
    VarFileInfo,
    VarStruct,
    VSVersionInfo,
)

_CONFIG = Path(__file__).parent.parent / "src" / "config.py"


def app_version() -> str:
    """APP_VERSION from src/config.py, for example "3.0.0-beta1"."""
    match = re.search(r'^APP_VERSION\s*=\s*"([^"]+)"', _CONFIG.read_text(encoding="utf-8"), re.M)
    if not match:
        raise RuntimeError(f"APP_VERSION not found in {_CONFIG}")
    return match.group(1)


def version_info(description: str, filename: str) -> VSVersionInfo:
    version = app_version()
    # Windows needs four numbers; "3.0.0-beta1" -> (3, 0, 0, 0).
    numbers = [int(n) for n in re.findall(r"\d+", version.split("-")[0])][:3]
    numbers += [0] * (4 - len(numbers))
    nums = tuple(numbers)
    return VSVersionInfo(
        ffi=FixedFileInfo(filevers=nums, prodvers=nums),
        kids=[
            StringFileInfo([
                StringTable("040904B0", [
                    StringStruct("CompanyName", "EasyScribe"),
                    StringStruct("FileDescription", description),
                    StringStruct("FileVersion", version),
                    StringStruct("InternalName", filename.rsplit(".", 1)[0]),
                    StringStruct("OriginalFilename", filename),
                    StringStruct("ProductName", "EasyScribe"),
                    StringStruct("ProductVersion", version),
                ])
            ]),
            VarFileInfo([VarStruct("Translation", [1033, 1200])]),
        ],
    )
