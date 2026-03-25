@echo off
setlocal enabledelayedexpansion

:: ============================================================
:: Transcriber7 - Windows Build Script
::
:: Requirements before running:
::   1. Python 3.10+ installed and on PATH
::   2. Model files placed at:
::        models\faster-whisper-large-v3-turbo\
::          config.json
::          model.bin
::          preprocessor_config.json
::          tokenizer.json
::          vocabulary.json
::
:: ffmpeg is downloaded automatically if not already present.
::
:: Usage:
::   build_windows.bat              (CPU build - default)
::   build_windows.bat --cuda 11.8  (CUDA 11.8 build)
::   build_windows.bat --cuda 12.1  (CUDA 12.1 build)
::   build_windows.bat --cuda 12.4  (CUDA 12.4 build)
::   build_windows.bat --cuda 12.6  (CUDA 12.6 build)
::
:: TIP: Run this script from cmd.exe, not PowerShell.
::      PowerShell treats > as a redirect operator which breaks pip version specs.
:: ============================================================

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

set PYTHON=python
set VENV_DIR=venv
set DIST_NAME=PortableTranscriber
set DIST_DIR=dist\%DIST_NAME%
set MODEL_NAME=faster-whisper-large-v3-turbo
set MODEL_SRC=models\%MODEL_NAME%
set FFMPEG_SRC=tools\ffmpeg
set TOOLS_DIR=tools

:: Parse optional --cuda argument
set CUDA_VERSION=
set TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

:parse_args
if "%~1"=="--cuda" (
    set CUDA_VERSION=%~2
    if "!CUDA_VERSION!"=="11.8"  set TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118
    if "!CUDA_VERSION!"=="12.1"  set TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
    if "!CUDA_VERSION!"=="12.4"  set TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
    if "!CUDA_VERSION!"=="12.6"  set TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
    shift
    shift
    goto parse_args
)

echo.
echo ============================================================
echo  Transcriber7 Build Script
if defined CUDA_VERSION (
    echo  Mode: GPU ^(CUDA !CUDA_VERSION!^)
) else (
    echo  Mode: CPU only
)
echo ============================================================
echo.

:: ─────────────────────────────────────────────────────────────────────────────
echo [Step 1/8] Checking Python installation...
:: ─────────────────────────────────────────────────────────────────────────────
%PYTHON% --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: 'python' not found on PATH.
    echo Install Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    exit /b 1
)
for /f "tokens=*" %%v in ('%PYTHON% --version 2^>^&1') do echo   Found: %%v

:: ─────────────────────────────────────────────────────────────────────────────
echo.
echo [Step 2/8] Checking model files...
:: ─────────────────────────────────────────────────────────────────────────────
if not exist "%MODEL_SRC%\model.bin" (
    echo ERROR: Model file not found at %MODEL_SRC%\model.bin
    echo.
    echo Place the following files in %MODEL_SRC%\:
    echo   config.json
    echo   model.bin
    echo   preprocessor_config.json
    echo   tokenizer.json
    echo   vocabulary.json
    echo.
    echo Download from: https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo
    exit /b 1
)
echo   Model files found at %MODEL_SRC%

:: ─────────────────────────────────────────────────────────────────────────────
echo.
echo [Step 3/8] Checking / downloading ffmpeg...
:: ─────────────────────────────────────────────────────────────────────────────
if exist "%FFMPEG_SRC%\ffmpeg.exe" (
    echo   ffmpeg.exe already present — skipping download
) else (
    echo   ffmpeg.exe not found — downloading from GitHub releases...

    if not exist "%TOOLS_DIR%" mkdir "%TOOLS_DIR%"
    if not exist "%FFMPEG_SRC%" mkdir "%FFMPEG_SRC%"

    :: Download the latest static GPL Windows 64-bit build from BtbN/FFmpeg-Builds
    set FFMPEG_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip
    set FFMPEG_ZIP=%TOOLS_DIR%\ffmpeg_dl.zip
    set FFMPEG_TMP=%TOOLS_DIR%\ffmpeg_tmp

    echo   Downloading !FFMPEG_URL!...
    powershell -Command "& { $ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri '!FFMPEG_URL!' -OutFile '!FFMPEG_ZIP!' }"
    if errorlevel 1 (
        echo ERROR: Failed to download ffmpeg.
        echo Please download manually from https://github.com/BtbN/FFmpeg-Builds/releases
        echo and place ffmpeg.exe and ffprobe.exe in %FFMPEG_SRC%\
        exit /b 1
    )

    echo   Extracting...
    powershell -Command "Expand-Archive -Path '!FFMPEG_ZIP!' -DestinationPath '!FFMPEG_TMP!' -Force"
    if errorlevel 1 (
        echo ERROR: Failed to extract ffmpeg archive.
        exit /b 1
    )

    :: The zip contains a subfolder like ffmpeg-master-latest-win64-gpl\bin\
    :: Use a wildcard to handle any subfolder name variation
    for /d %%D in ("%FFMPEG_TMP%\ffmpeg-*") do (
        if exist "%%D\bin\ffmpeg.exe" (
            copy /Y "%%D\bin\ffmpeg.exe" "%FFMPEG_SRC%\ffmpeg.exe" >nul
            copy /Y "%%D\bin\ffprobe.exe" "%FFMPEG_SRC%\ffprobe.exe" >nul
        )
    )

    :: Cleanup
    if exist "%FFMPEG_TMP%" rmdir /s /q "%FFMPEG_TMP%"
    if exist "%FFMPEG_ZIP%" del "%FFMPEG_ZIP%"

    if not exist "%FFMPEG_SRC%\ffmpeg.exe" (
        echo ERROR: ffmpeg.exe extraction failed. Check archive structure.
        exit /b 1
    )
    echo   ffmpeg downloaded and extracted successfully
)

:: ─────────────────────────────────────────────────────────────────────────────
echo.
echo [Step 4/8] Creating virtual environment...
:: ─────────────────────────────────────────────────────────────────────────────
if exist "%VENV_DIR%" (
    echo   Removing existing venv...
    rmdir /s /q "%VENV_DIR%"
)
%PYTHON% -m venv "%VENV_DIR%"
if errorlevel 1 (echo ERROR: Failed to create venv && exit /b 1)
call "%VENV_DIR%\Scripts\activate.bat"
echo   Virtual environment created

:: ─────────────────────────────────────────────────────────────────────────────
echo.
echo [Step 5/8] Installing Python dependencies...
:: ─────────────────────────────────────────────────────────────────────────────
python -m pip install --upgrade pip --quiet
if errorlevel 1 (echo ERROR: pip upgrade failed && exit /b 1)

:: Install PyTorch (CPU or CUDA variant)
if defined CUDA_VERSION (
    echo   Installing PyTorch with CUDA !CUDA_VERSION! from !TORCH_INDEX_URL!...
    pip install torch torchaudio --index-url !TORCH_INDEX_URL! --quiet
) else (
    echo   Installing PyTorch ^(CPU^)...
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
)
if errorlevel 1 (echo ERROR: PyTorch installation failed && exit /b 1)

:: Install remaining dependencies (excluding torch — already done above)
echo   Installing remaining dependencies...
pip install faster-whisper customtkinter tkinterdnd2 pyinstaller --quiet
if errorlevel 1 (echo ERROR: Dependency installation failed && exit /b 1)
echo   All dependencies installed

:: ─────────────────────────────────────────────────────────────────────────────
echo.
echo [Step 6/8] Running PyInstaller...
:: ─────────────────────────────────────────────────────────────────────────────
if exist "dist\%DIST_NAME%" rmdir /s /q "dist\%DIST_NAME%"
pyinstaller Transcriber7.spec --noconfirm
if errorlevel 1 (echo ERROR: PyInstaller build failed && exit /b 1)
echo   PyInstaller build complete

:: ─────────────────────────────────────────────────────────────────────────────
echo.
echo [Step 7/8] Copying model files and ffmpeg binaries...
:: ─────────────────────────────────────────────────────────────────────────────
echo   Copying model files...
if not exist "%DIST_DIR%\models\%MODEL_NAME%" mkdir "%DIST_DIR%\models\%MODEL_NAME%"
xcopy /E /I /Y "%MODEL_SRC%\*" "%DIST_DIR%\models\%MODEL_NAME%\" >nul
if errorlevel 1 (echo ERROR: Model copy failed && exit /b 1)

echo   Copying ffmpeg binaries...
if not exist "%DIST_DIR%\ffmpeg" mkdir "%DIST_DIR%\ffmpeg"
xcopy /E /I /Y "%FFMPEG_SRC%\*" "%DIST_DIR%\ffmpeg\" >nul
if errorlevel 1 (echo ERROR: ffmpeg copy failed && exit /b 1)

:: ─────────────────────────────────────────────────────────────────────────────
echo.
echo [Step 8/8] Creating runtime directories...
:: ─────────────────────────────────────────────────────────────────────────────
if not exist "%DIST_DIR%\logs" mkdir "%DIST_DIR%\logs"
if not exist "%DIST_DIR%\temp" mkdir "%DIST_DIR%\temp"
echo   Runtime directories created

:: ─────────────────────────────────────────────────────────────────────────────
echo.
echo ============================================================
echo  BUILD COMPLETE
echo  Output: %DIST_DIR%
echo ============================================================
echo.
echo Folder contents:
dir /B "%DIST_DIR%"
echo.
echo To run: double-click %DIST_DIR%\PortableTranscriber.exe
echo To distribute: zip the entire %DIST_DIR%\ folder.
echo.

endlocal
