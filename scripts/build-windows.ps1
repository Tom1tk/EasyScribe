# Build the single-file EasyScribe.exe on Windows.
#
# Prerequisites:
#   * Rust MSVC toolchain (rustup default stable-x86_64-pc-windows-msvc)
#   * CMake and the Visual Studio C++ toolchain. sherpa-onnx 1.12.9 is built
#     from the source vendored by sherpa-rs-sys with TTS disabled, so the
#     GPL-3.0 espeak-ng/piper components are not compiled in. (1.13.x changed
#     the C API; sherpa-rs 0.6.8 binds 1.12.9.)
#   * models/, ffmpeg/ (or tools/ffmpeg/) and
#     whispercpp/ggml-large-v3-turbo-q5_0.bin populated, plus assets/,
#     licenses/ and LICENSE
#
# Usage:  pwsh -File scripts/build-windows.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# The official 1.12.9 static libraries use the static MSVC runtime. Match it
# for Rust and for the whisper.cpp CMake build, otherwise the link fails.
# advapi32 supplies the ETW and registry imports that onnxruntime's static
# library does not always carry.
$env:RUSTFLAGS = (($env:RUSTFLAGS, "-C target-feature=+crt-static", "-C link-arg=advapi32.lib") -join " ").Trim()
$env:CMAKE_MSVC_RUNTIME_LIBRARY = "MultiThreaded"
$env:CMAKE_POLICY_DEFAULT_CMP0091 = "NEW"
$env:GGML_NATIVE = "OFF"
$env:GGML_AVX = "ON"
$env:GGML_AVX2 = "ON"
$env:GGML_FMA = "ON"
$env:GGML_F16C = "ON"

$payload = Join-Path $root "payload"
if (Test-Path $payload) { Remove-Item $payload -Recurse -Force }
New-Item -ItemType Directory -Force -Path $payload | Out-Null

$ffmpegSrc = if (Test-Path (Join-Path $root "ffmpeg")) {
    Join-Path $root "ffmpeg"
} elseif (Test-Path (Join-Path $root "tools\ffmpeg")) {
    Join-Path $root "tools\ffmpeg"
} else {
    throw "missing payload folder: expected ffmpeg/ or tools/ffmpeg/"
}
foreach ($name in @("models", "whispercpp", "assets", "licenses")) {
    $src = Join-Path $root $name
    if (-not (Test-Path $src)) { throw "missing payload folder: $src" }
    Copy-Item $src (Join-Path $payload $name) -Recurse
}
Copy-Item $ffmpegSrc (Join-Path $payload "ffmpeg") -Recurse

# The app needs the GGUF weights and the bundled whisper-cli (Vulkan when a
# driver exists, CPU otherwise) plus its DLLs.
$whisperDir = Join-Path $payload "whispercpp"
if (-not (Test-Path (Join-Path $whisperDir "ggml-large-v3-turbo-q5_0.bin"))) {
    throw "missing whispercpp/ggml-large-v3-turbo-q5_0.bin"
}
if (-not (Test-Path (Join-Path $whisperDir "whisper-cli.exe"))) {
    throw "missing whispercpp/whisper-cli.exe (build it with the Vulkan workflow)"
}
# Copy the VC++ runtime beside the CLI, matching the release workflow, so a
# clean PC does not need the redistributable installed system-wide.
if ($env:VCToolsRedistDir) {
    $crt = Get-ChildItem "$env:VCToolsRedistDir\x64\Microsoft.VC*.CRT" -Directory -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($crt) { Copy-Item "$($crt.FullName)\*.dll" $whisperDir -Force }
    $openmp = Get-ChildItem "$env:VCToolsRedistDir\x64\Microsoft.VC*.OpenMP" -Directory -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($openmp) { Copy-Item "$($openmp.FullName)\vcomp140.dll" $whisperDir -Force }
}
foreach ($dll in @(
    "whisper.dll", "ggml.dll", "ggml-base.dll", "ggml-cpu.dll",
    "ggml-vulkan.dll", "vulkan-1.dll",
    "vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "vcomp140.dll"
)) {
    if (-not (Test-Path (Join-Path $whisperDir $dll))) {
        throw "missing whispercpp/$dll (copy the whisper-cli build folder and the VC++ runtime DLLs)"
    }
}
Copy-Item (Join-Path $root "LICENSE") (Join-Path $payload "LICENSE")

# Build the app and packer in one cargo invocation so the static sherpa
# feature applies to both; a separate `cargo run -p easyscribe-pack` would
# rebuild core without it and download a dynamic sherpa just to package.
$target = if ($env:CARGO_TARGET_DIR) {
    $env:CARGO_TARGET_DIR
} else {
    Join-Path $root "target"
}

# sherpa-onnx 1.12.9's cmake/show-info.cmake calls wmic, which Windows 11 no
# longer ships. Patch the vendored copy after cargo fetch so the source build
# configures under CMake 4.
cargo fetch
if ($LASTEXITCODE -ne 0) { throw "cargo fetch failed" }
$showInfo = Get-ChildItem "$env:USERPROFILE\.cargo\registry\src" -Recurse -Filter show-info.cmake -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -match "sherpa-rs-sys" -and $_.FullName -match "sherpa-onnx" }
if (-not $showInfo) { throw "sherpa-onnx show-info.cmake not found" }
$old = 'string(REPLACE "\n" ";" SHERPA_ONNX_OS_LIST ${SHERPA_ONNX_OS_TWO_LINES})'
$new = 'string(REPLACE "\n" ";" SHERPA_ONNX_OS_LIST "${SHERPA_ONNX_OS_TWO_LINES}")'
$listLine = '  list(GET SHERPA_ONNX_OS_LIST 1 SHERPA_ONNX_OS)'
$listGuard = @'
  if(SHERPA_ONNX_OS_LIST)
    list(GET SHERPA_ONNX_OS_LIST 1 SHERPA_ONNX_OS)
  else()
    set(SHERPA_ONNX_OS "Windows")
  endif()
'@
foreach ($file in $showInfo) {
    $text = Get-Content $file.FullName -Raw
    if (-not $text.Contains($new)) {
        if (-not $text.Contains($old)) { throw "Unexpected show-info.cmake string line: $($file.FullName)" }
        $text = $text.Replace($old, $new)
    }
    if (-not $text.Contains('if(SHERPA_ONNX_OS_LIST)')) {
        if (-not $text.Contains($listLine)) { throw "Unexpected show-info.cmake list line: $($file.FullName)" }
        $text = $text.Replace($listLine, $listGuard)
    }
    Set-Content -Path $file.FullName -Value $text -NoNewline
}

# A restored target can keep the old, unpatched sherpa source in OUT_DIR.
if (Test-Path (Join-Path $target "release")) {
    foreach ($pattern in @("sherpa-rs-sys-*", "whisper-rs-sys-*", "whisper-rs-*")) {
        Get-ChildItem (Join-Path $target "release\build") -Directory -Filter $pattern -ErrorAction SilentlyContinue |
            Remove-Item -Recurse -Force
        Get-ChildItem (Join-Path $target "release\.fingerprint") -Directory -Filter $pattern -ErrorAction SilentlyContinue |
            Remove-Item -Recurse -Force
    }
}

cargo build --release -p easyscribe -p easyscribe-pack `
    --no-default-features --features easyscribe/release-static
if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }

$sherpaCache = Get-ChildItem (Join-Path $target "release\build") -Recurse -Filter CMakeCache.txt -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -match "sherpa-rs-sys" } | Select-Object -First 1
if (-not $sherpaCache) { throw "sherpa-onnx CMakeCache.txt was not produced" }
if (-not (Select-String -Path $sherpaCache.FullName -Pattern '^SHERPA_ONNX_ENABLE_TTS:BOOL=OFF$')) {
    throw "sherpa-onnx was not built with text-to-speech disabled"
}
Write-Host "sherpa-onnx TTS disabled (espeak-ng/piper not linked)."

$version = (Select-String -Path Cargo.toml -Pattern '^version\s*=\s*"([^"]+)"' | Select-Object -First 1).Matches[0].Groups[1].Value
if (-not $version) { throw "Could not read the workspace version from Cargo.toml" }
& (Join-Path $target "release\easyscribe-pack.exe") `
    --binary (Join-Path $target "release\easyscribe.exe") `
    --payload $payload `
    --out "dist\EasyScribe.exe" `
    --version $version
if ($LASTEXITCODE -ne 0) { throw "pack failed" }

Write-Host "Built dist\EasyScribe.exe"
Write-Host "Optional check: dist\EasyScribe.exe --selftest (extracts the runtime once)"
