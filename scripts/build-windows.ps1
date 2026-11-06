# Build the single-file EasyScribe.exe on Windows.
#
# Prerequisites:
#   * Rust MSVC toolchain (rustup default stable-x86_64-pc-windows-msvc)
#   * sherpa-onnx 1.12.9 Win x64 static libraries:
#       sherpa-onnx-v1.12.9-win-x64-static.tar.bz2
#     Set SHERPA_LIB_PATH to the extracted folder that contains lib/.
#     (1.13.x changed the C API; sherpa-rs 0.6.8 binds 1.12.9.)
#   * models/, ffmpeg/ (or tools/ffmpeg/) and
#     whispercpp/ggml-large-v3-turbo-q5_0.bin populated, plus assets/,
#     licenses/ and LICENSE
#
# Usage:  pwsh -File scripts/build-windows.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not $env:SHERPA_LIB_PATH) {
    throw "SHERPA_LIB_PATH must point at the extracted sherpa-onnx 1.12.9 static folder (the one containing lib/)."
}

# The official 1.12.9 static libraries use the static MSVC runtime. Match it
# for Rust and for the whisper.cpp CMake build, otherwise the link fails.
# advapi32 supplies the ETW and registry imports that onnxruntime and
# espeak-ng use; their static libs do not always carry the defaultlib.
$env:RUSTFLAGS = (($env:RUSTFLAGS, "-C target-feature=+crt-static", "-C link-arg=advapi32.lib") -join " ").Trim()
$env:CMAKE_MSVC_RUNTIME_LIBRARY = "MultiThreaded"
$env:CMAKE_POLICY_DEFAULT_CMP0091 = "NEW"
$env:GGML_NATIVE = "OFF"
$env:GGML_AVX = "ON"
$env:GGML_AVX2 = "ON"

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
Copy-Item (Join-Path $root "LICENSE") (Join-Path $payload "LICENSE")

# Build the app and packer in one cargo invocation so the static sherpa
# feature applies to both; a separate `cargo run -p easyscribe-pack` would
# rebuild core without it and download a dynamic sherpa just to package.
cargo build --release -p easyscribe -p easyscribe-pack `
    --no-default-features --features easyscribe/release-static
if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }

$target = if ($env:CARGO_TARGET_DIR) {
    $env:CARGO_TARGET_DIR
} else {
    Join-Path $root "target"
}

& (Join-Path $target "release\easyscribe-pack.exe") `
    --binary (Join-Path $target "release\easyscribe.exe") `
    --payload $payload `
    --out "dist\EasyScribe.exe" `
    --version "3.2.3"
if ($LASTEXITCODE -ne 0) { throw "pack failed" }

Write-Host "Built dist\EasyScribe.exe"
Write-Host "Optional check: dist\EasyScribe.exe --selftest (extracts the runtime once)"
