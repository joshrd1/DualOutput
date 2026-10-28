#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"

echo "=== Verifying Build Environment ==="
if ! command -v clang >/dev/null 2>&1; then
    echo "Error: 'clang' compiler not found."
    echo "Run: xcode-select --install"
    exit 1
fi

SDK_PATH="$(xcrun --sdk macosx --show-sdk-path 2>/dev/null || true)"
if [[ -z "${SDK_PATH}" || ! -d "${SDK_PATH}" ]]; then
    echo "Error: macOS SDK not found. Install Xcode Command Line Tools via: xcode-select --install"
    exit 1
fi

echo "Using clang: $(clang --version | head -n 1)"
echo "Using SDK:   ${SDK_PATH}"

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}/DualOutput.driver/Contents/MacOS"
mkdir -p "${BUILD_DIR}/DualOutput.app/Contents/MacOS"
mkdir -p "${BUILD_DIR}/DualOutput.app/Contents/Resources"

echo "=== Generating AppIcon.icns Desktop Icon ==="
clang \
    -target x86_64-apple-macos12.0 \
    -isysroot "${SDK_PATH}" \
    -ObjC \
    -fobjc-arc \
    -O2 \
    -framework Cocoa \
    "${PROJECT_DIR}/app/generate_icon.m" \
    -o "${BUILD_DIR}/generate_icon"

"${BUILD_DIR}/generate_icon" "${BUILD_DIR}/AppIcon.iconset"
iconutil -c icns "${BUILD_DIR}/AppIcon.iconset" -o "${BUILD_DIR}/DualOutput.app/Contents/Resources/AppIcon.icns"
rm -rf "${BUILD_DIR}/AppIcon.iconset" "${BUILD_DIR}/generate_icon"

echo "=== Building DualOutput.driver (CoreAudio HAL AudioServerPlugIn) ==="
clang \
    -target x86_64-apple-macos12.0 \
    -isysroot "${SDK_PATH}" \
    -std=c11 \
    -O3 \
    -Wall -Wextra -Werror \
    -bundle \
    -framework CoreAudio \
    -framework CoreFoundation \
    "${PROJECT_DIR}/driver/DualOutputPlugIn.c" \
    -o "${BUILD_DIR}/DualOutput.driver/Contents/MacOS/DualOutput"

cp "${PROJECT_DIR}/driver/Info.plist" "${BUILD_DIR}/DualOutput.driver/Contents/Info.plist"
codesign --force --deep --sign - "${BUILD_DIR}/DualOutput.driver"

echo "=== Building DualOutput.app (Native AppKit + AUHAL Controller) ==="
clang \
    -target x86_64-apple-macos12.0 \
    -isysroot "${SDK_PATH}" \
    -ObjC \
    -fobjc-arc \
    -std=c11 \
    -O3 \
    -Wall -Wextra -Werror \
    -framework Cocoa \
    -framework CoreAudio \
    -framework AudioToolbox \
    -framework AVFoundation \
    -framework CoreFoundation \
    "${PROJECT_DIR}/app/main.m" \
    -o "${BUILD_DIR}/DualOutput.app/Contents/MacOS/DualOutputApp"

cp "${PROJECT_DIR}/app/Info.plist" "${BUILD_DIR}/DualOutput.app/Contents/Info.plist"
codesign --force --deep --sign - "${BUILD_DIR}/DualOutput.app"

echo "=== Build Complete ==="
echo "Driver bundle: ${BUILD_DIR}/DualOutput.driver"
echo "App bundle:    ${BUILD_DIR}/DualOutput.app"
