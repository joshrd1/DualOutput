#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"
HAL_DIR="/Library/Audio/Plug-Ins/HAL"

if [[ ! -d "${BUILD_DIR}/DualOutput.driver" || ! -d "${BUILD_DIR}/DualOutput.app" ]]; then
    echo "Build artifacts not found. Running ./build.sh first..."
    "${PROJECT_DIR}/build.sh"
fi

echo "=== Installing DualOutput.driver to ${HAL_DIR} (requires administrator password) ==="
sudo mkdir -p "${HAL_DIR}"
sudo rm -rf "${HAL_DIR}/DualOutput.driver"
sudo cp -R "${BUILD_DIR}/DualOutput.driver" "${HAL_DIR}/DualOutput.driver"
sudo chown -R root:wheel "${HAL_DIR}/DualOutput.driver"
sudo chmod -R 755 "${HAL_DIR}/DualOutput.driver"

echo "=== Installing DualOutput.app to /Applications ==="
sudo rm -rf "/Applications/DualOutput.app"
sudo cp -R "${BUILD_DIR}/DualOutput.app" "/Applications/DualOutput.app"
sudo chown -R "$(id -un):staff" "/Applications/DualOutput.app"

echo "=== Restarting macOS Core Audio Daemon (coreaudiod) ==="
sudo launchctl kickstart -kp system/com.apple.audio.coreaudiod || sudo killall coreaudiod
sleep 2

echo "=== Verifying Dual Output Registration ==="
if system_profiler SPAudioDataType | grep -q "Dual Output"; then
    echo "SUCCESS: 'Dual Output' virtual audio device is active and registered with macOS."
else
    echo "WARNING: 'Dual Output' did not immediately appear in system_profiler. Wait 2 seconds or check Audio MIDI Setup."
fi

echo "Launching /Applications/DualOutput.app..."
open "/Applications/DualOutput.app"