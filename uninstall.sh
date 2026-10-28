#!/usr/bin/env bash
set -euo pipefail

HAL_DIR="/Library/Audio/Plug-Ins/HAL"

echo "=== Stopping DualOutput.app if running ==="
killall DualOutputApp 2>/dev/null || true

echo "=== Removing DualOutput.driver and DualOutput.app ==="
sudo rm -rf "${HAL_DIR}/DualOutput.driver"
sudo rm -rf "/Applications/DualOutput.app"

echo "=== Clearing User Preferences ==="
defaults delete com.dualoutput.app 2>/dev/null || true

echo "=== Restarting Core Audio Daemon (coreaudiod) ==="
sudo launchctl kickstart -kp system/com.apple.audio.coreaudiod || sudo killall coreaudiod
sleep 1

echo "Uninstall complete."