# Dual Output — macOS Virtual Audio Splitter & Synchronizer

A complete, production-ready virtual audio output driver (`DualOutput.driver`) and native real-time routing controller (`DualOutput.app`) for Intel x86-64 Macs running macOS Monterey (12.7.x).

## 1. Prerequisites & Verification

No paid Apple Developer account, third-party libraries, or full Xcode installation is required. Only Apple's **Command Line Tools for Xcode** are needed.

Verify your compiler in Terminal:
```bash
clang --version
xcode-select -p
```

## Crossover network

The controller includes a real-time Linkwitz-Riley crossover after each
destination path has been sample-rate converted:

- Enable **Crossover processing** and keep filters linked for a phase-aligned
  low/high pair; choose whether Output A or Output B receives the high band.
- Set the shared crossover frequency from 20 Hz to 20 kHz with the logarithmic
  slider or by dragging its marker on the response plot.
- Choose 12, 24, 36, or 48 dB per octave.
- Clear **Link A low-pass + B high-pass** to give each output its own bypass,
  low-pass, or high-pass filter, frequency, and slope.

All crossover settings are saved with the app configuration and in presets.
Existing configurations remain full range until crossover processing is enabled.

## Output routing matrices

Each output configuration includes a compact QLab-style routing matrix. The
matrix has Left and Right input rows (or one Mono row when that output's mono
switch is enabled), plus a column for every channel exposed by its selected
output device. Click a cell to activate or edit that
input-to-output-channel route; the light matrix displays its crosspoint gain in
dB, and its popover provides a per-route fader, exact-value field, and disable
button. Stereo devices start with the expected L→1 and R→2 routing; matrices
and their signal paths support up to 16 destination channels and are saved in
preferences and presets.
