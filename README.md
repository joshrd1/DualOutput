# Dual Output

Dual Output is a native macOS virtual-audio splitter and synchronizer. It
publishes a **Dual Output** virtual audio device that macOS and audio
applications can send sound to, then routes that signal to two independently
configured physical output devices. It is intended for combinations such as
speakers plus headphones, multi-speaker systems, or a low/high-band
crossover system.

The project contains two components:

- `DualOutput.driver` — a Core Audio HAL plug-in that provides the virtual
  audio device.
- `DualOutput.app` — the menu-bar controller that receives, synchronizes,
  processes, and routes audio to Output A and Output B.

The controller keeps the two output paths synchronized while accommodating
different device sample rates. Each path has its own enable state, device,
delay, mono option, gain trim, routing matrix, and crossover configuration.

![Dual Output Controller screenshot](App%20Screenshot.png)

> AI authorship: this project was made entirely with AI, using Gemini and Codex.

## Requirements

- An Intel (x86-64) Mac running macOS 12 Monterey or later.
- Apple’s Xcode Command Line Tools.

Install the Command Line Tools, if needed:

```bash
xcode-select --install
```

You can verify the toolchain with:

```bash
clang --version
xcode-select -p
```

## Build and install

Run these commands from this release folder:

```bash
./build.sh
./install.sh
```

`build.sh` creates both bundles in `build/`. `install.sh` installs the driver
to `/Library/Audio/Plug-Ins/HAL/`, installs the controller in
`/Applications/DualOutput.app`, restarts Core Audio, launches the controller,
and checks that the virtual device has registered.

The bundles are ad-hoc signed locally. If macOS asks for permission to open
the controller, approve it in the normal macOS security prompt or in
**System Settings/Preferences → Privacy & Security**.

### First launch

1. Allow the controller’s microphone permission request. If it was denied,
   enable it in **System Settings/Preferences → Privacy & Security →
   Microphone** and relaunch the app.
2. Keep **Dual Output Controller** running. The `Dual Output` virtual device
   is intentionally published only while the controller is running.
3. In macOS Sound settings or Control Center, choose **Dual Output** as the
   system output. The controller’s **Set as System Output** button performs
   the same action.
4. In the controller, choose the physical devices for **Output A** and
   **Output B**, enable the paths you need, and play audio.

The application lives in the menu bar. Closing the window or pressing
**Command-W** only hides the controller; routing continues in the background.
Use **Quit Dual Output** from the menu bar or app menu when you want to stop
routing and withdraw the virtual device.

## Signal path

Audio sent to the Dual Output virtual device is processed independently for
each destination path:

```text
Dual Output virtual device
  → sample-rate conversion and drift compensation
  → per-output delay and gain
  → shared 31-band graphic EQ
  → per-output crossover filter
  → optional mono fold-down
  → per-output routing matrix
  → Output A / Output B hardware device
```

The normal stereo defaults are Left → channel 1 and Right → channel 2 on each
selected output device. Device changes are saved and reconnected when the
device returns.

## Controller guide

### Output A and Output B

Each output path can be configured separately:

- **Enable** starts or stops that path without affecting the other one.
- **Mono** folds the signal to mono before the routing matrix.
- **Gain** applies a per-output trim from −24 dB to +6 dB. Use the slider or
  enter an exact value in the field.
- **Time Delay** adds 0–500 ms of delay in 0.5 ms steps, useful for aligning
  speakers at different distances or compensating for processing latency.
- **Meters** show stereo dBFS level, peak hold, and clip indication.

The global **Master** control adjusts both paths. **A / B** uses equal-power
balance, allowing one destination to be reduced while the other remains at
full level.

### Routing matrices

Each output has a compact routing matrix with a row for Left and Right input,
or one Mono row when mono is enabled. There is one column for each channel
offered by the selected destination device, up to 16 channels.

- Click a cell to inspect or set its crosspoint gain.
- In the popover, adjust the fader, type an exact dB value, or disable the
  route. Enter `-inf` to disable a route.
- Right-click a cell to toggle that route.
- Command-click a cell to disable it.

Routing and crosspoint gains are included in saved presets.

### 31-band graphic EQ

The shared 1/3-octave graphic EQ runs on both output paths before the
crossover. It provides 31 bands from 20 Hz to 20 kHz, each adjustable from
−12 dB to +12 dB.

- Enable it with **Enable 31-band Graphic EQ**.
- Click the response graph or **Open EQ Sliders…** for the full editor.
- Double-click or Option-click a fader to reset that band to 0 dB.
- Use **Flatten All (0 dB)** to reset every band.

The response graph previews the combined EQ curve. EQ settings are saved in
preferences and presets.

### Linkwitz-Riley crossover

Enable **Crossover processing** to filter the output paths after the EQ.

- With **Link crossover filters** enabled, Output A and Output B form a
  phase-aligned complementary low-pass/high-pass pair. Choose which output
  receives the high band, set a shared crossover frequency from 20 Hz to
  20 kHz, and choose a 12–96 dB/octave slope.
- With the filters unlinked, each path can independently use bypass,
  low-pass, or high-pass mode, frequency, and slope.
- Drag a marker on the response graph to change its frequency directly.

The crossover starts bypassed, so existing configurations remain full range
until processing is enabled.

### Presets and undo

The preset menu can save a new preset, overwrite an existing preset, rename
one, delete one, or load one. A preset stores device selections, output
enable/mono states, routing, master and balance, gain trim, delay, EQ, and
crossover settings. The active preset is marked **Edited** whenever the live
configuration no longer matches it.

Use **Command-Z** to undo a setting or preset-management action, and
**Shift-Command-Z** to redo it. The controller retains up to 100 undo levels.

## Troubleshooting

### “Dual Output” is missing from Sound settings

1. Confirm that **Dual Output Controller** is running; the virtual device is
   deliberately absent when it is not.
2. Run `./install.sh` again from this folder. It rebuilds when necessary,
   reinstalls the driver, restarts Core Audio, and launches the controller.
3. Confirm the driver is installed at
   `/Library/Audio/Plug-Ins/HAL/DualOutput.driver`.
4. If the controller reports that microphone access is denied, grant the
   permission and relaunch it.

### Audio is silent

1. Set **Dual Output** as the macOS system output, or choose it directly in
   the audio application.
2. Select and enable a physical device for at least one output path.
3. Check master volume, A/B balance, output trim, route gains, and whether a
   routing-matrix cell is disabled.
4. Verify that crossover processing is not intentionally filtering the band
   you expect to hear.

### An output device disconnects

The controller keeps its desired device selection and attempts to reconnect
when the device becomes available again. Select another device if you do not
want to wait for reconnection.

## Uninstall

To remove the app, driver, and saved Dual Output preferences, run:

```bash
./uninstall.sh
```

The script stops the controller, removes the installed bundles, deletes the
`com.dualoutput.app` preferences domain, and restarts Core Audio. It requires
an administrator password.

## Project layout

```text
app/                 Native AppKit controller source and app metadata
driver/              Core Audio HAL plug-in source and driver metadata
build.sh             Builds the driver and controller bundles
install.sh           Installs, launches, and verifies Dual Output
uninstall.sh         Removes installed components and preferences
App Screenshot.png   Controller screenshot
```
