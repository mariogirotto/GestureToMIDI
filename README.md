# Gesture2MIDI

A real-time hand gesture recognition system that converts hand poses and movements into MIDI messages for music production and live performance.

## Features

- **Real-time hand tracking** using MediaPipe Hands
- **Gesture recognition** with rotation-invariant feature extraction
- **3D rotation detection** (pitch, yaw, roll) for expressive control
- **MIDI output** via virtual MIDI ports
- **Continuous control** (CC messages) mapped to hand rotation
- **Note triggering** based on gesture detection
- **Per-hand configuration** for left and right hands
- **Calibration system** for rotation axes with midpoint/min/max settings

## Requirements

### Python Dependencies

```bash
pip install -r requirements.txt
```

- opencv-python==4.8.1.78
- mediapipe==0.10.9
- numpy==1.24.3
- python-rtmidi==1.5.0

### Virtual MIDI Driver (OS-Specific)

This application requires a virtual MIDI driver to create a software MIDI port that can be used by your DAW.

#### Windows

**Required:** [loopMIDI](https://www.tobias-erichsen.de/software/loopmidi.html) by Tobias Erichsen

1. Download and install loopMIDI from the official website
2. Launch loopMIDI
3. Click the **"+"** button to create a new virtual port
4. Name it (e.g., "Gesture MIDI Controller")
5. Keep loopMIDI running while using Gesture2MIDI

#### macOS

**Required:** Enable the built-in IAC (Inter-Application Communication) driver

1. Open **Audio MIDI Setup** (found in Applications > Utilities)
2. Click **Window > Show MIDI Studio** (or press ⌘2)
3. Double-click the **IAC Driver** icon
4. Check **"Device is online"**
5. Click **"+"** to add a port if none exists
6. Name it (e.g., "Gesture MIDI Controller")
7. Close the window

#### Linux

**Required:** ALSA MIDI support (usually pre-installed)

Most Linux distributions include ALSA MIDI support by default. If needed:

```bash
# Ubuntu/Debian
sudo apt-get install libasound2-dev

# Fedora
sudo dnf install alsa-lib-devel

# Create a virtual MIDI port
sudo modprobe snd-virmidi
```

## Installation

1. Clone or download this repository
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install and configure the virtual MIDI driver for your OS (see above)
4. Run the application:
   ```bash
   python gesture2midi.py
   ```

## Usage

### Controls

| Key | Action |
|-----|--------|
| `R` | Record a new gesture template |
| `S` | Save templates and mappings to disk |
| `M` | Map current gesture to a MIDI message |
| `C` | Configure rotation-to-CC mapping |
| `A` | Calibrate rotation axis (midpoint/min/max) |
| `Q` | Quit the application |

### Recording a Gesture

1. Press `R` to enter recording mode
2. Enter a name for the gesture (e.g., "fist", "open_palm", "peace")
3. Specify which hand (Left/Right)
4. Choose if the gesture is rotation-specific
5. Hold the gesture steady until recording completes

### Mapping Gestures to MIDI

1. First, record a gesture (see above)
2. Press `M` to map the gesture
3. Enter the gesture key (e.g., "fist_left")
4. Select message type: `note` or `cc`
5. Enter MIDI channel (0-15)
6. Enter note or CC number (0-127)
7. For CC messages, select the rotation axis (pitch/yaw/roll)

### Calibrating Rotation Axes

For smooth continuous control, calibrate each rotation axis:

1. Press `A` to start axis calibration
2. Select hand (Left/Right) and axis (pitch/yaw/roll)
3. Set **midpoint** position (neutral) - press SPACE
4. Set **minimum** position - press SPACE
5. Set **maximum** position - press SPACE

### Connecting to Your DAW

1. Start Gesture2MIDI and your virtual MIDI driver
2. Open your DAW (Ableton Live, FL Studio, Logic, etc.)
3. In your DAW's MIDI settings, select the virtual MIDI port as an input
4. Map MIDI CC or notes to parameters as desired

## Configuration Files

- `gesture_templates.json` - Stored gesture templates
- `gesture_midi_config.json` - MIDI mappings
- `rotation_calibration.json` - Rotation axis calibration data

## How It Works

1. **Hand Detection**: MediaPipe Hands detects 21 hand landmarks in real-time
2. **Feature Extraction**: Rotation-invariant features are extracted from finger positions
3. **Gesture Recognition**: Templates are matched using cosine similarity
4. **Rotation Tracking**: 3D hand orientation (pitch, yaw, roll) is calculated
5. **MIDI Output**: Recognized gestures trigger MIDI notes or CC messages

## Troubleshooting

### "No MIDI ports available!"
- Ensure your virtual MIDI driver is installed and running
- On Windows: Check that loopMIDI is running with at least one port created
- On macOS: Verify IAC Driver is enabled in Audio MIDI Setup

### Gestures not recognized
- Re-record the gesture in consistent lighting
- Ensure your hand is clearly visible to the camera
- Try adjusting your distance from the camera

### Jerky or unstable CC values
- Use the rotation axis calibration (`A` key) to set proper ranges
- Ensure good lighting for stable hand tracking

## License

This project is open source. Feel free to modify and distribute.

## Credits

- Built with [MediaPipe](https://mediapipe.dev/) for hand tracking
- MIDI functionality via [python-rtmidi](https://github.com/SpotlightKid/python-rtmidi)
