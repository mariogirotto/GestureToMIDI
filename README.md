# GestureToMIDI

A real-time hand gesture recognition system that converts hand poses and movements into MIDI messages for music production and interactive audio control.

## Features

- **Real-time Hand Detection**: Uses MediaPipe for robust hand landmark detection supporting both left and right hands
- **Gesture Recognition**: Recognizes multiple hand gestures including open hands, closed fists, finger positions, and symbolic gestures
- **Rotation-Aware Analysis**: Detects hand orientation with pitch, yaw, and roll angles for expressive control
- **MIDI Output**: Triggers MIDI notes, control changes (CC), and pitch bend messages based on recognized gestures
- **Continuous Control**: Maps hand movements and rotations to continuous MIDI parameters for smooth musical expression
- **Multi-hand Support**: Can track and process both left and right hands simultaneously
- **Customizable Mappings**: Easily configure gesture-to-MIDI associations for your specific needs

## Requirements

- Python 3.7+
- Webcam for real-time hand detection

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script:
```bash
python gesture2midi.py
```

The application will start capturing video from your webcam and begin recognizing hand gestures in real-time. Detected gestures are converted to MIDI messages that can be sent to your DAW or music software.

## How It Works

1. **Hand Detection**: Captures video from webcam and detects hand landmarks in real-time
2. **Feature Extraction**: Extracts geometric features from hand landmarks including joint angles and spatial relationships
3. **Gesture Recognition**: Analyzes hand poses to identify specific gestures
4. **Hand Orientation**: Calculates pitch, yaw, and roll angles for rotation-based control
5. **MIDI Generation**: Converts recognized gestures and parameters to MIDI messages
6. **MIDI Output**: Sends MIDI messages to external instruments or DAW plugins

## Hand Parameters

The system analyzes and monitors:
- **Joint Angles**: Angles between finger joints to detect hand pose and configuration
- **Rotation Angles**: Pitch (up/down tilt), yaw (left/right turn), roll (twist/rotation)
- **Scale Parameters**: Relative finger positions and distances for gesture variations

## MIDI Control

Gestures can generate:
- **Note Messages**: Trigger MIDI note on/off events for triggering sounds
- **Control Changes (CC)**: Send CC values (0-127) to control dynamic parameters
- **Pitch Bend**: Send pitch bend messages for continuous pitch expression

## Tips for Best Results

- Ensure good lighting for accurate hand detection
- Position your hands clearly within the webcam frame
- Use clear, deliberate gestures for reliable recognition
- The system uses smoothing to reduce jitter and improve stability
- Perform calibration at startup for optimal gesture detection

## Future Enhancements

- Interactive GUI for visualization and testing
- Gesture recording and custom template creation
- Advanced machine learning-based gesture recognition
- Multi-device MIDI output support
- Gesture sequence recording and playback
- Real-time gesture feedback and visualization

## License

Created as a hand gesture to MIDI conversion tool for interactive music production.
