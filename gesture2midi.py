import cv2
import mediapipe as mp
import numpy as np
import rtmidi
import json
import math
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
from pathlib import Path

# Configuration
CONFIG_FILE = "gesture_midi_config.json"
CALIBRATION_FRAMES = 15
SMOOTHING_WINDOW = 3

@dataclass
class MIDIMapping:
    """MIDI message configuration for a gesture"""
    message_type: str  # 'note', 'cc', 'pitch_bend'
    channel: int       # 0-15
    note_or_cc: int    # 0-127
    value_on: int      # 0-127
    value_off: int     # 0-127
    continuous: bool   # If True, maps gesture parameter (0.0-1.0) to MIDI value
    use_rotation: bool # If True, this gesture is rotation-specific (not invariant)
    rotation_axis: str = 'pitch'  # 'pitch', 'yaw', or 'roll' - which axis to use for continuous control

@dataclass 
class GestureTemplate:
    """Stored gesture signature"""
    name: str
    pose_features: np.ndarray
    hand_label: str
    threshold: float = 0.4

@dataclass
class RotationCalibration:
    """Stores calibration data for rotation axis"""
    midpoint: float
    min_val: float
    max_val: float

    def normalize(self, value: float) -> float:
        """Normalize value based on calibration (0.0 - 1.0 range)"""
        # Handle wrap-around for angles
        delta = value - self.midpoint
        if delta > 0.5:
            delta -= 1.0
        elif delta < -0.5:
            delta += 1.0

        # Calculate range from midpoint
        min_range = self.midpoint - self.min_val
        if min_range > 0.5:
            min_range -= 1.0
        elif min_range < -0.5:
            min_range += 1.0
        min_range = abs(min_range)

        max_range = self.max_val - self.midpoint
        if max_range > 0.5:
            max_range -= 1.0
        elif max_range < -0.5:
            max_range += 1.0
        max_range = abs(max_range)

        # Normalize
        if delta >= 0:
            normalized = 0.5 + (delta / (max_range * 2)) if max_range > 0 else 0.5
        else:
            normalized = 0.5 - (abs(delta) / (min_range * 2)) if min_range > 0 else 0.5

        return float(np.clip(normalized, 0.0, 1.0))

class HandFeatureExtractor:
    """Extracts geometric features from hand landmarks"""

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
            model_complexity=1
        )

    def get_hand_coordinate_system(self, landmarks, hand_label: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate orthonormal basis vectors for hand"""
        lm = landmarks.landmark
        points = np.array([[p.x, p.y, p.z] for p in lm])

        wrist = points[0]
        middle_mcp = points[9]
        index_mcp = points[5]
        pinky_mcp = points[17]

        # Y-axis: Wrist to Middle finger (along the hand)
        y_axis = middle_mcp - wrist
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)

        # X-axis: perpendicular to plane formed by Index-Middle-Pinky
        palm_vec = pinky_mcp - index_mcp
        x_axis = np.cross(palm_vec, y_axis)
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)

        # Z-axis: cross product (points out of palm)
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)

        # For right hands, flip Z so it consistently points out of the palm
        if hand_label == "Right":
            z_axis = -z_axis
            x_axis = -x_axis

        return wrist, x_axis, y_axis, z_axis

    def get_rotation_angles(self, landmarks, hand_label: str = None) -> Tuple[float, float, float]:
        """Extract stable Euler angles using rotation matrix decomposition"""
        wrist, x_axis, y_axis, z_axis = self.get_hand_coordinate_system(landmarks, hand_label)

        # Build rotation matrix with camera coordinate system
        # Camera: X right, Y down, Z forward (towards camera)
        # We need to map our hand axes to this
        R = np.column_stack([x_axis, y_axis, z_axis])

        # Extract angles using robust method that avoids gimbal lock
        # This uses the full rotation matrix to calculate stable angles

        # Pitch: Rotation around X-axis (tilting hand up/down)
        # Calculate from Y and Z components
        pitch = math.atan2(-R[2, 1], math.sqrt(R[0, 1]**2 + R[1, 1]**2))
        pitch_norm = (pitch + math.pi/2) / math.pi
        pitch_norm = float(np.clip(pitch_norm, 0.0, 1.0))

        # Yaw: Rotation around Y-axis (turning hand left/right)
        # Use X and Z components of the forward vector
        yaw = math.atan2(R[0, 2], R[2, 2])
        yaw_norm = (yaw + math.pi) / (2 * math.pi)
        yaw_norm = float(np.clip(yaw_norm, 0.0, 1.0))

        # Roll: Rotation around Z-axis (twisting hand/tilting side to side)
        # Calculate from X-axis orientation relative to horizontal plane
        # Use the angle between hand's X-axis and camera's horizontal plane
        roll = math.atan2(R[1, 0], R[0, 0])
        roll_norm = (roll + math.pi) / (2 * math.pi)
        roll_norm = float(np.clip(roll_norm, 0.0, 1.0))

        return pitch_norm, yaw_norm, roll_norm

    def get_rotation_invariant_features(self, landmarks) -> np.ndarray:
        """Extract features that don't change with hand rotation"""
        lm = landmarks.landmark
        points = np.array([[p.x, p.y, p.z] for p in lm])

        wrist = points[0]
        index_mcp = points[5]
        middle_mcp = points[9]
        pinky_mcp = points[17]

        y_axis = middle_mcp - wrist
        y_len = np.linalg.norm(y_axis)
        if y_len < 1e-8:
            return np.zeros(20, dtype=np.float32)
        y_axis = y_axis / y_len

        palm_vec = pinky_mcp - index_mcp
        palm_vec = palm_vec - np.dot(palm_vec, y_axis) * y_axis
        x_len = np.linalg.norm(palm_vec)
        if x_len < 1e-8:
            return np.zeros(20, dtype=np.float32)
        x_axis = palm_vec / x_len

        z_axis = np.cross(x_axis, y_axis)
        z_len = np.linalg.norm(z_axis)
        if z_len > 1e-8:
            z_axis = z_axis / z_len

        rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

        centered = points - wrist
        local_points = np.dot(centered, rotation_matrix)

        features = []

        fingers = [
            (5, 6, 7, 8, 'index'),
            (9, 10, 11, 12, 'middle'),
            (13, 14, 15, 16, 'ring'),
            (17, 18, 19, 20, 'pinky')
        ]

        thumb = (2, 3, 4)

        def joint_angle(p1_idx, p2_idx, p3_idx):
            v1 = local_points[p1_idx] - local_points[p2_idx]
            v2 = local_points[p3_idx] - local_points[p2_idx]
            norm = np.linalg.norm(v1) * np.linalg.norm(v2)
            if norm < 1e-8:
                return 0.0
            cos_angle = np.dot(v1, v2) / norm
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            return np.arccos(cos_angle) / math.pi

        hand_scale = np.linalg.norm(local_points[5] - local_points[17]) + 1e-8

        for mcp, pip, dip, tip, name in fingers:
            tip_dist = np.linalg.norm(local_points[tip])
            features.append(tip_dist / hand_scale)
            features.append(joint_angle(0, mcp, pip))
            features.append(joint_angle(mcp, pip, dip))

        thumb_mcp, thumb_ip, thumb_tip = thumb
        thumb_dist = np.linalg.norm(local_points[thumb_tip])
        features.append(thumb_dist / hand_scale)
        features.append(joint_angle(0, thumb_mcp, thumb_ip))
        features.append(joint_angle(thumb_mcp, thumb_ip, thumb_tip))

        thumb_tip_pos = local_points[4]
        for _, _, _, tip_idx, _ in fingers:
            dist = np.linalg.norm(thumb_tip_pos - local_points[tip_idx])
            features.append(dist / hand_scale)

        return np.array(features, dtype=np.float32)

    def get_global_features(self, landmarks, hand_label: str = None) -> Dict[str, float]:
        """Get non-invariant features"""
        pitch, yaw, roll = self.get_rotation_angles(landmarks, hand_label)
        lm = landmarks.landmark

        return {
            'pitch': pitch,
            'yaw': yaw,
            'roll': roll,
            'pos_x': lm[9].x,
            'pos_y': lm[9].y,
        }

class GestureRecognizer:
    """Recognizes gestures based on trained templates"""

    def __init__(self):
        self.templates: List[GestureTemplate] = []
        self.feature_extractor = HandFeatureExtractor()
        self.rotation_smoothing = {'Left': deque(maxlen=10), 'Right': deque(maxlen=10)}

    def add_template(self, name: str, features: np.ndarray, hand_label: str, rotation_specific: bool = False):
        """Record new gesture template"""
        template = GestureTemplate(
            name=name, 
            pose_features=features, 
            hand_label=hand_label,
            threshold=0.5 if rotation_specific else 0.4
        )

        for i, t in enumerate(self.templates):
            if t.name == name and t.hand_label == hand_label:
                self.templates[i] = template
                print(f"Updated template: {name} for {hand_label}")
                return

        self.templates.append(template)
        print(f"Added template: {name} for {hand_label} hand (rotation-specific: {rotation_specific})")

    def recognize(self, pose_features: np.ndarray, hand_label: str) -> Optional[str]:
        """Match features against templates"""
        if not self.templates:
            return None

        if np.linalg.norm(pose_features) < 1e-6:
            return None

        pose_norm = pose_features / (np.linalg.norm(pose_features) + 1e-8)

        best_match = None
        best_score = -1

        for template in self.templates:
            if template.hand_label != hand_label:
                continue

            if np.linalg.norm(template.pose_features) < 1e-6:
                continue

            template_norm = template.pose_features / (np.linalg.norm(template.pose_features) + 1e-8)
            similarity = np.dot(pose_norm, template_norm)
            eucl_dist = np.linalg.norm(pose_norm - template_norm)
            score = similarity - (eucl_dist * 0.5)

            if similarity > 0.85 and eucl_dist < 0.5 and score > best_score:
                best_score = score
                best_match = template.name

        return best_match

    def get_smooth_rotation(self, hand_label: str, new_rotation: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Apply exponential moving average"""
        self.rotation_smoothing[hand_label].append(new_rotation)
        if len(self.rotation_smoothing[hand_label]) < 3:
            return new_rotation

        avg = np.mean(self.rotation_smoothing[hand_label], axis=0)
        return tuple(avg)

    def save_templates(self, filename: str):
        """Save templates to JSON"""
        data = []
        for t in self.templates:
            data.append({
                'name': t.name,
                'hand_label': t.hand_label,
                'pose_features': t.pose_features.tolist(),
                'threshold': t.threshold
            })
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(self.templates)} templates")

    def load_templates(self, filename: str):
        """Load templates from JSON"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            self.templates = []
            for item in data:
                self.templates.append(GestureTemplate(
                    name=item['name'],
                    hand_label=item['hand_label'],
                    pose_features=np.array(item['pose_features'], dtype=np.float32),
                    threshold=item['threshold']
                ))
            print(f"Loaded {len(self.templates)} templates")
        except FileNotFoundError:
            print("No existing template file found")

class MIDIManager:
    """Handles virtual MIDI device output"""

    def __init__(self):
        self.midi_out = rtmidi.MidiOut()
        self.virtual_port_name = "Gesture MIDI Controller"
        self.active_notes = set()
        self.cc_states = {}

    def open_virtual_port(self):
        available_ports = self.midi_out.get_ports()
        print(f"Available MIDI ports: {available_ports}")

        target_idx = None
        for i, port_name in enumerate(available_ports):
            if any(keyword in port_name.lower() for keyword in ['loopmidi', 'gesture', 'virtual', 'loop']):
                target_idx = i
                print(f"Found virtual port: {port_name}")
                break

        if target_idx is not None:
            self.midi_out.open_port(target_idx)
            print(f"Connected to: {available_ports[target_idx]}")
        elif available_ports:
            target_idx = len(available_ports) - 1
            self.midi_out.open_port(target_idx)
            print(f"Using: {available_ports[target_idx]}")
            print("WARNING: If this is Microsoft GS Synth, install loopMIDI for DAW control")
        else:
            print("No MIDI ports available! Install loopMIDI first.")
            self.midi_out = None

    def send_cc(self, channel: int, cc_num: int, value: float, threshold: int = 2):
        if self.midi_out is None:
            return

        midi_val = int(np.clip(value * 127, 0, 127))
        key = (channel, cc_num)

        last_val = self.cc_states.get(key, -999)
        if abs(midi_val - last_val) >= threshold:
            msg = [0xB0 | channel, cc_num, midi_val]
            self.midi_out.send_message(msg)
            self.cc_states[key] = midi_val

    def send_message(self, mapping, value: float = None, is_active: bool = True):
        if self.midi_out is None:
            return

        if mapping.message_type == 'note':
            if is_active:
                msg = [0x90 | mapping.channel, mapping.note_or_cc, mapping.value_on]
                self.midi_out.send_message(msg)
                self.active_notes.add((mapping.channel, mapping.note_or_cc))
            else:
                msg = [0x80 | mapping.channel, mapping.note_or_cc, mapping.value_off]
                self.midi_out.send_message(msg)
                self.active_notes.discard((mapping.channel, mapping.note_or_cc))

        elif mapping.message_type == 'cc':
            if mapping.continuous and value is not None:
                self.send_cc(mapping.channel, mapping.note_or_cc, value)
            else:
                val = mapping.value_on if is_active else mapping.value_off
                msg = [0xB0 | mapping.channel, mapping.note_or_cc, val]
                self.midi_out.send_message(msg)

    def panic(self):
        if self.midi_out is None:
            return
        for ch, note in list(self.active_notes):
            msg = [0x80 | ch, note, 0]
            self.midi_out.send_message(msg)
        self.active_notes.clear()

class GestureMIDIMapper:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.recognizer = GestureRecognizer()
        self.midi = MIDIManager()
        self.mappings: Dict[str, MIDIMapping] = {}

        self.gesture_neutral_rotations = {}

        self.calibration_mode = False
        self.calibration_target = None
        self.calibration_buffer = []
        self.current_gestures = {'Left': None, 'Right': None}
        self.last_gesture_time = {'Left': 0, 'Right': 0}
        self.gesture_cooldown = 0.1

        self.rotation_cc = {
            'Left': {'pitch': None, 'yaw': None, 'roll': None},
            'Right': {'pitch': None, 'yaw': None, 'roll': None}
        }

        # New: Rotation calibration storage
        self.rotation_calibration = {
            'Left': {'pitch': None, 'yaw': None, 'roll': None},
            'Right': {'pitch': None, 'yaw': None, 'roll': None}
        }

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

    def setup(self):
        self.midi.open_virtual_port()
        self.recognizer.load_templates("gesture_templates.json")
        self.load_mappings()

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 60)

    def load_mappings(self):
        try:
            with open(CONFIG_FILE, 'r') as f:
                data = json.load(f)
                for name, m in data.items():
                    # Handle legacy mappings without rotation_axis
                    if 'rotation_axis' not in m:
                        m['rotation_axis'] = 'pitch'
                    self.mappings[name] = MIDIMapping(**m)

            # Load rotation calibration if exists
            try:
                with open("rotation_calibration.json", 'r') as f:
                    cal_data = json.load(f)
                    for hand in ['Left', 'Right']:
                        for axis in ['pitch', 'yaw', 'roll']:
                            if hand in cal_data and axis in cal_data[hand] and cal_data[hand][axis]:
                                self.rotation_calibration[hand][axis] = RotationCalibration(
                                    **cal_data[hand][axis]
                                )
            except FileNotFoundError:
                pass

        except FileNotFoundError:
            self.mappings = {}

    def save_mappings(self):
        data = {name: asdict(m) for name, m in self.mappings.items()}
        with open(CONFIG_FILE, 'w') as f:
            json.dump(data, f, indent=2)

        # Save rotation calibration
        cal_data = {}
        for hand in ['Left', 'Right']:
            cal_data[hand] = {}
            for axis in ['pitch', 'yaw', 'roll']:
                if self.rotation_calibration[hand][axis]:
                    cal_data[hand][axis] = asdict(self.rotation_calibration[hand][axis])
                else:
                    cal_data[hand][axis] = None
        with open("rotation_calibration.json", 'w') as f:
            json.dump(cal_data, f, indent=2)

    def process_hand(self, hand_landmarks, hand_label: str, frame):
        """Process hand and return gesture + rotation data"""
        pose_features = self.recognizer.feature_extractor.get_rotation_invariant_features(hand_landmarks)

        global_features = self.recognizer.feature_extractor.get_global_features(hand_landmarks, hand_label)

        pitch, yaw, roll = self.recognizer.get_smooth_rotation(
            hand_label, 
            self.recognizer.feature_extractor.get_rotation_angles(hand_landmarks, hand_label)
        )

        self.mp_draw.draw_landmarks(
            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
            self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2),
            self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=1)
        )

        if self.calibration_mode and self.calibration_target:
            if hand_label == self.calibration_target['hand']:
                self.calibration_buffer.append(pose_features)
                progress = len(self.calibration_buffer) / CALIBRATION_FRAMES
                cv2.putText(frame, f"RECORDING: {self.calibration_target['name']} [{int(progress*100)}%]", 
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                if len(self.calibration_buffer) >= CALIBRATION_FRAMES:
                    avg_pose = np.mean(self.calibration_buffer, axis=0)
                    self.recognizer.add_template(
                        self.calibration_target['name'], 
                        avg_pose, 
                        hand_label,
                        self.calibration_target.get('rotation_specific', False)
                    )
                    self.calibration_mode = False
                    self.calibration_buffer = []

        gesture = self.recognizer.recognize(pose_features, hand_label)

        y_pos = 100 if hand_label == 'Left' else 180
        color = (255, 255, 0) if hand_label == 'Left' else (0, 255, 255)

        cv2.putText(frame, f"{hand_label}: {gesture if gesture else '...'}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"P:{pitch:.2f} Y:{yaw:.2f} R:{roll:.2f}", 
                   (10, y_pos+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        return gesture, {'pitch': pitch, 'yaw': yaw, 'roll': roll}

    def handle_midi(self, gesture: str, hand_label: str, rotation: Dict[str, float]):
        current_time = time.time()
        gesture_key = f"{gesture}_{hand_label.lower()}" if gesture else None

        if gesture_key != self.current_gestures[hand_label]:
            prev = self.current_gestures[hand_label]
            if prev and prev in self.mappings:
                prev_map = self.mappings[prev]
                if prev_map.message_type == 'note':
                    self.midi.send_message(prev_map, is_active=False)

            if gesture_key and gesture_key in self.mappings:
                if current_time - self.last_gesture_time[hand_label] > self.gesture_cooldown:
                    mapping = self.mappings[gesture_key]
                    if not mapping.continuous:
                        self.midi.send_message(mapping, is_active=True)
                    self.last_gesture_time[hand_label] = current_time

                    if mapping.continuous:
                        self.gesture_neutral_rotations[gesture_key] = {
                            'pitch': rotation['pitch'],
                            'yaw': rotation['yaw'],
                            'roll': rotation['roll']
                        }

        if gesture_key and gesture_key in self.mappings:
            mapping = self.mappings[gesture_key]
            if mapping.continuous:
                # Check if we have calibration for this axis
                cal = self.rotation_calibration[hand_label].get(mapping.rotation_axis)
                if cal:
                    # Use calibrated value
                    value = cal.normalize(rotation[mapping.rotation_axis])
                else:
                    # Use legacy neutral-based calculation
                    neutral = self.gesture_neutral_rotations.get(gesture_key, {'pitch': 0.5, 'yaw': 0.5, 'roll': 0.5})
                    axis = mapping.rotation_axis
                    current_val = rotation[axis]
                    neutral_val = neutral[axis]

                    delta = current_val - neutral_val
                    if delta > 0.5:
                        delta -= 1.0
                    elif delta < -0.5:
                        delta += 1.0

                    sensitivity = 8.0  # Default sensitivity
                    value = 0.5 + (delta * sensitivity)
                    value = float(np.clip(value, 0.0, 1.0))

                self.midi.send_message(mapping, value=value)

        self.current_gestures[hand_label] = gesture_key

        # Handle global rotation CC with calibration
        for axis in ['pitch', 'yaw', 'roll']:
            cc_config = self.rotation_cc[hand_label][axis]
            if cc_config is not None:
                ch, cc_num = cc_config
                cal = self.rotation_calibration[hand_label].get(axis)
                if cal:
                    value = cal.normalize(rotation[axis])
                else:
                    value = rotation[axis]
                self.midi.send_cc(ch, cc_num, value)

    def run(self):
        print("\n=== Gesture MIDI Controller (Fixed Rotation) ===")
        print("Keys: [R]ecord  [S]ave  [M]ap  [C]C Config  [A]xis Calibrate  [Q]uit")
        print("\nRotation detection improved - Pitch/Yaw/Roll now calculated correctly")
        print("Sensitivity now works via explicit rotation_axis in mapping")
        print("Use [A] to calibrate rotation axes with midpoint/min/max\n")

        self.setup()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.recognizer.feature_extractor.hands.process(rgb)

            active_hands = {'Left': False, 'Right': False}

            if results.multi_hand_landmarks:
                for idx, (landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                    hand_label = handedness.classification[0].label
                    active_hands[hand_label] = True

                    gesture, rotation = self.process_hand(landmarks, hand_label, frame)

                    if not self.calibration_mode:
                        self.handle_midi(gesture, hand_label, rotation)

            for hand_label in ['Left', 'Right']:
                if not active_hands[hand_label] and self.current_gestures[hand_label]:
                    prev_key = self.current_gestures[hand_label]
                    if prev_key in self.mappings:
                        self.midi.send_message(self.mappings[prev_key], is_active=False)
                    self.current_gestures[hand_label] = None

            cv2.putText(frame, "Fixed Rotation Gesture MIDI", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if self.calibration_mode:
                cv2.putText(frame, "CALIBRATION MODE", (10, 400), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('Gesture MIDI', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.start_calibration()
            elif key == ord('s'):
                self.recognizer.save_templates("gesture_templates.json")
                self.save_mappings()
            elif key == ord('m'):
                self.map_current_gesture()
            elif key == ord('c'):
                self.configure_rotation_cc()
            elif key == ord('a'):
                self.calibrate_rotation_axis()

        self.shutdown()

    def start_calibration(self):
        print("\n--- Record Gesture ---")
        name = input("Gesture name (e.g., 'fist', 'open_palm', 'peace'): ")
        hand = input("Hand (Left/Right): ").capitalize()
        rot_spec = input("Rotation-specific? (y/n, default=n): ").lower() == 'y'

        if hand in ['Left', 'Right']:
            self.calibration_target = {
                'name': name, 
                'hand': hand,
                'rotation_specific': rot_spec
            }
            self.calibration_mode = True
            self.calibration_buffer = []
            print(f"Hold '{name}' gesture steady...")

    def calibrate_rotation_axis(self):
        """Calibrate rotation axis with midpoint, min, and max positions"""
        print("\n=== Rotation Axis Calibration ===")
        print("This will calibrate a rotation axis with midpoint, min, and max positions.")

        hand = input("Hand (Left/Right): ").capitalize()
        if hand not in ['Left', 'Right']:
            print("Invalid hand. Must be Left or Right.")
            return

        axis = input("Axis (pitch/yaw/roll): ").lower()
        if axis not in ['pitch', 'yaw', 'roll']:
            print("Invalid axis. Must be pitch, yaw, or roll.")
            return

        print(f"\nCalibrating {hand} hand {axis} axis.")
        print("You will set MIDPOINT, MINIMUM, and MAXIMUM positions.")
        print("Press SPACE to capture each position when ready.")
        print("Press ESC to cancel at any time.\n")

        calibration_values = {'midpoint': None, 'min': None, 'max': None}
        stages = ['midpoint', 'min', 'max']
        stage_idx = 0

        # Create a window for calibration
        calib_window_name = f"Calibrate {hand} {axis}"

        while stage_idx < len(stages):
            stage = stages[stage_idx]
            print(f"\nStage {stage_idx + 1}/3: Set {stage.upper()} position")
            print(f"Hold your hand in the {stage} position and press SPACE to capture.")
            print("Watch the camera feed for current rotation values.")

            captured = False
            while not captured:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.recognizer.feature_extractor.hands.process(rgb)

                # Display instructions
                cv2.putText(frame, f"Calibrating: {hand} {axis}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"Stage {stage_idx + 1}/3: {stage.upper()}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, "SPACE to capture, ESC to cancel", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                current_val = None
                if results.multi_hand_landmarks:
                    for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        if handedness.classification[0].label == hand:
                            pitch, yaw, roll = self.recognizer.feature_extractor.get_rotation_angles(landmarks, hand)
                            current_val = {'pitch': pitch, 'yaw': yaw, 'roll': roll}[axis]

                            # Draw landmarks
                            self.mp_draw.draw_landmarks(
                                frame, landmarks, self.mp_hands.HAND_CONNECTIONS,
                                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2),
                                self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=1)
                            )

                            # Show current value
                            cv2.putText(frame, f"Current {axis}: {current_val:.3f}", (10, 120), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                            # Show previously captured values
                            y_offset = 150
                            for prev_stage, prev_val in calibration_values.items():
                                if prev_val is not None:
                                    cv2.putText(frame, f"{prev_stage}: {prev_val:.3f}", 
                                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                               (255, 255, 0), 1)
                                    y_offset += 25
                            break
                else:
                    cv2.putText(frame, "No hand detected!", (10, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow('Gesture MIDI', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print("Calibration cancelled.")
                    return
                elif key == 32:  # SPACE
                    if current_val is not None:
                        calibration_values[stage] = current_val
                        print(f"Captured {stage}: {current_val:.3f}")
                        captured = True
                        stage_idx += 1
                    else:
                        print("No hand detected! Cannot capture.")

        # Validate calibration
        midpoint = calibration_values['midpoint']
        min_val = calibration_values['min']
        max_val = calibration_values['max']

        print(f"\n--- Calibration Summary ---")
        print(f"Midpoint: {midpoint:.3f}")
        print(f"Minimum:  {min_val:.3f}")
        print(f"Maximum:  {max_val:.3f}")

        # Check for valid range
        if abs(max_val - min_val) < 0.1:
            print("\nWARNING: Min and Max are very close. Calibration may not work well.")
            confirm = input("Save anyway? (y/n): ").lower()
            if confirm != 'y':
                print("Calibration discarded.")
                return

        # Save calibration
        self.rotation_calibration[hand][axis] = RotationCalibration(
            midpoint=midpoint,
            min_val=min_val,
            max_val=max_val
        )

        print(f"\nCalibration saved for {hand} hand {axis} axis!")
        print("This calibration will be used for both gesture-based CC and global rotation CC.")

    def configure_rotation_cc(self):
        """Setup 3-axis rotation as continuous CC"""
        print("\n--- Configure Rotation CC ---")
        hand = input("Hand (Left/Right): ").capitalize()
        if hand not in ['Left', 'Right']:
            return

        print(f"\nCurrent: Pitch={self.rotation_cc[hand]['pitch']}, "
              f"Yaw={self.rotation_cc[hand]['yaw']}, "
              f"Roll={self.rotation_cc[hand]['roll']}")

        axis = input("Axis (pitch/yaw/roll): ").lower()
        if axis not in ['pitch', 'yaw', 'roll']:
            return

        ch = int(input("MIDI Channel (0-15): "))
        cc = int(input("CC Number (0-127): "))

        self.rotation_cc[hand][axis] = (ch, cc)
        print(f"Set {hand} {axis} -> CC{cc} Ch{ch+1}")

    def map_current_gesture(self):
        print("\n--- Map Gesture ---")
        for hand in ['Left', 'Right']:
            if self.current_gestures[hand]:
                print(f"Current {hand}: {self.current_gestures[hand]}")

        gesture = input("Gesture key (e.g., 'fist_left'): ")
        msg_type = input("Type (note/cc): ")
        channel = int(input("Channel (0-15): "))
        note_cc = int(input("Note/CC number: "))

        rotation_axis = 'pitch'
        if msg_type == 'cc':
            rotation_axis = input("Rotation axis (pitch/yaw/roll, default=pitch): ").lower() or 'pitch'

        mapping = MIDIMapping(
            message_type=msg_type,
            channel=channel,
            note_or_cc=note_cc,
            value_on=127,
            value_off=0,
            continuous=(msg_type == 'cc'),
            use_rotation=False,
            rotation_axis=rotation_axis
        )

        self.mappings[gesture] = mapping
        print(f"Mapped {gesture} with rotation_axis={rotation_axis}")

    def shutdown(self):
        self.midi.panic()
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = GestureMIDIMapper()
    app.run()