import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import numpy as np
import rtmidi
import json
import math
import time
import threading
import queue
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
from pathlib import Path

# Configuration
CONFIG_FILE = "gesture_midi_config.json"
CALIBRATION_FRAMES = 15
SMOOTHING_WINDOW = 3

# Model download URL (auto-downloaded on first run if needed)
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"

# Performance settings
TARGET_FPS = 60
DETECTION_INTERVAL = 0.016  # Minimum time between detections (seconds) ~60fps max
MIDI_THROTTLE = 0.005  # Minimum time between MIDI messages

# Hand landmark connections for drawing (from original MediaPipe HAND_CONNECTIONS)
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm connections
    (5, 9), (9, 13), (13, 17)
]

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
    rotation_specific: bool = False

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

def download_model():
    """Download the hand landmarker model if not present"""
    if not Path(MODEL_PATH).exists():
        print(f"Downloading hand landmarker model to {MODEL_PATH}...")
        import urllib.request
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("Model downloaded successfully!")
        except Exception as e:
            print(f"Error downloading model: {e}")
            print("Please download manually from:")
            print(MODEL_URL)
            return False
    return True

class HandFeatureExtractor:
    """Extracts geometric features from hand landmarks"""

    def __init__(self):
        # Download model if needed
        if not download_model():
            raise RuntimeError("Could not download hand landmarker model")

        # Create HandLandmarker with the new Tasks API
        base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,  # Slightly lower for better performance
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.VIDEO,  # Optimized for video
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def get_hand_coordinate_system(self, landmarks, hand_label: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate orthonormal basis vectors for hand"""
        # Convert landmarks to numpy array (new API uses list of NormalizedLandmark)
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

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

    def get_rotation_angles(self, landmarks, hand_label: str = None, 
                             discontinuity_offsets: Dict[str, float] = None) -> Tuple[float, float, float]:
        """Extract stable Euler angles with configurable discontinuity positions"""
        wrist, x_axis, y_axis, z_axis = self.get_hand_coordinate_system(landmarks, hand_label)

        # Build rotation matrix with camera coordinate system
        # Camera: X right, Y down, Z forward (towards camera)
        # We need to map our hand axes to this
        R = np.column_stack([x_axis, y_axis, z_axis])

        # Extract raw angles using robust method that avoids gimbal lock
        # This uses the full rotation matrix to calculate stable angles

        # Pitch: Rotation around X-axis (tilting hand up/down)
        # Calculate from Y and Z components
        pitch = math.atan2(-R[2, 1], math.sqrt(R[0, 1]**2 + R[1, 1]**2))

        # Yaw: Rotation around Y-axis (turning hand left/right)
        # Use X and Z components of the forward vector
        yaw = math.atan2(R[0, 2], R[2, 2])

        # Roll: Rotation around Z-axis (twisting hand/tilting side to side)
        # Calculate from X-axis orientation relative to horizontal plane
        # Use the angle between hand's X-axis and camera's horizontal plane
        roll = math.atan2(R[1, 0], R[0, 0])

        # Default offsets (0 = no shift, discontinuity at original position)
        offsets = discontinuity_offsets or {}

        def normalize_with_offset(angle, offset=0.0):
            """
            Normalize angle to 0-1 with configurable discontinuity offset.
            offset: 0.0 = discontinuity at -π/π (default)
                    0.5 = discontinuity at 0 (neutral position)
                    Any value 0-1 rotates the seam position
            """
            # Shift angle by offset*2π before wrapping
            shifted = angle + (offset * 2 * math.pi)
            
            # Wrap to -π to π range
            while shifted > math.pi:
                shifted -= 2 * math.pi
            while shifted < -math.pi:
                shifted += 2 * math.pi
                
            # Now normalize to 0-1 (discontinuity is at the shifted position)
            normalized = (shifted + math.pi) / (2 * math.pi)
            return float(np.clip(normalized, 0.0, 1.0))

        # Get offsets for each axis (default 0)
        pitch_offset = offsets.get('pitch', 0.0)
        yaw_offset = offsets.get('yaw', 0.0)
        roll_offset = offsets.get('roll', 0.0)

        pitch_norm = normalize_with_offset(pitch, pitch_offset)
        yaw_norm = normalize_with_offset(yaw, yaw_offset)
        roll_norm = normalize_with_offset(roll, roll_offset)

        return pitch_norm, yaw_norm, roll_norm

    def get_rotation_invariant_features(self, landmarks) -> np.ndarray:
        """Extract features that don't change with hand rotation"""
        # Convert landmarks to numpy array
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

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
        # New API: landmarks is a list, access by index
        lm = landmarks

        return {
            'pitch': pitch,
            'yaw': yaw,
            'roll': roll,
            'pos_x': lm[9].x,
            'pos_y': lm[9].y,
        }

    def detect(self, rgb_image, timestamp_ms):
        """Process image and return hand landmarks using new API - VIDEO mode"""
        # Convert numpy array to MediaPipe Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        return self.landmarker.detect_for_video(mp_image, timestamp_ms)

class GestureRecognizer:
    """Recognizes gestures based on trained templates"""

    def __init__(self):
        self.templates: List[GestureTemplate] = []
        self.feature_extractor = HandFeatureExtractor()
        # Use simple exponential moving average instead of deque
        self.rotation_smoothing = {
            'Left': {'pitch': 0.5, 'yaw': 0.5, 'roll': 0.5, 'alpha': 0.3},
            'Right': {'pitch': 0.5, 'yaw': 0.5, 'roll': 0.5, 'alpha': 0.3}
        }

    def add_template(self, name: str, features: np.ndarray, hand_label: str, rotation_specific: bool = False):
        """Record new gesture template"""
        template = GestureTemplate(
            name=name, 
            pose_features=features, 
            hand_label=hand_label,
            threshold=0.5 if rotation_specific else 0.4,
            rotation_specific=rotation_specific
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
        """Apply exponential moving average - O(1) instead of O(N)"""
        smooth = self.rotation_smoothing[hand_label]
        alpha = smooth['alpha']
        
        smooth['pitch'] = alpha * new_rotation[0] + (1 - alpha) * smooth['pitch']
        smooth['yaw'] = alpha * new_rotation[1] + (1 - alpha) * smooth['yaw']
        smooth['roll'] = alpha * new_rotation[2] + (1 - alpha) * smooth['roll']
        
        return (smooth['pitch'], smooth['yaw'], smooth['roll'])

    def save_templates(self, filename: str):
        """Save templates to JSON"""
        data = []
        for t in self.templates:
            data.append({
                'name': t.name,
                'hand_label': t.hand_label,
                'pose_features': t.pose_features.tolist(),
                'threshold': t.threshold,
                'rotation_specific': t.rotation_specific
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
                    threshold=item.get('threshold', 0.4),
                    rotation_specific=item.get('rotation_specific', False)
                ))
            print(f"Loaded {len(self.templates)} templates")
        except FileNotFoundError:
            print("No existing template file found")

class MIDIManager:
    """Handles virtual MIDI device output with throttling"""

    def __init__(self):
        self.midi_out = rtmidi.MidiOut()
        self.virtual_port_name = "Gesture MIDI Controller"
        self.active_notes = set()
        self.cc_states = {}
        self.last_cc_time = {}  # Throttle CC messages
        self.lock = threading.Lock()

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

        # Throttle CC messages
        current_time = time.time()
        last_time = self.last_cc_time.get(key, 0)
        if current_time - last_time < MIDI_THROTTLE:
            return

        last_val = self.cc_states.get(key, -999)
        if abs(midi_val - last_val) >= threshold:
            with self.lock:
                msg = [0xB0 | channel, cc_num, midi_val]
                self.midi_out.send_message(msg)
                self.cc_states[key] = midi_val
                self.last_cc_time[key] = current_time

    def send_message(self, mapping, value: float = None, is_active: bool = True):
        if self.midi_out is None:
            return

        if mapping.message_type == 'note':
            with self.lock:
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
                with self.lock:
                    msg = [0xB0 | mapping.channel, mapping.note_or_cc, val]
                    self.midi_out.send_message(msg)

    def panic(self):
        if self.midi_out is None:
            return
        with self.lock:
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

        # NEW: Discontinuity offset configuration (0-1 range)
        # 0.0 = default (seam at -180°/180°)
        # 0.5 = seam at 0° (neutral position)
        self.discontinuity_offsets = {
            'Left': {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0},
            'Right': {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}
        }

        # Performance tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        self.last_detection_time = 0
        
        # Threading for non-blocking operations
        self.midi_queue = queue.Queue()
        self.midi_thread = threading.Thread(target=self._midi_worker, daemon=True)
        
        # Pre-allocate display buffer
        self.display_frame = None
        
        # Keyboard input handling
        self.key_buffer = []
        self.last_key_time = 0
        self.key_cooldown = 0.2  # Prevent double triggers

    def _midi_worker(self):
        """Background thread for MIDI operations"""
        while True:
            try:
                task = self.midi_queue.get(timeout=0.1)
                if task is None:
                    break
                func, args, kwargs = task
                func(*args, **kwargs)
            except queue.Empty:
                continue

    def setup(self):
        self.midi.open_virtual_port()
        self.recognizer.load_templates("gesture_templates.json")
        self.load_mappings()

        # Optimize camera settings for low latency
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for lower latency
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        # Start MIDI worker thread
        self.midi_thread.start()

    def load_mappings(self):
        try:
            with open(CONFIG_FILE, 'r') as f:
                data = json.load(f)

            # Extract rotation CC config if present
            if '_rotation_cc' in data:
                self.rotation_cc = data.pop('_rotation_cc')

            # Load discontinuity offsets if present
            if '_discontinuity_offsets' in data:
                loaded_offsets = data.pop('_discontinuity_offsets')
                for hand in ['Left', 'Right']:
                    if hand in loaded_offsets:
                        self.discontinuity_offsets[hand].update(loaded_offsets[hand])

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
        # Include rotation CC configuration
        data['_rotation_cc'] = self.rotation_cc
        # Include discontinuity offsets
        data['_discontinuity_offsets'] = self.discontinuity_offsets
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

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        """Draw landmarks using custom OpenCV implementation (solutions module removed)"""
        hand_landmarks_list = detection_result.hand_landmarks
        annotated_image = np.copy(rgb_image)

        # Get image dimensions
        h, w = annotated_image.shape[:2]

        # Loop through the detected hands to visualize
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]

            # Convert normalized coordinates to pixel coordinates
            landmarks_px = []
            for lm in hand_landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                landmarks_px.append((x, y))

            # Draw connections
            for connection in HAND_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(landmarks_px) and end_idx < len(landmarks_px):
                    start_point = landmarks_px[start_idx]
                    end_point = landmarks_px[end_idx]
                    cv2.line(annotated_image, start_point, end_point, (255, 0, 0), 2)

            # Draw landmarks
            for i, point in enumerate(landmarks_px):
                # Different color for different parts of the hand
                if i == 0:  # Wrist
                    color = (0, 0, 255)
                    size = 8
                elif i in [4, 8, 12, 16, 20]:  # Finger tips
                    color = (0, 255, 0)
                    size = 6
                else:  # Other joints
                    color = (0, 255, 255)
                    size = 4
                cv2.circle(annotated_image, point, size, color, -1)

        return annotated_image

    def process_hand(self, landmarks, hand_label: str, frame):
        """Process hand with configurable discontinuity offsets"""
        pose_features = self.recognizer.feature_extractor.get_rotation_invariant_features(landmarks)

        global_features = self.recognizer.feature_extractor.get_global_features(landmarks, hand_label)

        # Pass discontinuity offsets to rotation calculation
        offsets = self.discontinuity_offsets[hand_label]
        raw_angles = self.recognizer.feature_extractor.get_rotation_angles(
            landmarks, hand_label, discontinuity_offsets=offsets
        )
        
        pitch, yaw, roll = self.recognizer.get_smooth_rotation(hand_label, raw_angles)

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
        print("\n=== Gesture MIDI Controller (Optimized) ===")
        print("Keys: [R]ecord  [S]ave  [M]ap  [C]C Config  [A]xis Calibrate  [D]iscontinuity Offset  [Q]uit")
        print("\nOptimizations enabled:")
        print("- VIDEO mode for MediaPipe")
        print("- Frame skipping for detection")
        print("- Exponential smoothing (O(1))")
        print("- MIDI throttling")
        print("- Threaded MIDI output")
        print("- Buffer size minimized")
        print("- Configurable discontinuity offsets for rotation")
        print("")

        self.setup()

        # Create window and ensure it has focus
        cv2.namedWindow('Gesture MIDI', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Gesture MIDI', 640, 480)
        
        # Wait a moment for window to initialize
        time.sleep(0.1)
        
        # Force window to front (platform-specific workarounds)
        cv2.setWindowProperty('Gesture MIDI', cv2.WND_PROP_TOPMOST, 1)
        cv2.setWindowProperty('Gesture MIDI', cv2.WND_PROP_TOPMOST, 0)

        frame_time = 1.0 / TARGET_FPS
        last_frame_time = time.time()
        
        # For keyboard input - check multiple times per frame
        key_check_interval = 0.01  # Check for keys every 10ms
        last_key_check = 0

        while True:
            loop_start = time.time()
            
            # Check for keyboard input frequently (not just at end of loop)
            if loop_start - last_key_check >= key_check_interval:
                key = cv2.waitKey(1) & 0xFF
                last_key_check = loop_start
                
                if key != 255:  # Valid key pressed
                    # Prevent key repeat with cooldown
                    if loop_start - self.last_key_time >= self.key_cooldown:
                        self.last_key_time = loop_start
                        
                        if key == ord('q'):
                            break
                        elif key == ord('r'):
                            self.start_calibration()
                        elif key == ord('s'):
                            # Use thread for saving to avoid blocking
                            threading.Thread(target=self._save_data, daemon=True).start()
                        elif key == ord('m'):
                            self.map_current_gesture()
                        elif key == ord('c'):
                            self.configure_rotation_cc()
                        elif key == ord('a'):
                            self.calibrate_rotation_axis()
                        elif key == ord('d'):
                            self.configure_discontinuity_offset()
                        else:
                            # Debug: show what key was pressed
                            print(f"Key pressed: {chr(key) if 32 <= key <= 126 else key}")
            
            # Maintain consistent frame rate
            elapsed = loop_start - last_frame_time
            if elapsed < frame_time:
                time.sleep(max(0, frame_time - elapsed))
            
            ret, frame = self.cap.read()
            if not ret:
                continue

            last_frame_time = time.time()
            self.frame_count += 1

            # Calculate FPS every second
            if time.time() - self.last_fps_time >= 1.0:
                self.current_fps = self.frame_count
                self.frame_count = 0
                self.last_fps_time = time.time()

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Limit detection rate to avoid overloading
            current_time = time.time()
            should_detect = (current_time - self.last_detection_time) >= DETECTION_INTERVAL
            
            results = None
            if should_detect:
                timestamp_ms = int(current_time * 1000)
                results = self.recognizer.feature_extractor.detect(rgb, timestamp_ms)
                self.last_detection_time = current_time

            active_hands = {'Left': False, 'Right': False}

            if results and results.hand_landmarks:
                # Draw landmarks first
                display = self.draw_landmarks_on_image(rgb, results)
                display = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)

                for idx, landmarks in enumerate(results.hand_landmarks):
                    # Get handedness from results
                    if results.handedness and idx < len(results.handedness):
                        hand_label = results.handedness[idx][0].category_name
                    else:
                        # Fallback: infer from position
                        hand_label = 'Left' if idx == 0 else 'Right'

                    active_hands[hand_label] = True

                    gesture, rotation = self.process_hand(landmarks, hand_label, display)

                    if not self.calibration_mode:
                        self.handle_midi(gesture, hand_label, rotation)
                
                display_frame = display
            else:
                # No hands detected - convert back to BGR for display
                display_frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            for hand_label in ['Left', 'Right']:
                if not active_hands[hand_label] and self.current_gestures[hand_label]:
                    prev_key = self.current_gestures[hand_label]
                    if prev_key in self.mappings:
                        self.midi.send_message(self.mappings[prev_key], is_active=False)
                    self.current_gestures[hand_label] = None

            # FPS counter and status
            cv2.putText(display_frame, f"FPS: {self.current_fps}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "Gesture MIDI (Optimized)", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show discontinuity offset info
            y_offset = 90
            for hand in ['Left', 'Right']:
                roll_offset = self.discontinuity_offsets[hand]['roll']
                if roll_offset != 0.0:
                    seam_degrees = int((roll_offset * 360) - 180)
                    cv2.putText(display_frame, f"{hand} Roll seam: {seam_degrees}°", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
                    y_offset += 20

            if self.calibration_mode:
                cv2.putText(display_frame, "CALIBRATION MODE", (10, 400), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('Gesture MIDI', display_frame)

        self.shutdown()

    def _save_data(self):
        """Save data in background thread"""
        self.recognizer.save_templates("gesture_templates.json")
        self.save_mappings()

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
                timestamp_ms = int(time.time() * 1000)
                results = self.recognizer.feature_extractor.detect(rgb, timestamp_ms)

                # Display instructions
                display_frame = np.copy(frame)
                cv2.putText(display_frame, f"Calibrating: {hand} {axis}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Stage {stage_idx + 1}/3: {stage.upper()}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(display_frame, "SPACE to capture, ESC to cancel", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                current_val = None
                if results.hand_landmarks:
                    for idx, landmarks in enumerate(results.hand_landmarks):
                        # Get handedness
                        if results.handedness and idx < len(results.handedness):
                            current_hand = results.handedness[idx][0].category_name
                        else:
                            current_hand = 'Left' if idx == 0 else 'Right'

                        if current_hand == hand:
                            # Use current discontinuity offsets during calibration
                            offsets = self.discontinuity_offsets[hand]
                            pitch, yaw, roll = self.recognizer.feature_extractor.get_rotation_angles(
                                landmarks, hand, discontinuity_offsets=offsets
                            )
                            current_val = {'pitch': pitch, 'yaw': yaw, 'roll': roll}[axis]

                            # Draw landmarks
                            display_frame = self.draw_landmarks_on_image(rgb, results)
                            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)

                            # Show current value
                            cv2.putText(display_frame, f"Current {axis}: {current_val:.3f}", (10, 120), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                            # Show previously captured values
                            y_offset = 150
                            for prev_stage, prev_val in calibration_values.items():
                                if prev_val is not None:
                                    cv2.putText(display_frame, f"{prev_stage}: {prev_val:.3f}", 
                                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                               (255, 255, 0), 1)
                                    y_offset += 25
                            break
                else:
                    cv2.putText(display_frame, "No hand detected!", (10, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow('Gesture MIDI', display_frame)

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

    def configure_discontinuity_offset(self):
        """Set where the 0-1 discontinuity occurs for each rotation axis"""
        print("\n=== Configure Discontinuity Offset ===")
        print("This controls where the rotation value 'jumps' from 1 to 0.")
        print("Offset 0.0 = jump at ±180° (default)")
        print("Offset 0.5 = jump at 0° (neutral position)")
        print("Useful for: placing the discontinuity where your hand never goes")
        
        hand = input("Hand (Left/Right): ").capitalize()
        if hand not in ['Left', 'Right']:
            return
            
        axis = input("Axis (pitch/yaw/roll): ").lower()
        if axis not in ['pitch', 'yaw', 'roll']:
            return
            
        current = self.discontinuity_offsets[hand][axis]
        print(f"\nCurrent offset for {hand} {axis}: {current:.2f}")
        
        # Show where the seam currently is
        current_seam = (current * 360) - 180
        print(f"Current discontinuity position: {current_seam:+.0f}°")
        
        try:
            new_offset = float(input("New offset (0.0-1.0): "))
            new_offset = max(0.0, min(1.0, new_offset))  # Clamp to 0-1
            self.discontinuity_offsets[hand][axis] = new_offset
            
            # Calculate where the seam is now for user feedback
            seam_degrees = (new_offset * 360) - 180
            print(f"\nDiscontinuity moved to {seam_degrees:+.0f}°")
            print(f"(Values jump from 1→0 when hand passes {seam_degrees:+.0f}°)")
            
            # Show example ranges
            print(f"\nExample: With hand at neutral (0°), value is:")
            # Calculate what 0° maps to with this offset
            neutral_normalized = (0.5 - new_offset) % 1.0
            if neutral_normalized > 0.5:
                neutral_normalized -= 1.0
            print(f"  ~{neutral_normalized * 127:.0f} (MIDI 0-127)")
            
        except ValueError:
            print("Invalid input. Please enter a number between 0.0 and 1.0")

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