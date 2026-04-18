import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import tkinter as tk
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import cv2
import numpy as np
from collections import deque
import time
import csv
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# MediaPipe Tasks API
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    HAS_MEDIAPIPE = True
except ImportError:
    print("[WARNING] MediaPipe not found. Run 'pip install mediapipe'. Pose rules disabled.")
    HAS_MEDIAPIPE = False

# YOLOv8
try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] Ultralytics not found. Run 'pip install ultralytics'.")
    YOLO = None

# =============================================================================
# Configuration
USE_YOLO_HYBRID = True

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(SCRIPT_DIR, 'hybrid_pose_mobilenet_model_v2.h5')
POSE_MODEL  = os.path.join(SCRIPT_DIR, 'pose_landmarker_lite.task')
IMAGE_HEIGHT, IMAGE_WIDTH = 128, 128
SEQUENCE_LENGTH = 12
POSE_VECTOR_SIZE = 99
LSTM_CONFIDENCE_THRESHOLD = 0.70
LOG_FILE = os.path.join(SCRIPT_DIR, 'events.csv')

# Pose rule thresholds
RAISED_ARM_THRESH = 0.22
LUNGE_THRESH      = 0.28
STRIKE_THRESH     = 0.10
LEAN_THRESH       = 0.30
RULE_CONFIRM_N    = 3

# Landmark indices
IDX_NOSE       = 0
IDX_L_SHOULDER = 11; IDX_R_SHOULDER = 12
IDX_L_WRIST    = 15; IDX_R_WRIST    = 16
IDX_L_HIP      = 23; IDX_R_HIP      = 24
IDX_L_KNEE     = 25; IDX_R_KNEE     = 26

# Advanced Diagnostic Thresholds
VELOCITY_THRESH    = 0.18
PROXIMITY_THRESH   = 0.12
LOITER_SECONDS     = 10
LOITER_MOVE_THRESH = 0.04
WEAPON_CLASSES     = [43, 76, 73, 74]

# Detection toggles — controlled by the settings window (BooleanVar wired in after root created)
# Keys match the flag strings appended in the pipeline
DETECTION_TOGGLES = {
    "WEAPON":       True,
    "INTENT":       True,
    "POSE RULES":   True,
    "FALL":         True,
    "PERSON DOWN":  True,
    "RAPID MOVE":   True,
    "FLAILING":     True,
    "LOITERING":    True,
    "PROXIMITY":    True,
}

# Tier 3 Thresholds
FALL_THRESH    = 0.45
FLAIL_THRESH   = 0.08
FLAIL_MIN_FRAMES = 5

# =============================================================================
# Load Models
print("[INFO] Initializing LSTM Intent Model...")
try:
    lstm_model = load_model(MODEL_PATH)
    has_lstm = True
    print("[OK] LSTM Loaded.")
except Exception as e:
    print(f"[WARNING] Failed to load LSTM: {e}")
    lstm_model = None
    has_lstm = False

print("[INFO] Initializing YOLO Tracker...")
yolo_model = YOLO("yolov8s.pt") if YOLO else None

# Initialize MediaPipe PoseLandmarker
pose_landmarker = None
if HAS_MEDIAPIPE and os.path.exists(POSE_MODEL):
    try:
        _base_opts = mp_python.BaseOptions(model_asset_path=POSE_MODEL)
        _pose_opts = mp_vision.PoseLandmarkerOptions(
            base_options=_base_opts,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4,
            min_tracking_confidence=0.4
        )
        pose_landmarker = mp_vision.PoseLandmarker.create_from_options(_pose_opts)
        print("[OK] MediaPipe PoseLandmarker Loaded.")
    except Exception as e:
        print(f"[WARNING] PoseLandmarker failed to load: {e}")
        pose_landmarker = None
elif HAS_MEDIAPIPE:
    print(f"[WARNING] Pose model not found at {POSE_MODEL}. Pose rules disabled.")

# =============================================================================
# FIX 1: Thread-safe Audit Logger using a lock
_log_lock = threading.Lock()

def log_event(activity_type, confidence):
    """Append a detected threat event to events.csv (thread-safe)."""
    try:
        with _log_lock:
            file_exists = os.path.isfile(LOG_FILE)
            with open(LOG_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['Timestamp', 'Activity Type', 'Confidence'])
                writer.writerow([
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    activity_type,
                    f"{confidence:.2f}"
                ])
    except Exception as e:
        print(f"[LOG ERROR]: {e}")

# =============================================================================
# Frontend UI
root = tk.Tk()
root.title("Hybrid AI Surveillance - Professional Edition")
root.state('zoomed')
root.configure(bg="#0d1117")

heading = tk.Label(
    root, text="Suspicious Activity Detection", width=40,
    font=("Helvetica", 38, 'bold'), bg="#0d1117", fg="white"
)
heading.pack(pady=40)

subhead = tk.Label(
    root, text="YOLOv8  ·  MediaPipe Pose  ·  MobileNetV2-LSTM",
    font=("Courier", 14), bg="#0d1117", fg="#58a6ff"
)
subhead.pack()


# =============================================================================
# Pose Rule Engine

def _draw_skeleton(crop_bgr, landmarks):
    """Draw key joints on the crop using OpenCV directly."""
    h, w = crop_bgr.shape[:2]
    CONNECTIONS = [
        (IDX_L_SHOULDER, IDX_R_SHOULDER),
        (IDX_L_SHOULDER, IDX_L_WRIST),
        (IDX_R_SHOULDER, IDX_R_WRIST),
        (IDX_L_HIP,      IDX_R_HIP),
        (IDX_L_SHOULDER, IDX_L_HIP),
        (IDX_R_SHOULDER, IDX_R_HIP),
        (IDX_L_HIP,      IDX_L_KNEE),
        (IDX_R_HIP,      IDX_R_KNEE),
    ]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    annotated = crop_bgr.copy()
    for a, b in CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(annotated, pts[a], pts[b], (0, 255, 128), 2)
    for pt in pts:
        cv2.circle(annotated, pt, 4, (255, 255, 0), -1)
    return annotated


def run_pose_rules(crop_bgr, track_id, track_pose_prev, track_last_valid_pose, box=None, frame_shape=None):
    """
    Runs MediaPipe PoseLandmarker on the given person crop.

    FIX 2: Always returns a 4-tuple (raw_flag, annotated_crop, pose_vector, landmarks|None).
    The original code inconsistently returned 3- or 4-tuples causing index errors.
    """
    pose_vec = track_last_valid_pose.get(track_id, np.zeros((POSE_VECTOR_SIZE,), dtype='float32'))

    if pose_landmarker is None:
        return None, crop_bgr, pose_vec, None
    if crop_bgr.shape[0] < 48 or crop_bgr.shape[1] < 48:
        return None, crop_bgr, pose_vec, None

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)

    try:
        result = pose_landmarker.detect(mp_img)
    except Exception:
        return None, crop_bgr, pose_vec, None

    if not result.pose_landmarks:
        return None, crop_bgr, pose_vec, None

    lm = result.pose_landmarks[0]

    # FIX 3: Guard against out-of-range landmark index before arithmetic
    required_indices = [IDX_NOSE, IDX_L_SHOULDER, IDX_R_SHOULDER,
                        IDX_L_WRIST, IDX_R_WRIST, IDX_L_HIP, IDX_R_HIP,
                        IDX_L_KNEE, IDX_R_KNEE]
    if max(required_indices) >= len(lm):
        return None, crop_bgr, pose_vec, None

    vec = []
    for landmark in lm:
        if box is not None and frame_shape is not None:
            bx1, by1, bx2, by2 = box
            fh, fw = frame_shape[:2]
            # FIX 4: Avoid division by zero when frame dimensions are 0
            gx = landmark.x * (bx2 - bx1) / max(fw, 1) + bx1 / max(fw, 1)
            gy = landmark.y * (by2 - by1) / max(fh, 1) + by1 / max(fh, 1)
            vec.extend([gx, gy, landmark.z])
        else:
            vec.extend([landmark.x, landmark.y, landmark.z])

    pose_vec = np.array(vec, dtype='float32')
    track_last_valid_pose[track_id] = pose_vec

    nose    = lm[IDX_NOSE]
    l_sho   = lm[IDX_L_SHOULDER]; r_sho  = lm[IDX_R_SHOULDER]
    l_wrist = lm[IDX_L_WRIST];    r_wrist = lm[IDX_R_WRIST]
    l_hip   = lm[IDX_L_HIP];      r_hip   = lm[IDX_R_HIP]
    l_knee  = lm[IDX_L_KNEE];     r_knee  = lm[IDX_R_KNEE]

    shoulder_y = (l_sho.y + r_sho.y) / 2
    hip_y      = (l_hip.y + r_hip.y) / 2
    hip_x      = (l_hip.x + r_hip.x) / 2

    raw_flag = None

    both_raised = (l_wrist.y < shoulder_y - RAISED_ARM_THRESH and
                   r_wrist.y < shoulder_y - RAISED_ARM_THRESH)
    if both_raised:
        raw_flag = "RAISED ARMS"

    if not raw_flag:
        if l_knee.y < hip_y - STRIKE_THRESH or r_knee.y < hip_y - STRIKE_THRESH:
            raw_flag = "STRIKING POSTURE"

    if not raw_flag:
        if abs(nose.x - hip_x) > LEAN_THRESH:
            raw_flag = "AGGRESSIVE LEAN"

    if not raw_flag and track_id in track_pose_prev:
        if abs(hip_x - track_pose_prev[track_id]) > LUNGE_THRESH:
            raw_flag = "LUNGE DETECTED"

    track_pose_prev[track_id] = hip_x

    annotated = _draw_skeleton(crop_bgr, lm)
    # Always return 4-tuple
    return raw_flag, annotated, pose_vec, lm


# =============================================================================
# Behavioral Analysis Helpers

def get_center(x1, y1, x2, y2, fw, fh):
    # FIX 5: Guard division by zero for degenerate frame sizes
    fw = max(fw, 1)
    fh = max(fh, 1)
    return ((x1 + x2) / 2 / fw, (y1 + y2) / 2 / fh)

def check_velocity(track_id, cx, cy, now, track_center_history):
    if track_id not in track_center_history:
        track_center_history[track_id] = deque(maxlen=10)
    track_center_history[track_id].append((cx, cy, now))
    hist = track_center_history[track_id]
    if len(hist) < 4:
        return None
    dt   = max(hist[-1][2] - hist[0][2], 0.001)
    dist = ((hist[-1][0] - hist[0][0])**2 + (hist[-1][1] - hist[0][1])**2)**0.5
    return "RAPID MOVEMENT" if dist / dt > VELOCITY_THRESH else None

def check_loitering(track_id, cx, cy, now, track_first_seen, track_last_moved, track_center_history):
    if track_id not in track_first_seen:
        track_first_seen[track_id] = now
        track_last_moved[track_id] = now
    hist = track_center_history.get(track_id)
    if hist and len(hist) >= 2:
        dist = ((hist[-1][0] - hist[-2][0])**2 + (hist[-1][1] - hist[-2][1])**2)**0.5
        if dist > LOITER_MOVE_THRESH:
            track_last_moved[track_id] = now
    stay_time = now - track_last_moved.get(track_id, now)
    if stay_time > LOITER_SECONDS:
        return f"LOITERING ({int(stay_time)}s)"
    return None

def check_proximity(boxes, track_ids, fw, fh):
    conflicts = set()
    centers   = []
    fw = max(fw, 1); fh = max(fh, 1)
    for box, tid in zip(boxes, track_ids):
        bx1, by1, bx2, by2 = box
        centers.append((tid, (bx1+bx2)/2/fw, (by1+by2)/2/fh))
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            dist = ((centers[i][1]-centers[j][1])**2 + (centers[i][2]-centers[j][2])**2)**0.5
            if dist < PROXIMITY_THRESH:
                conflicts.add(centers[i][0])
                conflicts.add(centers[j][0])
    return conflicts

def weapon_near_person(px1, py1, px2, py2, weapon_boxes):
    for wb in weapon_boxes:
        wx1, wy1, wx2, wy2 = map(int, wb)
        wcx, wcy = (wx1+wx2)/2, (wy1+wy2)/2
        if px1 < wcx < px2 and py1 < wcy < py2:
            return True
    return False

# =============================================================================
# Tier 3 Behavioral Signals

def check_fall(track_id, x1, y1, x2, y2, now, track_bbox_history):
    h = y2 - y1
    if track_id not in track_bbox_history:
        track_bbox_history[track_id] = deque(maxlen=20)
    track_bbox_history[track_id].append((h, now))
    hist = track_bbox_history[track_id]
    if len(hist) < 10:
        return None
    prev_h = np.mean([val[0] for val in list(hist)[:5]])
    curr_h = np.mean([val[0] for val in list(hist)[-5:]])
    if prev_h > 0 and (prev_h - curr_h) / prev_h > FALL_THRESH:
        return "FALL DETECTED"
    return None

PRONE_RATIO      = 2.2    # bbox must be much wider than tall (was 1.6 — too sensitive)
PRONE_MIN_HEIGHT = 60     # ignore tiny boxes — crouching near camera looks wide but is small
PRONE_CONFIRM_N  = 8      # must be wide for this many consecutive frames

_prone_history = {}       # track_id -> deque of bool

def check_prone(track_id, x1, y1, x2, y2):
    w, h = (x2 - x1), (y2 - y1)
    if track_id not in _prone_history:
        _prone_history[track_id] = deque(maxlen=PRONE_CONFIRM_N)
    is_wide = (h > PRONE_MIN_HEIGHT) and (h > 0) and (w / h) > PRONE_RATIO
    _prone_history[track_id].append(is_wide)
    hist = _prone_history[track_id]
    if len(hist) == PRONE_CONFIRM_N and all(hist):
        return "PERSON DOWN"
    return None

def check_arm_flail(track_id, landmarks, track_wrist_history):
    if landmarks is None:
        return None
    try:
        # FIX 6: Guard against landmarks list being shorter than expected
        if len(landmarks) <= 16:
            return None
        lw, rw = landmarks[15], landmarks[16]
        pos = (lw.x, lw.y, rw.x, rw.y)

        if track_id not in track_wrist_history:
            track_wrist_history[track_id] = deque(maxlen=12)
        hist = track_wrist_history[track_id]

        if len(hist) > 0:
            prev = hist[-1]
            delta = ((pos[0]-prev[0])**2 + (pos[1]-prev[1])**2 +
                     (pos[2]-prev[2])**2 + (pos[3]-prev[3])**2)**0.5
            hist.append(pos)

            if delta > FLAIL_THRESH:
                flail_count = sum(1 for i in range(1, len(hist))
                                  if ((hist[i][0]-hist[i-1][0])**2 + (hist[i][1]-hist[i-1][1])**2)**0.5 > FLAIL_THRESH)
                if flail_count >= FLAIL_MIN_FRAMES:
                    return "ARM FLAILING"
        else:
            hist.append(pos)
    except Exception:
        pass
    return None

def get_time_multiplier():
    hr = datetime.now().hour
    if 22 <= hr or hr <= 5:
        return 1.3
    return 1.0

# FIX 7: Safe LSTM confidence parser extracted to a helper
def parse_lstm_confidence(lstm_verdict: str) -> float:
    """Safely extract numeric confidence from status strings like 'SUSPICIOUS (87%)'."""
    try:
        if '(' in lstm_verdict and '%' in lstm_verdict:
            inner = lstm_verdict.split('(')[1].rstrip('%)')
            return float(inner) / 100.0
    except (IndexError, ValueError):
        pass
    return 0.0

# =============================================================================
# Core Pipeline

def show_video(video_source):
    root.withdraw()

    # If the user closes/exits the tk window while video is running, stop gracefully
    stop_flag = [False]
    def _on_root_close():
        stop_flag[0] = True
    root.protocol("WM_DELETE_WINDOW", _on_root_close)

    # Robust webcam fallback chain
    if video_source == 0:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("[INFO] CAP_DSHOW failed, trying CAP_MSMF...")
            cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
        if not cap.isOpened():
            print("[INFO] CAP_MSMF failed, trying default backend...")
            cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"[ERROR] Could not open {video_source}")
        root.deiconify()
        return

    # Resolution lock for webcam only; buffer tuning applies to both
    if video_source == 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    # Tracking state
    track_buffers         = {}
    track_pose_buffers    = {}
    track_status          = {}
    track_scores          = {}
    track_pose_prev       = {}
    track_rule_flag       = {}
    track_rule_history    = {}
    track_last_valid_pose = {}
    track_center_history  = {}
    track_first_seen      = {}
    track_last_moved      = {}
    track_bbox_history    = {}
    track_wrist_history   = {}
    track_last_lm         = {}
    track_active_flags    = {}  # all active flags per track — drives side panel

    alert_sent    = False
    normal_frames = 0

    frame_count = 0
    start_time  = time.time()
    prev_time   = start_time

    executor          = ThreadPoolExecutor(max_workers=3)
    processing_tracks = set()

    def evaluate_intent_async(t_id, X_seq, X_pose_seq):
        try:
            preds = lstm_model({"image_input": X_seq, "pose_input": X_pose_seq}, training=False).numpy()[0]
            if t_id not in track_scores:
                track_scores[t_id] = deque(maxlen=3)
            track_scores[t_id].append(preds)
            avg  = np.mean(track_scores[t_id], axis=0)
            conf = np.max(avg)
            if np.argmax(avg) == 1 and conf > 0.65:
                track_status[t_id] = f"SUSPICIOUS ({avg[1]*100:.0f}%)"
            else:
                track_status[t_id] = f"Normal ({avg[0]*100:.0f}%)"
        except Exception as e:
            print(f"[LSTM Thread ERROR]: {e}")
        finally:
            processing_tracks.discard(t_id)

    font = cv2.FONT_HERSHEY_SIMPLEX

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            h_w, w_w = param
            if (w_w - 130) <= x <= (w_w - 10) and 12 <= y <= 60:
                stop_flag[0] = True

    win_name = 'Suspicious Activity Detection'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)
    cv2.setMouseCallback(win_name, mouse_callback, (720, 1280))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            curr_time = time.time()
            fps       = 1 / max(curr_time - prev_time, 0.0001)
            prev_time = curr_time
            frame_count += 1
            annotated_frame = frame.copy()

            # FIX 8: yolo_model guard — if YOLO failed to load skip entire detection block
            if yolo_model is None:
                cv2.putText(annotated_frame, "YOLO unavailable", (10, 50), font, 1, (0, 0, 255), 2)
                cv2.imshow(win_name, annotated_frame)
                if cv2.waitKey(1) & 0xFF in (27, ord('q')) or stop_flag[0]:
                    break
                continue

            results = yolo_model.track(frame, persist=True, classes=[0]+WEAPON_CLASSES, verbose=False)
            # FIX 9: Remove duplicate annotated_frame = frame.copy() that was overwriting annotations
            # (original had this line AFTER writing weapon boxes — erasing them)

            weapon_boxes = []
            person_boxes = []
            person_ids   = []

            if results and results[0].boxes is not None:
                for box_t, cls_t in zip(results[0].boxes.xyxy.cpu().numpy(),
                                        results[0].boxes.cls.int().cpu().tolist()):
                    if cls_t == 0:
                        person_boxes.append(box_t)
                    elif cls_t in WEAPON_CLASSES:
                        weapon_boxes.append(box_t)
                        wx1, wy1, wx2, wy2 = map(int, box_t)
                        cv2.rectangle(annotated_frame, (wx1, wy1), (wx2, wy2), (0, 0, 255), 3)
                        cv2.putText(annotated_frame, "WEAPON", (wx1, wy1 - 10), font, 0.8, (0, 0, 255), 2)

                if results[0].boxes.id is not None:
                    person_ids = [tid for tid, cls in zip(
                        results[0].boxes.id.int().cpu().tolist(),
                        results[0].boxes.cls.int().cpu().tolist()) if cls == 0]

                    for box, track_id in zip(person_boxes, person_ids):
                        x1, y1, x2, y2 = map(int, box)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                        crop = frame[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue

                        # FIX 10: Consistent 4-tuple unpack — run_pose_rules always returns 4 values now
                        if frame_count % 2 == 0 or track_id not in track_last_valid_pose:
                            raw_flag, skel_crop, pose_vec, current_lm = run_pose_rules(
                                crop, track_id, track_pose_prev, track_last_valid_pose,
                                (x1, y1, x2, y2), frame.shape)
                            track_rule_flag[track_id] = raw_flag
                            track_last_lm[track_id] = current_lm
                        else:
                            raw_flag   = track_rule_flag.get(track_id)
                            skel_crop  = crop
                            pose_vec   = track_last_valid_pose.get(track_id, np.zeros(POSE_VECTOR_SIZE, dtype='float32'))
                            current_lm = track_last_lm.get(track_id)

                        # Draw skeleton safely
                        try:
                            annotated_frame[y1:y2, x1:x2] = cv2.resize(skel_crop, (x2-x1, y2-y1))
                        except Exception:
                            pass

                        # Temporal rule confirmation
                        if track_id not in track_rule_history:
                            track_rule_history[track_id] = deque(maxlen=RULE_CONFIRM_N)
                        track_rule_history[track_id].append(raw_flag)

                        hist = track_rule_history[track_id]
                        if (len(hist) == RULE_CONFIRM_N and
                                all(f == hist[0] and f is not None for f in hist)):
                            track_rule_flag[track_id] = hist[0]
                        elif raw_flag is None:
                            track_rule_flag[track_id] = None

                        # LSTM buffer + async inference
                        if has_lstm:
                            try:
                                arr = preprocess_input(
                                    np.array(cv2.cvtColor(
                                        cv2.resize(crop, (IMAGE_WIDTH, IMAGE_HEIGHT)),
                                        cv2.COLOR_BGR2RGB), dtype='float32'))
                                if track_id not in track_buffers:
                                    track_buffers[track_id]      = deque(maxlen=SEQUENCE_LENGTH)
                                    track_pose_buffers[track_id] = deque(maxlen=SEQUENCE_LENGTH)
                                    track_status[track_id]       = "Tracking..."
                                track_buffers[track_id].append(arr)
                                track_pose_buffers[track_id].append(pose_vec)

                                if (len(track_buffers[track_id]) == SEQUENCE_LENGTH and
                                        (frame_count + track_id) % 3 == 0 and
                                        track_id not in processing_tracks):
                                    processing_tracks.add(track_id)
                                    X_img  = np.expand_dims(np.array(track_buffers[track_id]),      axis=0)
                                    X_pose = np.expand_dims(np.array(track_pose_buffers[track_id]), axis=0)
                                    executor.submit(evaluate_intent_async, track_id, X_img, X_pose)
                            except Exception as e:
                                print(f"[LSTM ERROR]: {e}")

                        # Advanced Diagnostic Checks
                        cx, cy = get_center(x1, y1, x2, y2, frame.shape[1], frame.shape[0])

                        vel_flag    = check_velocity(track_id, cx, cy, curr_time, track_center_history)
                        loiter_flag = check_loitering(track_id, cx, cy, curr_time,
                                                      track_first_seen, track_last_moved,
                                                      track_center_history)
                        weapon_flag = weapon_near_person(x1, y1, x2, y2, weapon_boxes)

                        fall_flag   = check_fall(track_id, x1, y1, x2, y2, curr_time, track_bbox_history)
                        prone_flag  = check_prone(track_id, x1, y1, x2, y2)
                        flail_flag  = check_arm_flail(track_id, current_lm, track_wrist_history)

                        # ── Collect ALL active flags for this track ───────────
                        lstm_verdict = track_status.get(track_id, "Tracking...")
                        rule_flag    = track_rule_flag.get(track_id)
                        lstm_conf    = parse_lstm_confidence(lstm_verdict)

                        time_mult   = get_time_multiplier()
                        is_lstm_sus = "SUSPICIOUS" in lstm_verdict and lstm_conf > (0.70 / time_mult)
                        is_rule_sus = rule_flag is not None

                        active_flags = []
                        if weapon_flag      and _toggle_vars["WEAPON"].get():      active_flags.append("WEAPON")
                        if is_lstm_sus      and _toggle_vars["INTENT"].get():      active_flags.append(f"INTENT {lstm_conf*100:.0f}%")
                        if is_rule_sus      and _toggle_vars["POSE RULES"].get():  active_flags.append(rule_flag)
                        if fall_flag        and _toggle_vars["FALL"].get():        active_flags.append("FALL")
                        if prone_flag       and _toggle_vars["PERSON DOWN"].get(): active_flags.append("PERSON DOWN")
                        if vel_flag         and _toggle_vars["RAPID MOVE"].get():  active_flags.append("RAPID MOVE")
                        if flail_flag       and _toggle_vars["FLAILING"].get():    active_flags.append("FLAILING")
                        if loiter_flag      and _toggle_vars["LOITERING"].get():   active_flags.append(loiter_flag)

                        # Severity → box border colour (worst wins for the bbox only)
                        if weapon_flag or fall_flag:
                            box_color = (0, 0, 255)          # red
                        elif is_lstm_sus or is_rule_sus:
                            box_color = (0, 80, 255)         # orange-red
                        elif vel_flag or flail_flag or prone_flag:
                            box_color = (0, 140, 255)        # orange
                        elif loiter_flag:
                            box_color = (30, 100, 180)       # amber
                        else:
                            box_color = (0, 200, 0)          # green — normal

                        # Small clean label directly on bounding box (ID + worst flag only)
                        short_label = f"ID:{track_id}"
                        if active_flags:
                            short_label += f" | {active_flags[0]}"
                        cv2.rectangle(annotated_frame, (x1, y1 - 28), (x1 + 260, y1), box_color, -1)
                        cv2.putText(annotated_frame, short_label,
                                    (x1 + 4, y1 - 8), font, 0.55, (255, 255, 255), 1)

                        # Store flags for side panel (drawn after all persons processed)
                        track_active_flags[track_id] = active_flags

                        # Alert trigger
                        is_alert = (weapon_flag or fall_flag or
                                    (is_lstm_sus and lstm_conf > LSTM_CONFIDENCE_THRESHOLD))
                        if is_alert:
                            if not alert_sent:
                                alert_sent = True
                                try:
                                    log_event(" | ".join(active_flags) or "SUSPICIOUS",
                                              max(lstm_conf, 0.85))
                                except Exception:
                                    pass
                            cv2.putText(annotated_frame, "! THREAT DETECTED !",
                                        (30, frame.shape[0] - 40), font, 1.6, (0, 0, 255), 4)
                            normal_frames = 0
                        else:
                            normal_frames += 1
                            if normal_frames > 90:
                                alert_sent = False

            # ── Proximity alert (on raw frame before resize) ──────────────
            conflict_ids = check_proximity(person_boxes, person_ids, frame.shape[1], frame.shape[0])
            if conflict_ids and _toggle_vars["PROXIMITY"].get():
                cv2.putText(annotated_frame,
                            f"CONFLICT ZONE: {len(conflict_ids)} PERSONS",
                            (30, frame.shape[0] - 80), font, 1.0, (0, 100, 255), 2)

            # ── Scale video to 720p ───────────────────────────────────────
            DISPLAY_HEIGHT = 720
            PANEL_W        = 280          # side panel width in pixels
            orig_h, orig_w = annotated_frame.shape[:2]
            scale    = DISPLAY_HEIGHT / max(orig_h, 1)
            scaled_w = int(orig_w * scale)
            video_frame = cv2.resize(annotated_frame, (scaled_w, DISPLAY_HEIGHT),
                                     interpolation=cv2.INTER_LINEAR)

            # ── Build side panel canvas ───────────────────────────────────
            panel = np.zeros((DISPLAY_HEIGHT, PANEL_W, 3), dtype=np.uint8)
            panel[:] = (18, 18, 24)   # very dark background

            # Panel header
            cv2.rectangle(panel, (0, 0), (PANEL_W, 50), (10, 10, 18), -1)
            cv2.putText(panel, "ACTIVE TRACKS", (10, 18),
                        font, 0.45, (100, 180, 255), 1)
            cv2.putText(panel, f"{len(track_active_flags)} person(s) tracked",
                        (10, 38), font, 0.38, (120, 120, 140), 1)
            cv2.line(panel, (0, 50), (PANEL_W, 50), (40, 40, 55), 1)

            # One card per tracked person
            ROW_H       = 28   # height per flag line
            CARD_PAD    = 8    # vertical gap between cards
            card_top    = 58

            # Severity colour map for flag pills
            FLAG_COLORS = {
                "WEAPON":      (40,  40,  220),
                "FALL":        (40,  40,  220),
                "INTENT":      (40,  100, 220),
                "RAISED ARMS": (40,  120, 200),
                "LUNGE":       (40,  120, 200),
                "STRIKING":    (40,  120, 200),
                "AGGRESSIVE":  (40,  120, 200),
                "RAPID MOVE":  (30,  160, 220),
                "FLAILING":    (30,  140, 200),
                "PERSON DOWN": (30,  80,  200),
                "LOITERING":   (60,  110, 160),
                "CONFLICT":    (60,  100, 200),
            }

            def flag_color(f):
                for key, col in FLAG_COLORS.items():
                    if key in f.upper():
                        return col
                return (60, 60, 80)

            # Remove stale tracks (IDs no longer in current frame)
            active_ids_this_frame = set(person_ids)
            stale = [tid for tid in track_active_flags if tid not in active_ids_this_frame]
            for tid in stale:
                del track_active_flags[tid]

            for tid in sorted(track_active_flags.keys()):
                flags = track_active_flags[tid]

                # How many lines does this card need?
                n_lines  = max(len(flags), 1)
                card_h   = 22 + n_lines * ROW_H + 6

                if card_top + card_h > DISPLAY_HEIGHT - 10:
                    break   # panel full

                # Card background
                has_critical = any(k in " ".join(flags).upper()
                                   for k in ("WEAPON", "FALL", "INTENT"))
                card_bg = (30, 18, 18) if has_critical else (22, 24, 32)
                cv2.rectangle(panel,
                              (6, card_top),
                              (PANEL_W - 6, card_top + card_h),
                              card_bg, -1)
                # Left accent bar — severity colour
                accent = (0, 0, 200) if has_critical else (50, 100, 180)
                cv2.rectangle(panel,
                              (6, card_top),
                              (10, card_top + card_h),
                              accent, -1)

                # Track ID header
                cv2.putText(panel, f"ID: {tid}",
                            (16, card_top + 16),
                            font, 0.48, (220, 220, 240), 1)

                if not flags:
                    cv2.putText(panel, "Normal",
                                (16, card_top + 16 + ROW_H),
                                font, 0.4, (80, 180, 80), 1)
                else:
                    for fi, flag_text in enumerate(flags):
                        fy = card_top + 20 + (fi + 1) * ROW_H
                        # Pill background
                        pill_col = flag_color(flag_text)
                        pill_x2  = min(16 + 8 + len(flag_text) * 7 + 8, PANEL_W - 10)
                        cv2.rectangle(panel,
                                      (16, fy - 13),
                                      (pill_x2, fy + 5),
                                      pill_col, -1)
                        cv2.putText(panel, flag_text,
                                    (20, fy),
                                    font, 0.4, (255, 255, 255), 1)

                # Thin separator line at card bottom
                cv2.line(panel,
                         (10, card_top + card_h),
                         (PANEL_W - 10, card_top + card_h),
                         (38, 38, 50), 1)

                card_top += card_h + CARD_PAD

            # ── Compose final display: video | panel ─────────────────────
            display_frame = np.hstack([video_frame, panel])
            h_f, w_f = display_frame.shape[:2]

            # ── Top HUD bar ───────────────────────────────────────────────
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (scaled_w, 90), (0, 0, 0), -1)
            display_frame = cv2.addWeighted(overlay, 0.55, display_frame, 0.45, 0)

            elapsed = time.time() - start_time
            cv2.putText(display_frame, "HYBRID AI  |  YOLO + Pose + LSTM",
                        (10, 34), font, 0.75, (0, 255, 255), 2)
            cv2.putText(display_frame,
                        f"FPS: {fps:.1f}  |  Elapsed: {int(elapsed)}s  |  Tracks: {len(track_active_flags)}",
                        (10, 68), font, 0.55, (180, 180, 180), 1)

            # ── STOP button ───────────────────────────────────────────────
            bx1, by1, bx2, by2 = scaled_w - 120, 14, scaled_w - 10, 56
            cv2.rectangle(display_frame, (bx1, by1), (bx2, by2), (0, 0, 160), -1)
            cv2.rectangle(display_frame, (bx1, by1), (bx2, by2), (255, 255, 255), 1)
            cv2.putText(display_frame, 'STOP', (bx1 + 18, by2 - 12),
                        font, 0.8, (255, 255, 255), 2)

            cv2.setMouseCallback(win_name, mouse_callback, (h_f, w_f))
            cv2.imshow(win_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q') or stop_flag[0]:
                print("[INFO] Detection stopped.")
                break

    finally:
        cap.release()
        executor.shutdown(wait=False)
        cv2.destroyAllWindows()
        try:
            # Restore normal close behaviour and show the launcher again
            root.protocol("WM_DELETE_WINDOW", root.destroy)
            root.deiconify()
        except Exception:
            pass  # root was already destroyed via Exit button — that's fine


# =============================================================================
# GUI — main launcher window

# BooleanVars for each toggle — created after root exists
_toggle_vars = {}   # key -> tk.BooleanVar

def open_settings():
    """Small floating window to enable/disable each detection type."""
    win = tk.Toplevel(root)
    win.title("Detection Settings")
    win.configure(bg="#0d1117")
    win.resizable(False, False)
    win.attributes("-topmost", True)

    tk.Label(win, text="Detection Toggles", font=("Helvetica", 14, "bold"),
             bg="#0d1117", fg="white").pack(pady=(14, 4))
    tk.Label(win, text="Uncheck to disable a detection type",
             font=("Helvetica", 10), bg="#0d1117", fg="#8b949e").pack(pady=(0, 10))

    frame = tk.Frame(win, bg="#0d1117")
    frame.pack(padx=20, pady=4)

    LABELS = {
        "WEAPON":      "Weapon detection",
        "INTENT":      "LSTM suspicious intent",
        "POSE RULES":  "Pose rules (raised arms, lunge…)",
        "FALL":        "Fall detection",
        "PERSON DOWN": "Person down (prone)",
        "RAPID MOVE":  "Rapid movement",
        "FLAILING":    "Arm flailing",
        "LOITERING":   "Loitering",
        "PROXIMITY":   "Proximity / conflict zone",
    }

    for key, label in LABELS.items():
        var = _toggle_vars[key]
        cb = tk.Checkbutton(
            frame, text=label, variable=var,
            font=("Helvetica", 11), bg="#0d1117", fg="white",
            selectcolor="#1f6feb", activebackground="#0d1117",
            activeforeground="white", anchor="w"
        )
        cb.pack(fill="x", pady=3)

    tk.Button(win, text="Close", command=win.destroy,
              width=14, font=("Helvetica", 11), bg="#21262d",
              fg="#8b949e", relief="flat", cursor="hand2").pack(pady=12)


def upload_video():
    f = askopenfilename(initialdir=SCRIPT_DIR, title='Select video',
                        filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov"), ("All", "*.*")])
    if f:
        show_video(f)

def use_webcam():
    show_video(0)

def do_exit():
    """Safe exit: ensures all threads and windows close immediately."""
    try:
        # Signal executor to stop if possible
        if 'executor' in locals():
            executor.shutdown(wait=False)
        root.quit()
        root.destroy()
        sys.exit(0)
    except Exception:
        os._exit(0) # Forced exit if tkinter is stuck

btn_frame = tk.Frame(root, bg="#0d1117")
btn_frame.pack(pady=30)

btn_style = dict(width=28, font=("Helvetica", 16, "bold"),
                 bg="#1f6feb", fg="white", relief="flat",
                 activebackground="#388bfd", cursor="hand2")

tk.Button(btn_frame, text="  Upload Video", command=upload_video,
          **btn_style).pack(pady=10)
tk.Button(btn_frame, text="  Use Webcam", command=use_webcam,
          **btn_style).pack(pady=10)
tk.Button(btn_frame, text="  Detection Settings", command=open_settings,
          width=28, font=("Helvetica", 14, "bold"), bg="#238636",
          fg="white", relief="flat", activebackground="#2ea043",
          cursor="hand2").pack(pady=10)
tk.Button(btn_frame, text="Exit", command=do_exit,
          width=28, font=("Helvetica", 14), bg="#21262d",
          fg="#8b949e", relief="flat", cursor="hand2").pack(pady=10)

# Initialise BooleanVars now that root exists
for k, v in DETECTION_TOGGLES.items():
    _toggle_vars[k] = tk.BooleanVar(value=v)

# CLI support
if len(sys.argv) > 1:
    arg = sys.argv[1]
    show_video(0 if arg == "0" else (
        arg if os.path.isabs(arg) else os.path.join(SCRIPT_DIR, arg)))
    try:
        root.destroy()
    except Exception:
        pass
else:
    root.mainloop()