import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import json
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import cv2
import numpy as np
from collections import deque, defaultdict
import time
import csv
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import notifier

# MediaPipe Tasks API
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    HAS_MEDIAPIPE = True
except ImportError:
    print("[WARNING] MediaPipe not found. Pose rules disabled.")
    HAS_MEDIAPIPE = False

# YOLOv10
try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] Ultralytics not found.")
    YOLO = None

# =============================================================================
# Global Configuration
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(SCRIPT_DIR, 'hybrid_pose_mobilenet_model_v2.h5')
POSE_MODEL   = os.path.join(SCRIPT_DIR, 'pose_landmarker_lite.task')
LOG_FILE     = os.path.join(SCRIPT_DIR, 'events.csv')
SNAPSHOT_DIR = os.path.join(SCRIPT_DIR, 'snapshots')
_SETTINGS_FILE = os.path.join(SCRIPT_DIR, 'settings.json')
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

IMAGE_HEIGHT, IMAGE_WIDTH = 128, 128
SEQUENCE_LENGTH           = 12
POSE_VECTOR_SIZE          = 99
LSTM_CONFIDENCE_THRESHOLD = 0.70

RAISED_ARM_THRESH = 0.22
LUNGE_THRESH      = 0.28
STRIKE_THRESH     = 0.10
LEAN_THRESH       = 0.30
RULE_CONFIRM_N    = 3

IDX_NOSE       = 0
IDX_L_SHOULDER = 11; IDX_R_SHOULDER = 12
IDX_L_WRIST    = 15; IDX_R_WRIST    = 16
IDX_L_HIP      = 23; IDX_R_HIP      = 24
IDX_L_KNEE     = 25; IDX_R_KNEE     = 26

VELOCITY_THRESH    = 0.18
PROXIMITY_THRESH   = 0.12
LOITER_SECONDS     = 10
LOITER_MOVE_THRESH = 0.04
WEAPON_CLASSES     = [43, 76, 73, 74]
FALL_THRESH        = 0.45
FLAIL_THRESH       = 0.08
FLAIL_MIN_FRAMES   = 5
PRONE_RATIO        = 2.2
PRONE_MIN_HEIGHT   = 60
PRONE_CONFIRM_N    = 8

DETECTION_TOGGLES = {
    "WEAPON": True, "INTENT": True, "POSE RULES": True,
    "FALL": True, "PERSON DOWN": True, "RAPID MOVE": True,
    "FLAILING": True, "LOITERING": True, "PROXIMITY": True,
}

# =============================================================================
# Shared AI Intelligence Hub (Loaded Once)
print("[INFO] Initializing Shared AI Models...")
_ai_lock = threading.Lock()

try:
    lstm_model = load_model(MODEL_PATH)
    has_lstm   = True
    print("[OK] LSTM Loaded.")
except Exception as e:
    print(f"[WARNING] LSTM Load failed: {e}")
    lstm_model = None
    has_lstm   = False

yolo_model = YOLO("yolov10n.pt") if YOLO else None
pose_landmarker = None
if HAS_MEDIAPIPE and os.path.exists(POSE_MODEL):
    try:
        _base_opts = mp_python.BaseOptions(model_asset_path=POSE_MODEL)
        _pose_opts = mp_vision.PoseLandmarkerOptions(
            base_options=_base_opts, running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1, min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4, min_tracking_confidence=0.4)
        pose_landmarker = mp_vision.PoseLandmarker.create_from_options(_pose_opts)
        print("[OK] MediaPipe PoseLandmarker Loaded.")
    except Exception as e:
        print(f"[WARNING] PoseLandmarker failed: {e}")

# =============================================================================
# Thread-safe Resources

_log_lock = threading.Lock()
_session_events = []
_session_lock   = threading.Lock()

def log_event(activity_type, confidence, camera_id="CAM"):
    try:
        with _log_lock:
            exists = os.path.isfile(LOG_FILE)
            with open(LOG_FILE, 'a', newline='') as f:
                w = csv.writer(f)
                if not exists:
                    w.writerow(['Timestamp', 'Camera', 'Activity Type', 'Confidence'])
                w.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            camera_id, activity_type, f"{confidence:.2f}"])
        with _session_lock:
            _session_events.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "type": f"[{camera_id}] {activity_type}",
                "conf": confidence
            })
    except Exception as e:
        print(f"[LOG ERROR]: {e}")

def save_snapshot(frame: np.ndarray, activity_type: str, camera_id="CAM") -> str:
    try:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = activity_type.replace(" ", "_").replace("|", "-")[:20]
        path = os.path.join(SNAPSHOT_DIR, f"{camera_id}_{ts}_{safe}.jpg")
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return path
    except Exception as e:
        print(f"[SNAPSHOT] Save failed: {e}")
        return None

# =============================================================================
# Stream Handler Class

class CameraStream:
    def __init__(self, source, camera_id, ui_label):
        self.source = source
        self.camera_id = camera_id
        self.ui_label = ui_label
        self.cap = None
        self.running = False
        self.thread = None
        
        # State
        self.track_buffers = {}
        self.track_pose_buffers = {}
        self.track_status = {}
        self.track_scores = {}
        self.track_active_flags = {}
        self.track_last_valid_pose = {}
        self.track_pose_prev = {}
        self.track_rule_history = {}
        self.track_rule_flag = {}
        self.track_center_history = {}
        self.track_first_seen = {}
        self.track_last_moved = {}
        self.track_bbox_history = {}
        self.track_wrist_history = {}
        self.track_last_lm = {}
        self.prone_history = {}
        self.flag_first_seen = {}
        
        self.alert_sent = False
        self.normal_frames = 0
        self.flash_frames = 0
        self.heatmap_acc = None
        self.heatmap_on = True
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.processing_tracks = set()

    def start(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            print(f"[ERROR] Could not open {self.camera_id}")
            return False
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()

    def _run_loop(self):
        frame_count = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret: break
            
            frame_count += 1
            if frame_count % 2 != 0: continue # Skip frames for multi-cam perf
            
            # 1. Processing
            processed_frame = self._process_frame(frame, frame_count)
            
            # 2. Update UI
            img = Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
            # Resize dynamically based on UI label size
            w, h = self.ui_label.winfo_width(), self.ui_label.winfo_height()
            if w > 10 and h > 10:
                img = img.resize((w, h), Image.Resampling.LANCZOS)
            
            tk_img = ImageTk.PhotoImage(image=img)
            self.ui_label.after(0, lambda: self.ui_label.configure(image=tk_img))
            self.ui_label.image = tk_img # Keep reference
            
        self.cap.release()

    def _process_frame(self, frame, frame_count):
        curr_time = time.time()
        annotated = frame.copy()
        h_f, w_f = frame.shape[:2]
        
        if self.heatmap_acc is None:
            self.heatmap_acc = np.zeros((h_f, w_f), dtype=np.float32)

        # AI Hub Lock (Shared Inference)
        with _ai_lock:
            results = yolo_model.track(frame, persist=True, classes=[0]+WEAPON_CLASSES, verbose=False)
        
        weapon_boxes = []; person_boxes = []; person_ids = []
        if results and results[0].boxes is not None:
            for box_t, cls_t in zip(results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.cls.int().cpu().tolist()):
                if cls_t == 0: person_boxes.append(box_t)
                elif cls_t in WEAPON_CLASSES:
                    weapon_boxes.append(box_t)
                    wx1, wy1, wx2, wy2 = map(int, box_t)
                    cv2.rectangle(annotated, (wx1, wy1), (wx2, wy2), (0, 0, 255), 2)
            
            if results[0].boxes.id is not None:
                person_ids = [tid for tid, cls in zip(results[0].boxes.id.int().cpu().tolist(), 
                                                       results[0].boxes.cls.int().cpu().tolist()) if cls == 0]
                
                for box, tid in zip(person_boxes, person_ids):
                    x1, y1, x2, y2 = map(int, box)
                    crop = frame[max(0, y1):min(h_f, y2), max(0, x1):min(w_f, x2)]
                    if crop.size == 0: continue
                    
                    # Pose & Intent logic...
                    raw_flag, pose_vec, lms = self._run_pose_analysis(crop, tid, (x1,y1,x2,y2), frame.shape)
                    
                    if has_lstm:
                        self._queue_intent_analysis(crop, tid, pose_vec)
                    
                    # Behavioral Flags
                    cx, cy = (x1+x2)/2/w_f, (y1+y2)/2/h_f
                    v_flag = self._check_velocity(tid, cx, cy, curr_time)
                    l_flag = self._check_loitering(tid, cx, cy, curr_time)
                    f_flag = self._check_fall(tid, x1, y1, x2, y2, curr_time)
                    w_flag = weapon_near_person(x1, y1, x2, y2, weapon_boxes)
                    
                    active = []
                    if w_flag and _toggle_vars["WEAPON"].get(): active.append("WEAPON")
                    if "SUSP" in self.track_status.get(tid,"") and _toggle_vars["INTENT"].get(): active.append("INTENT")
                    if raw_flag and _toggle_vars["POSE RULES"].get(): active.append(raw_flag)
                    if f_flag and _toggle_vars["FALL"].get(): active.append("FALL")
                    
                    self.track_active_flags[tid] = active
                    
                    # Alert Logic
                    if active and not self.alert_sent:
                        self.alert_sent = True
                        alert_lbl = " | ".join(active)
                        log_event(alert_lbl, 0.90, self.camera_id)
                        snap = save_snapshot(annotated, alert_lbl, self.camera_id)
                        notifier.send_alert(alert_lbl, 0.90, snap)
                        self.flash_frames = 30

        # Heatmap
        if self.heatmap_on and self.heatmap_acc.max() > 0:
            annotated = overlay_heatmap(annotated, self.heatmap_acc)
            
        # UI Overlays
        cv2.putText(annotated, f"{self.camera_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if self.flash_frames > 0:
            self.flash_frames -= 1
            if self.flash_frames % 4 < 2:
                cv2.rectangle(annotated, (0, 0), (w_f, h_f), (0, 0, 255), 10)
        
        return annotated

    def _run_pose_analysis(self, crop, tid, box, f_shape):
        if pose_landmarker is None: return None, np.zeros(POSE_VECTOR_SIZE), None
        with _ai_lock:
            res = pose_landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
        if not res.pose_landmarks: return None, np.zeros(POSE_VECTOR_SIZE), None
        lm = res.pose_landmarks[0]
        # simplified rule check
        nose = lm[0]; l_wrist = lm[15]; r_wrist = lm[16]; l_sho = lm[11]; r_sho = lm[12]
        sho_y = (l_sho.y + r_sho.y) / 2
        flag = "RAISED ARMS" if l_wrist.y < sho_y-0.2 and r_wrist.y < sho_y-0.2 else None
        return flag, np.zeros(POSE_VECTOR_SIZE), lm

    def _queue_intent_analysis(self, crop, tid, pose_vec):
        # Async intent analysis implementation...
        pass

    def _check_velocity(self, tid, cx, cy, now):
        if tid not in self.track_center_history: self.track_center_history[tid] = deque(maxlen=10)
        self.track_center_history[tid].append((cx, cy, now))
        return None

    def _check_loitering(self, tid, cx, cy, now): return None
    def _check_fall(self, tid, x1, y1, x2, y2, now): return None

def overlay_heatmap(frame, acc):
    norm = cv2.normalize(acc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    blur = cv2.GaussianBlur(norm, (51, 51), 0)
    cmap = cv2.applyColorMap(blur, cv2.COLORMAP_JET)
    mask = blur > 10
    frame[mask] = cv2.addWeighted(frame, 0.6, cmap, 0.4, 0)[mask]
    return frame

def weapon_near_person(px1,py1,px2,py2,wboxes):
    for wb in wboxes:
        wcx, wcy = (wb[0]+wb[2])/2, (wb[1]+wb[3])/2
        if px1 < wcx < px2 and py1 < wcy < py2: return True
    return False

# =============================================================================
# Main Multi-Stream Application

class MultiStreamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hybrid Multi-Camera Command Center")
        self.root.state('zoomed')
        self.root.configure(bg="#0d1117")
        
        self.streams = []
        self._build_ui()
        
    def _build_ui(self):
        # Header
        header = tk.Frame(self.root, bg="#161b22", height=60)
        header.pack(fill="x")
        tk.Label(header, text="COMMAND CENTER", font=("Helvetica", 18, "bold"), bg="#161b22", fg="white").pack(side="left", padx=20)
        
        ctrl = tk.Frame(header, bg="#161b22")
        ctrl.pack(side="right", padx=20)
        tk.Button(ctrl, text="+ Add Camera", command=self._add_camera, bg="#238636", fg="white", relief="flat", padx=10).pack(side="left", padx=5)
        tk.Button(ctrl, text="Dashboard", command=open_dashboard, bg="#1f6feb", fg="white", relief="flat", padx=10).pack(side="left", padx=5)
        tk.Button(ctrl, text="Settings", command=open_settings, bg="#30363d", fg="white", relief="flat", padx=10).pack(side="left", padx=5)
        
        # Grid Container
        self.grid_frame = tk.Frame(self.root, bg="#0d1117")
        self.grid_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
    def _add_camera(self):
        source = simpledialog.askstring("Add Stream", "Enter RTSP URL or Cam ID (0, 1):", parent=self.root)
        if source:
            if source.isdigit(): source = int(source)
            cam_id = f"CAM-{len(self.streams)+1}"
            
            # Create UI Panel
            panel = tk.Frame(self.grid_frame, bg="#161b22", highlightthickness=1, highlightbackground="#30363d")
            lbl = tk.Label(panel, bg="black")
            lbl.pack(fill="both", expand=True)
            
            # Arrange Grid
            n = len(self.streams) + 1
            cols = 2 if n > 1 else 1
            rows = (n + 1) // 2
            
            panel.grid(row=(n-1)//cols, column=(n-1)%cols, sticky="nsew", padx=5, pady=5)
            for i in range(rows): self.grid_frame.grid_rowconfigure(i, weight=1)
            for i in range(cols): self.grid_frame.grid_columnconfigure(i, weight=1)
            
            stream = CameraStream(source, cam_id, lbl)
            if stream.start():
                self.streams.append(stream)
            else:
                panel.destroy()

# =============================================================================
# Standard Modules Integration

def open_settings():
    # Previous settings logic...
    pass

def open_dashboard():
    # Previous dashboard logic...
    pass

_toggle_vars = {k: tk.BooleanVar(value=v) for k, v in DETECTION_TOGGLES.items()}

if __name__ == "__main__":
    root = tk.Tk()
    app = MultiStreamApp(root)
    root.mainloop()