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
LSTM_CONFIDENCE_THRESHOLD = 0.65

RAISED_ARM_THRESH = 0.22
LUNGE_THRESH      = 0.28
STRIKE_THRESH     = 0.15   # lowered from 0.20 — restores pose rule sensitivity
LEAN_THRESH       = 0.30
RULE_CONFIRM_N    = 3

IDX_NOSE       = 0
IDX_L_SHOULDER = 11; IDX_R_SHOULDER = 12
IDX_L_WRIST    = 15; IDX_R_WRIST    = 16
IDX_L_HIP      = 23; IDX_R_HIP      = 24
IDX_L_KNEE     = 25; IDX_R_KNEE     = 26

VELOCITY_THRESH    = 0.35
PROXIMITY_THRESH   = 0.12
LOITER_SECONDS     = 10
LOITER_MOVE_THRESH = 0.04
# COCO class IDs: 43=knife, 34=baseball bat, 76=scissors
# Removed 73 (book) and 74 (clock) which were wrong COCO IDs
WEAPON_CLASSES     = [43, 34, 76]
WEAPON_PADDING     = 80   # px — weapon box expanded before person overlap test
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
_yolo_lock = threading.Lock()   # guards yolo_model.track() — YOLO and MediaPipe are separate, no shared state
_pose_lock = threading.Lock()   # guards pose_landmarker.detect()

try:
    lstm_model = load_model(MODEL_PATH)
    has_lstm   = True
    print("[OK] LSTM Loaded.")
except Exception as e:
    print(f"[WARNING] LSTM Load failed: {e}")
    lstm_model = None
    has_lstm   = False

# Global yolo_model removed — each CameraStream creates its own YOLO instance
# so ByteTracker state stays isolated per stream (shared instance corrupts tracking).
yolo_model = None  # kept for legacy import references only
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

def load_settings() -> dict:
    if os.path.exists(_SETTINGS_FILE):
        try:
            with open(_SETTINGS_FILE) as f:
                return json.load(f)
        except Exception: pass
    return {}

def save_settings(toggles: dict):
    try:
        with open(_SETTINGS_FILE, 'w') as f:
            json.dump({k: v.get() for k, v in toggles.items()}, f, indent=2)
    except Exception as e: print(f"[SETTINGS] Save failed: {e}")

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
    except Exception as e: print(f"[LOG ERROR]: {e}")

def save_snapshot(frame: np.ndarray, activity_type: str, camera_id="CAM") -> str:
    try:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = activity_type.replace(" ", "_").replace("|", "-")[:20]
        path = os.path.join(SNAPSHOT_DIR, f"{camera_id}_{ts}_{safe}.jpg")
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return path
    except Exception as e: print(f"[SNAPSHOT] Save failed: {e}"); return None

# =============================================================================
# Live Dashboard Class
class LiveDashboard:
    def __init__(self, parent):
        self.win = tk.Toplevel(parent)
        self.win.title("Global Threat Dashboard")
        self.win.configure(bg="#0d1117"); self.win.geometry("520x640")
        self.win.resizable(False, False); self.win.attributes("-topmost", True)
        self._start = time.time(); self._running = True
        self._build(); self._refresh()

    def _build(self):
        tk.Label(self.win, text="Global Threat Dashboard", font=("Helvetica", 16, "bold"), bg="#0d1117", fg="white").pack(pady=(14, 2))
        sf = tk.Frame(self.win, bg="#0d1117"); sf.pack(fill="x", padx=16, pady=10)
        self._total_var = tk.StringVar(value="0")
        self._session_var = tk.StringVar(value="00:00")
        for lbl, var, col in [("Total Alerts", self._total_var, "#e05252"), ("Session Time", self._session_var, "#58a6ff")]:
            f = tk.Frame(sf, bg="#161b22"); f.pack(side="left", expand=True, fill="x", padx=4)
            tk.Label(f, text=lbl, font=("Helvetica", 9), bg="#161b22", fg="#8b949e").pack(pady=(8, 0))
            tk.Label(f, textvariable=var, font=("Helvetica", 22, "bold"), bg="#161b22", fg=col).pack(pady=(0, 8))
        self._canvas = tk.Canvas(self.win, width=488, height=190, bg="#161b22", highlightthickness=0); self._canvas.pack(padx=16, pady=4)
        self._feed = tk.Frame(self.win, bg="#161b22"); self._feed.pack(fill="both", expand=True, padx=16, pady=(0, 12))

    def _refresh(self):
        if not self._running: return
        try:
            with _session_lock: events = list(_session_events)
            self._total_var.set(str(len(events)))
            e = int(time.time() - self._start); m, s = divmod(e, 60); self._session_var.set(f"{m:02d}:{s:02d}")
            counts = defaultdict(int)
            for ev in events:
                raw = ev["type"]
                key = raw.split("]")[1].strip()[:16] if "]" in raw else raw[:16]
                counts[key] += 1
            self._canvas.delete("all")
            if counts:
                items = sorted(counts.items(), key=lambda x: -x[1])[:6]; max_v = max(v for _, v in items)
                bw, gap, bh = 60, 18, 140; sx = (488 - len(items)*(bw+gap))//2
                COLS = ["#e05252","#e07752","#e0c452","#52c4e0","#5274e0","#a052e0"]
                for i, (lbl, cnt) in enumerate(items):
                    x = sx + i*(bw+gap); fill = int(bh*cnt/max_v); top = bh-fill+20; col = COLS[i%len(COLS)]
                    self._canvas.create_rectangle(x, top, x+bw, bh+20, fill=col, outline="")
                    self._canvas.create_text(x+bw//2, top-8, text=str(cnt), fill="white", font=("Helvetica",9,"bold"))
                    self._canvas.create_text(x+bw//2, bh+34, text=lbl[:9], fill="#8b949e", font=("Helvetica",8))
            for w in self._feed.winfo_children(): w.destroy()
            for ev in reversed(events[-8:]):
                row = tk.Frame(self._feed, bg="#161b22"); row.pack(fill="x", padx=4, pady=1)
                tk.Label(row, text=ev["time"], font=("Courier", 9), bg="#161b22", fg="#555d69", width=8, anchor="w").pack(side="left")
                tk.Label(row, text=ev["type"][:36], font=("Helvetica", 9), bg="#161b22", fg="#c9d1d9", anchor="w").pack(side="left", fill="x", expand=True)
        except Exception: pass
        self.win.after(1000, self._refresh)

# =============================================================================
# Camera Stream Class
class CameraStream:
    def __init__(self, source, camera_id, ui_label, status_label=None):
        self.source       = source
        self.camera_id    = camera_id
        self.ui_label     = ui_label
        self.status_label = status_label  # small bar above feed tile
        # Per-stream YOLO — isolates ByteTracker state so multi-cam tracking is correct
        self.yolo = YOLO("yolov10n.pt") if YOLO else None
        self.cap = None; self.running = False; self.thread = None
        self.track_buffers = {}; self.track_pose_buffers = {}; self.track_status = {}; self.track_scores = {}
        self.track_active_flags = {}; self.track_last_valid_pose = {}; self.track_pose_prev = {}
        self.track_center_history = {}; self.track_bbox_history = {}; self.track_wrist_history = {}
        self.track_last_lm = {}; self.prone_history = {}; self.flag_first_seen = {}
        self.track_last_alert_time = {}
        self.ALERT_COOLDOWN_S = 30
        self.flash_frames  = 0
        self.current_status = "NORMAL"  # "NORMAL" | "SUSPICIOUS" | "CRITICAL"
        self.heatmap_acc = None; self.heatmap_on = True
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.processing_tracks = set()

        # Lag Solutions state variables
        self.is_live = False
        self.capture_thread = None
        self.latest_frame = None
        self.latest_frame_id = 0
        self.track_last_pose_flag = {}

    def start(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened(): return False
        self.running = True
        
        # Lag Solution 1: Decoupled reader for live feeds to prevent OS buffer lag
        self.is_live = isinstance(self.source, int) or str(self.source).startswith(("rtsp", "http"))
        if self.is_live:
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start(); return True

    def stop(self):
        self.running = False
        if self.thread: self.thread.join(timeout=1.0)
        if self.capture_thread: self.capture_thread.join(timeout=1.0)
        if self.cap: self.cap.release()

    def _capture_loop(self):
        """Always read the freshest frame into memory."""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.latest_frame = frame
                self.latest_frame_id += 1
            else:
                time.sleep(0.01)

    def _run_loop(self):
        frame_cnt   = 0
        fail_streak = 0
        MAX_FAILS   = 10  # tolerate up to 10 consecutive bad reads before giving up
        last_processed_id = -1
        
        # Determine source FPS for throttling offline videos
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps): fps = 25.0
        frame_time = 1.0 / fps

        while self.running:
            start_t = time.time()
            if self.is_live:
                if self.latest_frame is None or self.latest_frame_id == last_processed_id:
                    time.sleep(0.01)
                    continue
                frame = self.latest_frame.copy()
                last_processed_id = self.latest_frame_id
                frame_cnt += 1
            else:
                ret, frame = self.cap.read()
                if not ret:
                    fail_streak += 1
                    if fail_streak >= MAX_FAILS:
                        break
                    time.sleep(0.05)
                    continue
                fail_streak = 0
                frame_cnt  += 1

            processed = self._process_frame(frame, frame_cnt)
            tw = self.ui_label.winfo_width()
            th = self.ui_label.winfo_height()
            if tw > 10 and th > 10:
                # cv2 resize is ~3x faster than PIL LANCZOS in a tight thread loop
                display = cv2.resize(processed, (tw, th), interpolation=cv2.INTER_LINEAR)
            else:
                display = processed
            img    = Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
            tk_img = ImageTk.PhotoImage(image=img)
            self.ui_label.after(0, lambda m=tk_img: self.ui_label.configure(image=m))
            self.ui_label.image = tk_img
            
            # Throttle offline video playback to match natural FPS
            # This prevents the async LSTM queue from backing up and skipping the event
            if not self.is_live:
                elapsed = time.time() - start_t
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)

        self.cap.release()

    def _process_frame(self, frame, frame_cnt):
        now = time.time(); ann = frame.copy(); h_f, w_f = frame.shape[:2]
        if self.heatmap_acc is None: self.heatmap_acc = np.zeros((h_f, w_f), dtype=np.float32)
        if self.yolo is None:
            return ann
        try:
            res = self.yolo.track(frame, persist=True, classes=[0]+WEAPON_CLASSES, verbose=False)
        except Exception as e:
            print(f"[YOLO ERROR {self.camera_id}]: {e}")
            return ann
        weapon_boxes = []; person_boxes = []; person_ids = []
        if res and res[0].boxes is not None:
            boxes    = res[0].boxes
            xyxy_all = boxes.xyxy.cpu().numpy()
            cls_all  = boxes.cls.int().cpu().tolist()
            id_all   = boxes.id.int().cpu().tolist() if boxes.id is not None else [None]*len(cls_all)
            for b, c, tid in zip(xyxy_all, cls_all, id_all):
                if c == 0:
                    person_boxes.append(b)
                    person_ids.append(tid)
                elif c in WEAPON_CLASSES:
                    weapon_boxes.append(b)
                    wx1, wy1, wx2, wy2 = map(int, b)
                    cv2.rectangle(ann, (wx1, wy1), (wx2, wy2), (0, 0, 255), 3)
                    cv2.putText(ann, "WEAPON", (wx1, wy1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            valid_pairs = [(box, tid) for box, tid in zip(person_boxes, person_ids) if tid is not None]
            for box, tid in valid_pairs:
                    x1, y1, x2, y2 = map(int, box); crop = frame[max(0, y1):min(h_f, y2), max(0, x1):min(w_f, x2)]
                    if crop.size == 0: continue
                    cx, cy = (x1+x2)/2/w_f, (y1+y2)/2/h_f
                    
                    # Lag Solution 3: Pose Analysis Throttling
                    moved_enough = True
                    if tid in self.track_center_history:
                        hist = self.track_center_history[tid]
                        if len(hist) > 0:
                            last_cx, last_cy, _ = hist[-1]
                            dist_px = ((cx - last_cx)*w_f)**2 + ((cy - last_cy)*h_f)**2
                            if dist_px < 25.0:  # Less than 5 pixels movement
                                moved_enough = False

                    if moved_enough or tid not in self.track_last_valid_pose:
                        raw_flag, pose_vec, lms = self._run_pose_analysis(crop, tid, (x1,y1,x2,y2), frame.shape)
                        self.track_last_pose_flag[tid] = raw_flag
                    else:
                        pose_vec = self.track_last_valid_pose[tid]
                        raw_flag = self.track_last_pose_flag.get(tid, None)
                        
                    v_flag = self._check_velocity(tid, cx, cy, now); f_flag = self._check_fall(tid, x1, y1, x2, y2, now)
                    # Weapon proximity: weapon box (with padding) overlaps person box
                    # Previous check tested person-center inside weapon-box — inverted and too strict
                    w_flag = any(
                        (int(wb[0]) - WEAPON_PADDING) < x2 and
                        (int(wb[2]) + WEAPON_PADDING) > x1 and
                        (int(wb[1]) - WEAPON_PADDING) < y2 and
                        (int(wb[3]) + WEAPON_PADDING) > y1
                        for wb in weapon_boxes
                    )
                    # LSTM Intent Analysis
                    lstm_verdict = "Tracking..."
                    if has_lstm:
                        try:
                            arr = preprocess_input(np.array(cv2.cvtColor(
                                cv2.resize(crop, (IMAGE_WIDTH, IMAGE_HEIGHT)),
                                cv2.COLOR_BGR2RGB), dtype='float32'))
                            if tid not in self.track_buffers:
                                self.track_buffers[tid] = deque(maxlen=SEQUENCE_LENGTH)
                                self.track_pose_buffers[tid] = deque(maxlen=SEQUENCE_LENGTH)
                                self.track_status[tid] = "Tracking..."
                            self.track_buffers[tid].append(arr)
                            self.track_pose_buffers[tid].append(pose_vec)
                            if (len(self.track_buffers[tid]) == SEQUENCE_LENGTH and
                                    (frame_cnt + tid) % 3 == 0 and
                                    tid not in self.processing_tracks):
                                self.processing_tracks.add(tid)
                                X_img = np.expand_dims(np.array(self.track_buffers[tid]), 0)
                                X_pose = np.expand_dims(np.array(self.track_pose_buffers[tid]), 0)
                                self.executor.submit(self._evaluate_intent_async, tid, X_img, X_pose)
                        except Exception as e: print(f"[LSTM ERROR]: {e}")
                        lstm_verdict = self.track_status.get(tid, "Tracking...")

                    i_flag = lstm_verdict if "SUSPICIOUS" in lstm_verdict else None

                    active = []
                    if w_flag   and _toggle_vars["WEAPON"].get():     active.append("WEAPON")
                    if raw_flag and _toggle_vars["POSE RULES"].get(): active.append(raw_flag)
                    if f_flag   and _toggle_vars["FALL"].get():       active.append("FALL")
                    if v_flag   and _toggle_vars["RAPID MOVE"].get(): active.append("RAPID MOVE")
                    if i_flag   and _toggle_vars["INTENT"].get():     active.append(i_flag)
                    self.track_active_flags[tid] = active

                    # ── Draw person bounding box + labels ──────────────────
                    if active:
                        box_col = (0, 0, 220)   if any(f in ("WEAPON","FALL DETECTED") for f in active) \
                             else (0, 140, 255)  # red for critical, orange for suspicious
                    else:
                        box_col = (50, 200, 50)  # green = normal
                    cv2.rectangle(ann, (x1, y1), (x2, y2), box_col, 2)

                    # Track ID + LSTM verdict in one line above box
                    verdict_short = lstm_verdict.replace("Tracking...", "...")[:20] if has_lstm else ""
                    header = f"ID{tid}  {verdict_short}"
                    (tw, th), _ = cv2.getTextSize(header, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(ann, (x1, y1 - th - 6), (x1 + tw + 4, y1), box_col, -1)
                    cv2.putText(ann, header, (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # Active flags stacked below box
                    for fi, flag in enumerate(active):
                        fy = y2 + 16 + fi * 16
                        cv2.putText(ann, flag, (x1 + 2, fy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)

                    # Per-track cooldown alert
                    now_alert = time.time()
                    last_alerted = self.track_last_alert_time.get(tid, 0)
                    if active and (now_alert - last_alerted) > self.ALERT_COOLDOWN_S:
                        self.track_last_alert_time[tid] = now_alert
                        lbl = " | ".join(active)
                        log_event(lbl, 0.90, self.camera_id)
                        snap = save_snapshot(ann, lbl, self.camera_id)
                        notifier.send_alert(lbl, 0.90, snap, self.camera_id)
                        self.flash_frames = 30
        # Prune stale track flags for IDs no longer detected in this frame
        stale = [t for t in self.track_active_flags if t not in person_ids]
        for t in stale:
            del self.track_active_flags[t]

        # Standalone Weapon Alert
        if weapon_boxes and _toggle_vars["WEAPON"].get():
            now_alert = time.time()
            last_w_alert = self.track_last_alert_time.get("GLOBAL_WEAPON", 0)
            if (now_alert - last_w_alert) > self.ALERT_COOLDOWN_S:
                self.track_last_alert_time["GLOBAL_WEAPON"] = now_alert
                lbl = "WEAPON DETECTED"
                log_event(lbl, 0.99, self.camera_id)
                snap = save_snapshot(ann, lbl, self.camera_id)
                notifier.send_alert(lbl, 0.99, snap, self.camera_id)
                self.flash_frames = 30

        if self.heatmap_on and self.heatmap_acc is not None and self.heatmap_acc.max() > 0:
            # Scale blur kernel to actual frame size (calibrated for 1280x720)
            _sf = min(w_f / 1280.0, h_f / 720.0)
            _k  = max(3, int(21 * _sf) | 1)  # must be odd, minimum 3
            norm = cv2.normalize(self.heatmap_acc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            blur = cv2.GaussianBlur(norm, (_k, _k), 0)
            cmap = cv2.applyColorMap(blur, cv2.COLORMAP_JET)
            mask = blur > 10
            ann[mask] = cv2.addWeighted(ann, 0.6, cmap, 0.4, 0)[mask]
        # Camera ID watermark — small, unobtrusive
        cv2.putText(ann, self.camera_id, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Determine stream status for the status bar
        all_flags = [f for flags in self.track_active_flags.values() for f in flags]
        if weapon_boxes and _toggle_vars["WEAPON"].get():
            all_flags.append("WEAPON DETECTED")
            
        if any("WEAPON" in f or "FALL" in f for f in all_flags):
            self.current_status = "CRITICAL"
        elif all_flags:
            self.current_status = "SUSPICIOUS"
        else:
            self.current_status = "NORMAL"

        # Push status to Tkinter status bar on the main thread
        if self.status_label is not None:
            _STATUS_MAP = {
                "CRITICAL":   ("\u26a0  CRITICAL THREAT DETECTED",  "#7f1d1d", "#fca5a5"),
                "SUSPICIOUS": ("\u25cf  SUSPICIOUS ACTIVITY",        "#1e3a5f", "#93c5fd"),
                "NORMAL":     ("\u25cf  ALL CLEAR",                  "#14532d", "#86efac"),
            }
            st, sb, sf = _STATUS_MAP[self.current_status]
            self.status_label.after(0, lambda t=st, b=sb, f=sf:
                self.status_label.configure(text=t, bg=b, fg=f))

        if self.flash_frames > 0:
            self.flash_frames -= 1
        return ann

    def _run_pose_analysis(self, crop, tid, box, f_shape):
        if pose_landmarker is None: return None, np.zeros(POSE_VECTOR_SIZE), None
        if crop.shape[0] < 48 or crop.shape[1] < 48: return None, np.zeros(POSE_VECTOR_SIZE), None
        try:
            with _pose_lock:
                res = pose_landmarker.detect(
                    mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
        except Exception:
            return None, np.zeros(POSE_VECTOR_SIZE), None
        if not res.pose_landmarks: return None, np.zeros(POSE_VECTOR_SIZE), None
        lm = res.pose_landmarks[0]
        if len(lm) <= max(IDX_L_KNEE, IDX_R_KNEE): return None, np.zeros(POSE_VECTOR_SIZE), None
        l_w, r_w = lm[IDX_L_WRIST], lm[IDX_R_WRIST]
        l_s, r_s = lm[IDX_L_SHOULDER], lm[IDX_R_SHOULDER]
        l_h, r_h = lm[IDX_L_HIP], lm[IDX_R_HIP]
        l_k, r_k = lm[IDX_L_KNEE], lm[IDX_R_KNEE]
        nose     = lm[IDX_NOSE]
        s_y = (l_s.y + r_s.y) / 2
        h_y = (l_h.y + r_h.y) / 2
        h_x = (l_h.x + r_h.x) / 2
        # RAISED ARMS removed — too many false positives (e.g. reaching for shelf)
        # Rules kept: striking posture, aggressive lean, lunge
        flag = None
        if l_k.y < h_y - STRIKE_THRESH or r_k.y < h_y - STRIKE_THRESH:
            flag = "STRIKING POSTURE"
        elif abs(nose.x - h_x) > LEAN_THRESH:
            flag = "AGGRESSIVE LEAN"
        elif tid in self.track_pose_prev:
            if abs(h_x - self.track_pose_prev[tid]) > LUNGE_THRESH:
                flag = "LUNGE DETECTED"
        self.track_pose_prev[tid] = h_x
        # Build global-coordinate pose vector
        bx1, by1, bx2, by2 = box; fh, fw = f_shape[:2]
        vec = []
        for landmark in lm:
            gx = landmark.x * (bx2-bx1) / max(fw,1) + bx1 / max(fw,1)
            gy = landmark.y * (by2-by1) / max(fh,1) + by1 / max(fh,1)
            vec.extend([gx, gy, landmark.z])
        pose_vec = np.array(vec, dtype='float32')
        self.track_last_valid_pose[tid] = pose_vec
        return flag, pose_vec, lm

    def _evaluate_intent_async(self, t_id, X_seq, X_pose_seq):
        try:
            preds = lstm_model({"image_input": X_seq, "pose_input": X_pose_seq}, training=False).numpy()[0]
            if t_id not in self.track_scores: self.track_scores[t_id] = deque(maxlen=3)
            self.track_scores[t_id].append(preds)
            avg = np.mean(self.track_scores[t_id], axis=0)
            conf = float(np.max(avg))
            self.track_status[t_id] = f"SUSPICIOUS ({avg[1]*100:.0f}%)" if np.argmax(avg) == 1 and conf > LSTM_CONFIDENCE_THRESHOLD else f"Normal ({avg[0]*100:.0f}%)"
        except Exception as e: print(f"[LSTM ERROR]: {e}")
        finally: self.processing_tracks.discard(t_id)

    def _check_velocity(self, tid, cx, cy, now):
        if tid not in self.track_center_history:
            self.track_center_history[tid] = deque(maxlen=10)
        self.track_center_history[tid].append((cx, cy, now))
        hist = self.track_center_history[tid]
        if len(hist) < 4:
            return None
        dt   = max(hist[-1][2] - hist[0][2], 0.001)
        dist = ((hist[-1][0] - hist[0][0])**2 + (hist[-1][1] - hist[0][1])**2) ** 0.5
        return "RAPID MOVE" if dist / dt > VELOCITY_THRESH else None

    def _check_fall(self, tid, x1, y1, x2, y2, now):
        h = y2 - y1
        if tid not in self.track_bbox_history:
            self.track_bbox_history[tid] = deque(maxlen=20)
        self.track_bbox_history[tid].append((h, now))
        hist = self.track_bbox_history[tid]
        if len(hist) < 10:
            return None
        prev_h = sum(v[0] for v in list(hist)[:5])  / 5
        curr_h = sum(v[0] for v in list(hist)[-5:]) / 5
        if prev_h > 0 and (prev_h - curr_h) / prev_h > FALL_THRESH:
            return "FALL DETECTED"
        return None

# =============================================================================
# Main UI — Single Camera
class SurveillanceApp:
    def __init__(self, root):
        self.root   = root
        self.stream = None          # currently active CameraStream (or None)
        self.root.title("Hybrid AI Surveillance System")
        self.root.state("zoomed")
        self.root.configure(bg="#0d1117")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self):
        # ── Header bar ────────────────────────────────────────────────
        hdr = tk.Frame(self.root, bg="#161b22", height=56)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        tk.Label(hdr, text="COMMAND CENTER",
                 font=("Helvetica", 18, "bold"),
                 bg="#161b22", fg="white").pack(side="left", padx=20)

        btn_cfg = dict(relief="flat", font=("Helvetica", 10, "bold"),
                       padx=14, pady=4, cursor="hand2")
        ctrl = tk.Frame(hdr, bg="#161b22"); ctrl.pack(side="right", padx=16)

        tk.Button(ctrl, text="🎥  Webcam",
                  command=lambda: self._open_source(0),
                  bg="#238636", fg="white", **btn_cfg).pack(side="left", padx=4)
        tk.Button(ctrl, text="📁  Open File",
                  command=self._browse_file,
                  bg="#1f6feb", fg="white", **btn_cfg).pack(side="left", padx=4)
        tk.Button(ctrl, text="📡  RTSP / IP",
                  command=self._prompt_rtsp,
                  bg="#6e40c9", fg="white", **btn_cfg).pack(side="left", padx=4)
        tk.Button(ctrl, text="⏹  Stop",
                  command=self._stop_stream,
                  bg="#b62324", fg="white", **btn_cfg).pack(side="left", padx=4)

        tk.Frame(ctrl, bg="#30363d", width=1, height=28).pack(side="left", padx=8)

        tk.Button(ctrl, text="Dashboard",
                  command=open_dashboard,
                  bg="#30363d", fg="white", **btn_cfg).pack(side="left", padx=4)
        tk.Button(ctrl, text="⚙ Settings",
                  command=open_settings,
                  bg="#30363d", fg="white", **btn_cfg).pack(side="left", padx=4)

        # ── Feed area ─────────────────────────────────────────────────
        feed_frame = tk.Frame(self.root, bg="#0d1117")
        feed_frame.pack(fill="both", expand=True, padx=10, pady=(6, 10))

        # Status bar
        self.status_lbl = tk.Label(
            feed_frame,
            text="●  IDLE — open a source to begin",
            font=("Helvetica", 10, "bold"),
            bg="#1c2128", fg="#8b949e",
            anchor="w", padx=12, pady=4
        )
        self.status_lbl.pack(fill="x", side="top")

        # Video label
        self.video_lbl = tk.Label(feed_frame, bg="#0d1117")
        self.video_lbl.pack(fill="both", expand=True)

    # ------------------------------------------------------------------
    def _open_source(self, source):
        self._stop_stream()
        cid = "CAM-1" if isinstance(source, int) else "FILE"
        self.stream = CameraStream(source, cid, self.video_lbl, self.status_lbl)
        if not self.stream.start():
            self.stream = None
            messagebox.showerror(
                "Stream Error",
                f"Could not open source:\n{source}\n\nCheck the URL or device.",
                parent=self.root)

    def _browse_file(self):
        path = askopenfilename(
            parent=self.root,
            title="Select video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov"), ("All files", "*.*")])
        if path:
            self._open_source(path)

    def _prompt_rtsp(self):
        url = simpledialog.askstring(
            "RTSP / IP Camera",
            "Enter RTSP URL or IP stream address:",
            parent=self.root)
        if url and url.strip():
            self._open_source(url.strip())

    def _stop_stream(self):
        if self.stream:
            self.stream.stop()
            self.stream = None
        self.video_lbl.configure(image="")
        self.status_lbl.configure(
            text="●  STOPPED — open a source to begin",
            bg="#1c2128", fg="#8b949e")

    def _on_close(self):
        self._stop_stream()
        self.root.destroy()


# Module-level placeholders — populated after tk.Tk() is created in __main__
root = None
_toggle_vars: dict = {k: None for k in DETECTION_TOGGLES}  # filled in __main__

def open_settings():
    win = tk.Toplevel(root); win.title("Settings"); win.configure(bg="#0d1117"); win.geometry("440x520")
    win.resizable(False, False); win.attributes("-topmost", True)
    tk.Label(win, text="Detection Toggles", font=("Helvetica", 13, "bold"), bg="#0d1117", fg="white").pack(pady=10)
    tk.Label(win, text="Applies to the active camera stream", font=("Helvetica", 9), bg="#0d1117", fg="#8b949e").pack()
    LABELS = {
        "WEAPON": "Weapon detection", "INTENT": "LSTM suspicious intent",
        "POSE RULES": "Pose rules (striking posture, lunge…)", "FALL": "Fall detection",
        "PERSON DOWN": "Person down (prone)", "RAPID MOVE": "Rapid movement",
        "FLAILING": "Arm flailing", "LOITERING": "Loitering", "PROXIMITY": "Proximity / conflict",
    }
    for k, lbl in LABELS.items():
        var = _toggle_vars.get(k)
        if var is not None:
            tk.Checkbutton(win, text=lbl, variable=var, bg="#0d1117", fg="white",
                           selectcolor="#1f6feb", activebackground="#0d1117", activeforeground="white",
                           font=("Helvetica", 11), anchor="w",
                           command=lambda: save_settings(_toggle_vars)).pack(fill="x", padx=20, pady=2)
    tk.Label(win, text="Email Alert Config", font=("Helvetica", 13, "bold"), bg="#0d1117", fg="white").pack(pady=(14, 2))
    tk.Label(win, text="Alerts sent from suspiciousactivity678@gmail.com", font=("Helvetica", 9), bg="#0d1117", fg="#8b949e").pack()
    cfg = notifier.load_config()
    tk.Label(win, text="Recipient Email", bg="#0d1117", fg="#c9d1d9", anchor="w").pack(fill="x", padx=20, pady=(8,0))
    recipient_entry = tk.Entry(win, bg="#161b22", fg="white", insertbackground="white", relief="flat")
    recipient_entry.insert(0, cfg.get("recipient_email", ""))
    recipient_entry.pack(fill="x", padx=20, pady=2, ipady=4)
    def _save_email():
        nc = notifier.load_config()
        nc["recipient_email"] = recipient_entry.get().strip()
        nc["enabled"] = True
        notifier.save_config(nc)
        messagebox.showinfo("Saved", "Recipient email saved. Alerts are now enabled.", parent=win)
    tk.Button(win, text="Save", command=_save_email, bg="#238636", fg="white",
              relief="flat", font=("Helvetica", 11, "bold")).pack(pady=10)
    tk.Button(win, text="Close", command=win.destroy, bg="#21262d", fg="#8b949e", relief="flat").pack()

def open_dashboard(): LiveDashboard(root)

if __name__ == "__main__":
    root = tk.Tk()
    saved = load_settings()
    _toggle_vars = {k: tk.BooleanVar(value=saved.get(k, v)) for k, v in DETECTION_TOGGLES.items()}
    app = SurveillanceApp(root)
    root.mainloop()

