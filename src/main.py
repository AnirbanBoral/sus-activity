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
            for ev in events: counts[ev["type"].split("]")[1].strip()[:16]] += 1
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
    def __init__(self, source, camera_id, ui_label):
        self.source = source; self.camera_id = camera_id; self.ui_label = ui_label
        self.cap = None; self.running = False; self.thread = None
        self.track_buffers = {}; self.track_pose_buffers = {}; self.track_status = {}
        self.track_active_flags = {}; self.track_last_valid_pose = {}; self.track_pose_prev = {}
        self.track_center_history = {}; self.track_bbox_history = {}; self.track_wrist_history = {}
        self.track_last_lm = {}; self.prone_history = {}; self.flag_first_seen = {}
        self.alert_sent = False; self.normal_frames = 0; self.flash_frames = 0
        self.heatmap_acc = None; self.heatmap_on = True; self.executor = ThreadPoolExecutor(max_workers=2)
        self.processing_tracks = set()

    def start(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened(): return False
        self.running = True; self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start(); return True

    def stop(self):
        self.running = False
        if self.thread: self.thread.join(timeout=1.0)
        if self.cap: self.cap.release()

    def _run_loop(self):
        frame_cnt   = 0
        fail_streak = 0
        MAX_FAILS   = 10  # tolerate up to 10 consecutive bad reads before giving up

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                fail_streak += 1
                if fail_streak >= MAX_FAILS:
                    break
                time.sleep(0.05)
                continue
            fail_streak = 0
            frame_cnt  += 1
            if frame_cnt % 2 != 0:
                continue
            processed = self._process_frame(frame, frame_cnt)
            img = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
            w, h = self.ui_label.winfo_width(), self.ui_label.winfo_height()
            if w > 10 and h > 10:
                img = img.resize((w, h), Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(image=img)
            self.ui_label.after(0, lambda m=tk_img: self.ui_label.configure(image=m))
            self.ui_label.image = tk_img

        self.cap.release()

    def _process_frame(self, frame, frame_cnt):
        now = time.time(); ann = frame.copy(); h_f, w_f = frame.shape[:2]
        if self.heatmap_acc is None: self.heatmap_acc = np.zeros((h_f, w_f), dtype=np.float32)
        with _yolo_lock: res = yolo_model.track(frame, persist=True, classes=[0]+WEAPON_CLASSES, verbose=False)
        weapon_boxes = []; person_boxes = []; person_ids = []
        if res and res[0].boxes is not None:
            for b, c in zip(res[0].boxes.xyxy.cpu().numpy(), res[0].boxes.cls.int().cpu().tolist()):
                if c == 0: person_boxes.append(b)
                elif c in WEAPON_CLASSES:
                    weapon_boxes.append(b)
                    wx1, wy1, wx2, wy2 = map(int, b)
                    cv2.rectangle(ann, (wx1, wy1), (wx2, wy2), (0, 0, 255), 3)
                    cv2.putText(ann, "WEAPON", (wx1, wy1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if res[0].boxes.id is not None:
                person_ids = [tid for tid, cls in zip(res[0].boxes.id.int().cpu().tolist(), res[0].boxes.cls.int().cpu().tolist()) if cls == 0]
                for box, tid in zip(person_boxes, person_ids):
                    x1, y1, x2, y2 = map(int, box); crop = frame[max(0, y1):min(h_f, y2), max(0, x1):min(w_f, x2)]
                    if crop.size == 0: continue
                    raw_flag, pose_vec, lms = self._run_pose_analysis(crop, tid, (x1,y1,x2,y2), frame.shape)
                    cx, cy = (x1+x2)/2/w_f, (y1+y2)/2/h_f
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
                    active = []
                    if w_flag   and _toggle_vars["WEAPON"].get():     active.append("WEAPON")
                    if raw_flag and _toggle_vars["POSE RULES"].get(): active.append(raw_flag)
                    if f_flag   and _toggle_vars["FALL"].get():       active.append("FALL")
                    if v_flag   and _toggle_vars["RAPID MOVE"].get(): active.append("RAPID MOVE")
                    self.track_active_flags[tid] = active
                    if active and not self.alert_sent:
                        self.alert_sent = True; lbl = " | ".join(active); log_event(lbl, 0.90, self.camera_id)
                        snap = save_snapshot(ann, lbl, self.camera_id); notifier.send_alert(lbl, 0.90, snap); self.flash_frames = 30
                        _sf = min(w_f / 1280.0, h_f / 720.0)
                        _r  = max(10, int(60 * _sf))
                        cv2.circle(self.heatmap_acc, (int((x1+x2)/2), int((y1+y2)/2)), _r, 1.0, -1)
        if self.heatmap_on and self.heatmap_acc is not None and self.heatmap_acc.max() > 0:
            # Scale blur kernel to actual frame size (calibrated for 1280x720)
            _sf = min(w_f / 1280.0, h_f / 720.0)
            _k  = max(3, int(21 * _sf) | 1)  # must be odd, minimum 3
            norm = cv2.normalize(self.heatmap_acc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            blur = cv2.GaussianBlur(norm, (_k, _k), 0)
            cmap = cv2.applyColorMap(blur, cv2.COLORMAP_JET)
            mask = blur > 10
            ann[mask] = cv2.addWeighted(ann, 0.6, cmap, 0.4, 0)[mask]
        cv2.putText(ann, f"{self.camera_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if self.flash_frames > 0:
            self.flash_frames -= 1
            if self.flash_frames % 4 < 2: cv2.rectangle(ann, (0, 0), (w_f, h_f), (0, 0, 255), 10)
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
# Main UI Logic
class MultiStreamApp:
    def __init__(self, root):
        self.root = root; self.root.title("Hybrid AI Command Center"); self.root.state('zoomed'); self.root.configure(bg="#0d1117")
        self.streams = []; self._build_ui()

    def _build_ui(self):
        h = tk.Frame(self.root, bg="#161b22", height=60); h.pack(fill="x")
        tk.Label(h, text="COMMAND CENTER", font=("Helvetica", 18, "bold"), bg="#161b22", fg="white").pack(side="left", padx=20)
        c = tk.Frame(h, bg="#161b22"); c.pack(side="right", padx=20)
        tk.Button(c, text="+ Add Camera", command=self._add_camera, bg="#238636", fg="white", relief="flat", padx=15).pack(side="left", padx=5)
        tk.Button(c, text="Dashboard", command=open_dashboard, bg="#1f6feb", fg="white", relief="flat", padx=15).pack(side="left", padx=5)
        tk.Button(c, text="Settings", command=open_settings, bg="#30363d", fg="white", relief="flat", padx=15).pack(side="left", padx=5)
        self.grid = tk.Frame(self.root, bg="#0d1117"); self.grid.pack(fill="both", expand=True, padx=10, pady=10)

    def _add_camera(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("Add Stream"); dlg.configure(bg="#0d1117")
        dlg.geometry("420x185"); dlg.resizable(False, False)
        dlg.attributes("-topmost", True); dlg.grab_set()

        tk.Label(dlg, text="Enter RTSP URL, device ID (0, 1), or pick a video file:",
                 bg="#0d1117", fg="#c9d1d9", font=("Helvetica", 10),
                 wraplength=380, justify="left").pack(padx=16, pady=(14, 6), anchor="w")

        entry_var = tk.StringVar()
        row = tk.Frame(dlg, bg="#0d1117"); row.pack(fill="x", padx=16)
        entry = tk.Entry(row, textvariable=entry_var, bg="#161b22", fg="white",
                         insertbackground="white", relief="flat", font=("Helvetica", 11))
        entry.pack(side="left", fill="x", expand=True, ipady=5)

        def _browse():
            path = askopenfilename(
                parent=dlg, title="Select video file",
                filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov"), ("All files", "*.*")])
            if path: entry_var.set(path)

        tk.Button(row, text="Browse…", command=_browse,
                  bg="#30363d", fg="white", relief="flat", padx=8).pack(side="left", padx=(6, 0))

        def _confirm():
            src = entry_var.get().strip()
            if not src: dlg.destroy(); return
            if src.isdigit(): src = int(src)
            dlg.destroy()
            cid = f"CAM-{len(self.streams) + 1}"
            p = tk.Frame(self.grid, bg="#161b22", highlightthickness=1, highlightbackground="#30363d")
            l = tk.Label(p, bg="black"); l.pack(fill="both", expand=True)
            n = len(self.streams) + 1; cols = 2 if n > 1 else 1; rows = (n + 1) // 2
            p.grid(row=(n-1)//cols, column=(n-1)%cols, sticky="nsew", padx=5, pady=5)
            for i in range(rows): self.grid.grid_rowconfigure(i, weight=1)
            for i in range(cols): self.grid.grid_columnconfigure(i, weight=1)
            s = CameraStream(src, cid, l)
            if s.start():
                self.streams.append(s)
            else:
                p.destroy()
                messagebox.showerror("Stream Error",
                                     f"Could not open:\n{src}\n\nCheck the URL or file path.",
                                     parent=self.root)

        btn_row = tk.Frame(dlg, bg="#0d1117"); btn_row.pack(pady=12)
        tk.Button(btn_row, text="Connect", command=_confirm, bg="#238636", fg="white",
                  relief="flat", font=("Helvetica", 11, "bold"), padx=16).pack(side="left", padx=6)
        tk.Button(btn_row, text="Cancel", command=dlg.destroy,
                  bg="#21262d", fg="#8b949e", relief="flat", padx=16).pack(side="left")
        dlg.bind("<Return>", lambda e: _confirm())
        entry.focus_set()

# Module-level placeholders — populated after tk.Tk() is created in __main__
root = None
_toggle_vars: dict = {k: None for k in DETECTION_TOGGLES}  # filled in __main__

def open_settings():
    win = tk.Toplevel(root); win.title("Settings"); win.configure(bg="#0d1117"); win.geometry("440x620")
    win.resizable(False, False); win.attributes("-topmost", True)
    tk.Label(win, text="Detection Toggles", font=("Helvetica", 13, "bold"), bg="#0d1117", fg="white").pack(pady=10)
    tk.Label(win, text="Applies to all active camera streams", font=("Helvetica", 9), bg="#0d1117", fg="#8b949e").pack()
    LABELS = {
        "WEAPON": "Weapon detection", "INTENT": "LSTM suspicious intent",
        "POSE RULES": "Pose rules (raised arms, lunge…)", "FALL": "Fall detection",
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
    tk.Label(win, text="Use a Gmail App Password", font=("Helvetica", 9), bg="#0d1117", fg="#8b949e").pack()
    cfg = notifier.load_config(); entries = {}
    for lbl, k, show in [("Sender Gmail", "sender_email", ""), ("App Password", "sender_password", "*"), ("Recipient", "recipient_email", "")]:
        tk.Label(win, text=lbl, bg="#0d1117", fg="#c9d1d9", anchor="w").pack(fill="x", padx=20)
        e = tk.Entry(win, bg="#161b22", fg="white", insertbackground="white", relief="flat", show=show)
        e.insert(0, cfg.get(k, "")); e.pack(fill="x", padx=20, pady=2, ipady=4); entries[k] = e
    def _save_email():
        nc = notifier.load_config()
        for fk, ew in entries.items(): nc[fk] = ew.get().strip()
        notifier.save_config(nc); messagebox.showinfo("Saved", "Email settings saved.", parent=win)
    tk.Button(win, text="Save Email Settings", command=_save_email, bg="#238636", fg="white",
              relief="flat", font=("Helvetica", 11, "bold")).pack(pady=10)
    tk.Button(win, text="Close", command=win.destroy, bg="#21262d", fg="#8b949e", relief="flat").pack()

def open_dashboard(): LiveDashboard(root)

if __name__ == "__main__":
    root = tk.Tk()
    # BooleanVar MUST be created after tk.Tk() root window exists
    saved = load_settings()
    _toggle_vars = {k: tk.BooleanVar(value=saved.get(k, v)) for k, v in DETECTION_TOGGLES.items()}
    app = MultiStreamApp(root)
    root.mainloop()