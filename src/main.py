import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import json
import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog, messagebox
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
    print("[WARNING] MediaPipe not found. Run 'pip install mediapipe'. Pose rules disabled.")
    HAS_MEDIAPIPE = False

# YOLOv8/v10
try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] Ultralytics not found. Run 'pip install ultralytics'.")
    YOLO = None

# =============================================================================
# Configuration
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
_prone_history     = {}

DETECTION_TOGGLES = {
    "WEAPON": True, "INTENT": True, "POSE RULES": True,
    "FALL": True, "PERSON DOWN": True, "RAPID MOVE": True,
    "FLAILING": True, "LOITERING": True, "PROXIMITY": True,
}

# =============================================================================
# Settings persistence

def load_settings() -> dict:
    if os.path.exists(_SETTINGS_FILE):
        try:
            with open(_SETTINGS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_settings(toggles: dict):
    try:
        with open(_SETTINGS_FILE, 'w') as f:
            json.dump({k: v.get() for k, v in toggles.items()}, f, indent=2)
    except Exception as e:
        print(f"[SETTINGS] Save failed: {e}")

# =============================================================================
# Load Models
print("[INFO] Initializing LSTM Intent Model...")
try:
    lstm_model = load_model(MODEL_PATH)
    has_lstm   = True
    print("[OK] LSTM Loaded.")
except Exception as e:
    print(f"[WARNING] Failed to load LSTM: {e}")
    lstm_model = None
    has_lstm   = False

print("[INFO] Initializing YOLO Tracker...")
# Preserve YOLOv10n upgrade
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
elif HAS_MEDIAPIPE:
    print(f"[WARNING] Pose model not found at {POSE_MODEL}.")

# =============================================================================
# Thread-safe logger
_log_lock = threading.Lock()

def log_event(activity_type, confidence):
    try:
        with _log_lock:
            exists = os.path.isfile(LOG_FILE)
            with open(LOG_FILE, 'a', newline='') as f:
                w = csv.writer(f)
                if not exists:
                    w.writerow(['Timestamp', 'Activity Type', 'Confidence'])
                w.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            activity_type, f"{confidence:.2f}"])
    except Exception as e:
        print(f"[LOG ERROR]: {e}")

# =============================================================================
# Session event store (feeds live dashboard)
_session_events = []
_session_lock   = threading.Lock()

def record_session_event(activity_type: str, confidence: float):
    with _session_lock:
        _session_events.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "type": activity_type,
            "conf": confidence
        })

# =============================================================================
# Snapshot saver

def save_snapshot(frame: np.ndarray, activity_type: str) -> str:
    try:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = activity_type.replace(" ", "_").replace("|", "-")[:30]
        path = os.path.join(SNAPSHOT_DIR, f"alert_{ts}_{safe}.jpg")
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return path
    except Exception as e:
        print(f"[SNAPSHOT] Save failed: {e}")
        return None

# =============================================================================
# Live Dashboard

class LiveDashboard:
    def __init__(self, parent):
        self.win = tk.Toplevel(parent)
        self.win.title("Live Threat Dashboard")
        self.win.configure(bg="#0d1117")
        self.win.geometry("520x640")
        self.win.resizable(False, False)
        self.win.attributes("-topmost", True)
        self._start = time.time()
        self._running = True
        self._build()
        self._refresh()

    def _build(self):
        tk.Label(self.win, text="Live Threat Dashboard",
                 font=("Helvetica", 16, "bold"), bg="#0d1117", fg="white").pack(pady=(14, 2))
        tk.Label(self.win, text="Auto-refreshes every second",
                 font=("Helvetica", 9), bg="#0d1117", fg="#8b949e").pack()

        sf = tk.Frame(self.win, bg="#0d1117")
        sf.pack(fill="x", padx=16, pady=10)
        self._total_var   = tk.StringVar(value="0")
        self._session_var = tk.StringVar(value="00:00")
        for lbl, var, col in [("Total Alerts", self._total_var, "#e05252"),
                               ("Session Time",  self._session_var, "#58a6ff")]:
            f = tk.Frame(sf, bg="#161b22")
            f.pack(side="left", expand=True, fill="x", padx=4)
            tk.Label(f, text=lbl, font=("Helvetica", 9),
                     bg="#161b22", fg="#8b949e").pack(pady=(8, 0))
            tk.Label(f, textvariable=var, font=("Helvetica", 22, "bold"),
                     bg="#161b22", fg=col).pack(pady=(0, 8))

        tk.Label(self.win, text="Threats by Type",
                 font=("Helvetica", 11, "bold"), bg="#0d1117", fg="#c9d1d9").pack(anchor="w", padx=16)
        self._canvas = tk.Canvas(self.win, width=488, height=190,
                                  bg="#161b22", highlightthickness=0)
        self._canvas.pack(padx=16, pady=4)

        tk.Label(self.win, text="Recent Alerts",
                 font=("Helvetica", 11, "bold"), bg="#0d1117", fg="#c9d1d9").pack(
            anchor="w", padx=16, pady=(10, 2))
        self._feed = tk.Frame(self.win, bg="#161b22")
        self._feed.pack(fill="both", expand=True, padx=16, pady=(0, 12))

    def _refresh(self):
        if not self._running:
            return
        try:
            with _session_lock:
                events = list(_session_events)

            self._total_var.set(str(len(events)))
            e = int(time.time() - self._start)
            m, s = divmod(e, 60)
            self._session_var.set(f"{m:02d}:{s:02d}")

            counts = defaultdict(int)
            for ev in events:
                counts[ev["type"].split("|")[0].strip()[:16]] += 1

            self._canvas.delete("all")
            if counts:
                items   = sorted(counts.items(), key=lambda x: -x[1])[:6]
                max_val = max(v for _, v in items)
                bw, gap, bh = 60, 18, 140
                total_w     = len(items) * (bw + gap)
                sx          = (488 - total_w) // 2
                COLS = ["#e05252","#e07752","#e0c452","#52c4e0","#5274e0","#a052e0"]
                for i, (lbl, cnt) in enumerate(items):
                    x    = sx + i * (bw + gap)
                    fill = int(bh * cnt / max(max_val, 1))
                    top  = bh - fill + 20
                    col  = COLS[i % len(COLS)]
                    self._canvas.create_rectangle(x, top, x+bw, bh+20, fill=col, outline="")
                    self._canvas.create_text(x+bw//2, top-8, text=str(cnt),
                                              fill="white", font=("Helvetica", 9, "bold"))
                    self._canvas.create_text(x+bw//2, bh+34, text=lbl[:9],
                                              fill="#8b949e", font=("Helvetica", 8))
            else:
                self._canvas.create_text(244, 95, text="No alerts yet",
                                          fill="#444d56", font=("Helvetica", 12))

            for w in self._feed.winfo_children():
                w.destroy()
            for ev in reversed(events[-8:]):
                row = tk.Frame(self._feed, bg="#161b22")
                row.pack(fill="x", padx=4, pady=1)
                cc = "#e05252" if ev["conf"] > 0.80 else "#e0a052"
                tk.Label(row, text=ev["time"], font=("Courier", 9),
                         bg="#161b22", fg="#555d69", width=8, anchor="w").pack(side="left")
                tk.Label(row, text=f"{ev['conf']*100:.0f}%", font=("Helvetica", 9, "bold"),
                         bg="#161b22", fg=cc, width=5).pack(side="left")
                tk.Label(row, text=ev["type"][:36], font=("Helvetica", 9),
                         bg="#161b22", fg="#c9d1d9", anchor="w").pack(side="left", fill="x", expand=True)
        except Exception as e:
            print(f"[DASHBOARD] {e}")
        self.win.after(1000, self._refresh)

    def destroy(self):
        self._running = False
        try:
            self.win.destroy()
        except Exception:
            pass

# =============================================================================
# Pose Rule Engine

def _draw_skeleton(crop_bgr, landmarks):
    h, w = crop_bgr.shape[:2]
    CONNECTIONS = [(IDX_L_SHOULDER,IDX_R_SHOULDER),(IDX_L_SHOULDER,IDX_L_WRIST),
                   (IDX_R_SHOULDER,IDX_R_WRIST),(IDX_L_HIP,IDX_R_HIP),
                   (IDX_L_SHOULDER,IDX_L_HIP),(IDX_R_SHOULDER,IDX_R_HIP),
                   (IDX_L_HIP,IDX_L_KNEE),(IDX_R_HIP,IDX_R_KNEE)]
    pts = [(int(lm.x*w), int(lm.y*h)) for lm in landmarks]
    ann = crop_bgr.copy()
    for a, b in CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(ann, pts[a], pts[b], (0, 255, 128), 2)
    for pt in pts:
        cv2.circle(ann, pt, 4, (255, 255, 0), -1)
    return ann


def run_pose_rules(crop_bgr, track_id, track_pose_prev, track_last_valid_pose,
                   box=None, frame_shape=None):
    pose_vec = track_last_valid_pose.get(track_id, np.zeros((POSE_VECTOR_SIZE,), dtype='float32'))
    if pose_landmarker is None or crop_bgr.shape[0] < 48 or crop_bgr.shape[1] < 48:
        return None, crop_bgr, pose_vec, None
    try:
        result = pose_landmarker.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)))
    except Exception:
        return None, crop_bgr, pose_vec, None
    if not result.pose_landmarks:
        return None, crop_bgr, pose_vec, None
    lm = result.pose_landmarks[0]
    req = [IDX_NOSE,IDX_L_SHOULDER,IDX_R_SHOULDER,IDX_L_WRIST,IDX_R_WRIST,
           IDX_L_HIP,IDX_R_HIP,IDX_L_KNEE,IDX_R_KNEE]
    if max(req) >= len(lm):
        return None, crop_bgr, pose_vec, None
    vec = []
    for landmark in lm:
        if box is not None and frame_shape is not None:
            bx1,by1,bx2,by2 = box; fh,fw = frame_shape[:2]
            vec.extend([landmark.x*(bx2-bx1)/max(fw,1)+bx1/max(fw,1),
                        landmark.y*(by2-by1)/max(fh,1)+by1/max(fh,1), landmark.z])
        else:
            vec.extend([landmark.x, landmark.y, landmark.z])
    pose_vec = np.array(vec, dtype='float32')
    track_last_valid_pose[track_id] = pose_vec
    nose=lm[IDX_NOSE]; l_sho=lm[IDX_L_SHOULDER]; r_sho=lm[IDX_R_SHOULDER]
    l_wrist=lm[IDX_L_WRIST]; r_wrist=lm[IDX_R_WRIST]
    l_hip=lm[IDX_L_HIP]; r_hip=lm[IDX_R_HIP]; l_knee=lm[IDX_L_KNEE]; r_knee=lm[IDX_R_KNEE]
    shoulder_y=(l_sho.y+r_sho.y)/2; hip_y=(l_hip.y+r_hip.y)/2; hip_x=(l_hip.x+r_hip.x)/2
    raw_flag = None
    if l_wrist.y < shoulder_y-RAISED_ARM_THRESH and r_wrist.y < shoulder_y-RAISED_ARM_THRESH:
        raw_flag = "RAISED ARMS"
    if not raw_flag and (l_knee.y < hip_y-STRIKE_THRESH or r_knee.y < hip_y-STRIKE_THRESH):
        raw_flag = "STRIKING POSTURE"
    if not raw_flag and abs(nose.x-hip_x) > LEAN_THRESH:
        raw_flag = "AGGRESSIVE LEAN"
    if not raw_flag and track_id in track_pose_prev:
        if abs(hip_x-track_pose_prev[track_id]) > LUNGE_THRESH:
            raw_flag = "LUNGE DETECTED"
    track_pose_prev[track_id] = hip_x
    return raw_flag, _draw_skeleton(crop_bgr, lm), pose_vec, lm

# =============================================================================
# Behavioral Helpers

def get_center(x1,y1,x2,y2,fw,fh):
    return ((x1+x2)/2/max(fw,1), (y1+y2)/2/max(fh,1))

def check_velocity(tid,cx,cy,now,hist_d):
    if tid not in hist_d: hist_d[tid]=deque(maxlen=10)
    hist_d[tid].append((cx,cy,now)); h=hist_d[tid]
    if len(h)<4: return None
    dt=max(h[-1][2]-h[0][2],0.001)
    dist=((h[-1][0]-h[0][0])**2+(h[-1][1]-h[0][1])**2)**0.5
    return "RAPID MOVEMENT" if dist/dt>VELOCITY_THRESH else None

def check_loitering(tid,cx,cy,now,first_d,moved_d,hist_d):
    if tid not in first_d: first_d[tid]=now; moved_d[tid]=now
    h=hist_d.get(tid)
    if h and len(h)>=2:
        if ((h[-1][0]-h[-2][0])**2+(h[-1][1]-h[-2][1])**2)**0.5 > LOITER_MOVE_THRESH:
            moved_d[tid]=now
    stay=now-moved_d.get(tid,now)
    return f"LOITERING ({int(stay)}s)" if stay>LOITER_SECONDS else None

def check_proximity(boxes,tids,fw,fh):
    conflicts,centers=[],[]
    fw=max(fw,1); fh=max(fh,1)
    for box,tid in zip(boxes,tids):
        bx1,by1,bx2,by2=box
        centers.append((tid,(bx1+bx2)/2/fw,(by1+by2)/2/fh))
    c_set=set()
    for i in range(len(centers)):
        for j in range(i+1,len(centers)):
            if ((centers[i][1]-centers[j][1])**2+(centers[i][2]-centers[j][2])**2)**0.5 < PROXIMITY_THRESH:
                c_set.add(centers[i][0]); c_set.add(centers[j][0])
    return c_set

def weapon_near_person(px1,py1,px2,py2,wboxes):
    for wb in wboxes:
        wx1,wy1,wx2,wy2=map(int,wb)
        wcx,wcy=(wx1+wx2)/2,(wy1+wy2)/2
        if px1<wcx<px2 and py1<wcy<py2: return True
    return False

def check_fall(tid,x1,y1,x2,y2,now,hist_d):
    h=y2-y1
    if tid not in hist_d: hist_d[tid]=deque(maxlen=20)
    hist_d[tid].append((h,now)); bh=hist_d[tid]
    if len(bh)<10: return None
    prev_h=np.mean([v[0] for v in list(bh)[:5]])
    curr_h=np.mean([v[0] for v in list(bh)[-5:]])
    return "FALL DETECTED" if prev_h>0 and (prev_h-curr_h)/prev_h>FALL_THRESH else None

def check_prone(tid,x1,y1,x2,y2):
    w,h=x2-x1,y2-y1
    if tid not in _prone_history: _prone_history[tid]=deque(maxlen=PRONE_CONFIRM_N)
    is_wide=h>PRONE_MIN_HEIGHT and h>0 and w/h>PRONE_RATIO
    _prone_history[tid].append(is_wide); ph=_prone_history[tid]
    return "PERSON DOWN" if len(ph)==PRONE_CONFIRM_N and all(ph) else None

def check_arm_flail(tid,lms,hist_d):
    if lms is None or len(lms)<=16: return None
    try:
        lw,rw=lms[15],lms[16]; pos=(lw.x,lw.y,rw.x,rw.y)
        if tid not in hist_d: hist_d[tid]=deque(maxlen=12)
        h=hist_d[tid]
        if h:
            delta=sum((a-b)**2 for a,b in zip(pos,h[-1]))**0.5
            h.append(pos)
            if delta>FLAIL_THRESH:
                cnt=sum(1 for i in range(1,len(h))
                        if ((h[i][0]-h[i-1][0])**2+(h[i][1]-h[i-1][1])**2)**0.5>FLAIL_THRESH)
                if cnt>=FLAIL_MIN_FRAMES: return "ARM FLAILING"
        else: h.append(pos)
    except Exception: pass
    return None

def get_time_multiplier():
    hr=datetime.now().hour
    return 1.3 if (22<=hr or hr<=5) else 1.0

def parse_lstm_confidence(v:str)->float:
    try:
        if '(' in v and '%' in v:
            return float(v.split('(')[1].rstrip('%)')) / 100.0
    except Exception: pass
    return 0.0

# =============================================================================
# Core Pipeline

def show_video(video_source):
    global _dashboard_instance
    root.withdraw()
    stop_flag=[False]
    root.protocol("WM_DELETE_WINDOW", lambda: stop_flag.__setitem__(0,True))

    if video_source==0:
        cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
        if not cap.isOpened(): cap=cv2.VideoCapture(0,cv2.CAP_MSMF)
        if not cap.isOpened(): cap=cv2.VideoCapture(0)
    else:
        cap=cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"[ERROR] Could not open {video_source}"); root.deiconify(); return

    if video_source==0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,2)

    track_buffers={}; track_pose_buffers={}; track_status={}; track_scores={}
    track_pose_prev={}; track_rule_flag={}; track_rule_history={}
    track_last_valid_pose={}; track_center_history={}; track_first_seen={}
    track_last_moved={}; track_bbox_history={}; track_wrist_history={}
    track_last_lm={}; track_active_flags={}; flag_first_seen={}

    alert_sent=False; normal_frames=0; flash_frames=0; event_count=0
    fps_history=deque(maxlen=30); heatmap_acc=None; heatmap_on=[True]
    frame_count=0; start_time=time.time(); prev_time=start_time
    executor=ThreadPoolExecutor(max_workers=3); processing_tracks=set()

    def evaluate_intent_async(t_id,X_seq,X_pose_seq):
        try:
            preds=lstm_model({"image_input":X_seq,"pose_input":X_pose_seq},training=False).numpy()[0]
            if t_id not in track_scores: track_scores[t_id]=deque(maxlen=3)
            track_scores[t_id].append(preds)
            avg=np.mean(track_scores[t_id],axis=0); conf=float(np.max(avg))
            track_status[t_id]=(f"SUSPICIOUS ({avg[1]*100:.0f}%)" if np.argmax(avg)==1 and conf>0.65
                                 else f"Normal ({avg[0]*100:.0f}%)")
        except Exception as e: print(f"[LSTM ERROR]: {e}")
        finally: processing_tracks.discard(t_id)

    font=cv2.FONT_HERSHEY_SIMPLEX; btn_coords=[0,0,0,0]
    def mouse_cb(event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONDOWN:
            if btn_coords[0]<=x<=btn_coords[2] and btn_coords[1]<=y<=btn_coords[3]:
                stop_flag[0]=True

    win_name='Suspicious Activity Detection'
    cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name,1560,720)
    cv2.setMouseCallback(win_name,mouse_cb)

    FLAG_COLORS={"WEAPON":(40,40,220),"FALL":(40,40,220),"INTENT":(40,100,220),
                 "RAISED ARMS":(40,120,200),"LUNGE":(40,120,200),"STRIKING":(40,120,200),
                 "AGGRESSIVE":(40,120,200),"RAPID MOVE":(30,160,220),"FLAILING":(30,140,200),
                 "PERSON DOWN":(30,80,200),"LOITERING":(60,110,160),"CONFLICT":(60,100,200)}
    def flag_color(f):
        for key,col in FLAG_COLORS.items():
            if key in f.upper(): return col
        return (60,60,80)

    try:
        while True:
            ret,frame=cap.read()
            if not ret: break

            curr_time=time.time()
            fps_history.append(1/max(curr_time-prev_time,0.0001))
            fps=sum(fps_history)/len(fps_history); prev_time=curr_time
            frame_count+=1; annotated_frame=frame.copy()

            if heatmap_acc is None:
                heatmap_acc=np.zeros((frame.shape[0],frame.shape[1]),dtype=np.float32)

            if yolo_model is None:
                cv2.putText(annotated_frame,"YOLO unavailable",(10,50),font,1,(0,0,255),2)
                cv2.imshow(win_name,annotated_frame)
                if cv2.waitKey(1)&0xFF in (27,ord('q')) or stop_flag[0]: break
                continue

            # Performance optimization: run YOLO every 2nd frame
            if frame_count % 2 == 0:
                results=yolo_model.track(frame,persist=True,classes=[0]+WEAPON_CLASSES,verbose=False)
            
            weapon_boxes=[]; person_boxes=[]; person_ids=[]

            if results and results[0].boxes is not None:
                for box_t,cls_t in zip(results[0].boxes.xyxy.cpu().numpy(),
                                        results[0].boxes.cls.int().cpu().tolist()):
                    if cls_t==0: person_boxes.append(box_t)
                    elif cls_t in WEAPON_CLASSES:
                        weapon_boxes.append(box_t)
                        wx1,wy1,wx2,wy2=map(int,box_t)
                        cv2.rectangle(annotated_frame,(wx1,wy1),(wx2,wy2),(0,0,255),3)
                        cv2.putText(annotated_frame,"WEAPON",(wx1,wy1-10),font,0.8,(0,0,255),2)

                if results[0].boxes.id is not None:
                    person_ids=[tid for tid,cls in zip(
                        results[0].boxes.id.int().cpu().tolist(),
                        results[0].boxes.cls.int().cpu().tolist()) if cls==0]

                    for box,track_id in zip(person_boxes,person_ids):
                        x1,y1,x2,y2=map(int,box)
                        x1,y1=max(0,x1),max(0,y1)
                        x2,y2=min(frame.shape[1],x2),min(frame.shape[0],y2)
                        crop=frame[y1:y2,x1:x2]
                        if crop.size==0: continue

                        if frame_count%2==0 or track_id not in track_last_valid_pose:
                            raw_flag,skel_crop,pose_vec,current_lm=run_pose_rules(
                                crop,track_id,track_pose_prev,track_last_valid_pose,
                                (x1,y1,x2,y2),frame.shape)
                            track_rule_flag[track_id]=raw_flag
                            track_last_lm[track_id]=current_lm
                        else:
                            raw_flag=track_rule_flag.get(track_id); skel_crop=crop
                            pose_vec=track_last_valid_pose.get(track_id,np.zeros(POSE_VECTOR_SIZE,dtype='float32'))
                            current_lm=track_last_lm.get(track_id)

                        try: annotated_frame[y1:y2,x1:x2]=cv2.resize(skel_crop,(x2-x1,y2-y1))
                        except Exception: pass

                        if track_id not in track_rule_history:
                            track_rule_history[track_id]=deque(maxlen=RULE_CONFIRM_N)
                        track_rule_history[track_id].append(raw_flag)
                        hist=track_rule_history[track_id]
                        if len(hist)==RULE_CONFIRM_N and all(f==hist[0] and f is not None for f in hist):
                            track_rule_flag[track_id]=hist[0]
                        elif raw_flag is None: track_rule_flag[track_id]=None

                        if has_lstm:
                            try:
                                arr=preprocess_input(np.array(cv2.cvtColor(
                                    cv2.resize(crop,(IMAGE_WIDTH,IMAGE_HEIGHT)),
                                    cv2.COLOR_BGR2RGB),dtype='float32'))
                                if track_id not in track_buffers:
                                    track_buffers[track_id]=deque(maxlen=SEQUENCE_LENGTH)
                                    track_pose_buffers[track_id]=deque(maxlen=SEQUENCE_LENGTH)
                                    track_status[track_id]="Tracking..."
                                track_buffers[track_id].append(arr)
                                track_pose_buffers[track_id].append(pose_vec)
                                if (len(track_buffers[track_id])==SEQUENCE_LENGTH and
                                        (frame_count+track_id)%3==0 and
                                        track_id not in processing_tracks):
                                    processing_tracks.add(track_id)
                                    X_img=np.expand_dims(np.array(track_buffers[track_id]),0)
                                    X_pose=np.expand_dims(np.array(track_pose_buffers[track_id]),0)
                                    executor.submit(evaluate_intent_async,track_id,X_img,X_pose)
                            except Exception as e: print(f"[LSTM ERROR]: {e}")

                        cx,cy=get_center(x1,y1,x2,y2,frame.shape[1],frame.shape[0])
                        vel_flag=check_velocity(track_id,cx,cy,curr_time,track_center_history)
                        loiter_flag=check_loitering(track_id,cx,cy,curr_time,
                                                     track_first_seen,track_last_moved,track_center_history)
                        weapon_flag=weapon_near_person(x1,y1,x2,y2,weapon_boxes)
                        fall_flag=check_fall(track_id,x1,y1,x2,y2,curr_time,track_bbox_history)
                        prone_flag=check_prone(track_id,x1,y1,x2,y2)
                        flail_flag=check_arm_flail(track_id,current_lm,track_wrist_history)

                        lstm_verdict=track_status.get(track_id,"Tracking...")
                        rule_flag=track_rule_flag.get(track_id)
                        lstm_conf=parse_lstm_confidence(lstm_verdict)
                        time_mult=get_time_multiplier()
                        is_lstm_sus="SUSPICIOUS" in lstm_verdict and lstm_conf>(0.70/time_mult)
                        is_rule_sus=rule_flag is not None

                        active_flags=[]
                        if weapon_flag  and _toggle_vars["WEAPON"].get():      active_flags.append("WEAPON")
                        if is_lstm_sus  and _toggle_vars["INTENT"].get():      active_flags.append(f"INTENT {lstm_conf*100:.0f}%")
                        if is_rule_sus  and _toggle_vars["POSE RULES"].get():  active_flags.append(rule_flag)
                        if fall_flag    and _toggle_vars["FALL"].get():        active_flags.append("FALL")
                        if prone_flag   and _toggle_vars["PERSON DOWN"].get(): active_flags.append("PERSON DOWN")
                        if vel_flag     and _toggle_vars["RAPID MOVE"].get():  active_flags.append("RAPID MOVE")
                        if flail_flag   and _toggle_vars["FLAILING"].get():    active_flags.append("FLAILING")
                        if loiter_flag  and _toggle_vars["LOITERING"].get():   active_flags.append(loiter_flag)

                        if track_id not in flag_first_seen: flag_first_seen[track_id]={}
                        for f in active_flags: flag_first_seen[track_id].setdefault(f,curr_time)
                        for f in [k for k in flag_first_seen[track_id] if k not in active_flags]:
                            del flag_first_seen[track_id][f]

                        if weapon_flag or fall_flag: box_color=(0,0,255)
                        elif is_lstm_sus or is_rule_sus: box_color=(0,80,255)
                        elif vel_flag or flail_flag or prone_flag: box_color=(0,140,255)
                        elif loiter_flag: box_color=(30,100,180)
                        else: box_color=(0,200,0)

                        draw_color=box_color
                        if flash_frames>0 and (flash_frames%6)<3: draw_color=(255,255,255)
                        short_label=(f"ID:{track_id} | {active_flags[0]}" if active_flags
                                     else f"ID:{track_id}  Normal")
                        cv2.rectangle(annotated_frame,(x1,y1-28),(x1+260,y1),draw_color,-1)
                        cv2.putText(annotated_frame,short_label,(x1+4,y1-8),font,0.55,(255,255,255),1)
                        track_active_flags[track_id]=active_flags

                        is_alert=(weapon_flag or fall_flag or
                                  (is_lstm_sus and lstm_conf>LSTM_CONFIDENCE_THRESHOLD))
                        if is_alert:
                            if heatmap_acc is not None:
                                cv2.circle(heatmap_acc,(int((x1+x2)/2),int((y1+y2)/2)),60,1.0,-1)
                            if not alert_sent:
                                alert_sent=True; flash_frames=40; event_count+=1
                                alert_label=" | ".join(active_flags) or "SUSPICIOUS"
                                conf_val=max(lstm_conf,0.85)
                                log_event(alert_label,conf_val)
                                record_session_event(alert_label,conf_val)
                                snap_path=save_snapshot(annotated_frame,alert_label)
                                notifier.send_alert(activity_type=alert_label,
                                                    confidence=conf_val,
                                                    snapshot_path=snap_path)
                            cv2.putText(annotated_frame,"! THREAT DETECTED !",
                                        (30,frame.shape[0]-40),font,1.6,(0,0,255),4)
                            normal_frames=0
                        else:
                            normal_frames+=1
                            if normal_frames>90: alert_sent=False

            if flash_frames>0: flash_frames-=1

            conflict_ids=check_proximity(person_boxes,person_ids,frame.shape[1],frame.shape[0])
            if conflict_ids and _toggle_vars["PROXIMITY"].get():
                cv2.putText(annotated_frame,f"CONFLICT ZONE: {len(conflict_ids)} PERSONS",
                            (30,frame.shape[0]-80),font,1.0,(0,100,255),2)

            # Heatmap overlay
            if heatmap_on[0] and heatmap_acc is not None and heatmap_acc.max()>0:
                norm=cv2.normalize(heatmap_acc,None,0,255,cv2.NORM_MINMAX)
                blur=cv2.GaussianBlur(norm.astype(np.uint8),(51,51),0)
                cmap=cv2.applyColorMap(blur,cv2.COLORMAP_JET)
                mask=blur>10
                annotated_frame[mask]=cv2.addWeighted(annotated_frame,0.55,cmap,0.45,0)[mask]

            # Scale to 720p
            DISPLAY_HEIGHT=720; PANEL_W=290
            orig_h,orig_w=annotated_frame.shape[:2]
            scale=DISPLAY_HEIGHT/max(orig_h,1); scaled_w=int(orig_w*scale)
            video_frame=cv2.resize(annotated_frame,(scaled_w,DISPLAY_HEIGHT),interpolation=cv2.INTER_LINEAR)

            # Side panel
            panel=np.zeros((DISPLAY_HEIGHT,PANEL_W,3),dtype=np.uint8); panel[:]=(18,18,24)
            cv2.rectangle(panel,(0,0),(PANEL_W,52),(10,10,18),-1)
            cv2.putText(panel,"ACTIVE TRACKS",(10,18),font,0.45,(100,180,255),1)
            cv2.putText(panel,f"{len(track_active_flags)} person(s) in frame",
                        (10,38),font,0.38,(120,120,140),1)
            cv2.line(panel,(0,52),(PANEL_W,52),(40,40,55),1)

            ROW_H=26; CARD_PAD=6; card_top=60
            for tid in [t for t in list(track_active_flags) if t not in set(person_ids)]:
                del track_active_flags[tid]

            for tid in sorted(track_active_flags.keys()):
                flags=track_active_flags[tid]; lstm_v=track_status.get(tid,"")
                conf=parse_lstm_confidence(lstm_v)
                card_h=18+14+max(len(flags),1)*ROW_H+8
                if card_top+card_h>DISPLAY_HEIGHT-42: break
                has_crit=any(k in " ".join(flags).upper() for k in ("WEAPON","FALL","INTENT"))
                card_bg=(30,18,18) if has_crit else (22,24,32)
                cv2.rectangle(panel,(6,card_top),(PANEL_W-6,card_top+card_h),card_bg,-1)
                accent=(0,0,200) if has_crit else (50,100,180)
                cv2.rectangle(panel,(6,card_top),(10,card_top+card_h),accent,-1)
                cv2.putText(panel,f"ID: {tid}",(14,card_top+14),font,0.46,(220,220,240),1)
                bx,by=14,card_top+20; bw,bh_px=PANEL_W-28,8
                cv2.rectangle(panel,(bx,by),(bx+bw,by+bh_px),(40,40,50),-1)
                fill=int(bw*conf)
                if fill>0:
                    bc=(0,0,200) if conf>0.70 else (0,160,200) if conf>0.40 else (0,180,80)
                    cv2.rectangle(panel,(bx,by),(bx+fill,by+bh_px),bc,-1)
                cv2.putText(panel,f"{conf*100:.0f}%",(bx+bw+3,by+7),font,0.3,(160,160,180),1)
                if not flags:
                    cv2.putText(panel,"Normal",(14,card_top+20+ROW_H),font,0.38,(60,180,60),1)
                else:
                    for fi,ft in enumerate(flags):
                        fy=card_top+20+14+(fi+1)*ROW_H; pc=flag_color(ft)
                        px2=min(14+len(ft)*7+10,PANEL_W-36)
                        cv2.rectangle(panel,(14,fy-12),(px2,fy+4),pc,-1)
                        cv2.putText(panel,ft,(18,fy),font,0.37,(255,255,255),1)
                        age=curr_time-flag_first_seen.get(tid,{}).get(ft,curr_time)
                        if age>=1: cv2.putText(panel,f"{int(age)}s",(px2+4,fy),font,0.32,(140,140,160),1)
                cv2.line(panel,(10,card_top+card_h),(PANEL_W-10,card_top+card_h),(38,38,50),1)
                card_top+=card_h+CARD_PAD

            footer_y=DISPLAY_HEIGHT-30
            cv2.rectangle(panel,(0,footer_y-14),(PANEL_W,DISPLAY_HEIGHT),(10,10,18),-1)
            cv2.line(panel,(0,footer_y-14),(PANEL_W,footer_y-14),(40,40,55),1)
            ev_col=(80,80,200) if event_count==0 else (80,80,255)
            cv2.putText(panel,f"Events: {event_count}  H=heatmap  C=clear",
                        (10,footer_y+4),font,0.30,ev_col,1)
            snap_count=len([f for f in os.listdir(SNAPSHOT_DIR) if f.endswith('.jpg')])
            cv2.putText(panel,f"Snapshots: {snap_count}",(10,footer_y+18),font,0.30,(80,80,100),1)

            # Compose
            display_frame=np.hstack([video_frame,panel])
            h_f,w_f=display_frame.shape[:2]
            overlay=display_frame.copy()
            cv2.rectangle(overlay,(0,0),(scaled_w,78),(0,0,0),-1)
            display_frame=cv2.addWeighted(overlay,0.6,display_frame,0.4,0)
            elapsed=int(time.time()-start_time)
            cv2.putText(display_frame,"HYBRID AI  |  YOLO + Pose + LSTM",
                        (10,30),font,0.7,(0,255,255),2)
            cv2.putText(display_frame,
                        f"FPS: {fps:.1f}  |  {elapsed}s elapsed  |  Tracks: {len(track_active_flags)}",
                        (10,62),font,0.5,(170,170,170),1)
            _bx1=scaled_w-120; _by1=14; _bx2=scaled_w-10; _by2=56
            btn_coords[0],btn_coords[1]=_bx1,_by1; btn_coords[2],btn_coords[3]=_bx2,_by2
            cv2.rectangle(display_frame,(_bx1,_by1),(_bx2,_by2),(0,0,160),-1)
            cv2.rectangle(display_frame,(_bx1,_by1),(_bx2,_by2),(255,255,255),1)
            cv2.putText(display_frame,'STOP',(_bx1+18,_by2-12),font,0.8,(255,255,255),2)

            cv2.imshow(win_name,display_frame)
            key=cv2.waitKey(1)&0xFF
            if key==27 or key==ord('q') or stop_flag[0]: break
            elif key in (ord('c'),ord('C')):
                alert_sent=False; flash_frames=0; normal_frames=0
                track_active_flags.clear(); flag_first_seen.clear()
            elif key in (ord('h'),ord('H')):
                heatmap_on[0]=not heatmap_on[0]
                print(f"[INFO] Heatmap {'ON' if heatmap_on[0] else 'OFF'}")
            elif key in (ord('r'),ord('R')):
                if heatmap_acc is not None: heatmap_acc[:]=0
                print("[INFO] Heatmap reset.")

    finally:
        cap.release(); executor.shutdown(wait=False); cv2.destroyAllWindows()
        try:
            root.protocol("WM_DELETE_WINDOW",root.destroy)
            root.deiconify()
        except Exception: pass

# =============================================================================
# GUI

_toggle_vars={}

def open_settings():
    win=tk.Toplevel(root); win.title("Detection Settings")
    win.configure(bg="#0d1117"); win.resizable(False,False)
    win.attributes("-topmost",True); win.geometry("440x620")

    tab_bar=tk.Frame(win,bg="#161b22"); tab_bar.pack(fill="x")
    content=tk.Frame(win,bg="#0d1117"); content.pack(fill="both",expand=True)
    det_f=tk.Frame(content,bg="#0d1117"); email_f=tk.Frame(content,bg="#0d1117")
    frames={"detection":det_f,"email":email_f}

    def show_tab(name):
        for f in frames.values(): f.pack_forget()
        frames[name].pack(fill="both",expand=True,padx=16,pady=8)

    for lbl,key in [("Detection Toggles","detection"),("Email Alerts","email")]:
        tk.Button(tab_bar,text=lbl,command=lambda k=key:show_tab(k),
                  font=("Helvetica",10,"bold"),bg="#1f6feb",fg="white",
                  relief="flat",activebackground="#388bfd",cursor="hand2",
                  padx=14,pady=6).pack(side="left",padx=2,pady=4)

    tk.Label(det_f,text="Detection Toggles",font=("Helvetica",13,"bold"),
             bg="#0d1117",fg="white").pack(pady=(10,4))
    tk.Label(det_f,text="Uncheck to disable a detection type",
             font=("Helvetica",9),bg="#0d1117",fg="#8b949e").pack(pady=(0,8))
    LABELS={"WEAPON":"Weapon detection","INTENT":"LSTM suspicious intent",
            "POSE RULES":"Pose rules (raised arms, lunge…)","FALL":"Fall detection",
            "PERSON DOWN":"Person down (prone)","RAPID MOVE":"Rapid movement",
            "FLAILING":"Arm flailing","LOITERING":"Loitering","PROXIMITY":"Proximity / conflict zone"}
    for key,label in LABELS.items():
        tk.Checkbutton(det_f,text=label,variable=_toggle_vars[key],
                       font=("Helvetica",11),bg="#0d1117",fg="white",
                       selectcolor="#1f6feb",activebackground="#0d1117",
                       activeforeground="white",anchor="w",
                       command=lambda:save_settings(_toggle_vars)).pack(fill="x",pady=2)

    tk.Label(email_f,text="Email Alert Configuration",font=("Helvetica",13,"bold"),
             bg="#0d1117",fg="white").pack(pady=(10,2))
    tk.Label(email_f,text="Use a Gmail App Password.\nmyaccount.google.com/apppasswords",
             font=("Helvetica",9),bg="#0d1117",fg="#8b949e",justify="left").pack(pady=(0,8),anchor="w")
    cfg=notifier.load_config(); flds={}
    def field(lbl,key,show=""):
        tk.Label(email_f,text=lbl,font=("Helvetica",10),bg="#0d1117",fg="#c9d1d9",anchor="w").pack(fill="x")
        e=tk.Entry(email_f,font=("Helvetica",11),bg="#161b22",fg="white",
                   insertbackground="white",relief="flat",show=show)
        e.insert(0,cfg.get(key,"")); e.pack(fill="x",pady=(0,6),ipady=4); flds[key]=e
    field("Sender Gmail address","sender_email")
    field("App Password","sender_password",show="*")
    field("Recipient email","recipient_email")
    enabled_var=tk.BooleanVar(value=cfg.get("enabled",False))
    tk.Checkbutton(email_f,text="Enable email alerts",variable=enabled_var,
                   font=("Helvetica",11),bg="#0d1117",fg="white",
                   selectcolor="#1f6feb",activebackground="#0d1117",
                   activeforeground="white").pack(anchor="w",pady=4)
    def save_email():
        nc=notifier.load_config()
        nc["sender_email"]=flds["sender_email"].get().strip()
        nc["sender_password"]=flds["sender_password"].get().strip()
        nc["recipient_email"]=flds["recipient_email"].get().strip()
        nc["enabled"]=enabled_var.get()
        notifier.save_config(nc); messagebox.showinfo("Saved","Email settings saved.",parent=win)
    def test_email():
        save_email(); notifier.send_alert("TEST ALERT",0.99,None)
        messagebox.showinfo("Sent","Test email dispatched. Check inbox in ~10s.",parent=win)
    br=tk.Frame(email_f,bg="#0d1117"); br.pack(pady=8)
    tk.Button(br,text="Save",command=save_email,width=12,
              font=("Helvetica",11,"bold"),bg="#238636",fg="white",
              relief="flat",cursor="hand2").pack(side="left",padx=4)
    tk.Button(br,text="Send Test Email",command=test_email,width=16,
              font=("Helvetica",11),bg="#6e40c9",fg="white",
              relief="flat",cursor="hand2").pack(side="left",padx=4)

    show_tab("detection")
    tk.Button(win,text="Close",command=win.destroy,width=14,
              font=("Helvetica",11),bg="#21262d",fg="#8b949e",
              relief="flat",cursor="hand2").pack(pady=10)


def open_dashboard():
    global _dashboard_instance
    if _dashboard_instance is not None:
        try: _dashboard_instance.win.lift(); return
        except Exception: pass
    _dashboard_instance=LiveDashboard(root)

def upload_video():
    f=askopenfilename(initialdir=SCRIPT_DIR,title='Select video',
                      filetypes=[("Video files","*.mp4 *.avi *.mkv *.mov"),("All","*.*")])
    if f: show_video(f)

def use_webcam(): show_video(0)

def use_cctv():
    url=simpledialog.askstring("CCTV Stream",
        "Enter RTSP URL:\n(e.g. rtsp://admin:pass@192.168.1.64:554/stream1)",parent=root)
    if url and url.strip(): show_video(url.strip())

def open_snapshots():
    import subprocess,platform
    if platform.system()=="Windows": os.startfile(SNAPSHOT_DIR)
    elif platform.system()=="Darwin": subprocess.Popen(["open",SNAPSHOT_DIR])
    else: subprocess.Popen(["xdg-open",SNAPSHOT_DIR])

def do_exit():
    try: root.destroy()
    except Exception: pass

# Layout
btn_frame=tk.Frame(root,bg="#0d1117"); btn_frame.pack(pady=30)
BS=dict(width=28,font=("Helvetica",16,"bold"),bg="#1f6feb",fg="white",
        relief="flat",activebackground="#388bfd",cursor="hand2")
tk.Button(btn_frame,text="  Upload Video",     command=upload_video, **BS).pack(pady=8)
tk.Button(btn_frame,text="  Use Webcam",       command=use_webcam,   **BS).pack(pady=8)
tk.Button(btn_frame,text="  Connect CCTV Stream",command=use_cctv,
          width=28,font=("Helvetica",16,"bold"),bg="#6e40c9",fg="white",
          relief="flat",activebackground="#8957e5",cursor="hand2").pack(pady=8)
tk.Button(btn_frame,text="  Live Dashboard",   command=open_dashboard,
          width=28,font=("Helvetica",14,"bold"),bg="#0e6e40",fg="white",
          relief="flat",activebackground="#16a34a",cursor="hand2").pack(pady=8)
tk.Button(btn_frame,text="  Detection Settings",command=open_settings,
          width=28,font=("Helvetica",14,"bold"),bg="#238636",fg="white",
          relief="flat",activebackground="#2ea043",cursor="hand2").pack(pady=8)
tk.Button(btn_frame,text="  View Snapshots",   command=open_snapshots,
          width=28,font=("Helvetica",14),bg="#30363d",fg="#c9d1d9",
          relief="flat",activebackground="#444c56",cursor="hand2").pack(pady=8)
tk.Button(btn_frame,text="Exit",command=do_exit,width=28,
          font=("Helvetica",14),bg="#21262d",fg="#8b949e",
          relief="flat",cursor="hand2").pack(pady=8)

saved_settings=load_settings()
for k,v in DETECTION_TOGGLES.items():
    _toggle_vars[k]=tk.BooleanVar(value=saved_settings.get(k,v))

if len(sys.argv)>1:
    arg=sys.argv[1]
    show_video(0 if arg=="0" else (arg if os.path.isabs(arg) else os.path.join(SCRIPT_DIR,arg)))
    try: root.destroy()
    except Exception: pass
else:
    root.mainloop() 