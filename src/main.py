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
from concurrent.futures import ThreadPoolExecutor

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# MediaPipe Tasks API (compatible with protobuf 7.x / TF 2.21)
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
MODEL_PATH  = os.path.join(SCRIPT_DIR, 'mobilenet_model.h5')
POSE_MODEL  = os.path.join(SCRIPT_DIR, 'pose_landmarker_lite.task')
IMAGE_HEIGHT, IMAGE_WIDTH = 128, 128
SEQUENCE_LENGTH = 12

# Pose rule thresholds (tuned to reduce false positives)
RAISED_ARM_THRESH = 0.22   # wrist must be well above shoulder (was 0.10)
LUNGE_THRESH      = 0.28   # dramatic hip shift only (was 0.18)
STRIKE_THRESH     = 0.10   # knee clearly above hip (was 0.05)
LEAN_THRESH       = 0.30   # aggressive lean only (was 0.22)
RULE_CONFIRM_N    = 3      # rule must fire this many consecutive checks to display

# Landmark indices
IDX_NOSE       = 0
IDX_L_SHOULDER = 11; IDX_R_SHOULDER = 12
IDX_L_WRIST    = 15; IDX_R_WRIST    = 16
IDX_L_HIP      = 23; IDX_R_HIP      = 24
IDX_L_KNEE     = 25; IDX_R_KNEE     = 26

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

# Initialize MediaPipe PoseLandmarker (Tasks API)
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
    print("          Run: python src/setup_models.py  to download it.")

# =============================================================================
# Frontend UI
root = tk.Tk()
root.state('zoomed')
title_text = "Suspicious Activity Detection  |  YOLOv8 + Pose + LSTM"
root.title(title_text)

bg_image_path = os.path.join(SCRIPT_DIR, "back5.png")
if os.path.exists(bg_image_path):
    img = Image.open(bg_image_path)
    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    bg = img.resize((w, h), Image.LANCZOS)
    bg_img = ImageTk.PhotoImage(bg)
    bg_lbl = tk.Label(root, image=bg_img)
    bg_lbl.place(x=0, y=0)
else:
    root.configure(bg="#192841")

heading = tk.Label(
    root, text="Suspicious Activity Detection", width=40,
    font=("Times New Roman", 38, 'bold'), bg="#192841", fg="white"
)
heading.pack(pady=40)

subhead = tk.Label(
    root, text="YOLOv8  ·  MediaPipe Pose  ·  MobileNetV2-LSTM",
    font=("Courier", 14), bg="#192841", fg="#00e5ff"
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


def run_pose_rules(crop_bgr, track_id, track_pose_prev):
    """
    Runs MediaPipe PoseLandmarker on the given person crop.
    Returns (rule_flag: str | None, annotated_crop: ndarray)
    """
    if pose_landmarker is None:
        return None, crop_bgr
    if crop_bgr.shape[0] < 48 or crop_bgr.shape[1] < 48:
        return None, crop_bgr

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)

    try:
        result = pose_landmarker.detect(mp_img)
    except Exception:
        return None, crop_bgr

    if not result.pose_landmarks:
        return None, crop_bgr

    lm = result.pose_landmarks[0]   # first (and only) person in the crop

    # Landmark indices
    nose      = lm[IDX_NOSE]
    l_sho     = lm[IDX_L_SHOULDER]; r_sho  = lm[IDX_R_SHOULDER]
    l_wrist   = lm[IDX_L_WRIST];    r_wrist = lm[IDX_R_WRIST]
    l_hip     = lm[IDX_L_HIP];      r_hip   = lm[IDX_R_HIP]
    l_knee    = lm[IDX_L_KNEE];     r_knee  = lm[IDX_R_KNEE]

    shoulder_y = (l_sho.y + r_sho.y) / 2
    hip_y      = (l_hip.y + r_hip.y) / 2
    hip_x      = (l_hip.x + r_hip.x) / 2

    raw_flag = None

    # Rule 1 — Raised arms (both wrists well above shoulder)
    both_raised = (l_wrist.y < shoulder_y - RAISED_ARM_THRESH and
                   r_wrist.y < shoulder_y - RAISED_ARM_THRESH)
    if both_raised:
        raw_flag = "RAISED ARMS"

    # Rule 2 — Striking / kicking posture
    if not raw_flag:
        if l_knee.y < hip_y - STRIKE_THRESH or r_knee.y < hip_y - STRIKE_THRESH:
            raw_flag = "STRIKING POSTURE"

    # Rule 3 — Aggressive forward lean
    if not raw_flag:
        if abs(nose.x - hip_x) > LEAN_THRESH:
            raw_flag = "AGGRESSIVE LEAN"

    # Rule 4 — Lunge (large lateral hip shift between frames)
    if not raw_flag and track_id in track_pose_prev:
        if abs(hip_x - track_pose_prev[track_id]) > LUNGE_THRESH:
            raw_flag = "LUNGE DETECTED"

    track_pose_prev[track_id] = hip_x

    annotated = _draw_skeleton(crop_bgr, lm)
    return raw_flag, annotated


# =============================================================================
# Core Pipeline

def show_video(video_source):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open {video_source}")
        return

    track_buffers       = {}
    track_status        = {}
    track_scores        = {}
    track_pose_prev     = {}
    track_rule_flag     = {}   # confirmed rule flag
    track_rule_history  = {}   # deque of recent raw flags for temporal smoothing

    alert_sent  = False
    prev_time   = time.time()
    frame_count = 0

    executor = ThreadPoolExecutor(max_workers=3)
    processing_tracks = set()

    def evaluate_intent_async(t_id, X_seq):
        nonlocal alert_sent
        try:
            preds = lstm_model(X_seq, training=False).numpy()[0]
            if t_id not in track_scores:
                track_scores[t_id] = deque(maxlen=3)
            track_scores[t_id].append(preds)
            avg  = np.mean(track_scores[t_id], axis=0)
            conf = np.max(avg)
            if np.argmax(avg) == 1 and conf > 0.65:
                track_status[t_id] = f"SUSPICIOUS ({avg[1]*100:.0f}%)"
                if not alert_sent:
                    try:
                        import notifier; notifier.send_alert()
                    except Exception:
                        pass
                    alert_sent = True
            else:
                track_status[t_id] = f"Normal ({avg[0]*100:.0f}%)"
        except Exception as e:
            print(f"[LSTM Thread ERROR]: {e}")
        finally:
            processing_tracks.discard(t_id)

    font = cv2.FONT_HERSHEY_SIMPLEX
    stop_flag = [False]

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            h_w, w_w = param
            if (w_w - 130) <= x <= (w_w - 10) and 12 <= y <= 60:
                stop_flag[0] = True

    win_name = 'Suspicious Activity Detection'
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, mouse_callback, (640, 480))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_time = time.time()
        fps = 1 / max(curr_time - prev_time, 0.0001)
        prev_time = curr_time
        frame_count += 1
        annotated_frame = frame.copy()

        if USE_YOLO_HYBRID and yolo_model:
            results = yolo_model.track(frame, persist=True, classes=[0], verbose=False)
            annotated_frame = results[0].plot()

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes     = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    # ── Pose Rule Engine (every 3rd frame) ───────────────
                    if frame_count % 3 == 0:
                        raw_flag, skel_crop = run_pose_rules(crop, track_id, track_pose_prev)

                        # Temporal confirmation — only confirm if rule fires N times in a row
                        if track_id not in track_rule_history:
                            track_rule_history[track_id] = deque(maxlen=RULE_CONFIRM_N)
                        track_rule_history[track_id].append(raw_flag)

                        hist = track_rule_history[track_id]
                        if (len(hist) == RULE_CONFIRM_N and
                                all(f == hist[0] and f is not None for f in hist)):
                            track_rule_flag[track_id] = hist[0]  # confirmed
                        elif raw_flag is None:
                            track_rule_flag[track_id] = None      # reset on normal frame

                        try:
                            if skel_crop.shape == crop.shape:
                                annotated_frame[y1:y2, x1:x2] = skel_crop
                        except Exception:
                            pass

                    # ── LSTM Buffer & Async Inference ─────────────────────
                    if has_lstm:
                        try:
                            arr = preprocess_input(
                                np.array(cv2.cvtColor(
                                    cv2.resize(crop, (IMAGE_WIDTH, IMAGE_HEIGHT)),
                                    cv2.COLOR_BGR2RGB), dtype='float32'))
                            if track_id not in track_buffers:
                                track_buffers[track_id] = deque(maxlen=SEQUENCE_LENGTH)
                                track_status[track_id]  = "Tracking..."
                            track_buffers[track_id].append(arr)

                            if (len(track_buffers[track_id]) == SEQUENCE_LENGTH and
                                    (frame_count + track_id) % 5 == 0 and
                                    track_id not in processing_tracks):
                                processing_tracks.add(track_id)
                                X = np.expand_dims(np.array(track_buffers[track_id]), axis=0)
                                executor.submit(evaluate_intent_async, track_id, X)
                        except Exception as e:
                            print(f"[LSTM ERROR]: {e}")

                    # ── Combined Verdict ──────────────────────────────────
                    lstm_verdict = track_status.get(track_id, "Tracking...")
                    rule_flag    = track_rule_flag.get(track_id)  # confirmed flag only
                    lstm_conf    = float(lstm_verdict.split('(')[1].rstrip('%)')) / 100.0 if '(' in lstm_verdict else 0.0
                    is_lstm_sus  = "SUSPICIOUS" in lstm_verdict and lstm_conf > 0.75  # raised bar
                    is_rule_sus  = rule_flag is not None

                    if is_lstm_sus and is_rule_sus:
                        display_text = f"SUSPICIOUS | {rule_flag}"
                        box_color    = (0, 0, 230)    # Red
                    elif is_lstm_sus:
                        display_text = lstm_verdict
                        box_color    = (0, 50, 220)   # Red-orange
                    elif is_rule_sus:
                        display_text = f"ALERT: {rule_flag}"
                        box_color    = (0, 120, 255)  # Orange
                    else:
                        display_text = lstm_verdict
                        box_color    = (0, 190, 60)   # Green

                    banner_w = min(420, frame.shape[1] - x1)
                    cv2.rectangle(annotated_frame, (x1, y1 - 38), (x1 + banner_w, y1), box_color, -1)
                    cv2.putText(annotated_frame,
                                f"ID:{track_id} | {display_text}",
                                (x1 + 5, y1 - 10), font, 0.56, (255, 255, 255), 2)

        # ── Scale to 720p ───────────────────────────────────────────────
        DISPLAY_HEIGHT = 720
        orig_h, orig_w = annotated_frame.shape[:2]
        scale = DISPLAY_HEIGHT / orig_h
        display_frame = cv2.resize(annotated_frame,
                                   (int(orig_w * scale), DISPLAY_HEIGHT),
                                   interpolation=cv2.INTER_LINEAR)

        # ── HUD ─────────────────────────────────────────────────────────
        h_f, w_f = display_frame.shape[:2]
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w_f, 90), (0, 0, 0), -1)
        display_frame = cv2.addWeighted(overlay, 0.55, display_frame, 0.45, 0)
        cv2.putText(display_frame, "HYBRID  |  YOLO + Pose + LSTM",
                    (10, 45), font, 1.1, (0, 255, 255), 2)
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 80), font, 0.8, (255, 255, 255), 2)

        # ── STOP button ──────────────────────────────────────────────────
        bx1, by1, bx2, by2 = w_f - 130, 12, w_f - 10, 60
        cv2.rectangle(display_frame, (bx1, by1), (bx2, by2), (0, 0, 180), -1)
        cv2.rectangle(display_frame, (bx1, by1), (bx2, by2), (255, 255, 255), 2)
        cv2.putText(display_frame, 'STOP', (bx1 + 14, by2 - 12), font, 1.0, (255, 255, 255), 2)

        cv2.setMouseCallback(win_name, mouse_callback, (h_f, w_f))
        cv2.imshow(win_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q') or stop_flag[0]:
            print("[INFO] Detection stopped.")
            break

    cap.release()
    executor.shutdown(wait=False)
    cv2.destroyAllWindows()


# =============================================================================
# GUI Buttons
def upload_video():
    f = askopenfilename(initialdir=SCRIPT_DIR, title='Select video',
                        filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov"), ("All", "*.*")])
    if f:
        show_video(f)

def use_webcam():
    show_video(0)

btn_frame = tk.Frame(root, bg="#192841")
btn_frame.pack(pady=30)

tk.Button(btn_frame, command=upload_video, text="Upload Video",
          width=25, font=("Times new roman", 25, "bold"), bg="cyan", fg="black").pack(pady=15)
tk.Button(btn_frame, command=use_webcam, text="Use Webcam",
          width=25, font=("Times new roman", 25, "bold"), bg="orange", fg="black").pack(pady=15)
tk.Button(btn_frame, command=root.destroy, text="Exit",
          width=25, font=("Times new roman", 25, "bold"), bg="red", fg="white").pack(pady=15)

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