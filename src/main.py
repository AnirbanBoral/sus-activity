import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF C++ warnings

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

# MediaPipe Pose
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    HAS_MEDIAPIPE = True
except ImportError:
    print("[WARNING] MediaPipe not found. Run 'pip install mediapipe'. Pose rules disabled.")
    HAS_MEDIAPIPE = False

# YOLOv8 imports
try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] Ultralytics not found. Run 'pip install ultralytics'.")
    YOLO = None

# =============================================================================
# Configuration
USE_YOLO_HYBRID = True

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'mobilenet_model.h5')
IMAGE_HEIGHT, IMAGE_WIDTH = 128, 128
SEQUENCE_LENGTH = 12

# Pose rule-engine thresholds (normalized 0–1, relative to person crop)
RAISED_ARM_THRESH   = 0.10   # wrist above shoulder by this much → raised arms
LUNGE_THRESH        = 0.18   # hip x-shift between frames → lunge
STRIKE_THRESH       = 0.05   # knee above hip → striking posture
LEAN_THRESH         = 0.22   # nose x ahead of hip → aggressive lean

# =============================================================================
# Load Global Models
print("[INFO] Initializing Spatio-Temporal Intent (LSTM) Model...")
try:
    lstm_model = load_model(MODEL_PATH)
    has_lstm = True
    print("[OK] LSTM Loaded.")
except Exception as e:
    print(f"[WARNING] Failed to load LSTM: {e}")
    lstm_model = None
    has_lstm = False

print("[INFO] Initializing YOLO Spatial Tracker...")
yolo_model = YOLO("yolov8s.pt") if YOLO else None

# =============================================================================
# Frontend UI
root = tk.Tk()
root.state('zoomed')
title_text = "Suspicious Activity Detection (YOLOv8 + Pose + LSTM)"
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
    root, text=title_text, width=50,
    font=("Times New Roman", 36, 'bold'), bg="#192841", fg="white"
)
heading.pack(pady=50)


# =============================================================================
# Pose Rule Engine

def run_pose_rules(crop_bgr, track_id, track_pose_prev):
    """
    Runs MediaPipe Pose on the given person crop and applies geometric rules.
    Returns (rule_flag: str | None, skeleton_annotated_crop: ndarray)
    """
    if not HAS_MEDIAPIPE:
        return None, crop_bgr

    h, w = crop_bgr.shape[:2]
    if h < 32 or w < 32:
        return None, crop_bgr

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,          # Lite model — fastest on CPU
        smooth_landmarks=True,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4
    ) as pose:
        result = pose.process(crop_rgb)

    rule_flag = None

    if result.pose_landmarks:
        lm = result.pose_landmarks.landmark

        # --- Key joint indices ---
        # 0=nose, 11=L_shoulder, 12=R_shoulder
        # 15=L_wrist, 16=R_wrist
        # 23=L_hip, 24=R_hip
        # 25=L_knee, 26=R_knee
        nose      = lm[0]
        l_sho     = lm[11];  r_sho = lm[12]
        l_wrist   = lm[15];  r_wrist = lm[16]
        l_hip     = lm[23];  r_hip   = lm[24]
        l_knee    = lm[25];  r_knee  = lm[26]

        shoulder_y = (l_sho.y + r_sho.y) / 2
        hip_y      = (l_hip.y + r_hip.y) / 2
        hip_x      = (l_hip.x + r_hip.x) / 2

        # Rule 1 — Raised arms (wrist significantly above shoulder)
        if (l_wrist.y < shoulder_y - RAISED_ARM_THRESH or
                r_wrist.y < shoulder_y - RAISED_ARM_THRESH):
            rule_flag = "RAISED ARMS"

        # Rule 2 — Striking posture (knee above hip level)
        if not rule_flag:
            if (l_knee.y < hip_y - STRIKE_THRESH or
                    r_knee.y < hip_y - STRIKE_THRESH):
                rule_flag = "STRIKING POSTURE"

        # Rule 3 — Aggressive lean (nose far ahead of hips laterally)
        if not rule_flag:
            if abs(nose.x - hip_x) > LEAN_THRESH:
                rule_flag = "AGGRESSIVE LEAN"

        # Rule 4 — Lunge (hip shifts sharply between frames)
        if not rule_flag and track_id in track_pose_prev:
            prev_hip_x = track_pose_prev[track_id]
            if abs(hip_x - prev_hip_x) > LUNGE_THRESH:
                rule_flag = "LUNGE DETECTED"

        # Update hip position for next frame
        track_pose_prev[track_id] = hip_x

        # Draw skeleton on the crop for visual debugging
        annotated = crop_bgr.copy()
        mp_drawing.draw_landmarks(
            annotated,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
        return rule_flag, annotated

    return rule_flag, crop_bgr


# =============================================================================
# Core Pipeline

def show_video(video_source):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open {video_source}")
        return

    track_buffers   = {}   # track_id → deque of preprocessed crop arrays
    track_status    = {}   # track_id → last LSTM verdict string
    track_scores    = {}   # track_id → deque of raw softmax outputs (smoothing)
    track_pose_prev = {}   # track_id → previous hip_x for lunge detection
    track_rule_flag = {}   # track_id → last pose rule flag string or None

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

            avg_preds  = np.mean(track_scores[t_id], axis=0)
            confidence = np.max(avg_preds)

            if np.argmax(avg_preds) == 1 and confidence > 0.65:
                track_status[t_id] = f"SUSPICIOUS ({avg_preds[1]*100:.0f}%)"
                if not alert_sent:
                    try:
                        import notifier
                        notifier.send_alert()
                    except Exception:
                        pass
                    alert_sent = True
            else:
                track_status[t_id] = f"Normal ({avg_preds[0]*100:.0f}%)"
        except Exception as e:
            print(f"[LSTM Thread ERROR]: {e}")
        finally:
            processing_tracks.discard(t_id)

    font = cv2.FONT_HERSHEY_SIMPLEX
    stop_flag = [False]

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            h_win, w_win = param
            if (w_win - 130) <= x <= (w_win - 10) and 12 <= y <= 60:
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
            # ====================== HYBRID MODE ======================
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

                    # ── Pose Rule Engine ──────────────────────────────
                    if HAS_MEDIAPIPE and frame_count % 3 == 0:
                        rule_flag, skeleton_crop = run_pose_rules(crop, track_id, track_pose_prev)
                        track_rule_flag[track_id] = rule_flag
                        # Paint skeleton back onto annotated frame
                        try:
                            crop_h = y2 - y1
                            crop_w = x2 - x1
                            if skeleton_crop.shape[:2] == (crop_h, crop_w):
                                annotated_frame[y1:y2, x1:x2] = skeleton_crop
                        except Exception:
                            pass

                    # ── LSTM Buffer & Async Inference ─────────────────
                    if has_lstm:
                        try:
                            crop_resized = cv2.resize(crop, (IMAGE_WIDTH, IMAGE_HEIGHT))
                            crop_rgb     = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
                            arr = preprocess_input(np.array(crop_rgb, dtype='float32'))

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
                            print(f"[LSTM Processing ERROR]: {e}")

                    # ── Combined Verdict Display ──────────────────────
                    lstm_verdict = track_status.get(track_id, "Tracking...")
                    rule_flag    = track_rule_flag.get(track_id)

                    # Final suspicion logic: LSTM *or* rule engine → escalate
                    is_lstm_suspicious = "SUSPICIOUS" in lstm_verdict
                    is_rule_suspicious = rule_flag is not None

                    if is_lstm_suspicious and is_rule_suspicious:
                        # Both agree → highest confidence
                        display_text = f"⚠ SUSPICIOUS | {rule_flag}"
                        box_color    = (0, 0, 255)   # Red
                    elif is_lstm_suspicious:
                        display_text = lstm_verdict
                        box_color    = (0, 60, 255)  # Red-orange
                    elif is_rule_suspicious:
                        display_text = f"⚠ {rule_flag}"
                        box_color    = (0, 140, 255)  # Orange
                    else:
                        display_text = lstm_verdict
                        box_color    = (0, 200, 80)   # Green

                    # Draw verdict banner above bounding box
                    banner_w = min(400, frame.shape[1] - x1)
                    cv2.rectangle(annotated_frame, (x1, y1 - 38), (x1 + banner_w, y1), box_color, -1)
                    cv2.putText(annotated_frame,
                                f"ID:{track_id} | {display_text}",
                                (x1 + 5, y1 - 10), font, 0.56, (255, 255, 255), 2)

        # ── Scale display to 720p ─────────────────────────────────────
        DISPLAY_HEIGHT = 720
        orig_h, orig_w = annotated_frame.shape[:2]
        scale     = DISPLAY_HEIGHT / orig_h
        display_w = int(orig_w * scale)
        display_frame = cv2.resize(annotated_frame, (display_w, DISPLAY_HEIGHT),
                                   interpolation=cv2.INTER_LINEAR)

        # ── HUD Top Bar ───────────────────────────────────────────────
        h_frame, w_frame = display_frame.shape[:2]
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w_frame, 90), (0, 0, 0), -1)
        display_frame = cv2.addWeighted(overlay, 0.55, display_frame, 0.45, 0)

        cv2.putText(display_frame, "HYBRID SYSTEM  |  YOLO + Pose + LSTM",
                    (10, 45), font, 1.1, (0, 255, 255), 2)
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 80), font, 0.8, (255, 255, 255), 2)

        # ── STOP button ───────────────────────────────────────────────
        btn_x1, btn_y1 = w_frame - 130, 12
        btn_x2, btn_y2 = w_frame - 10,  60
        cv2.rectangle(display_frame, (btn_x1, btn_y1), (btn_x2, btn_y2), (0, 0, 180), -1)
        cv2.rectangle(display_frame, (btn_x1, btn_y1), (btn_x2, btn_y2), (255, 255, 255), 2)
        cv2.putText(display_frame, 'STOP', (btn_x1 + 14, btn_y2 - 12), font, 1.0, (255, 255, 255), 2)

        cv2.setMouseCallback(win_name, mouse_callback, (h_frame, w_frame))
        cv2.imshow(win_name, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q') or stop_flag[0]:
            print("[INFO] Detection stopped.")
            break

    cap.release()
    executor.shutdown(wait=False)
    cv2.destroyAllWindows()


# =============================================================================
# Button Commands
def upload_video():
    fileName = askopenfilename(
        initialdir=SCRIPT_DIR,
        title='Select video to test',
        filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov"), ("All files", "*.*")]
    )
    if fileName:
        show_video(fileName)

def use_webcam():
    show_video(0)

# =============================================================================
# GUI
btn_frame = tk.Frame(root, bg="#192841")
btn_frame.pack(pady=20)

tk.Button(btn_frame, command=upload_video, text="Upload Video for Testing",
          width=25, font=("Times new roman", 25, "bold"), bg="cyan", fg="black"
          ).pack(pady=20)

tk.Button(btn_frame, command=use_webcam, text="Use Webcam",
          width=25, font=("Times new roman", 25, "bold"), bg="orange", fg="black"
          ).pack(pady=20)

tk.Button(btn_frame, command=root.destroy, text="Exit",
          width=25, font=("Times new roman", 25, "bold"), bg="red", fg="white"
          ).pack(pady=20)

# CLI support
if len(sys.argv) > 1:
    video_arg = sys.argv[1]
    if video_arg == "0":
        show_video(0)
    else:
        if not os.path.isabs(video_arg):
            video_arg = os.path.join(SCRIPT_DIR, video_arg)
        if os.path.isfile(video_arg):
            show_video(video_arg)
        else:
            print(f"[ERROR] File not found: {video_arg}")
    try:
        root.destroy()
    except Exception:
        pass
else:
    root.mainloop()