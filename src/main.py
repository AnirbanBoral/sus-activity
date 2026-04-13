import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF C++ warnings

import sys
import tkinter as tk
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import cv2
import numpy as np
from collections import deque
import random
import time
from concurrent.futures import ThreadPoolExecutor

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# YOLOv8 imports for PyTorch tracking
try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] Ultralytics module not found. Run 'pip install ultralytics' first.")
    YOLO = None

# =============================================================================
# Configuration
# Change this variable to False to use the deprecated, older (non-hybrid) MobileNetV2+LSTM logic
USE_YOLO_HYBRID = True 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'mobilenet_model.h5')
IMAGE_HEIGHT, IMAGE_WIDTH = 128, 128
SEQUENCE_LENGTH = 12

# =============================================================================
# Load Global Models
print("[INFO] Initializing Spatio-Temporal Intent (LSTM) Model...")
try:
    lstm_model = load_model(MODEL_PATH)
    has_lstm = True
    print("[OK] LSTM Loaded.")
except Exception as e:
    print(f"[WARNING] Failed to load LSTM {MODEL_PATH}. {e}")
    lstm_model = None
    has_lstm = False

print("[INFO] Initializing Spatial Tracking (YOLOv8) Model...")
if YOLO:
    yolo_model = YOLO("yolov8s.pt")  # Using small model for real-time inference with higher accuracy
else:
    yolo_model = None


# =============================================================================
# Frontend UI
root = tk.Tk()
root.state('zoomed')
title_text = "Suspicious Activity Detection (YOLOv8 + LSTM)" if USE_YOLO_HYBRID else "Suspicious Activity Detection (MobilenetV2 + LSTM)"
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
    root, text=title_text, width=35,
    font=("Times New Roman", 45, 'bold'), bg="#192841", fg="white"
)
heading.pack(pady=50)


# =============================================================================
# Core Pipeline

def show_video(video_source):
    """
    Run video feed. If USE_YOLO_HYBRID=True, tracks humans individually.
    If False, evaluates the entire frame at once (deprecated method).
    """
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open {video_source}")
        return

    track_buffers = {}
    track_status = {}
    track_scores = {}
    alert_sent = False
    prev_time = time.time()
    
    executor = ThreadPoolExecutor(max_workers=3)
    processing_tracks = set()

    def evaluate_intent_async(t_id, X_seq):
        nonlocal alert_sent
        try:
            preds = lstm_model(X_seq, training=False).numpy()[0]
            if t_id not in track_scores:
                track_scores[t_id] = deque(maxlen=3)
            track_scores[t_id].append(preds)
            
            avg_preds = np.mean(track_scores[t_id], axis=0)
            confidence = np.max(avg_preds)
            
            if np.argmax(avg_preds) == 1 and confidence > 0.65:
                track_status[t_id] = f"SUSPICIOUS ({avg_preds[1]*100:.0f}%)"
                if not alert_sent:
                    try:
                        import notifier
                        notifier.send_alert()
                    except Exception as e:
                        print(f"[ERROR sending alert]: {e}")
                    alert_sent = True
            else:
                track_status[t_id] = f"Normal ({avg_preds[0]*100:.0f}%)"
        except Exception as e:
            print(f"[ERROR in LSTM Thread]: {e}")
        finally:
            processing_tracks.discard(t_id)
            
    legacy_processing = set()
    def evaluate_legacy_async(X_seq):
        nonlocal legacy_label, legacy_color, alert_sent
        try:
            predicted = lstm_model(X_seq, training=False).numpy()
            confidence = np.max(predicted[0])
            
            if np.argmax(predicted[0]) == 1 and confidence >= 0.65:
                legacy_label = "Suspicious Activity Detected"
                legacy_color = (0, 0, 255)
                if not alert_sent:
                    try:
                        import notifier
                        notifier.send_alert()
                    except Exception as e:
                        print(f"[ERROR sending alert]: {e}")
                    alert_sent = True
            else:
                legacy_label = "Normal Activity Detected"
                legacy_color = (0, 255, 0)
        except Exception as e:
            print(f"[ERROR in Legacy Thread]: {e}")
        finally:
            legacy_processing.discard('legacy')

    # Legacy buffer for non-hybrid mode
    legacy_buffer = deque(maxlen=SEQUENCE_LENGTH)
    legacy_label = "Normal Activity detected"
    legacy_color = (0, 255, 0)
    frame_count = 0

    font = cv2.FONT_HERSHEY_SIMPLEX
    stop_flag = [False]

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            h_win, w_win = param
            btn_x1, btn_y1 = w_win - 130, 12
            btn_x2, btn_y2 = w_win - 10, 60
            if btn_x1 <= x <= btn_x2 and btn_y1 <= y <= btn_y2:
                stop_flag[0] = True
                
    win_name = 'Hybrid Pipeline' if USE_YOLO_HYBRID else 'Legacy MobileNet-LSTM'
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
            # ======================== HYBRID MODE ========================
            results = yolo_model.track(frame, persist=True, classes=[0], verbose=False)
            annotated_frame = results[0].plot()

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    crop = frame[y1:y2, x1:x2]
                    
                    if crop.size == 0:
                        continue
                        
                    if has_lstm:
                        try:
                            crop_resized = cv2.resize(crop, (IMAGE_WIDTH, IMAGE_HEIGHT))
                            crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
                            arr = np.array(crop_rgb, dtype='float32')
                            arr = preprocess_input(arr)
                            
                            if track_id not in track_buffers:
                                track_buffers[track_id] = deque(maxlen=SEQUENCE_LENGTH)
                                track_status[track_id] = "Tracking Intent..."
                                
                            track_buffers[track_id].append(arr)
                            
                            if len(track_buffers[track_id]) == SEQUENCE_LENGTH and (frame_count + track_id) % 5 == 0:
                                if track_id not in processing_tracks:
                                    processing_tracks.add(track_id)
                                    X = np.expand_dims(np.array(track_buffers[track_id]), axis=0)
                                    executor.submit(evaluate_intent_async, track_id, X)
                        except Exception as e:
                            import traceback
                            print(f"[ERROR in LSTM Processing]: {e}")
                            traceback.print_exc()

                    status_text = track_status.get(track_id, "Tracking Intent...")
                    color = (0, 0, 255) if "SUSPICIOUS" in status_text else (0, 255, 0)
                    cv2.rectangle(annotated_frame, (x1, y1 - 35), (x1 + 300, y1), color, -1)
                    cv2.putText(annotated_frame, f"ID: {track_id} | {status_text}", (x1 + 5, y1 - 10),
                                font, 0.6, (255, 255, 255), 2)
        else:
            # ======================== LEGACY MODE ========================
            if has_lstm:
                img_bgr = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_CUBIC)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                X_img = np.array(img_rgb).reshape(IMAGE_WIDTH, IMAGE_HEIGHT, 3).astype('float32')
                X_img = preprocess_input(X_img)
                legacy_buffer.append(X_img)

                if len(legacy_buffer) == SEQUENCE_LENGTH and frame_count % 3 == 0:
                    if 'legacy' not in legacy_processing:
                        legacy_processing.add('legacy')
                        X_seq = np.expand_dims(np.array(legacy_buffer), axis=0)
                        executor.submit(evaluate_legacy_async, X_seq)

            # Draw Legacy UI Elements
            cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 110), font, 1.0, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, legacy_label, (10, 150), font, 1.0, legacy_color, 2, cv2.LINE_AA)


        # Scale up display frame for visibility
        DISPLAY_HEIGHT = 720
        orig_h, orig_w = annotated_frame.shape[:2]
        scale = DISPLAY_HEIGHT / orig_h
        display_w = int(orig_w * scale)
        display_frame = cv2.resize(annotated_frame, (display_w, DISPLAY_HEIGHT), interpolation=cv2.INTER_LINEAR)

        # Draw Global UI Canvas (Top Bar)
        h_frame, w_frame = display_frame.shape[:2]
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w_frame, 90), (0, 0, 0), -1)
        display_frame = cv2.addWeighted(overlay, 0.5, display_frame, 0.5, 0)

        header_txt = "HYBRID SYSTEM ACTIVE (YOLO + LSTM)" if USE_YOLO_HYBRID else "LEGACY SYSTEM ACTIVE (LSTM ONLY)"
        header_clr = (0, 255, 255) if USE_YOLO_HYBRID else (255, 100, 0)
        cv2.putText(display_frame, header_txt, (10, 45), font, 1.2, header_clr, 2)
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 80), font, 0.8, (255, 255, 255), 2)

        # STOP button in top-right
        btn_x1, btn_y1 = w_frame - 130, 12
        btn_x2, btn_y2 = w_frame - 10, 60
        cv2.rectangle(display_frame, (btn_x1, btn_y1), (btn_x2, btn_y2), (0, 0, 200), -1)
        cv2.rectangle(display_frame, (btn_x1, btn_y1), (btn_x2, btn_y2), (255, 255, 255), 2)
        cv2.putText(display_frame, 'STOP', (btn_x1 + 14, btn_y2 - 12), font, 1.0, (255, 255, 255), 2)

        cv2.setMouseCallback(win_name, mouse_callback, (h_frame, w_frame))

        cv2.imshow(win_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q') or stop_flag[0]:
            print("[INFO] Detection stopped by user.")
            break

    cap.release()
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
# Main GUI buttons
btn_frame = tk.Frame(root, bg="#192841")
btn_frame.pack(pady=20)

btn_upload = tk.Button(
    btn_frame, command=upload_video,
    text="Upload Video for Testing", width=25,
    font=("Times new roman", 25, "bold"), bg="cyan", fg="black"
)
btn_upload.pack(pady=20)

btn_webcam = tk.Button(
    btn_frame, command=use_webcam,
    text="Use Webcam", width=25,
    font=("Times new roman", 25, "bold"), bg="orange", fg="black"
)
btn_webcam.pack(pady=20)

btn_exit = tk.Button(
    btn_frame, command=root.destroy,
    text="Exit", width=25,
    font=("Times new roman", 25, "bold"), bg="red", fg="white"
)
btn_exit.pack(pady=20)


# Support CLI arguments
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