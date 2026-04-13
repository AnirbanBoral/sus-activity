import os
import sys
import cv2
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from ultralytics import YOLO

def verify_pipeline():
    model_path = os.path.join('src', 'mobilenet_model.h5')
    video_path = os.path.join('src', 'manual_test', 'fight.avi')
    
    if not os.path.exists(video_path):
        print(f"[ERROR] Video {video_path} not found!")
        return False
        
    print("[TEST] Loading LSTM...")
    try:
        lstm_model = load_model(model_path)
    except Exception as e:
        print(f"[ERROR] Failed to load dummy LSTM: {e}")
        return False
        
    print("[TEST] Loading YOLOv8s...")
    try:
        yolo_model = YOLO('yolov8s.pt')
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO: {e}")
        return False
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Could not open video file.")
        return False
        
    track_buffers = {}
    track_scores = {}
    
    frame_count = 0
    predictions = 0
    
    print(f"[TEST] Processing {video_path}...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Test YOLOv8 tracking
        try:
            results = yolo_model.track(frame, persist=True, classes=[0], verbose=False)
        except Exception as e:
            print(f"[ERROR] YOLO logic failed on frame {frame_count} - {e}")
            return False
            
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                crop = frame[y1:y2, x1:x2]
                
                if crop.size == 0: continue
                
                # Test Buffer Appending
                try:
                    crop_resized = cv2.resize(crop, (128, 128))
                    crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
                    arr = preprocess_input(np.expand_dims(np.array(crop_rgb, dtype='float32'), axis=0))[0]
                    
                    if track_id not in track_buffers:
                        track_buffers[track_id] = deque(maxlen=12)
                        
                    track_buffers[track_id].append(arr)
                except Exception as e:
                    print(f"[ERROR] Cropping/Buffering failed - {e}")
                    return False
                    
                # Test Intent Analysis
                if len(track_buffers[track_id]) == 12 and (frame_count + track_id) % 5 == 0:
                    try:
                        seq = np.expand_dims(np.array(track_buffers[track_id]), axis=0)
                        preds = lstm_model(seq, training=False).numpy()[0]
                        predictions += 1
                        
                        if track_id not in track_scores:
                            track_scores[track_id] = deque(maxlen=3)
                        track_scores[track_id].append(preds)
                        
                        avg_preds = np.mean(track_scores[track_id], axis=0)
                        confidence = np.max(avg_preds)
                        print(f"  -> Frame {frame_count} | Person {track_id} Intent: {'SUSPICIOUS' if np.argmax(avg_preds)==1 else 'Normal'} ({confidence*100:.1f}%)")
                    except Exception as e:
                        print(f"[ERROR] LSTM prediction failed - {e}")
                        return False

    cap.release()
    print(f"[SUCCESS] Test fully completed! Parsed {frame_count} frames and successfully ran LSTM Analysis {predictions} times without crashing.")
    return True

if __name__ == '__main__':
    verify_pipeline()
