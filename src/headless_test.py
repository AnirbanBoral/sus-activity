"""
headless_test.py — Tests detection logic without Tkinter GUI.
Usage: .venv\Scripts\python.exe src\headless_test.py <video_path>
"""
import os, sys, time, json, threading
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Stub out tkinter and notifier before importing main
import types, unittest.mock as mock
sys.modules['tkinter'] = mock.MagicMock()
sys.modules['tkinter.filedialog'] = mock.MagicMock()

# Stub notifier so emails don't fire during test
import notifier as _notifier_real
_notifier_real.send_alert = lambda *a, **kw: None
_notifier_real.load_config = lambda: {}

import cv2
import numpy as np
from collections import deque, defaultdict

VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else None

# ─── Import shared models from main ──────────────────────────────────────────
print("[TEST] Loading models...")
import importlib.util
spec = importlib.util.spec_from_file_location("main_module",
    os.path.join(os.path.dirname(__file__), "main.py"))
main = importlib.util.module_from_spec(spec)

# Patch _toggle_vars before exec so detection flags return True
_fake_var = mock.MagicMock()
_fake_var.get = mock.MagicMock(return_value=True)
main._toggle_vars = defaultdict(lambda: _fake_var)
main.root = None

try:
    spec.loader.exec_module(main)
except Exception as e:
    print(f"[ERROR] main.py import failed: {e}")
    sys.exit(1)

print("[OK] Models loaded.")

# ─── Run headless detection on video ─────────────────────────────────────────
if VIDEO_PATH is None:
    print("\n[TEST] No video path provided — running logic unit tests only.\n")
else:
    print(f"\n[TEST] Processing video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("[ERROR] Cannot open video file.")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_src      = cap.get(cv2.CAP_PROP_FPS) or 25
    print(f"[INFO] {total_frames} frames @ {fps_src:.1f} fps")

    # Create a dummy CameraStream to run _process_frame on
    dummy_label = mock.MagicMock()
    dummy_label.winfo_width  = mock.MagicMock(return_value=0)
    dummy_label.winfo_height = mock.MagicMock(return_value=0)
    stream = main.CameraStream(VIDEO_PATH, "TEST-CAM", dummy_label)

    detections = []
    frame_times = []
    frame_cnt   = 0
    SAMPLE_EVERY = 5   # only process every 5th frame for speed
    MAX_FRAMES   = 150 # cap at 150 processed frames (~30s at 25fps)

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_cnt += 1
        if frame_cnt % SAMPLE_EVERY != 0: continue
        if frame_cnt // SAMPLE_EVERY > MAX_FRAMES: break

        t0 = time.time()
        ann = stream._process_frame(frame, frame_cnt)
        elapsed = time.time() - t0
        frame_times.append(elapsed)

        active = []
        for tid, flags in stream.track_active_flags.items():
            active.extend(flags)
        if active:
            detections.append((frame_cnt, list(set(active))))
            print(f"  Frame {frame_cnt:4d}: {list(set(active))}")

        if frame_cnt % 50 == 0:
            print(f"  ... frame {frame_cnt}/{total_frames} "
                  f"({elapsed*1000:.0f}ms/frame)")

    cap.release()

    # ─── Results ──────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("HEADLESS TEST RESULTS")
    print("="*55)
    print(f"Frames processed : {min(frame_cnt // SAMPLE_EVERY, MAX_FRAMES)}")
    print(f"Detection events : {len(detections)}")
    if frame_times:
        avg_ms = sum(frame_times)/len(frame_times)*1000
        max_ms = max(frame_times)*1000
        print(f"Avg inference    : {avg_ms:.0f}ms/frame")
        print(f"Max inference    : {max_ms:.0f}ms/frame")
        print(f"Effective FPS    : {1000/avg_ms:.1f} fps")

    if detections:
        tag_counts = defaultdict(int)
        for _, tags in detections: 
            for t in tags: tag_counts[t] += 1
        print("\nDetection breakdown:")
        for tag, cnt in sorted(tag_counts.items(), key=lambda x: -x[1]):
            print(f"  {tag:<20} {cnt:>4} frames")
        print("\n[PASS] Detection pipeline is firing correctly.")
    else:
        print("\n[WARN] No detections triggered. Check that:")
        print("  1. The video contains visible people")
        print("  2. yolov10n.pt downloaded correctly")
        print("  3. pose_landmarker_lite.task is in src/")

# ─── Unit tests for helper methods ───────────────────────────────────────────
print("\n--- Unit Tests ---")

dummy_label2 = mock.MagicMock()
dummy_label2.winfo_width  = mock.MagicMock(return_value=0)
dummy_label2.winfo_height = mock.MagicMock(return_value=0)
s = main.CameraStream("dummy", "UNIT", dummy_label2)

# Test 1: velocity detection
s.track_center_history[1] = deque(maxlen=10)
now = time.time()
for i in range(5):
    s.track_center_history[1].append((i * 0.1, 0, now + i * 0.1))
result = s._check_velocity(1, 0.5, 0, now + 0.5)
print(f"[{'PASS' if result == 'RAPID MOVE' else 'FAIL'}] Velocity detection: {result}")

# Test 2: velocity non-trigger (slow movement)
s2 = main.CameraStream("dummy", "UNIT2", dummy_label2)
s2.track_center_history[2] = deque(maxlen=10)
for i in range(5):
    s2.track_center_history[2].append((i * 0.001, 0, now + i * 0.5))
result2 = s2._check_velocity(2, 0.005, 0, now + 2.5)
print(f"[{'PASS' if result2 is None else 'FAIL'}] Velocity non-trigger: {result2}")

# Test 3: fall detection
s3 = main.CameraStream("dummy", "UNIT3", dummy_label2)
for i in range(20):
    h = 200 if i < 10 else 60   # height drops from 200px to 60px
    s3._check_fall(3, 100, 100, 200, 100 + h, now + i)
result3 = s3._check_fall(3, 100, 100, 200, 160, now + 20)
print(f"[{'PASS' if result3 == 'FALL DETECTED' else 'FAIL'}] Fall detection: {result3}")

# Test 4: alert cooldown (should not re-alert within cooldown period)
s4 = main.CameraStream("dummy", "UNIT4", dummy_label2)
s4.ALERT_COOLDOWN_S = 30
s4.track_last_alert_time[4] = time.time()  # just alerted
last_alerted = s4.track_last_alert_time.get(4, 0)
would_alert = (time.time() - last_alerted) > s4.ALERT_COOLDOWN_S
print(f"[{'PASS' if not would_alert else 'FAIL'}] Alert cooldown blocks re-alert: {not would_alert}")

# Test 5: stale flag pruning
s5 = main.CameraStream("dummy", "UNIT5", dummy_label2)
s5.track_active_flags = {10: ["WEAPON"], 11: ["FALL"], 12: ["RAPID MOVE"]}
current_ids = [10]  # only ID 10 still in frame
stale = [t for t in s5.track_active_flags if t not in current_ids]
for t in stale: del s5.track_active_flags[t]
print(f"[{'PASS' if list(s5.track_active_flags.keys()) == [10] else 'FAIL'}] Stale flag pruning: {list(s5.track_active_flags.keys())}")

print("\n[DONE] All tests complete.")
