"""Quick headless test: Run mobilenet_model.h5 on test video sequences and print predictions."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from PIL import Image
from collections import deque
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'mobilenet_model.h5')
IMG_SIZE = 64
SEQ_LEN = 15

def extract_prefix_and_frame(filename):
    idx = filename.rfind('_')
    if idx == -1: return None, None
    prefix = filename[:idx]
    frame_str = filename[idx+1:].split('.')[0]
    try: return prefix, int(frame_str)
    except ValueError: return None, None

def test_folder(model, folder_path, expected_label):
    """Run model on all sequences in a folder, return accuracy."""
    video_groups = {}
    for root, _, files in os.walk(folder_path):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in ['.png', '.jpg', '.jpeg']:
                prefix, frame_num = extract_prefix_and_frame(f)
                if prefix is not None:
                    filepath = os.path.join(root, f)
                    if prefix not in video_groups: video_groups[prefix] = []
                    video_groups[prefix].append((frame_num, filepath))

    sequences = []
    for prefix, frames in video_groups.items():
        frames.sort(key=lambda x: x[0])
        buf = []
        for _, fp in frames:
            try:
                img = Image.open(fp).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
                arr = np.array(img, dtype='float16')
                arr = preprocess_input(arr)
                buf.append(arr)
                if len(buf) == SEQ_LEN:
                    sequences.append(np.array(buf))
                    buf = []
            except: pass

    if not sequences:
        return None, 0, 0

    X = np.array(sequences)
    preds = model.predict(X, verbose=0)
    pred_labels = np.argmax(preds, axis=1)
    correct = int(np.sum(pred_labels == expected_label))
    acc = correct / len(preds) * 100
    return acc, correct, len(preds)

def main():
    print(f"Loading model: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print("[OK] Model loaded.\n")

    suspicious_dir = os.path.join(SCRIPT_DIR, 'data', 'suspicious')
    normal_dir = os.path.join(SCRIPT_DIR, 'data', 'normal')

    # Test a few suspicious categories
    test_cats = []
    for d in [suspicious_dir, normal_dir]:
        label = 1 if 'suspicious' in d else 0
        if os.path.isdir(d):
            for name in sorted(os.listdir(d)):
                if name.startswith('Test_') and os.path.isdir(os.path.join(d, name)):
                    test_cats.append((os.path.join(d, name), name, label))

    if not test_cats:
        print("No Test_ folders found!")
        return

    print(f"{'Category':<25} {'Seqs':>6} {'Correct':>8} {'Accuracy':>10}  {'Result'}")
    print("-" * 65)

    total_correct = 0
    total_seqs = 0

    for folder, name, label in test_cats[:6]:  # Test first 6 categories for speed
        acc, correct, count = test_folder(model, folder, label)
        if acc is None:
            print(f"{name:<25} {'0':>6} {'--':>8} {'N/A':>10}  SKIP")
            continue
        total_correct += correct
        total_seqs += count
        status = "PASS" if acc >= 70 else "FAIL"
        print(f"{name:<25} {count:>6} {correct:>8} {acc:>9.1f}%  {status}")

    print("-" * 65)
    overall = (total_correct / total_seqs * 100) if total_seqs > 0 else 0
    print(f"{'OVERALL (sampled)':<25} {total_seqs:>6} {total_correct:>8} {overall:>9.1f}%")

if __name__ == '__main__':
    main()
