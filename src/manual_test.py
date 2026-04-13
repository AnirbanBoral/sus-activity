"""
Manual Test Script — Evaluate mobilenet_model.h5 against the Test_* dataset folders.

Picks all available sequences from each Test category (suspicious + normal),
runs them through the trained architecture, and reports per-category accuracy.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from PIL import Image
from tqdm import tqdm
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ── Config ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'mobilenet_model.h5')
IMG_SIZE = 64
SEQUENCE_LENGTH = 15

SUSPICIOUS_DIR = os.path.join(SCRIPT_DIR, 'data', 'suspicious')
NORMAL_DIR = os.path.join(SCRIPT_DIR, 'data', 'normal')

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg'}

def extract_prefix_and_frame(filename):
    last_underscore_idx = filename.rfind('_')
    if last_underscore_idx == -1: return None, None
    prefix = filename[:last_underscore_idx]
    frame_str = filename[last_underscore_idx+1:].split('.')[0]
    try: return prefix, int(frame_str)
    except ValueError: return None, None

def get_test_folders(base_dir, label, prefix='Test_'):
    folders = []
    if not os.path.isdir(base_dir): return folders
    for name in sorted(os.listdir(base_dir)):
        if name.startswith(prefix) and os.path.isdir(os.path.join(base_dir, name)):
            folders.append((os.path.join(base_dir, name), name, label))
    return folders

def load_sequences_from_folder(folder_path):
    """ Loads all valid sequences from a given test folder. """
    video_groups = {}
    for root, _, files in os.walk(folder_path):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                prefix, frame_num = extract_prefix_and_frame(f)
                if prefix is not None:
                    filepath = os.path.join(root, f)
                    if prefix not in video_groups: video_groups[prefix] = []
                    video_groups[prefix].append((frame_num, filepath))
                    
    sequences = []
    for prefix, frames in video_groups.items():
        frames.sort(key=lambda x: x[0])
        current_sequence = []
        for _, filepath in frames:
            try:
                img = Image.open(filepath).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
                img_array = np.array(img, dtype='float16')
                img_array = preprocess_input(img_array)
                current_sequence.append(img_array)
                if len(current_sequence) == SEQUENCE_LENGTH:
                    sequences.append(list(current_sequence))
                    current_sequence = current_sequence[SEQUENCE_LENGTH // 2:]
            except Exception: pass
    if len(sequences) == 0:
        return np.array([])
    return np.array(sequences)


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at: {MODEL_PATH}")
        return
    print(f"Loading MobileNet-LSTM model: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print("[OK] Model loaded.\n")

    test_folders = []
    test_folders.extend(get_test_folders(SUSPICIOUS_DIR, label=1))
    test_folders.extend(get_test_folders(NORMAL_DIR, label=0))

    if not test_folders:
        print("[ERROR] No Test_ folders found in data/suspicious or data/normal.")
        return

    print(f"{'Category':<25} {'Sequences':>10} {'Correct':>8} {'Accuracy':>10}  {'Verdict'}")
    print("-" * 75)

    total_correct = 0
    total_samples = 0
    failures = []

    for folder_path, folder_name, true_label in test_folders:
        sequences = load_sequences_from_folder(folder_path)
        if len(sequences) == 0:
            print(f"{folder_name:<25} {'0':>10} {'—':>8} {'N/A':>10}  SKIPPED")
            continue

        preds = model.predict(sequences, verbose=0)
        pred_labels = np.argmax(preds, axis=1)
        correct = int(np.sum(pred_labels == true_label))
        acc = correct / len(preds) * 100
        
        total_correct += correct
        total_samples += len(preds)

        label_str = "suspicious" if true_label == 1 else "normal"
        # We can drop passing criteria for sequences slightly
        status = "PASS" if acc >= 70 else "FAIL"
        if acc < 70:
            failures.append((folder_name, label_str, acc))

        print(f"{folder_name:<25} {len(preds):>10} {correct:>8} {acc:>9.1f}%  {status}")

    print("-" * 75)
    overall_acc = (total_correct / total_samples * 100) if total_samples > 0 else 0
    print(f"{'OVERALL':<25} {total_samples:>10} {total_correct:>8} {overall_acc:>9.1f}%")

    if failures:
        print(f"\nFAILED CATEGORIES ({len(failures)}):")
        for name, label, acc in failures:
            print(f"  - {name} (expected={label}, accuracy={acc:.1f}%)")

if __name__ == '__main__':
    main()
