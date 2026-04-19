import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_curve, auc)
import cv2
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Configuration — must match trainer.py ────────────────────────────────────
IMAGE_HEIGHT, IMAGE_WIDTH = 128, 128
SEQUENCE_LENGTH  = 12
IMAGE_CHANNELS   = 3
POSE_VECTOR_SIZE = 99
BATCH_SIZE       = 4

base_dir   = os.path.dirname(os.path.abspath(__file__))
PERF_DIR   = os.path.join(base_dir, 'model_performance')
MODEL_PATH = os.path.join(base_dir, 'hybrid_pose_mobilenet_model_v2.h5')
POSE_MODEL = os.path.join(base_dir, 'pose_landmarker_lite.task')
DATA_DIR_SUSPICIOUS = os.path.join(base_dir, 'data', 'suspicious')
DATA_DIR_NORMAL     = os.path.join(base_dir, 'data', 'normal')

os.makedirs(PERF_DIR, exist_ok=True)


# ── MediaPipe init ────────────────────────────────────────────────────────────
pose_landmarker = None
if os.path.exists(POSE_MODEL):
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
    except Exception as e:
        print(f"[WARNING] Pose model failed: {e}. Pose vectors will be zeros.")


def get_pose_vector(image_path):
    if pose_landmarker is None:
        return np.zeros((POSE_VECTOR_SIZE,), dtype='float32')
    try:
        bgr    = cv2.imread(image_path)
        if bgr is None:
            return np.zeros((POSE_VECTOR_SIZE,), dtype='float32')
        rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = pose_landmarker.detect(mp_img)
        if not result.pose_landmarks:
            return np.zeros((POSE_VECTOR_SIZE,), dtype='float32')
        lm  = result.pose_landmarks[0]
        vec = []
        for landmark in lm:
            vec.extend([landmark.x, landmark.y, landmark.z])
        return np.array(vec, dtype='float32')
    except Exception:
        return np.zeros((POSE_VECTOR_SIZE,), dtype='float32')


def extract_prefix_and_frame(filename):
    last_idx = filename.rfind('_')
    if last_idx == -1:
        return None, None
    prefix    = filename[:last_idx]
    frame_str = filename[last_idx + 1:].split('.')[0]
    try:
        return prefix, int(frame_str)
    except ValueError:
        return None, None


def load_sequences(directory, label, max_seqs=300):
    """Load up to max_seqs sequences from a directory."""
    video_groups = {}
    for root_dir, _, files in os.walk(directory):
        for f in files:
            if os.path.splitext(f)[1].lower() in ['.png', '.jpg', '.jpeg']:
                prefix, frame_num = extract_prefix_and_frame(f)
                if prefix is not None:
                    filepath = os.path.join(root_dir, f)
                    if prefix not in video_groups:
                        video_groups[prefix] = []
                    video_groups[prefix].append((frame_num, filepath))

    sequences, labels = [], []
    for prefix, frames in video_groups.items():
        frames.sort(key=lambda x: x[0])
        current = []
        for _, filepath in frames:
            current.append(filepath)
            if len(current) == SEQUENCE_LENGTH:
                sequences.append(list(current))
                labels.append(label)
                current = []

    # Cap and shuffle reproducibly
    combined = list(zip(sequences, labels))
    np.random.default_rng(42).shuffle(combined)
    combined = combined[:max_seqs]
    if not combined:
        return [], []
    sequences, labels = zip(*combined)
    return list(sequences), list(labels)


def sequences_to_arrays(seq_paths_list):
    """Convert list of sequence path lists to numpy arrays."""
    n = len(seq_paths_list)
    X_img  = np.zeros((n, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype='float32')
    X_pose = np.zeros((n, SEQUENCE_LENGTH, POSE_VECTOR_SIZE), dtype='float32')

    for i, seq_paths in enumerate(seq_paths_list):
        for j, filepath in enumerate(seq_paths):
            X_pose[i, j] = get_pose_vector(filepath)
            try:
                img = Image.open(filepath).convert('RGB').resize((IMAGE_WIDTH, IMAGE_HEIGHT))
                X_img[i, j] = preprocess_input(np.array(img, dtype='float32'))
            except Exception:
                pass
        if (i + 1) % 50 == 0:
            print(f"  Loaded {i+1}/{n} sequences...")

    return X_img, X_pose


def run_evaluation():
    # ── Load model ────────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at {MODEL_PATH}")
        print("        Run trainer.py first to generate the model.")
        return

    print(f"[INFO] Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("[OK] Model loaded.")

    # ── Load test data ────────────────────────────────────────────────────
    if not os.path.exists(DATA_DIR_NORMAL) or not os.path.exists(DATA_DIR_SUSPICIOUS):
        print("[ERROR] data/normal or data/suspicious not found.")
        print("        Cannot run real evaluation without test data.")
        return

    print("\n[INFO] Loading Normal sequences...")
    normal_seqs, normal_labels = load_sequences(DATA_DIR_NORMAL, label=0, max_seqs=300)
    print(f"  Loaded {len(normal_seqs)} Normal sequences.")

    print("[INFO] Loading Suspicious sequences...")
    susp_seqs, susp_labels = load_sequences(DATA_DIR_SUSPICIOUS, label=1, max_seqs=300)
    print(f"  Loaded {len(susp_seqs)} Suspicious sequences.")

    all_seqs   = normal_seqs + susp_seqs
    all_labels = normal_labels + susp_labels

    if len(all_seqs) == 0:
        print("[ERROR] No data loaded. Check your data/ directories.")
        return

    print(f"\n[INFO] Converting {len(all_seqs)} sequences to arrays...")
    X_img, X_pose = sequences_to_arrays(all_seqs)
    y_true = np.array(all_labels)

    # ── Inference ─────────────────────────────────────────────────────────
    print("\n[INFO] Running inference...")
    raw_preds = model.predict(
        {"image_input": X_img, "pose_input": X_pose},
        batch_size=BATCH_SIZE, verbose=1
    )
    y_pred       = np.argmax(raw_preds, axis=1)
    y_scores_pos = raw_preds[:, 1]   # probability of Suspicious

    # ── 1. Confusion Matrix ───────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Suspicious'],
                yticklabels=['Normal', 'Suspicious'])
    plt.title('Confusion Matrix: Suspicious Activity Detection')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    cm_path = os.path.join(PERF_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {cm_path}")

    # ── 2. ROC-AUC Curve ─────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_true, y_scores_pos)
    roc_auc     = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curve — Suspicious Activity Detection')
    plt.legend(loc='lower right'); plt.grid(alpha=0.3)
    roc_path = os.path.join(PERF_DIR, 'roc_auc_curve.png')
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {roc_path}")

    # ── 3. Classification Report ──────────────────────────────────────────
    report = classification_report(y_true, y_pred,
                                    target_names=['Normal', 'Suspicious'])
    print("\nClassification Report:")
    print(report)

    report_path = os.path.join(PERF_DIR, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("Classification Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(report)
        f.write(f"\nROC-AUC Score: {roc_auc:.4f}\n")
    print(f"Saved: {report_path}")

    # ── 4. Confidence Distribution ────────────────────────────────────────
    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.hist(y_scores_pos[y_true == 0], bins=20, color='steelblue', alpha=0.7, label='Normal')
    plt.hist(y_scores_pos[y_true == 1], bins=20, color='tomato',    alpha=0.7, label='Suspicious')
    plt.xlabel('Predicted Suspicious Probability')
    plt.ylabel('Count'); plt.title('Confidence Distribution')
    plt.legend(); plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    tn, fp, fn, tp = cm.ravel()
    metrics_vals = [tp/(tp+fn)*100, tn/(tn+fp)*100, tp/(tp+fp)*100,
                    (tp+tn)/(tp+tn+fp+fn)*100]
    metrics_names = ['Recall\n(Suspicious)', 'Specificity\n(Normal)',
                     'Precision\n(Suspicious)', 'Accuracy']
    colors = ['tomato', 'steelblue', 'orange', 'purple']
    bars = plt.bar(metrics_names, metrics_vals, color=colors, alpha=0.8)
    for bar, val in zip(bars, metrics_vals):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    plt.ylim(0, 115); plt.title('Key Performance Metrics'); plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    dist_path = os.path.join(PERF_DIR, 'performance_summary.png')
    plt.savefig(dist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {dist_path}")

    # ── 5. Summary Markdown ───────────────────────────────────────────────
    tn, fp, fn, tp = cm.ravel()
    recall      = tp / max(tp + fn, 1) * 100
    precision   = tp / max(tp + fp, 1) * 100
    specificity = tn / max(tn + fp, 1) * 100
    accuracy    = (tp + tn) / max(tp + tn + fp + fn, 1) * 100

    summary = f"""# Model Performance Report

## Real Evaluation Results 
All metrics below are computed from actual model inference on {len(all_seqs)} test sequences.

## Metrics Summary
| Metric | Value |
|---|---|
| Overall Accuracy | {accuracy:.1f}% |
| Suspicious Recall | {recall:.1f}% |
| Suspicious Precision | {precision:.1f}% |
| Normal Specificity | {specificity:.1f}% |
| ROC-AUC Score | {roc_auc:.4f} |

## Confusion Matrix
|  | Predicted Normal | Predicted Suspicious |
|---|---|---|
| **Actual Normal** | {tn} (TN) | {fp} (FP) |
| **Actual Suspicious** | {fn} (FN) | {tp} (TP) |

## Visualization Gallery
- confusion_matrix.png
- roc_auc_curve.png
- performance_summary.png

## Notes
- Recall of {recall:.1f}% means the system catches {recall:.0f}% of real threats
- Precision of {precision:.1f}% means {precision:.0f}% of alarms are genuine
- ROC-AUC of {roc_auc:.3f} shows discriminative power across all thresholds
"""
    summary_path = os.path.join(PERF_DIR, 'performance_summary.md')
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"Saved: {summary_path}")

    print(f"\n[DONE] All reports saved to {PERF_DIR}/")
    print(f"  Accuracy:  {accuracy:.1f}%")
    print(f"  Recall:    {recall:.1f}%")
    print(f"  Precision: {precision:.1f}%")
    print(f"  ROC-AUC:   {roc_auc:.4f}")


if __name__ == "__main__":
    run_evaluation()
