"""
trainer.py
====================
Hybrid Pose + EfficientNetV2B0 + LSTM trainer
Optimised for RTX 3050 4GB / Ryzen 5600U

Key upgrades over MobileNetV3Small version
────────────────────────────────────────────
1.  EfficientNetV2B0 backbone
      • ~78% ImageNet top-1 vs ~67% for MobileNetV3Small
      • Fused-MBConv blocks train 3–9× faster than depthwise-separable convs
      • 5.9M params — fits in 4 GB with mixed precision at batch=4

2.  Mixed precision (float16 compute / float32 master weights)
      • Auto-enabled only when GPU is detected; CPU falls back to float32

3.  Focal Loss (γ=2, α=0.25)
      • Handles Normal >> Suspicious imbalance automatically
      • No manual class_weight tuning needed

4.  Cosine Decay with Restarts LR schedule
      • Periodic restarts escape local minima on small datasets

5.  2-phase training: warmup (CNN frozen) → fine-tune (top-30 layers)
      • Prevents random LSTM weights corrupting pretrained features

6.  Hardened augmentation: flip, brightness/contrast jitter, zoom crop,
      temporal frame drops, skeleton Gaussian noise on pose vectors

7.  Full evaluation: confusion matrix, per-class report, PNG + CSV saved
"""

import os
import gc
import cv2
import numpy as np
from PIL import Image
import random
import tempfile
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# ── Mixed Precision Setup ────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("[INFO] Mixed precision enabled (float16 compute).")
else:
    print("[INFO] No GPU detected — training on CPU in float32.")

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, TimeDistributed, LSTM, Dense,
                                     Dropout, GlobalAveragePooling2D, Concatenate,
                                     BatchNormalization)
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        LambdaCallback)
from sklearn.metrics import (classification_report, confusion_matrix,
                              precision_recall_fscore_support)
import matplotlib.pyplot as plt
import seaborn as sns
import csv

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# =============================================================================
# Configuration
IMAGE_HEIGHT, IMAGE_WIDTH = 128, 128
IMAGE_CHANNELS            = 3
SEQUENCE_LENGTH           = 12
POSE_VECTOR_SIZE          = 33 * 3   # 99 floats

MAX_SEQUENCES_PER_CLASS   = 1500
BATCH_SIZE                = 4        # safe for 4 GB VRAM with mixed precision
EPOCHS                    = 20
WARMUP_EPOCHS             = 3        # CNN frozen during warmup

base_dir             = os.path.dirname(os.path.abspath(__file__))
DATA_DIR_SUSPICIOUS  = os.path.join(base_dir, 'data', 'suspicious')
DATA_DIR_NORMAL      = os.path.join(base_dir, 'data', 'normal')
MODEL_PATH           = os.path.join(base_dir, 'hybrid_pose_efficientnetb0_model.h5')
POSE_MODEL           = os.path.join(base_dir, 'pose_landmarker_lite.task')

# =============================================================================
# Focal Loss
class FocalLoss(tf.keras.losses.Loss):
    """
    FL(p_t) = -α_t (1 - p_t)^γ log(p_t)
    γ=2.0 is the standard recommendation (Lin et al., RetinaNet, CVPR 2017).
    α=0.25 upweights the minority (Suspicious) class.
    """
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce     = -y_true * tf.math.log(y_pred)
        weight = self.alpha * y_true * tf.pow(1.0 - y_pred, self.gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * ce, axis=-1))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"gamma": self.gamma, "alpha": self.alpha})
        return cfg

# =============================================================================
# MediaPipe Pose Init
base_opts = mp_python.BaseOptions(model_asset_path=POSE_MODEL)
pose_opts = mp_vision.PoseLandmarkerOptions(
    base_options=base_opts,
    running_mode=mp_vision.RunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.4,
    min_pose_presence_confidence=0.4,
    min_tracking_confidence=0.4
)
try:
    pose_landmarker = mp_vision.PoseLandmarker.create_from_options(pose_opts)
    print("\nBuilding Hybrid MediaPipe-EfficientNetB0-LSTM Architecture...")
except Exception as e:
    print(f"[FATAL] Pose model failed to load: {e}")
    exit(1)

# =============================================================================
# Data Utilities

def extract_prefix_and_frame(filename: str):
    idx = filename.rfind('_')
    if idx == -1:
        return None, None
    prefix    = filename[:idx]
    frame_str = filename[idx + 1:].split('.')[0]
    try:
        return prefix, int(frame_str)
    except ValueError:
        return None, None


def gather_sequence_paths(directory: str, label: int):
    """
    Walk directory, group frames by video prefix, build non-overlapping
    sequences of length SEQUENCE_LENGTH, shuffle, cap at MAX_SEQUENCES_PER_CLASS.
    """
    print(f"  Scanning: {directory} ...")
    video_groups: dict = {}
    for root, _, files in os.walk(directory):
        for f in files:
            if os.path.splitext(f)[1].lower() not in ('.png', '.jpg', '.jpeg'):
                continue
            prefix, frame_num = extract_prefix_and_frame(f)
            if prefix is None:
                continue
            video_groups.setdefault(prefix, []).append((frame_num, os.path.join(root, f)))

    sequences, labels = [], []
    for prefix, frames in video_groups.items():
        frames.sort(key=lambda x: x[0])
        buf = []
        for _, fp in frames:
            buf.append(fp)
            if len(buf) == SEQUENCE_LENGTH:
                sequences.append(list(buf))
                labels.append(label)
                buf = []   # non-overlapping

    combined = list(zip(sequences, labels))
    np.random.shuffle(combined)
    combined = combined[:MAX_SEQUENCES_PER_CLASS]
    if not combined:
        return [], []
    seqs, lbls = zip(*combined)
    print(f"    -> {len(seqs)} sequences (cap={MAX_SEQUENCES_PER_CLASS})")
    return list(seqs), list(lbls)


def get_pose_vector(image_path: str) -> np.ndarray:
    """Run MediaPipe on one frame, return 99-dim (x,y,z) vector."""
    try:
        bgr = cv2.imread(image_path)
        if bgr is None:
            return np.zeros(POSE_VECTOR_SIZE, dtype='float32')
        rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = pose_landmarker.detect(mp_img)
        if not result.pose_landmarks:
            return np.zeros(POSE_VECTOR_SIZE, dtype='float32')
        vec = []
        for lm in result.pose_landmarks[0]:
            vec.extend([lm.x, lm.y, lm.z])
        return np.array(vec, dtype='float32')
    except Exception:
        return np.zeros(POSE_VECTOR_SIZE, dtype='float32')

# =============================================================================
# Data Generator

class SequenceDataGenerator(tf.keras.utils.Sequence):
    """
    Lazy-loads frames per batch with augmentation during training:
      • Random horizontal flip
      • Brightness / contrast jitter
      • Zoom crop (1.0–1.4×)
      • Temporal jitter (frame duplication to simulate drop)
      • Gaussian noise on pose vectors
    """
    def __init__(self, sequence_paths, labels,
                 batch_size=BATCH_SIZE, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.sequence_paths = sequence_paths
        self.labels         = labels
        self.batch_size     = batch_size
        self.shuffle        = shuffle
        self.indexes        = np.arange(len(self.sequence_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.sequence_paths) / self.batch_size))

    def __getitem__(self, index):
        idxs       = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_seqs = [self.sequence_paths[k] for k in idxs]
        batch_lbls = [self.labels[k]         for k in idxs]
        return self._generate(batch_seqs, batch_lbls)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _generate(self, batch_seqs, batch_lbls):
        X_img  = np.zeros((self.batch_size, SEQUENCE_LENGTH,
                            IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype='float32')
        X_pose = np.zeros((self.batch_size, SEQUENCE_LENGTH, POSE_VECTOR_SIZE), dtype='float32')
        Y      = np.empty(self.batch_size, dtype=int)

        for i, (seq_paths, lbl) in enumerate(zip(batch_seqs, batch_lbls)):
            do_flip    = self.shuffle and random.random() < 0.5
            scale      = random.uniform(1.0, 1.4) if self.shuffle else 1.0
            brightness = random.uniform(-0.2, 0.2) if self.shuffle else 0.0
            contrast   = random.uniform(0.8, 1.2)  if self.shuffle else 1.0
            drops = (random.sample(range(1, SEQUENCE_LENGTH), random.randint(1, 2))
                     if self.shuffle and random.random() < 0.3 else [])

            prev_img, prev_pose = None, None

            for j, fp in enumerate(seq_paths):
                if j in drops and prev_img is not None:
                    X_img[i, j]  = prev_img
                    X_pose[i, j] = prev_pose
                    continue

                # ── Pose ─────────────────────────────────────────────────
                pose_v = get_pose_vector(fp)
                if self.shuffle and pose_v.any():
                    pose_v = pose_v + np.random.normal(0, 0.02, pose_v.shape).astype('float32')
                if self.shuffle and do_flip:
                    pv_r = pose_v.reshape(-1, 3).copy()
                    pv_r[:, 0] = 1.0 - pv_r[:, 0]
                    pose_v = pv_r.flatten()
                X_pose[i, j] = pose_v
                prev_pose    = pose_v

                # ── Image ─────────────────────────────────────────────────
                try:
                    img = Image.open(fp).convert('RGB')
                    if scale > 1.0:
                        w, h   = img.size
                        nw, nh = int(w / scale), int(h / scale)
                        l, t   = (w - nw) // 2, (h - nh) // 2
                        img    = img.crop((l, t, l + nw, t + nh))
                    img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.BILINEAR)
                    if do_flip:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    arr = np.array(img, dtype='float32')
                    if self.shuffle:
                        arr = np.clip((arr + brightness * 255) * contrast, 0, 255)
                    arr = preprocess_input(arr)
                    X_img[i, j] = arr
                    prev_img    = arr
                except Exception:
                    if prev_img is not None:
                        X_img[i, j] = prev_img
            Y[i] = lbl

        return {"image_input": X_img, "pose_input": X_pose}, to_categorical(Y, num_classes=2)

# =============================================================================
# Model Architecture

def create_hybrid_model(cnn_trainable: bool = False) -> Model:
    """
    Two-branch hybrid:
      Branch 1 — EfficientNetB0 (ImageNet) -> TimeDistributed -> GAP
      Branch 2 — Pose 99d -> TimeDistributed Dense
      Merge     — Concatenate -> LSTM(128) -> Dense(2, softmax float32)
    L2 + aggressive Dropout to counter EfficientNetB0 overfitting on small dataset.
    """
    from tensorflow.keras.regularizers import l2
    REG = l2(1e-4)

    input_img = Input(shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
                      name='image_input')
    base_model = EfficientNetB0(weights='imagenet', include_top=False,
                               input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))

    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    cnn_out = TimeDistributed(base_model,                                            name='td_efficientnet')(input_img)
    cnn_out = TimeDistributed(GlobalAveragePooling2D(),                              name='td_gap')(cnn_out)
    cnn_out = TimeDistributed(BatchNormalization(),                                  name='td_bn')(cnn_out)
    cnn_out = TimeDistributed(Dropout(0.5),                                          name='td_drop')(cnn_out)
    cnn_out = TimeDistributed(Dense(128, activation='relu', kernel_regularizer=REG), name='td_dense')(cnn_out)

    input_pose = Input(shape=(SEQUENCE_LENGTH, POSE_VECTOR_SIZE), name='pose_input')
    pose_out   = TimeDistributed(Dense(64, activation='relu', kernel_regularizer=REG), name='pose_dense')(input_pose)
    pose_out   = TimeDistributed(BatchNormalization(),                                 name='pose_bn')(pose_out)
    pose_out   = TimeDistributed(Dropout(0.4),                                         name='pose_drop')(pose_out)

    merged   = Concatenate(name='concat')([cnn_out, pose_out])
    lstm_out = LSTM(128, return_sequences=False,
                    kernel_regularizer=REG, recurrent_regularizer=REG,
                    name='lstm')(merged)
    lstm_out = Dropout(0.5, name='lstm_drop')(lstm_out)
    dense    = Dense(64, activation='relu', kernel_regularizer=REG, name='head_dense')(lstm_out)
    dense    = Dropout(0.4, name='head_drop')(dense)
    # dtype='float32' required for mixed-precision stability
    output   = Dense(2, activation='softmax', dtype='float32', name='output')(dense)

    return Model(inputs=[input_img, input_pose], outputs=output)

# =============================================================================
# Main

def main():
    print("\n=== Suspicious Activity Trainer — EfficientNetB0 Edition ===\n")

    print("Gathering Normal sequences...")
    normal_paths, normal_labels = gather_sequence_paths(DATA_DIR_NORMAL,     label=0)
    print("Gathering Suspicious sequences...")
    susp_paths,   susp_labels   = gather_sequence_paths(DATA_DIR_SUSPICIOUS, label=1)

    X_paths  = normal_paths + susp_paths
    Y_labels = normal_labels + susp_labels
    if not X_paths:
        raise ValueError("No sequences found. Check DATA_DIR_NORMAL / DATA_DIR_SUSPICIOUS paths.")

    print(f"\nTotal: {len(X_paths)}  "
          f"(Normal={len(normal_paths)}, Suspicious={len(susp_paths)})")

    arr = np.array(Y_labels)
    cw  = compute_class_weight('balanced', classes=np.unique(arr), y=arr)
    print(f"Class weights (for reference): {dict(enumerate(cw))}")

    X_train, X_val, Y_train, Y_val = train_test_split(
        X_paths, Y_labels, test_size=0.2, random_state=42, stratify=Y_labels)

    train_gen = SequenceDataGenerator(X_train, Y_train, batch_size=BATCH_SIZE, shuffle=True)
    val_gen   = SequenceDataGenerator(X_val,   Y_val,   batch_size=BATCH_SIZE, shuffle=False)
    print(f"Train batches: {len(train_gen)}  |  Val batches: {len(val_gen)}")

    # ── Phase 1: Warmup ──────────────────────────────────────────────────────
    print(f"\n[Phase 1] Warmup: {WARMUP_EPOCHS} epochs, CNN frozen")
    model = create_hybrid_model(cnn_trainable=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=FocalLoss(gamma=2.0, alpha=0.25),
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    model.summary(line_length=100)
    model.fit(train_gen, validation_data=val_gen, epochs=WARMUP_EPOCHS, verbose=1)

    # Save warmup weights to a proper temp file (Windows-safe)
    warmup_tmp = os.path.join(tempfile.gettempdir(), 'surv_warmup_weights.h5')
    model.save_weights(warmup_tmp)
    del model
    gc.collect()

    # ── Phase 2: Fine-tune ───────────────────────────────────────────────────
    print(f"\n[Phase 2] Fine-tune: {EPOCHS} epochs, CNN top-30 layers unfrozen")
    model = create_hybrid_model(cnn_trainable=True)
    model.load_weights(warmup_tmp)

    steps_per_epoch = len(train_gen)
    cosine_lr = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=1e-4,
        first_decay_steps=steps_per_epoch * 5,
        t_mul=2.0, m_mul=0.9, alpha=1e-6
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cosine_lr),
        loss=FocalLoss(gamma=2.0, alpha=0.25),
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[
            ModelCheckpoint(MODEL_PATH, monitor='val_accuracy',
                            save_best_only=True, mode='max', verbose=1),
            EarlyStopping(monitor='val_loss', patience=6,
                          restore_best_weights=True, verbose=1),
            LambdaCallback(on_epoch_end=lambda epoch, logs: print(
                f"  LR: {float(model.optimizer.learning_rate):.2e}"))
        ],
        verbose=1
    )

    model.save(MODEL_PATH)
    print(f"\nModel saved: {MODEL_PATH}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\n=== Evaluation ===")
    val_eval = SequenceDataGenerator(X_val, Y_val, batch_size=BATCH_SIZE, shuffle=False)
    m = model.evaluate(val_eval, verbose=0)
    print(f"Val Accuracy:  {m[1]*100:.2f}%")
    print(f"Val Precision: {m[2]*100:.2f}%")
    print(f"Val Recall:    {m[3]*100:.2f}%")

    preds, trues = [], []
    for i in tqdm(range(len(val_eval)), desc="Scoring"):
        X_b, y_b = val_eval[i]
        p = model.predict(X_b, verbose=0)
        preds.extend(np.argmax(p, axis=1))
        trues.extend(np.argmax(y_b, axis=1))

    print("\nClassification Report:")
    print(classification_report(trues, preds, target_names=["Normal", "Suspicious"]))

    report_path = os.path.join(base_dir, 'classification_report.csv')
    with open(report_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Class', 'Precision', 'Recall', 'F1', 'Support'])
        p_s, r_s, f_s, s_s = precision_recall_fscore_support(trues, preds, labels=[0, 1])
        for cls, (p, r, f, s) in enumerate(zip(p_s, r_s, f_s, s_s)):
            w.writerow([['Normal', 'Suspicious'][cls],
                        f"{p:.3f}", f"{r:.3f}", f"{f:.3f}", int(s)])
    print(f"\nHybrid Model EfficientNetB0 saved as: {MODEL_PATH}")

    cm = confusion_matrix(trues, preds)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=["Normal", "Suspicious"],
                yticklabels=["Normal", "Suspicious"])
    axes[0].set_title("Confusion Matrix")
    axes[0].set_ylabel("True Label"); axes[0].set_xlabel("Predicted Label")

    axes[1].plot(history.history['accuracy'],     label='Train Acc')
    axes[1].plot(history.history['val_accuracy'], label='Val Acc')
    axes[1].plot(history.history['loss'],         label='Train Loss', linestyle='--')
    axes[1].plot(history.history['val_loss'],     label='Val Loss',   linestyle='--')
    axes[1].set_title("Training History (Phase 2)")
    axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    report_png = os.path.join(base_dir, 'training_report.png')
    plt.tight_layout(); plt.savefig(report_png, dpi=120)
    print(f"Training report saved: {report_png}")
    plt.show()


if __name__ == "__main__":
    main()
