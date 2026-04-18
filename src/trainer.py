import os
import gc
import cv2
import numpy as np
from PIL import Image
import random
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, TimeDistributed, LSTM, Dense, Dropout, GlobalAveragePooling2D, Concatenate
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Configuration ────────────────────────────────────────────────────────────
IMAGE_HEIGHT, IMAGE_WIDTH = 128, 128
SEQUENCE_LENGTH = 12
IMAGE_CHANNELS = 3
POSE_VECTOR_SIZE = 33 * 3  # 33 landmarks, each (x, y, z)

MAX_SEQUENCES_PER_CLASS = 1500  # Cap aggressively so it doesn't OOM memory during feature extraction

base_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR_SUSPICIOUS = os.path.join(base_dir, 'data', 'suspicious')
DATA_DIR_NORMAL = os.path.join(base_dir, 'data', 'normal')
MODEL_PATH = os.path.join(base_dir, 'hybrid_pose_mobilenet_model.h5')
POSE_MODEL = os.path.join(base_dir, 'pose_landmarker_lite.task')

# ── Init MediaPipe Pose ──────────────────────────────────────────────────────
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
except Exception as e:
    print(f"[FATAL] Pose model failed to load. Is pose_landmarker_lite.task in {base_dir}?")
    print(e)
    exit(1)


def extract_prefix_and_frame(filename):
    last_underscore_idx = filename.rfind('_')
    if last_underscore_idx == -1: return None, None
    prefix = filename[:last_underscore_idx]
    frame_str = filename[last_underscore_idx+1:].split('.')[0]
    try: return prefix, int(frame_str)
    except ValueError: return None, None

def gather_sequence_paths(directory, label):
    print(f"Mapping sequence paths in: {directory}...")
    video_groups = {}
    
    for root, _, files in os.walk(directory):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in ['.png', '.jpg', '.jpeg']:
                prefix, frame_num = extract_prefix_and_frame(f)
                if prefix is not None:
                    filepath = os.path.join(root, f)
                    if prefix not in video_groups: video_groups[prefix] = []
                    video_groups[prefix].append((frame_num, filepath))
                    
    sequences_paths = []
    labels = []
    
    for prefix, frames in video_groups.items():
        frames.sort(key=lambda x: x[0])
        current_sequence = []
        for _, filepath in frames:
            current_sequence.append(filepath)
            if len(current_sequence) == SEQUENCE_LENGTH:
                sequences_paths.append(list(current_sequence))
                labels.append(label)
                current_sequence = []
    
    combined = list(zip(sequences_paths, labels))
    np.random.shuffle(combined)
    combined = combined[:MAX_SEQUENCES_PER_CLASS]
    sequences_paths, labels = zip(*combined) if combined else ([], [])
    sequences_paths, labels = list(sequences_paths), list(labels)
    print(f"  Using {len(sequences_paths)} sequences (capped).")
    return sequences_paths, labels

def get_pose_vector(image_path):
    """ Reads image, runs mediapipe pose, and flattens the 33 landmarks into a 99-dim vector. """
    try:
        bgr = cv2.imread(image_path)
        if bgr is None:
            return np.zeros((POSE_VECTOR_SIZE,), dtype='float32')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = pose_landmarker.detect(mp_img)
        if not result.pose_landmarks:
            return np.zeros((POSE_VECTOR_SIZE,), dtype='float32')
        
        lm = result.pose_landmarks[0]
        # flatten 33 landmarks (x,y,z)
        vec = []
        for landmark in lm:
            vec.extend([landmark.x, landmark.y, landmark.z])
        return np.array(vec, dtype='float32')
    except:
        return np.zeros((POSE_VECTOR_SIZE,), dtype='float32')

class SequenceDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, sequence_paths, labels, batch_size=4, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.sequence_paths = sequence_paths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.sequence_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
    def __len__(self):
        return int(np.floor(len(self.sequence_paths) / self.batch_size))
        
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_seqs = [self.sequence_paths[k] for k in indexes]
        batch_labels = [self.labels[k] for k in indexes]
        X, y = self.__data_generation(batch_seqs, batch_labels)
        return X, y
        
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
    def __data_generation(self, batch_seqs, batch_labels):
        X_img = np.empty((self.batch_size, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype='float32')
        X_pose = np.empty((self.batch_size, SEQUENCE_LENGTH, POSE_VECTOR_SIZE), dtype='float32')
        y = np.empty((self.batch_size), dtype=int)
        
        for i, seq_paths in enumerate(batch_seqs):
            scale_factor = random.uniform(1.0, 1.4) if self.shuffle else 1.0
            
            # Temporal Jitter: occasionally drop 1-2 frames to simulate lag
            temporal_drops = []
            if self.shuffle and random.random() < 0.3:
                num_drops = random.randint(1, 2)
                temporal_drops = random.sample(range(1, SEQUENCE_LENGTH), num_drops)
            
            for j, filepath in enumerate(seq_paths):
                # Apply Temporal Jitter by duplicating previous frame
                if j in temporal_drops:
                    X_pose[i, j, :] = X_pose[i, j-1, :]
                    X_img[i, j,] = X_img[i, j-1,]
                    continue
                    
                # 1. Pose Vector
                pose_v = get_pose_vector(filepath)
                # Skeleton Gaussian Noise
                if self.shuffle and np.any(pose_v):
                    noise = np.random.normal(0, 0.02, pose_v.shape)
                    pose_v = pose_v + noise
                X_pose[i, j, :] = pose_v
                
                # 2. Image Array
                try:
                    img = Image.open(filepath).convert('RGB')
                    if scale_factor > 1.0:
                        w, h = img.size
                        new_w, new_h = int(w / scale_factor), int(h / scale_factor)
                        left, top = (w - new_w) // 2, (h - new_h) // 2
                        img = img.crop((left, top, left + new_w, top + new_h))
                    
                    img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
                    img_array = np.array(img, dtype='float32')
                    img_array = preprocess_input(img_array)
                    X_img[i, j,] = img_array
                except Exception:
                    X_img[i, j,] = 0.0
            y[i] = batch_labels[i]
        return {"image_input": X_img, "pose_input": X_pose}, to_categorical(y, num_classes=2)

def create_hybrid_model():
    # Model 1: Image Branch
    input_img = Input(shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), name='image_input')
    base_model = MobileNetV2(weights='imagenet', include_top=False, 
                             input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
    
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    cnn_out = TimeDistributed(base_model)(input_img)
    cnn_out = TimeDistributed(GlobalAveragePooling2D())(cnn_out)
    cnn_out = TimeDistributed(Dropout(0.3))(cnn_out)
    cnn_out = TimeDistributed(Dense(128, activation='relu'))(cnn_out)
    
    # Model 2: Pose Branch
    input_pose = Input(shape=(SEQUENCE_LENGTH, POSE_VECTOR_SIZE), name='pose_input')
    pose_out = TimeDistributed(Dense(64, activation='relu'))(input_pose)
    pose_out = TimeDistributed(Dropout(0.2))(pose_out)
    
    # Concatenate Branches
    concat = Concatenate()([cnn_out, pose_out])
    
    # Temporal LSTM
    lstm_out = LSTM(128, return_sequences=False)(concat)
    lstm_out = Dropout(0.5)(lstm_out)
    
    # Classification Head
    dense_out = Dense(64, activation='relu')(lstm_out)
    dense_out = Dropout(0.2)(dense_out)
    output = Dense(2, activation='softmax')(dense_out)
    
    model = Model(inputs=[input_img, input_pose], outputs=output)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model

def main():
    print("=== Mapping Normal Data Paths ===")
    normal_paths, normal_labels = gather_sequence_paths(DATA_DIR_NORMAL, label=0)
    
    print("\n=== Mapping Suspicious Data Paths ===")
    suspicious_paths, suspicious_labels = gather_sequence_paths(DATA_DIR_SUSPICIOUS, label=1)
    
    X_paths = normal_paths + suspicious_paths
    Y_labels = normal_labels + suspicious_labels
    
    if len(X_paths) == 0:
        raise ValueError("No valid sequential data found!")
        
    print(f"\nTotal Sequence Pathways Mapped: {len(X_paths)}") 
    
    # Split the PATHS, not the images
    X_train, X_test, Y_train, Y_test = train_test_split(X_paths, Y_labels, test_size=0.2, random_state=42, stratify=Y_labels)
    
    # Extremely small batch size to avoid GPU memory overflow with 128x128 images.
    train_generator = SequenceDataGenerator(X_train, Y_train, batch_size=4, shuffle=True)
    val_generator = SequenceDataGenerator(X_test, Y_test, batch_size=4, shuffle=False)
    
    print("\nBuilding Hybrid MediaPipe-MobileNet-LSTM Architecture...")
    model = create_hybrid_model()
    model.summary()
    
    # Callbacks
    checkpoint_path = os.path.join(base_dir, 'hybrid_pose_mobilenet_model_v2.h5')
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)

    print("\nStarting Multimodal Training Phase...")
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=15,
        callbacks=[checkpoint, reduce_lr, early_stop]
    )
    
    model.save(checkpoint_path)
    print(f"\nHybrid Model V2 saved successfully as: {checkpoint_path}")
    
    # Unpack evaluate carefully (returns [loss, accuracy, precision, recall])
    metrics = model.evaluate(val_generator, verbose=0)
    print(f"Final Validation Accuracy: {metrics[1] * 100:.2f}%")
    
    print("\n=== Generating Classification Report and Confusion Matrix ===")
    val_generator = SequenceDataGenerator(X_test, Y_test, batch_size=4, shuffle=False)
    predictions = []
    true_labels = []
    print("Evaluating Test Set...")
    for i in tqdm(range(len(val_generator)), desc="Scoring"):
        X_batch, y_batch = val_generator[i]
        preds = model.predict(X_batch, verbose=0)
        predictions.extend(np.argmax(preds, axis=1))
        true_labels.extend(np.argmax(y_batch, axis=1))
        
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=["Normal", "Suspicious"]))
    
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal", "Suspicious"], yticklabels=["Normal", "Suspicious"])
    plt.title("Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = os.path.join(base_dir, 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(cm_path)
    print(f"Saved Confusion Matrix to: {cm_path}")

if __name__ == "__main__":
    main()
