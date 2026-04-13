import os
import gc
import cv2
import numpy as np
from PIL import Image
import random
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ── Configuration ────────────────────────────────────────────────────────────
IMAGE_HEIGHT, IMAGE_WIDTH = 128, 128  # Enhanced resolution for MobileNet visibility
SEQUENCE_LENGTH = 12
IMAGE_CHANNELS = 3

# Increased per-class limits to train on a robust dataset (5x more data)
MAX_SEQUENCES_PER_CLASS = 2500

base_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR_SUSPICIOUS = os.path.join(base_dir, 'data', 'suspicious')
DATA_DIR_NORMAL = os.path.join(base_dir, 'data', 'normal')
MODEL_PATH = os.path.join(base_dir, 'mobilenet_model.h5')

def extract_prefix_and_frame(filename):
    last_underscore_idx = filename.rfind('_')
    if last_underscore_idx == -1: return None, None
    prefix = filename[:last_underscore_idx]
    frame_str = filename[last_underscore_idx+1:].split('.')[0]
    try: return prefix, int(frame_str)
    except ValueError: return None, None

def gather_sequence_paths(directory, label):
    """
    Groups absolute paths of frames into sequences of 15.
    Memory efficient design streams paths instead of loading all images into RAM.
    """
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
                # Non-overlapping window to keep sequence count manageable
                current_sequence = []
    
    # Shuffle and cap to MAX_SEQUENCES_PER_CLASS
    combined = list(zip(sequences_paths, labels))
    np.random.shuffle(combined)
    combined = combined[:MAX_SEQUENCES_PER_CLASS]
    sequences_paths, labels = zip(*combined) if combined else ([], [])
    sequences_paths, labels = list(sequences_paths), list(labels)

    print(f"  Using {len(sequences_paths)} sequences (capped).")
    return sequences_paths, labels

class SequenceDataGenerator(tf.keras.utils.Sequence):
    """
    Loads sequences from disk directly into the GPU inside small batches during training.
    """
    def __init__(self, sequence_paths, labels, batch_size=8, shuffle=True, **kwargs):
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
        X = np.empty((self.batch_size, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype='float32')
        y = np.empty((self.batch_size), dtype=int)
        
        for i, seq_paths in enumerate(batch_seqs):
            # Apply consistent Spatial Scale Zoom across the whole sequence randomly
            scale_factor = random.uniform(1.0, 1.4) if self.shuffle else 1.0
            
            for j, filepath in enumerate(seq_paths):
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
                    X[i, j,] = img_array
                except Exception:
                    X[i, j,] = 0.0 # Zero-padding fallback to prevent batch crashing
            y[i] = batch_labels[i]
            
        return X, to_categorical(y, num_classes=2)

def create_mobilenet_lstm_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, 
                             input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
    
    # Domain Adaptation: Unfreeze the last 20 layers (roughly the final conv blocks of MobileNetV2)
    # so the CNN learns "surveillance" features instead of just generic "ImageNet" features.
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model = Sequential()
    # Apply MobileNet feature extraction
    model.add(TimeDistributed(base_model, input_shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)))
    # Flatten feature block
    model.add(TimeDistributed(GlobalAveragePooling2D()))
    model.add(TimeDistributed(Dropout(0.3)))
    
    # LSTM with increased capacity to 128 units to map complex temporal changes
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.5))
    
    # Final dense classification (0 = Normal, 1 = Suspicious)
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))
    
    # Lower learning rate (1e-4) because we are partially fine-tuning MobileNetV2
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model

def main():
    print("=== Mapping Normal Data Paths (Label 0) ===")
    normal_paths, normal_labels = gather_sequence_paths(DATA_DIR_NORMAL, label=0)
    
    print("\n=== Mapping Suspicious Data Paths (Label 1) ===")
    suspicious_paths, suspicious_labels = gather_sequence_paths(DATA_DIR_SUSPICIOUS, label=1)
    
    X_paths = normal_paths + suspicious_paths
    Y_labels = normal_labels + suspicious_labels
    del normal_paths, normal_labels, suspicious_paths, suspicious_labels
    gc.collect()
    
    if len(X_paths) == 0:
        raise ValueError("No valid sequential data found!")
        
    print(f"\nTotal Sequence Pathways Mapped: {len(X_paths)}") 
    
    # Split the PATHS, not the images
    X_train, X_test, Y_train, Y_test = train_test_split(X_paths, Y_labels, test_size=0.2, random_state=42, stratify=Y_labels)
    
    # Extremely small batch size to avoid GPU memory overflow with 128x128 images.
    train_generator = SequenceDataGenerator(X_train, Y_train, batch_size=4, shuffle=True)
    val_generator = SequenceDataGenerator(X_test, Y_test, batch_size=4, shuffle=False)
    
    print("\nBuilding MobileNetV2-LSTM Architecture...")
    model = create_mobilenet_lstm_model()
    
    # Add robust Early Stopping to combat overfitting on large data during fine-tuning
    early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    
    print("\nStarting Generator Training Phase...")
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=20, # More epochs + early stopping
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    model.save(MODEL_PATH)
    print(f"\nMobileNetV2-LSTM Model saved successfully as: {MODEL_PATH}")
    
    loss, acc = model.evaluate(val_generator, verbose=0)
    print(f"Final Validation Accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    main()
