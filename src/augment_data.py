"""
Data Augmentation Script for Sequence Data
This script creates artificial variations of your existing image sequences 
by applying coherent augmentations (blurring, dimming) across the entire sequence.
This helps the model learn that blurry or dark images can still be 'Normal'.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR_NORMAL = os.path.join(SCRIPT_DIR, 'data', 'normal')

def extract_prefix_and_frame(filename):
    """Extracts the base prefix and frame number from a file like 'vid1_0.jpg'"""
    last_underscore_idx = filename.rfind('_')
    if last_underscore_idx == -1: return None, None
    prefix = filename[:last_underscore_idx]
    frame_str = filename[last_underscore_idx+1:].split('.')[0]
    try: return prefix, int(frame_str), filename[last_underscore_idx+1:]
    except ValueError: return None, None, None

def get_sequences(directory):
    """Groups images into their logical video sequences"""
    print(f"Scanning {directory}...")
    video_groups = {}
    
    for root, _, files in os.walk(directory):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in ['.png', '.jpg', '.jpeg']:
                prefix, frame_num, suffix = extract_prefix_and_frame(f)
                if prefix is not None:
                    filepath = os.path.join(root, f)
                    if prefix not in video_groups: 
                        video_groups[prefix] = []
                    video_groups[prefix].append((frame_num, filepath, root, suffix))
    return video_groups

def augment_sequence(frames_data, aug_type="blur"):
    """
    Applies the EXACT same augmentation uniformly across all frames in a sequence 
    to prevent temporal jitter that would confuse the LSTM.
    """
    # Define coherent variables for the whole sequence
    blur_kernel = random.choice([(5,5), (7,7), (9,9)])
    brightness_factor = random.uniform(0.3, 0.7) if aug_type == "dim" else 1.0
    contrast_alpha = random.uniform(1.2, 1.5) if aug_type == "jitter" else 1.0
    contrast_beta = random.randint(-20, 20) if aug_type == "jitter" else 0
    
    for _, filepath, root, suffix in frames_data:
        img = cv2.imread(filepath)
        if img is None: continue
        
        # Apply augmentation
        if aug_type == "blur":
            aug_img = cv2.GaussianBlur(img, blur_kernel, 0)
            new_prefix = f"{os.path.basename(filepath).split('_')[0]}_augBlur"
        elif aug_type == "dim":
            # Simulate low-light/darkness
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv = np.array(hsv, dtype=np.float64)
            hsv[:, :, 2] = hsv[:, :, 2] * brightness_factor
            hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
            hsv = np.array(hsv, dtype=np.uint8)
            aug_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            new_prefix = f"{os.path.basename(filepath).split('_')[0]}_augDim"
        elif aug_type == "flip":
            aug_img = cv2.flip(img, 1) # Horizontal flip
            new_prefix = f"{os.path.basename(filepath).split('_')[0]}_augFlip"
        elif aug_type == "jitter":
            aug_img = cv2.convertScaleAbs(img, alpha=contrast_alpha, beta=contrast_beta)
            new_prefix = f"{os.path.basename(filepath).split('_')[0]}_augJitter"
        
        # Save back to disk
        new_filename = f"{new_prefix}_{suffix}"
        new_filepath = os.path.join(root, new_filename)
        cv2.imwrite(new_filepath, aug_img)

def main():
    print("=== Suspicious Activity Sequence Augmenter ===")
    
    if not os.path.exists(DATA_DIR_NORMAL):
        print(f"[ERROR] Directory not found: {DATA_DIR_NORMAL}")
        return

    # 1. Grab all Normal sequences
    normal_sequences = get_sequences(DATA_DIR_NORMAL)
    prefix_list = list(normal_sequences.keys())
    print(f"Found {len(prefix_list)} unique 'Normal' sequences.")
    
    # 2. Augment them cleanly
    print("Generating 'Blurry' variations to simulate failing tracking crops...")
    for prefix in tqdm(prefix_list, desc="Blurring Sequences"):
        augment_sequence(normal_sequences[prefix], aug_type="blur")
        
    print("Generating 'Dim' variations to simulate low-light CCTV...")
    for prefix in tqdm(prefix_list, desc="Dimming Sequences"):
        augment_sequence(normal_sequences[prefix], aug_type="dim")

    print("Generating 'Flipped' variations to increase spatial variance...")
    for prefix in tqdm(prefix_list, desc="Flipping Sequences"):
        augment_sequence(normal_sequences[prefix], aug_type="flip")

    print("Generating 'Jitter' variations for contrast/brightness...")
    for prefix in tqdm(prefix_list, desc="Jitter Sequences"):
        augment_sequence(normal_sequences[prefix], aug_type="jitter")

    print("\n[SUCCESS] Dataset massively bloated with coherent adversarial sequences.")
    print("Run trainer.py again to let the LSTM learn these new negative examples!")

if __name__ == "__main__":
    main()
