# Hybrid Multimodal Suspicious Activity Detection System

A high-performance computer vision pipeline designed to detect violent or suspicious actions (e.g., fighting, lunging, striking) in real-time camera feeds. This system uses a multi-layered approach combining spatial tracking, geometric pose estimation, and temporal sequence modeling.

## 🧠 System Architecture

The pipeline leverages a **Hybrid YOLOv8 + MediaPipe + LSTM** architecture for maximum accuracy and real-world robustness:
*   **Stage 1: Spatial Tracker (YOLOv8)**: Handles global object detection and person tracking.
*   **Stage 2: Geometric Layer (MediaPipe)**: Extracts 33 skeletal keypoints (99-dim vector), converting raw pixels into a geometric manifold that is invariant to lighting, clothing, or background changes.
*   **Stage 3: Intent Engine (LSTM)**: Analyzes a rolling 12-frame window of skeleton data to identify temporal patterns of suspicious behavior.
*   **Stage 4: Geometric Rule Engine**: Calculates joint velocities and strike vectors for redundant safety-overrides.

## 🚀 Key Results
- **Training Accuracy:** 97.32%
- **Final Test Accuracy:** 89.00%
- **Suspicious Action Precision:** 93.00%
- **Adversarial Robustness:** Trained on over 100,000 augmented samples (Blur, Dim, Jitter) to ensure performance in poor CCTV conditions.

## 🛠 Usage

### 1. Initialize Environment
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Real-Time Demo
```bash
# Run on a test video
python src/main.py "manual_test/fight.avi"

# Run on live webcam
python src/main.py 0
```

## 📊 Evaluation
The final confusion matrix and classification report are generated automatically during training and can be found in the `src/` directory as `confusion_matrix.png`.

---
*Note: The primary 11GB training dataset and augmented image folders are excluded from this repository via .gitignore due to size constraints.*
