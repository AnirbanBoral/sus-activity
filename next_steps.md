# Status Report & Next Steps for Suspicious Activity Detection

## Current Model: Deeper Custom CNN (VGG-style)
- **Architecture**: 3 blocks (filters 32 -> 64 -> 128), BatchNormalization, and Dropout (0.2 to 0.5).
- **Training State**: Pre-trained on 10,000 images (randomly sampled) with 5 epochs completed before interruption.
- **Data Augmentation**: Augmented with horizontal flips, small rotations (0.1), and zooms (0.1).

## Current Performance (manual_test.py)
- **Goal**: >= 90% accuracy across ALL categories.
- **Results**:
  - `Test_Abuse`: ~92.5% (PASS)
  - `Test_Arrest`: ~76.0% (FAIL - requires more distinctive data or training).
- **Video Detection (main.py)**:
  - Detects **Fighting** as suspicious successfully.
  - Still produces **False Positives** on high-motion normal actions (Walking/Running).

---

## Technical Recommendations for Next Model iteration

### 1. Hard-Negative Mining (The "Running/Walking" problem)
- **Problem**: The model sees high entropy/movement and flags it as suspicious.
- **Solution**: Explicitly add more images of "Normal Walking" and "Normal Running" into the `data/normal` training folder. This forces the CNN to learn that "fast movement" isn't always "fighting."

### 2. Temporal Processing (Moving from CNN to LSTM)
- **Problem**: Single frames don't tell the whole story. Fighting involves a sequence of specific poses.
- **Solution**: Implement an **LRCN (Long-term Recurrent Convolutional Network)**. Feed a sequence of frames (~15-30) into the CNN, then pass those features into an **LSTM** layer before classification.

### 3. Hyperparameter Adjustments
- **Learning Rate**: Use a lower initial rate (`5e-4` or `1e-4`) if using a pre-trained base.
- **Batch Size**: 64 works well for the RTX 3050 (4GB), but if you hit OOM (Out of Memory), drop it to 32.

---

## Workflow Commands
To resume training:
```powershell
.\.venv\Scripts\python.exe src/simple_cnn.py
```
To run the automated category tests:
```powershell
.\.venv\Scripts\python.exe src/manual_test.py
```
To run the GUI for testing videos:
```powershell
.\.venv\Scripts\python.exe src/main.py
```
