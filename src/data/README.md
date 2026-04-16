# Dataset Directory

This directory is excluded from version control due to its size.

## Expected Structure

```
src/data/
├── normal/
│   ├── sequence_0001/
│   │   ├── frame_001.jpg
│   │   ├── frame_002.jpg
│   │   └── ...
│   └── sequence_XXXX/
└── suspicious/
    ├── sequence_0001/
    │   ├── frame_001.jpg
    │   └── ...
    └── sequence_XXXX/
```

## Source

The dataset used for training is sourced from Kaggle:
- [CCTV Anomaly Detection Dataset](https://www.kaggle.com/datasets)

Each class folder should contain subfolders, one per sequence, with sequential JPEG frames inside.

## Training

Once the data is placed correctly, run:
```bash
python src/trainer.py
```
