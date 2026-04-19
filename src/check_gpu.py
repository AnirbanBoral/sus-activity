import tensorflow as tf
import torch
import os

print("="*50)
print("  HARDWARE ACCELERATION DIAGNOSTIC")
print("="*50)

# TensorFlow Check
gpus = tf.config.list_physical_devices('GPU')
print(f"TensorFlow Version: {tf.__version__}")
print(f"TF GPU Found: {len(gpus) > 0}")
for gpu in gpus:
    print(f"  - {gpu.name}")

if len(gpus) == 0:
    print("\n[!] WARNING: TensorFlow is NOT using your GPU.")
    print("    On Windows, for TF > 2.10, you must use WSL2 or downgrade to 2.10.0.")

# PyTorch Check
print("\n" + "="*50)
print(f"PyTorch Version: {torch.__version__}")
print(f"Torch CUDA Found: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  - Device: {torch.cuda.get_device_name(0)}")
    print(f"  - CUDA Version: {torch.version.cuda}")

print("="*50)
