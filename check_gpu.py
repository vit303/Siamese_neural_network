import tensorflow as tf
import torch
import subprocess
import sys

print("=== Информация о системе ===")
print(f"Python version: {sys.version}")
print(f"TensorFlow version: {tf.__version__}")

print("\n=== Проверка GPU ===")
print("TensorFlow devices:")
print(tf.config.list_physical_devices())

print("\nPyTorch проверка:")
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"PyTorch GPU count: {torch.cuda.device_count()}")
    print(f"PyTorch GPU name: {torch.cuda.get_device_name(0)}")

print("\n=== NVIDIA информация ===")
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("nvidia-smi работает:")
        print(result.stdout[:500])  # Первые 500 символов
    else:
        print("nvidia-smi не доступен")
except FileNotFoundError:
    print("nvidia-smi не установлен")

print("\n=== Драйверы CUDA ===")
try:
    result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
    if result.returncode == 0:
        print("NVCC version:")
        print(result.stdout)
    else:
        print("NVCC не доступен")
except FileNotFoundError:
    print("NVCC не установлен")