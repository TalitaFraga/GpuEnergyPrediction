import subprocess
import sys

batch_sizes = [8, 16, 32, 64]

for batch in batch_sizes:
    print(f"\nExecutando batch_size={batch}")

    subprocess.run([
        sys.executable,     
        "src/train_with_logger.py",
        "--batch_size", str(batch),
        "--epochs", "1",
        "--dataset_size", "1000",
        "--image_size", "64"
    ], check=True)

print("\nTodos os experimentos concluídos!")