import torch
import subprocess
import numpy as np
import time
from cellpose import models

def print_nvidia_smi():
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    print(result.stdout)

print("=== GPU state before model load ===")
print_nvidia_smi()

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch version: {torch.__version__}")

# in cellpose 2.x, CellposeModel with model_type='nuclei' is correct
model = models.Cellpose(gpu=True, model_type='nuclei')
print(f"Model device: {next(model.cp.net.parameters()).device}")

print("=== GPU state after model load ===")
print_nvidia_smi()

# cellpose 2.x eval returns (masks, flows, styles, diams)
fake = np.random.randint(0, 1000, (5, 64, 64), dtype=np.uint16).astype(np.float32)
t = time.time()
masks, flows, styles, diams = model.eval(
    fake,
    do_3D=True,
    anisotropy=7.0,
    diameter=10,
    z_axis=0,
    channels=[0, 0],  # grayscale
)
elapsed = time.time() - t

print("=== GPU state after inference ===")
print_nvidia_smi()

print(f"Test inference took {elapsed:.1f}s")
print(f"Masks shape: {masks.shape}, unique labels: {masks.max()}")
print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")