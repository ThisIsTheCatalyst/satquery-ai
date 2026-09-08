# SatQuery AI — Google Colab Guide

## Target Hardware

- **Minimum**: T4 GPU (16GB VRAM), 12GB RAM — use `configs/colab_8gb.yaml`
- **Comfortable**: A100 GPU (40GB VRAM), 25GB RAM — use `configs/colab_16gb.yaml`

The siamese-difference change detector and fusion architecture check work on CPU.
GeoChat VQA needs GPU for reasonable speed.

## Setup (copy into first Colab cell)

```python
# Cell 1: Clone and install
!git clone https://github.com/your-repo/satquery-ai
%cd satquery-ai

!pip install -r requirements-colab.txt -q

# Verify
!python scripts/check_environment.py
```

```python
# Cell 2: Check GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
      if torch.cuda.is_available() else "CPU only")
```

```python
# Cell 3: Smoke test (no data needed)
!python scripts/smoke_test.py --no-load
```

## Download Minimal Data (for demo)

```python
# Download a small BigEarthNet subset for fusion training
import os
os.makedirs("data/raw/bigearthnet", exist_ok=True)

# Option A: From BigEarthNet website (requires registration)
# !wget -P data/raw/ https://bigearth.net/downloads/BigEarthNet-S2-v1.0.tar.gz

# Option B: Use synthetic data for architecture demo
!python -c "
import numpy as np, os, json
from PIL import Image

# Create synthetic BigEarthNet-like data
os.makedirs('data/raw/bigearthnet/images', exist_ok=True)
with open('data/raw/bigearthnet/BigEarthNet.txt', 'w') as f:
    for i in range(100):
        labels = np.random.choice(['Forest', 'Urban fabric', 'Water bodies', 'Pastures'], 2)
        # Create dummy 4-band 64x64 image
        arr = np.random.randint(0, 3000, (4, 64, 64), dtype=np.uint16)
        np.save(f'data/raw/bigearthnet/images/patch_{i:04d}.npy', arr)
        f.write(f'patch_{i:04d}\t{\",\".join(labels)}\n')
print('Synthetic data created.')
"
```

## Train Fusion Model

```python
# Full training (takes ~10-15 min on T4)
!python scripts/train_fusion.py --config configs/colab_8gb.yaml

# Quick smoke run (1 epoch, synthetic data)
!python scripts/train_fusion.py --config configs/colab_8gb.yaml --max-samples 50
```

## Run the Demo

```python
import numpy as np, yaml
from src.agent.controller import AgentController

with open("configs/colab_8gb.yaml") as f:
    config = yaml.safe_load(f)

controller = AgentController(config, device="cuda" if torch.cuda.is_available() else "cpu")

# VQA demo
image = np.random.rand(4, 64, 64).astype("float32")
result = controller.run({"image": image}, "What is the dominant land cover?")
print("Task:", result["task"])
print("Answer:", result["text"])
print("Confidence:", result["confidence"])
print("Model used:", result["metadata"]["parameters"].get("fallback_level", "unknown"))
```

## Memory Management

If you hit OOM:
```python
# Free GPU memory between models
import torch, gc
torch.cuda.empty_cache()
gc.collect()
```

Use gradient checkpointing for GeoChat training:
```yaml
# In colab_8gb.yaml
training:
  gradient_checkpointing: true
  batch_size: 1
  gradient_accumulation_steps: 8
```

## Common Issues

| Error | Fix |
|-------|-----|
| `CUDA OOM` | Use `configs/colab_8gb.yaml`, reduce `batch_size` to 1 |
| `rasterio not found` | `!pip install rasterio` |
| `No module named peft` | `!pip install peft accelerate` |
| `GeoChat checkpoint not found` | VQA falls back to CLIP/baseline automatically |
| `VisTA not found` | Change falls back to siamese-difference automatically |
