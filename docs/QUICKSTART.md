# SatQuery AI — Quickstart Guide

## Prerequisites

- Python 3.9+
- Git
- 8 GB+ RAM (GPU recommended for VQA/captioning; change detection and fusion CPU-capable)

## 1. Clone and Install

```bash
git clone <your-repo-url> satquery-ai
cd satquery-ai
pip install -r requirements.txt
```

For Google Colab (minimal install):
```bash
pip install -r requirements-colab.txt
```

## 2. Verify Environment

```bash
python scripts/check_environment.py
```

## 3. Run Smoke Test (No Data Needed)

```bash
# Schema/routing/preprocessing checks + model baseline checks
python scripts/smoke_test.py --no-load

# Full model checks (loads fallback models)
python scripts/smoke_test.py
```

## 4. Run the Agent (Demo)

```python
import numpy as np
import yaml
from src.agent.controller import AgentController

with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

controller = AgentController(config, device="cpu")

# Single-image VQA
image = np.random.rand(4, 64, 64).astype("float32")  # replace with real image
result = controller.run(
    images={"image": image},
    query="What is the dominant land cover type?"
)
print(result["text"])
print(result["trace"])
```

## 5. Train the Fusion Model

```bash
# Small subset for Colab 8GB
python scripts/train_fusion.py --config configs/colab_8gb.yaml

# Full dataset on local GPU
python scripts/train_fusion.py --config configs/local_gpu.yaml
```

## 6. Train VQA (requires GeoChat)

```bash
python scripts/train_vqa.py --config configs/colab_8gb.yaml
```

## 7. Evaluate

```bash
# VQA on RSVQA (requires downloaded benchmark)
python scripts/evaluate.py --task vqa --dataset rsvqa --write-results

# Change on CDVQA
python scripts/evaluate.py --task change --dataset cdvqa --write-results

# Generate evaluation report
python -m src.evaluation.generate_report
```

## 8. FastAPI Backend

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Test:
curl -X POST http://localhost:8000/health
```

## Colab Notebooks

Open the notebooks in order:
1. `notebooks/00_setup_and_environment.ipynb` — Install everything
2. `notebooks/01_data_download_and_validation.ipynb` — Download datasets
3. `notebooks/02_remote_sensing_adaptation.ipynb` — Adapt model to RS domain
4. `notebooks/SatQuery_AI_Master_Colab.ipynb` — Full demo in one notebook

## Troubleshooting

**OOM on GPU**: Use `configs/colab_8gb.yaml` (batch_size=2, image_size=224, fp16=True).

**rasterio not found**: `pip install rasterio` or `conda install rasterio`.

**GeoChat not available**: The VQA and captioning models automatically fall back to
CLIP-based and keyword-statistics baselines. All results still pass the RSModelResult schema.

**VisTA not available**: Change detection automatically uses the siamese-difference baseline.
This baseline is fully functional and returns valid change masks and descriptions.

**"No module named torch"**: Install PyTorch first:
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
