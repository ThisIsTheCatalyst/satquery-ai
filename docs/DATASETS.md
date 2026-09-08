# SatQuery AI — Dataset Guide

## Dataset Roles

**IMPORTANT**: Training on evaluation datasets is prevented by a machine-checked guardrail
(`src/common/constants.py::assert_usable_for_training`). This raises immediately if a
training DataLoader is requested for an eval-only dataset.

## Training Datasets

### BigEarthNet.txt
- **Purpose**: Remote-sensing domain adaptation for GeoChat VQA/captioning
- **Format**: Tab-separated `<patch_id>\t<label1,label2,...>` or comma-separated
- **Images**: Sentinel-2 GeoTIFF patches (120×120 pixels, 12 bands)
- **Labels**: Multi-label CORINE land-cover classes
- **Download**: https://bigearth.net/
- **Config path**: `datasets.bigearthnet.labels_file`, `datasets.bigearthnet.image_dir`

### BigEarthNet-MM
- **Purpose**: Optical-SAR fusion training
- **Format**: Paired Sentinel-2 (optical) + Sentinel-1 (SAR) patches
- **Labels**: Multi-label CORINE classes
- **Download**: https://bigearth.net/
- **Config path**: `datasets.bigearthnet_mm.root`

## Evaluation Datasets

### VRSBench
- **Tasks**: VQA, captioning, grounding
- **Format**: JSON with image paths, questions/answers/bboxes
- **Config**: `datasets.vrsbench.root`
- **Note**: eval-only — training split requests raise immediately

### RSVQA (Low-Resolution)
- **Task**: VQA only
- **Format**: JSON with question/answer pairs and question types
- **Config**: `datasets.rsvqa.root`

### CDVQA
- **Task**: Change-VQA (bi-temporal)
- **Format**: JSON with T1/T2 image paths, question/answer, optional mask
- **Config**: `datasets.cdvqa.root`

### ISRO/SAC (Hidden Judging Set)
- **Task**: All tasks (Cartosat-2S optical + RISAT SAR)
- **Access**: Only during judging — NOT available for development
- **Sensor adapters**: `sensor_adapters/cartosat2s.py`, `sensor_adapters/risat.py` (STUBS)

## Dataset Setup

```bash
# Download BigEarthNet (optical only, ~66 GB)
wget https://bigearth.net/downloads/BigEarthNet-S2-v1.0.tar.gz
tar -xf BigEarthNet-S2-v1.0.tar.gz -C data/raw/bigearthnet/images/

# BigEarthNet-MM (optical + SAR, ~120 GB)
wget https://bigearth.net/downloads/BigEarthNet-MM-v1.0.tar.gz
tar -xf BigEarthNet-MM-v1.0.tar.gz -C data/raw/bigearthnet_mm/
```

For Colab, use only a subset (see `configs/colab_8gb.yaml`):
```yaml
datasets:
  bigearthnet:
    max_samples: 5000
```

## Annotation Schema Assumptions

Each dataset class documents its assumed annotation format. If real files differ,
the loader will fail loudly with a `FileNotFoundError` or descriptive `KeyError`.
See each class's docstring in `src/preprocessing/dataset_loader.py`.

## Evaluation Results

Evaluation JSONs are written to `outputs/eval_results/` with schema:
```json
{
    "model": "GeoChat",
    "dataset": "rsvqa",
    "metric": "accuracy",
    "score": 0.72,
    "n_samples": 1000,
    "all_metrics": {...}
}
```

Generate a combined Markdown report:
```bash
python -m src.evaluation.generate_report
```
