# SatQuery AI — Architecture

## Overview

SatQuery AI is a multimodal remote sensing analysis system that routes natural language
queries to specialist models depending on the input imagery type.

## Pipeline

```
User Input (images + query)
        ↓
   Input Validator
   (format, mode detection, co-registration check)
        ↓
   Task Router
   (keyword + mode-based classification)
        ↓
   Tool Registry
   (lazy-loads specialist models)
        ↓
   Specialist Model
   (VQA / Captioning / Grounding / Change / Fusion)
        ↓
   RSModelResult
   (frozen schema: task, text, confidence, spatial_evidence, metadata)
        ↓
   Execution Trace
   (query, mode, task, tools_used, confidence, latency)
        ↓
   API Response
```

## Supported Input Modes

| Mode | Keys | Tasks |
|------|------|-------|
| Single image | `{"image": ...}` | VQA, Captioning, Grounding |
| Cross-modal pair | `{"image_optical": ..., "image_sar": ...}` | Fusion |
| Bi-temporal pair | `{"image_t1": ..., "image_t2": ...}` | Change Detection |

## Specialist Models

### VQA (Person 1)
- Primary: GeoChat + LoRA (remote-sensing adapted VLM)
- Fallback: CLIP embedding similarity over RS vocabulary
- Baseline: keyword + image statistics (always available)

### Captioning (Person 1)
- Shared GeoChat checkpoint with VQA, different prompt
- Template fallback based on image statistics

### Grounding (Person 2)
- Primary: GeoChat prompted for bounding box output
- CLIP fallback: 4×4 patch similarity
- Heuristic baseline: quadrant analysis

### Change Detection (Person 3)
- Primary: VisTA (pretrained)
- Fallback: Siamese absolute-difference + Otsu threshold (functional, fully tested)

### Optical-SAR Fusion (Person 4)
- Architecture: ResNet-18 (optical) + ResNet-18 (SAR) + FusionHead (MLP)
- 19-class multi-label classification (BigEarthNet/CORINE taxonomy)
- Three-way ablation: optical-only / SAR-only / optical+SAR

## Shared Interfaces (Frozen)

### RSModelResult
```python
{
    "task": str,           # "vqa" | "captioning" | "grounding" | "change" | "fusion"
    "text": str,           # Human-readable answer/description
    "confidence": float,   # [0, 1]
    "spatial_evidence": {
        "type": str,       # "bbox" | "mask" | "none"
        "source": str,     # "image" | "image_t1" | "image_t2" | "optical" | "sar"
        "bbox": list,      # [x1, y1, x2, y2] or None
        "mask": ndarray,   # (H, W) or None
    },
    "metadata": {
        "model": str,
        "backbone": str,
        "checkpoint": str,
        "dataset": str,
        "input_modalities": list,
        "parameters": dict,
    },
    "status": str,         # "success" | "error" | "low_confidence"
    "inference_seconds": float,
}
```

## Sensor Registry

```
Sentinel-2  →  sentinel2.py  (normalize: /10000, clip [0,1])
Sentinel-1  →  sentinel1.py  (to_db, min-max scale [-25,0] dB)
Cartosat-2S →  cartosat2s.py (STUB — not implemented)
RISAT       →  risat.py      (STUB — not implemented)
```

## Dataset Roles (Guardrailed)

| Dataset | Role | Usable for Training |
|---------|------|---------------------|
| BigEarthNet.txt | RS domain adaptation | ✓ |
| BigEarthNet-MM | Optical-SAR fusion | ✓ |
| VRSBench | VQA/captioning/grounding eval | ✗ |
| RSVQA | VQA eval | ✗ |
| CDVQA | Change-VQA eval | ✗ |
| ISRO/SAC | Final judging | ✗ |
