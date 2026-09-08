# SatQuery AI — Model Implementation Status

Last updated: 2025-08

## Summary Table

| Model | Task | Status | Fallback | Notes |
|-------|------|--------|----------|-------|
| GeoChat VQA | vqa | PARTIAL | keyword_statistics_baseline | Requires GeoChat checkpoint |
| GeoChat Captioning | captioning | PARTIAL | statistics_template_baseline | Shares VQA checkpoint |
| Grounding (GeoChat/CLIP) | grounding | PARTIAL | heuristic_quadrant | CLIP fallback available |
| Change (VisTA + siamese-diff) | change | FUNCTIONAL | siamese_difference ✓ | Siamese fallback works end-to-end |
| Optical-SAR Fusion | fusion | FUNCTIONAL | random_init_fallback | FusionHead implemented; needs BEN-MM training |

## Detailed Status

### VQA (vqa_v1)

**Status: PARTIAL — Fallback baseline functional**

- **Level 1 (GeoChat LoRA)**: Requires `MBZUAI/geochat-7B` base weights + a trained LoRA adapter.
  Training script: `python scripts/train_vqa.py --config configs/colab_8gb.yaml`.
  Without checkpoint, automatically falls through to Level 2/3.
- **Level 2 (CLIP-RS)**: Uses `openai/clip-vit-base-patch32` for embedding-based VQA.
  Downloads from HuggingFace on first run (~600MB).
- **Level 3 (Keyword+Statistics Baseline)**: Zero dependencies, always available.
  Returns answers based on query keywords + image band statistics.
  Labeled `low_confidence` in results — never mistaken for a trained model.

**Reported accurately in**: `metadata.parameters.fallback_level`

### Captioning (caption_v1)

**Status: PARTIAL — Template baseline functional**

Shares GeoChat checkpoint with VQA (different prompt). Template-based fallback generates
scene descriptions from image statistics. BLEU-4 and ROUGE-L metrics implemented in
`src/evaluation/metrics.py::caption_scores`.

### Grounding (grounding_v1)

**Status: PARTIAL — Heuristic baseline functional**

- Primary: GeoChat prompted for bbox output, parsed by `parse_bbox_from_text()`.
- CLIP fallback: 4x4 patch similarity (implemented, needs CLIP download).
- Heuristic baseline: quadrant-based grounding from keywords + image statistics.
  Always returns a valid `[x1, y1, x2, y2]` bbox in pixel coordinates.

Coordinate handling (`parse_bbox_from_text`, `rescale_bbox`, `clip_bbox`) is fully
implemented and unit-tested independently of any model.

### Change Detection (change_v1)

**Status: FUNCTIONAL — Siamese-difference baseline works end-to-end**

- **VisTA (primary)**: Research repo, requires manual clone+install.
  See `src/models/change_model.py::_try_load_vista()` for exact setup commands.
  If unavailable, automatically falls back to siamese-difference.
- **Siamese-difference (fallback, always functional)**:
  - Computes per-band absolute difference between T1 and T2.
  - Otsu's method for automatic threshold (no hand-tuned constant).
  - Returns change mask, percentage changed, spatial description.
  - Confidence: heuristic magnitude-based (capped at 0.75 — never overconfident).
  - Labeled `low_confidence` in results — never pretends to be VisTA.

Evaluation metrics (pixel F1, precision, recall, IoU, Cohen's kappa) implemented in
`src/evaluation/metrics.py::change_f1`.

### Optical-SAR Fusion (fusion_v1)

**Status: FUNCTIONAL — Architecture complete, needs BigEarthNet-MM training**

- **Architecture**: ResNet-18 encoders (optical + SAR) + FusionHead (256-dim MLP).
  Fully implemented in `src/models/fusion_model.py`.
- **19-class multi-label classification** over BigEarthNet/CORINE taxonomy.
- **Three-way ablation**: `predict()`, `predict_optical_only()`, `predict_sar_only()`.
- **Without trained checkpoint**: returns predictions but reports `low_confidence`.
  Encoder weights: ImageNet-pretrained (if downloadable) or random init fallback.
  Both recorded in `metadata.parameters.encoder_weights`.
- **Training**: `python scripts/train_fusion.py --config configs/colab_8gb.yaml`.

## What Is NOT Implemented

| Feature | Status | Notes |
|---------|--------|-------|
| Cartosat-2S normalization | NOT IMPLEMENTED | Deliberate stub — see `sensor_adapters/cartosat2s.py` |
| RISAT normalization | NOT IMPLEMENTED | Deliberate stub — see `sensor_adapters/risat.py` |
| VisTA integration | NOT IMPLEMENTED | Requires manual clone from GitHub |
| GeoChat full fine-tuning | NOT IMPLEMENTED | Only LoRA script exists |
| Segmentation masks for fusion | NOT IMPLEMENTED | Stretch goal — FusionHead returns class probs, not masks |
| SAM grounding fallback | NOT IMPLEMENTED | Requires `pip install segment-anything` + checkpoint |
| Speckle filtering (SAR) | NOT IMPLEMENTED | Known gap — see `sentinel1.py` docstring |

## Fallback Hierarchy (per model)

### VQA
```
GeoChat LoRA (L1) → CLIP-RS embedding (L2) → Keyword+Statistics baseline (L3)
```

### Captioning
```
GeoChat caption prompt (L1) → Statistics+Template baseline (L2)
```

### Grounding
```
GeoChat bbox parsing (L1) → CLIP patch similarity (L2) → Heuristic quadrant (L3)
```

### Change
```
VisTA (L1) → Siamese-difference + Otsu (L2, functional ✓)
```

### Fusion
```
Trained FusionHead + ResNet encoders (L1) → Same arch, untrained (L2, functional ✓)
```
