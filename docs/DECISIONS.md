# SatQuery AI — Architecture & Implementation Decisions

## D1: Fallback Hierarchy for Every Model

**Decision**: Every model implements an explicit 2-3 level fallback hierarchy, recorded
in `ResultMetadata.parameters["fallback_level"]`. The highest level tried is always reported.

**Rationale**: The problem statement mandates reliable end-to-end functionality. Heavy models
(GeoChat, VisTA) may be unavailable in resource-constrained environments. A system that
crashes when a checkpoint is missing is less useful than one that degrades gracefully.

**Implementation**: VQA: GeoChat→CLIP→Keyword+Stats. Grounding: GeoChat→CLIP-patch→Heuristic.
Change: VisTA→Siamese-diff (always functional). Fusion: Trained head→Untrained head (always functional).

## D2: Siamese-Difference as Change Detection Baseline

**Decision**: Implement Otsu-thresholded absolute difference as the guaranteed change detection
baseline instead of relying solely on VisTA.

**Rationale**: VisTA is a research repository (not pip-installable) that requires manual setup.
The baseline delivers real value: it detects change area, percentage, and spatial location.
It is honest about its limitations (confidence capped at 0.75, status=`low_confidence`).

**Source**: `src/models/change_model.py::_predict_fallback`

## D3: FusionModel Outputs Multi-Label Classification, Not Free Text

**Decision**: The fusion model outputs probabilities over 19 CORINE land-cover classes with a
thin template wrapper for the `text` field, NOT open-ended language generation.

**Rationale**: Open-ended generation from fused optical+SAR features would require a VLM decoder
we don't have pretrained for that task. Multi-label classification is semantically appropriate
for land-cover analysis, achievable with a lightweight MLP head, and allows honest evaluation
(F1, per-class accuracy) without requiring a generative VLM.

**Source**: `src/models/fusion_model.py`, `BIGEARTHNET_19_LABELS`

## D4: Three-Way Ablation Is Mandatory and Non-Negotiable

**Decision**: FusionModel always exposes `predict()` (optical+SAR), `predict_optical_only()`,
and `predict_sar_only()`. The evaluation NEVER asserts fusion >= optical.

**Rationale**: Per the problem statement, the ablation is mandatory. An earlier notebook draft
had `assert fusion >= optical` — this was removed. If SAR or optical-only actually performs
better, we report that honestly. Fabricating ablation results would undermine the scientific
validity of the system.

## D5: Cartosat-2S and RISAT Adapters Are Deliberate Stubs

**Decision**: `sensor_adapters/cartosat2s.py` and `sensor_adapters/risat.py` raise
`NotImplementedError` with exact, actionable error messages rather than silently applying
Sentinel normalization constants.

**Rationale**: Applying Sentinel-2 normalization (REFLECTANCE_SCALE=10000) to Cartosat-2S data
with unknown radiometric range would silently corrupt model inputs on the final judging set.
A loud failure is better than silent wrong output. The adapters will be completed when ISRO/SAC
sample data or spec sheets become available.

## D6: No LLM-Based Query Classification (Deterministic Router)

**Decision**: `task_router.py` uses keyword matching + mode constraints, not an LLM.

**Rationale**: The problem statement says "deterministic, auditable controller." Adding an LLM
for query classification introduces latency, cost, and non-determinism for marginal gain — the
keyword approach already handles the main patterns. The router is documented as the place to
add an LLM classifier if keyword matching proves insufficient.

## D7: Grounding SAM Fallback Not Implemented (Full Implementation)

**Decision**: SAM is documented as a fallback in the original code but not implemented in
the grounding model. Replaced by CLIP-patch similarity grounding.

**Rationale**: SAM without a seed point has no text understanding. The original SAM fallback
`_predict_sam()` always returned an error about needing a seed. CLIP-patch similarity provides
actual text-conditioned grounding without requiring SAM's ~360MB checkpoint.

## D8: Dataset Guardrail at DataLoader Level

**Decision**: `get_dataloader()` raises immediately if `split="train"` is requested for a
dataset where `DATASET_ROLES[dataset]["usable_for_training"] = False`.

**Rationale**: Under deadline pressure, it's easy to accidentally add an eval-only dataset to
a training loop. A machine-checked guardrail prevents this from silently invalidating results.

## D9: Preprocessing Is Two-Stage

**Decision**: Preprocessing separates dataset-level (band selection + sensor normalization) from
model-level (resize to backbone's expected input size).

**Rationale**: The same raw image read from Sentinel-2 should normalize to [0,1] the same way
regardless of which model it's going to. But GeoChat expects 336×336 while the fusion head
expects 256×256 — that resize is model-specific, not sensor-specific.
