# API Contracts

These are the exact interfaces the four workstreams must respect so the
pieces plug together without a big-bang integration at the end. Treat this
file as a contract — change it only after agreement, and update all callers.
This is the one document everyone should actually read before splitting up;
everything else is detail.

## 0. The three things that actually matter

Everything below exists to nail down exactly three things. If these three
are frozen correctly, the four of you can diverge on architecture, prompts,
libraries — anything downstream — and still merge cleanly.

1. **Output shape** every model returns → `RSModelResult` (§2)
2. **Input shape** (typed samples) data loaders return → `ImageSample` /
   `PairSample` / `FusionSample` (§1)
3. **Dataset semantics** — which dataset is for training vs. eval-only → `DATASET_ROLES` (§5)

## 1. Data loading (Person 1 → everyone)

```python
# src/preprocessing/dataset_loader.py
def get_dataloader(
    task: str,          # "vqa" | "captioning" | "grounding" | "change" | "fusion"
    split: str,          # "train" | "val" | "test"
    dataset: str = None, # "bigearthnet" | "bigearthnet_mm" | "vrsbench" | "rsvqa" | "cdvqa" | "isro_sac"
    batch_size: int = 8,
    config: dict = None,
) -> "torch.utils.data.DataLoader":
    """Requesting split="train" for a dataset marked eval-only in
    src.common.constants.DATASET_ROLES raises immediately (see §5) —
    this is enforced, not just documented.
    """
```

Underlying per-item samples are typed (`src/common/schemas.py`), not ad-hoc
dicts/tuples — this stops four people from inventing four different shapes:

```python
ImageSample(image=array, modality="optical"|"sar", sensor="Sentinel-2"|...,
            metadata={...}, query=..., label=...)
PairSample(image_t1=..., image_t2=..., sensor=..., metadata_t1={...},
           metadata_t2={...}, query=..., label=...)
FusionSample(optical=..., sar=..., optical_sensor=..., sar_sensor=...,
             metadata={...}, query=..., label=...)
```

Model `predict()` signatures stay **explicit kwargs**, not one of these
objects — see the note at the top of `src/common/schemas.py`. Convert
sample → kwargs at the call site (dataset loader / eval harness / agent).

## 2. Models (Person 1-4 → agent integration)

Every model implements `BaseRSModel` (`src/models/base_model.py`) and
returns the shared result contract defined in `src/common/schemas.py`
(`RSModelResult`) — **THIS SCHEMA IS FROZEN**, don't hand-roll a different shape:

```python
class BaseRSModel:
    name: str                 # unique id used in tool_registry, e.g. "vqa_v1"
    task: str                 # "vqa" | "captioning" | "grounding" | "change" | "fusion"
    backbone: str             # e.g. "GeoChat", "VisTA", "OpticalSAR-FeatureFusion"

    def load(self, checkpoint_path: str, device: str = "cpu") -> None: ...
    def health_check(self) -> dict: ...   # used by scripts/smoke_test.py

    def predict(self, **inputs) -> dict:
        """Must return RSModelResult(...).to_dict():
        {
          "task": str,
          "text": str,                          # NOT "answer" — captions/descriptions
                                                  # aren't "answering" anything
          "confidence": float,                   # 0-1
          "spatial_evidence": {
              "type": "bbox" | "mask" | "none",
              "source": "image" | "image_t1" | "image_t2" |
                        "optical" | "sar" | "fused",   # REQUIRED — disambiguates which
                                                        # image/coordinate frame the
                                                        # evidence belongs to
              "bbox": [x1,y1,x2,y2] | None,       # pixel coords, in `source`'s frame
              "mask": np.ndarray | None,          # HxW, same frame as `source`
          },
          "metadata": {
              "model": str, "backbone": str, "checkpoint": str, "dataset": str,
              "input_modalities": [...],          # e.g. ["optical"], ["optical","sar"], ["t1","t2"]
              "parameters": {...},
          },
          "status": "success" | "error" | "low_confidence",
          "inference_seconds": float,
        }
        Use RSModelResult(...).to_dict() or self._empty_result(...) — never
        construct this dict by hand.
        """
```

Concrete expected call signatures (what `predict(**inputs)` receives) and
which backbone/owner each maps to for the current sprint:

| Model | Owner | Backbone | inputs kwargs | `input_modalities` |
|---|---|---|---|---|
| VQA | Person 1 | GeoChat (LoRA-adapted) | `image, query` | `["optical"]` (or `["sar"]`) |
| Captioning | Person 1 | GeoChat (shared checkpoint) | `image` (query optional) | `["optical"]` |
| Grounding | Person 2 | GeoChat → GeoGround/SAM fallback | `image, query` | `["optical"]` |
| Change | Person 3 | VisTA (pretrained) | `image_t1, image_t2, query` | `["t1","t2"]` |
| Fusion | Person 4 | Optical+SAR dual encoder + fusion head | `image_optical, image_sar, query` | `["optical","sar"]` |

**Fusion note:** `text` for the fusion model is a **templated wrapper**
around a multi-label classification output, not open-ended generation —
see `src/models/fusion_model.py` for why. Fusion must also implement
`predict_optical_only()` and `predict_sar_only()` for the mandatory 3-way
ablation.

## 3. Agent (Person 4, consumes 1–2)

```python
# src/agent/input_validator.py
def validate_input(images: dict, config: dict) -> dict:
    """images: {"image": path} or {"image_t1":.., "image_t2":..} or
    {"image_optical":.., "image_sar":..}
    Returns: {"valid": bool, "mode": "single"|"cross_modal"|"bi_temporal",
              "errors": list[str], "metadata": dict}
    """

# src/agent/task_router.py
def route_task(query: str, mode: str) -> str:
    """Returns one of: "vqa","captioning","grounding","change","fusion" """

# src/agent/tool_registry.py
class ToolRegistry:
    def get(self, task: str) -> "BaseRSModel": ...
    def register(self, model: "BaseRSModel", checkpoint_path: str) -> None: ...

# src/agent/controller.py
class AgentController:
    def run(self, images: dict, query: str) -> dict:
        """Full pipeline: validate -> route -> select tool(s) -> predict ->
        integrate -> log trace. Returns:
        {
          "success": bool,
          "task": str | None,
          "text": str | None,
          "confidence": float | None,
          "spatial_evidence": {"type":.., "source":.., "bbox":.., "mask":..} | None,
          "metadata": {...} | None,
          "status": str | None,
          "errors": list[str],
          "trace": dict,   # see execution_trace.py — the auditable execution summary
        }
        """
```

## 4. Sensors — the domain-gap seam (all of Layer B, mainly Person 3 & 4)

```python
# src/preprocessing/sensor_registry.py
def get_adapter(sensor_name: str):
    """sensor_name: "Sentinel-2" | "Sentinel-1" | "Cartosat-2S" | "RISAT"
    Returns the adapter module — call adapter.normalize(raw_array).
    Cartosat-2S/RISAT adapters are deliberate stubs (NotImplementedError)
    until real sample data/spec is available — see their module docstrings.
    Never guess their radiometric constants and ship them silently.
    """
```

Development/training data is Sentinel-based; the hidden ISRO/SAC judging
set is Cartosat-2S (optical) + RISAT (SAR) — a different sensor pair with a
different radiometric range. This seam exists so that gap doesn't produce a
silent, undetected failure at judging time (see `docs/DECISIONS.md`).

## 5. Datasets (shared, frozen)

```python
# src/common/constants.py
DATASET_ROLES = {
    "bigearthnet":    {"usable_for_training": True,  ...},  # Person 1
    "bigearthnet_mm": {"usable_for_training": True,  ...},  # Person 4
    "vrsbench":       {"usable_for_training": False, ...},  # eval-only
    "rsvqa":          {"usable_for_training": False, ...},  # eval-only
    "cdvqa":          {"usable_for_training": False, ...},  # eval-only (stretch fine-tune only)
    "isro_sac":       {"usable_for_training": False, ...},  # never seen during dev
}
```

`get_dataloader(..., split="train", dataset=X)` raises if `X`'s
`usable_for_training` is `False` — a real guardrail, not just a norm
someone can violate under deadline pressure.

## 6. Evaluation (Person 4, all 4 people write into this)

```python
# src/evaluation/metrics.py
def vqa_accuracy(preds: list[str], golds: list[str]) -> float: ...
def caption_scores(preds: list[str], golds: list[str]) -> dict:  # bleu, rouge, cider
def grounding_iou(pred_boxes: list, gold_boxes: list) -> float: ...
def change_f1(pred_masks, gold_masks) -> dict:  # precision, recall, f1, kappa
def fusion_ablation(optical_metric, sar_metric, fusion_metric) -> dict:  # honest 3-way report
```

Each person's notebook writes one file: `src/evaluation/results/<person>.json`
(schema in `docs/DATASETS.md` §Evaluation). `src/evaluation/generate_report.py`
globs all four and prints a combined markdown table.

## 7. Config (shared)

Everything reads from `configs/config.yaml`. No hardcoded paths. Preprocessing
is per-dataset (band indices/normalization/sensor — see §4) and per-model
(`image_size` — different backbones expect different input sizes), NOT a
single global block. If you add a new key, document it inline in the YAML
with a comment. Changing the top-level schema (not just adding a value
inside your own section) needs a team sync.
