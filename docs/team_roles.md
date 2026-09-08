# Team Roles & Timeline

## ⚡ Sprint mode: 2-day ML-only phase (current)

If you're reading this because the deadline is ~2 days away: **ignore the
6-week plan further down this file for now.** Use the sprint plan below.
The 6-week plan is what this repo's roles collapse back into once the ML
backend is proven and you're ready to build the agent + GUI.

### Rule #1 for the sprint

> Nobody changes the common contract (`src/common/schemas.py`,
> `src/common/constants.py`, `configs/config.yaml`'s top-level shape,
> `BaseRSModel`) unilaterally. Model internals — architecture, prompts,
> weights, notebook code — can change freely inside your own file/notebook.

### What NOT to touch during the sprint

Do not spend sprint time on: `app/` (Streamlit GUI), `src/agent/`
(controller/router/registry wiring), FastAPI, deployment, or training a
model nobody assigned you (e.g. don't train a new grounding model if
GeoChat grounding already works). These come after the 4 ML notebooks
produce stable, evaluated inference functions.

### The 4 notebooks (one per person, one ML workstream each)

| # | Notebook | Owner | Backbone | Train? | Priority |
|---|----------|-------|----------|--------|----------|
| 1 | `notebooks/01_vqa_geochat.ipynb` | Person 1 | **GeoChat** | **Yes — LoRA** on BigEarthNet.txt | 🔴 P1 — mandatory RS adaptation + VQA |
| 2 | `notebooks/02_grounding.ipynb` | Person 2 | GeoChat first → GeoGround/SAM fallback | Usually **no** | 🟠 P4 — only add a dedicated model if GeoChat grounding IoU is too low |
| 3 | `notebooks/03_change_vista.ipynb` | Person 3 | **VisTA** | **No initially** (pretrained) | 🔴 P2 — mandatory bi-temporal change |
| 4 | `notebooks/04_optical_sar_fusion.ipynb` | Person 4 | Optical + SAR dual encoder + fusion head | **Yes — fusion head only** | 🔴 P3 — mandatory cross-modal, hardest slot |

That's **2 actual training workloads** (Person 1, Person 4) and **2
pretrained/inference-first workstreams** (Person 2, Person 3) — realistic
for two days.

### Notebook structure (keep it consistent across all 4)

```text
1. Environment
2. Load shared configuration (configs/config.yaml)
3. Dataset inspection
4. Preprocessing (via src/preprocessing/*, not ad-hoc code)
5. Load pretrained backbone
6. Baseline inference (before any training)
7. Training/adaptation (only Person 1 & Person 4)
8. Evaluation (via src/evaluation/metrics.py)
9. Save checkpoint (path from configs/config.yaml -> models.<task>.checkpoint)
10. Export a predict() function matching src.common.schemas.RSModelResult
```

Every notebook already has this skeleton filled in with `TODO`s and
commented-out example code — see the actual `.ipynb` files.

### Common input/output contract (all 4 must satisfy this)

**Input** — model-specific kwargs (see `docs/api_contracts.md`):
`(image, query)` for vqa/captioning/grounding,
`(image_t1, image_t2, query)` for change,
`(image_optical, image_sar, query)` for fusion.

**Output** — every `predict()` returns
`src.common.schemas.RSModelResult(...).to_dict()`:

```python
{
  "task": "vqa" | "captioning" | "grounding" | "change" | "fusion",
  "text": str,
  "confidence": float,
  "spatial_evidence": {"type": "bbox"|"mask"|"none", "source": "image"|"image_t1"|
                        "image_t2"|"optical"|"sar"|"fused",
                        "bbox": [...] | None, "mask": np.ndarray | None},
  "metadata": {"model": str, "backbone": str, "checkpoint": str, "dataset": str,
               "input_modalities": [...], "parameters": {...}},
  "status": "success" | "error" | "low_confidence",
  "inference_seconds": float,
}
```

See `docs/api_contracts.md` §2 for the full rationale (why `text` not
`answer`, why `source` on the evidence is load-bearing). Also see
`docs/DECISIONS.md` and `docs/DATASETS.md` for the frozen dataset-role table
and the sensor domain-gap seam (Sentinel dev data vs. Cartosat/RISAT hidden
eval set).

Use `RSModelResult(...).to_dict()` or `BaseRSModel._empty_result(...)` —
don't hand-roll a differently-shaped dict, or integration breaks later.

### What each person must hand over at the end of the sprint

Not "my notebook runs" — actually hand over:

- **Person 1**: GeoChat LoRA adapter checkpoint + working `predict()` for
  VQA and captioning + RSVQA/VRSBench evaluation numbers + a before/after
  (pretrained vs. adapted) comparison. **Do this first** — Person 2's
  GeoChat-grounding path depends on this checkpoint existing.
- **Person 2**: grounding `predict()` (GeoChat-based or fallback) + IoU
  result on a VRSBench subset + a couple of visualized bounding-box
  overlays. Decision rule (freeze **before** you see results, per
  `configs/config.yaml -> datasets.vrsbench.grounding_iou_threshold`):
  mean IoU ≥ 0.30 → ship GeoChat-only; below → escalate to the SAM
  fallback. Visualize boxes on real images before trusting an IoU number —
  it catches coordinate-parsing bugs an aggregate metric would hide.
- **Person 3**: **Day 1, first thing** — confirm VisTA is actually
  obtainable (pretrained weights, working repo/env). This is the highest
  unknown-unknown risk in the sprint since it's less mainstream than
  GeoChat/SAM. If broken/unavailable, fall back to a simple
  siamese-difference approach or a GeoChat-twice-plus-diff-prompt approach
  — worse quality, guaranteed working. Then: VisTA `predict()` returning
  both a change answer and a change mask + CDVQA evaluation numbers. Run
  T1 and T2 through the *same* sensor adapter/normalization — asymmetric
  preprocessing silently degrades change detection.
- **Person 4**: fusion head checkpoint + `predict()`,
  `predict_optical_only()`, `predict_sar_only()` + an honest 3-way ablation
  table (optical / SAR / optical+SAR). Fusion output is a **multi-label
  land-cover classification**, with `text` as a templated sentence built
  from it — not open-ended generation (see `src/models/fusion_model.py`).
  If fusion loses to optical-only, report that — it's a legitimate finding,
  not a bug to hide.

Everyone also runs `scripts/smoke_test.py` before declaring done — it's the
single objective status board for "ML backend complete" (all four lines
`[PASS]`, all four outputs validate against `RSModelResult`).

Everyone also provides one small test sample (image(s) + query + expected
output) so integration can be smoke-tested without re-running training.

### Checkpoint convention

Don't use `model_final_v2_new.pt`. Use the structure already reflected in
`configs/config.yaml`:

```text
models/checkpoints/
├── geochat/lora_adapter/
├── grounding/model/          # only if the fallback path is used
├── vista/pretrained/
└── fusion/fusion_head/
```

### After the sprint: integration order

1. Merge each notebook's stable `predict()` into `src/models/*.py` (files
   already exist with `TODO` markers matching each backbone).
2. Wire `src/agent/tool_registry.py` to actually call `model.load(...)`
   (currently commented out on purpose — see file).
3. Swap `AgentController.run()`'s stub call for the real
   `tool.predict(**params)` (one line, flagged with `TODO` in
   `src/agent/controller.py`).
4. Only then build/polish `app/app.py` (already wired to the controller and
   runs today with stub answers, so this step is mostly cosmetic).

---

## 🐢 Longer-project roles (6+ weeks) — reference only, not the current plan

If this project continues past the initial sprint into full production
hardening, the roles below are a reasonable way to divide the *remaining*
work (data robustness, agent intelligence, reporting, UI polish). They are
not how the initial 2-day ML sprint should be organized — see above.

| # | Role | Owns | Key files |
|---|------|------|-----------|
| 1 | **Data & Preprocessing hardening** | Edge cases in GeoTIFF I/O, per-sensor band/normalization config, tiling for large scenes, dataset QA | `src/preprocessing/*` |
| 2 | **Single-image models hardening** | GeoChat LoRA v2, captioning quality, grounding fallback (GeoGround/SAM) if needed | `src/models/vqa_model.py`, `captioning_model.py`, `grounding_model.py` |
| 3 | **Multi-image models hardening** | VisTA fine-tuning (stretch goal), fusion head v2, richer masks | `src/models/change_model.py`, `fusion_model.py` |
| 4 | **Agentic orchestration + GUI + evaluation** | Real task routing (LLM-based), tool registry wiring, execution trace, report generation, Streamlit polish | `src/agent/*`, `app/app.py`, `src/evaluation/*` |

### Suggested Timeline (6-week reference)

| Week | Person 1 | Person 2 | Person 3 | Person 4 |
|------|----------|----------|----------|----------|
| 1 | GeoTIFF I/O + dataset survey | GeoChat VQA v1 | VisTA inference v1 | Fusion data prep (BigEarthNet-MM) |
| 2 | Per-sensor preprocessing (Cartosat/RISAT vs Sentinel) | LoRA fine-tuning v1 | CDVQA evaluation | Fusion head v1 |
| 3 | BigEarthNet adaptation pipeline hardening | Grounding (GeoChat, then fallback if needed) | VisTA fine-tuning (if pursued) | Fusion evaluation + baseline comparison |
| 4 | Data QA, tiling for large scenes | VQA + captioning v2, eval | Change model v2, eval | Fusion v2, eval |
| 5 | Support agent integration testing | Model hardening | Model hardening | Model hardening |
| 6 | Buffer / bug fixes | Agent wiring, GUI, execution trace, report generation, ISRO/SAC eval run, polish (Person 4 leads, others support) |

## Metrics Ownership (implemented in `src/evaluation/metrics.py`)

- **VQA**: accuracy / exact-match, plus soft accuracy for open-ended answers.
- **Captioning**: BLEU-4, ROUGE-L, CIDEr.
- **Grounding**: IoU / mIoU against ground-truth boxes.
- **Change VQA/description**: accuracy (VQA-style) or BLEU/ROUGE (description) + F1/Kappa
  if a change mask is produced.
- **Fusion**: task-dependent (e.g. IoU for built-up/water extraction masks) +
  mandatory optical-only vs. optical+SAR comparison.
- All metrics normalized (0–1) before combination, per the problem statement.
