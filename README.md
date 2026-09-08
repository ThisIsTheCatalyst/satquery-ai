# SatQuery AI

**Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis**
ISRO SIH Problem Statement 26167

---

## What it does

Ask a natural-language question about satellite imagery and get back an answer,
a confidence score, spatial evidence (bounding box or change mask), and an
auditable trace of how the answer was produced.

```
Input (image(s) + query)
    ↓  Input Validator      — mode detection, co-registration check
    ↓  Task Router          — picks the specialist from query + input shape
    ↓  Tool Registry        — lazy-loads that specialist
    ↓  Specialist Model     — VQA / Captioning / Grounding / Change / Fusion
    ↓  RSModelResult        — one frozen schema for all five tasks
    ↓  Execution Trace      — query, mode, task, tools, confidence, latency
```

The agent layer is the contribution: five heterogeneous models behind one
query interface, one result contract, and one audit trail.

---

## Quick start

```bash
pip install -r requirements-demo.txt
python scripts/prepare_data.py        # EuroSAT (~90 MB)
python scripts/eval_clip_eurosat.py   # verify CLIP is working
streamlit run app/app.py
```

Colab: open `notebooks/DEMO.ipynb` and run all cells (~8 minutes).

---

## Tasks and how each is implemented

| Task | Method | Training | Honest status |
|---|---|---|---|
| VQA | CLIP ViT-B/32 zero-shot over a 10-class RS vocabulary | None | Working |
| Captioning | CLIP top-2 classes → templated sentence | None | Working; surface text is templated, not generated |
| Grounding | CLIP scored over a 4×4 tile grid, peak region expanded | None | Working; box snaps to tile resolution |
| Change detection | Per-band absolute difference + Otsu threshold | None (unsupervised) | Working |
| Optical–SAR fusion | Two frozen ResNet-18 encoders + 2-layer MLP head | Head only, ~15 min | Needs a paired optical/SAR subset |

**No model in this repo is fine-tuned on a remote-sensing task.** VQA,
captioning and grounding are zero-shot; change detection is unsupervised;
only the small fusion head is trained. That is a deliberate scope decision
for a prototype, and it is why the numbers below are measured rather than
claimed.

---

## Measured results

Run the scripts to reproduce. Fill in the numbers you obtain.

| Metric | Script | Result |
|---|---|---|
| CLIP zero-shot top-1, EuroSAT (10 classes, chance = 10%) | `scripts/eval_clip_eurosat.py` | _run to populate_ |
| Change detection pixel F1, OSCD | `scripts/eval_change_oscd.py` | _run to populate_ |
| Fusion macro-F1 (optical / SAR / fused) | `scripts/train_fusion.py` | _run to populate_ |

---

## Datasets

| Dataset | Size | Role |
|---|---|---|
| EuroSAT (RGB) | ~90 MB | Labelled Sentinel-2 chips — VQA/captioning demo and accuracy measurement |
| OSCD | ~500 MB | Real bi-temporal Sentinel-2 pairs with ground-truth change masks |
| Optical+SAR subset | small | Fusion training and ablation |

EuroSAT downloads automatically. OSCD requires accepting terms at
<https://rcdaudt.github.io/oscd/> and extracting to `data/raw/oscd/`.

---

## Sensor handling

Development data is Sentinel-1/2. The ISRO judging set is Cartosat-2S +
RISAT — a different sensor pair with different band layouts and calibration.
`src/preprocessing/sensor_registry.py` selects a per-sensor adapter so band
selection and normalisation are never hardcoded to Sentinel.

The Cartosat-2S and RISAT adapters are **deliberate stubs** that raise
`NotImplementedError`. Implementing them requires SAC band and calibration
specifications; guessing the normalisation would be worse than failing
loudly.

---

## Known limitations

1. Grounding localises at tile resolution (1/grid of the image side), so box
   edges snap to grid lines.
2. Captions are templated from classification output, not generated.
3. The VQA vocabulary is the 10 EuroSAT land-cover classes — it will not
   answer counting or fine-grained attribute questions.
4. Fusion requires a paired optical/SAR subset; the `--synthetic` flag
   smoke-tests the pipeline but its numbers are not results.
5. Cartosat-2S / RISAT adapters unimplemented.

---

## Repository layout

```
src/models/clip_backend.py    shared CLIP instance (VQA + captioning + grounding)
src/models/*_model.py         the five specialists
src/agent/                    validator, router, registry, controller, trace
src/data/                     EuroSAT and OSCD loaders
src/common/schemas.py         RSModelResult — frozen result contract
scripts/prepare_data.py       one-command data setup
scripts/eval_*.py             the two measured numbers
scripts/train_fusion.py       fusion head + 3-way ablation
app/app.py                    Streamlit demo
notebooks/DEMO.ipynb          Colab entry point
tests/                        61 tests
```

## Tests

```bash
python -m pytest tests/ -q
```
