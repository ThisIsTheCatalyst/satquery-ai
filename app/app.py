"""
SatQuery AI — demo UI.

Run:  streamlit run app/app.py

Design intent: a judge should be able to produce a correct, image-conditioned
answer within about ten seconds of the page loading, without uploading
anything. Hence the built-in gallery of real EuroSAT and OSCD imagery, and
the preset query buttons.
"""

import sys, os, json, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import streamlit as st
from PIL import Image as PILImage, ImageDraw

st.set_page_config(page_title="SatQuery AI", page_icon="🛰️", layout="wide")

# ── resources ────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading models (first run only)…")
def get_controller():
    import yaml
    from src.agent.controller import AgentController
    cfg_path = "configs/demo.yaml" if os.path.exists("configs/demo.yaml") \
        else "configs/config.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    return AgentController(cfg, device="cpu"), cfg


@st.cache_data(show_spinner=False)
def load_gallery():
    """Real EuroSAT chips for the single-image tasks."""
    try:
        from src.data.eurosat import load_demo_subset, demo_subset_exists
        if not demo_subset_exists():
            return []
        return [(a, c, f) for a, c, f in load_demo_subset()]
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def load_pairs():
    """Real OSCD bi-temporal pairs for change detection."""
    try:
        from src.data.oscd import load_demo_pairs, demo_pairs_exist
        if not demo_pairs_exist():
            return []
        return load_demo_pairs()
    except Exception:
        return []


controller, CFG = get_controller()
GALLERY = load_gallery()
PAIRS = load_pairs()

# ── helpers ──────────────────────────────────────────────────────────────────

def to_chw(hwc: np.ndarray) -> np.ndarray:
    return hwc.transpose(2, 0, 1).astype("float32") / 255.0


def draw_box(hwc: np.ndarray, bbox) -> np.ndarray:
    img = PILImage.fromarray(hwc.astype("uint8")).convert("RGB")
    # upscale small chips so the box is visible on screen
    if max(img.size) < 256:
        img = img.resize((img.width * 4, img.height * 4), PILImage.NEAREST)
    w, h = img.size
    x1, y1, x2, y2 = bbox
    if max(bbox) <= 1.0:
        x1, x2, y1, y2 = x1 * w, x2 * w, y1 * h, y2 * h
    ImageDraw.Draw(img).rectangle([x1, y1, x2, y2], outline=(255, 40, 40), width=4)
    return np.array(img)


def overlay(hwc: np.ndarray, mask: np.ndarray) -> np.ndarray:
    img = hwc.astype("float32").copy()
    m = np.asarray(mask, dtype="float32")
    if m.max() > 1.0:
        m = m / m.max()
    m = np.array(
        PILImage.fromarray((m * 255).astype("uint8"))
        .resize((hwc.shape[1], hwc.shape[0]), PILImage.NEAREST)
    ).astype("float32") / 255.0
    img[:, :, 0] = np.clip(img[:, :, 0] + m * 170, 0, 255)
    img[:, :, 1] = np.clip(img[:, :, 1] * (1 - m * 0.55), 0, 255)
    img[:, :, 2] = np.clip(img[:, :, 2] * (1 - m * 0.55), 0, 255)
    return img.astype("uint8")


PRESETS = {
    "Single image": [
        "What is the land cover in this image?",
        "Describe this satellite image.",
        "Is there water in this image?",
        "Highlight the residential area.",
        "Where is the vegetation?",
    ],
    "Bi-temporal pair": [
        "What changed between these two dates?",
        "Where did the change occur?",
    ],
    "Optical + SAR": [
        "Identify the land cover using both optical and SAR.",
    ],
}

# ── sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("🛰️ SatQuery AI")
mode = st.sidebar.radio("Input mode", list(PRESETS.keys()))

images_hwc = {}

if mode == "Single image":
    src = st.sidebar.radio("Source", ["Sample (EuroSAT)", "Upload"],
                           horizontal=True)
    if src == "Sample (EuroSAT)" and GALLERY:
        classes = sorted({c for _, c, _ in GALLERY})
        cls = st.sidebar.selectbox("Land-cover class", classes)
        opts = [(a, f) for a, c, f in GALLERY if c == cls]
        i = st.sidebar.slider("Image", 0, max(len(opts) - 1, 0), 0)
        images_hwc["image"] = opts[i][0]
        st.sidebar.caption(f"Ground truth: **{cls}** · `{opts[i][1]}`")
    elif src == "Sample (EuroSAT)":
        st.sidebar.warning("No EuroSAT subset found. "
                           "Run `python scripts/prepare_data.py`.")
    else:
        f = st.sidebar.file_uploader("Image", type=["png", "jpg", "jpeg", "tif", "tiff"])
        if f:
            images_hwc["image"] = np.array(PILImage.open(f).convert("RGB"))

elif mode == "Bi-temporal pair":
    src = st.sidebar.radio("Source", ["Sample (OSCD)", "Upload"], horizontal=True)
    if src == "Sample (OSCD)" and PAIRS:
        names = [p["name"] for p in PAIRS]
        n = st.sidebar.selectbox("Scene", names)
        p = next(x for x in PAIRS if x["name"] == n)
        images_hwc["image_t1"] = p["t1"]
        images_hwc["image_t2"] = p["t2"]
        st.session_state["_gt"] = p["gt"]
        st.sidebar.caption("Real Sentinel-2 pair"
                           + (" · ground truth available" if p["gt"] is not None else ""))
    elif src == "Sample (OSCD)":
        st.sidebar.warning("No OSCD pairs found. "
                           "Run `python scripts/prepare_data.py`.")
    else:
        a = st.sidebar.file_uploader("T1", type=["png", "jpg", "jpeg", "tif"], key="t1")
        b = st.sidebar.file_uploader("T2", type=["png", "jpg", "jpeg", "tif"], key="t2")
        if a: images_hwc["image_t1"] = np.array(PILImage.open(a).convert("RGB"))
        if b: images_hwc["image_t2"] = np.array(PILImage.open(b).convert("RGB"))

else:  # optical + SAR
    a = st.sidebar.file_uploader("Optical", type=["png", "jpg", "jpeg", "tif"], key="o")
    b = st.sidebar.file_uploader("SAR", type=["png", "jpg", "jpeg", "tif"], key="s")
    if a: images_hwc["image_optical"] = np.array(PILImage.open(a).convert("RGB"))
    if b: images_hwc["image_sar"] = np.array(PILImage.open(b).convert("RGB"))
    st.sidebar.info("Fusion needs a co-registered optical/SAR pair.")

st.sidebar.markdown("---")
preset = st.sidebar.selectbox("Preset query", ["(type your own)"] + PRESETS[mode])
query = st.sidebar.text_area("Query", value="" if preset == "(type your own)" else preset,
                             height=80)
run = st.sidebar.button("▶ Run", type="primary", use_container_width=True)

# ── main ─────────────────────────────────────────────────────────────────────

st.title("SatQuery AI")
st.caption("Natural-language queries over satellite imagery · ISRO SIH 26167")

left, right = st.columns([1, 1])

with left:
    st.subheader("Input")
    if images_hwc:
        for k, v in images_hwc.items():
            st.image(v, caption=k, use_column_width=True)
    else:
        st.info("Pick a sample or upload an image in the sidebar.")

with right:
    st.subheader("Result")
    if not run:
        st.info("Choose a query and press **Run**.")

if run:
    if not images_hwc:
        st.error("No image selected.")
        st.stop()
    if not query.strip():
        st.error("Enter a query.")
        st.stop()

    payload = {k: to_chw(v) for k, v in images_hwc.items()}
    t0 = time.time()
    with st.spinner("Running pipeline…"):
        result = controller.run(images=payload, query=query)
    latency = (time.time() - t0) * 1000

    with right:
        if not result.get("success", True):
            st.error("Pipeline error: " + str(result.get("errors")))
        else:
            task = result.get("task", "?")
            st.markdown(f"**Task routed to:** `{task}`")
            st.markdown(f"### {result.get('text','')}")

            conf = float(result.get("confidence", 0.0))
            st.progress(min(max(conf, 0.0), 1.0),
                        text=f"Confidence {conf:.0%} · {latency:.0f} ms · "
                             f"status: {result.get('status','—')}")

            ev = result.get("spatial_evidence") or {}
            base_key = list(images_hwc)[0]

            if ev.get("type") == "bbox" and ev.get("bbox"):
                st.image(draw_box(images_hwc[base_key], ev["bbox"]),
                         caption="Grounding — CLIP patch localisation",
                         use_column_width=True)

            elif ev.get("type") == "mask" and ev.get("mask") is not None:
                base = images_hwc.get("image_t1", images_hwc[base_key])
                c1, c2 = st.columns(2)
                c1.image(base, caption="T1 (before)", use_column_width=True)
                c2.image(overlay(base, np.asarray(ev["mask"])),
                         caption="Predicted change (red)", use_column_width=True)
                gt = st.session_state.get("_gt")
                if gt is not None:
                    st.image(overlay(base, gt),
                             caption="Ground truth change mask",
                             use_column_width=True)

            meta = result.get("metadata", {}) or {}
            params = meta.get("parameters", {}) or {}
            st.caption(
                f"model `{meta.get('model','—')}` · "
                f"backbone `{meta.get('backbone','—')}` · "
                f"path `{params.get('fallback_level','—')}`"
            )
            if params.get("top_k"):
                st.caption("Top-3: " + ", ".join(
                    f"{l} {p:.0%}" for l, p in params["top_k"]))

    with st.expander("Execution trace — how the agent reached this answer"):
        st.json(result.get("trace", {}))

    dl = {k: v for k, v in result.items() if k != "spatial_evidence"}
    dl["spatial_evidence"] = {
        k: (None if k == "mask" else v)
        for k, v in (result.get("spatial_evidence") or {}).items()
    }
    st.download_button("⬇ Download result JSON",
                       data=json.dumps(dl, default=str, indent=2),
                       file_name=f"satquery_{result.get('task','result')}.json",
                       mime="application/json")
