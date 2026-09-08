"""Tests for the CLIP-backed specialists.

The load-bearing test here is test_answer_depends_on_image: an earlier
keyword-matching implementation passed every other test while being blind to
the image, so that property is asserted explicitly.
"""
import numpy as np
import pytest

import src.models.clip_backend as cb


class _FakeCLIP:
    available = True
    model_id = "fake"
    prompt_template = "a satellite photo of {}"

    def classify(self, image, labels=None):
        labels = labels or cb.RS_LABELS
        arr = cb.CLIPBackend._to_hwc_uint8(image).astype(float) / 255.0
        seed = {"a sea or lake": arr[..., 2].mean() * 3,
                "forest": arr[..., 1].mean() * 3,
                "a residential area": arr.mean() * 2}
        s = np.array([seed.get(l, 0.25) for l in labels])
        e = np.exp(s * 3)
        return sorted(zip(labels, (e / e.sum()).tolist()), key=lambda t: -t[1])

    def score_patches(self, image, text, grid=4):
        arr = cb.CLIPBackend._to_hwc_uint8(image).astype(float) / 255.0
        h, w = arr.shape[:2]
        ph, pw = max(h // grid, 1), max(w // grid, 1)
        heat = np.zeros((grid, grid), np.float32)
        for gy in range(grid):
            for gx in range(grid):
                p = arr[gy * ph:(gy + 1) * ph, gx * pw:(gx + 1) * pw]
                heat[gy, gx] = p.mean() if p.size else 0.0
        r = heat.max() - heat.min()
        return (heat - heat.min()) / r if r > 1e-8 else np.zeros_like(heat)


@pytest.fixture(autouse=True)
def _patch_clip(monkeypatch):
    monkeypatch.setattr(cb, "get_shared_clip", lambda *a, **k: _FakeCLIP())
    import src.models.vqa_model as v
    import src.models.captioning_model as c
    import src.models.grounding_model as g
    for m in (v, c, g):
        monkeypatch.setattr(m, "get_shared_clip", lambda *a, **k: _FakeCLIP())


def _img(rgb):
    a = np.zeros((64, 64, 3), np.uint8)
    for i, v in enumerate(rgb):
        a[..., i] = v
    return a.transpose(2, 0, 1).astype(np.float32) / 255.0


WATER = _img((30, 40, 190))
FOREST = _img((40, 160, 40))


def test_answer_depends_on_image():
    from src.models.vqa_model import VQAModel
    m = VQAModel(config={}); m.load()
    q = "What is the land cover in this image?"
    assert m.predict(image=WATER, query=q)["text"] != \
           m.predict(image=FOREST, query=q)["text"]


def test_vqa_schema():
    from src.models.vqa_model import VQAModel
    m = VQAModel(config={}); m.load()
    r = m.predict(image=WATER, query="What is here?")
    for k in ("task", "text", "confidence", "spatial_evidence", "metadata", "status"):
        assert k in r
    assert 0.0 <= r["confidence"] <= 1.0
    assert r["task"] == "vqa"


def test_vqa_yes_no():
    from src.models.vqa_model import VQAModel
    m = VQAModel(config={}); m.load()
    r = m.predict(image=WATER, query="Is there water in this image?")
    assert r["metadata"]["parameters"]["mode"] == "yes_no"
    assert r["text"].startswith(("Yes", "No"))


def test_vqa_missing_image_errors():
    from src.models.vqa_model import VQAModel
    m = VQAModel(config={}); m.load()
    assert m.predict(image=None, query="x")["status"] == "error"


def test_caption_mentions_class():
    from src.models.captioning_model import CaptioningModel
    m = CaptioningModel(config={}); m.load()
    assert "forest" in m.predict(image=FOREST, query="describe")["text"].lower()


def test_grounding_bbox_valid_and_json_safe():
    import json
    from src.models.grounding_model import GroundingModel
    m = GroundingModel(config={}); m.load()
    r = m.predict(image=WATER, query="Highlight the water body.")
    b = r["spatial_evidence"]["bbox"]
    assert b and b[0] < b[2] and b[1] < b[3]
    assert 0.0 <= min(b) and max(b) <= 1.0
    assert all(isinstance(v, float) for v in b)
    json.dumps(b)


def test_target_phrase_extraction():
    from src.models.grounding_model import extract_target_phrase
    assert extract_target_phrase("Highlight the water body in the image.") == "water body"
    assert "river" in extract_target_phrase("Where is the river?")


def test_heatmap_region_selection():
    from src.models.grounding_model import _region_from_heatmap
    h = np.zeros((4, 4), np.float32); h[2, 1] = 1.0; h[2, 2] = 0.9
    b, c = _region_from_heatmap(h, 0.8)
    assert b == [0.25, 0.5, 0.75, 0.75]
    assert 0.0 <= c <= 0.9


def test_chw_and_hwc_both_accepted():
    conv = cb.CLIPBackend._to_hwc_uint8
    assert conv(np.zeros((3, 32, 32), np.float32)).shape == (32, 32, 3)
    assert conv(np.zeros((32, 32, 3), np.uint8)).shape == (32, 32, 3)
    assert conv(np.zeros((4, 32, 32), np.float32)).shape == (32, 32, 3)  # drops NIR
    assert conv(np.zeros((2, 32, 32), np.float32)).shape == (32, 32, 3)  # SAR VV/VH
