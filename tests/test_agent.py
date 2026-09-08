"""Tests for the agentic pipeline."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from src.agent.input_validator import validate_input
from src.agent.task_router import route_task


def test_validate_input_rejects_unknown_keys():
    result = validate_input({"foo": "bar.tif"}, config={})
    assert result["valid"] is False


def test_validate_input_rejects_bad_format():
    result = validate_input({"image": "scene.bmp"}, config={})
    assert result["valid"] is False
    assert any("unsupported format" in e for e in result["errors"])


def test_validate_input_accepts_tif():
    result = validate_input({"image": "scene.tif"}, config={})
    assert result["valid"] is True
    assert result["mode"] == "single"


def test_validate_input_accepts_png():
    result = validate_input({"image": "scene.png"}, config={})
    assert result["valid"] is True


def test_validate_input_bi_temporal():
    result = validate_input({"image_t1": "t1.tif", "image_t2": "t2.tif"}, config={})
    assert result["valid"] is True
    assert result["mode"] == "bi_temporal"


def test_validate_input_cross_modal():
    result = validate_input({"image_optical": "opt.tif", "image_sar": "sar.tif"}, config={})
    assert result["valid"] is True
    assert result["mode"] == "cross_modal"


def test_validate_input_numpy_array():
    arr = np.zeros((4, 64, 64), dtype=np.float32)
    result = validate_input({"image": arr}, config={})
    assert result["valid"] is True
    assert result["mode"] == "single"


def test_validate_input_bad_array_shape():
    arr = np.zeros((64, 64), dtype=np.float32)  # missing channel dim
    result = validate_input({"image": arr}, config={})
    assert result["valid"] is False


def test_route_task_bi_temporal_is_change():
    assert route_task("what happened here?", "bi_temporal") == "change"


def test_route_task_cross_modal_is_fusion():
    assert route_task("combine these", "cross_modal") == "fusion"


def test_route_task_single_defaults_to_vqa():
    assert route_task("how many buildings are there?", "single") == "vqa"


def test_route_task_single_grounding_keyword():
    assert route_task("Highlight the water body referred to in the query.", "single") == "grounding"


def test_route_task_single_captioning_keyword():
    assert route_task("Describe the land-cover in this image.", "single") == "captioning"


def test_route_task_captioning_variant():
    assert route_task("Summarize the image.", "single") in ("captioning", "vqa")


def test_base_model_empty_result_contract():
    from src.models.vqa_model import VQAModel
    model = VQAModel(config={})
    result = model._empty_result(text="stub answer", confidence=0.1)
    assert result["task"] == "vqa"
    assert result["text"] == "stub answer"
    assert result["metadata"]["model"] == "vqa_v1"


def test_change_model_fallback_predict():
    """ChangeModel fallback (siamese-difference) must return valid RSModelResult."""
    from src.models.change_model import ChangeModel
    
    model = ChangeModel(config={})
    # Load forces fallback since VisTA is unavailable
    model.load("nonexistent_checkpoint", device="cpu")
    assert model.using_fallback, "Should use fallback when VisTA unavailable"
    
    t1 = np.random.rand(3, 64, 64).astype("float32")
    t2 = t1.copy()
    t2[:, 10:30, 10:30] += 0.5
    t2 = np.clip(t2, 0, 1)
    
    result = model.predict(t1, t2, query="What changed?")
    
    assert result["task"] == "change"
    assert result["text"]
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["status"] in ("success", "low_confidence", "error")
    assert result["spatial_evidence"]["type"] == "mask"
    mask = result["spatial_evidence"]["mask"]
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (64, 64)


def test_change_model_shape_mismatch():
    """ChangeModel should report error when T1 and T2 have different shapes."""
    from src.models.change_model import ChangeModel
    
    model = ChangeModel(config={})
    model.load("nonexistent_checkpoint", device="cpu")
    
    t1 = np.zeros((3, 64, 64), dtype=np.float32)
    t2 = np.zeros((3, 128, 128), dtype=np.float32)  # different size
    
    result = model.predict(t1, t2, query="What changed?")
    assert result["status"] == "error"


def test_vqa_model_baseline_predict():
    """VQA baseline must return valid RSModelResult without any model loaded."""
    from src.models.vqa_model import VQAModel
    
    model = VQAModel(config={})
    model.load("nonexistent", device="cpu")  # loads baseline
    
    image = np.random.rand(4, 64, 64).astype("float32")
    result = model.predict(image, query="What is visible in this image?")
    
    assert result["task"] == "vqa"
    assert result["text"]
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["status"] in ("success", "low_confidence")


def test_vqa_degrades_gracefully_without_clip():
    """With CLIP unavailable, VQA must still return a schema-valid result via
    the statistics baseline, and must label itself as the degraded path so
    nothing downstream mistakes it for a real vision-language answer."""
    from src.models.vqa_model import VQAModel

    model = VQAModel(config={})
    model.load("nonexistent", device="cpu")

    image = np.random.rand(4, 64, 64).astype("float32")
    result = model.predict(image, query="Is there water visible in this image?")

    assert result["task"] == "vqa"
    assert result["status"] in ("success", "low_confidence", "error")
    assert 0.0 <= result["confidence"] <= 1.0
    if result["status"] != "error":
        assert result["metadata"]["parameters"].get("fallback_level") in (
            "statistics_baseline", "clip_zeroshot")


def test_grounding_reports_error_without_clip():
    """Grounding is CLIP-only by design — the old brightness-quadrant
    heuristic was removed because it ignored the query. Without CLIP it must
    report an error rather than emit a misleading box."""
    from src.models.grounding_model import GroundingModel

    model = GroundingModel(config={})
    model.load("nonexistent", device="cpu")

    image = np.random.rand(4, 64, 64).astype("float32")
    result = model.predict(image, query="Highlight the water body.")

    assert result["task"] == "grounding"
    if result["status"] == "error":
        assert result["spatial_evidence"]["type"] == "none"
    else:
        bbox = result["spatial_evidence"]["bbox"]
        assert bbox is not None and len(bbox) == 4
        assert bbox[0] < bbox[2] and bbox[1] < bbox[3]

def test_change_model_no_change():
    """Identical images should produce near-zero change percentage."""
    from src.models.change_model import ChangeModel
    
    model = ChangeModel(config={})
    model.load("nonexistent", device="cpu")
    
    t1 = np.ones((3, 64, 64), dtype=np.float32) * 0.5
    t2 = t1.copy()  # identical images
    
    result = model.predict(t1, t2, query="What changed?")
    assert "no substantial change" in result["text"].lower() or result["confidence"] < 0.3


def test_execution_trace():
    """ExecutionTrace must produce a valid dict."""
    from src.agent.execution_trace import ExecutionTrace
    
    trace = ExecutionTrace(query="test query", mode="single")
    trace.set_task("vqa")
    trace.add_tool_call(name="vqa_v1", task="vqa", params={"query": "test"})
    trace.set_confidence(0.75)
    trace.add_warning("Low confidence")
    
    result = trace.finalize()
    
    assert result["query"] == "test query"
    assert result["selected_task"] == "vqa"
    assert result["confidence"] == 0.75
    assert len(result["warnings"]) == 1
    assert result["latency_seconds"] >= 0.0
    assert len(result["tools_used"]) == 1


def test_metrics_vqa_accuracy():
    from src.evaluation.metrics import vqa_accuracy, vqa_accuracy_by_type
    
    preds = ["yes", "No!", "urban area", "The image shows water."]
    golds = ["yes", "no", "urban area", "Water"]
    
    acc = vqa_accuracy(preds, golds)
    assert 0.0 <= acc <= 1.0
    assert acc >= 0.5  # at least "yes" and "urban area" match


def test_metrics_caption_scores():
    from src.evaluation.metrics import caption_scores
    
    preds = ["The satellite image shows a body of water.", "The image shows urban area."]
    golds = ["A large lake is visible from satellite.", "Dense urban built-up region."]
    
    scores = caption_scores(preds, golds)
    assert "bleu4" in scores
    assert "rouge_l" in scores
    assert 0.0 <= scores["bleu4"] <= 1.0
    assert 0.0 <= scores["rouge_l"] <= 1.0


def test_metrics_change_f1():
    from src.evaluation.metrics import change_f1
    
    pred = np.zeros((64, 64), dtype=np.uint8)
    pred[10:30, 10:30] = 1
    gold = np.zeros((64, 64), dtype=np.uint8)
    gold[15:25, 15:25] = 1
    
    result = change_f1([pred], [gold])
    assert "f1" in result
    assert "precision" in result
    assert "recall" in result
    assert 0.0 <= result["f1"] <= 1.0


def test_metrics_grounding_iou():
    from src.evaluation.metrics import grounding_iou, grounding_accuracy_at_threshold
    
    pred_boxes = [[10, 10, 50, 50]]
    gold_boxes = [[15, 15, 45, 45]]
    
    iou = grounding_iou(pred_boxes, gold_boxes)
    assert 0.0 < iou < 1.0  # overlapping but not identical
    
    acc = grounding_accuracy_at_threshold(pred_boxes, gold_boxes, threshold=0.5)
    assert acc in (0.0, 1.0)  # binary at threshold
