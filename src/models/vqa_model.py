"""
VQA specialist — CLIP-backed.

Design note (important for the write-up):
  An earlier iteration answered by matching keywords in the *question*
  against a fixed answer list and using mean pixel brightness to choose
  between them. That is not visual question answering — the answer barely
  depends on the image. This version routes every query through CLIP, so
  the answer is conditioned on the pixels. The statistics baseline is
  retained only for the case where CLIP cannot be loaded at all, and it
  labels itself as such in the result metadata.

Two question shapes are handled:
  * open questions  -> zero-shot classify over the RS label vocabulary
  * yes/no questions -> score the concept named in the question against a
                        set of distractors and threshold the probability
"""

import re
import time
from typing import List, Optional, Tuple

import numpy as np

from src.common.schemas import RSModelResult, SpatialEvidence, ResultMetadata, empty_result
from src.models.base_model import BaseRSModel
from src.models.clip_backend import (
    get_shared_clip, RS_LABELS, LABEL_DISPLAY,
)

_YES_NO_PREFIXES = ("is ", "are ", "does ", "do ", "has ", "have ",
                    "can ", "was ", "were ", "any ", "there ")

# Concepts a yes/no question might ask about, mapped to the CLIP prompt used
# to score them and the distractors they compete against.
_CONCEPTS = {
    "water":      ("a sea or lake", ["a river"]),
    "river":      ("a river", ["a sea or lake"]),
    "lake":       ("a sea or lake", ["a river"]),
    "sea":        ("a sea or lake", ["a river"]),
    "forest":     ("forest", []),
    "tree":       ("forest", []),
    "vegetation": ("herbaceous vegetation", ["forest", "pasture land"]),
    "crop":       ("annual cropland", ["permanent cropland"]),
    "farm":       ("annual cropland", ["permanent cropland", "pasture land"]),
    "agricult":   ("annual cropland", ["permanent cropland", "pasture land"]),
    "urban":      ("a residential area", ["an industrial area"]),
    "built":      ("a residential area", ["an industrial area"]),
    "building":   ("a residential area", ["an industrial area"]),
    "residential": ("a residential area", []),
    "industrial": ("an industrial area", []),
    "road":       ("a highway or road", []),
    "highway":    ("a highway or road", []),
    "pasture":    ("pasture land", []),
    "grass":      ("pasture land", ["herbaceous vegetation"]),
}


def _is_yes_no(query: str) -> bool:
    q = query.lower().strip()
    return any(q.startswith(p) for p in _YES_NO_PREFIXES)


def _concept_in_query(query: str) -> Optional[Tuple[str, List[str]]]:
    """Find which land-cover concept a yes/no question is asking about."""
    q = query.lower()
    for key, (label, distractors) in _CONCEPTS.items():
        if key in q:
            return label, distractors
    return None


def _stats_baseline(image: np.ndarray, query: str) -> Tuple[str, float]:
    """Last-resort answer when CLIP is unavailable. Clearly weaker; the
    result metadata flags it so nothing downstream mistakes it for a real
    vision-language answer."""
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] <= 4:
        arr = arr.transpose(1, 2, 0)
    if arr.max() > 1.0:
        arr = arr / 255.0
    mean_v, std_v = float(arr.mean()), float(arr.std())
    if mean_v < 0.30:
        return "a water body", 0.30
    if std_v > 0.18 and mean_v > 0.45:
        return "a built-up area", 0.30
    return "vegetated land", 0.30


class VQAModel(BaseRSModel):
    name = "vqa_v1"
    task = "vqa"
    backbone = "CLIP-ViT-B/32"

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.clip = None
        self.checkpoint = "openai/clip-vit-base-patch32"
        self._using_fallback = True

    def load(self, checkpoint_path: str = None, device: str = "cpu") -> None:
        cfg = (self.config.get("clip") or {})
        model_id = cfg.get("model_id", self.checkpoint)
        template = cfg.get("prompt_template", "a satellite photo of {}")
        self.clip = get_shared_clip(model_id, template, device=device)
        self._using_fallback = not self.clip.available
        self.checkpoint = model_id if self.clip.available else "statistics_baseline"
        self.backbone = "CLIP-ViT-B/32" if self.clip.available else "statistics_baseline"

    def predict(self, image: np.ndarray = None, query: str = "", **kwargs) -> dict:
        t0 = time.time()
        if image is None:
            return empty_result(self.task, "No image provided.", status="error")

        try:
            if self.clip is not None and self.clip.available:
                text, conf, params = self._clip_answer(image, query)
            else:
                text, conf = _stats_baseline(image, query)
                params = {"fallback_level": "statistics_baseline"}
        except Exception as exc:                       # noqa: BLE001
            return empty_result(self.task, f"VQA failed: {exc}", status="error")

        status = "success" if conf >= 0.35 else "low_confidence"
        return RSModelResult(
            task=self.task,
            text=text,
            confidence=float(conf),
            spatial_evidence=SpatialEvidence(type="none", source="image"),
            metadata=ResultMetadata(
                model=self.name,
                backbone=self.backbone,
                checkpoint=self.checkpoint,
                dataset="zero-shot (no task-specific training)",
                input_modalities=["optical"],
                parameters=params,
            ),
            status=status,
            inference_seconds=time.time() - t0,
        ).to_dict()

    # -- internals ---------------------------------------------------------

    def _clip_answer(self, image, query):
        if _is_yes_no(query):
            hit = _concept_in_query(query)
            if hit is not None:
                target, distractors = hit
                candidates = [target] + (distractors or [])
                others = [l for l in RS_LABELS if l not in candidates][:6]
                ranked = self.clip.classify(image, candidates + others)
                score = sum(p for l, p in ranked if l in candidates)
                top_label, _ = ranked[0]
                if score >= 0.35:
                    answer = f"Yes — the image appears to contain {LABEL_DISPLAY.get(target, target)}."
                    conf = score
                else:
                    answer = (f"No — the image is dominated by "
                              f"{LABEL_DISPLAY.get(top_label, top_label)} instead.")
                    conf = 1.0 - score
                return answer, conf, {"fallback_level": "clip_zeroshot",
                                      "mode": "yes_no", "concept": target}

        ranked = self.clip.classify(image, RS_LABELS)
        top, p1 = ranked[0]
        second, p2 = ranked[1]
        disp1 = LABEL_DISPLAY.get(top, top)
        disp2 = LABEL_DISPLAY.get(second, second)

        if p1 - p2 < 0.10:
            text = (f"The image most likely shows {disp1}, though {disp2} "
                    f"is a close alternative.")
        else:
            text = f"The image shows {disp1}."
        return text, p1, {"fallback_level": "clip_zeroshot",
                          "mode": "open",
                          "top_k": [[l, round(p, 4)] for l, p in ranked[:3]]}
