"""
Captioning specialist — CLIP-backed, templated output.

Rationale: generating free-form text needs a language decoder (a 7B VLM),
which we deliberately cut. Instead we take CLIP's top-2 land-cover
predictions and fill a template. The caption is therefore fully grounded in
what the model actually perceived, and the templating is stated openly in
the metadata rather than presented as generation.

This is a real, defensible design for a prototype: the perception is
learned, the surface realisation is deterministic.
"""

import time
import numpy as np

from src.common.schemas import RSModelResult, SpatialEvidence, ResultMetadata, empty_result
from src.models.base_model import BaseRSModel
from src.models.clip_backend import get_shared_clip, RS_LABELS, LABEL_DISPLAY

# Class-specific phrasing so captions do not all read identically.
_CONTEXT = {
    "annual cropland": "regularly shaped field parcels under seasonal cultivation",
    "forest": "dense tree cover with a continuous canopy",
    "herbaceous vegetation": "low-growing vegetation with no clear field boundaries",
    "a highway or road": "a linear transport corridor cutting across the scene",
    "an industrial area": "large flat-roofed structures and open hardstanding",
    "pasture land": "open grazing land with irregular boundaries",
    "permanent cropland": "structured planting typical of orchards or vineyards",
    "a residential area": "dense small-footprint buildings and a street network",
    "a river": "a sinuous water channel crossing the scene",
    "a sea or lake": "a large uniform body of open water",
}


class CaptioningModel(BaseRSModel):
    name = "caption_v1"
    task = "captioning"
    backbone = "CLIP-ViT-B/32"

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.clip = None
        self.checkpoint = "openai/clip-vit-base-patch32"

    def load(self, checkpoint_path: str = None, device: str = "cpu") -> None:
        cfg = (self.config.get("clip") or {})
        model_id = cfg.get("model_id", self.checkpoint)
        template = cfg.get("prompt_template", "a satellite photo of {}")
        self.clip = get_shared_clip(model_id, template, device=device)
        self.checkpoint = model_id if self.clip.available else "statistics_baseline"
        self.backbone = "CLIP-ViT-B/32" if self.clip.available else "statistics_baseline"

    def predict(self, image: np.ndarray = None, query: str = "", **kwargs) -> dict:
        t0 = time.time()
        if image is None:
            return empty_result(self.task, "No image provided.", status="error")

        if self.clip is None or not self.clip.available:
            return empty_result(
                self.task,
                "Captioning requires the CLIP backend, which is unavailable.",
                status="error",
            )

        try:
            ranked = self.clip.classify(image, RS_LABELS)
        except Exception as exc:                       # noqa: BLE001
            return empty_result(self.task, f"Captioning failed: {exc}", status="error")

        (top, p1), (second, p2) = ranked[0], ranked[1]
        d1 = LABEL_DISPLAY.get(top, top)
        d2 = LABEL_DISPLAY.get(second, second)
        ctx = _CONTEXT.get(top, "")

        caption = f"This satellite image primarily shows {d1}"
        if ctx:
            caption += f", characterised by {ctx}"
        caption += "."
        if p2 > 0.15:
            caption += f" Areas consistent with {d2} are also present."

        status = "success" if p1 >= 0.35 else "low_confidence"
        return RSModelResult(
            task=self.task,
            text=caption,
            confidence=float(p1),
            spatial_evidence=SpatialEvidence(type="none", source="image"),
            metadata=ResultMetadata(
                model=self.name,
                backbone=self.backbone,
                checkpoint=self.checkpoint,
                dataset="zero-shot (no task-specific training)",
                input_modalities=["optical"],
                parameters={
                    "fallback_level": "clip_zeroshot",
                    "surface_realisation": "templated",
                    "top_k": [[l, round(p, 4)] for l, p in ranked[:3]],
                },
            ),
            status=status,
            inference_seconds=time.time() - t0,
        ).to_dict()
