"""
Grounding specialist — CLIP sliding-window localisation.

Replaces an earlier heuristic that snapped the box to the brightest/darkest
image quadrant. That produced a box that was always exactly a quarter of the
image and was driven by brightness rather than by the query, which is not
grounding in any meaningful sense.

Method here:
  1. Split the image into a grid x grid tile grid.
  2. Score every tile against the queried phrase with CLIP.
  3. Take the peak tile, then grow the region to include neighbouring tiles
     scoring above `expand_ratio` of the peak.
  4. Return the tight box around that region.

Known limitation, stated openly: localisation is at tile resolution, so the
box edges snap to grid lines. On the 64x64 EuroSAT chips a 4x4 grid means
16x16 pixel tiles, which is coarse; on the larger OSCD tiles it is much
more meaningful. This is a resolution limit of a training-free method, not
a bug.
"""

import re
import time
from typing import List, Tuple, Optional

import numpy as np

from src.common.schemas import RSModelResult, SpatialEvidence, ResultMetadata, empty_result
from src.models.base_model import BaseRSModel
from src.models.clip_backend import get_shared_clip

_STOPWORDS = {
    "highlight", "show", "find", "locate", "where", "is", "are", "the", "a",
    "an", "in", "on", "of", "this", "image", "satellite", "please", "me",
    "point", "out", "identify", "region", "area", "and", "to", "at", "which",
}


def extract_target_phrase(query: str) -> str:
    """Pull the thing to be localised out of the instruction.

    'Highlight the water body in the image.' -> 'water body'
    Falls back to the whole query if nothing survives filtering.
    """
    q = query.lower().strip().rstrip(".?!")
    tokens = re.findall(r"[a-z]+", q)
    kept = [t for t in tokens if t not in _STOPWORDS]
    return " ".join(kept) if kept else q


def parse_bbox_from_text(text: str) -> Optional[List[float]]:
    """Parse '[x1, y1, x2, y2]' out of model text. Retained from the original
    implementation — still used if a text-generating backbone is swapped in
    later."""
    m = re.search(r"\[?\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]?", text)
    if not m:
        return None
    vals = [float(g) for g in m.groups()]
    return vals if vals[2] > vals[0] and vals[3] > vals[1] else None


def clip_bbox(bbox: List[float], lo: float = 0.0, hi: float = 1.0) -> List[float]:
    return [min(max(v, lo), hi) for v in bbox]


def rescale_bbox(bbox: List[float], from_size: Tuple[int, int],
                 to_size: Tuple[int, int]) -> List[float]:
    fw, fh = from_size
    tw, th = to_size
    return [bbox[0] / fw * tw, bbox[1] / fh * th, bbox[2] / fw * tw, bbox[3] / fh * th]


def visualize_bbox(image_hwc: np.ndarray, bbox: List[float],
                   color=(255, 0, 0), width: int = 3) -> np.ndarray:
    """Draw a normalised [x1,y1,x2,y2] box on an HWC uint8 image."""
    from PIL import Image as PILImage, ImageDraw
    img = PILImage.fromarray(np.asarray(image_hwc).astype(np.uint8)).convert("RGB")
    w, h = img.size
    x1, y1, x2, y2 = bbox
    if max(bbox) <= 1.0:                       # normalised -> pixels
        x1, x2 = x1 * w, x2 * w
        y1, y2 = y1 * h, y2 * h
    ImageDraw.Draw(img).rectangle([x1, y1, x2, y2], outline=color, width=width)
    return np.array(img)


def _region_from_heatmap(heat: np.ndarray, expand_ratio: float = 0.80
                         ) -> Tuple[List[float], float]:
    """Peak tile plus neighbours above expand_ratio * peak -> normalised box."""
    grid = heat.shape[0]
    peak = float(heat.max())
    selected = heat >= (peak * expand_ratio) if peak > 0 else heat >= 1.0
    if not selected.any():
        gy, gx = np.unravel_index(int(np.argmax(heat)), heat.shape)
        selected = np.zeros_like(heat, dtype=bool)
        selected[gy, gx] = True

    ys, xs = np.where(selected)
    y1 = float(ys.min()) / grid
    y2 = float(ys.max() + 1) / grid
    x1 = float(xs.min()) / grid
    x2 = float(xs.max() + 1) / grid

    # Confidence: how far the peak tile stands out from the mean tile score.
    # Scaled by 0.5 so a clean single-tile peak lands near 0.75 rather than
    # pinning at the ceiling — an untrained localiser should not report 95%.
    mean_score = float(heat.mean())
    margin = peak - mean_score
    confidence = float(np.clip(0.30 + 0.5 * margin, 0.0, 0.90))
    return [float(v) for v in clip_bbox([x1, y1, x2, y2])], confidence


class GroundingModel(BaseRSModel):
    name = "grounding_v1"
    task = "grounding"
    backbone = "CLIP-ViT-B/32"

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.clip = None
        self.checkpoint = "openai/clip-vit-base-patch32"
        gcfg = ((config or {}).get("models", {}).get("grounding", {}))
        self.grid = int(gcfg.get("grid", 4))
        self.expand_ratio = float(gcfg.get("expand_ratio", 0.80))

    def load(self, checkpoint_path: str = None, device: str = "cpu") -> None:
        cfg = (self.config.get("clip") or {})
        model_id = cfg.get("model_id", self.checkpoint)
        template = cfg.get("prompt_template", "a satellite photo of {}")
        self.clip = get_shared_clip(model_id, template, device=device)
        self.checkpoint = model_id if self.clip.available else "unavailable"
        self.backbone = "CLIP-ViT-B/32" if self.clip.available else "unavailable"

    def predict(self, image: np.ndarray = None, query: str = "", **kwargs) -> dict:
        t0 = time.time()
        if image is None:
            return empty_result(self.task, "No image provided.", status="error")
        if self.clip is None or not self.clip.available:
            return empty_result(
                self.task,
                "Grounding requires the CLIP backend, which is unavailable.",
                status="error",
            )

        target = extract_target_phrase(query)
        try:
            heat = self.clip.score_patches(image, target, grid=self.grid)
        except Exception as exc:                       # noqa: BLE001
            return empty_result(self.task, f"Grounding failed: {exc}", status="error")

        bbox, conf = _region_from_heatmap(heat, self.expand_ratio)
        text = (f"Localised '{target}' to the highlighted region "
                f"(normalised box {[round(v, 3) for v in bbox]}).")

        status = "success" if conf >= 0.35 else "low_confidence"
        return RSModelResult(
            task=self.task,
            text=text,
            confidence=float(conf),
            spatial_evidence=SpatialEvidence(
                type="bbox", source="image", bbox=bbox,
                mask=heat.astype(np.float32),     # tile heatmap, for overlay
            ),
            metadata=ResultMetadata(
                model=self.name,
                backbone=self.backbone,
                checkpoint=self.checkpoint,
                dataset="zero-shot (no task-specific training)",
                input_modalities=["optical"],
                parameters={
                    "fallback_level": "clip_patch_sliding_window",
                    "target_phrase": target,
                    "grid": self.grid,
                    "expand_ratio": self.expand_ratio,
                    "localisation_resolution": f"1/{self.grid} of image side",
                },
            ),
            status=status,
            inference_seconds=time.time() - t0,
        ).to_dict()
