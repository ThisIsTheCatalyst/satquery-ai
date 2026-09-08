"""
Common contracts — THIS FILE IS FROZEN.

Rule for the team (2-day ML sprint): nobody changes these shapes on their
own once agreed. Model internals (architecture, weights, prompts) can
change freely inside each person's notebook/module — but:

  1. every dataset loader returns `ImageSample` / `PairSample` / `FusionSample`
     (Layer A — data loading), and
  2. every model's predict() returns something that satisfies `RSModelResult`
     (Layer B -> Layer C boundary)

...so the four independent notebooks merge into the agent with zero rewrites.

Design notes (why this shape, not a simpler one):
- `text` not `answer` — captioning/description isn't "answering" anything;
  keep the field generic across all 5 tasks.
- `SpatialEvidence.source` is the single most important field here. Without
  it, a bbox from a change-detection result is ambiguous: is it in T1's or
  T2's coordinate frame? For fusion: optical or SAR pixel space? Every
  producer of evidence MUST set `source` correctly.
- `status` lets a model signal "ran but low confidence" vs "errored" vs
  "succeeded" explicitly, instead of the agent inferring it from confidence
  alone.
- Deliberately NOT adding more fields than this (no per-span confidence, no
  nested timing breakdowns, etc.) until something concrete needs them —
  every field here is one all 4 people must populate correctly.

Owner: shared / agreed by all 4 people before starting ML work.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict
import numpy as np


# ---------------------------------------------------------------------------
# Output contract — what every specialist model's predict() must return.
# ---------------------------------------------------------------------------

@dataclass
class SpatialEvidence:
    """Spatial evidence backing an answer — bounding box and/or mask, always
    tagged with which image it refers to via `source`.
    """
    type: str = "none"            # "bbox" | "mask" | "none"
    source: str = "image"         # "image" | "image_t1" | "image_t2" |
                                   # "optical" | "sar" | "fused"
    bbox: Optional[List[float]] = None   # [x1,y1,x2,y2] pixel coords, in `source`'s frame
    mask: Optional[np.ndarray] = None    # HxW array, same coordinate frame as `source`


@dataclass
class ResultMetadata:
    """Provenance info the agent/trace/report need."""
    model: str = ""                     # e.g. "GeoChat", "VisTA", "OpticalSARFusion"
    backbone: str = ""                  # underlying architecture, if different from `model`
    checkpoint: str = ""                # path/tag of the weights used
    dataset: str = ""                   # dataset used for adaptation/training
    input_modalities: List[str] = field(default_factory=list)  # e.g. ["optical"], ["optical","sar"]
    parameters: dict = field(default_factory=dict)             # task params actually applied


@dataclass
class RSModelResult:
    """The single return type every specialist model's predict() must produce.

    Usage in a model:
        return RSModelResult(
            task="vqa",
            text="The image shows a mix of built-up area and cropland.",
            confidence=0.82,
            spatial_evidence=SpatialEvidence(),
            metadata=ResultMetadata(model="GeoChat", checkpoint="geochat_lora_v1",
                                     dataset="BigEarthNet.txt",
                                     input_modalities=["optical"]),
        ).to_dict()
    """
    task: str                              # "vqa" | "captioning" | "grounding" | "change" | "fusion"
    text: str
    confidence: float
    spatial_evidence: SpatialEvidence = field(default_factory=SpatialEvidence)
    metadata: ResultMetadata = field(default_factory=ResultMetadata)
    status: str = "success"                # "success" | "error" | "low_confidence"
    inference_seconds: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        # asdict() recurses through numpy arrays and can mangle them —
        # patch the mask back in directly rather than trusting dataclasses'
        # recursive conversion.
        d["spatial_evidence"]["mask"] = self.spatial_evidence.mask
        return d


def empty_result(task: str, text: str = "", confidence: float = 0.0,
                  model: str = "", checkpoint: str = "", dataset: str = "",
                  status: str = "error") -> dict:
    """Convenience helper for stub/placeholder/error responses that still
    satisfy the contract. Default status="error" since this is normally
    called from a caught-exception or not-yet-implemented path; pass
    status="success" explicitly for legitimate stub demos.
    """
    return RSModelResult(
        task=task,
        text=text,
        confidence=confidence,
        spatial_evidence=SpatialEvidence(),
        metadata=ResultMetadata(model=model, checkpoint=checkpoint, dataset=dataset),
        status=status,
    ).to_dict()


# ---------------------------------------------------------------------------
# Input contract — what dataset loaders return (Layer A).
#
# These are for the DATA LOADING layer only. Model predict() signatures stay
# explicit kwargs (image=..., query=..., image_t1=..., image_optical=..., etc.)
# rather than taking one of these objects directly — kwargs surface typos as
# immediate TypeErrors and are self-documenting at each model's call site.
# Typed samples exist so four people don't invent four different dict/tuple
# shapes when reading datasets; convert sample -> kwargs at the call site.
# ---------------------------------------------------------------------------

@dataclass
class ImageSample:
    """A single optical/multispectral OR SAR image + its provenance."""
    image: np.ndarray
    modality: str                  # "optical" | "sar"
    sensor: str = "unknown"        # "Sentinel-2" | "Sentinel-1" | "Cartosat-2S" | "RISAT" | "unknown"
    metadata: dict = field(default_factory=dict)   # crs, bounds, resolution, source_path, ...
    query: Optional[str] = None
    label: Optional[object] = None  # task-dependent ground truth (answer/caption/bbox/labels)


@dataclass
class PairSample:
    """Two spatially-corresponding images of the same area at different
    times — bi-temporal / change tasks.
    """
    image_t1: np.ndarray
    image_t2: np.ndarray
    sensor: str = "unknown"
    metadata_t1: dict = field(default_factory=dict)
    metadata_t2: dict = field(default_factory=dict)
    query: Optional[str] = None
    label: Optional[object] = None  # change answer / change mask


@dataclass
class FusionSample:
    """Co-registered optical + SAR pair of the same area — cross-modal tasks."""
    optical: np.ndarray
    sar: np.ndarray
    optical_sensor: str = "unknown"
    sar_sensor: str = "unknown"
    metadata: dict = field(default_factory=dict)
    query: Optional[str] = None
    label: Optional[object] = None  # multi-label land-cover targets, mask, etc.
