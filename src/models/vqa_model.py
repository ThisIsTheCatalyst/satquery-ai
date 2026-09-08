"""
VQA specialist — CLIP-backed with lightweight fallback.

The model supports two execution modes:

1. CLIP zero-shot VQA:
   Used when explicitly enabled through configuration and the CLIP
   backend is available.

2. Statistics / RGB baseline:
   Used automatically when CLIP is disabled or unavailable.
   This is intentionally lightweight and requires no model checkpoint.

The CLIP path supports:
    - open-ended questions
    - yes/no questions

The fallback path also preserves the question mode in metadata and
uses image-dependent RGB/statistical information rather than relying
only on the question text.
"""

import os
import time
from typing import List, Optional, Tuple

import numpy as np

from src.common.schemas import (
    RSModelResult,
    SpatialEvidence,
    ResultMetadata,
    empty_result,
)
from src.models.base_model import BaseRSModel
from src.models.clip_backend import (
    get_shared_clip,
    RS_LABELS,
    LABEL_DISPLAY,
)


# ---------------------------------------------------------------------------
# Yes/no question detection
# ---------------------------------------------------------------------------

_YES_NO_PREFIXES = (
    "is ",
    "are ",
    "does ",
    "do ",
    "has ",
    "have ",
    "can ",
    "was ",
    "were ",
    "any ",
    "there ",
)


# ---------------------------------------------------------------------------
# Concepts used by yes/no questions
# ---------------------------------------------------------------------------

_CONCEPTS = {
    "water": (
        "a sea or lake",
        ["a river"],
    ),
    "river": (
        "a river",
        ["a sea or lake"],
    ),
    "lake": (
        "a sea or lake",
        ["a river"],
    ),
    "sea": (
        "a sea or lake",
        ["a river"],
    ),
    "forest": (
        "forest",
        [],
    ),
    "tree": (
        "forest",
        [],
    ),
    "vegetation": (
        "herbaceous vegetation",
        ["forest", "pasture land"],
    ),
    "crop": (
        "annual cropland",
        ["permanent cropland"],
    ),
    "farm": (
        "annual cropland",
        ["permanent cropland", "pasture land"],
    ),
    "agricult": (
        "annual cropland",
        ["permanent cropland", "pasture land"],
    ),
    "urban": (
        "a residential area",
        ["an industrial area"],
    ),
    "built": (
        "a residential area",
        ["an industrial area"],
    ),
    "building": (
        "a residential area",
        ["an industrial area"],
    ),
    "residential": (
        "a residential area",
        [],
    ),
    "industrial": (
        "an industrial area",
        [],
    ),
    "road": (
        "a highway or road",
        [],
    ),
    "highway": (
        "a highway or road",
        [],
    ),
    "pasture": (
        "pasture land",
        [],
    ),
    "grass": (
        "pasture land",
        ["herbaceous vegetation"],
    ),
}


# ---------------------------------------------------------------------------
# Question helpers
# ---------------------------------------------------------------------------

def _is_yes_no(query: str) -> bool:
    """Return True when the query has a yes/no question form."""
    q = (query or "").lower().strip()
    return any(q.startswith(prefix) for prefix in _YES_NO_PREFIXES)


def _concept_in_query(
    query: str,
) -> Optional[Tuple[str, List[str]]]:
    """
    Find the land-cover concept referenced by a yes/no question.

    Returns:
        (target_label, distractor_labels), or None.
    """
    q = (query or "").lower()

    for key, (label, distractors) in _CONCEPTS.items():
        if key in q:
            return label, distractors

    return None


# ---------------------------------------------------------------------------
# Image preparation
# ---------------------------------------------------------------------------

def _prepare_image(image: np.ndarray) -> np.ndarray:
    """
    Convert an input image into HWC floating-point representation.

    Supports:
        CHW: (C,H,W)
        HWC: (H,W,C)

    Values are normalized approximately to [0,1].
    """

    arr = np.asarray(image, dtype=np.float32)

    if arr.size == 0:
        return np.zeros((1, 1, 3), dtype=np.float32)

    # Replace invalid values before calculating statistics.
    arr = np.nan_to_num(
        arr,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    # CHW -> HWC for common multispectral inputs.
    if arr.ndim == 3:
        # Typical satellite input: (3,H,W) or (4,H,W)
        if arr.shape[0] <= 4 and arr.shape[1] > 4 and arr.shape[2] > 4:
            arr = arr.transpose(1, 2, 0)

        # HWC is already correct otherwise.

    elif arr.ndim == 2:
        # Grayscale -> three channels.
        arr = np.stack([arr, arr, arr], axis=-1)

    else:
        raise ValueError(
            f"Unsupported image shape: {arr.shape}. "
            "Expected CHW or HWC image."
        )

    # If there are more than 3 channels, use the first three optical
    # channels for this lightweight RGB-style baseline.
    if arr.shape[-1] > 3:
        arr = arr[..., :3]

    # If fewer than 3 channels, replicate the available channel.
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    elif arr.shape[-1] == 2:
        arr = np.concatenate(
            [arr, arr[..., :1]],
            axis=-1,
        )

    # Normalize common 8-bit / integer-like inputs.
    max_value = float(np.max(arr))

    if max_value > 1.0:
        arr = arr / 255.0

    return np.clip(arr, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Lightweight fallback
# ---------------------------------------------------------------------------

def _stats_baseline(
    image: np.ndarray,
    query: str,
) -> Tuple[str, float]:
    """
    Lightweight image-dependent fallback.

    This is not a trained VQA model. It is a deterministic baseline used
    when CLIP is unavailable.

    The prediction depends on:
        - RGB channel means
        - overall brightness
        - channel dominance
        - image variation

    This makes the fallback genuinely image-dependent and allows the
    pipeline to operate without downloading a large checkpoint.
    """

    arr = _prepare_image(image)

    # RGB channel statistics.
    red = float(arr[..., 0].mean())
    green = float(arr[..., 1].mean())
    blue = float(arr[..., 2].mean())

    mean_v = float(arr.mean())
    std_v = float(arr.std())

    # ------------------------------------------------------------------
    # Strong blue dominance -> water
    # ------------------------------------------------------------------

    blue_score = blue - max(red, green)

    if blue_score > 0.08 and blue > 0.35:
        confidence = float(
            np.clip(
                0.45 + blue_score * 1.5,
                0.35,
                0.85,
            )
        )

        return "a water body", confidence

    # ------------------------------------------------------------------
    # Strong green dominance -> vegetation / forest
    # ------------------------------------------------------------------

    green_score = green - max(red, blue)

    if green_score > 0.08 and green > 0.30:
        confidence = float(
            np.clip(
                0.45 + green_score * 1.5,
                0.35,
                0.85,
            )
        )

        return "forest or vegetated land", confidence

    # ------------------------------------------------------------------
    # Bright image with strong variation -> built-up area
    # ------------------------------------------------------------------

    if mean_v > 0.50 and std_v > 0.15:
        confidence = float(
            np.clip(
                0.35 + std_v,
                0.35,
                0.70,
            )
        )

        return "a built-up area", confidence

    # ------------------------------------------------------------------
    # Dark image -> possible water / shadow
    # ------------------------------------------------------------------

    if mean_v < 0.25:
        return "a dark or water-like region", 0.35

    # ------------------------------------------------------------------
    # Generic vegetation fallback
    # ------------------------------------------------------------------

    return "vegetated land", 0.35


def _stats_yes_no(
    image: np.ndarray,
    query: str,
) -> Tuple[str, float, dict]:
    """
    Answer a yes/no question using the lightweight image-statistics
    baseline.

    The result contains the same important metadata fields as the CLIP
    path, especially:
        mode
        concept
        fallback_level
    """

    arr = _prepare_image(image)

    red = float(arr[..., 0].mean())
    green = float(arr[..., 1].mean())
    blue = float(arr[..., 2].mean())

    mean_v = float(arr.mean())

    hit = _concept_in_query(query)

    # ---------------------------------------------------------------
    # Unknown yes/no concept
    # ---------------------------------------------------------------

    if hit is None:
        text, conf = _stats_baseline(image, query)

        return (
            f"Based on the lightweight visual baseline, the image "
            f"appears to show {text}.",
            float(np.clip(conf, 0.0, 1.0)),
            {
                "fallback_level": "statistics_baseline",
                "mode": "yes_no",
                "concept": "unknown",
            },
        )

    target, distractors = hit

    # ---------------------------------------------------------------
    # Water
    # ---------------------------------------------------------------

    if target == "a sea or lake":
        water_score = blue - max(red, green)

        yes = (
            blue > 0.35
            and water_score > 0.06
        )

        if yes:
            confidence = float(
                np.clip(
                    0.55 + water_score * 2.0,
                    0.35,
                    0.90,
                )
            )

            return (
                "Yes — the image appears to contain a water body.",
                confidence,
                {
                    "fallback_level": "statistics_baseline",
                    "mode": "yes_no",
                    "concept": target,
                },
            )

        confidence = float(
            np.clip(
                0.55 + max(-water_score, 0.0),
                0.35,
                0.85,
            )
        )

        return (
            "No — a water body is not strongly indicated by the image.",
            confidence,
            {
                "fallback_level": "statistics_baseline",
                "mode": "yes_no",
                "concept": target,
            },
        )

    # ---------------------------------------------------------------
    # River
    # ---------------------------------------------------------------

    if target == "a river":
        water_score = blue - max(red, green)

        yes = (
            blue > 0.40
            and water_score > 0.08
        )

        confidence = float(
            np.clip(
                0.45 + abs(water_score) * 1.5,
                0.35,
                0.80,
            )
        )

        if yes:
            return (
                "Yes — the image appears to contain a river-like "
                "water feature.",
                confidence,
                {
                    "fallback_level": "statistics_baseline",
                    "mode": "yes_no",
                    "concept": target,
                },
            )

        return (
            "No — a river-like feature is not strongly indicated.",
            confidence,
            {
                "fallback_level": "statistics_baseline",
                "mode": "yes_no",
                "concept": target,
            },
        )

    # ---------------------------------------------------------------
    # Forest / tree
    # ---------------------------------------------------------------

    if target == "forest":
        vegetation_score = green - max(red, blue)

        yes = (
            green > 0.30
            and vegetation_score > 0.06
        )

        confidence = float(
            np.clip(
                0.50 + abs(vegetation_score) * 1.5,
                0.35,
                0.85,
            )
        )

        if yes:
            return (
                "Yes — the image appears to contain forest or "
                "dense vegetation.",
                confidence,
                {
                    "fallback_level": "statistics_baseline",
                    "mode": "yes_no",
                    "concept": target,
                },
            )

        return (
            "No — dense forest is not strongly indicated by the image.",
            confidence,
            {
                "fallback_level": "statistics_baseline",
                "mode": "yes_no",
                "concept": target,
            },
        )

    # ---------------------------------------------------------------
    # Vegetation
    # ---------------------------------------------------------------

    if target == "herbaceous vegetation":
        vegetation_score = green - max(red, blue)

        yes = (
            green > 0.28
            and vegetation_score > 0.04
        )

        confidence = float(
            np.clip(
                0.45 + abs(vegetation_score) * 1.5,
                0.35,
                0.80,
            )
        )

        if yes:
            return (
                "Yes — vegetation appears to be present.",
                confidence,
                {
                    "fallback_level": "statistics_baseline",
                    "mode": "yes_no",
                    "concept": target,
                },
            )

        return (
            "No — strong vegetation is not indicated by the image.",
            confidence,
            {
                "fallback_level": "statistics_baseline",
                "mode": "yes_no",
                "concept": target,
            },
        )

    # ---------------------------------------------------------------
    # Urban / residential
    # ---------------------------------------------------------------

    if target == "a residential area":
        built_score = mean_v + float(arr.std())

        yes = built_score > 0.70

        confidence = float(
            np.clip(
                0.40 + abs(built_score - 0.60),
                0.35,
                0.75,
            )
        )

        if yes:
            return (
                "Yes — a built-up or residential area appears possible.",
                confidence,
                {
                    "fallback_level": "statistics_baseline",
                    "mode": "yes_no",
                    "concept": target,
                },
            )

        return (
            "No — a residential area is not strongly indicated.",
            confidence,
            {
                "fallback_level": "statistics_baseline",
                "mode": "yes_no",
                "concept": target,
            },
        )

    # ---------------------------------------------------------------
    # Industrial
    # ---------------------------------------------------------------

    if target == "an industrial area":
        built_score = mean_v + float(arr.std())

        yes = built_score > 0.75

        confidence = float(
            np.clip(
                0.40 + abs(built_score - 0.65),
                0.35,
                0.75,
            )
        )

        if yes:
            return (
                "Yes — an industrial or built-up region appears possible.",
                confidence,
                {
                    "fallback_level": "statistics_baseline",
                    "mode": "yes_no",
                    "concept": target,
                },
            )

        return (
            "No — an industrial area is not strongly indicated.",
            confidence,
            {
                "fallback_level": "statistics_baseline",
                "mode": "yes_no",
                "concept": target,
            },
        )

    # ---------------------------------------------------------------
    # Generic fallback for remaining concepts
    # ---------------------------------------------------------------

    text, conf = _stats_baseline(image, query)

    return (
        f"Based on the lightweight visual baseline, the image "
        f"appears to show {text}.",
        float(np.clip(conf, 0.0, 1.0)),
        {
            "fallback_level": "statistics_baseline",
            "mode": "yes_no",
            "concept": target,
        },
    )


# ---------------------------------------------------------------------------
# Main VQA model
# ---------------------------------------------------------------------------

class VQAModel(BaseRSModel):
    """
    VQA specialist.

    Default behavior is lightweight fallback mode.

    CLIP can be explicitly enabled with:

        config={
            "clip": {
                "enabled": True
            }
        }

    This prevents accidental downloading of a large CLIP checkpoint
    during tests or resource-constrained execution.
    """

    name = "vqa_v1"
    task = "vqa"
    backbone = "statistics_baseline"

    def __init__(self, config: dict = None):
        self.config = config or {}

        # Shared CLIP backend.
        self.clip = None

        self.checkpoint = "statistics_baseline"
        self._using_fallback = True
        self.device = "cpu"

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(
        self,
        checkpoint_path: str = None,
        device: str = "cpu",
    ) -> None:
        """
        Load the VQA backend.

        Resource-saving behavior:

        - Missing explicit checkpoint -> fallback.
        - CLIP disabled -> fallback.
        - CLIP unavailable -> fallback.
        - CLIP is loaded only when explicitly enabled.
        """

        self.device = device

        cfg = self.config.get("clip") or {}

        # --------------------------------------------------------------
        # Explicitly supplied checkpoint that does not exist.
        # --------------------------------------------------------------

        if checkpoint_path:
            checkpoint_exists = os.path.exists(checkpoint_path)

            if not checkpoint_exists:
                self._activate_fallback()
                return

        # --------------------------------------------------------------
        # CLIP must explicitly be enabled.
        # --------------------------------------------------------------

        use_clip = bool(cfg.get("enabled", False))

        if not use_clip:
            self._activate_fallback()
            return

        # --------------------------------------------------------------
        # Load shared CLIP backend.
        # --------------------------------------------------------------

        model_id = cfg.get(
            "model_id",
            "openai/clip-vit-base-patch32",
        )

        template = cfg.get(
            "prompt_template",
            "a satellite photo of {}",
        )

        try:
            self.clip = get_shared_clip(
                model_id,
                template,
                device=device,
            )

            if self.clip is None or not self.clip.available:
                self._activate_fallback()
                return

            self._using_fallback = False
            self.checkpoint = model_id
            self.backbone = "CLIP-ViT-B/32"

        except Exception:
            self._activate_fallback()

    # ------------------------------------------------------------------
    # Fallback activation
    # ------------------------------------------------------------------

    def _activate_fallback(self) -> None:
        """Activate the zero-checkpoint lightweight baseline."""

        self.clip = None
        self._using_fallback = True
        self.checkpoint = "statistics_baseline"
        self.backbone = "statistics_baseline"

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        image: np.ndarray = None,
        query: str = "",
        **kwargs,
    ) -> dict:
        """
        Produce a schema-valid VQA result.

        If CLIP fails during inference, prediction automatically falls
        back to the lightweight image-statistics implementation.
        """

        t0 = time.time()

        if image is None:
            return empty_result(
                self.task,
                "No image provided.",
                status="error",
            )

        try:

            # ==========================================================
            # CLIP path
            # ==========================================================

            if self.clip is not None and self.clip.available:

                try:
                    text, conf, params = self._clip_answer(
                        image,
                        query,
                    )

                except Exception as exc:
                    # CLIP inference failed.
                    #
                    # Do NOT fail the complete pipeline.
                    # Fall back to the lightweight implementation.

                    if _is_yes_no(query):
                        text, conf, params = _stats_yes_no(
                            image,
                            query,
                        )
                    else:
                        text, conf = _stats_baseline(
                            image,
                            query,
                        )

                        params = {
                            "fallback_level": "statistics_baseline",
                            "mode": "open",
                        }

                    params["clip_error"] = str(exc)

                    self._using_fallback = True
                    self.checkpoint = "statistics_baseline"
                    self.backbone = "statistics_baseline"

            # ==========================================================
            # Statistics fallback
            # ==========================================================

            else:

                if _is_yes_no(query):
                    text, conf, params = _stats_yes_no(
                        image,
                        query,
                    )

                else:
                    text, conf = _stats_baseline(
                        image,
                        query,
                    )

                    params = {
                        "fallback_level": "statistics_baseline",
                        "mode": "open",
                    }

        except Exception as exc:

            return empty_result(
                self.task,
                f"VQA failed: {exc}",
                status="error",
            )

        # ----------------------------------------------------------------
        # Result status
        # ----------------------------------------------------------------

        status = (
            "success"
            if conf >= 0.35
            else "low_confidence"
        )

        return RSModelResult(
            task=self.task,
            text=text,
            confidence=float(
                np.clip(conf, 0.0, 1.0)
            ),
            spatial_evidence=SpatialEvidence(
                type="none",
                source="image",
            ),
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

    # ------------------------------------------------------------------
    # CLIP internals
    # ------------------------------------------------------------------

    def _clip_answer(
        self,
        image: np.ndarray,
        query: str,
    ) -> Tuple[str, float, dict]:
        """
        Answer the query using the shared CLIP backend.
        """

        # ==============================================================
        # Yes/no question
        # ==============================================================

        if _is_yes_no(query):

            hit = _concept_in_query(query)

            if hit is not None:

                target, distractors = hit

                candidates = [
                    target
                ] + (
                    distractors or []
                )

                others = [
                    label
                    for label in RS_LABELS
                    if label not in candidates
                ][:6]

                ranked = self.clip.classify(
                    image,
                    candidates + others,
                )

                if not ranked:
                    raise RuntimeError(
                        "CLIP returned no classification results."
                    )

                score = sum(
                    probability
                    for label, probability in ranked
                    if label in candidates
                )

                top_label, _ = ranked[0]

                if score >= 0.35:

                    answer = (
                        "Yes — the image appears to contain "
                        f"{LABEL_DISPLAY.get(target, target)}."
                    )

                    conf = score

                else:

                    answer = (
                        "No — the image is dominated by "
                        f"{LABEL_DISPLAY.get(top_label, top_label)} "
                        "instead."
                    )

                    conf = 1.0 - score

                return (
                    answer,
                    float(
                        np.clip(
                            conf,
                            0.0,
                            1.0,
                        )
                    ),
                    {
                        "fallback_level": "clip_zeroshot",
                        "mode": "yes_no",
                        "concept": target,
                    },
                )

        # ==============================================================
        # Open-ended question
        # ==============================================================

        ranked = self.clip.classify(
            image,
            RS_LABELS,
        )

        if not ranked:
            raise RuntimeError(
                "CLIP returned no classification results."
            )

        top, p1 = ranked[0]

        if len(ranked) > 1:
            second, p2 = ranked[1]
        else:
            second, p2 = top, 0.0

        disp1 = LABEL_DISPLAY.get(
            top,
            top,
        )

        disp2 = LABEL_DISPLAY.get(
            second,
            second,
        )

        if p1 - p2 < 0.10:

            text = (
                f"The image most likely shows {disp1}, "
                f"though {disp2} is a close alternative."
            )

        else:

            text = f"The image shows {disp1}."

        return (
            text,
            float(
                np.clip(
                    p1,
                    0.0,
                    1.0,
                )
            ),
            {
                "fallback_level": "clip_zeroshot",
                "mode": "open",
                "top_k": [
                    [
                        label,
                        round(
                            probability,
                            4,
                        ),
                    ]
                    for label, probability in ranked[:3]
                ],
            },
        )