"""
Shared CLIP backend — the vision core of SatQuery AI.

One CLIP instance is loaded process-wide and shared by the VQA, captioning
and grounding specialists. This is deliberate: CLIP is ~600 MB and loading
it three times would triple demo cold-start for zero benefit.

Why CLIP and not a remote-sensing-specific VLM:
  CLIP's pretraining corpus contains a substantial amount of aerial and
  satellite imagery, so zero-shot land-cover classification works out of the
  box at well above chance. That buys a genuinely image-conditioned answer
  with no training, no 14 GB checkpoint, and no LoRA pipeline. The measured
  accuracy is reported honestly by scripts/eval_clip_eurosat.py rather than
  claimed.

Prompt engineering matters more than model size here: "a satellite photo of
{label}" substantially outperforms the bare label, because it moves the text
embedding into the aerial-imagery region of CLIP's joint space.
"""

from typing import List, Tuple, Optional
import numpy as np

_DEFAULT_MODEL_ID = "openai/clip-vit-base-patch32"
_DEFAULT_TEMPLATE = "a satellite photo of {}"

# Canonical land-cover vocabulary. Aligned with EuroSAT's 10 classes but
# phrased in natural language, since CLIP responds to phrasing, not to
# dataset-internal class names ("HerbaceousVegetation" is not English).
RS_LABELS = [
    "annual cropland",
    "forest",
    "herbaceous vegetation",
    "a highway or road",
    "an industrial area",
    "pasture land",
    "permanent cropland",
    "a residential area",
    "a river",
    "a sea or lake",
]

# Maps EuroSAT's directory class names onto the natural-language prompts above,
# so evaluation can score CLIP's output against ground truth.
EUROSAT_TO_LABEL = {
    "AnnualCrop": "annual cropland",
    "Forest": "forest",
    "HerbaceousVegetation": "herbaceous vegetation",
    "Highway": "a highway or road",
    "Industrial": "an industrial area",
    "Pasture": "pasture land",
    "PermanentCrop": "permanent cropland",
    "Residential": "a residential area",
    "River": "a river",
    "SeaLake": "a sea or lake",
}
LABEL_TO_EUROSAT = {v: k for k, v in EUROSAT_TO_LABEL.items()}

# Short human-readable forms for captions and answers.
LABEL_DISPLAY = {
    "annual cropland": "annual cropland",
    "forest": "forest",
    "herbaceous vegetation": "herbaceous vegetation",
    "a highway or road": "a highway",
    "an industrial area": "an industrial area",
    "pasture land": "pasture",
    "permanent cropland": "permanent cropland",
    "a residential area": "a residential area",
    "a river": "a river",
    "a sea or lake": "a lake or sea",
}


class CLIPBackend:
    """Thin wrapper over HuggingFace CLIP exposing the two operations the
    specialists need: whole-image classification and patch-wise scoring.
    """

    def __init__(self, model_id: str = _DEFAULT_MODEL_ID,
                 prompt_template: str = _DEFAULT_TEMPLATE):
        self.model_id = model_id
        self.prompt_template = prompt_template
        self.model = None
        self.processor = None
        self.device = "cpu"
        self.available = False

    # -- lifecycle ---------------------------------------------------------

    def load(self, device: str = "cpu") -> bool:
        """Load CLIP. Returns True on success, False if transformers/torch
        are unavailable — callers fall back to the statistics baseline.
        """
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self.model = CLIPModel.from_pretrained(self.model_id)
            self.processor = CLIPProcessor.from_pretrained(self.model_id)
            self.model.eval()
            if device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to("cuda")
                self.device = "cuda"
            else:
                self.device = "cpu"
            self.available = True
        except Exception as exc:                       # noqa: BLE001
            print(f"[clip_backend] CLIP unavailable, falling back: {exc}")
            self.available = False
        return self.available

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _to_hwc_uint8(image: np.ndarray) -> np.ndarray:
        """Accept CHW float [0,1] or HWC uint8; always return HWC uint8 RGB.

        Models elsewhere in the codebase pass CHW float arrays (the internal
        convention), while PIL/Streamlit produce HWC uint8. Normalising here
        keeps that mess out of the three specialist models.
        """
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[0] in (1, 2, 3, 4) and arr.shape[0] < arr.shape[-1]:
            arr = arr.transpose(1, 2, 0)              # CHW -> HWC
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.shape[-1] == 2:                       # SAR VV/VH -> pseudo-RGB
            arr = np.stack([arr[..., 0], arr[..., 1], arr[..., 0]], axis=-1)
        elif arr.shape[-1] > 3:
            arr = arr[..., :3]                         # drop NIR etc. for CLIP
        if arr.dtype != np.uint8:
            amax = float(np.nanmax(arr)) if arr.size else 1.0
            arr = arr / amax if amax > 1.0 else arr
            arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
        return arr

    def _embed(self, pil_images: List, texts: List[str]):
        """Return (image_features, text_features), both L2-normalised."""
        import torch

        inputs = self.processor(text=texts, images=pil_images,
                                return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            img_f = self.model.get_image_features(pixel_values=inputs["pixel_values"])
            txt_f = self.model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
        return img_f, txt_f

    # -- public API --------------------------------------------------------

    def classify(self, image: np.ndarray,
                 candidate_labels: Optional[List[str]] = None
                 ) -> List[Tuple[str, float]]:
        """Zero-shot classify one image against candidate labels.

        Returns [(label, probability), ...] sorted descending. Probabilities
        are a softmax over cosine similarities scaled by CLIP's learned
        logit_scale, so they are comparable across calls.
        """
        if not self.available:
            raise RuntimeError("CLIPBackend.classify called before load()")

        from PIL import Image as PILImage
        import torch

        labels = candidate_labels or RS_LABELS
        prompts = [self.prompt_template.format(l) for l in labels]

        hwc = self._to_hwc_uint8(image)
        pil = PILImage.fromarray(hwc).convert("RGB")

        img_f, txt_f = self._embed([pil], prompts)
        logit_scale = self.model.logit_scale.exp()
        logits = (logit_scale * img_f @ txt_f.T).squeeze(0)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        ranked = sorted(zip(labels, probs.tolist()), key=lambda t: -t[1])
        return [(l, float(p)) for l, p in ranked]

    def score_patches(self, image: np.ndarray, text: str,
                      grid: int = 4) -> np.ndarray:
        """Slide a grid x grid window over the image and score each patch
        against `text`. Returns a (grid, grid) float array of similarities,
        min-max normalised to [0,1].

        This is what makes grounding real rather than heuristic: the box
        comes from where CLIP actually finds the queried concept, not from
        image brightness statistics.
        """
        if not self.available:
            raise RuntimeError("CLIPBackend.score_patches called before load()")

        from PIL import Image as PILImage
        import torch

        hwc = self._to_hwc_uint8(image)
        h, w = hwc.shape[:2]
        ph, pw = max(h // grid, 1), max(w // grid, 1)

        patches, coords = [], []
        for gy in range(grid):
            for gx in range(grid):
                y0, x0 = gy * ph, gx * pw
                y1 = h if gy == grid - 1 else (gy + 1) * ph
                x1 = w if gx == grid - 1 else (gx + 1) * pw
                crop = hwc[y0:y1, x0:x1]
                if crop.size == 0:
                    crop = hwc
                # CLIP's encoder expects 224x224; small patches are upsampled.
                patches.append(
                    PILImage.fromarray(crop).convert("RGB").resize((224, 224),
                                                                   PILImage.BICUBIC)
                )
                coords.append((gy, gx))

        prompt = self.prompt_template.format(text)
        img_f, txt_f = self._embed(patches, [prompt])
        sims = (img_f @ txt_f.T).squeeze(-1).cpu().numpy()

        heat = np.zeros((grid, grid), dtype=np.float32)
        for (gy, gx), s in zip(coords, sims):
            heat[gy, gx] = float(s)

        rng = heat.max() - heat.min()
        return (heat - heat.min()) / rng if rng > 1e-8 else np.zeros_like(heat)


# Process-wide singleton so the three specialists share one set of weights.
_SHARED: Optional[CLIPBackend] = None


def get_shared_clip(model_id: str = _DEFAULT_MODEL_ID,
                    prompt_template: str = _DEFAULT_TEMPLATE,
                    device: str = "cpu") -> CLIPBackend:
    """Return the shared CLIP instance, loading it on first call."""
    global _SHARED
    if _SHARED is None:
        _SHARED = CLIPBackend(model_id, prompt_template)
        _SHARED.load(device=device)
    return _SHARED
