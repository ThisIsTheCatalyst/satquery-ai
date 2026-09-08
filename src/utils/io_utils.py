"""I/O utilities for saving evidence, reports, and checkpoints."""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np


def ensure_dir(path: str) -> str:
    """Create directory (and parents) if it doesn't exist. Returns path."""
    os.makedirs(path, exist_ok=True)
    return path


def save_mask_as_image(mask: np.ndarray, path: str, colormap: bool = True) -> str:
    """Save a binary/probability mask as a PNG image.
    
    mask: (H,W) array, values in [0,1] or binary {0,1}.
    path: output path.
    colormap: if True, apply a red colormap (change=red, no-change=black).
    """
    ensure_dir(os.path.dirname(path) or ".")
    try:
        from PIL import Image as PILImage
        
        mask_u8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
        if colormap:
            # Red channel = change, green/blue = 0
            rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
            rgb[:, :, 0] = mask_u8
            img = PILImage.fromarray(rgb)
        else:
            img = PILImage.fromarray(mask_u8, mode="L")
        img.save(path)
        return path
    except Exception as e:
        print(f"[io_utils] Could not save mask to {path}: {e}")
        return path


def save_bbox_visualization(image: np.ndarray, bbox: list, path: str,
                             color=(255, 0, 0), thickness: int = 2) -> str:
    """Save image with bounding box overlay as PNG."""
    ensure_dir(os.path.dirname(path) or ".")
    try:
        from PIL import Image as PILImage, ImageDraw
        
        # image: (C,H,W) float32 in [0,1]
        arr = image[:3] if image.shape[0] >= 3 else np.repeat(image[:1], 3, 0)
        arr_uint8 = (np.clip(arr, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)
        img = PILImage.fromarray(arr_uint8)
        draw = ImageDraw.Draw(img)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        for t in range(thickness):
            draw.rectangle([x1+t, y1+t, x2-t, y2-t], outline=color)
        img.save(path)
        return path
    except Exception as e:
        print(f"[io_utils] Could not save bbox visualization to {path}: {e}")
        return path


def save_side_by_side(img1: np.ndarray, img2: np.ndarray, path: str,
                       label1: str = "T1", label2: str = "T2") -> str:
    """Save two images side by side as PNG."""
    ensure_dir(os.path.dirname(path) or ".")
    try:
        from PIL import Image as PILImage, ImageDraw, ImageFont
        
        def to_pil(arr):
            a = arr[:3] if arr.shape[0] >= 3 else np.repeat(arr[:1], 3, 0)
            return PILImage.fromarray((np.clip(a, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8))
        
        pil1, pil2 = to_pil(img1), to_pil(img2)
        w = pil1.width + pil2.width + 4
        h = max(pil1.height, pil2.height) + 20
        combined = PILImage.new("RGB", (w, h), (50, 50, 50))
        combined.paste(pil1, (0, 20))
        combined.paste(pil2, (pil1.width + 4, 20))
        draw = ImageDraw.Draw(combined)
        draw.text((4, 2), label1, fill=(255, 255, 255))
        draw.text((pil1.width + 8, 2), label2, fill=(255, 255, 255))
        combined.save(path)
        return path
    except Exception as e:
        print(f"[io_utils] Could not save side-by-side to {path}: {e}")
        return path


def save_json(data: dict, path: str) -> str:
    """Save dict as JSON with numpy/array handling."""
    ensure_dir(os.path.dirname(path) or ".")
    
    def _serialize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_serialize)
    return path


def load_json(path: str) -> dict:
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def save_evidence(result: dict, output_dir: str, prefix: str = "evidence") -> dict:
    """Save spatial evidence (mask/bbox) from an RSModelResult to output_dir.
    Returns updated evidence dict with file paths added.
    """
    ensure_dir(output_dir)
    evidence = result.get("spatial_evidence", {}) or {}
    ev_type = evidence.get("type", "none")
    
    if ev_type == "mask":
        mask = evidence.get("mask")
        if mask is not None and isinstance(mask, np.ndarray):
            mask_path = os.path.join(output_dir, f"{prefix}_change_mask.png")
            save_mask_as_image(mask, mask_path)
            evidence = dict(evidence)
            evidence["mask_path"] = mask_path
            evidence["mask"] = None  # don't serialize the array itself in the result
    
    elif ev_type == "bbox":
        bbox = evidence.get("bbox")
        if bbox is not None:
            evidence = dict(evidence)
            evidence["bbox_saved"] = True
    
    return evidence
