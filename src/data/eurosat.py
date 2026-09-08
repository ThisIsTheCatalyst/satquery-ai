"""
EuroSAT loader.

Why EuroSAT: 27,000 labelled Sentinel-2 patches across 10 land-cover classes
at 64x64 RGB, roughly 90 MB. It is the smallest real satellite dataset that
comes with ground-truth labels, which is what lets us report an honest
zero-shot accuracy number instead of claiming one. Downloads in under a
minute on Colab.

Only a 200-image subset (20 per class) is cached for the demo — the UI
gallery and the CLIP evaluation both read from that subset, so the demo is
reproducible and fast.
"""

import json
import os
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

_HF_REPO = "blanchon/EuroSAT_RGB"
_TORCHVISION_ROOT = "data/raw/eurosat_tv"


def _download_via_torchvision(root: str) -> Optional[Path]:
    """Primary path: torchvision handles the download and extraction."""
    try:
        from torchvision.datasets import EuroSAT
        EuroSAT(root=root, download=True)
        for cand in Path(root).rglob("2750"):
            if cand.is_dir():
                return cand
        for cand in Path(root).rglob("AnnualCrop"):
            return cand.parent
    except Exception as exc:                          # noqa: BLE001
        print(f"[eurosat] torchvision download failed: {exc}")
    return None


def _download_via_hf(root: str) -> Optional[Path]:
    """Fallback path: HuggingFace datasets mirror."""
    try:
        from datasets import load_dataset
        from PIL import Image as PILImage

        ds = load_dataset(_HF_REPO, split="train")
        out = Path(root) / "hf_extracted"
        out.mkdir(parents=True, exist_ok=True)
        names = ds.features["label"].names
        for i, rec in enumerate(ds):
            cls = names[rec["label"]]
            d = out / cls
            d.mkdir(exist_ok=True)
            img = rec["image"]
            if not isinstance(img, PILImage.Image):
                img = PILImage.fromarray(np.array(img))
            img.convert("RGB").save(d / f"{cls}_{i}.png")
        return out
    except Exception as exc:                          # noqa: BLE001
        print(f"[eurosat] HuggingFace download failed: {exc}")
    return None


def download_eurosat(root: str = _TORCHVISION_ROOT) -> Optional[Path]:
    """Download EuroSAT and return the directory containing class folders."""
    Path(root).mkdir(parents=True, exist_ok=True)
    src = _download_via_torchvision(root) or _download_via_hf(root)
    if src is None:
        print("[eurosat] Could not obtain EuroSAT automatically. "
              "Download manually from https://zenodo.org/records/7711810 "
              "and extract so that class folders sit under the root.")
    return src


def build_demo_subset(src_dir: Path,
                      out_dir: str = "data/demo/eurosat",
                      n_per_class: int = 20,
                      seed: int = 42) -> Path:
    """Copy n_per_class images per class into out_dir and write labels.json.

    Deterministic given `seed`, so the gallery a judge sees is the same set
    the reported accuracy was measured on.
    """
    src_dir = Path(src_dir)
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    labels = {}

    for cls in EUROSAT_CLASSES:
        cdir = src_dir / cls
        if not cdir.is_dir():
            print(f"[eurosat] missing class dir: {cdir}")
            continue
        files = sorted([p for p in cdir.iterdir()
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif")])
        if not files:
            continue
        pick = rng.choice(len(files), size=min(n_per_class, len(files)),
                          replace=False)
        for k, idx in enumerate(pick):
            dst = out / f"{cls}_{k:03d}.png"
            _save_as_png(files[int(idx)], dst)
            labels[dst.name] = cls

    (out / "labels.json").write_text(json.dumps(labels, indent=2))
    print(f"[eurosat] demo subset: {len(labels)} images -> {out}")
    return out


def _save_as_png(src_path: Path, dst_path: Path) -> None:
    from PIL import Image as PILImage
    PILImage.open(src_path).convert("RGB").save(dst_path)


def load_demo_subset(out_dir: str = "data/demo/eurosat"
                     ) -> List[Tuple[np.ndarray, str, str]]:
    """Load the cached subset as [(HWC uint8 array, class_name, filename)]."""
    from PIL import Image as PILImage

    out = Path(out_dir)
    lbl_path = out / "labels.json"
    if not lbl_path.exists():
        raise FileNotFoundError(
            f"No demo subset at {out}. Run scripts/prepare_data.py first."
        )
    labels = json.loads(lbl_path.read_text())
    items = []
    for fname, cls in sorted(labels.items()):
        arr = np.array(PILImage.open(out / fname).convert("RGB"))
        items.append((arr, cls, fname))
    return items


def demo_subset_exists(out_dir: str = "data/demo/eurosat") -> bool:
    return (Path(out_dir) / "labels.json").exists()
