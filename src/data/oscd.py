"""
OSCD (Onera Satellite Change Detection) loader.

Why OSCD: 24 real co-registered Sentinel-2 image pairs with pixel-level
ground-truth change masks, around 500 MB. It is the smallest real bi-temporal
dataset that ships with labels, which is what turns the change-detection
demo from "here is a mask" into "here is a mask and its measured F1".

Six pairs are cached for the demo. Tiles are also large enough (hundreds of
pixels per side) that CLIP patch-grounding produces a meaningful box, unlike
64x64 EuroSAT chips.
"""

import json
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

_ZENODO_HINT = "https://rcdaudt.github.io/oscd/"


def download_oscd(root: str = "data/raw/oscd") -> Optional[Path]:
    """OSCD requires accepting terms, so this checks for a manual extract
    rather than pretending it can be fetched unattended.
    """
    p = Path(root)
    if p.exists() and any(p.iterdir()):
        return p
    print(f"[oscd] Not found at {root}. Download from {_ZENODO_HINT} "
          f"and extract there (expects <city>/imgs_1/, imgs_2/, cm/).")
    return None


def _read_rgb_from_bands(city_dir: Path, sub: str) -> Optional[np.ndarray]:
    """OSCD ships per-band Sentinel-2 TIFFs. Compose B04/B03/B02 into RGB."""
    from PIL import Image as PILImage

    d = city_dir / sub
    if not d.is_dir():
        return None

    band = {}
    for p in d.iterdir():
        n = p.name.upper()
        for b in ("B04", "B03", "B02"):
            if b in n:
                band[b] = p

    if len(band) == 3:
        chans = []
        for b in ("B04", "B03", "B02"):
            a = np.array(PILImage.open(band[b])).astype(np.float32)
            lo, hi = np.percentile(a, 2), np.percentile(a, 98)
            a = np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)
            chans.append((a * 255).astype(np.uint8))
        return np.stack(chans, axis=-1)

    # Some redistributions ship a ready-made RGB file.
    for p in d.iterdir():
        if p.suffix.lower() in (".png", ".jpg", ".tif", ".tiff"):
            try:
                return np.array(PILImage.open(p).convert("RGB"))
            except Exception:                          # noqa: BLE001
                continue
    return None


def _read_mask(city_dir: Path) -> Optional[np.ndarray]:
    from PIL import Image as PILImage
    cm = city_dir / "cm"
    if not cm.is_dir():
        return None
    for p in cm.iterdir():
        if p.suffix.lower() in (".png", ".tif", ".tiff"):
            m = np.array(PILImage.open(p).convert("L"))
            return (m > 0).astype(np.uint8)
    return None


def build_demo_pairs(src_root: Path,
                     out_dir: str = "data/demo/oscd",
                     n_pairs: int = 6,
                     max_side: int = 512) -> Path:
    """Cache n_pairs of (T1, T2, ground-truth mask) as PNGs."""
    from PIL import Image as PILImage

    src_root = Path(src_root)
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    cities = sorted([d for d in src_root.iterdir() if d.is_dir()])
    manifest = []

    for city in cities:
        if len(manifest) >= n_pairs:
            break
        t1 = _read_rgb_from_bands(city, "imgs_1")
        t2 = _read_rgb_from_bands(city, "imgs_2")
        gt = _read_mask(city)
        if t1 is None or t2 is None:
            continue

        def _fit(a, resample=PILImage.BILINEAR):
            img = PILImage.fromarray(a)
            if max(img.size) > max_side:
                r = max_side / max(img.size)
                img = img.resize((int(img.width * r), int(img.height * r)), resample)
            return np.array(img)

        t1, t2 = _fit(t1), _fit(t2)
        rec = {"name": city.name,
               "t1": f"{city.name}_t1.png",
               "t2": f"{city.name}_t2.png",
               "gt": None}
        PILImage.fromarray(t1).save(out / rec["t1"])
        PILImage.fromarray(t2).save(out / rec["t2"])

        if gt is not None:
            gt_img = PILImage.fromarray((gt * 255).astype(np.uint8)).resize(
                (t1.shape[1], t1.shape[0]), PILImage.NEAREST)
            rec["gt"] = f"{city.name}_gt.png"
            gt_img.save(out / rec["gt"])

        manifest.append(rec)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[oscd] demo pairs: {len(manifest)} -> {out}")
    return out


def load_demo_pairs(out_dir: str = "data/demo/oscd") -> List[dict]:
    """Return [{name, t1, t2, gt}] with arrays loaded (gt may be None)."""
    from PIL import Image as PILImage

    out = Path(out_dir)
    man = out / "manifest.json"
    if not man.exists():
        raise FileNotFoundError(
            f"No OSCD demo pairs at {out}. Run scripts/prepare_data.py first."
        )
    pairs = []
    for rec in json.loads(man.read_text()):
        item = {
            "name": rec["name"],
            "t1": np.array(PILImage.open(out / rec["t1"]).convert("RGB")),
            "t2": np.array(PILImage.open(out / rec["t2"]).convert("RGB")),
            "gt": None,
        }
        if rec.get("gt"):
            g = np.array(PILImage.open(out / rec["gt"]).convert("L"))
            item["gt"] = (g > 127).astype(np.uint8)
        pairs.append(item)
    return pairs


def demo_pairs_exist(out_dir: str = "data/demo/oscd") -> bool:
    return (Path(out_dir) / "manifest.json").exists()
