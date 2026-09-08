"""
GeoTIFF / TIFF I/O utilities.
Owner: Person 1 (Data & Preprocessing).

Responsibilities:
- Read GeoTIFF/TIFF files (optical, multispectral, SAR) with rasterio.
- Extract bands, CRS, transform (georeferencing), resolution, bounding box.
- Validate that two images are co-registered (same CRS/extent/resolution)
  for cross-modal and bi-temporal pairs.
- Fall back to PNG/JPEG reading for the approved public benchmark datasets
  (no georeferencing expected there).
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

try:
    import rasterio
    from rasterio.warp import transform_bounds
except ImportError:
    rasterio = None  # allow module import even before deps are installed

try:
    from PIL import Image
except ImportError:
    Image = None


@dataclass
class RSImage:
    """Container for a loaded remote-sensing image."""
    array: np.ndarray            # shape (C, H, W)
    crs: Optional[str]           # e.g. "EPSG:32643", None for PNG/JPEG
    transform: Optional[object]  # affine transform, None for PNG/JPEG
    bounds: Optional[tuple]      # (minx, miny, maxx, maxy)
    resolution: Optional[tuple]  # (x_res, y_res) in CRS units
    band_count: int
    dtype: str
    source_path: str


def read_image(path: str) -> RSImage:
    """Read a GeoTIFF/TIFF/PNG/JPEG into an RSImage.

    TODO(Person 1):
      - Use rasterio for .tif/.tiff (preserves georeferencing).
      - Use PIL/opencv for .png/.jpg (benchmark datasets only, no CRS).
      - Handle multi-band stacks (optical: R/G/B/NIR, SAR: VV/VH).
      - Handle nodata values / masking.
    """
    ext = path.lower().rsplit(".", 1)[-1]
    if ext in ("tif", "tiff"):
        return _read_geotiff(path)
    elif ext in ("png", "jpg", "jpeg"):
        return _read_plain_image(path)
    raise ValueError(f"Unsupported format: {ext}. Supported: tif/tiff/png/jpg/jpeg")


def _read_geotiff(path: str) -> RSImage:
    if rasterio is None:
        raise ImportError("rasterio is required to read GeoTIFF files. pip install rasterio")
    with rasterio.open(path) as src:
        array = src.read()  # (C, H, W)
        return RSImage(
            array=array,
            crs=str(src.crs) if src.crs else None,
            transform=src.transform,
            bounds=tuple(src.bounds),
            resolution=(src.res[0], src.res[1]),
            band_count=src.count,
            dtype=str(src.dtypes[0]),
            source_path=path,
        )


def _read_plain_image(path: str) -> RSImage:
    """PNG/JPEG reader for the benchmark datasets (VRSBench/RSVQA/CDVQA),
    which carry no georeferencing. Always returns array shape (C,H,W) —
    converts to RGB first so grayscale/paletted/RGBA PNGs (all seen in the
    wild in these benchmarks) come out as a consistent 3-band array rather
    than silently varying band counts across samples in the same dataset.
    """
    if Image is None:
        raise ImportError("Pillow is required to read PNG/JPEG files. pip install Pillow")
    with Image.open(path) as im:
        im = im.convert("RGB")
        array = np.array(im)  # (H, W, C)
        array = np.transpose(array, (2, 0, 1))  # -> (C, H, W)
        return RSImage(
            array=array,
            crs=None,
            transform=None,
            bounds=None,
            resolution=None,
            band_count=array.shape[0],
            dtype=str(array.dtype),
            source_path=path,
        )


def check_coregistration(img_a: RSImage, img_b: RSImage, tolerance_px: float = 1.0) -> dict:
    """Check whether two RSImages are co-registered (same CRS, comparable
    extent/resolution) — required for cross-modal (optical+SAR) and
    bi-temporal pairs.

    Returns: {"co_registered": bool, "reason": str | None}

    Behaviour by case:
    - Both images have no CRS (benchmark PNG/JPEG, e.g. CDVQA T1/T2 pairs):
      co-registration is asserted structurally instead — same (H, W) shape,
      since these datasets don't carry georeferencing to check against.
      A shape mismatch is flagged; a shape match does NOT prove true
      geographic alignment (the benchmark is assumed pre-aligned by its
      authors) — this is a documented assumption, not a guarantee.
    - One has a CRS and the other doesn't: always a hard failure — mixing a
      georeferenced GeoTIFF with a non-georeferenced PNG/JPEG in one pair is
      almost certainly a data-pipeline bug, never intentional.
    - Both have a CRS (real GeoTIFF pairs — BigEarthNet-MM, ISRO/SAC):
      reprojects img_b's bounds into img_a's CRS if they differ, then
      checks (a) resolution matches within `tolerance_px` fractions of a
      pixel and (b) the bounds actually overlap (not just "close" — a
      spatial IoU-style overlap check, since two images can have identical
      resolution and still be entirely different tiles).
    """
    a_has_crs = img_a.crs is not None
    b_has_crs = img_b.crs is not None

    if not a_has_crs and not b_has_crs:
        if img_a.array.shape[-2:] != img_b.array.shape[-2:]:
            return {
                "co_registered": False,
                "reason": (
                    f"Neither image has CRS metadata (benchmark PNG/JPEG "
                    f"pair) and shapes differ: {img_a.array.shape[-2:]} vs "
                    f"{img_b.array.shape[-2:]}."
                ),
            }
        return {"co_registered": True, "reason": None}

    if a_has_crs != b_has_crs:
        return {
            "co_registered": False,
            "reason": (
                f"CRS mismatch: one image is georeferenced ({img_a.crs!r}) "
                f"and the other is not ({img_b.crs!r}). Mixing a GeoTIFF "
                f"with a non-georeferenced PNG/JPEG in a pair is not valid."
            ),
        }

    if rasterio is None:
        raise ImportError("rasterio is required to compare georeferenced images.")

    bounds_b = img_b.bounds
    if str(img_a.crs) != str(img_b.crs):
        bounds_b = transform_bounds(img_b.crs, img_a.crs, *img_b.bounds)

    res_a_x, res_a_y = img_a.resolution
    res_b_x, res_b_y = img_b.resolution
    if abs(res_a_x - res_b_x) > tolerance_px * abs(res_a_x) or \
       abs(res_a_y - res_b_y) > tolerance_px * abs(res_a_y):
        return {
            "co_registered": False,
            "reason": (
                f"Resolution mismatch beyond tolerance ({tolerance_px}px): "
                f"{img_a.resolution} vs {img_b.resolution}."
            ),
        }

    ax0, ay0, ax1, ay1 = img_a.bounds
    bx0, by0, bx1, by1 = bounds_b
    overlap_x0, overlap_y0 = max(ax0, bx0), max(ay0, by0)
    overlap_x1, overlap_y1 = min(ax1, bx1), min(ay1, by1)
    if overlap_x1 <= overlap_x0 or overlap_y1 <= overlap_y0:
        return {
            "co_registered": False,
            "reason": (
                f"No spatial overlap between bounds {img_a.bounds} and "
                f"(reprojected) {bounds_b}."
            ),
        }

    return {"co_registered": True, "reason": None}


def extract_metadata(img: RSImage) -> dict:
    """Return a JSON-serializable metadata dict for the execution trace / UI."""
    return {
        "crs": img.crs,
        "bounds": img.bounds,
        "resolution": img.resolution,
        "band_count": img.band_count,
        "dtype": img.dtype,
        "shape": img.array.shape if img.array is not None else None,
        "source_path": img.source_path,
    }
