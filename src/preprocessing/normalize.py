"""
Band selection, normalization, resizing, and tiling.
Owner: Person 1 (Data & Preprocessing).

STATUS (ML audit pass): select_bands/normalize/resize/tile_image are now
REAL implementations (previously NotImplementedError stubs). See each
function's docstring for the exact method and its assumptions. Sensor-aware
normalization for Sentinel-1/2 (and, once confirmed, Cartosat-2S/RISAT)
should still go through src/preprocessing/sensor_registry.py rather than the
generic string-keyed `normalize()` here — that function remains the right
tool for benchmark PNG/JPEG datasets (vrsbench/rsvqa/cdvqa) that carry no
real sensor metadata.
"""

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - exercised only if opencv missing
    cv2 = None

# ImageNet normalization constants (RGB order), for the "imagenet" method —
# matches what most HuggingFace vision processors (incl. CLIP-family
# encoders GeoChat is built on) expect for a 3-channel RGB input.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def select_bands(array: np.ndarray, band_indices: list) -> np.ndarray:
    """array: (C,H,W). Select a subset of bands (e.g. optical R/G/B/NIR).

    Validates indices are in range (raises IndexError with a clear message
    instead of numpy's less-obvious one) and accepts any indexable sequence
    (list/tuple/np.ndarray of ints).
    """
    c = array.shape[0]
    bad = [i for i in band_indices if i < 0 or i >= c]
    if bad:
        raise IndexError(
            f"select_bands: index/indices {bad} out of range for an array "
            f"with {c} bands (valid range 0..{c - 1})."
        )
    return array[list(band_indices), :, :]


def normalize(array: np.ndarray, method: str = "per_band_minmax") -> np.ndarray:
    """Normalize pixel values to [0, 1] (or z-score) per band.

    Methods used across configs/config.yaml's per-dataset blocks:
      - "per_band_minmax": per-band min/max scaling to [0, 1], computed from
        the array itself (no fixed constants) — a reasonable generic
        fallback when no sensor-specific adapter applies.
      - "percentile_2_98": clip to the 2nd/98th percentile per band, then
        min/max scale — robust to outliers, used for Sentinel-2 optical
        reflectance (bigearthnet, bigearthnet_mm, isro_sac optical).
        Prefer sensor_adapters/sentinel2.py's physically-meaningful fixed
        REFLECTANCE_SCALE for actual Sentinel-2 training data; this
        percentile method is the appropriate choice for benchmark PNG/JPEG
        imagery with no known reflectance scale.
      - "db_scale_minmax": SAR-specific — convert linear backscatter to dB
        (10*log10), then min/max scale to [0, 1] using the same DB_MIN/MAX
        convention as sensor_adapters/sentinel1.py. Prefer that adapter
        directly for real Sentinel-1 data; this generic path is for
        benchmark SAR-like imagery without a known sensor.
      - "imagenet": normalize with ImageNet mean/std. Assumes a 3-channel
        (RGB-ordered) array already scaled to [0, 1] — run
        "per_band_minmax" or "percentile_2_98" first if the input is raw DN.
      - "zscore": per-band mean/std normalization (computed from the array).

    All methods return float32. Unknown methods raise ValueError (not
    NotImplementedError) since this is now a complete, closed set.
    """
    array = array.astype(np.float32)

    if method == "per_band_minmax":
        out = np.empty_like(array)
        for c in range(array.shape[0]):
            band = array[c]
            lo, hi = float(band.min()), float(band.max())
            out[c] = (band - lo) / (hi - lo) if hi > lo else np.zeros_like(band)
        return out

    if method == "percentile_2_98":
        out = np.empty_like(array)
        for c in range(array.shape[0]):
            band = array[c]
            lo, hi = np.percentile(band, 2), np.percentile(band, 98)
            band = np.clip(band, lo, hi)
            out[c] = (band - lo) / (hi - lo) if hi > lo else np.zeros_like(band)
        return out

    if method == "db_scale_minmax":
        # Mirrors sensor_adapters/sentinel1.py's DB_MIN/DB_MAX convention.
        # Kept as a local constant (not an import) so this generic path
        # never silently drifts if the sensor adapter's constants change —
        # any real Sentinel-1 training should go through the adapter
        # directly, not this fallback.
        db_min, db_max = -25.0, 0.0
        eps = 1e-6
        linear = np.clip(array, eps, None)
        db = 10.0 * np.log10(linear)
        db = np.clip(db, db_min, db_max)
        return (db - db_min) / (db_max - db_min)

    if method == "imagenet":
        if array.shape[0] != 3:
            raise ValueError(
                f"'imagenet' normalization expects a 3-channel RGB array, "
                f"got {array.shape[0]} channels. Select RGB bands first."
            )
        mean = IMAGENET_MEAN.reshape(3, 1, 1)
        std = IMAGENET_STD.reshape(3, 1, 1)
        return (array - mean) / std

    if method == "zscore":
        out = np.empty_like(array)
        for c in range(array.shape[0]):
            band = array[c]
            mean, std = float(band.mean()), float(band.std())
            out[c] = (band - mean) / std if std > 1e-8 else np.zeros_like(band)
        return out

    raise ValueError(
        f"Unknown normalization method '{method}'. Known methods: "
        f"per_band_minmax, percentile_2_98, db_scale_minmax, imagenet, zscore."
    )


def resize(array: np.ndarray, target_size: int) -> np.ndarray:
    """Resize (C,H,W) array to (C, target_size, target_size).

    Uses cv2.resize per-band (bilinear) to support arbitrary band counts —
    cv2 itself only resizes up to 4 channels at once, and RS imagery
    routinely has more (e.g. Sentinel-2's 12 bands), so we loop over bands
    rather than trying a single multi-channel call. Falls back to a plain
    nearest-neighbour numpy resize if opencv isn't installed (keeps this
    importable/usable in a minimal env; document the quality difference).
    """
    c, h, w = array.shape
    if h == target_size and w == target_size:
        return array

    if cv2 is not None:
        out = np.empty((c, target_size, target_size), dtype=array.dtype)
        for i in range(c):
            out[i] = cv2.resize(
                array[i], (target_size, target_size), interpolation=cv2.INTER_LINEAR
            )
        return out

    # Fallback: nearest-neighbour via index mapping (no opencv dependency).
    row_idx = (np.linspace(0, h - 1, target_size)).astype(np.int64)
    col_idx = (np.linspace(0, w - 1, target_size)).astype(np.int64)
    return array[:, row_idx][:, :, col_idx]


def tile_image(array: np.ndarray, tile_size: int = 256, overlap: int = 32) -> list:
    """Split a large (C,H,W) scene into overlapping tiles for models with a
    fixed input size. Returns a list of dicts:
        {"tile": np.ndarray (C, tile_size, tile_size), "offset": (x, y)}
    where `offset` is the top-left pixel coordinate of the tile in the
    ORIGINAL image — grounding/change outputs on a tile can be mapped back
    to full-image coordinates via bbox + offset.

    Edge tiles are shifted inward (not padded) so every tile is full-size;
    this means the last tile in each row/column may overlap its neighbour
    by more than `overlap` when the scene doesn't divide evenly — documented
    behaviour, not a bug, since padding would introduce fake pixels into
    exactly the border region most likely to matter for grounding/change.
    """
    if tile_size <= overlap:
        raise ValueError(f"tile_size ({tile_size}) must be > overlap ({overlap})")

    c, h, w = array.shape
    stride = tile_size - overlap

    if h <= tile_size and w <= tile_size:
        return [{"tile": array, "offset": (0, 0)}]

    def _starts(dim_size):
        if dim_size <= tile_size:
            return [0]
        starts = list(range(0, dim_size - tile_size + 1, stride))
        if starts[-1] != dim_size - tile_size:
            starts.append(dim_size - tile_size)  # ensure full coverage to the edge
        return starts

    tiles = []
    for y in _starts(h):
        for x in _starts(w):
            tiles.append({
                "tile": array[:, y:y + tile_size, x:x + tile_size],
                "offset": (x, y),
            })
    return tiles


def preprocess_pipeline(array: np.ndarray, config: dict, dataset: str,
                         model: str = None, modality: str = "optical") -> np.ndarray:
    """End-to-end preprocessing entry point used by dataset_loader.py.

    Preprocessing is now two-stage and NOT globally shared, per the team's
    per-dataset / per-model schema in configs/config.yaml:

    1. Dataset-level (sensor-specific): band selection + normalization come
       from `config["datasets"][dataset]` — e.g. Sentinel-2 bands/scaling
       for "bigearthnet" differ from Cartosat-2S bands/scaling for
       "isro_sac". `modality`: "optical" | "sar" picks which band/normalize
       keys to read (datasets with only one modality, like plain
       BigEarthNet, just use the unprefixed keys).
    2. Model-level (backbone-specific): final resize to whatever the target
       model's `image_size` expects, from `config["models"][model]` — e.g.
       GeoChat vs VisTA vs the fusion encoders may all expect different
       input sizes. Pass `model` (a key into config["models"]) to apply this
       stage; omit it to skip resizing (e.g. during raw dataset inspection).

    NOTE: for Sentinel-1/2 and (once implemented) Cartosat-2S/RISAT imagery,
    prefer routing normalization through src/preprocessing/sensor_registry.py
    (physically meaningful, sensor-specific constants) rather than the
    generic string-keyed methods below. The methods here remain useful for
    benchmark PNG/JPEG datasets (vrsbench/rsvqa/cdvqa) that carry no real
    sensor metadata (sensor="unknown" in configs/config.yaml).

    TODO(whoever wires up dataset_loader.py first): confirm actual band
    indices/normalization methods per dataset in configs/config.yaml before
    relying on this for real training.
    """
    ds_cfg = config["datasets"][dataset]

    if modality == "optical" and "optical_bands" in ds_cfg:
        bands = ds_cfg["optical_bands"]
        norm_method = ds_cfg.get("optical_normalization", ds_cfg.get("normalization"))
    elif modality == "sar" and "sar_bands" in ds_cfg:
        bands = ds_cfg["sar_bands"]
        norm_method = ds_cfg.get("sar_normalization", ds_cfg.get("normalization"))
    else:
        bands = ds_cfg.get("optical_bands") or list(range(array.shape[0]))
        norm_method = ds_cfg.get("normalization", "per_band_minmax")

    if bands is not None:
        array = select_bands(array, bands)
    array = normalize(array, method=norm_method)

    if model is not None:
        target_size = config["models"][model]["image_size"]
        array = resize(array, target_size)

    return array
