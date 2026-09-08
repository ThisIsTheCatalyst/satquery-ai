"""
Sentinel-2 (optical) sensor adapter.
Owner: Person 1 (Data & Preprocessing).

This is the sensor behind BigEarthNet / BigEarthNet-MM's optical half —
your primary development/training data. Physically meaningful normalization
(not a generic min-max) matters because the SAME adapter interface is later
implemented for Cartosat-2S (see cartosat2s.py) — if this file's assumptions
(band order, reflectance scale) are hardcoded elsewhere instead of living
here, the Cartosat swap becomes a search-and-replace across four people's
notebooks instead of one file.
"""

import numpy as np

# Sentinel-2 L2A reflectance is typically scaled 0-10000 (DN), representing
# 0-1 surface reflectance (Sen2Cor convention). Confirm this matches whatever
# BigEarthNet's actual pixel value range is before relying on it for training.
REFLECTANCE_SCALE = 10000.0

# Band order as configured in configs/config.yaml (datasets.bigearthnet.optical_bands)
BAND_NAMES = {2: "Blue (B2)", 3: "Green (B3)", 4: "Red (B4)", 8: "NIR (B8)"}


def normalize(array: np.ndarray) -> np.ndarray:
    """array: (C,H,W) raw Sentinel-2 DN values (already band-selected).
    Returns float32 array scaled to approximately [0, 1] reflectance.
    """
    array = array.astype(np.float32) / REFLECTANCE_SCALE
    return np.clip(array, 0.0, 1.0)


def describe() -> dict:
    return {
        "sensor": "Sentinel-2",
        "modality": "optical",
        "reflectance_scale": REFLECTANCE_SCALE,
        "band_names": BAND_NAMES,
    }
