"""
Sentinel-1 (SAR) sensor adapter.
Owner: Person 4 (Optical-SAR fusion), also used by Person 3 (change) if a
bi-temporal pair includes SAR.

SAR needs different handling from optical: raw backscatter is linear power,
usually converted to dB before normalization, and speckle noise is present.
Document exactly what's done here so nobody on the team invents an
undocumented, inconsistent SAR preprocessing method in their own notebook.
"""

import numpy as np

BAND_NAMES = {1: "VV", 2: "VH"}

# Typical dB range for Sentinel-1 GRD backscatter over land; used to min-max
# scale after dB conversion. Confirm against your actual data distribution
# before trusting this for training — these are reasonable defaults, not
# measured constants.
DB_MIN, DB_MAX = -25.0, 0.0

# Speckle filtering (e.g. Lee/Frost filter) is explicitly OUT OF SCOPE for
# the initial sprint — the seam exists (see the `speckle_filter` param) but
# no filter is applied by default. Guessing filter constants without ground
# truth wastes time; leave this as a documented known gap for now (see
# docs/DECISIONS.md).


def to_db(array: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Convert linear backscatter power to dB: 10 * log10(x)."""
    array = np.clip(array.astype(np.float32), eps, None)
    return 10.0 * np.log10(array)


def normalize(array: np.ndarray, already_db: bool = False,
              speckle_filter: bool = False) -> np.ndarray:
    """array: (C,H,W) raw or dB-scaled Sentinel-1 backscatter (already band-selected).

    speckle_filter: deliberately unimplemented this sprint — see module
    docstring. Raises if requested so nobody silently gets a no-op.
    """
    if speckle_filter:
        raise NotImplementedError(
            "Speckle filtering is an explicitly out-of-scope stretch goal "
            "for this sprint (see docs/DECISIONS.md). Pass speckle_filter=False."
        )
    db = array.astype(np.float32) if already_db else to_db(array)
    db = np.clip(db, DB_MIN, DB_MAX)
    return (db - DB_MIN) / (DB_MAX - DB_MIN)


def describe() -> dict:
    return {
        "sensor": "Sentinel-1",
        "modality": "sar",
        "db_range": (DB_MIN, DB_MAX),
        "band_names": BAND_NAMES,
        "speckle_filter": "not implemented (known gap, see docs/DECISIONS.md)",
    }
