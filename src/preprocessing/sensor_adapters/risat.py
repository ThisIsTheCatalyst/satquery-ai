"""
RISAT (SAR) sensor adapter — STUB.
Owner: whoever gets ISRO/SAC sample data first. See cartosat2s.py's
docstring for the full rationale — same reasoning applies here, and doubly
so for SAR: RISAT's dB calibration constants, VV/VH (or other polarization)
layout, and speckle characteristics are almost certainly different from
Sentinel-1's, and a wrong silent guess here would corrupt both the change
and fusion tasks on the hidden evaluation set.
"""

import numpy as np

# TODO: confirm actual RISAT polarization/band layout once sample data or
# spec sheet is available.
BAND_NAMES = {}  # unconfirmed

# TODO: confirm actual dB range for RISAT backscatter — do not copy
# Sentinel-1's DB_MIN/DB_MAX as a guess; RISAT's frequency band and
# acquisition mode likely give a different dynamic range.
DB_MIN, DB_MAX = None, None


def normalize(array: np.ndarray) -> np.ndarray:
    raise NotImplementedError(
        "RISAT normalization constants are unconfirmed — this is a "
        "deliberate stub (see module docstring and docs/DECISIONS.md). "
        "Fill in DB_MIN/DB_MAX/BAND_NAMES from real RISAT spec or sample "
        "data before implementing, not from a guess."
    )


def describe() -> dict:
    return {
        "sensor": "RISAT",
        "modality": "sar",
        "status": "stub — unconfirmed band layout/dB range",
    }
