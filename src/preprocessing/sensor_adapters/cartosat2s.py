"""
Cartosat-2S (optical) sensor adapter — STUB.
Owner: whoever gets ISRO/SAC sample data first (not assigned during the
2-day ML sprint; this file just needs to exist so the seam is real).

WHY THIS FILE EXISTS EVEN THOUGH WE HAVE NO CARTOSAT DATA YET:
The hidden final-judging set uses Cartosat-2S optical + RISAT SAR — a
DIFFERENT sensor from the Sentinel-1/2 data all four notebooks train and
validate against. If Sentinel assumptions (band order, reflectance scale
~0-10000, resolution ~10m) are hardcoded into shared preprocessing, Cartosat
imagery will be silently mis-normalized and every downstream model will get
garbage input with NO error thrown — the worst kind of failure, because it
looks like it's working right up until judging.

Building this seam costs ~30 minutes now. It means that when/if ISRO/SAC
sample data appears, someone fills in ONE file instead of hunting through
four people's notebooks for hardcoded Sentinel-2 assumptions.

DO NOT guess real Cartosat-2S radiometric constants and ship them silently —
that's worse than an explicit NotImplementedError, because a wrong silent
guess is indistinguishable from a working pipeline until judging day.
"""

import numpy as np

# TODO: confirm actual Cartosat-2S band layout/order once sample data or
# spec sheet is available (see configs/config.yaml -> datasets.isro_sac).
BAND_NAMES = {}  # unconfirmed

# TODO: confirm actual radiometric range (DN scale, whether already
# reflectance-calibrated, bit depth). Do not copy Sentinel-2's
# REFLECTANCE_SCALE = 10000.0 as a guess.
REFLECTANCE_SCALE = None


def normalize(array: np.ndarray) -> np.ndarray:
    raise NotImplementedError(
        "Cartosat-2S normalization constants are unconfirmed — this is a "
        "deliberate stub (see module docstring and docs/DECISIONS.md). "
        "Fill in REFLECTANCE_SCALE/BAND_NAMES from real Cartosat-2S spec "
        "or sample data before implementing, not from a guess."
    )


def describe() -> dict:
    return {
        "sensor": "Cartosat-2S",
        "modality": "optical",
        "status": "stub — unconfirmed band layout/radiometric range",
    }
