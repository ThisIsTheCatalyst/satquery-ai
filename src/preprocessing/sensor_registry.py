"""
Sensor registry — single dispatch point from a sensor name (as recorded in
ImageSample/PairSample/FusionSample.sensor, or configs/config.yaml's
per-dataset `sensor`/`optical_sensor`/`sar_sensor` keys) to its adapter
module. Owner: shared.

Usage:
    from src.preprocessing.sensor_registry import get_adapter
    adapter = get_adapter("Sentinel-2")
    array = adapter.normalize(raw_array)

This is the seam described in cartosat2s.py/risat.py: when ISRO/SAC sample
data appears, someone implements those two adapters' normalize() functions
and every caller of get_adapter() picks it up automatically — no changes
needed anywhere else.
"""

from src.common.constants import (
    SENSOR_SENTINEL1, SENSOR_SENTINEL2, SENSOR_CARTOSAT2S, SENSOR_RISAT,
    SENSOR_UNKNOWN,
)
from src.preprocessing.sensor_adapters import sentinel1, sentinel2, cartosat2s, risat

_REGISTRY = {
    SENSOR_SENTINEL2: sentinel2,
    SENSOR_SENTINEL1: sentinel1,
    SENSOR_CARTOSAT2S: cartosat2s,
    SENSOR_RISAT: risat,
}


def get_adapter(sensor_name: str):
    """Return the adapter module for `sensor_name`.

    Raises KeyError for anything not in the registry (including
    SENSOR_UNKNOWN — benchmark PNG/JPEG datasets should go through the
    common/model-specific preprocessing path in normalize.py, not a sensor
    adapter, since they carry no real sensor metadata).
    """
    if sensor_name not in _REGISTRY:
        raise KeyError(
            f"No sensor adapter registered for '{sensor_name}'. "
            f"Known sensors: {list(_REGISTRY.keys())}. "
            f"Benchmark datasets with sensor='{SENSOR_UNKNOWN}' should use "
            f"the common preprocessing path instead (see normalize.py)."
        )
    return _REGISTRY[sensor_name]


def list_sensors() -> list:
    return list(_REGISTRY.keys())
