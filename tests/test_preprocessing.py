"""Tests for preprocessing utilities."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from src.preprocessing.normalize import select_bands, normalize, resize, tile_image


def test_select_bands_shape():
    arr = np.random.rand(4, 32, 32).astype("float32")
    out = select_bands(arr, [0, 1, 2])
    assert out.shape == (3, 32, 32)


def test_select_bands_single_band():
    arr = np.random.rand(4, 32, 32).astype("float32")
    out = select_bands(arr, [3])
    assert out.shape == (1, 32, 32)


def test_select_bands_out_of_range():
    arr = np.random.rand(4, 32, 32).astype("float32")
    with pytest.raises(IndexError):
        select_bands(arr, [4])


def test_normalize_per_band_minmax():
    arr = np.random.rand(3, 32, 32).astype("float32") * 100
    out = normalize(arr, "per_band_minmax")
    assert out.dtype == np.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_normalize_percentile_2_98():
    arr = np.random.rand(3, 64, 64).astype("float32") * 10000
    out = normalize(arr, "percentile_2_98")
    assert out.dtype == np.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_normalize_db_scale_minmax():
    # SAR-like data: small positive linear backscatter
    arr = np.random.uniform(1e-5, 1.0, (2, 32, 32)).astype("float32")
    out = normalize(arr, "db_scale_minmax")
    assert out.dtype == np.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_normalize_imagenet_requires_3_channels():
    arr = np.random.rand(3, 32, 32).astype("float32")
    out = normalize(arr, "imagenet")
    assert out.shape == (3, 32, 32)
    
    arr_4ch = np.random.rand(4, 32, 32).astype("float32")
    with pytest.raises(ValueError):
        normalize(arr_4ch, "imagenet")


def test_normalize_unknown_method():
    arr = np.random.rand(3, 32, 32).astype("float32")
    with pytest.raises(ValueError):
        normalize(arr, "nonexistent_method")


def test_resize_square():
    arr = np.random.rand(4, 128, 128).astype("float32")
    out = resize(arr, 64)
    assert out.shape == (4, 64, 64)


def test_resize_noop():
    arr = np.random.rand(3, 64, 64).astype("float32")
    out = resize(arr, 64)
    assert out.shape == (3, 64, 64)
    assert np.allclose(out, arr)


def test_tile_image_no_tiling_needed():
    arr = np.random.rand(4, 64, 64).astype("float32")
    tiles = tile_image(arr, tile_size=256, overlap=32)
    assert len(tiles) == 1
    assert tiles[0]["offset"] == (0, 0)
    assert tiles[0]["tile"].shape == (4, 64, 64)


def test_tile_image_2x2():
    arr = np.random.rand(4, 512, 512).astype("float32")
    tiles = tile_image(arr, tile_size=256, overlap=0)
    assert len(tiles) >= 4
    for t in tiles:
        assert t["tile"].shape == (4, 256, 256)


def test_tile_image_tile_gt_overlap():
    with pytest.raises(ValueError):
        tile_image(np.zeros((3, 512, 512)), tile_size=32, overlap=64)


def test_sensor_registry():
    from src.preprocessing.sensor_registry import get_adapter, list_sensors
    
    sensors = list_sensors()
    assert "Sentinel-2" in sensors
    assert "Sentinel-1" in sensors
    
    s2 = get_adapter("Sentinel-2")
    info = s2.describe()
    assert info["modality"] == "optical"


def test_sentinel2_normalize():
    from src.preprocessing.sensor_adapters.sentinel2 import normalize as s2_norm
    
    # Sentinel-2 DN values (0-10000)
    arr = np.random.uniform(0, 10000, (4, 32, 32)).astype("float32")
    out = s2_norm(arr)
    assert out.dtype == np.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_sentinel1_normalize():
    from src.preprocessing.sensor_adapters.sentinel1 import normalize as s1_norm
    
    # SAR backscatter: small positive linear values
    arr = np.random.uniform(1e-6, 0.5, (2, 32, 32)).astype("float32")
    out = s1_norm(arr, already_db=False)
    assert out.dtype == np.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_cartosat_stub():
    from src.preprocessing.sensor_adapters.cartosat2s import normalize as c_norm
    with pytest.raises(NotImplementedError):
        c_norm(np.zeros((4, 32, 32)))


def test_bigearthnet_adapter_parsing():
    """Test BigEarthNet label parsing with in-memory data."""
    import tempfile
    from src.preprocessing.bigearthnet_adapter import parse_bigearthnet_labels, labels_to_caption
    
    # Test with tab-separated format
    content = "patch_001\tConiferous forest,Water bodies\npatch_002\tPastures\n"
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(content)
        tmp = f.name
    
    try:
        records = parse_bigearthnet_labels(tmp)
        assert len(records) == 2
        assert records[0]["image_id"] == "patch_001"
        assert "Coniferous forest" in records[0]["labels"]
        assert "Water bodies" in records[0]["labels"]
    finally:
        os.unlink(tmp)
    
    # Test caption generation
    caption = labels_to_caption(["Forest", "Water bodies"])
    assert "forest" in caption.lower() or "water" in caption.lower()


def test_geotiff_utils_plain_image():
    """Test reading a synthetic PNG (no rasterio needed)."""
    import tempfile
    from PIL import Image
    from src.preprocessing.geotiff_utils import read_image
    
    arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img.save(f.name)
        tmp = f.name
    
    try:
        rs_img = read_image(tmp)
        assert rs_img.array.shape == (3, 64, 64)
        assert rs_img.crs is None  # PNG has no georeferencing
        assert rs_img.band_count == 3
    finally:
        os.unlink(tmp)
