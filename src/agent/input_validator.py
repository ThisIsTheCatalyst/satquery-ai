"""
Input validator — checks number, modality, format, metadata, and
co-registration compatibility of uploaded images before task routing.
"""

import numpy as np
from src.preprocessing.geotiff_utils import read_image, check_coregistration, extract_metadata

ALLOWED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def validate_input(images: dict, config: dict) -> dict:
    """images: one of
        {"image": path_or_array}                         -> single image
        {"image_optical": ..., "image_sar": ...}         -> cross-modal pair
        {"image_t1": ..., "image_t2": ...}               -> bi-temporal pair

    Returns:
        {
          "valid": bool,
          "mode": "single" | "cross_modal" | "bi_temporal" | None,
          "errors": list[str],
          "metadata": dict,
        }
    """
    errors = []
    metadata = {}

    keys = set(images.keys())
    if keys == {"image"}:
        mode = "single"
    elif keys == {"image_optical", "image_sar"}:
        mode = "cross_modal"
    elif keys == {"image_t1", "image_t2"}:
        mode = "bi_temporal"
    else:
        return {"valid": False, "mode": None,
                "errors": [f"Unrecognized image key combination: {keys}. "
                           f"Expected one of: {{image}}, {{image_optical,image_sar}}, "
                           f"{{image_t1,image_t2}}"],
                "metadata": {}}

    # Format check (only for string paths)
    for key, value in images.items():
        if isinstance(value, str):
            ext = "." + value.lower().rsplit(".", 1)[-1] if "." in value else ""
            if ext not in ALLOWED_EXTENSIONS:
                errors.append(f"{key}: unsupported format '{ext}'. "
                              f"Supported: {sorted(ALLOWED_EXTENSIONS)}")
        elif isinstance(value, np.ndarray):
            # Validate array shape: must be (C, H, W) with C >= 1
            if value.ndim != 3 or value.shape[0] < 1:
                errors.append(f"{key}: expected (C,H,W) array, got shape {value.shape}")
        elif hasattr(value, 'numpy'):
            # Torch tensor
            shape = tuple(value.shape)
            if len(shape) != 3 or shape[0] < 1:
                errors.append(f"{key}: expected (C,H,W) tensor, got shape {shape}")
        else:
            errors.append(f"{key}: unsupported image type {type(value).__name__}. "
                          f"Expected a file path (str) or numpy array.")

    if errors:
        return {"valid": False, "mode": mode, "errors": errors, "metadata": {}}

    # Load metadata from file paths (best-effort — don't fail hard on I/O errors)
    for key, value in images.items():
        if isinstance(value, str):
            try:
                rs_img = read_image(value)
                metadata[key] = extract_metadata(rs_img)
            except Exception as e:
                metadata[key] = {"source_path": value, "load_error": str(e)}
        elif isinstance(value, np.ndarray):
            metadata[key] = {"shape": value.shape, "dtype": str(value.dtype),
                             "source": "numpy_array"}
        else:
            metadata[key] = {"source": "tensor"}

    # Co-registration check for paired inputs (file paths only — arrays assumed pre-aligned)
    if mode in ("cross_modal", "bi_temporal"):
        paths = [v for v in images.values() if isinstance(v, str)]
        if len(paths) == 2:
            try:
                imgs = [read_image(p) for p in paths]
                result = check_coregistration(imgs[0], imgs[1])
                if not result["co_registered"]:
                    errors.append(f"Images are not co-registered: {result['reason']}")
            except Exception as e:
                # Non-fatal: log as warning in metadata, don't reject the input
                metadata["coregistration_check"] = f"Could not verify: {e}"

    return {
        "valid": len(errors) == 0,
        "mode": mode,
        "errors": errors,
        "metadata": metadata,
    }
