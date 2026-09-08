"""
SatQuery AI — FastAPI backend.

Usage:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000

Or:
    python -m src.api.main

Endpoints:
    POST /analyze   — Main analysis endpoint
    GET  /health    — Health check
    GET  /status    — Model status
"""

import base64
import io
import os
import tempfile
import time
from typing import List, Optional, Dict, Any

import numpy as np
import yaml

try:
    from fastapi import FastAPI, HTTPException, UploadFile, File, Form
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("WARNING: FastAPI not installed. Run: pip install fastapi uvicorn")

from src.agent.controller import AgentController

# Load config once at module level
_CONFIG_PATH = os.environ.get("SATQUERY_CONFIG", "configs/config.yaml")
_DEVICE = os.environ.get("SATQUERY_DEVICE", "cpu")

with open(_CONFIG_PATH) as f:
    _CONFIG = yaml.safe_load(f)

_CONTROLLER: Optional[AgentController] = None


def _get_controller() -> AgentController:
    global _CONTROLLER
    if _CONTROLLER is None:
        _CONTROLLER = AgentController(_CONFIG, device=_DEVICE)
    return _CONTROLLER


if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="SatQuery AI",
        description="Interactive Vision-Language Assistant for Remote Sensing Image Analysis",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------------------------------------------------------------------------
    # Request / Response models
    # ---------------------------------------------------------------------------

    class ImageInput(BaseModel):
        """Base64-encoded image with its role."""
        key: str           # "image" | "image_t1" | "image_t2" | "image_optical" | "image_sar"
        data: str          # base64-encoded PNG/JPEG/TIFF content
        filename: str      # original filename for format detection

    class AnalyzeRequest(BaseModel):
        query: str
        images: List[ImageInput]

    class AnalyzeResponse(BaseModel):
        success: bool
        task: Optional[str]
        text: Optional[str]
        confidence: Optional[float]
        status: Optional[str]
        errors: List[str]
        trace: Dict[str, Any]
        spatial_evidence: Optional[Dict[str, Any]] = None
        metadata: Optional[Dict[str, Any]] = None
        latency_ms: float

    # ---------------------------------------------------------------------------
    # Endpoints
    # ---------------------------------------------------------------------------

    @app.get("/health")
    def health():
        """Health check — always returns 200 if the server is running."""
        return {"status": "ok", "timestamp": time.time()}

    @app.get("/status")
    def status():
        """Model registry status."""
        try:
            ctrl = _get_controller()
            reg_status = ctrl.registry.status()
        except Exception as e:
            reg_status = {"error": str(e)}
        return {
            "status": "ok",
            "config": _CONFIG_PATH,
            "device": _DEVICE,
            "models": reg_status,
        }

    @app.post("/analyze", response_model=AnalyzeResponse)
    def analyze(request: AnalyzeRequest):
        """Main analysis endpoint.
        
        Send base64-encoded images with their roles and a natural language query.
        The controller routes the query to the appropriate specialist model and
        returns a structured result with an execution trace.
        
        Example:
            {
                "query": "What is the dominant land cover?",
                "images": [
                    {
                        "key": "image",
                        "data": "<base64-encoded TIFF/PNG>",
                        "filename": "scene.tif"
                    }
                ]
            }
        """
        t_start = time.time()
        
        if not request.images:
            raise HTTPException(status_code=400, detail="At least one image is required.")
        
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
        # Decode images to numpy arrays
        images = {}
        for img_input in request.images:
            try:
                img_bytes = base64.b64decode(img_input.data)
                arr = _decode_image_bytes(img_bytes, img_input.filename)
                images[img_input.key] = arr
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not decode image '{img_input.key}' ({img_input.filename}): {e}"
                )
        
        # Run controller
        try:
            ctrl = _get_controller()
            result = ctrl.run(images, request.query)
        except Exception as e:
            return AnalyzeResponse(
                success=False, task=None, text=None, confidence=None,
                status="error", errors=[f"Internal error: {e}"],
                trace={}, latency_ms=(time.time() - t_start) * 1000,
            )
        
        # Sanitize spatial evidence for JSON (numpy arrays -> None, paths kept)
        se = result.get("spatial_evidence") or {}
        if isinstance(se, dict) and "mask" in se:
            # Masks can't be directly serialized to JSON
            mask = se.get("mask")
            if isinstance(mask, np.ndarray):
                se = dict(se)
                se["mask"] = None  # client should request a separate /evidence endpoint
        
        return AnalyzeResponse(
            success=result.get("success", False),
            task=result.get("task"),
            text=result.get("text"),
            confidence=result.get("confidence"),
            status=result.get("status"),
            errors=result.get("errors", []),
            trace=result.get("trace", {}),
            spatial_evidence=se if se else None,
            metadata=result.get("metadata"),
            latency_ms=(time.time() - t_start) * 1000,
        )

    @app.post("/analyze/files")
    async def analyze_files(
        query: str = Form(...),
        files: List[UploadFile] = File(...),
        keys: str = Form(...),  # comma-separated keys matching the files
    ):
        """Alternative endpoint for multipart form data (easier for web forms)."""
        key_list = [k.strip() for k in keys.split(",")]
        if len(key_list) != len(files):
            raise HTTPException(
                status_code=400,
                detail=f"Number of keys ({len(key_list)}) must match number of files ({len(files)})."
            )
        
        t_start = time.time()
        images = {}
        for key, upload in zip(key_list, files):
            content = await upload.read()
            arr = _decode_image_bytes(content, upload.filename or "image.png")
            images[key] = arr
        
        ctrl = _get_controller()
        result = ctrl.run(images, query)
        result["latency_ms"] = (time.time() - t_start) * 1000
        
        # Sanitize for JSON
        if result.get("spatial_evidence") and isinstance(
            result["spatial_evidence"].get("mask"), np.ndarray
        ):
            result["spatial_evidence"] = dict(result["spatial_evidence"])
            result["spatial_evidence"]["mask"] = None
        
        return JSONResponse(content=_jsonify(result))


    def _decode_image_bytes(content: bytes, filename: str) -> np.ndarray:
        """Decode image bytes to a (C,H,W) float32 numpy array."""
        ext = os.path.splitext(filename)[1].lower()
        
        if ext in (".tif", ".tiff"):
            # Write to temp file for rasterio
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                f.write(content)
                tmp_path = f.name
            try:
                from src.preprocessing.geotiff_utils import read_image
                rs_img = read_image(tmp_path)
                arr = rs_img.array.astype(np.float32)
            finally:
                os.unlink(tmp_path)
        else:
            # PNG/JPEG via PIL
            from PIL import Image as PILImage
            img = PILImage.open(io.BytesIO(content)).convert("RGB")
            arr = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
        
        return arr


    def _jsonify(obj):
        """Recursively convert numpy arrays and other non-serializable types."""
        if isinstance(obj, np.ndarray):
            return None  # Don't serialize large arrays
        if isinstance(obj, dict):
            return {k: _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_jsonify(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj


if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)
    except ImportError:
        print("uvicorn not installed. Run: pip install uvicorn fastapi")
