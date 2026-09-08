"""
Agent controller — the core orchestration entry point.
Pipeline: validate input -> route task -> select tool(s) -> run inference ->
integrate outputs -> log execution trace -> return response.
"""

import numpy as np

from src.agent.input_validator import validate_input
from src.agent.task_router import route_task
from src.agent.tool_registry import ToolRegistry
from src.agent.execution_trace import ExecutionTrace
from src.preprocessing.geotiff_utils import read_image
from src.preprocessing.normalize import preprocess_pipeline


class AgentController:
    def __init__(self, config: dict, device: str = "cpu"):
        self.config = config
        self.device = device
        self.registry = ToolRegistry(config, device=device)
        self.confidence_threshold = config.get("agent", {}).get("confidence_threshold", 0.4)

    def run(self, images: dict, query: str) -> dict:
        """images: see input_validator.validate_input for expected keys.
        Accepts either file paths (str) or pre-loaded numpy arrays.
        query: natural-language question/instruction from the user.
        """
        # 1. Validate input
        validation = validate_input(images, self.config)
        trace = ExecutionTrace(query=query, mode=validation.get("mode"))
        trace.set_input_metadata(validation.get("metadata", {}))

        if not validation["valid"]:
            return self._error_response(validation["errors"], trace, task=None)

        mode = validation["mode"]

        # 2. Route task
        task = route_task(query, mode)
        trace.set_task(task)

        # 3. Select + execute tool
        try:
            tool = self.registry.get(task)
        except KeyError as e:
            trace.add_warning(str(e))
            return self._error_response([str(e)], trace, task=task)

        # 4. Build predict kwargs (load + preprocess images if paths were given)
        try:
            params = self._build_predict_kwargs(images, query, mode, task)
        except Exception as e:
            trace.add_warning(f"Image loading/preprocessing failed: {e}")
            return self._error_response([f"Image preprocessing error: {e}"], trace, task=task)

        trace.add_tool_call(name=tool.name, task=task, params={"query": query, "mode": mode})

        # 5. Run inference
        try:
            result = tool.predict(**params)
        except Exception as e:
            trace.add_warning(f"Model inference failed: {type(e).__name__}: {e}")
            result = tool._empty_result(
                text=f"Inference error: {type(e).__name__}: {e}",
                confidence=0.0,
                status="error",
            )

        # 6. Confidence flag
        conf = result.get("confidence", 0.0) or 0.0
        if conf < self.confidence_threshold:
            trace.add_warning(
                f"Low confidence ({conf:.2f} < {self.confidence_threshold})"
            )
        trace.set_confidence(conf)

        return {
            "success": True,
            "task": result["task"],
            "text": result["text"],
            "confidence": result["confidence"],
            "spatial_evidence": result.get("spatial_evidence"),
            "metadata": result.get("metadata"),
            "status": result.get("status"),
            "errors": [],
            "trace": trace.finalize(),
        }

    def _error_response(self, errors: list, trace: "ExecutionTrace", task: str = None) -> dict:
        return {
            "success": False,
            "task": task,
            "text": None,
            "confidence": None,
            "spatial_evidence": None,
            "metadata": None,
            "status": "error",
            "errors": errors,
            "trace": trace.finalize(),
        }

    def _build_predict_kwargs(self, images: dict, query: str, mode: str, task: str) -> dict:
        """Map image paths/arrays -> preprocessed numpy arrays in model kwargs."""
        model_key = task  # config["models"][task]["image_size"]
        
        def _load_and_preprocess(key_or_path, modality="optical"):
            """Load if string path, convert if tensor, ensure numpy float32."""
            data = images.get(key_or_path) if isinstance(key_or_path, str) else key_or_path
            if data is None:
                data = key_or_path  # direct value
            
            if isinstance(data, str):
                # It's a file path
                rs_img = read_image(data)
                arr = rs_img.array
            elif hasattr(data, 'numpy'):
                arr = data.numpy()
            elif isinstance(data, np.ndarray):
                arr = data
            else:
                arr = np.array(data)
            
            # Normalize if not already in [0,1] range
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            if arr.max() > 1.0 and arr.max() <= 255.0:
                arr = arr / 255.0
            elif arr.max() > 255.0:
                arr = arr / arr.max()
            
            return arr
        
        if mode == "single":
            img = _load_and_preprocess(images.get("image", "image"), "optical")
            return {"image": img, "query": query}
        
        if mode == "cross_modal":
            opt = _load_and_preprocess(images.get("image_optical", "image_optical"), "optical")
            sar = _load_and_preprocess(images.get("image_sar", "image_sar"), "sar")
            return {"image_optical": opt, "image_sar": sar, "query": query}
        
        if mode == "bi_temporal":
            t1 = _load_and_preprocess(images.get("image_t1", "image_t1"), "optical")
            t2 = _load_and_preprocess(images.get("image_t2", "image_t2"), "optical")
            return {"image_t1": t1, "image_t2": t2, "query": query}
        
        raise ValueError(f"Unknown mode: {mode}")
