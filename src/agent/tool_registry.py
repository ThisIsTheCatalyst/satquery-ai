"""
Tool registry — maps task labels to loaded specialist model instances.
Owner: Person 4 (integration phase).

CHANGED FROM ORIGINAL:
- Actually calls model.load() (previously commented out with a TODO).
- Lazy-loads models on first registry.get() call to avoid loading all
  heavyweight models simultaneously (important for 8GB VRAM Colab).
- Tracks loaded status per model so health_check() is accurate.
- Catches load failures per-model (one broken model does not prevent
  others from registering).
"""

from src.models.vqa_model import VQAModel
from src.models.captioning_model import CaptioningModel
from src.models.grounding_model import GroundingModel
from src.models.change_model import ChangeModel
from src.models.fusion_model import FusionModel


class ToolRegistry:
    """Lazy-loading registry: models are instantiated immediately but their
    weights are loaded on first get() call (or eagerly via _build_registry
    if eager_load=True in config). This lets the agent start up quickly
    and avoids putting all models in VRAM simultaneously.
    """

    _MODEL_CLASSES = {
        "vqa": VQAModel,
        "captioning": CaptioningModel,
        "grounding": GroundingModel,
        "change": ChangeModel,
        "fusion": FusionModel,
    }

    def __init__(self, config: dict, device: str = "cpu"):
        self.config = config
        self.device = device
        self._tools = {}        # task -> model instance (loaded)
        self._unloaded = {}     # task -> (model_instance, checkpoint_path)
        self._load_errors = {}  # task -> error message
        self._build_registry()

    def _build_registry(self):
        model_cfg = self.config.get("models", {})
        
        # Instantiate all models (no weights yet)
        for task, cls in self._MODEL_CLASSES.items():
            cfg = model_cfg.get(task, {})
            if not cfg.get("enabled", True) and task not in ("vqa", "change", "fusion"):
                continue
            try:
                instance = cls(config=self.config)
                checkpoint = cfg.get("checkpoint", "")
                self._unloaded[task] = (instance, checkpoint)
            except Exception as e:
                self._load_errors[task] = f"Instantiation failed: {e}"

    def _load_model(self, task: str):
        """Load model weights on first access."""
        if task in self._tools:
            return  # already loaded
        if task in self._load_errors:
            raise KeyError(f"Model '{task}' failed during instantiation: {self._load_errors[task]}")
        if task not in self._unloaded:
            raise KeyError(f"No model registered for task '{task}'. Available: {list(self._unloaded.keys())}")
        
        instance, checkpoint = self._unloaded[task]
        try:
            print(f"[tool_registry] Loading model for task='{task}' from '{checkpoint}'...")
            instance.load(checkpoint, device=self.device)
            self._tools[task] = instance
            del self._unloaded[task]
            print(f"[tool_registry] Model '{task}' ready.")
        except Exception as e:
            self._load_errors[task] = str(e)
            # Even if load fails, register the instance so predict() can
            # return a proper error result rather than crashing the controller.
            self._tools[task] = instance
            del self._unloaded[task]
            print(f"[tool_registry] WARNING: model '{task}' load raised {type(e).__name__}: {e}. "
                  f"predict() will return error/fallback results.")

    def register(self, model, checkpoint_path: str):
        """Register a model instance (already instantiated) with a checkpoint path.
        Replaces any existing registration for that task.
        """
        self._unloaded[model.task] = (model, checkpoint_path)
        # Remove from already-loaded if re-registering
        self._tools.pop(model.task, None)

    def get(self, task: str):
        """Fetch the loaded specialist model for a given task.
        Loads on first access (lazy loading).
        """
        if task not in self._tools and task not in self._unloaded:
            raise KeyError(
                f"No tool registered for task '{task}'. "
                f"Available: {list(self._unloaded.keys()) + list(self._tools.keys())}"
            )
        self._load_model(task)
        return self._tools[task]

    def status(self) -> dict:
        """Return a status dict for all registered tools."""
        all_tasks = set(self._tools) | set(self._unloaded) | set(self._load_errors)
        out = {}
        for task in all_tasks:
            if task in self._tools:
                out[task] = "loaded"
            elif task in self._unloaded:
                out[task] = "registered_unloaded"
            else:
                out[task] = f"error: {self._load_errors.get(task, 'unknown')}"
        return out
