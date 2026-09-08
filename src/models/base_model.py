"""
Base interface every specialist model must implement.
Shared contract — see docs/api_contracts.md and src/common/schemas.py.
Owner: shared (Person 1-4 implement subclasses, agent depends on this).

IMPORTANT: predict() must return a dict matching src.common.schemas.RSModelResult
(use RSModelResult(...).to_dict() or the empty_result() helper — do not
hand-roll a differently-shaped dict, or the agent/evaluation code breaks).

Input side stays explicit kwargs (image=..., query=..., image_t1=...,
image_optical=..., etc.) rather than a single typed object — see the note
at the top of src/common/schemas.py for why. Data LOADING (dataset_loader.py)
uses the typed ImageSample/PairSample/FusionSample instead; convert to
kwargs at the call site.
"""

from abc import ABC, abstractmethod

from src.common.schemas import empty_result


class BaseRSModel(ABC):
    """All specialist models (VQA, captioning, grounding, change, fusion)
    subclass this so the agent controller can call any of them uniformly.
    """

    name: str = "base_model"        # unique id, used in tool_registry
    task: str = "undefined"         # one of src.common.constants.ALL_TASKS
    backbone: str = "undefined"     # e.g. "GeoChat", "VisTA" — for metadata/trace

    @abstractmethod
    def load(self, checkpoint_path: str, device: str = "cpu") -> None:
        """Load model weights onto `device`."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, **inputs) -> dict:
        """Run inference. MUST return a dict shaped like
        src.common.schemas.RSModelResult.to_dict():
        {
          "task": str,
          "text": str,
          "confidence": float,          # 0.0 - 1.0
          "spatial_evidence": {
              "type": "bbox" | "mask" | "none",
              "source": "image" | "image_t1" | "image_t2" | "optical" | "sar" | "fused",
              "bbox": [...] | None,
              "mask": np.ndarray | None,
          },
          "metadata": {
              "model": str, "backbone": str, "checkpoint": str, "dataset": str,
              "input_modalities": [...], "parameters": {...},
          },
          "status": "success" | "error" | "low_confidence",
          "inference_seconds": float,
        }
        Use self._empty_result(...) for stubs/errors so the shape is always correct.
        """
        raise NotImplementedError

    def torch_module(self):
        """Return the underlying torch.nn.Module that actually holds trainable
        weights, so shared training code (src/training/trainer_utils.py) can
        call .parameters()/.state_dict()/.load_state_dict() on the right
        object without knowing each model's internal attribute names.

        BUG THIS FIXES: BaseRSModel is a thin ABC wrapper, NOT an nn.Module —
        it deliberately holds `self.model` (or, for fusion, several
        sub-modules) as plain attributes so predict() can stay simple. Code
        that called `model.parameters()` / `model.state_dict()` directly on
        a BaseRSModel instance (as trainer_utils.py originally did) raises
        AttributeError the moment real model internals exist. Every
        trainable subclass must override this; the default here raises so
        the failure is loud and immediate instead of a silent no-op that
        "trains" zero parameters.

        For models with more than one trainable component (e.g. FusionModel
        has a fusion_head plus optionally-unfrozen encoders), return an
        nn.ModuleList or nn.ModuleDict bundling everything that should
        receive gradients — trainer_utils only needs something with a
        working .parameters()/.state_dict().
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not override torch_module() — "
            f"trainer_utils needs this to build an optimizer / save a "
            f"checkpoint. Return the underlying nn.Module (or an "
            f"nn.ModuleDict of the trainable pieces)."
        )

    def health_check(self) -> dict:
        """Cheap self-test used by scripts/smoke_test.py: confirms the model
        loads and can run one forward pass on a trivial dummy input. Override
        in subclasses once load()/predict() are implemented — the default
        here just reports whether load() has been called.
        """
        return {
            "name": self.name,
            "task": self.task,
            "backbone": self.backbone,
            "loaded": getattr(self, "checkpoint_path", None) is not None,
        }

    def _empty_result(self, text: str = "", confidence: float = 0.0,
                       checkpoint: str = "", status: str = "error") -> dict:
        """Convenience helper subclasses use to build a contract-compliant
        placeholder/error result without hand-rolling the dict shape."""
        return empty_result(
            task=self.task, text=text, confidence=confidence,
            model=self.name, checkpoint=checkpoint, status=status,
        )
