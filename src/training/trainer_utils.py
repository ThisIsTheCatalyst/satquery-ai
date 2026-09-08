"""
Shared training utilities: seeding, checkpointing, logging, optimizer/scheduler
setup, so each train_*.py script isn't reinventing the same boilerplate.
Owner: shared, initial version by whoever starts training first.

CHANGE LOG (ML audit pass):
- FIXED a real bug: build_optimizer/save_checkpoint/load_checkpoint used to
  call model.parameters() / model.state_dict() directly on a BaseRSModel
  instance. BaseRSModel is a plain ABC wrapper, not an nn.Module — it holds
  the real weights in an attribute (self.model, or several attributes for
  fusion). This raised AttributeError as soon as any subclass had real
  internals. Fixed by routing through BaseRSModel.torch_module() (see
  src/models/base_model.py), which every trainable model must implement.
- Added: AMP (torch.amp), gradient accumulation, checkpoint resume,
  best-checkpoint tracking, structured checkpoint metadata (per the ML
  audit's "checkpoint metadata" requirement), and a CPU-safe path so smoke
  tests never pretend a 1-batch CPU run is production training.
"""

import datetime
import json
import os
import random
import subprocess
from contextlib import nullcontext

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    """Seed python/numpy/torch/CUDA. NOTE: this does not guarantee bit-exact
    determinism across all CUDA ops (some kernels are non-deterministic by
    design) — see docs/ML_AUDIT.md for the reproducibility caveat we
    document rather than silently claim isn't there.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preferred: str = "cuda") -> str:
    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_git_commit() -> str:
    """Best-effort short git SHA for checkpoint/experiment provenance.
    Returns "unknown" outside a git repo or if git isn't available — never
    raises, since this is metadata, not a hard requirement.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Optimizer / scheduler
# ---------------------------------------------------------------------------

def _trainable_torch_module(model):
    """Resolve the real nn.Module behind a BaseRSModel wrapper.

    Accepts either a BaseRSModel (calls .torch_module()) or a plain
    nn.Module directly (returned as-is) so this file also works from
    contexts that already have the raw module (e.g. quick notebook cells).
    """
    if isinstance(model, torch.nn.Module):
        return model
    if hasattr(model, "torch_module"):
        return model.torch_module()
    raise TypeError(
        f"Don't know how to get trainable parameters from {type(model)}. "
        f"Pass an nn.Module, or a BaseRSModel subclass that implements "
        f"torch_module()."
    )


def build_optimizer(model, config: dict):
    """Build AdamW over the model's trainable parameters only (important for
    LoRA/frozen-encoder setups where most parameters have requires_grad=False
    and must not accumulate dead optimizer state).
    """
    t = config["training"]
    module = _trainable_torch_module(model)
    params = [p for p in module.parameters() if p.requires_grad]
    if not params:
        raise ValueError(
            "build_optimizer: zero trainable parameters found. If you froze "
            "encoders intentionally, make sure the fusion head / LoRA "
            "adapter params still have requires_grad=True."
        )
    return torch.optim.AdamW(params, lr=t["lr"], weight_decay=t["weight_decay"])


def build_scheduler(optimizer, config: dict, num_training_steps: int):
    """Linear warmup + linear decay, driven by config['training']['warmup_ratio'].
    Returns None if num_training_steps <= 0 (e.g. an empty/smoke dataloader)
    so callers can skip scheduler.step() safely rather than crashing on a
    degenerate LambdaLR.
    """
    if num_training_steps <= 0:
        return None
    warmup_ratio = config["training"].get("warmup_ratio", 0.0)
    warmup_steps = int(num_training_steps * warmup_ratio)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return step / max(1, warmup_steps)
        remaining = num_training_steps - warmup_steps
        if remaining <= 0:
            return 1.0
        progress = (step - warmup_steps) / remaining
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def amp_context(device: str, enabled: bool):
    """Return the right autocast context manager for the device, or a no-op
    context if AMP is disabled/unsupported. Using torch.amp (not the
    deprecated torch.cuda.amp) per the ML audit's PyTorch-version note.
    CPU autocast exists in modern torch but rarely helps — disabled by
    default for device=="cpu" even if `enabled` is True, so smoke tests on
    CPU don't silently get slower/weirder numerics for no benefit.
    """
    if enabled and device == "cuda":
        return torch.amp.autocast(device_type="cuda")
    return nullcontext()


def get_grad_scaler(device: str, enabled: bool):
    """GradScaler is only meaningful for CUDA fp16 AMP; return a
    disabled-but-valid scaler otherwise so callers can always call
    scaler.scale(loss)/scaler.step(opt)/scaler.update() uniformly.
    """
    return torch.amp.GradScaler(device="cuda" if device == "cuda" else "cpu",
                                 enabled=(enabled and device == "cuda"))


# ---------------------------------------------------------------------------
# Checkpointing — now with structured metadata (ML audit requirement #32)
# ---------------------------------------------------------------------------

def save_checkpoint(model, path: str, epoch: int, optimizer=None, scheduler=None,
                     config: dict = None, dataset: str = "", split: str = "",
                     seed: int = None, extra_metadata: dict = None,
                     is_best: bool = False) -> dict:
    """Save weights + a structured metadata sidecar.

    Writes:
      {path}                 — torch.save state dict (model/optimizer/scheduler)
      {path}.meta.json       — human-readable provenance (see fields below)
    If is_best, additionally writes {path}.best and {path}.best.meta.json so
    "last" and "best" checkpoints never silently overwrite each other.

    Metadata fields (per ML_AUDIT.md checkpoint-metadata requirement):
    model/backbone/task, dataset, split, epoch, optimizer name, learning
    rate, seed, preprocessing version (from config, if present), full
    resolved config, git commit, and wall-clock date.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    module = _trainable_torch_module(model)

    state = {"model_state_dict": module.state_dict(), "epoch": epoch}
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state, path)

    metadata = {
        "model": getattr(model, "name", type(model).__name__),
        "backbone": getattr(model, "backbone", "unknown"),
        "task": getattr(model, "task", "unknown"),
        "dataset": dataset,
        "dataset_split": split,
        "epoch": epoch,
        "optimizer": type(optimizer).__name__ if optimizer is not None else None,
        "learning_rate": config["training"]["lr"] if config else None,
        "seed": seed,
        "preprocessing_version": (config or {}).get("preprocessing_version", "unversioned"),
        "git_commit": get_git_commit(),
        "date_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "config_snapshot": config,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    with open(path + ".meta.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    if is_best:
        torch.save(state, path + ".best")
        with open(path + ".best.meta.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    return metadata


def load_checkpoint(model, path: str, optimizer=None, scheduler=None,
                     map_location: str = "cpu") -> dict:
    """Load weights (+ optimizer/scheduler state for resume) into the
    underlying torch module of a BaseRSModel. Returns the saved metadata
    dict if a {path}.meta.json sidecar exists, else {"epoch": ...} only.
    """
    module = _trainable_torch_module(model)
    state = torch.load(path, map_location=map_location, weights_only=False)
    module.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])

    meta_path = path + ".meta.json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        meta["epoch"] = state.get("epoch", meta.get("epoch", 0))
        return meta
    return {"epoch": state.get("epoch", 0)}


def find_resumable_checkpoint(path: str) -> str:
    """Return `path` if a resumable checkpoint file already exists there,
    else None. Kept as a tiny function (not inlined in every train_*.py)
    so resume logic is identical everywhere.
    """
    return path if os.path.exists(path) else None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class SimpleLogger:
    """Minimal stdout + file logger; swap for tensorboard/wandb if preferred."""

    def __init__(self, log_file: str = None):
        self.log_file = log_file
        if log_file:
            os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    def log(self, step: int, metrics: dict):
        msg = f"[step {step}] " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(msg)
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(msg + "\n")


# ---------------------------------------------------------------------------
# Early stopping — small, stateful helper shared by every train_*.py
# ---------------------------------------------------------------------------

class EarlyStopper:
    """Tracks the best validation metric seen so far and reports whether
    training should stop. Higher-is-better by default (accuracy/F1/IoU);
    pass mode="min" for loss-like metrics.
    """

    def __init__(self, patience: int = 3, mode: str = "max", min_delta: float = 0.0):
        assert mode in ("max", "min")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.num_bad_epochs = 0

    def step(self, value: float) -> bool:
        """Call once per epoch with the validation metric. Returns True if
        `value` is the new best (caller should save a "best" checkpoint),
        and sets self.should_stop when patience is exhausted.
        """
        is_better = (
            self.best is None
            or (self.mode == "max" and value > self.best + self.min_delta)
            or (self.mode == "min" and value < self.best - self.min_delta)
        )
        if is_better:
            self.best = value
            self.num_bad_epochs = 0
            return True
        self.num_bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.num_bad_epochs >= self.patience
