"""
Optical-SAR fusion / joint information extraction model. MANDATORY per
problem statement (cross-modal pair analysis). Owner: Person 4.
Sprint plan: notebooks/04_optical_sar_fusion.ipynb.

Strategy (Path A — the practical choice for a 2-day deadline): feature-level
fusion, NOT a new large multimodal foundation model, and NOT open-ended
language generation from fused features (too ambitious with no pretrained
VLM backbone to lean on for this slot).

  - Separate pretrained encoders for optical and SAR (frozen by default —
    see `freeze_encoders` in configs/config.yaml), features concatenated,
    and a small trainable fusion head on top -> MULTI-LABEL LAND-COVER
    CLASSIFICATION output (built-up / water / vegetation / forest / ...,
    the standard 19-class BigEarthNet/CORINE nomenclature — see
    BIGEARTHNET_19_LABELS below), not free text.
  - `text` is a THIN TEMPLATED WRAPPER around the classification output.
  - A lightweight segmentation head for spatial masks is a stretch goal —
    not implemented this pass; spatial_evidence stays type="none" until
    that lands (see docs/ML_AUDIT.md).
  - Train on BigEarthNet-MM (paired Sentinel-2 optical + Sentinel-1 SAR).
  - MUST run and report a real THREE-WAY ABLATION: optical-only, SAR-only,
    optical+SAR fusion (src/evaluation/metrics.py::fusion_ablation). Report
    all three honestly, even if fusion loses to optical-only — never
    hardcode `assert fusion >= optical` (an earlier draft of
    notebooks/04_optical_sar_fusion.ipynb did exactly this; removed during
    the ML audit — see docs/ML_AUDIT.md and docs/DECISIONS.md).

ENCODER WEIGHTS — HONEST STATUS: both encoders attempt to load ImageNet-
pretrained torchvision weights. In an offline/sandboxed environment (no
route to download.pytorch.org), that download fails and the encoder falls
back to random initialization — this is caught explicitly, logged, and
recorded in ResultMetadata.parameters["encoder_weights"] as
"imagenet_pretrained" or "random_init_fallback" so nobody mistakes a
randomly-initialized encoder's output for a real result. A model trained
with random-init encoders is a legitimate SMOKE_TEST / architecture
validation, NOT a reportable benchmark number — see docs/ML_AUDIT.md.
"""

import numpy as np
import torch
import torch.nn as nn

from src.models.base_model import BaseRSModel
from src.common.schemas import RSModelResult, SpatialEvidence, ResultMetadata
from src.common.constants import BACKBONE_FUSION
from src.training.trainer_utils import get_git_commit  # noqa: F401 (kept importable for symmetry with other model files' provenance needs)


# Standard 19-class BigEarthNet / CORINE Land Cover nomenclature (the
# BEN-19 scheme used by reBEN and most BigEarthNet-MM benchmarks). Matches
# configs/config.yaml's `label_taxonomy: bigearthnet_corine`. If the real
# BigEarthNet-MM annotation file uses a different class count/order,
# update this list AND note the change in docs/DECISIONS.md — every
# trained checkpoint's class-probability vector is meaningless without a
# fixed, documented label order.
BIGEARTHNET_19_LABELS = [
    "Urban fabric", "Industrial or commercial units", "Arable land",
    "Permanent crops", "Pastures", "Complex cultivation patterns",
    "Agriculture with natural vegetation", "Agro-forestry areas",
    "Broad-leaved forest", "Coniferous forest", "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland, shrub", "Beaches, dunes, sands",
    "Inland wetlands", "Coastal wetlands", "Inland waters", "Marine waters",
]


def _build_resnet_encoder(in_channels: int, feature_dim: int = 512) -> nn.Module:
    """ResNet-18 backbone adapted to `in_channels` input bands, with the
    classification head stripped so it returns a `feature_dim`-length
    feature vector. Chosen over a heavier backbone specifically for the
    2-day-sprint constraint (small, CPU-trainable, well-understood).

    Attempts ImageNet-pretrained weights first; falls back to random init
    with a clearly logged warning if the download is unavailable (offline
    sandbox, firewalled network, etc.) — see module docstring's "ENCODER
    WEIGHTS" section. Returns (module, weights_status_string).
    """
    from torchvision.models import resnet18, ResNet18_Weights

    weights_status = "random_init_fallback"
    try:
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        weights_status = "imagenet_pretrained"
    except Exception as e:  # noqa: BLE001 — genuinely any download/network error is fine to fall back on
        backbone = resnet18(weights=None)
        print(
            f"[fusion_model] WARNING: could not download ImageNet-pretrained "
            f"ResNet-18 weights ({type(e).__name__}: {e}). Falling back to "
            f"random initialization. This is fine for a smoke test / "
            f"architecture check, but a model trained on top of a randomly "
            f"initialized encoder is NOT a valid benchmark result — see "
            f"docs/ML_AUDIT.md."
        )

    if in_channels != 3:
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(
            in_channels, old_conv.out_channels, kernel_size=old_conv.kernel_size,
            stride=old_conv.stride, padding=old_conv.padding, bias=False,
        )
        if weights_status == "imagenet_pretrained":
            # Average the pretrained 3-channel filters across channels, then
            # repeat to the new channel count — a standard, well-behaved way
            # to adapt an RGB-pretrained conv1 to N-band input (better than
            # random init even though the extra bands start "uninformed").
            with torch.no_grad():
                mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
                new_conv.weight.copy_(mean_weight.repeat(1, in_channels, 1, 1))
        backbone.conv1 = new_conv

    backbone.fc = nn.Identity()  # -> (N, 512) feature vector, no classification head
    return backbone, weights_status


class FusionHead(nn.Module):
    """The ONLY part trained from scratch this sprint (per team decision —
    see docs/DECISIONS.md). Concatenates optical+SAR features and decodes
    into multi-label land-cover logits. Small and CPU-trainable by design.
    """

    def __init__(self, optical_dim: int, sar_dim: int, num_classes: int,
                 hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(optical_dim + sar_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        # Zero-tensors matching each branch's feature dim, used by
        # predict_optical_only/predict_sar_only to zero out the missing
        # modality rather than needing a structurally different head.
        self.optical_dim = optical_dim
        self.sar_dim = sar_dim

    def forward(self, optical_features: torch.Tensor, sar_features: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([optical_features, sar_features], dim=-1)
        return self.net(fused)  # logits, shape (N, num_classes)


class FusionModel(BaseRSModel):
    name = "fusion_v1"
    task = "fusion"
    backbone = BACKBONE_FUSION  # "OpticalSAR-FeatureFusion"

    def __init__(self, config: dict = None):
        self.config = config
        self.optical_encoder = None
        self.sar_encoder = None
        self.fusion_head = None
        self.device = "cpu"
        self.checkpoint_path = None
        self.labels = BIGEARTHNET_19_LABELS
        self.encoder_weights_status = {"optical": None, "sar": None}
        self.text_threshold = 0.5  # probability cutoff for "significant" in the template

    def load(self, checkpoint_path: str, device: str = "cpu") -> None:
        """Build both encoders (frozen by default) + fusion head, then load
        a trained fusion-head checkpoint if one exists at checkpoint_path.
        A missing checkpoint is NOT an error here — it means "freshly
        initialized, not yet trained," which is a valid state for a smoke
        test (health_check) but not for real predictions; predict() reports
        status accordingly via ResultMetadata rather than silently
        pretending an untrained head is a working model.
        """
        self.device = device
        self.checkpoint_path = checkpoint_path
        fusion_cfg = self.config["models"]["fusion"] if self.config else {}
        ds_cfg = (self.config["datasets"].get("bigearthnet_mm", {}) if self.config else {})

        n_optical_bands = len(ds_cfg.get("optical_bands", [2, 3, 4, 8]))
        n_sar_bands = len(ds_cfg.get("sar_bands", [1, 2]))

        self.optical_encoder, self.encoder_weights_status["optical"] = \
            _build_resnet_encoder(n_optical_bands)
        self.sar_encoder, self.encoder_weights_status["sar"] = \
            _build_resnet_encoder(n_sar_bands)

        # freeze_encoders defaults True (per team decision — only the fusion
        # head is trained this sprint); set False in config to allow light
        # encoder fine-tuning later without touching this code.
        freeze_encoders = fusion_cfg.get("freeze_encoders", True)
        for p in self.optical_encoder.parameters():
            p.requires_grad = not freeze_encoders
        for p in self.sar_encoder.parameters():
            p.requires_grad = not freeze_encoders

        num_classes = len(self.labels)
        self.fusion_head = FusionHead(optical_dim=512, sar_dim=512, num_classes=num_classes)

        self._checkpoint_loaded = False
        try:
            state = torch.load(checkpoint_path, map_location=device, weights_only=False)
            self.fusion_head.load_state_dict(state["model_state_dict"])
            self._checkpoint_loaded = True
        except FileNotFoundError:
            print(
                f"[fusion_model] No trained fusion-head checkpoint at "
                f"'{checkpoint_path}' — fusion_head is freshly initialized "
                f"(untrained). predict() will still run but its output is "
                f"not meaningful until src/training/train_fusion.py has "
                f"produced a real checkpoint there."
            )

        self.optical_encoder.to(device).eval()
        self.sar_encoder.to(device).eval()
        self.fusion_head.to(device)
        self.fusion_head.eval()  # predict-only by default; train_step() flips to .train()

    def torch_module(self) -> nn.Module:
        """Bundle everything that could receive gradients. Even when
        encoders are frozen (requires_grad=False), including them here is
        harmless — build_optimizer filters to requires_grad params anyway —
        and this way flipping freeze_encoders=False in config to allow
        light encoder fine-tuning "just works" without touching this method.
        """
        return nn.ModuleDict({
            "optical_encoder": self.optical_encoder,
            "sar_encoder": self.sar_encoder,
            "fusion_head": self.fusion_head,
        })

    # -- inference -----------------------------------------------------

    def _to_tensor(self, array) -> torch.Tensor:
        if isinstance(array, np.ndarray):
            array = torch.from_numpy(array).float()
        if array.dim() == 3:
            array = array.unsqueeze(0)  # add batch dim
        return array.to(self.device)

    def _labels_to_text(self, probs: np.ndarray) -> str:
        """Template-fill the top label(s) into a sentence — see module
        docstring for why this is a template, not open-ended generation.
        """
        above = [(self.labels[i], p) for i, p in enumerate(probs) if p >= self.text_threshold]
        above.sort(key=lambda kv: -kv[1])
        if not above:
            top_idx = int(np.argmax(probs))
            return f"The area shows some indication of {self.labels[top_idx].lower()}, though confidence is low."
        if len(above) == 1:
            return f"The area shows significant {above[0][0].lower()}."
        names = [name.lower() for name, _ in above[:3]]
        if len(names) == 2:
            joined = f"{names[0]} and {names[1]}"
        else:
            joined = ", ".join(names[:-1]) + f", and {names[-1]}"
        return f"The area shows significant {joined}."

    def _forward(self, optical_arr, sar_arr) -> np.ndarray:
        with torch.no_grad():
            optical_t = self._to_tensor(optical_arr)
            sar_t = self._to_tensor(sar_arr)
            optical_feat = self.optical_encoder(optical_t)
            sar_feat = self.sar_encoder(sar_t)
            logits = self.fusion_head(optical_feat, sar_feat)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        return probs

    def _result_from_probs(self, probs: np.ndarray, modalities: list,
                            elapsed: float) -> dict:
        status = "success" if getattr(self, "_checkpoint_loaded", False) else "low_confidence"
        confidence = float(np.max(probs))
        return RSModelResult(
            task="fusion",
            text=self._labels_to_text(probs),
            confidence=confidence,
            spatial_evidence=SpatialEvidence(),  # segmentation head is a stretch goal, not built this pass
            metadata=ResultMetadata(
                model=self.name, backbone=self.backbone,
                checkpoint=self.checkpoint_path or "",
                dataset="BigEarthNet-MM",
                input_modalities=modalities,
                parameters={
                    "class_probabilities": {lbl: float(p) for lbl, p in zip(self.labels, probs)},
                    "encoder_weights": self.encoder_weights_status,
                    "checkpoint_loaded": getattr(self, "_checkpoint_loaded", False),
                },
            ),
            status=status,
            inference_seconds=elapsed,
        ).to_dict()

    def predict(self, image_optical, image_sar, query: str, **kwargs) -> dict:
        import time
        if self.fusion_head is None:
            return self._empty_result(
                text="Fusion model not loaded — call load() first.", status="error")
        start = time.time()
        probs = self._forward(image_optical, image_sar)
        return self._result_from_probs(probs, ["optical", "sar"], time.time() - start)

    def predict_optical_only(self, image_optical, query: str, **kwargs) -> dict:
        """Baseline leg of the mandatory 3-way ablation: zero the SAR
        branch's features (not the raw input) so the SAME fusion head
        weights are used for all three legs — an apples-to-apples
        comparison of what the trained head does with/without each
        modality, rather than three separately-trained heads.
        """
        import time
        if self.fusion_head is None:
            return self._empty_result(
                text="Fusion model not loaded — call load() first.", status="error")
        start = time.time()
        with torch.no_grad():
            optical_t = self._to_tensor(image_optical)
            optical_feat = self.optical_encoder(optical_t)
            sar_feat = torch.zeros(optical_feat.shape[0], self.fusion_head.sar_dim,
                                    device=self.device)
            logits = self.fusion_head(optical_feat, sar_feat)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        return self._result_from_probs(probs, ["optical"], time.time() - start)

    def predict_sar_only(self, image_sar, query: str, **kwargs) -> dict:
        """SAR-only leg of the ablation — see predict_optical_only for the
        zero-features rationale (same fusion head, optical branch zeroed)."""
        import time
        if self.fusion_head is None:
            return self._empty_result(
                text="Fusion model not loaded — call load() first.", status="error")
        start = time.time()
        with torch.no_grad():
            sar_t = self._to_tensor(image_sar)
            sar_feat = self.sar_encoder(sar_t)
            optical_feat = torch.zeros(sar_feat.shape[0], self.fusion_head.optical_dim,
                                        device=self.device)
            logits = self.fusion_head(optical_feat, sar_feat)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        return self._result_from_probs(probs, ["sar"], time.time() - start)

    def health_check(self) -> dict:
        base = super().health_check()
        base["checkpoint_loaded"] = getattr(self, "_checkpoint_loaded", False)
        base["encoder_weights"] = self.encoder_weights_status
        return base


def train_step(model: FusionModel, batch: dict, optimizer, config: dict) -> float:
    """One training step for the fusion head (encoders frozen by default —
    see load()'s freeze_encoders handling). Multi-label BCE loss over the
    19-class BigEarthNet/CORINE taxonomy.

    batch: as produced by BigEarthNetMMDataset + the dataset_loader collate
    fn — {"image_optical": Tensor(N,C,H,W), "image_sar": Tensor(N,C,H,W),
    "label": List[List[str]]}. Converts the string-label lists to a
    multi-hot target tensor here (kept out of the dataset class so the
    dataset stays label-representation-agnostic).
    """
    model.fusion_head.train()
    optimizer.zero_grad()

    optical = batch["image_optical"].to(model.device)
    sar = batch["image_sar"].to(model.device)
    target = torch.zeros(len(batch["label"]), len(model.labels), device=model.device)
    for i, label_list in enumerate(batch["label"]):
        for lbl in label_list:
            if lbl in model.labels:
                target[i, model.labels.index(lbl)] = 1.0

    optical_feat = model.optical_encoder(optical)
    sar_feat = model.sar_encoder(sar)
    logits = model.fusion_head(optical_feat, sar_feat)

    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(logits, target)
    loss.backward()
    optimizer.step()
    model.fusion_head.eval()
    return float(loss.item())
