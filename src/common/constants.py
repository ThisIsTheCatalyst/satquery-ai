"""
Shared constants. Owner: shared, agreed once and not changed unilaterally.
"""

# Task labels the router/agent/evaluation all use.
TASK_VQA = "vqa"
TASK_CAPTIONING = "captioning"
TASK_GROUNDING = "grounding"
TASK_CHANGE = "change"
TASK_FUSION = "fusion"

ALL_TASKS = [TASK_VQA, TASK_CAPTIONING, TASK_GROUNDING, TASK_CHANGE, TASK_FUSION]

# Input modes, decided by input_validator from the images passed in.
MODE_SINGLE = "single"
MODE_CROSS_MODAL = "cross_modal"     # optical + SAR pair
MODE_BI_TEMPORAL = "bi_temporal"     # image at t1 + t2

MODE_TASK_MAP = {
    MODE_SINGLE: [TASK_VQA, TASK_CAPTIONING, TASK_GROUNDING],
    MODE_CROSS_MODAL: [TASK_FUSION],
    MODE_BI_TEMPORAL: [TASK_CHANGE],
}

# Backbone names — the actual models each workstream owns for this sprint.
# Keep these in sync with configs/config.yaml -> models.*.backbone
BACKBONE_VQA_CAPTIONING = "CLIP-ViT-B/32"           # zero-shot, no fine-tuning
BACKBONE_GROUNDING_PRIMARY = "CLIP-ViT-B/32"        # patch sliding-window localisation
BACKBONE_GROUNDING_FALLBACK = []                    # none — CLIP-only by design
BACKBONE_CHANGE = "siamese_difference"              # unsupervised classical baseline
BACKBONE_FUSION = "OpticalSAR-FeatureFusion"        # Person 4 (dual encoder + fusion head)

ALLOWED_IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

# ---------------------------------------------------------------------------
# Sensor names — used in ImageSample/PairSample/FusionSample.sensor and by
# src/preprocessing/sensor_registry.py to pick the right adapter. Development
# data is Sentinel-based; the hidden ISRO/SAC judging set is Cartosat-2S
# (optical) + RISAT (SAR) — a DIFFERENT sensor with a different radiometric
# range/band layout. Never assume Sentinel preprocessing applies to it.
# ---------------------------------------------------------------------------
SENSOR_SENTINEL2 = "Sentinel-2"     # optical, BigEarthNet / BigEarthNet-MM
SENSOR_SENTINEL1 = "Sentinel-1"     # SAR, BigEarthNet-MM
SENSOR_CARTOSAT2S = "Cartosat-2S"   # optical, ISRO/SAC hidden eval set
SENSOR_RISAT = "RISAT"              # SAR, ISRO/SAC hidden eval set
SENSOR_UNKNOWN = "unknown"          # benchmark PNG/JPEG with no sensor metadata

# ---------------------------------------------------------------------------
# Dataset roles — freeze this table. It converts "VRSBench is eval-only" from
# a norm someone can accidentally violate under deadline pressure into a
# machine-checkable guardrail: src/preprocessing/dataset_loader.py raises if
# a training DataLoader is requested for a dataset with usable_for_training
# = False. Do NOT let a teammate silently repurpose an eval-only dataset for
# training — if a stretch-goal fine-tune is genuinely needed (e.g. CDVQA),
# flip the flag deliberately and note it in docs/DECISIONS.md.
# ---------------------------------------------------------------------------
DATASET_ROLES = {
    "bigearthnet": {
        "purpose": "RS domain adaptation (image<->text) for GeoChat",
        "usable_for_training": True,
        "owner": "person_1",
    },
    "bigearthnet_mm": {
        "purpose": "Optical-SAR fusion development + training",
        "usable_for_training": True,
        "owner": "person_4",
    },
    "vrsbench": {
        "purpose": "VQA / captioning / grounding evaluation",
        "usable_for_training": False,
        "owner": "person_1_and_2",
    },
    "rsvqa": {
        "purpose": "VQA evaluation",
        "usable_for_training": False,
        "owner": "person_1",
    },
    "cdvqa": {
        "purpose": "Change-VQA evaluation (+ optional stretch fine-tune)",
        "usable_for_training": False,  # flip deliberately + log in DECISIONS.md if pursued
        "owner": "person_3",
    },
    "isro_sac": {
        "purpose": "Final hidden evaluation (Cartosat-2S optical + RISAT SAR)",
        "usable_for_training": False,  # never seen during development, by design
        "owner": None,
    },
}


def assert_usable_for_training(dataset_name: str) -> None:
    """Guardrail called from src/preprocessing/dataset_loader.get_dataloader
    when split="train". Raises loudly instead of silently letting someone
    train on an eval-only dataset under deadline pressure.
    """
    role = DATASET_ROLES.get(dataset_name)
    if role is None:
        raise ValueError(
            f"Unknown dataset '{dataset_name}' — not in DATASET_ROLES. "
            f"Add it to src/common/constants.py::DATASET_ROLES first."
        )
    if not role["usable_for_training"]:
        raise ValueError(
            f"Dataset '{dataset_name}' is eval-only ({role['purpose']}). "
            f"Requesting a training split for it is almost certainly a bug. "
            f"If a stretch-goal fine-tune genuinely needs this, flip "
            f"usable_for_training in DATASET_ROLES deliberately and record "
            f"why in docs/DECISIONS.md."
        )
