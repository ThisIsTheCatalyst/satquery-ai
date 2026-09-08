"""
Unified dataset loading interface — contract every training script relies on.
Owner: Person 1 (Data & Preprocessing).

See docs/api_contracts.md section 1 for the exact batch schema.

============================================================================
ML AUDIT FINDING (fixed here): get_dataloader() previously always
instantiated one generic `RSDataset` class regardless of `dataset=`, and
that class's `_load_index`/`__getitem__` were both bare NotImplementedError
stubs. There was no actual mechanism to read BigEarthNet vs. VRSBench vs.
CDVQA differently even though configs/config.yaml, docs/DATASETS.md, and
every training script all assumed one existed. This is now a real factory
(`_DATASET_CLASSES`) dispatching to one subclass per benchmark.

ASSUMED ANNOTATION SCHEMAS: the team does not yet have the real VRSBench/
RSVQA/CDVQA/BigEarthNet-MM annotation files in-repo. Per the audit
instructions ("do not invent dataset schemas; isolate the adapter and
document the assumption"), each dataset class below assumes one specific,
clearly-documented JSON layout (the most common convention in the RS-VQA/
captioning/grounding literature) and fails loudly — FileNotFoundError or a
descriptive KeyError — rather than silently returning wrong data if the
real file doesn't match. CONFIRM against the real files before trusting
these for an actual training run; update the docstring of the relevant
class (not just a comment) once confirmed, since that docstring is the
single source of truth the next person reads.
============================================================================
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.preprocessing.geotiff_utils import read_image
from src.preprocessing.normalize import preprocess_pipeline
from src.common.constants import assert_usable_for_training
from src.common.schemas import ImageSample, PairSample, FusionSample


# ---------------------------------------------------------------------------
# Base class — shared index/getitem contract. Subclasses implement
# _load_index() (reading the dataset's own annotation format) and
# __getitem__() (reading + preprocessing images, returning model-ready kwargs
# plus a "label" for evaluation).
# ---------------------------------------------------------------------------

class RSDataset(Dataset):
    """Base dataset class. Subclass per benchmark (VRSBench/RSVQA/CDVQA/...)
    but always return the common batch schema documented in api_contracts.md.
    """

    def __init__(self, task: str, split: str, config: dict, dataset_name: str = None):
        self.task = task
        self.split = split
        self.config = config
        self.dataset_name = dataset_name
        self.samples = self._load_index()

    def _load_index(self) -> list:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _load_index()."
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raise NotImplementedError(
            f"{type(self).__name__} must implement __getitem__()."
        )

    # -- shared helpers used by every subclass -----------------------------

    def _preprocess(self, array: np.ndarray, modality: str = "optical") -> np.ndarray:
        return preprocess_pipeline(
            array, self.config, dataset=self.dataset_name, model=self.task,
            modality=modality,
        )

    def _annotation_path(self, filename: str) -> str:
        root = self.config["datasets"][self.dataset_name]["root"]
        path = os.path.join(root, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{type(self).__name__}: expected annotation file at "
                f"'{path}' (dataset root '{root}' + split '{self.split}'). "
                f"See this class's docstring for the assumed layout — "
                f"update it if the real dataset differs."
            )
        return path


# ---------------------------------------------------------------------------
# BigEarthNet — RS domain adaptation (image <-> text), Person 1 / GeoChat.
# Reuses src/preprocessing/bigearthnet_adapter.py, which owns the actual
# BigEarthNet.txt parsing + label->caption logic (see that module's
# "ASSUMED SCHEMA" docstring — the assumption lives there, not duplicated
# here).
# ---------------------------------------------------------------------------

class BigEarthNetDataset(RSDataset):
    """Image<->text pairs for GeoChat domain adaptation.

    __getitem__ returns:
        {"image": np.ndarray (C,H,W) preprocessed, "query": None,
         "label": str (the generated pseudo-caption, used as the LoRA
         target text during adaptation), "sensor": "Sentinel-2"}
    """

    def _load_index(self) -> list:
        from src.preprocessing.bigearthnet_adapter import build_pretraining_pairs
        ds_cfg = self.config["datasets"][self.dataset_name]
        pairs = build_pretraining_pairs(ds_cfg["labels_file"], ds_cfg["image_dir"])
        # No standard train/val/test split file is assumed to exist yet for
        # BigEarthNet.txt itself — deterministic 90/5/5 split by index so
        # re-running this doesn't reshuffle which patches land in val/test.
        n = len(pairs)
        if self.split == "train":
            return pairs[: int(0.9 * n)]
        elif self.split == "val":
            return pairs[int(0.9 * n): int(0.95 * n)]
        else:
            return pairs[int(0.95 * n):]

    def __getitem__(self, idx):
        rec = self.samples[idx]
        rsimage = read_image(rec["image_path"])
        array = self._preprocess(rsimage.array, modality="optical")
        return {
            "image": array,
            "query": None,
            "label": rec["text"],
            "sensor": "Sentinel-2",
        }


# ---------------------------------------------------------------------------
# BigEarthNet-MM — optical-SAR fusion development + training, Person 4.
#
# ASSUMED SCHEMA: a JSON file `{root}/{split}.json` — a list of records:
#   {"optical_path": "<path relative to root>",
#    "sar_path": "<path relative to root>",
#    "labels": ["Water bodies", "Pastures", ...]}   # multi-label, CORINE
# This mirrors the standard BigEarthNet-MM distribution convention (paired
# optical/SAR patches keyed by the same tile id, multi-label CORINE
# annotation) — confirm against the real archive's file layout before
# trusting this for training and update this docstring once confirmed.
# ---------------------------------------------------------------------------

class BigEarthNetMMDataset(RSDataset):
    """Co-registered optical+SAR pairs with multi-label land-cover targets.

    __getitem__ returns:
        {"image_optical": np.ndarray, "image_sar": np.ndarray, "query": None,
         "label": List[str] (multi-label land-cover classes)}
    """

    def _load_index(self) -> list:
        path = self._annotation_path(f"{self.split}.json")
        with open(path) as f:
            records = json.load(f)
        if not isinstance(records, list):
            raise ValueError(
                f"{path}: expected a JSON list of records, got {type(records)}."
            )
        return records

    def __getitem__(self, idx):
        rec = self.samples[idx]
        root = self.config["datasets"][self.dataset_name]["root"]
        optical_img = read_image(os.path.join(root, rec["optical_path"]))
        sar_img = read_image(os.path.join(root, rec["sar_path"]))
        optical_arr = self._preprocess(optical_img.array, modality="optical")
        sar_arr = self._preprocess(sar_img.array, modality="sar")
        return {
            "image_optical": optical_arr,
            "image_sar": sar_arr,
            "query": None,
            "label": rec["labels"],
        }


# ---------------------------------------------------------------------------
# VRSBench — single-image captioning/VQA/grounding EVALUATION ONLY.
#
# ASSUMED SCHEMA: `{root}/{task}_{split}.json` — a list of records, shape
# depends on `task`:
#   vqa:        {"image": "<relative path>", "question": str, "answer": str}
#   captioning: {"image": "<relative path>", "caption": str}
#   grounding:  {"image": "<relative path>", "query": str,
#                "bbox": [x1, y1, x2, y2]}   # pixel coords in the ORIGINAL
#                                            # (pre-resize) image
# This follows VRSBench's published task-separated JSON convention. Confirm
# the exact field names against the real download before trusting this.
# ---------------------------------------------------------------------------

class VRSBenchDataset(RSDataset):
    """Task-dependent single-image benchmark (vqa / captioning / grounding).

    __getitem__ returns task-appropriate kwargs + "label":
      vqa:        {"image":.., "query": question, "label": answer}
      captioning: {"image":.., "query": None,     "label": caption}
      grounding:  {"image":.., "query": query,    "label": bbox,
                   "orig_size": (H, W)}   # needed to rescale bbox after
                                          # preprocess_pipeline's resize —
                                          # see docs/api_contracts.md note
                                          # on grounding coordinate handling
    """

    def _load_index(self) -> list:
        path = self._annotation_path(f"{self.task}_{self.split}.json")
        with open(path) as f:
            records = json.load(f)
        if not isinstance(records, list):
            raise ValueError(f"{path}: expected a JSON list, got {type(records)}.")
        return records

    def __getitem__(self, idx):
        rec = self.samples[idx]
        root = self.config["datasets"][self.dataset_name]["root"]
        rsimage = read_image(os.path.join(root, rec["image"]))
        orig_h, orig_w = rsimage.array.shape[-2:]
        array = self._preprocess(rsimage.array, modality="optical")

        if self.task == "vqa":
            return {"image": array, "query": rec["question"], "label": rec["answer"]}
        if self.task == "captioning":
            return {"image": array, "query": None, "label": rec["caption"]}
        if self.task == "grounding":
            return {
                "image": array, "query": rec["query"], "label": rec["bbox"],
                "orig_size": (orig_h, orig_w),
            }
        raise ValueError(f"VRSBenchDataset does not support task '{self.task}'.")


# ---------------------------------------------------------------------------
# RSVQA — VQA EVALUATION ONLY, Person 1.
#
# ASSUMED SCHEMA: `{root}/{split}.json` — list of
#   {"image": "<relative path>", "question": str, "answer": str,
#    "type": str}   # e.g. "yes/no", "count", "land-cover" — RSVQA's own
#                    # question-type taxonomy, useful for per-type accuracy
# ---------------------------------------------------------------------------

class RSVQADataset(RSDataset):
    """VQA-only benchmark. __getitem__ returns
    {"image":.., "query": question, "label": answer, "question_type": type}.
    """

    def _load_index(self) -> list:
        path = self._annotation_path(f"{self.split}.json")
        with open(path) as f:
            records = json.load(f)
        if not isinstance(records, list):
            raise ValueError(f"{path}: expected a JSON list, got {type(records)}.")
        return records

    def __getitem__(self, idx):
        rec = self.samples[idx]
        root = self.config["datasets"][self.dataset_name]["root"]
        rsimage = read_image(os.path.join(root, rec["image"]))
        array = self._preprocess(rsimage.array, modality="optical")
        return {
            "image": array, "query": rec["question"], "label": rec["answer"],
            "question_type": rec.get("type", "unknown"),
        }


# ---------------------------------------------------------------------------
# CDVQA — Change-VQA EVALUATION ONLY (+ optional stretch fine-tune), Person 3.
#
# ASSUMED SCHEMA: `{root}/{split}.json` — list of
#   {"image_t1": "<relative path>", "image_t2": "<relative path>",
#    "question": str, "answer": str, "mask": "<relative path, optional>"}
# `mask`, if present, is a single-channel PNG/TIFF of the same (H, W) as
# image_t2 — absent for pure change-VQA-without-grounding samples.
# ---------------------------------------------------------------------------

class CDVQADataset(RSDataset):
    """Bi-temporal change-VQA benchmark. __getitem__ returns
    {"image_t1":.., "image_t2":.., "query": question, "label": answer,
     "gold_mask": np.ndarray | None}.
    """

    def _load_index(self) -> list:
        path = self._annotation_path(f"{self.split}.json")
        with open(path) as f:
            records = json.load(f)
        if not isinstance(records, list):
            raise ValueError(f"{path}: expected a JSON list, got {type(records)}.")
        return records

    def __getitem__(self, idx):
        rec = self.samples[idx]
        root = self.config["datasets"][self.dataset_name]["root"]
        img_t1 = read_image(os.path.join(root, rec["image_t1"]))
        img_t2 = read_image(os.path.join(root, rec["image_t2"]))
        arr_t1 = self._preprocess(img_t1.array, modality="optical")
        arr_t2 = self._preprocess(img_t2.array, modality="optical")

        gold_mask = None
        if rec.get("mask"):
            mask_img = read_image(os.path.join(root, rec["mask"]))
            gold_mask = mask_img.array[0]  # single-channel

        return {
            "image_t1": arr_t1, "image_t2": arr_t2, "query": rec["question"],
            "label": rec["answer"], "gold_mask": gold_mask,
        }


_DATASET_CLASSES = {
    "bigearthnet": BigEarthNetDataset,
    "bigearthnet_mm": BigEarthNetMMDataset,
    "vrsbench": VRSBenchDataset,
    "rsvqa": RSVQADataset,
    "cdvqa": CDVQADataset,
}


def _collate(batch: list) -> dict:
    """Custom collate: image arrays -> stacked float tensors (they're all
    the same size post-preprocess_pipeline resize); everything else
    (queries, string/list labels, optional masks) stays a plain Python list
    since VQA-style batches mix fixed-size tensors with variable-length
    text/labels — torch's default_collate chokes on that mix.
    """
    out = {}
    keys = batch[0].keys()
    for key in keys:
        values = [item[key] for item in batch]
        if isinstance(values[0], np.ndarray) and key.startswith("image"):
            out[key] = torch.from_numpy(np.stack(values)).float()
        else:
            out[key] = values
    return out


def get_dataloader(
    task: str,
    split: str,
    dataset: str = None,
    batch_size: int = 8,
    config: dict = None,
) -> DataLoader:
    """Single entry point every training/eval script should call.

    Example:
        loader = get_dataloader(task="vqa", split="train", dataset="rsvqa", config=cfg)

    Guardrail: requesting split="train" for a dataset marked eval-only in
    src.common.constants.DATASET_ROLES (e.g. VRSBench, RSVQA, CDVQA,
    ISRO/SAC) raises immediately instead of silently training on the wrong
    data — see constants.py::assert_usable_for_training for why this is
    enforced here rather than left as a documentation-only norm.

    `dataset` is required in practice (there is no meaningful "generic"
    dataset) — omitting it raises ValueError immediately rather than
    silently falling through to an undispatched base class, which is what
    happened before this audit fix.
    """
    if dataset is None:
        raise ValueError(
            "get_dataloader: `dataset` is required — pass one of "
            f"{list(_DATASET_CLASSES.keys())}. There is no generic fallback."
        )
    if split == "train":
        assert_usable_for_training(dataset)

    dataset_cls = _DATASET_CLASSES.get(dataset)
    if dataset_cls is None:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Known: {list(_DATASET_CLASSES.keys())}."
        )

    ds = dataset_cls(task=task, split=split, config=config, dataset_name=dataset)
    shuffle = split == "train"
    num_workers = 0 if os.environ.get("SATQUERY_SMOKE_TEST") else 4
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                       num_workers=num_workers, collate_fn=_collate)
