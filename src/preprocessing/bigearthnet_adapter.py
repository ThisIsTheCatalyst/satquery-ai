"""
BigEarthNet.txt adapter — builds image-text pairs used for remote-sensing
domain adaptation of the base VLM (mandatory per problem statement).
Owner: Person 1 (Data & Preprocessing).

============================================================================
ASSUMED SCHEMA — READ BEFORE TRUSTING THIS FOR REAL TRAINING
============================================================================
Per the ML audit instructions ("do not invent dataset schemas; if uncertain,
isolate the adapter and document the assumption"), this parser assumes the
following format for BigEarthNet.txt, based on the standard BigEarthNet /
reBEN multi-label distribution convention (patch_id <TAB> comma-separated
CORINE labels):

    S2A_MSIL2A_20180526T101031_N9999_R022_T33UUP_00_00	Coniferous forest,Water bodies
    S2A_MSIL2A_20180526T101031_N9999_R022_T33UUP_00_01	Pastures

i.e. one line per patch: `<patch_id><TAB><label1>,<label2>,...`, and the
corresponding image is expected at `{image_dir}/{patch_id}.tif`.

THIS IS AN ASSUMPTION, NOT A CONFIRMED FACT — the team has not yet been
handed the actual BigEarthNet.txt file. `parse_bigearthnet_labels` therefore:
  1. tries the assumed tab-separated format first;
  2. falls back to comma-separated (`patch_id,label1,label2,...`) if no tab
     is found on the first non-empty line;
  3. raises a clear, actionable ValueError (not a silent misparse) if
     neither pattern matches, naming the exact line that broke.

Before real training: open the actual BigEarthNet.txt, confirm which of the
two delimiter conventions it uses (or note a third one), and update
docs/DECISIONS.md with what was found. Do not delete this docstring's
"ASSUMED SCHEMA" section until that confirmation happens — it's the seam
this whole module depends on.
"""

from typing import List, Dict
import random


def parse_bigearthnet_labels(labels_file: str) -> List[Dict]:
    """Parse BigEarthNet.txt into a list of {"image_id": str, "labels": [str,...]}.

    See module docstring for the assumed format and the tab/comma fallback
    logic. Blank lines are skipped. Whitespace around labels is stripped so
    "Water bodies, Pastures" and "Water bodies,Pastures" parse identically.
    """
    records: List[Dict] = []
    delimiter = None  # decided from the first non-empty line, then fixed

    with open(labels_file) as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            if delimiter is None:
                if "\t" in line:
                    delimiter = "\t"
                elif "," in line:
                    delimiter = ","
                else:
                    raise ValueError(
                        f"{labels_file}:{line_no}: cannot determine delimiter "
                        f"— line has neither a tab nor a comma: {line!r}. "
                        f"Update the ASSUMED SCHEMA in "
                        f"src/preprocessing/bigearthnet_adapter.py once the "
                        f"real format is confirmed."
                    )

            if delimiter == "\t":
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    raise ValueError(
                        f"{labels_file}:{line_no}: expected "
                        f"'<image_id>\\t<label1,label2,...>', got: {line!r}"
                    )
                image_id, label_str = parts
            else:
                parts = line.split(",")
                if len(parts) < 2:
                    raise ValueError(
                        f"{labels_file}:{line_no}: expected "
                        f"'<image_id>,<label1>,<label2>,...', got: {line!r}"
                    )
                image_id, label_str = parts[0], ",".join(parts[1:])

            labels = [lbl.strip() for lbl in label_str.split(",") if lbl.strip()]
            if not labels:
                raise ValueError(
                    f"{labels_file}:{line_no}: image_id {image_id!r} has no "
                    f"labels after parsing: {line!r}"
                )
            records.append({"image_id": image_id.strip(), "labels": labels})

    if not records:
        raise ValueError(f"{labels_file}: no records parsed — file empty?")
    return records


# Multiple phrasing templates so the adapted VLM doesn't overfit to one
# caption pattern (explicitly called out as a requirement in the original
# TODO). {labels} is replaced with a natural-language, Oxford-comma-joined
# list of the CORINE land-cover labels.
_CAPTION_TEMPLATES = [
    "This image shows {labels}.",
    "The satellite image contains {labels}.",
    "A remote sensing scene depicting {labels}.",
    "This patch is characterized by {labels}.",
    "Land cover in this image includes {labels}.",
    "Visible in this scene: {labels}.",
]


def _join_labels_naturally(labels: List[str]) -> str:
    labels = [lbl[0].lower() + lbl[1:] if lbl else lbl for lbl in labels]
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def labels_to_caption(labels: List[str], seed: int = None) -> str:
    """Convert a list of land-cover labels into a natural-language
    pseudo-caption for image-text pretraining, e.g.
    ["Coniferous forest", "Water bodies"] ->
    "This image shows coniferous forest and water bodies."

    `seed` makes template choice reproducible when passed (e.g. hash of the
    image_id) — pass None for genuinely random variety across an epoch.
    """
    if not labels:
        raise ValueError("labels_to_caption: empty labels list")
    rng = random.Random(seed)
    template = rng.choice(_CAPTION_TEMPLATES)
    return template.format(labels=_join_labels_naturally(labels))


def build_pretraining_pairs(labels_file: str, image_dir: str) -> List[Dict]:
    """Combine parsed labels + generated captions into pretraining samples:
    [{"image_path": str, "text": str, "labels": [str,...]}, ...]

    This is the dataset used to domain-adapt the base VLM before task-specific
    fine-tuning (VQA/captioning/grounding/change/fusion).

    Caption template choice is seeded off the image_id (via Python's hash)
    so re-running this function is reproducible without needing to persist
    which template was picked per sample.
    """
    records = parse_bigearthnet_labels(labels_file)
    pairs = []
    for r in records:
        seed = hash(r["image_id"]) & 0xFFFFFFFF
        pairs.append({
            "image_path": f"{image_dir}/{r['image_id']}.tif",
            "text": labels_to_caption(r["labels"], seed=seed),
            "labels": r["labels"],
        })
    return pairs
