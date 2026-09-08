#!/usr/bin/env python3
"""
Measure CLIP zero-shot land-cover accuracy on the EuroSAT demo subset.

This produces the single most important number in the project: proof that
the VQA/captioning path is genuinely image-conditioned rather than echoing
the question. Run it before building anything on top of CLIP.

Expected: top-1 accuracy well above the 10% chance level for 10 classes.
If it comes out near chance, the prompt template or the preprocessing is
wrong — fix that here, not later.

Usage:
    python scripts/eval_clip_eurosat.py
    python scripts/eval_clip_eurosat.py --device cuda
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="data/demo/eurosat")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="outputs/eval_results/clip_eurosat.json")
    args = ap.parse_args()

    from src.data.eurosat import load_demo_subset, demo_subset_exists
    from src.models.clip_backend import (
        get_shared_clip, RS_LABELS, EUROSAT_TO_LABEL, LABEL_TO_EUROSAT,
    )

    if not demo_subset_exists(args.subset):
        print(f"ERROR: no demo subset at {args.subset}. "
              f"Run: python scripts/prepare_data.py")
        return 1

    items = load_demo_subset(args.subset)
    print(f"Loaded {len(items)} images across "
          f"{len(set(c for _, c, _ in items))} classes")

    clip = get_shared_clip(device=args.device)
    if not clip.available:
        print("ERROR: CLIP failed to load. Check transformers/torch install.")
        return 1

    correct = 0
    per_class = defaultdict(lambda: {"n": 0, "correct": 0})
    confusion = defaultdict(lambda: defaultdict(int))

    for i, (arr, true_cls, fname) in enumerate(items):
        ranked = clip.classify(arr, RS_LABELS)
        pred_label = ranked[0][0]
        pred_cls = LABEL_TO_EUROSAT.get(pred_label, pred_label)

        hit = pred_cls == true_cls
        correct += int(hit)
        per_class[true_cls]["n"] += 1
        per_class[true_cls]["correct"] += int(hit)
        confusion[true_cls][pred_cls] += 1

        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(items)}  running acc="
                  f"{correct/(i+1):.1%}")

    acc = correct / max(len(items), 1)
    chance = 1.0 / len(RS_LABELS)

    print("\n" + "=" * 58)
    print(f"CLIP zero-shot top-1 accuracy : {acc:.1%}")
    print(f"Chance level (10 classes)     : {chance:.1%}")
    print(f"Improvement over chance       : {acc/chance:.1f}x")
    print("=" * 58)
    print(f"\n{'Class':<24}{'N':>5}{'Correct':>9}{'Acc':>8}")
    print("-" * 46)
    for cls in sorted(per_class):
        s = per_class[cls]
        print(f"{cls:<24}{s['n']:>5}{s['correct']:>9}"
              f"{s['correct']/max(s['n'],1):>8.0%}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": clip.model_id,
        "prompt_template": clip.prompt_template,
        "n_images": len(items),
        "top1_accuracy": acc,
        "chance_level": chance,
        "per_class": {k: dict(v) for k, v in per_class.items()},
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }, indent=2))
    print(f"\nWritten: {out}")

    if acc < 0.20:
        print("\nWARNING: accuracy is close to chance. Check the prompt "
              "template and image preprocessing before continuing.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
