#!/usr/bin/env python3
"""
Measure the siamese-difference change detector against OSCD ground truth.

The detector is unsupervised (per-band absolute difference + Otsu
thresholding), so there is no training step. This script just reports how
well that classical baseline actually does, per pair and overall.

For an unsupervised baseline on OSCD, an F1 in the 0.25-0.40 band is a
normal, defensible result. Report whatever comes out — a modest honest
number is worth more in a viva than an impressive unverifiable one.

Usage:
    python scripts/eval_change_oscd.py
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def prf1(pred: np.ndarray, gt: np.ndarray):
    pred = pred.astype(bool).ravel()
    gt = gt.astype(bool).ravel()
    tp = int(np.sum(pred & gt))
    fp = int(np.sum(pred & ~gt))
    fn = int(np.sum(~pred & gt))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    inter = tp
    union = tp + fp + fn
    iou = inter / union if union else 0.0
    return prec, rec, f1, iou, tp, fp, fn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/demo/oscd")
    ap.add_argument("--out", default="outputs/eval_results/change_oscd.json")
    ap.add_argument("--save-masks", action="store_true",
                    help="write predicted masks as PNGs for the slides")
    args = ap.parse_args()

    from src.data.oscd import load_demo_pairs, demo_pairs_exist
    from src.models.change_model import ChangeModel

    if not demo_pairs_exist(args.pairs):
        print(f"ERROR: no OSCD pairs at {args.pairs}. "
              f"Run: python scripts/prepare_data.py")
        return 1

    pairs = load_demo_pairs(args.pairs)
    model = ChangeModel(config={})
    model.load("none", device="cpu")
    print(f"Change model backbone: {model.backbone}")

    results, agg = [], {"tp": 0, "fp": 0, "fn": 0}

    for p in pairs:
        if p["gt"] is None:
            print(f"  {p['name']}: no ground truth, skipping scoring")
            continue

        t1 = p["t1"].transpose(2, 0, 1).astype(np.float32) / 255.0
        t2 = p["t2"].transpose(2, 0, 1).astype(np.float32) / 255.0

        res = model.predict(image_t1=t1, image_t2=t2,
                            query="What changed between these two dates?")
        mask = res.get("spatial_evidence", {}).get("mask")
        if mask is None:
            print(f"  {p['name']}: no mask returned")
            continue

        mask = np.asarray(mask)
        gt = p["gt"]
        if mask.shape != gt.shape:
            from PIL import Image as PILImage
            mask = np.array(
                PILImage.fromarray((mask * 255).astype(np.uint8))
                .resize((gt.shape[1], gt.shape[0]), PILImage.NEAREST)
            ) > 127

        prec, rec, f1, iou, tp, fp, fn = prf1(mask, gt)
        agg["tp"] += tp
        agg["fp"] += fp
        agg["fn"] += fn
        results.append({"name": p["name"], "precision": prec, "recall": rec,
                        "f1": f1, "iou": iou})
        print(f"  {p['name']:<16} P={prec:.3f} R={rec:.3f} "
              f"F1={f1:.3f} IoU={iou:.3f}")

        if args.save_masks:
            from PIL import Image as PILImage
            d = Path("outputs/evidence")
            d.mkdir(parents=True, exist_ok=True)
            PILImage.fromarray((np.asarray(mask) * 255).astype(np.uint8)).save(
                d / f"{p['name']}_pred.png")

    gp = agg["tp"] / (agg["tp"] + agg["fp"]) if (agg["tp"] + agg["fp"]) else 0.0
    gr = agg["tp"] / (agg["tp"] + agg["fn"]) if (agg["tp"] + agg["fn"]) else 0.0
    gf = 2 * gp * gr / (gp + gr) if (gp + gr) else 0.0

    print("\n" + "=" * 52)
    print(f"Aggregate (pixel-level, all pairs pooled)")
    print(f"  Precision : {gp:.3f}")
    print(f"  Recall    : {gr:.3f}")
    print(f"  F1        : {gf:.3f}")
    print("=" * 52)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "method": "siamese_absolute_difference + otsu",
        "supervised": False,
        "n_pairs_scored": len(results),
        "per_pair": results,
        "aggregate": {"precision": gp, "recall": gr, "f1": gf},
    }, indent=2))
    print(f"\nWritten: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
