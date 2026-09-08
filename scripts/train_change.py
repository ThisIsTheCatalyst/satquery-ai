#!/usr/bin/env python3
"""
Train/evaluate the change detection model.

Usage:
    python scripts/train_change.py --config configs/colab_8gb.yaml

The default change model uses the siamese-difference baseline (no training
needed). This script exists to:
1. Evaluate the baseline on CDVQA-format data.
2. (Stretch goal) Fine-tune VisTA if available.
"""
import argparse
import os
import sys
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    parser = argparse.ArgumentParser(description="Change detection training/evaluation")
    parser.add_argument("--config", default="configs/colab_8gb.yaml")
    parser.add_argument("--eval-only", action="store_true", default=True,
                        help="Only evaluate (no training) — default for siamese-diff baseline")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if "_extends" in config:
        base_path = config.pop("_extends")
        with open(base_path) as f:
            base = yaml.safe_load(f)
        base.update(config)
        config = base

    from src.models.change_model import ChangeModel
    from src.training.trainer_utils import set_seed, get_device

    device = get_device(config["training"].get("device", "cuda"))
    set_seed(config["training"].get("seed", 42))
    
    model = ChangeModel(config=config)
    checkpoint = config["models"]["change"]["checkpoint"]
    model.load(checkpoint, device=device)
    
    print(f"\nChange Detection Model")
    print(f"  backbone: {model.backbone}")
    print(f"  using_fallback: {model.using_fallback}")
    print()
    
    if args.eval_only:
        print("Evaluation mode (siamese-difference baseline requires no training).")
        print("To evaluate on CDVQA, use:")
        print("  python -m src.evaluation.evaluate --task change --dataset cdvqa \\")
        print("         --config configs/config.yaml")
        
        # Run a quick synthetic evaluation
        import numpy as np
        from src.evaluation.metrics import change_f1
        
        print("\nRunning synthetic evaluation...")
        pred_masks = []
        gold_masks = []
        for i in range(10):
            rng = np.random.RandomState(i)
            t1 = rng.rand(4, 64, 64).astype("float32")
            t2 = t1.copy()
            # Add change in one region
            t2[:, 10:30, 10:30] += rng.rand(4, 20, 20) * 0.5
            t2 = np.clip(t2, 0, 1)
            
            result = model.predict(t1, t2, query="What changed?")
            mask = result["spatial_evidence"]["mask"]
            pred_masks.append(mask)
            
            gold = np.zeros((64, 64), dtype=np.uint8)
            gold[10:30, 10:30] = 1
            gold_masks.append(gold)
        
        metrics = change_f1(pred_masks, gold_masks)
        print(f"  Synthetic evaluation (10 samples):")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")
    else:
        print("VisTA fine-tuning not yet implemented (stretch goal).")
        print("See docs/DECISIONS.md for details.")
        sys.exit(0)

    print("\nDone.")


if __name__ == "__main__":
    main()
