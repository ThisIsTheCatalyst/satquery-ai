#!/usr/bin/env python3
"""
Unified evaluation script — runs a trained model over a benchmark.

Usage:
    python scripts/evaluate.py --task vqa --dataset rsvqa \\
                                --config configs/config.yaml \\
                                --write-results

    python scripts/evaluate.py --task change --dataset cdvqa \\
                                --config configs/config.yaml
"""
import argparse
import os
import sys
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.evaluate import evaluate, _write_result_json


def main():
    parser = argparse.ArgumentParser(description="SatQuery AI evaluation")
    parser.add_argument("--task", required=True,
                        choices=["vqa", "captioning", "grounding", "change", "fusion"])
    parser.add_argument("--dataset", required=True,
                        help="e.g. vrsbench, rsvqa, cdvqa, bigearthnet_mm")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--write-results", action="store_true",
                        help="Write results JSON to outputs/eval_results/")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"\nEvaluating task={args.task} on dataset={args.dataset}")
    print(f"  config: {args.config}")
    print()

    try:
        results = evaluate(args.task, args.dataset, config)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        print("Make sure the dataset files exist at the paths in configs/config.yaml.")
        sys.exit(1)

    print(f"Results:")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        elif isinstance(v, list) and k == "per_class_f1":
            print(f"  {k}: [... {len(v)} classes ...]")
        else:
            print(f"  {k}: {v}")

    if args.write_results:
        model_cfg = config["models"].get(args.task, {})
        path = _write_result_json(
            args.task, args.dataset, config, results,
            model_name=model_cfg.get("backbone", args.task)
        )
        print(f"\nResults written to: {path}")


if __name__ == "__main__":
    main()
