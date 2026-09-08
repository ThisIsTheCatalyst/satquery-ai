"""
Evaluation harness — runs a trained model over a benchmark test split and
reports normalized metrics. Used for VRSBench/RSVQA/CDVQA and, at final
judging time, the ISRO/SAC evaluation set.
Owner: Person 4 (Agentic Orchestration + GUI / Evaluation).

Usage:
    python -m src.evaluation.evaluate --task vqa --dataset rsvqa --config configs/config.yaml

STATUS (ML audit pass): two real bugs fixed here:
  1. TASK_METRIC_FN had no "fusion" entry even though --task fusion is a
     valid argparse choice — running it raised ValueError immediately. Now
     routes to metrics.multilabel_f1 (fusion's `label` is a multi-hot
     vector, not a single string, so it needs its own collection path —
     see `_collect_predictions` below).
  2. The eval loop body was literally `pass` — no predictions were ever
     collected, so `preds`/`golds` were always empty and every metric
     silently returned its degenerate "no data" value (e.g. accuracy=0.0),
     which is indistinguishable from "the model got everything wrong." Now
     actually calls tool.predict(**kwargs) per sample and collects results
     against the sample's `label`.
"""

import argparse
import json
import os

import numpy as np
import yaml

from src.preprocessing.dataset_loader import get_dataloader
from src.agent.tool_registry import ToolRegistry
from src.evaluation import metrics as M


def _predict_kwargs_for_task(task: str, batch: dict, i: int) -> dict:
    """Pull the i-th sample's model kwargs out of a collated batch dict.
    Mirrors AgentController._build_predict_kwargs's per-task shape but
    reads from a DataLoader batch instead of a request dict.
    """
    if task in ("vqa", "captioning", "grounding"):
        return {"image": batch["image"][i], "query": batch["query"][i]}
    if task == "change":
        return {"image_t1": batch["image_t1"][i], "image_t2": batch["image_t2"][i],
                "query": batch["query"][i]}
    if task == "fusion":
        return {"image_optical": batch["image_optical"][i],
                "image_sar": batch["image_sar"][i], "query": batch["query"][i]}
    raise ValueError(f"Unknown task '{task}'")


def _collect_predictions(task: str, tool, loader) -> dict:
    """Run `tool.predict()` over every sample in `loader` and collect
    whatever each task's metric function needs. Returns a dict whose shape
    depends on `task` (see the branches below) rather than a single
    preds/golds pair, since grounding needs boxes, change needs masks, and
    fusion needs class-probability vectors — none of which fit the
    "list of strings" shape vqa/captioning use.
    """
    if task in ("vqa", "captioning"):
        preds, golds = [], []
        for batch in loader:
            for i in range(len(batch["query"])):
                kwargs = _predict_kwargs_for_task(task, batch, i)
                result = tool.predict(**kwargs)
                preds.append(result["text"])
                golds.append(batch["label"][i])
        return {"preds": preds, "golds": golds}

    if task == "grounding":
        pred_boxes, gold_boxes = [], []
        for batch in loader:
            for i in range(len(batch["query"])):
                kwargs = _predict_kwargs_for_task(task, batch, i)
                result = tool.predict(**kwargs)
                bbox = result["spatial_evidence"]["bbox"]
                if bbox is not None:
                    pred_boxes.append(bbox)
                    gold_boxes.append(batch["label"][i])
        return {"pred_boxes": pred_boxes, "gold_boxes": gold_boxes}

    if task == "change":
        preds, golds, pred_masks, gold_masks = [], [], [], []
        for batch in loader:
            for i in range(len(batch["query"])):
                kwargs = _predict_kwargs_for_task(task, batch, i)
                result = tool.predict(**kwargs)
                preds.append(result["text"])
                golds.append(batch["label"][i])
                pred_masks.append(result["spatial_evidence"]["mask"])
                gold_masks.append(batch["gold_mask"][i])
        return {"preds": preds, "golds": golds,
                "pred_masks": pred_masks, "gold_masks": gold_masks}

    if task == "fusion":
        pred_probs, gold_labels = [], []
        for batch in loader:
            for i in range(len(batch["query"])):
                kwargs = _predict_kwargs_for_task(task, batch, i)
                result = tool.predict(**kwargs)
                probs = result["metadata"]["parameters"].get("class_probabilities")
                if probs is None:
                    raise KeyError(
                        "fusion result metadata.parameters missing "
                        "'class_probabilities' — FusionModel.predict() must "
                        "populate this for evaluation (see fusion_model.py)."
                    )
                pred_probs.append(probs)
                gold_labels.append(batch["label"][i])
        return {"pred_probs": np.array(pred_probs), "gold_labels": np.array(gold_labels)}

    raise ValueError(f"No prediction-collection logic for task '{task}'")


def _compute_metrics(task: str, collected: dict) -> dict:
    if task == "vqa":
        return {"accuracy": M.vqa_accuracy(collected["preds"], collected["golds"])}
    if task == "captioning":
        return M.caption_scores(collected["preds"], collected["golds"])
    if task == "grounding":
        return {
            "iou": M.grounding_iou(collected["pred_boxes"], collected["gold_boxes"]),
            "accuracy_at_0.5": M.grounding_accuracy_at_threshold(
                collected["pred_boxes"], collected["gold_boxes"], threshold=0.5),
        }
    if task == "change":
        result = {"accuracy": M.vqa_accuracy(collected["preds"], collected["golds"])}
        result.update(M.change_f1(collected["pred_masks"], collected["gold_masks"]))
        return result
    if task == "fusion":
        return M.multilabel_f1(collected["pred_probs"], collected["gold_labels"])
    raise ValueError(f"No metric function for task '{task}'")


def evaluate(task: str, dataset: str, config: dict) -> dict:
    loader = get_dataloader(task=task, split="test", dataset=dataset, config=config,
                             batch_size=config["training"]["batch_size"])
    registry = ToolRegistry(config)
    tool = registry.get(task)

    collected = _collect_predictions(task, tool, loader)
    metrics = _compute_metrics(task, collected)
    metrics["n_samples"] = len(loader.dataset)
    return metrics


def _write_result_json(task: str, dataset: str, config: dict, metrics: dict,
                        model_name: str) -> str:
    """Write the per-person results JSON that generate_report.py globs
    (schema in docs/DATASETS.md). Picks the first scalar float metric as
    `score` for the summary table; the full dict is still there for anyone
    who wants more than one number.
    """
    primary_metric = next(
        (k for k, v in metrics.items() if isinstance(v, (int, float)) and k != "n_samples"),
        None,
    )
    out_dir = config["paths"]["eval_results"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{task}_{dataset}.json")
    with open(out_path, "w") as f:
        json.dump({
            "model": model_name,
            "dataset": dataset,
            "metric": primary_metric or "n/a",
            "score": metrics.get(primary_metric) if primary_metric else None,
            "n_samples": metrics.get("n_samples", 0),
            "all_metrics": metrics,
        }, f, indent=2)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True,
                         choices=["vqa", "captioning", "grounding", "change", "fusion"])
    parser.add_argument("--dataset", required=True,
                         help="e.g. vrsbench, rsvqa, cdvqa, isro_sac, bigearthnet_mm")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--write-results", action="store_true",
                         help="also write src/evaluation/results/<task>_<dataset>.json")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    results = evaluate(args.task, args.dataset, config)
    print(f"Results for task={args.task} dataset={args.dataset}:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, (int, float)) else f"  {k}: {v}")

    if args.write_results:
        model_cfg = config["models"].get(args.task, {})
        path = _write_result_json(args.task, args.dataset, config, results,
                                   model_name=model_cfg.get("backbone", args.task))
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
