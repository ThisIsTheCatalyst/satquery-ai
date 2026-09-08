#!/usr/bin/env python3
"""
scripts/smoke_test.py — the single objective "is ML done" status board.

Tests all five specialist models with dummy inputs, validates the frozen
RSModelResult schema, and tests the controller end-to-end.

Usage:
    python scripts/smoke_test.py

    # With GPU device:
    python scripts/smoke_test.py --device cuda

    # Skip loading heavyweight models (fastest check):
    python scripts/smoke_test.py --no-load

Exit 0 if all checks pass, 1 if any fail.
"""

import sys
import os
import argparse
import traceback

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent.tool_registry import ToolRegistry
from src.agent.controller import AgentController
from src.agent.task_router import route_task
from src.agent.input_validator import validate_input
from src.common.constants import ALL_TASKS

REQUIRED_TOP_LEVEL_KEYS = {
    "task", "text", "confidence", "spatial_evidence", "metadata",
    "status", "inference_seconds",
}
REQUIRED_EVIDENCE_KEYS = {"type", "source", "bbox", "mask"}
REQUIRED_METADATA_KEYS = {
    "model", "backbone", "checkpoint", "dataset", "input_modalities", "parameters",
}


def _dummy_image(c=4, size=64):
    """Synthetic remote-sensing image — float32 (C,H,W) in [0,1]."""
    rng = np.random.RandomState(42)
    # Realistic: slightly heterogeneous values (not all zeros)
    arr = rng.uniform(0.1, 0.8, (c, size, size)).astype(np.float32)
    # Add a water-like dark patch in one quadrant
    arr[:, :size//4, :size//4] = 0.1
    return arr


def _validate_result(result: dict) -> list:
    """Returns a list of schema violations (empty = valid)."""
    problems = []
    missing = REQUIRED_TOP_LEVEL_KEYS - set(result.keys())
    if missing:
        problems.append(f"missing top-level keys: {missing}")
        return problems

    se = result.get("spatial_evidence") or {}
    if REQUIRED_EVIDENCE_KEYS - set(se.keys()):
        problems.append(f"spatial_evidence missing keys: {REQUIRED_EVIDENCE_KEYS - set(se.keys())}")

    md = result.get("metadata") or {}
    if REQUIRED_METADATA_KEYS - set(md.keys()):
        problems.append(f"metadata missing keys: {REQUIRED_METADATA_KEYS - set(md.keys())}")

    if result.get("status") not in ("success", "error", "low_confidence"):
        problems.append(f"invalid status: {result.get('status')!r}")

    conf = result.get("confidence")
    if conf is None or not (0.0 <= float(conf) <= 1.0):
        problems.append(f"confidence out of [0,1]: {conf}")

    if not result.get("text"):
        problems.append("text is empty or None")

    return problems


MODEL_CHECKS = [
    ("GeoChat VQA",        "vqa",
     dict(image=_dummy_image(), query="What land-cover types are visible in this satellite image?")),
    ("GeoChat Captioning", "captioning",
     dict(image=_dummy_image(), query="Describe the scene visible in this satellite image.")),
    ("Grounding",          "grounding",
     dict(image=_dummy_image(), query="Highlight the water body.")),
    ("Change Detection",   "change",
     dict(image_t1=_dummy_image(), image_t2=_dummy_image(),
          query="What changed between these two dates?")),
    ("Optical-SAR Fusion", "fusion",
     dict(image_optical=_dummy_image(), image_sar=_dummy_image(c=2),
          query="Identify built-up and water-covered regions.")),
]

UNIT_CHECKS = [
    ("Schema RSModelResult", _validate_result),  # filled in during main
    ("Task router: bi-temporal -> change",
     lambda: route_task("what changed?", "bi_temporal") == "change"),
    ("Task router: cross_modal -> fusion",
     lambda: route_task("combine them", "cross_modal") == "fusion"),
    ("Task router: grounding keyword",
     lambda: route_task("Highlight the water body.", "single") == "grounding"),
    ("Input validator: bad format",
     lambda: not validate_input({"image": "test.bmp"}, {})["valid"]),
    ("Input validator: single tif",
     lambda: validate_input({"image": "test.tif"}, {})["valid"]),
    ("Input validator: bi-temporal pair",
     lambda: validate_input({"image_t1": "t1.tif", "image_t2": "t2.tif"}, {})["valid"]),
    ("Normalize: per_band_minmax",
     lambda: __import__("src.preprocessing.normalize", fromlist=["normalize"]).normalize(
         _dummy_image(), "per_band_minmax").max() <= 1.0),
    ("Select bands",
     lambda: __import__("src.preprocessing.normalize", fromlist=["select_bands"]).select_bands(
         _dummy_image(6), [0, 1, 2]).shape[0] == 3),
]


def main():
    parser = argparse.ArgumentParser(description="SatQuery AI smoke test")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--no-load", action="store_true",
                        help="Skip loading heavyweight model weights (tests schema/routing only)")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.environ["SATQUERY_SMOKE_TEST"] = "1"  # disables multiprocessing in dataset_loader
    
    print()
    print("=" * 60)
    print("  SatQuery AI — Smoke Test")
    print("=" * 60)
    
    # Unit checks
    print("\n--- Unit Checks ---")
    unit_results = []
    
    # Schema check
    from src.common.schemas import RSModelResult, SpatialEvidence, ResultMetadata
    schema_result = RSModelResult(
        task="vqa", text="test", confidence=0.7,
        spatial_evidence=SpatialEvidence(type="bbox", source="image", bbox=[0, 0, 10, 10]),
        metadata=ResultMetadata(model="m", backbone="b", checkpoint="c", dataset="d",
                                input_modalities=["optical"]),
        status="success",
    ).to_dict()
    schema_ok = len(_validate_result(schema_result)) == 0
    unit_results.append(("Schema RSModelResult", "PASS" if schema_ok else "FAIL", ""))
    
    for name, fn in UNIT_CHECKS[1:]:  # skip first (schema, handled above)
        try:
            ok = fn()
            unit_results.append((name, "PASS" if ok else "FAIL", "" if ok else "returned False"))
        except Exception as e:
            unit_results.append((name, "FAIL", f"{type(e).__name__}: {e}"))
    
    for name, status, note in unit_results:
        prefix = "✓" if status == "PASS" else "✗"
        line = f"  {prefix} {name:<45}"
        if note:
            line += f"  ({note})"
        print(line)
    
    # Model checks
    print("\n--- Model Checks ---")
    if args.no_load:
        print("  (skipped — --no-load flag set)")
        model_results = [(label, "SKIP", "--no-load") for label, _, _ in MODEL_CHECKS]
    else:
        registry = ToolRegistry(config, device=args.device)
        model_results = []
        
        for label, task, kwargs in MODEL_CHECKS:
            try:
                tool = registry.get(task)
                out = tool.predict(**kwargs)
                problems = _validate_result(out)
                if problems:
                    model_results.append((label, "FAIL", "; ".join(problems)))
                else:
                    model_results.append((label, "PASS",
                                         f"status={out['status']} conf={out['confidence']:.2f}"))
            except NotImplementedError as e:
                model_results.append((label, "TODO", f"predict() not implemented: {e}"))
            except Exception as e:
                tb = traceback.format_exc().strip().split("\n")[-1]
                model_results.append((label, "FAIL", f"{type(e).__name__}: {e}"))
        
        for label, status, note in model_results:
            prefix = {"PASS": "✓", "FAIL": "✗", "TODO": "○", "SKIP": "-"}.get(status, "?")
            print(f"  {prefix} [{status:4s}] {label:<25}  {note}")
    
    # Controller end-to-end check
    print("\n--- Controller End-to-End ---")
    e2e_results = []
    if not args.no_load:
        controller = AgentController(config, device=args.device)
        
        e2e_cases = [
            ("VQA (single image)", 
             {"image": _dummy_image()},
             "What is the dominant land cover?"),
            ("Change (bi-temporal)",
             {"image_t1": _dummy_image(), "image_t2": _dummy_image()},
             "What changed between T1 and T2?"),
            ("Fusion (optical+SAR)",
             {"image_optical": _dummy_image(), "image_sar": _dummy_image(c=2)},
             "Identify land cover types using both images."),
        ]
        
        for label, images, query in e2e_cases:
            try:
                result = controller.run(images, query)
                ok = result.get("success") or result.get("status") != "error"
                task = result.get("task", "?")
                e2e_results.append((label, "PASS" if ok else "FAIL",
                                    f"task={task} status={result.get('status')}"))
            except Exception as e:
                e2e_results.append((label, "FAIL", f"{type(e).__name__}: {e}"))
        
        for label, status, note in e2e_results:
            prefix = "✓" if status == "PASS" else "✗"
            print(f"  {prefix} {label:<30}  {note}")
    else:
        print("  (skipped — --no-load flag set)")
        e2e_results = []
    
    # Summary
    print()
    print("=" * 60)
    all_results = unit_results + model_results + e2e_results
    n_pass = sum(1 for _, s, _ in all_results if s == "PASS")
    n_fail = sum(1 for _, s, _ in all_results if s == "FAIL")
    n_todo = sum(1 for _, s, _ in all_results if s == "TODO")
    n_skip = sum(1 for _, s, _ in all_results if s == "SKIP")
    
    print(f"  Results: {n_pass} PASS / {n_fail} FAIL / {n_todo} TODO / {n_skip} SKIP")
    print()
    
    if n_fail > 0:
        print("  ✗ Some checks FAILED — see above for details.")
        sys.exit(1)
    elif n_todo > 0:
        print("  ○ Some models TODO (predict() not yet implemented).")
        sys.exit(0)  # not a failure, just incomplete
    else:
        print("  ✓ All checks PASS — SatQuery AI backend functional.")
        sys.exit(0)


if __name__ == "__main__":
    main()
