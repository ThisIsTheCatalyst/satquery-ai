"""
Globs each person's src/evaluation/results/<person>.json and prints a
combined markdown table — the single artifact for "here's what we built"
in a judge conversation. Costs almost nothing once each notebook's last
cell is already writing its JSON (see docs/DATASETS.md for the schema).

Usage:
    python -m src.evaluation.generate_report
    python -m src.evaluation.generate_report --results-dir src/evaluation/results
"""

import argparse
import glob
import json
import os


def load_results(results_dir: str) -> list:
    records = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            try:
                records.append(json.load(f))
            except json.JSONDecodeError as e:
                print(f"WARNING: skipping unparsable {path}: {e}")
    return records


def to_markdown(records: list) -> str:
    if not records:
        return "_No evaluation results found yet — each person's notebook should " \
               "write src/evaluation/results/<person>.json in its last cell._"

    lines = ["| Model | Dataset | Metric | Score | n_samples |",
             "|---|---|---|---|---|"]
    for r in records:
        score = r.get("score")
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
        lines.append(
            f"| {r.get('model', '—')} | {r.get('dataset', '—')} | "
            f"{r.get('metric', '—')} | {score_str} | {r.get('n_samples', '—')} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="src/evaluation/results")
    args = parser.parse_args()

    records = load_results(args.results_dir)
    print(to_markdown(records))


if __name__ == "__main__":
    main()
