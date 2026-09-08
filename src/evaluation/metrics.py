"""
Task-specific metrics. Owner: Person 4 (Agentic Orchestration + GUI / Evaluation).

Scores should be normalized to [0, 1] before combination, per the problem
statement's "Scores will be normalised before combining different metrics."

STATUS (ML audit pass): caption_scores, change_f1, and fusion_ablation are
now real implementations (previously NotImplementedError stubs, or in
fusion_ablation's case, documented in docs/api_contracts.md §6 but entirely
absent from this file — evaluate.py's TASK_METRIC_FN dict had no "fusion"
entry at all, so `evaluate.py --task fusion` raised immediately). BLEU/ROUGE
are implemented from scratch (no nltk/rouge-score dependency) specifically
to avoid nltk's punkt-tokenizer download requirement, which needs network
access the team may not reliably have mid-sprint — see the functions'
docstrings for exactly what's computed and how it differs from the
nltk/rouge-score/pycocoevalcap versions named in requirements.txt.
"""

import re
from collections import Counter
from typing import List, Optional
import numpy as np


# ---------------------------------------------------------------------------
# VQA / Change-VQA
# ---------------------------------------------------------------------------

_ARTICLES = {"a", "an", "the"}
_YES_SYNONYMS = {"yes", "yeah", "yep", "true", "correct"}
_NO_SYNONYMS = {"no", "nope", "false", "incorrect"}


def _normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation, drop articles, collapse whitespace, and
    canonicalize yes/no synonyms. This is the "soft match" the original
    TODO asked for — a generative VLM saying "Yes, there is water visible."
    should count as correct against gold "yes", not fail exact-match.
    """
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = [t for t in text.split() if t not in _ARTICLES]
    text = " ".join(tokens)
    if text in _YES_SYNONYMS:
        return "yes"
    if text in _NO_SYNONYMS:
        return "no"
    return text


def vqa_accuracy(preds: List[str], golds: List[str]) -> float:
    """Accuracy after normalization (lowercase, strip punctuation/articles,
    yes/no synonym canonicalization) — see _normalize_answer. This is
    "soft" exact-match, not a fuzzy/embedding similarity metric; it fixes
    the specific false negatives most common in RS-VQA (article/punctuation
    noise, yes/yeah/true style variation) without over-crediting genuinely
    wrong answers the way a similarity threshold can.
    """
    if not preds:
        return 0.0
    if len(preds) != len(golds):
        raise ValueError(f"preds/golds length mismatch: {len(preds)} vs {len(golds)}")
    correct = sum(
        1 for p, g in zip(preds, golds)
        if _normalize_answer(p) == _normalize_answer(g)
    )
    return correct / len(preds)


def vqa_accuracy_by_type(preds: List[str], golds: List[str],
                          question_types: List[str]) -> dict:
    """Per-question-type accuracy (e.g. "yes/no", "count", "land-cover" for
    RSVQA) — an aggregate accuracy hides failures concentrated in one
    question type, which is exactly the kind of gap the ML audit asked
    evaluation to surface rather than hide behind one number.
    """
    by_type = {}
    for p, g, t in zip(preds, golds, question_types):
        by_type.setdefault(t, {"preds": [], "golds": []})
        by_type[t]["preds"].append(p)
        by_type[t]["golds"].append(g)
    return {t: vqa_accuracy(v["preds"], v["golds"]) for t, v in by_type.items()}


# ---------------------------------------------------------------------------
# Captioning — BLEU-4 and ROUGE-L implemented from scratch (see module
# docstring for why: avoids nltk's punkt download requirement mid-sprint).
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return re.sub(r"[^\w\s]", "", text.lower()).split()


def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _sentence_bleu4(pred_tokens: List[str], gold_tokens: List[str]) -> float:
    """Single-reference BLEU-4 with the standard brevity penalty. Returns 0
    if the prediction is shorter than 4 tokens (n=4 has no valid n-grams) —
    matches standard BLEU convention rather than raising.
    """
    if len(pred_tokens) == 0:
        return 0.0
    precisions = []
    for n in range(1, 5):
        pred_ngrams = _ngrams(pred_tokens, n)
        gold_ngrams = _ngrams(gold_tokens, n)
        if not pred_ngrams:
            precisions.append(0.0)
            continue
        overlap = sum((pred_ngrams & gold_ngrams).values())
        total = sum(pred_ngrams.values())
        precisions.append(overlap / total if total > 0 else 0.0)

    if min(precisions) == 0.0:
        return 0.0
    geo_mean = np.exp(np.mean(np.log(precisions)))

    bp = 1.0 if len(pred_tokens) > len(gold_tokens) else \
        np.exp(1 - len(gold_tokens) / max(1, len(pred_tokens)))
    return float(bp * geo_mean)


def _lcs_length(a: List[str], b: List[str]) -> int:
    """Standard O(len(a)*len(b)) longest-common-subsequence DP, used by
    ROUGE-L. Fine for caption-length sequences (tens of tokens).
    """
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def _rouge_l(pred_tokens: List[str], gold_tokens: List[str]) -> float:
    """ROUGE-L F1 (harmonic mean of LCS-based precision and recall)."""
    if not pred_tokens or not gold_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, gold_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def caption_scores(preds: List[str], golds: List[str]) -> dict:
    """BLEU-4 and ROUGE-L for captioning, averaged over the batch.

    Implemented from scratch (single-reference) rather than via
    nltk/pycocoevalcap — see module docstring. This means scores are NOT
    directly comparable to published VRSBench leaderboard numbers that use
    multi-reference COCO-style evaluation; report that caveat alongside any
    number from this function rather than presenting it as the official
    benchmark score. CIDEr is intentionally not implemented here (needs a
    corpus-level IDF computed via pycocoevalcap) — leave as a documented gap
    rather than a fake approximation.
    """
    if not preds:
        return {"bleu4": 0.0, "rouge_l": 0.0, "n_samples": 0}
    if len(preds) != len(golds):
        raise ValueError(f"preds/golds length mismatch: {len(preds)} vs {len(golds)}")

    bleu_scores, rouge_scores = [], []
    for p, g in zip(preds, golds):
        p_tok, g_tok = _tokenize(p), _tokenize(g)
        bleu_scores.append(_sentence_bleu4(p_tok, g_tok))
        rouge_scores.append(_rouge_l(p_tok, g_tok))

    return {
        "bleu4": float(np.mean(bleu_scores)),
        "rouge_l": float(np.mean(rouge_scores)),
        "n_samples": len(preds),
    }


# ---------------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------------

def grounding_iou(pred_boxes: List[list], gold_boxes: List[list]) -> float:
    """Mean IoU between predicted and ground-truth bounding boxes.
    Each box: [x1, y1, x2, y2].
    """
    if not pred_boxes:
        return 0.0
    ious = [_iou(p, g) for p, g in zip(pred_boxes, gold_boxes)]
    return float(np.mean(ious))


def grounding_accuracy_at_threshold(pred_boxes: List[list], gold_boxes: List[list],
                                     threshold: float = 0.5) -> float:
    """Fraction of predictions with IoU >= threshold — the "thresholded
    accuracy" metric named in docs/api_contracts.md's grounding row,
    distinct from (and often more interpretable to a judge than) mean IoU.
    """
    if not pred_boxes:
        return 0.0
    ious = [_iou(p, g) for p, g in zip(pred_boxes, gold_boxes)]
    return float(np.mean([1.0 if iou >= threshold else 0.0 for iou in ious]))


def _iou(box_a: list, box_b: list) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Change
# ---------------------------------------------------------------------------

def change_f1(pred_masks: List[Optional[np.ndarray]],
              gold_masks: List[Optional[np.ndarray]]) -> dict:
    """Pixel-wise precision / recall / F1 / Cohen's kappa for binary change
    masks, aggregated over all provided (non-None) mask pairs.

    Falls back gracefully if masks are None (i.e. only change-VQA/
    description was produced, no spatial mask) — pairs where EITHER side is
    None are skipped, and if every pair is skipped, returns zeros with
    n_masks_evaluated=0 rather than raising, so a caller can distinguish
    "no masks available to score" from "masks scored badly".

    Kappa uses the standard 2x2-confusion-matrix formula rather than
    sklearn.metrics.cohen_kappa_score, to avoid adding scikit-learn as a
    hard dependency for one metric already computable from the same
    tp/fp/fn/tn counts used for precision/recall.
    """
    tp = fp = fn = tn = 0
    n_evaluated = 0
    for pred, gold in zip(pred_masks, gold_masks):
        if pred is None or gold is None:
            continue
        if pred.shape != gold.shape:
            raise ValueError(
                f"change_f1: shape mismatch pred {pred.shape} vs gold "
                f"{gold.shape} — masks must be the same (H, W)."
            )
        pred_bin = (pred > 0).astype(np.uint8)
        gold_bin = (gold > 0).astype(np.uint8)
        tp += int(np.sum((pred_bin == 1) & (gold_bin == 1)))
        fp += int(np.sum((pred_bin == 1) & (gold_bin == 0)))
        fn += int(np.sum((pred_bin == 0) & (gold_bin == 1)))
        tn += int(np.sum((pred_bin == 0) & (gold_bin == 0)))
        n_evaluated += 1

    if n_evaluated == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "kappa": 0.0,
                "iou": 0.0, "n_masks_evaluated": 0}

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    total = tp + fp + fn + tn
    po = (tp + tn) / total if total > 0 else 0.0
    pe = (((tp + fp) * (tp + fn)) + ((fn + tn) * (fp + tn))) / (total ** 2) if total > 0 else 0.0
    kappa = (po - pe) / (1 - pe) if (1 - pe) != 0 else 0.0

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "kappa": float(kappa), "iou": iou, "n_masks_evaluated": n_evaluated,
    }


# ---------------------------------------------------------------------------
# Fusion — multi-label classification metrics + the mandatory 3-way ablation
# ---------------------------------------------------------------------------

def multilabel_f1(pred_probs: np.ndarray, gold_labels: np.ndarray,
                   threshold: float = 0.5) -> dict:
    """Macro/micro/per-class F1 for multi-label land-cover classification.

    pred_probs: (N, K) array of per-class probabilities.
    gold_labels: (N, K) binary array (multi-hot).
    threshold: probability cutoff for a positive prediction.
    """
    if pred_probs.shape != gold_labels.shape:
        raise ValueError(
            f"multilabel_f1: shape mismatch preds {pred_probs.shape} vs "
            f"golds {gold_labels.shape}."
        )
    pred_bin = (pred_probs >= threshold).astype(np.uint8)
    gold_bin = gold_labels.astype(np.uint8)

    n_classes = pred_bin.shape[1]
    per_class_f1 = []
    tp_total = fp_total = fn_total = 0
    for k in range(n_classes):
        tp = int(np.sum((pred_bin[:, k] == 1) & (gold_bin[:, k] == 1)))
        fp = int(np.sum((pred_bin[:, k] == 1) & (gold_bin[:, k] == 0)))
        fn = int(np.sum((pred_bin[:, k] == 0) & (gold_bin[:, k] == 1)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class_f1.append(f1)
        tp_total += tp
        fp_total += fp
        fn_total += fn

    macro_f1 = float(np.mean(per_class_f1)) if per_class_f1 else 0.0
    micro_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    micro_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                if (micro_precision + micro_recall) > 0 else 0.0)
    exact_match = float(np.mean(np.all(pred_bin == gold_bin, axis=1)))

    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "per_class_f1": per_class_f1,
        "exact_match_accuracy": exact_match,
    }


def fusion_ablation(optical_metric: dict, sar_metric: dict, fusion_metric: dict) -> dict:
    """Assemble the mandatory 3-way ablation report (optical-only / SAR-only
    / optical+SAR), documented in docs/api_contracts.md §6 but missing from
    this file until this audit pass. Each input is a metrics dict as
    returned by multilabel_f1 (or any dict sharing a "macro_f1" key).

    Deliberately makes NO pass/fail judgement and does NOT assert fusion
    beats optical — per fusion_model.py / docs/DECISIONS.md, "never force
    fusion to beat optical" is a hard team rule; a training script that
    hardcodes `assert fusion >= optical` (as an earlier draft of
    notebooks/04_optical_sar_fusion.ipynb did) contradicts that rule and
    was removed. This function only computes the deltas and honestly
    labels which leg (if any) won.
    """
    for name, m in [("optical", optical_metric), ("sar", sar_metric), ("fusion", fusion_metric)]:
        if "macro_f1" not in m:
            raise ValueError(f"fusion_ablation: {name}_metric missing 'macro_f1' key.")

    deltas = {
        "fusion_minus_optical": fusion_metric["macro_f1"] - optical_metric["macro_f1"],
        "fusion_minus_sar": fusion_metric["macro_f1"] - sar_metric["macro_f1"],
    }
    best_leg = max(
        [("optical", optical_metric["macro_f1"]),
         ("sar", sar_metric["macro_f1"]),
         ("fusion", fusion_metric["macro_f1"])],
        key=lambda kv: kv[1],
    )[0]

    return {
        "optical": optical_metric,
        "sar": sar_metric,
        "fusion": fusion_metric,
        "deltas": deltas,
        "best_leg": best_leg,
        "fusion_helped": deltas["fusion_minus_optical"] > 0 and deltas["fusion_minus_sar"] > 0,
    }


# ---------------------------------------------------------------------------
# Score normalization
# ---------------------------------------------------------------------------

def normalize_score(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clip + linearly rescale a raw metric into [0, 1] for combination
    across heterogeneous metrics, per the evaluation requirement."""
    value = max(min_val, min(max_val, value))
    if max_val == min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)
