"""
Task router — classifies a natural-language query (+ input mode) into one of
the supported task labels. This is the "agentic" query-interpretation step.
Owner: Person 4 (Agentic Orchestration + GUI).
"""

TASK_LABELS = ["vqa", "captioning", "grounding", "change", "fusion"]

# Simple keyword-based first pass. Replace/augment with an LLM classifier
# (e.g. prompt a small instruction-tuned model to output one of TASK_LABELS)
# once the rule-based version is working end-to-end.
_KEYWORDS = {
    "change": ["changed", "change", "before and after", "increased", "decreased",
               "difference between", "over time"],
    "grounding": ["highlight", "locate", "where is", "point to", "mark the", "show me the region"],
    "captioning": ["describe", "caption", "summarize the image", "what is in this image"],
    "fusion": ["optical and sar", "sar and optical", "combine", "together to identify",
               "using both images"],
}


def route_task(query: str, mode: str) -> str:
    """Return one of TASK_LABELS.

    mode: "single" | "cross_modal" | "bi_temporal" — constrains which tasks
    are even possible (e.g. "change" only valid for bi_temporal).
    """
    q = query.lower()

    if mode == "bi_temporal":
        return "change"
    if mode == "cross_modal":
        return "fusion"

    # mode == "single": choose between vqa / captioning / grounding
    for task in ("grounding", "captioning"):
        if any(kw in q for kw in _KEYWORDS[task]):
            return task

    # TODO(Person 4): consider an LLM-based fallback classifier for queries
    # that don't match keywords, instead of always defaulting to "vqa".
    return "vqa"
