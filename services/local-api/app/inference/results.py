from __future__ import annotations


def classify_inference_result(positive: int, negative: int, *, succeeded: bool) -> str:
    """Single source of truth for AI result classification."""
    if not succeeded:
        return "unchecked"
    if positive > 0:
        return "positive"
    if negative > 0:
        return "negative"
    return "needs_review"
