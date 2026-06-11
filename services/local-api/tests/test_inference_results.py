from __future__ import annotations

import pytest

from app.inference.results import classify_inference_result


@pytest.mark.parametrize(
    ("positive", "negative", "succeeded", "expected"),
    [
        (0, 0, False, "unchecked"),
        (5, 3, False, "unchecked"),  # not succeeded always wins
        (1, 0, True, "positive"),
        (2, 7, True, "positive"),  # positive takes precedence over negative
        (0, 1, True, "negative"),
        (0, 0, True, "needs_review"),
    ],
)
def test_classify_inference_result(positive, negative, succeeded, expected):
    assert classify_inference_result(positive, negative, succeeded=succeeded) == expected
