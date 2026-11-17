
import numpy as np
import pytest

from src.hybrid_model import (
    DEFAULT_COLLABORATIVE_WEIGHT,
    DEFAULT_CONTENT_WEIGHT,
    DEFAULT_POPULARITY_WEIGHT,
    DEFAULT_READABILITY_WEIGHT,
    normalize_scores_array
)


def test_normalize_scores_array_range():
    raw_scores = np.array([10.0, 20.0, 50.0, 100.0])
    normalized = normalize_scores_array(raw_scores)

    assert float(normalized.min()) == 0.0
    assert float(normalized.max()) == 1.0
    assert len(normalized) == 4


def test_normalize_scores_array_identical_values():
    identical_scores = np.array([5.0, 5.0, 5.0])
    normalized = normalize_scores_array(identical_scores)

    assert float(normalized.sum()) == 0.0


def test_hybrid_weights_sum_to_one():
    total_weight = (
        DEFAULT_COLLABORATIVE_WEIGHT
        + DEFAULT_CONTENT_WEIGHT
        + DEFAULT_READABILITY_WEIGHT
        + DEFAULT_POPULARITY_WEIGHT
    )
    assert round(total_weight, 6) == 1.0


def test_hybrid_weighted_combination():
    collaborative_score = 1.0
    content_score = 0.5
    readability_score = 0.4
    popularity_score = 0.2

    expected_hybrid = (
        (DEFAULT_COLLABORATIVE_WEIGHT * collaborative_score)
        + (DEFAULT_CONTENT_WEIGHT * content_score)
        + (DEFAULT_READABILITY_WEIGHT * readability_score)
        + (DEFAULT_POPULARITY_WEIGHT * popularity_score)
    )
    assert round(expected_hybrid, 3) == 0.600
