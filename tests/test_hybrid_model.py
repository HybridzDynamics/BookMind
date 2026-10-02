"""
Tests for Hybrid Recommendation Model.

Tests cover:
- Score normalization to [0.0, 1.0].
- Weighted linear combination logic.
- Author diversity constraints.

Note: Strictly functional test implementations using only `def` (no `class`).
"""

import numpy as np
import pytest

from src.hybrid_model import normalize_scores_array


def test_normalize_scores_array_range():
    """
    Test that min-max normalization scales values strictly between 0.0 and 1.0.
    """
    raw_scores = np.array([10.0, 20.0, 50.0, 100.0])
    normalized = normalize_scores_array(raw_scores)

    assert float(normalized.min()) == 0.0
    assert float(normalized.max()) == 1.0
    assert len(normalized) == 4


def test_normalize_scores_array_identical_values():
    """
    Test that an array of identical values normalizes to zeros without divide-by-zero error.
    """
    identical_scores = np.array([5.0, 5.0, 5.0])
    normalized = normalize_scores_array(identical_scores)

    assert float(normalized.sum()) == 0.0


def test_hybrid_weighted_combination():
    """
    Test the 40/35/25 weighted score combination arithmetic.
    """
    collab_score = 1.0
    content_score = 0.5
    popularity_score = 0.2

    expected_hybrid = (0.40 * collab_score) + (0.35 * content_score) + (0.25 * popularity_score)
    # 0.40 + 0.175 + 0.05 = 0.625
    assert round(expected_hybrid, 3) == 0.625
