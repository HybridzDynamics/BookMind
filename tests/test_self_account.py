import numpy as np
import pytest

from src.self_account import normalize_scores_array


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


def test_normalize_scores_array_empty():
    empty_scores = np.array([])
    normalized = normalize_scores_array(empty_scores)

    assert len(normalized) == 0


def test_normalize_scores_array_single_value():
    single_score = np.array([42.0])
    normalized = normalize_scores_array(single_score)

    assert float(normalized.sum()) == 0.0
    assert len(normalized) == 1
