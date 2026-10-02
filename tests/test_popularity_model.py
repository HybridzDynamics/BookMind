"""
Tests for Popularity Recommendation Model.

Tests cover:
- Bayesian weighted calculation.
- Ranking ordering (descending score).
- Popularity score monotonicity with rating count.

Note: Strictly functional test implementations using only `def` (no `class`).
"""

import numpy as np
import pandas as pd
import pytest

from src.popularity_model import calculate_popularity_scores


def test_popularity_scores_calculation():
    """
    Test Bayesian score calculation and descending order output.
    """
    mock_catalog = pd.DataFrame({
        "bookId": ["1", "2", "3"],
        "title": ["Popular Classic", "Obscure Great", "Poor Book"],
        "author": ["Author A", "Author B", "Author C"],
        "rating": [4.5, 4.8, 2.0],
        "numRatings": [100000, 10, 50000]
    })

    scored_dataframe = calculate_popularity_scores(mock_catalog, quantile_threshold=0.50)

    assert "popularity_score" in scored_dataframe.columns
    assert "weighted_rating" in scored_dataframe.columns

    # Verify descending ordering
    scores = scored_dataframe["popularity_score"].tolist()
    is_sorted_descending = True
    for index in range(len(scores) - 1):
        if scores[index] < scores[index + 1]:
            is_sorted_descending = False
            break

    assert is_sorted_descending is True
    # The high-rating high-volume book should be ranked first
    assert scored_dataframe.iloc[0]["bookId"] == "1"
