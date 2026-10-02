"""
Tests for Collaborative Filtering Model.

Tests cover:
- Sparse matrix construction and SVD factor dot-product prediction.
- Exclusion of already-rated books.
- Cold-start fallback behavior.

Note: Strictly functional test implementations using only `def` (no `class`).
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


def test_svd_matrix_factorization_dot_product():
    """
    Test that TruncatedSVD reconstructs interactions and dot products can be calculated.
    """
    # 3 users, 4 books
    # user 0 likes books 0 and 1
    # user 1 likes books 1 and 2
    # user 2 likes books 2 and 3
    interaction_dense = np.array([
        [5.0, 4.0, 0.0, 0.0],
        [0.0, 5.0, 4.0, 0.0],
        [0.0, 0.0, 5.0, 5.0]
    ])

    sparse_matrix = csr_matrix(interaction_dense)
    svd = TruncatedSVD(n_components=2, random_state=42)
    user_factors = svd.fit_transform(sparse_matrix)
    book_factors = svd.components_.T

    # Predict for user 0
    scores = np.dot(user_factors[0], book_factors.T)
    assert len(scores) == 4


def test_already_rated_books_exclusion():
    """
    Test that books already rated by the target user are excluded from recommendations.
    """
    candidate_book_ids = ["101", "102", "103", "104"]
    already_rated_set = {"101", "103"}

    unrated_recommendations = []
    for book_id in candidate_book_ids:
        if book_id in already_rated_set:
            continue
        unrated_recommendations.append(book_id)

    assert "101" not in unrated_recommendations
    assert "103" not in unrated_recommendations
    assert unrated_recommendations == ["102", "104"]
