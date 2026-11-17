
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


def test_svd_matrix_factorization_dot_product():
    interaction_dense = np.array([
        [5.0, 4.0, 0.0, 0.0],
        [0.0, 5.0, 4.0, 0.0],
        [0.0, 0.0, 5.0, 5.0]
    ])

    sparse_matrix = csr_matrix(interaction_dense)
    svd = TruncatedSVD(n_components=2, random_state=42)
    user_factors = svd.fit_transform(sparse_matrix)
    book_factors = svd.components_.T

    scores = np.dot(user_factors[0], book_factors.T)
    assert len(scores) == 4


def test_already_rated_books_exclusion():
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
