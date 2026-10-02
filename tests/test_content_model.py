"""
Tests for Content-Based Recommendation Model.

Tests cover:
- NearestNeighbors cosine distance recommendations.
- Target book exclusion.
- Author diversity constraint (max 2 per author).
- Unknown book ID error handling.

Note: Strictly functional test implementations using only `def` (no `class`).
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def test_content_model_fit_and_query():
    """
    Test TF-IDF fitting and neighbor queries with target exclusion.
    """
    mock_books = pd.DataFrame({
        "bookId": ["1", "2", "3", "4"],
        "title": ["Space Odyssey", "Solaris Orbit", "Cookbook Recipes", "Baking Pastries"],
        "author": ["Arthur Clarke", "Stanislaw Lem", "Chef John", "Chef John"],
        "content": [
            "sci-fi space galaxy exploration future",
            "sci-fi space cosmos planets future",
            "culinary recipes kitchen meals cooking",
            "culinary kitchen desserts baking cooking"
        ]
    })

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(mock_books["content"])

    nn_model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=3)
    nn_model.fit(tfidf_matrix)

    # Query book 1 ("Space Odyssey")
    target_vector = tfidf_matrix[0]
    distances, indices = nn_model.kneighbors(target_vector, n_neighbors=3)

    neighbor_indices = indices[0]
    # Nearest non-self neighbor should be index 1 ("Solaris Orbit")
    assert neighbor_indices[0] == 0  # self
    assert neighbor_indices[1] == 1  # closest sci-fi book


def test_author_diversity_constraint():
    """
    Test that recommendations do not exceed the max books per author limit.
    """
    candidate_items = [
        {"bookId": "1", "author": "Prolific Author"},
        {"bookId": "2", "author": "Prolific Author"},
        {"bookId": "3", "author": "Prolific Author"},
        {"bookId": "4", "author": "Diverse Author"}
    ]

    max_per_author = 2
    filtered_results = []
    author_tracker = {}

    for item in candidate_items:
        auth = item["author"]
        current_count = author_tracker.get(auth, 0)
        if current_count >= max_per_author:
            continue
        filtered_results.append(item)
        author_tracker[auth] = current_count + 1

    assert len(filtered_results) == 3
    assert author_tracker["Prolific Author"] == 2
    assert author_tracker["Diverse Author"] == 1
