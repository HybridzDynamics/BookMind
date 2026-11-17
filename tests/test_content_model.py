
import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def test_content_model_fit_and_query():
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

    target_vector = tfidf_matrix[0]
    distances, indices = nn_model.kneighbors(target_vector, n_neighbors=3)

    neighbor_indices = indices[0]
    assert neighbor_indices[0] == 0
    assert neighbor_indices[1] == 1


def test_author_diversity_constraint():
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
