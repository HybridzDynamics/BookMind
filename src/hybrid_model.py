"""
BookMind Hybrid Recommendation Model.

This module combines Collaborative Filtering (40%), Content-Based Filtering (35%),
and Popularity Priors (25%) into an ensemble hybrid recommendation engine.
All component scores are normalized to [0.0, 1.0]. It enforces author diversity
(maximum 2 books per author), cold-start fallback to popularity recommendations
for new users, and outputs explainable recommendation reasons.

Note: Strictly functional implementation using only `def` functions (no `class`).
"""

import sys
from pathlib import Path
from typing import Any, Dict, List
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from src.popularity_model import get_popular_books

POPULARITY_MODEL_PATH = PROJECT_ROOT / "models" / "popularity_model.pkl"
CONTENT_MODEL_PATH = PROJECT_ROOT / "models" / "content_model.pkl"
COLLABORATIVE_MODEL_PATH = PROJECT_ROOT / "models" / "collaborative_model.pkl"
HYBRID_MODEL_PATH = PROJECT_ROOT / "models" / "hybrid_model.pkl"

DEFAULT_COLLABORATIVE_WEIGHT = 0.40
DEFAULT_CONTENT_WEIGHT = 0.35
DEFAULT_POPULARITY_WEIGHT = 0.25


def normalize_scores_array(scores_input: np.ndarray) -> np.ndarray:
    """
    Normalize an array of numerical scores to the range [0.0, 1.0].

    Parameters
    ----------
    scores_input : np.ndarray
        Array of unnormalized raw scores.

    Returns
    -------
    np.ndarray
        Normalized scores array.
    """
    scores_array = np.asarray(scores_input, dtype=float)

    if scores_array.size == 0:
        return scores_array

    minimum_score = float(scores_array.min())
    maximum_score = float(scores_array.max())

    if maximum_score == minimum_score:
        return np.zeros_like(scores_array, dtype=float)
    else:
        normalized_array = (scores_array - minimum_score) / (maximum_score - minimum_score)
        return normalized_array


def train_hybrid_model(
    popularity_path: Path = POPULARITY_MODEL_PATH,
    content_path: Path = CONTENT_MODEL_PATH,
    collaborative_path: Path = COLLABORATIVE_MODEL_PATH,
    model_output_path: Path = HYBRID_MODEL_PATH
) -> Dict[str, Any]:
    """
    Train and serialize the ensemble hybrid recommendation model.

    Parameters
    ----------
    popularity_path : Path
        Path to `popularity_model.pkl`.
    content_path : Path
        Path to `content_model.pkl`.
    collaborative_path : Path
        Path to `collaborative_model.pkl`.
    model_output_path : Path
        Path where `hybrid_model.pkl` will be stored.

    Returns
    -------
    dict
        The hybrid model bundle.

    Raises
    ------
    FileNotFoundError
        If any of the required prerequisite model files are missing.
    """
    if not popularity_path.exists():
        error_message = f"Missing popularity model at: {popularity_path}. Run popularity_model.py first."
        raise FileNotFoundError(error_message)

    if not content_path.exists():
        error_message = f"Missing content model at: {content_path}. Run content_model.py first."
        raise FileNotFoundError(error_message)

    if not collaborative_path.exists():
        error_message = f"Missing collaborative model at: {collaborative_path}. Run collaborative_model.py first."
        raise FileNotFoundError(error_message)

    print("Loading base models for hybrid ensemble...")
    popularity_dataframe = joblib.load(popularity_path)
    content_model_bundle = joblib.load(content_path)
    collaborative_model_bundle = joblib.load(collaborative_path)

    catalog_books = collaborative_model_bundle["books"].copy()
    catalog_books["bookId"] = catalog_books["bookId"].astype(str)

    # Build popularity score lookup
    popularity_lookup = {}
    for _, row in popularity_dataframe.iterrows():
        book_id_key = str(row["bookId"])
        popularity_value = float(row.get("popularity_score", 0.0))
        popularity_lookup[book_id_key] = popularity_value

    raw_popularity_list = []
    for book_id in catalog_books["bookId"]:
        score_value = popularity_lookup.get(str(book_id), 0.0)
        raw_popularity_list.append(score_value)

    raw_popularity_array = np.array(raw_popularity_list, dtype=float)
    normalized_popularity_scores = normalize_scores_array(raw_popularity_array)

    hybrid_artifacts = {
        "books": catalog_books,
        "content_model": content_model_bundle,
        "collaborative_model": collaborative_model_bundle,
        "popularity_scores": normalized_popularity_scores,
        "weights": {
            "collaborative": DEFAULT_COLLABORATIVE_WEIGHT,
            "content": DEFAULT_CONTENT_WEIGHT,
            "popularity": DEFAULT_POPULARITY_WEIGHT
        }
    }

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(hybrid_artifacts, model_output_path)
    print(f"Hybrid model successfully saved to: {model_output_path}")

    return hybrid_artifacts


def load_hybrid_model(model_path: Path = HYBRID_MODEL_PATH) -> Dict[str, Any]:
    """
    Load the serialized hybrid recommendation model from disk.

    Parameters
    ----------
    model_path : Path
        Path to `hybrid_model.pkl`.

    Returns
    -------
    dict
        Hybrid model artifacts dictionary.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist.
    """
    if not model_path.exists():
        error_message = f"Hybrid model not found at: {model_path}. Train it first using train_hybrid_model()."
        raise FileNotFoundError(error_message)

    loaded_model = joblib.load(model_path)
    return loaded_model


def recommend_for_user(
    user_id: str,
    number_of_books: int = 10,
    max_books_per_author: int = 2,
    collaborative_weight: float = DEFAULT_COLLABORATIVE_WEIGHT,
    content_weight: float = DEFAULT_CONTENT_WEIGHT,
    popularity_weight: float = DEFAULT_POPULARITY_WEIGHT,
    model_path: Path = HYBRID_MODEL_PATH
) -> List[Dict[str, Any]]:
    """
    Generate hybrid book recommendations for a user.
    Falls back gracefully to popularity recommendations for cold-start / unknown users.
    Applies author diversity (max 2 books per author).

    Parameters
    ----------
    user_id : str
        Target user identifier.
    number_of_books : int, default=10
        Number of recommendations to return.
    max_books_per_author : int, default=2
        Maximum allowed books per author in the recommendation list.
    collaborative_weight : float, default=0.40
        Weight applied to collaborative filtering scores.
    content_weight : float, default=0.35
        Weight applied to user-profile content similarity scores.
    popularity_weight : float, default=0.25
        Weight applied to global popularity prior scores.
    model_path : Path
        Path to `hybrid_model.pkl`.

    Returns
    -------
    list of dict
        Top hybrid recommendations containing sub-scores and explainable reasons.
    """
    target_user_id = str(user_id).strip()
    hybrid_bundle = load_hybrid_model(model_path)

    collaborative_bundle = hybrid_bundle["collaborative_model"]
    user_to_index = collaborative_bundle["user_to_index"]

    # Cold-start fallback: Unknown user or user without training ratings
    if target_user_id not in user_to_index:
        popular_df = get_popular_books(number_of_books=number_of_books)
        fallback_list = []
        author_counts = {}
        for _, pop_row in popular_df.iterrows():
            author_val = str(pop_row["author"]).strip()
            current_count = author_counts.get(author_val, 0)
            if current_count >= max_books_per_author:
                continue

            fallback_item = {
                "bookId": str(pop_row["bookId"]),
                "title": str(pop_row["title"]),
                "author": author_val,
                "hybrid_score": float(pop_row.get("popularity_score", 0.0)),
                "collaborative_score": 0.0,
                "content_score": 0.0,
                "popularity_score": float(pop_row.get("popularity_score", 0.0)),
                "reason": "Top trending popular book (cold-start recommendation for new reader)",
                "matched_author": False,
                "image_url": str(pop_row.get("image_url", ""))
            }
            fallback_list.append(fallback_item)
            author_counts[author_val] = current_count + 1
            if len(fallback_list) >= number_of_books:
                break
        return fallback_list

    # 1. Compute Collaborative Filtering Scores
    user_internal_index = user_to_index[target_user_id]
    user_latent_factor = collaborative_bundle["user_factors"][user_internal_index]
    book_latent_factors = collaborative_bundle["book_factors"]
    raw_collab_scores = np.dot(user_latent_factor, book_latent_factors.T)

    collab_book_ids = collaborative_bundle["book_ids"]
    collab_scores_lookup = {}
    for index, book_id in enumerate(collab_book_ids):
        collab_scores_lookup[str(book_id)] = float(raw_collab_scores[index])

    catalog_books = hybrid_bundle["books"]
    aligned_collab_list = []
    for book_id in catalog_books["bookId"]:
        score = collab_scores_lookup.get(str(book_id), 0.0)
        aligned_collab_list.append(score)

    normalized_collab_scores = normalize_scores_array(np.array(aligned_collab_list, dtype=float))

    # 2. Compute Content-Based Scores from User's High-Rated Books
    ratings_dataframe = collaborative_bundle["ratings"]
    user_ratings_subset = ratings_dataframe[ratings_dataframe["user_id"] == target_user_id]

    already_rated_book_ids = set()
    for book_id_val in user_ratings_subset["book_id"]:
        already_rated_book_ids.add(str(book_id_val))

    liked_ratings_subset = user_ratings_subset[user_ratings_subset["rating"] >= 4.0]
    liked_book_ids = set()
    for book_id_val in liked_ratings_subset["book_id"]:
        liked_book_ids.add(str(book_id_val))

    # Record authors user liked for match explanation
    user_liked_authors = set()
    for liked_id in liked_book_ids:
        liked_rows = catalog_books[catalog_books["bookId"] == liked_id]
        if len(liked_rows) > 0:
            user_liked_authors.add(str(liked_rows.iloc[0]["author"]).strip())

    content_bundle = hybrid_bundle["content_model"]
    content_books = content_bundle["books"]
    content_tfidf_matrix = content_bundle["tfidf_matrix"]

    total_catalog_books = len(catalog_books)
    normalized_content_scores = np.zeros(total_catalog_books, dtype=float)

    if len(liked_book_ids) > 0:
        liked_indices = []
        for index, book_id in enumerate(content_books["bookId"]):
            if str(book_id) in liked_book_ids:
                liked_indices.append(index)

        if len(liked_indices) > 0:
            user_profile_sparse = content_tfidf_matrix[liked_indices].mean(axis=0)
            user_profile_vector = np.asarray(user_profile_sparse)

            raw_similarities = cosine_similarity(user_profile_vector, content_tfidf_matrix).ravel()

            content_scores_lookup = {}
            for index, book_id in enumerate(content_books["bookId"]):
                content_scores_lookup[str(book_id)] = float(raw_similarities[index])

            aligned_content_list = []
            for book_id in catalog_books["bookId"]:
                score = content_scores_lookup.get(str(book_id), 0.0)
                aligned_content_list.append(score)

            normalized_content_scores = normalize_scores_array(np.array(aligned_content_list, dtype=float))

    # 3. Retrieve Pre-computed Normalized Popularity Scores
    normalized_popularity_scores = hybrid_bundle["popularity_scores"]

    # 4. Weighted Linear Combination
    combined_hybrid_scores = (
        (collaborative_weight * normalized_collab_scores)
        + (content_weight * normalized_content_scores)
        + (popularity_weight * normalized_popularity_scores)
    )

    # Sort indices by hybrid score descending
    sorted_indices = np.argsort(combined_hybrid_scores)
    sorted_indices = sorted_indices[::-1]

    recommendations_list = []
    author_counts_tracker = {}

    for candidate_index in sorted_indices:
        book_row = catalog_books.iloc[candidate_index]
        book_id_string = str(book_row["bookId"])

        # Exclude books the user already evaluated
        if book_id_string in already_rated_book_ids:
            continue

        author_name = str(book_row["author"]).strip()

        # Enforce author diversity
        current_author_count = author_counts_tracker.get(author_name, 0)
        if current_author_count >= max_books_per_author:
            continue

        matched_author_flag = False
        if author_name in user_liked_authors:
            matched_author_flag = True

        # Construct explainable reason
        c_score = float(normalized_collab_scores[candidate_index])
        t_score = float(normalized_content_scores[candidate_index])

        if matched_author_flag:
            reason_str = f"Matches your reading interest in books by {author_name}"
        elif c_score > 0.6:
            reason_str = "Readers with similar book taste rated this book highly"
        elif t_score > 0.5:
            reason_str = "Content and genre strongly match books you rated 4+ stars"
        else:
            reason_str = "Trending highly-rated title aligned with your reading profile"

        image_url = ""
        if "image_url" in book_row:
            image_url = str(book_row["image_url"])

        record = {
            "bookId": book_id_string,
            "title": str(book_row["title"]),
            "author": author_name,
            "hybrid_score": round(float(combined_hybrid_scores[candidate_index]), 4),
            "collaborative_score": round(c_score, 4),
            "content_score": round(t_score, 4),
            "popularity_score": round(float(normalized_popularity_scores[candidate_index]), 4),
            "reason": reason_str,
            "matched_author": matched_author_flag,
            "image_url": image_url
        }
        recommendations_list.append(record)
        author_counts_tracker[author_name] = current_author_count + 1

        if len(recommendations_list) >= number_of_books:
            break

    return recommendations_list


if __name__ == "__main__":
    train_hybrid_model()
