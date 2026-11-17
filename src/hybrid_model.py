import sys
from functools import lru_cache
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

DEFAULT_COLLABORATIVE_WEIGHT = 0.35
DEFAULT_CONTENT_WEIGHT = 0.30
DEFAULT_READABILITY_WEIGHT = 0.15
DEFAULT_POPULARITY_WEIGHT = 0.20


def normalize_scores_array(scores_input: np.ndarray) -> np.ndarray:
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

    if "difficulty" in catalog_books.columns:
        difficulty_scores_array = (
            pd.to_numeric(catalog_books["difficulty"], errors="coerce")
            .fillna(0.5)
            .to_numpy(dtype=float)
        )
    else:
        difficulty_scores_array = np.full(len(catalog_books), 0.5, dtype=float)

    hybrid_artifacts = {
        "books": catalog_books,
        "content_model": content_model_bundle,
        "collaborative_model": collaborative_model_bundle,
        "popularity_scores": normalized_popularity_scores,
        "difficulty_scores": difficulty_scores_array,
        "weights": {
            "collaborative": DEFAULT_COLLABORATIVE_WEIGHT,
            "content": DEFAULT_CONTENT_WEIGHT,
            "readability": DEFAULT_READABILITY_WEIGHT,
            "popularity": DEFAULT_POPULARITY_WEIGHT
        }
    }

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(hybrid_artifacts, model_output_path)
    print(f"Hybrid model successfully saved to: {model_output_path}")

    load_hybrid_model.cache_clear()
    return hybrid_artifacts


@lru_cache(maxsize=4)
def load_hybrid_model(model_path: Path = HYBRID_MODEL_PATH) -> Dict[str, Any]:
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
    readability_weight: float = DEFAULT_READABILITY_WEIGHT,
    popularity_weight: float = DEFAULT_POPULARITY_WEIGHT,
    model_path: Path = HYBRID_MODEL_PATH
) -> List[Dict[str, Any]]:
    target_user_id = str(user_id).strip()
    hybrid_bundle = load_hybrid_model(model_path)

    collaborative_bundle = hybrid_bundle["collaborative_model"]
    user_to_index = collaborative_bundle["user_to_index"]

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
                "readability_score": 0.0,
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

    if "user_item_matrix" in collaborative_bundle:
        user_matrix_row = collaborative_bundle["user_item_matrix"][user_internal_index]
        rated_column_indices = user_matrix_row.indices
        rated_column_data = user_matrix_row.data
        already_rated_book_ids = set()
        liked_book_ids = set()
        for col_idx, rating_val in zip(rated_column_indices, rated_column_data):
            rated_book_id_str = str(collab_book_ids[col_idx])
            already_rated_book_ids.add(rated_book_id_str)
            if float(rating_val) >= 4.0:
                liked_book_ids.add(rated_book_id_str)
    else:
        ratings_dataframe = collaborative_bundle["ratings"]
        user_ratings_subset = ratings_dataframe[ratings_dataframe["user_id"] == target_user_id]

        already_rated_book_ids = set()
        for book_id_val in user_ratings_subset["book_id"]:
            already_rated_book_ids.add(str(book_id_val))

        liked_ratings_subset = user_ratings_subset[user_ratings_subset["rating"] >= 4.0]
        liked_book_ids = set()
        for book_id_val in liked_ratings_subset["book_id"]:
            liked_book_ids.add(str(book_id_val))

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

    difficulty_lookup = {}
    if "difficulty" in catalog_books.columns:
        for book_id_val, diff_val in zip(catalog_books["bookId"], catalog_books["difficulty"]):
            difficulty_lookup[str(book_id_val)] = float(diff_val)

    liked_difficulty_values = []
    for liked_id in liked_book_ids:
        if liked_id in difficulty_lookup:
            liked_difficulty_values.append(difficulty_lookup[liked_id])

    if len(liked_difficulty_values) > 0:
        preferred_difficulty = float(np.mean(liked_difficulty_values))
    else:
        all_difficulty_values = []
        for rated_id in already_rated_book_ids:
            if rated_id in difficulty_lookup:
                all_difficulty_values.append(difficulty_lookup[rated_id])
        if len(all_difficulty_values) > 0:
            preferred_difficulty = float(np.mean(all_difficulty_values))
        else:
            preferred_difficulty = 0.5

    difficulty_scores_array = hybrid_bundle.get("difficulty_scores")
    if difficulty_scores_array is None:
        if "difficulty" in catalog_books.columns:
            difficulty_scores_array = (
                pd.to_numeric(catalog_books["difficulty"], errors="coerce")
                .fillna(0.5)
                .to_numpy(dtype=float)
            )
        else:
            difficulty_scores_array = np.full(total_catalog_books, 0.5, dtype=float)

    readability_scores = 1.0 - np.abs(difficulty_scores_array.astype(float) - preferred_difficulty)
    readability_scores = np.clip(readability_scores, 0.0, 1.0)

    normalized_popularity_scores = hybrid_bundle["popularity_scores"]

    total_weight = collaborative_weight + content_weight + readability_weight + popularity_weight
    if total_weight <= 0.0:
        total_weight = 1.0

    combined_hybrid_scores = (
        (collaborative_weight / total_weight) * normalized_collab_scores
        + (content_weight / total_weight) * normalized_content_scores
        + (readability_weight / total_weight) * readability_scores
        + (popularity_weight / total_weight) * normalized_popularity_scores
    )

    sorted_indices = np.argsort(combined_hybrid_scores)
    sorted_indices = sorted_indices[::-1]

    recommendations_list = []
    author_counts_tracker = {}

    for candidate_index in sorted_indices:
        book_row = catalog_books.iloc[candidate_index]
        book_id_string = str(book_row["bookId"])

        if book_id_string in already_rated_book_ids:
            continue

        author_name = str(book_row["author"]).strip()

        current_author_count = author_counts_tracker.get(author_name, 0)
        if current_author_count >= max_books_per_author:
            continue

        matched_author_flag = False
        if author_name in user_liked_authors:
            matched_author_flag = True

        c_score = float(normalized_collab_scores[candidate_index])
        t_score = float(normalized_content_scores[candidate_index])
        r_score = float(readability_scores[candidate_index])

        if matched_author_flag:
            reason_str = f"Matches your reading interest in books by {author_name}"
        elif c_score > 0.6:
            reason_str = "Readers with similar book taste rated this book highly"
        elif t_score > 0.5:
            reason_str = "Content and genre strongly match books you rated 4+ stars"
        elif r_score > 0.8:
            reason_str = "Reading difficulty closely matches the books you enjoy"
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
            "readability_score": round(r_score, 4),
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
