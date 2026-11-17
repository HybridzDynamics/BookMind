import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from src.popularity_model import get_popular_books

USER_RATINGS_PATH = PROJECT_ROOT / "data" / "processed" / "user_ratings.csv"
CLEAN_BOOKS_PATH = PROJECT_ROOT / "data" / "processed" / "books_clean.csv"
COLLABORATIVE_MODEL_PATH = PROJECT_ROOT / "models" / "collaborative_model.pkl"


def train_collaborative_model(
    ratings_path: Path = USER_RATINGS_PATH,
    clean_books_path: Path = CLEAN_BOOKS_PATH,
    model_output_path: Path = COLLABORATIVE_MODEL_PATH,
    latent_factors: int = 50
) -> Dict[str, Any]:
    if not ratings_path.exists():
        error_message = f"Ratings file not found at: {ratings_path}. Run prepare_ratings.py first."
        raise FileNotFoundError(error_message)

    if not clean_books_path.exists():
        error_message = f"Cleaned books file not found at: {clean_books_path}. Run data_cleaning.py first."
        raise FileNotFoundError(error_message)

    print(f"Loading ratings from: {ratings_path}")
    ratings_dataframe = pd.read_csv(ratings_path)
    ratings_dataframe["user_id"] = ratings_dataframe["user_id"].astype(str)
    ratings_dataframe["book_id"] = ratings_dataframe["book_id"].astype(str)
    ratings_dataframe["rating"] = pd.to_numeric(ratings_dataframe["rating"], errors="coerce")
    ratings_dataframe = ratings_dataframe.dropna(subset=["user_id", "book_id", "rating"])

    print(f"Loading books from: {clean_books_path}")
    books_dataframe = pd.read_csv(clean_books_path)
    books_dataframe["bookId"] = books_dataframe["bookId"].astype(str)

    valid_book_ids_set = set(books_dataframe["bookId"].unique())
    valid_ratings_mask = ratings_dataframe["book_id"].isin(valid_book_ids_set)
    filtered_ratings = ratings_dataframe[valid_ratings_mask]

    grouped = filtered_ratings.groupby(["user_id", "book_id"], as_index=False)
    deduplicated_ratings = grouped["rating"].mean()

    unique_users = deduplicated_ratings["user_id"].unique().tolist()
    unique_books = deduplicated_ratings["book_id"].unique().tolist()

    user_to_index = {}
    for user_index, user_id in enumerate(unique_users):
        user_to_index[user_id] = user_index

    book_to_index = {}
    for book_index, book_id in enumerate(unique_books):
        book_to_index[book_id] = book_index

    row_indices = deduplicated_ratings["user_id"].map(user_to_index)
    col_indices = deduplicated_ratings["book_id"].map(book_to_index)
    rating_values = deduplicated_ratings["rating"].astype(float)

    total_users_count = len(unique_users)
    total_books_count = len(unique_books)
    interaction_shape = (total_users_count, total_books_count)

    interaction_matrix = csr_matrix(
        (rating_values, (row_indices, col_indices)),
        shape=interaction_shape
    )

    minimum_matrix_dim = min(total_users_count, total_books_count)
    if minimum_matrix_dim <= 2:
        error_message = "Interaction matrix dimension is too small to perform SVD decomposition."
        raise ValueError(error_message)

    components_count = latent_factors
    if components_count >= minimum_matrix_dim:
        components_count = minimum_matrix_dim - 1

    print(f"Training TruncatedSVD with {components_count} components on matrix shape {interaction_shape}...")
    svd_model = TruncatedSVD(n_components=components_count, random_state=42)
    user_factors = svd_model.fit_transform(interaction_matrix).astype(np.float32)
    book_factors = svd_model.components_.T.astype(np.float32)
    user_item_matrix = interaction_matrix.astype(np.float32)

    model_artifacts = {
        "user_item_matrix": user_item_matrix,
        "books": books_dataframe,
        "user_ids": unique_users,
        "book_ids": unique_books,
        "user_to_index": user_to_index,
        "book_to_index": book_to_index,
        "user_factors": user_factors,
        "book_factors": book_factors,
        "latent_factors": components_count
    }

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_artifacts, model_output_path)

    total_ratings = len(deduplicated_ratings)
    print(f"Collaborative model saved to: {model_output_path}")
    print(f"Stats: Number of users = {total_users_count:,}, Number of books = {total_books_count:,}, Number of ratings = {total_ratings:,}, Number of SVD factors = {components_count}")

    load_collaborative_model.cache_clear()
    return model_artifacts


@lru_cache(maxsize=4)
def load_collaborative_model(model_path: Path = COLLABORATIVE_MODEL_PATH) -> Dict[str, Any]:
    if not model_path.exists():
        error_message = f"Collaborative model not found at: {model_path}. Run train_collaborative_model() first."
        raise FileNotFoundError(error_message)

    loaded_model = joblib.load(model_path)
    return loaded_model


def recommend_for_user(
    user_id: str,
    number_of_books: int = 10,
    max_books_per_author: int = 2,
    model_path: Path = COLLABORATIVE_MODEL_PATH
) -> List[Dict[str, Any]]:
    target_user_id = str(user_id).strip()
    model = load_collaborative_model(model_path)

    user_to_index = model["user_to_index"]

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
                "predicted_score": float(pop_row.get("popularity_score", 0.0)),
                "reason": "Top trending book across BookMind readers (recommended for new reader)",
                "image_url": str(pop_row.get("image_url", ""))
            }
            fallback_list.append(fallback_item)
            author_counts[author_val] = current_count + 1
            if len(fallback_list) >= number_of_books:
                break
        return fallback_list

    user_internal_index = user_to_index[target_user_id]
    target_user_factor = model["user_factors"][user_internal_index]
    book_factors = model["book_factors"]

    predicted_scores = np.dot(target_user_factor, book_factors.T)

    books_dataframe = model["books"]
    book_ids_list = model["book_ids"]

    if "user_item_matrix" in model:
        user_matrix_row = model["user_item_matrix"][user_internal_index]
        already_rated_book_ids = set()
        for rated_column_index in user_matrix_row.indices:
            already_rated_book_ids.add(str(book_ids_list[rated_column_index]))
    else:
        ratings_dataframe = model["ratings"]
        user_rating_rows = ratings_dataframe[ratings_dataframe["user_id"] == target_user_id]
        already_rated_book_ids = set(user_rating_rows["book_id"].astype(str).unique())

    sorted_candidate_indices = np.argsort(predicted_scores)
    sorted_candidate_indices = sorted_candidate_indices[::-1]

    recommendations_list = []
    author_counts_tracker = {}

    for candidate_index in sorted_candidate_indices:
        book_id_string = str(book_ids_list[candidate_index])

        if book_id_string in already_rated_book_ids:
            continue

        matching_rows = books_dataframe[books_dataframe["bookId"] == book_id_string]
        if len(matching_rows) == 0:
            continue

        book_row = matching_rows.iloc[0]
        author_name = str(book_row["author"]).strip()

        current_author_count = author_counts_tracker.get(author_name, 0)
        if current_author_count >= max_books_per_author:
            continue

        raw_score = float(predicted_scores[candidate_index])

        image_url = ""
        if "image_url" in book_row:
            image_url = str(book_row["image_url"])

        record = {
            "bookId": book_id_string,
            "title": str(book_row["title"]),
            "author": author_name,
            "predicted_score": round(raw_score, 4),
            "reason": "Readers with similar reading tastes and ratings also enjoyed this book",
            "image_url": image_url
        }
        recommendations_list.append(record)
        author_counts_tracker[author_name] = current_author_count + 1

        if len(recommendations_list) >= number_of_books:
            break

    return recommendations_list


if __name__ == "__main__":
    train_collaborative_model()
