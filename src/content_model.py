from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_BOOKS_PATH = PROJECT_ROOT / "data" / "processed" / "books_clean.csv"
CONTENT_MODEL_PATH = PROJECT_ROOT / "models" / "content_model.pkl"


def train_content_model(
    clean_books_path: Path = CLEAN_BOOKS_PATH,
    model_output_path: Path = CONTENT_MODEL_PATH,
    max_features: int = 50000
) -> Dict[str, Any]:
    if not clean_books_path.exists():
        error_message = f"Cleaned books file not found at: {clean_books_path}. Run data_cleaning.py first."
        raise FileNotFoundError(error_message)

    print(f"Loading cleaned books from: {clean_books_path}")
    books_dataframe = pd.read_csv(clean_books_path)
    books_dataframe["bookId"] = books_dataframe["bookId"].astype(str)
    books_dataframe["content"] = books_dataframe["content"].fillna("").astype(str)

    print(f"Building TF-IDF feature matrix (max_features={max_features:,})...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2
    )
    tfidf_matrix = vectorizer.fit_transform(books_dataframe["content"])

    print("Fitting NearestNeighbors model with cosine metric...")
    nearest_neighbors_model = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=50,
        n_jobs=-1
    )
    nearest_neighbors_model.fit(tfidf_matrix)

    model_artifacts = {
        "books": books_dataframe,
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "nearest_neighbors": nearest_neighbors_model
    }

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_artifacts, model_output_path)
    print(f"Content-based model saved to: {model_output_path} (Matrix shape: {tfidf_matrix.shape})")

    load_content_model.cache_clear()
    return model_artifacts


@lru_cache(maxsize=4)
def load_content_model(model_path: Path = CONTENT_MODEL_PATH) -> Dict[str, Any]:
    if not model_path.exists():
        error_message = f"Content model not found at: {model_path}. Train it first using train_content_model()."
        raise FileNotFoundError(error_message)

    loaded_model = joblib.load(model_path)
    return loaded_model


def recommend_similar_books(
    book_id: str,
    number_of_books: int = 10,
    max_books_per_author: int = 2,
    model_path: Path = CONTENT_MODEL_PATH
) -> List[Dict[str, Any]]:
    target_id_string = str(book_id).strip()
    model = load_content_model(model_path)

    books_dataframe = model["books"]
    matching_indices = books_dataframe.index[books_dataframe["bookId"] == target_id_string].tolist()

    if len(matching_indices) == 0:
        error_message = f"Book ID '{target_id_string}' was not found in the catalog."
        raise ValueError(error_message)

    target_index = matching_indices[0]
    target_book_row = books_dataframe.iloc[target_index]
    target_book_title = str(target_book_row["title"])
    target_vector = model["tfidf_matrix"][target_index]

    query_k = min(number_of_books * 4 + 1, len(books_dataframe))

    distances, indices = model["nearest_neighbors"].kneighbors(
        target_vector,
        n_neighbors=query_k
    )

    neighbor_distances = distances[0]
    neighbor_indices = indices[0]

    recommendations_list = []
    author_counts_tracker = {}

    for distance, index in zip(neighbor_distances, neighbor_indices):
        if index == target_index:
            continue

        book_row = books_dataframe.iloc[index]
        author_name = str(book_row["author"]).strip()

        current_author_count = author_counts_tracker.get(author_name, 0)
        if current_author_count >= max_books_per_author:
            continue

        similarity_score = round(float(1.0 - distance), 4)

        image_url = ""
        if "image_url" in book_row:
            image_url = str(book_row["image_url"])

        record = {
            "bookId": str(book_row["bookId"]),
            "title": str(book_row["title"]),
            "author": author_name,
            "similarity_score": similarity_score,
            "reason": f"Similar themes, genres, and narrative keywords to '{target_book_title}'",
            "image_url": image_url
        }
        recommendations_list.append(record)
        author_counts_tracker[author_name] = current_author_count + 1

        if len(recommendations_list) >= number_of_books:
            break

    return recommendations_list


if __name__ == "__main__":
    train_content_model()
