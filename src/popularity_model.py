"""
BookMind Popularity Recommendation Model.

This module implements a weighted popularity recommendation model.
It computes an adjusted Bayesian weighted rating (similar to IMDB's formula)
combined with logarithmic scaling of total ratings count to reward books that
have both consistently high scores and a solid volume of ratings.

Formula:
    Weighted Rating (WR) = (v / (v + m)) * R + (m / (v + m)) * C
    Popularity Score     = WR * log(1 + v)

Where:
    v = number of ratings for the book (`numRatings`)
    m = minimum ratings threshold (e.g., 60th percentile of ratings count)
    R = average rating of the book (`rating`)
    C = overall mean rating across all books in the catalog
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_BOOKS_PATH = PROJECT_ROOT / "data" / "processed" / "books_clean.csv"
POPULARITY_MODEL_PATH = PROJECT_ROOT / "models" / "popularity_model.pkl"


def calculate_popularity_scores(
    books_dataframe: pd.DataFrame,
    quantile_threshold: float = 0.60
) -> pd.DataFrame:
    """
    Compute popularity scores for all books in the DataFrame using Bayesian weighting.

    Parameters
    ----------
    books_dataframe : pd.DataFrame
        DataFrame containing book metadata with columns ['bookId', 'rating', 'numRatings'].
    quantile_threshold : float, default=0.60
        Quantile used to determine the minimum number of ratings required (m).

    Returns
    -------
    pd.DataFrame
        DataFrame sorted by `popularity_score` in descending order.
    """
    ranking_dataframe = books_dataframe.copy()

    # Ensure rating and rating count are strictly numeric
    rating_series = pd.to_numeric(ranking_dataframe["rating"], errors="coerce").fillna(0.0)
    ranking_dataframe["rating"] = rating_series

    num_ratings_series = pd.to_numeric(ranking_dataframe["numRatings"], errors="coerce").fillna(0)
    ranking_dataframe["numRatings"] = num_ratings_series

    # Global mean rating across all catalog books
    mean_catalog_rating = float(ranking_dataframe["rating"].mean())

    # Minimum ratings threshold (m)
    calculated_threshold = float(ranking_dataframe["numRatings"].quantile(quantile_threshold))
    if calculated_threshold < 1.0:
        minimum_ratings_threshold = 1.0
    else:
        minimum_ratings_threshold = calculated_threshold

    ratings_count = ranking_dataframe["numRatings"]
    average_rating = ranking_dataframe["rating"]

    # Compute Bayesian weighted rating
    weight_for_book = ratings_count / (ratings_count + minimum_ratings_threshold)
    weight_for_prior = minimum_ratings_threshold / (ratings_count + minimum_ratings_threshold)

    weighted_rating = (weight_for_book * average_rating) + (weight_for_prior * mean_catalog_rating)
    ranking_dataframe["weighted_rating"] = weighted_rating.round(4)

    # Scale weighted rating by the log of ratings count to reward high-volume books
    log_rating_count = np.log1p(ratings_count)
    ranking_dataframe["popularity_score"] = (ranking_dataframe["weighted_rating"] * log_rating_count).round(4)

    # Sort descending by final popularity score
    sorted_dataframe = ranking_dataframe.sort_values(by="popularity_score", ascending=False)
    final_dataframe = sorted_dataframe.reset_index(drop=True)

    return final_dataframe


def train_popularity_model(
    clean_books_path: Path = CLEAN_BOOKS_PATH,
    model_output_path: Path = POPULARITY_MODEL_PATH
) -> pd.DataFrame:
    """
    Train and serialize the popularity recommendation model artifact.

    Parameters
    ----------
    clean_books_path : Path
        Path to `books_clean.csv`.
    model_output_path : Path
        Path where the serialized popularity model will be saved.

    Returns
    -------
    pd.DataFrame
        The ranked popularity DataFrame.

    Raises
    ------
    FileNotFoundError
        If the cleaned books dataset does not exist.
    """
    if not clean_books_path.exists():
        error_message = f"Cleaned books file not found at: {clean_books_path}. Run data_cleaning.py first."
        raise FileNotFoundError(error_message)

    print(f"Loading cleaned books from: {clean_books_path}")
    books_dataframe = pd.read_csv(clean_books_path)

    print("Computing popularity scores...")
    popularity_dataframe = calculate_popularity_scores(books_dataframe)

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(popularity_dataframe, model_output_path)
    print(f"Popularity model successfully saved to: {model_output_path}")

    return popularity_dataframe


def load_popularity_model(model_path: Path = POPULARITY_MODEL_PATH) -> pd.DataFrame:
    """
    Load the serialized popularity recommendation model from disk.

    Parameters
    ----------
    model_path : Path
        Path to the saved `popularity_model.pkl` file.

    Returns
    -------
    pd.DataFrame
        Ranked popularity DataFrame.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist.
    """
    if not model_path.exists():
        error_message = f"Popularity model not found at: {model_path}. Train it first using train_popularity_model()."
        raise FileNotFoundError(error_message)

    loaded_model = joblib.load(model_path)
    return loaded_model


def get_popular_books(
    number_of_books: int = 10,
    model_path: Path = POPULARITY_MODEL_PATH
) -> pd.DataFrame:
    """
    Retrieve the top N most popular books.

    Parameters
    ----------
    number_of_books : int, default=10
        Number of recommendations to return.
    model_path : Path
        Path to the serialized popularity model.

    Returns
    -------
    pd.DataFrame
        Top N popular books.
    """
    if number_of_books < 1:
        number_of_books = 1

    popularity_dataframe = load_popularity_model(model_path)
    top_books = popularity_dataframe.head(number_of_books)
    return top_books


if __name__ == "__main__":
    train_popularity_model()
    top_recommendations = get_popular_books(10)
    display_columns = ["bookId", "title", "author", "rating", "numRatings", "popularity_score"]
    available_display_columns = []
    for column_name in display_columns:
        if column_name in top_recommendations.columns:
            available_display_columns.append(column_name)
    print("\nTop 10 Popular Books:")
    print(top_recommendations[available_display_columns].to_string(index=False))
