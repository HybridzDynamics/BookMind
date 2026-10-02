"""
BookMind Ratings Preparation Module.

This module processes raw interaction records from `data/raw/ratings.csv`
and matches them against cleaned books from `data/processed/books_clean.csv`.
The result is saved to `data/processed/user_ratings.csv` for collaborative filtering
and model evaluation.
"""

from pathlib import Path
import csv
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_RATINGS_PATH = PROJECT_ROOT / "data" / "raw" / "ratings.csv"
CLEAN_BOOKS_PATH = PROJECT_ROOT / "data" / "processed" / "books_clean.csv"
PROCESSED_RATINGS_PATH = PROJECT_ROOT / "data" / "processed" / "user_ratings.csv"


def prepare_ratings_data(
    raw_ratings_path: Path = RAW_RATINGS_PATH,
    clean_books_path: Path = CLEAN_BOOKS_PATH,
    output_ratings_path: Path = PROCESSED_RATINGS_PATH
) -> pd.DataFrame:
    """
    Filter, clean, and deduplicate user ratings against the catalog of cleaned books.

    Parameters
    ----------
    raw_ratings_path : Path
        Filesystem path to the raw `ratings.csv` file.
    clean_books_path : Path
        Filesystem path to the processed `books_clean.csv` file.
    output_ratings_path : Path
        Filesystem path where the cleaned `user_ratings.csv` will be saved.

    Returns
    -------
    pd.DataFrame
        Cleaned ratings DataFrame containing columns ['user_id', 'book_id', 'rating'].

    Raises
    ------
    FileNotFoundError
        If raw ratings or cleaned books files are missing.
    ValueError
        If required rating columns are missing from the raw input file.
    """
    if not raw_ratings_path.exists():
        error_message = f"Missing raw ratings file at: {raw_ratings_path}"
        raise FileNotFoundError(error_message)

    if not clean_books_path.exists():
        error_message = f"Missing clean books file at: {clean_books_path}. Please run data_cleaning.py first."
        raise FileNotFoundError(error_message)

    print(f"Reading cleaned books from: {clean_books_path}")
    books_dataframe = pd.read_csv(clean_books_path)
    valid_book_ids_set = set(books_dataframe["bookId"].astype(str).unique())

    print(f"Reading raw user ratings from: {raw_ratings_path}")
    columns_to_load = ["user_id", "book_id", "rating"]
    ratings_dataframe = pd.read_csv(raw_ratings_path, usecols=columns_to_load)

    # Standardize IDs as strings
    ratings_dataframe["user_id"] = ratings_dataframe["user_id"].astype(str)
    ratings_dataframe["book_id"] = ratings_dataframe["book_id"].astype(str)

    # Ensure rating is numeric
    numeric_ratings = pd.to_numeric(ratings_dataframe["rating"], errors="coerce")
    ratings_dataframe["rating"] = numeric_ratings

    # Drop null values
    ratings_dataframe = ratings_dataframe.dropna(subset=["user_id", "book_id", "rating"])

    # Retain ratings within the valid 1 to 5 scale
    valid_rating_range_mask = (ratings_dataframe["rating"] >= 1.0) & (ratings_dataframe["rating"] <= 5.0)
    ratings_dataframe = ratings_dataframe[valid_rating_range_mask]

    # Filter to only books present in the cleaned book catalog
    is_valid_book_mask = ratings_dataframe["book_id"].isin(valid_book_ids_set)
    ratings_dataframe = ratings_dataframe[is_valid_book_mask]

    # Deduplicate repeated ratings by taking the average rating per user-book pair
    grouped_ratings = ratings_dataframe.groupby(["user_id", "book_id"], as_index=False)
    deduplicated_ratings = grouped_ratings["rating"].mean()

    # Round ratings to two decimal places for clean storage
    deduplicated_ratings["rating"] = deduplicated_ratings["rating"].round(2)

    # Ensure destination folder exists and save to CSV
    output_ratings_path.parent.mkdir(parents=True, exist_ok=True)
    deduplicated_ratings.to_csv(
        output_ratings_path,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_ALL,
        lineterminator="\n"
    )

    total_ratings_count = len(deduplicated_ratings)
    total_users_count = deduplicated_ratings["user_id"].nunique()
    total_books_count = deduplicated_ratings["book_id"].nunique()

    print(f"Saved {total_ratings_count:,} ratings across {total_users_count:,} users and {total_books_count:,} books to: {output_ratings_path}")

    return deduplicated_ratings


if __name__ == "__main__":
    prepare_ratings_data()
