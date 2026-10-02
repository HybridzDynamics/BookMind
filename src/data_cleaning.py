"""
BookMind Data Cleaning Module.

This module processes raw Goodbooks-10k datasets:
- `data/raw/books.csv`: Core catalog metadata (title, authors, ratings, counts).
- `data/raw/book_tags.csv` & `data/raw/tags.csv`: Rich genre and topical tags.

It normalizes text, extracts and attaches top genres per book, creates composite
NLP content strings, and writes the clean dataset to `data/processed/books_clean.csv`.

Note: Strictly functional implementation using only `def` functions (no `class`).
"""

from pathlib import Path
import csv
from typing import Dict, List
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_BOOKS_PATH = PROJECT_ROOT / "data" / "raw" / "books.csv"
RAW_BOOK_TAGS_PATH = PROJECT_ROOT / "data" / "raw" / "book_tags.csv"
RAW_TAGS_PATH = PROJECT_ROOT / "data" / "raw" / "tags.csv"
CLEAN_BOOKS_PATH = PROJECT_ROOT / "data" / "processed" / "books_clean.csv"


def clean_text_field(value: object) -> str:
    """
    Clean and normalize a text string by removing extraneous whitespace.

    Parameters
    ----------
    value : object
        Input value to normalize.

    Returns
    -------
    str
        Cleaned text string, or empty string if input is missing.
    """
    if pd.isna(value):
        return ""
    else:
        text_string = str(value)
        words = text_string.split()
        return " ".join(words)


def extract_genres_from_tags(
    book_tags_file: Path = RAW_BOOK_TAGS_PATH,
    tags_file: Path = RAW_TAGS_PATH,
    max_genres_per_book: int = 5
) -> Dict[str, str]:
    """
    Extract meaningful book genres by cross-referencing book_tags and tags files.

    Parameters
    ----------
    book_tags_file : Path
        Path to `book_tags.csv`.
    tags_file : Path
        Path to `tags.csv`.
    max_genres_per_book : int, default=5
        Maximum number of genre tags to retain per book.

    Returns
    -------
    dict
        Mapping from goodreads_book_id (string) to comma-separated genres string.
    """
    if not book_tags_file.exists():
        print(f"Notice: {book_tags_file} not found; skipping genre tag extraction.")
        return {}

    if not tags_file.exists():
        print(f"Notice: {tags_file} not found; skipping genre tag extraction.")
        return {}

    print(f"Extracting genre tags from {tags_file} and {book_tags_file}...")

    # Excluded noise tags (shelf markers, format indicators, personal book list labels)
    excluded_keywords = [
        "to-read", "favorites", "currently-reading", "owned", "default",
        "books-i-own", "read-in-", "wish-list", "shelfari", "audiobook",
        "ebook", "kindle", "library", "my-books", "bought", "all-time",
        "re-read", "dnf", "abandoned", "audio", "part-of-a-series", "own",
        "i-own", "my-library", "e-book", "e-books", "have"
    ]

    tags_df = pd.read_csv(tags_file)
    valid_tag_lookup = {}

    for _, tag_row in tags_df.iterrows():
        tag_id_num = int(tag_row["tag_id"])
        raw_tag_name = str(tag_row["tag_name"]).lower().strip()

        is_disqualified = False
        if len(raw_tag_name) < 3:
            is_disqualified = True
        elif raw_tag_name.startswith("-") or raw_tag_name.isdigit():
            is_disqualified = True
        else:
            for pattern in excluded_keywords:
                if pattern in raw_tag_name:
                    is_disqualified = True
                    break

        if not is_disqualified:
            # Clean hyphens in genre names for cleaner text representation
            genre_name_clean = raw_tag_name.replace("-", " ")
            valid_tag_lookup[tag_id_num] = genre_name_clean

    book_tags_df = pd.read_csv(book_tags_file)
    book_tags_df["goodreads_book_id"] = book_tags_df["goodreads_book_id"].astype(str)
    book_tags_df["tag_id"] = pd.to_numeric(book_tags_df["tag_id"], errors="coerce").fillna(-1).astype(int)

    # Filter to meaningful tags only
    valid_tag_id_set = set(valid_tag_lookup.keys())
    matching_tags_mask = book_tags_df["tag_id"].isin(valid_tag_id_set)
    filtered_book_tags = book_tags_df[matching_tags_mask]

    sorted_book_tags = filtered_book_tags.sort_values(
        by=["goodreads_book_id", "count"],
        ascending=[True, False]
    )

    book_genres_map = {}
    current_goodreads_id = None
    accumulated_genres = []

    for _, row in sorted_book_tags.iterrows():
        book_gid = str(row["goodreads_book_id"])
        tag_num = int(row["tag_id"])
        genre_label = valid_tag_lookup.get(tag_num, "")

        if book_gid != current_goodreads_id:
            if current_goodreads_id is not None:
                book_genres_map[current_goodreads_id] = ", ".join(accumulated_genres)
            current_goodreads_id = book_gid
            accumulated_genres = []

        if len(accumulated_genres) < max_genres_per_book:
            if genre_label not in accumulated_genres:
                accumulated_genres.append(genre_label)

    if current_goodreads_id is not None:
        book_genres_map[current_goodreads_id] = ", ".join(accumulated_genres)

    print(f"Extracted genres for {len(book_genres_map):,} books.")
    return book_genres_map


def clean_books_data(
    raw_path: Path = RAW_BOOKS_PATH,
    output_path: Path = CLEAN_BOOKS_PATH
) -> pd.DataFrame:
    """
    Clean the Goodbooks-10k books dataset, extract genres from tag files if available,
    and save the processed result.

    Parameters
    ----------
    raw_path : Path
        Path to raw `books.csv`.
    output_path : Path
        Path where `books_clean.csv` will be written.

    Returns
    -------
    pd.DataFrame
        Cleaned books DataFrame.

    Raises
    ------
    FileNotFoundError
        If `books.csv` is missing from `raw_path`.
    ValueError
        If required columns are absent in `books.csv`.
    """
    if not raw_path.exists():
        error_message = f"Raw books file not found at: {raw_path}. Place Goodbooks-10k books.csv there."
        raise FileNotFoundError(error_message)

    print(f"Loading raw books catalog from: {raw_path}")
    raw_dataframe = pd.read_csv(raw_path)

    # Strip column name whitespace
    clean_columns = []
    for column_name in raw_dataframe.columns:
        stripped_name = str(column_name).strip()
        clean_columns.append(stripped_name)
    raw_dataframe.columns = clean_columns

    required_columns = [
        "book_id",
        "title",
        "authors",
        "average_rating",
        "ratings_count"
    ]

    missing_columns = []
    for required_column in required_columns:
        if required_column not in raw_dataframe.columns:
            missing_columns.append(required_column)

    if len(missing_columns) > 0:
        available_columns = list(raw_dataframe.columns)
        error_message = f"Missing required columns: {missing_columns}. Available: {available_columns}"
        raise ValueError(error_message)

    cleaned_dataframe = pd.DataFrame()

    cleaned_dataframe["bookId"] = raw_dataframe["book_id"].astype(str)
    cleaned_dataframe["title"] = raw_dataframe["title"].map(clean_text_field)
    cleaned_dataframe["author"] = raw_dataframe["authors"].map(clean_text_field)

    numeric_rating = pd.to_numeric(raw_dataframe["average_rating"], errors="coerce")
    cleaned_dataframe["rating"] = numeric_rating.fillna(0.0)

    numeric_count = pd.to_numeric(raw_dataframe["ratings_count"], errors="coerce")
    cleaned_dataframe["numRatings"] = numeric_count.fillna(0).astype(int)

    # Extract genres from tags if tags files exist
    extracted_genres_map = extract_genres_from_tags()

    if len(extracted_genres_map) > 0 and "goodreads_book_id" in raw_dataframe.columns:
        genres_column_values = []
        for gid_val in raw_dataframe["goodreads_book_id"].astype(str):
            genre_str = extracted_genres_map.get(gid_val, "")
            genres_column_values.append(genre_str)
        cleaned_dataframe["genres"] = genres_column_values
    elif "genres" in raw_dataframe.columns:
        cleaned_dataframe["genres"] = raw_dataframe["genres"].map(clean_text_field)
    else:
        cleaned_dataframe["genres"] = ""

    if "description" in raw_dataframe.columns:
        cleaned_dataframe["description"] = raw_dataframe["description"].map(clean_text_field)
    else:
        cleaned_dataframe["description"] = ""

    if "language_code" in raw_dataframe.columns:
        cleaned_dataframe["language_code"] = raw_dataframe["language_code"].map(clean_text_field)
    else:
        cleaned_dataframe["language_code"] = ""

    if "image_url" in raw_dataframe.columns:
        cleaned_dataframe["image_url"] = raw_dataframe["image_url"].map(clean_text_field)
    else:
        cleaned_dataframe["image_url"] = ""

    # Build composite content string used by content-based TF-IDF models
    combined_content_series = (
        cleaned_dataframe["title"]
        + " "
        + cleaned_dataframe["author"]
        + " "
        + cleaned_dataframe["description"]
        + " "
        + cleaned_dataframe["genres"]
        + " "
        + cleaned_dataframe["language_code"]
    )
    cleaned_dataframe["content"] = combined_content_series.str.strip()

    # Deduplicate and filter out books with empty titles
    cleaned_dataframe = cleaned_dataframe.drop_duplicates(subset=["bookId"])
    title_length_mask = cleaned_dataframe["title"].str.len() > 0
    cleaned_dataframe = cleaned_dataframe[title_length_mask].reset_index(drop=True)

    # Save to processed directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_ALL,
        lineterminator="\n"
    )

    total_books_count = len(cleaned_dataframe)
    books_with_genres = (cleaned_dataframe["genres"].str.len() > 0).sum()
    print(f"Successfully cleaned and saved {total_books_count:,} books to: {output_path} ({books_with_genres:,} books with genres)")

    return cleaned_dataframe


if __name__ == "__main__":
    clean_books_data()
