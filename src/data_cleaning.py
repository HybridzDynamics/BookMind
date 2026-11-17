
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
    if not book_tags_file.exists():
        print(f"Notice: {book_tags_file} not found; skipping genre tag extraction.")
        return {}

    if not tags_file.exists():
        print(f"Notice: {tags_file} not found; skipping genre tag extraction.")
        return {}

    print(f"Extracting genre tags from {tags_file} and {book_tags_file}...")

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
            genre_name_clean = raw_tag_name.replace("-", " ")
            valid_tag_lookup[tag_id_num] = genre_name_clean

    book_tags_df = pd.read_csv(book_tags_file)
    book_tags_df["goodreads_book_id"] = book_tags_df["goodreads_book_id"].astype(str)
    book_tags_df["tag_id"] = pd.to_numeric(book_tags_df["tag_id"], errors="coerce").fillna(-1).astype(int)

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
    if not raw_path.exists():
        error_message = f"Raw books file not found at: {raw_path}. Place Goodbooks-10k books.csv there."
        raise FileNotFoundError(error_message)

    print(f"Loading raw books catalog from: {raw_path}")
    raw_dataframe = pd.read_csv(raw_path)

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

    star_columns = ["ratings_1", "ratings_2", "ratings_3", "ratings_4", "ratings_5"]
    star_totals = None
    low_counts = None
    for star_column in star_columns:
        if star_column in raw_dataframe.columns:
            star_numeric = pd.to_numeric(raw_dataframe[star_column], errors="coerce").fillna(0.0)
            if star_totals is None:
                star_totals = star_numeric.copy()
            else:
                star_totals = star_totals + star_numeric
            if star_column in ("ratings_1", "ratings_2"):
                if low_counts is None:
                    low_counts = star_numeric.copy()
                else:
                    low_counts = low_counts + star_numeric

    if star_totals is None or low_counts is None:
        cleaned_dataframe["difficulty"] = 0.5
    else:
        has_ratings_mask = star_totals > 0
        raw_difficulty = pd.Series(0.5, index=raw_dataframe.index, dtype=float)
        raw_difficulty.loc[has_ratings_mask] = (
            low_counts.loc[has_ratings_mask] / star_totals.loc[has_ratings_mask]
        )
        valid_difficulty = raw_difficulty[has_ratings_mask]
        if len(valid_difficulty) > 0:
            diff_min = float(valid_difficulty.min())
            diff_max = float(valid_difficulty.max())
            if diff_max > diff_min:
                normalized_difficulty = (raw_difficulty - diff_min) / (diff_max - diff_min)
            else:
                normalized_difficulty = pd.Series(0.5, index=raw_dataframe.index, dtype=float)
        else:
            normalized_difficulty = pd.Series(0.5, index=raw_dataframe.index, dtype=float)
        cleaned_dataframe["difficulty"] = normalized_difficulty.round(4).astype(float)

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

    cleaned_dataframe = cleaned_dataframe.drop_duplicates(subset=["bookId"])
    title_length_mask = cleaned_dataframe["title"].str.len() > 0
    cleaned_dataframe = cleaned_dataframe[title_length_mask].reset_index(drop=True)

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
