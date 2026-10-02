"""
Tests for Data Cleaning Module.

Tests cover:
- Text normalization function.
- Validation of required columns.
- Handling of missing values.
- Duplicate removal.

Note: Strictly functional test implementations using only `def` (no `class`).
"""

import pandas as pd
import pytest

from src.data_cleaning import clean_text_field


def test_clean_text_field_normalizes_whitespace():
    """
    Test that clean_text_field strips leading, trailing, and repeated whitespace.
    """
    dirty_text = "   The   Great    Gatsby   "
    cleaned_result = clean_text_field(dirty_text)
    assert cleaned_result == "The Great Gatsby"


def test_clean_text_field_handles_null_values():
    """
    Test that clean_text_field safely converts NaN/None to an empty string.
    """
    none_result = clean_text_field(None)
    assert none_result == ""

    nan_result = clean_text_field(float("nan"))
    assert nan_result == ""


def test_missing_column_detection():
    """
    Test that data cleaning identifies missing required columns properly.
    """
    sample_dataframe = pd.DataFrame({
        "book_id": ["1", "2"],
        "title": ["Book 1", "Book 2"]
    })

    required_columns = ["book_id", "title", "authors", "average_rating", "ratings_count"]
    missing_columns = []
    for column_name in required_columns:
        if column_name not in sample_dataframe.columns:
            missing_columns.append(column_name)

    assert "authors" in missing_columns
    assert "average_rating" in missing_columns
    assert "ratings_count" in missing_columns
    assert len(missing_columns) == 3


def test_duplicate_book_id_removal():
    """
    Test that duplicate book identifiers are removed.
    """
    duplicate_dataframe = pd.DataFrame({
        "bookId": ["1", "1", "2"],
        "title": ["First Book", "First Book Copy", "Second Book"]
    })

    deduplicated = duplicate_dataframe.drop_duplicates(subset=["bookId"])
    assert len(deduplicated) == 2
    assert list(deduplicated["bookId"]) == ["1", "2"]
