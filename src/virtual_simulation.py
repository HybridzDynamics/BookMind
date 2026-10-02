"""
BookMind Virtual User Activity Simulation Module.

This module simulates realistic user sessions (e.g. searching catalog keywords,
retrieving recommendations via content, collaborative, or hybrid engines)
without modifying or injecting artificial ratings into the clean ground-truth dataset.
Activity logs are persisted to `data/processed/virtual_user_activity.csv`
for observability and behavioral analytics.
"""

import sys
from datetime import datetime
from pathlib import Path
import random
from typing import Any, Dict, List
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

CLEAN_BOOKS_PATH = PROJECT_ROOT / "data" / "processed" / "books_clean.csv"
USER_RATINGS_PATH = PROJECT_ROOT / "data" / "processed" / "user_ratings.csv"
SIMULATION_LOG_PATH = PROJECT_ROOT / "data" / "processed" / "virtual_user_activity.csv"


def run_virtual_simulation(
    number_of_users: int = 5,
    searches_per_user: int = 3,
    model_name: str = "hybrid",
    random_seed: int = 42,
    output_log_path: Path = SIMULATION_LOG_PATH
) -> pd.DataFrame:
    """
    Simulate user queries and recommendation requests and log results.

    Parameters
    ----------
    number_of_users : int, default=5
        Number of virtual user sessions to simulate.
    searches_per_user : int, default=3
        Number of catalog interactions performed by each simulated user.
    model_name : str, default='hybrid'
        Recommendation algorithm to use ('hybrid', 'collaborative', or 'content').
    random_seed : int, default=42
        Seed for deterministic random behavior.
    output_log_path : Path
        Path where session logs are saved as CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame of logged user simulation interactions.

    Raises
    ------
    FileNotFoundError
        If processed books or ratings files are not present.
    """
    if not CLEAN_BOOKS_PATH.exists():
        error_message = f"Cleaned books file missing at: {CLEAN_BOOKS_PATH}. Run data_cleaning.py first."
        raise FileNotFoundError(error_message)

    if not USER_RATINGS_PATH.exists():
        error_message = f"User ratings file missing at: {USER_RATINGS_PATH}. Run prepare_ratings.py first."
        raise FileNotFoundError(error_message)

    random.seed(random_seed)

    books_dataframe = pd.read_csv(CLEAN_BOOKS_PATH)
    ratings_dataframe = pd.read_csv(USER_RATINGS_PATH)

    books_dataframe["bookId"] = books_dataframe["bookId"].astype(str)
    ratings_dataframe["user_id"] = ratings_dataframe["user_id"].astype(str)

    unique_user_ids = ratings_dataframe["user_id"].unique().tolist()
    total_available_users = len(unique_user_ids)

    sample_user_count = min(number_of_users, total_available_users)
    selected_user_ids = random.sample(unique_user_ids, sample_user_count)

    from src.content_model import recommend_similar_books
    from src.collaborative_model import recommend_for_user as get_collaborative_recommendations
    from src.hybrid_model import recommend_for_user as get_hybrid_recommendations

    simulation_records = []

    for user_index, real_user_id in enumerate(selected_user_ids, start=1):
        virtual_user_identifier = f"virtual_user_{user_index}"

        for search_step in range(1, searches_per_user + 1):
            random_sample_row = books_dataframe.sample(1).iloc[0]
            selected_book_id = str(random_sample_row["bookId"])
            selected_book_title = str(random_sample_row["title"])

            title_words = selected_book_title.split()
            if len(title_words) > 0:
                search_query_term = title_words[0]
            else:
                search_query_term = "fiction"

            recommendations_list = []
            status_message = "success"

            try:
                if model_name == "content":
                    recommendations_list = recommend_similar_books(
                        book_id=selected_book_id,
                        number_of_books=5
                    )
                elif model_name == "collaborative":
                    recommendations_list = get_collaborative_recommendations(
                        user_id=real_user_id,
                        number_of_books=5
                    )
                else:
                    recommendations_list = get_hybrid_recommendations(
                        user_id=real_user_id,
                        number_of_books=5
                    )
            except Exception as exception_instance:
                status_message = f"error: {str(exception_instance)}"

            log_entry = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "virtual_user": virtual_user_identifier,
                "real_user_id": real_user_id,
                "search_number": search_step,
                "query": search_query_term,
                "selected_book_id": selected_book_id,
                "selected_book_title": selected_book_title,
                "model": model_name,
                "recommendation_count": len(recommendations_list),
                "status": status_message
            }
            simulation_records.append(log_entry)

    simulation_dataframe = pd.DataFrame(simulation_records)
    output_log_path.parent.mkdir(parents=True, exist_ok=True)
    simulation_dataframe.to_csv(output_log_path, index=False)

    print(f"Simulation completed with {len(simulation_dataframe)} activity records logged to: {output_log_path}")
    return simulation_dataframe


if __name__ == "__main__":
    simulation_results = run_virtual_simulation()
    print(simulation_results.to_string(index=False))
